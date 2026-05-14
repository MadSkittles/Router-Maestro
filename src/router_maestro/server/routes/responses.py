"""Responses API route for Codex models."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException

from router_maestro.providers import ProviderError
from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.routing import Router, get_router
from router_maestro.server.schemas import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.utils import get_logger
from router_maestro.utils.reasoning import VALID_EFFORTS

logger = get_logger("server.routes.responses")

router = APIRouter()


def generate_id(prefix: str) -> str:
    """Generate a unique ID with given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def sse_event(data: dict[str, Any]) -> str:
    """Format data as SSE event with event type field."""
    event_type = data.get("type", "")
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def extract_text_from_content(content: str | list[Any]) -> str:
    """Extract text from content which can be a string or list of content blocks."""
    if isinstance(content, str):
        return content

    texts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") in ("input_text", "output_text"):
                texts.append(block.get("text", ""))
            elif "text" in block:
                texts.append(block.get("text", ""))
        elif hasattr(block, "text"):
            texts.append(block.text)
    return "".join(texts)


def convert_content_to_serializable(content: Any) -> Any:
    """Convert content to JSON-serializable format.

    Handles Pydantic models and nested structures.
    """
    if isinstance(content, str):
        return content
    if hasattr(content, "model_dump"):
        return content.model_dump(exclude_none=True)
    if isinstance(content, list):
        return [convert_content_to_serializable(item) for item in content]
    if isinstance(content, dict):
        return {k: convert_content_to_serializable(v) for k, v in content.items()}
    return content


def convert_input_to_internal(
    input_data: str | list[Any],
) -> str | list[dict[str, Any]]:
    """Convert the incoming input format to internal format.

    Preserves the original content format (string or array) as the upstream
    Copilot API accepts both formats. Converts Pydantic models to dicts.
    """
    if isinstance(input_data, str):
        return input_data

    items = []
    for item in input_data:
        if isinstance(item, dict):
            item_type = item.get("type", "message")

            if item_type == "message" or (item_type is None and "role" in item):
                role = item.get("role", "user")
                content = item.get("content", "")
                # Convert content to serializable format
                content = convert_content_to_serializable(content)
                items.append({"type": "message", "role": role, "content": content})

            elif item_type == "function_call":
                fc_item: dict[str, Any] = {
                    "type": "function_call",
                    "id": item.get("id"),
                    "call_id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments", "{}"),
                    "status": item.get("status", "completed"),
                }
                # Preserve MCP namespace verbatim. Copilot CAPI rejects the
                # next turn with ``Missing namespace for function_call 'X'``
                # if a previously-namespaced call is round-tripped without it.
                if item.get("namespace") is not None:
                    fc_item["namespace"] = item["namespace"]
                items.append(fc_item)

            elif item_type == "function_call_output":
                output = item.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.get("call_id"),
                        "output": output,
                    }
                )

            elif item_type == "reasoning":
                # Echoed back from a prior turn — preserve the full shape so
                # Copilot can correlate chain-of-thought across turns. Mirrors
                # vscode-copilot-chat responsesApi.ts:216-230 (extractThinkingData).
                #
                # Codex CLI's ``ResponseItem::Reasoning`` marks ``id`` as
                # ``#[serde(default, skip_serializing)]`` (see openai/codex
                # codex-rs/protocol/src/models.rs), so it NEVER sends the id
                # back on round-trip. Copilot CAPI signs ``encrypted_content``
                # against the upstream id and rejects (id, blob) pairs that
                # don't match. Without an id, the blob is unverifiable, so we
                # MUST strip it — otherwise Copilot 400s with ``Encrypted
                # content could not be decrypted``. (See openai/codex#17541
                # and the parallel litellm bug BerriAI/litellm#22189.)
                reasoning_item: dict[str, Any] = {
                    "type": "reasoning",
                    "id": item.get("id"),
                    "summary": item.get("summary", []) or [],
                }
                if item.get("encrypted_content") and item.get("id"):
                    reasoning_item["encrypted_content"] = item["encrypted_content"]
                items.append(reasoning_item)
            else:
                items.append(convert_content_to_serializable(item))

        elif hasattr(item, "model_dump"):
            # Pydantic model - convert to dict
            items.append(item.model_dump(exclude_none=True))

        elif hasattr(item, "role") and hasattr(item, "content"):
            # Object with role and content attributes
            content = convert_content_to_serializable(item.content)
            items.append({"type": "message", "role": item.role, "content": content})

    return items


def convert_tools_to_internal(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert tools to internal format."""
    if not tools:
        return None
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            result.append(tool)
        elif hasattr(tool, "model_dump"):
            result.append(tool.model_dump(exclude_none=True))
        else:
            result.append(dict(tool))
    return result


def convert_tool_choice_to_internal(
    tool_choice: str | Any | None,
) -> str | dict[str, Any] | None:
    """Convert tool_choice to internal format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        return tool_choice
    if hasattr(tool_choice, "model_dump"):
        return tool_choice.model_dump(exclude_none=True)
    return dict(tool_choice)


def make_text_content(text: str) -> dict[str, Any]:
    """Create output_text content block."""
    return {"type": "output_text", "text": text, "annotations": []}


def make_usage(raw_usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Create properly structured usage object matching OpenAI spec."""
    if not raw_usage:
        return None

    input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("output_tokens", 0)
    upstream_input_details = raw_usage.get("input_tokens_details") or {}
    upstream_output_details = raw_usage.get("output_tokens_details") or {}

    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": upstream_input_details.get("cached_tokens", 0),
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": upstream_output_details.get("reasoning_tokens", 0),
        },
        "total_tokens": input_tokens + output_tokens,
    }


def make_message_item(msg_id: str, text: str, status: str = "completed") -> dict[str, Any]:
    """Create message output item."""
    return {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "content": [make_text_content(text)],
        "status": status,
    }


def make_function_call_item(
    fc_id: str,
    call_id: str,
    name: str,
    arguments: str,
    status: str = "completed",
    namespace: str | None = None,
) -> dict[str, Any]:
    """Create function_call output item."""
    item: dict[str, Any] = {
        "type": "function_call",
        "id": fc_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }
    if namespace is not None:
        item["namespace"] = namespace
    return item


@router.post("/api/openai/v1/responses")
async def create_response(request: ResponsesRequest):
    """Handle Responses API requests (for Codex models)."""
    request_id = generate_id("req")
    start_time = time.time()

    logger.info(
        "Received responses request: req_id=%s, model=%s, stream=%s, has_tools=%s, reasoning=%s",
        request_id,
        request.model,
        request.stream,
        request.tools is not None,
        request.reasoning,
    )

    model_router = get_router()

    input_value = convert_input_to_internal(request.input)

    internal_request = InternalResponsesRequest(
        model=request.model,
        input=input_value,
        stream=request.stream,
        instructions=request.instructions,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        tools=convert_tools_to_internal(request.tools),
        tool_choice=convert_tool_choice_to_internal(request.tool_choice),
        parallel_tool_calls=request.parallel_tool_calls,
    )

    if request.reasoning:
        effort = str(request.reasoning.get("effort", "")).lower() or None
        if effort and effort in VALID_EFFORTS:
            internal_request.reasoning_effort = effort

    pre_rewrite_model = internal_request.model
    pre_rewrite_effort = internal_request.reasoning_effort
    internal_request = await model_router.rewrite_to_reasoning_variant(internal_request)
    logger.info(
        "Reasoning resolved: req_id=%s pre_model=%s pre_effort=%s post_model=%s post_effort=%s",
        request_id,
        pre_rewrite_model,
        pre_rewrite_effort,
        internal_request.model,
        internal_request.reasoning_effort,
    )

    if request.stream:
        return sse_streaming_response(
            stream_response(model_router, internal_request, request_id, start_time),
        )

    try:
        response, provider_name = await model_router.responses_completion(internal_request)

        usage = None
        if response.usage:
            usage = ResponsesUsage(
                input_tokens=response.usage.get("input_tokens", 0),
                output_tokens=response.usage.get("output_tokens", 0),
                total_tokens=response.usage.get("total_tokens", 0),
            )

        response_id = generate_id("resp")
        output: list[dict[str, Any]] = []

        # Emit the reasoning item BEFORE the message so the (id, blob) pair
        # round-trips back to Copilot intact. ``thinking_id`` must be the
        # upstream id (Copilot signs ``thinking_signature`` against it) — see
        # base.py ResponsesResponse for the rationale.
        if response.thinking_id or response.thinking:
            reasoning_item: dict[str, Any] = {
                "type": "reasoning",
                "id": response.thinking_id or generate_id("rs"),
                "summary": (
                    [{"type": "summary_text", "text": response.thinking}]
                    if response.thinking
                    else []
                ),
            }
            if response.thinking_signature:
                reasoning_item["encrypted_content"] = response.thinking_signature
            output.append(reasoning_item)

        if response.content:
            message_id = generate_id("msg")
            output.append(make_message_item(message_id, response.content))

        if response.tool_calls:
            for tc in response.tool_calls:
                fc_id = generate_id("fc")
                output.append(
                    make_function_call_item(
                        fc_id,
                        tc.call_id,
                        tc.name,
                        tc.arguments,
                        namespace=tc.namespace,
                    )
                )

        return ResponsesResponse(
            id=response_id,
            model=response.model,
            status="completed",
            output=output,
            usage=usage,
        )
    except ProviderError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Responses request failed: req_id=%s, elapsed=%.1fms, error=%s",
            request_id,
            elapsed_ms,
            e,
        )
        raise HTTPException(status_code=e.status_code, detail=str(e))


@dataclass
class _StreamMessageState:
    """Tracks the open message during Responses API streaming."""

    output_items: list[dict[str, Any]] = field(default_factory=list)
    output_index: int = 0
    content_index: int = 0
    current_message_id: str | None = None
    accumulated_content: str = ""
    message_started: bool = False
    # Reasoning summary state. ``current_reasoning_id`` is the upstream
    # reasoning item id (e.g. ``rs_…``) when Copilot supplies it on the
    # first ``reasoning_summary_text.delta``, falling back to a locally-
    # generated ``rs-…`` only if the upstream never sent one.
    # ``upstream_encrypted_content`` is the verifiable blob from
    # ``output_item.done.item.encrypted_content``; it is signed against the
    # upstream id, so the (id, blob) pair MUST round-trip together — pairing
    # the blob with a local id 400s the next turn with ``Encrypted content
    # could not be decrypted``.
    current_reasoning_id: str | None = None
    upstream_encrypted_content: str | None = None
    accumulated_reasoning: str = ""
    reasoning_started: bool = False
    summary_index: int = 0
    reasoning_item_count: int = 0

    def close_open_message(self, advance_index: bool = True) -> list[str]:
        """Close the currently open message.

        Returns SSE events to yield. Does nothing if no message is open.
        """
        if not self.message_started or not self.current_message_id:
            return []
        events = _close_message_events(
            self.current_message_id,
            self.output_index,
            self.content_index,
            self.accumulated_content,
        )
        self.output_items.append(
            make_message_item(self.current_message_id, self.accumulated_content)
        )
        if advance_index:
            self.output_index += 1
        self.message_started = False
        self.current_message_id = None
        return events

    def close_open_reasoning(self, advance_index: bool = True) -> list[str]:
        """Close any open reasoning summary block + reasoning item.

        Emits the OpenAI Responses-API event sequence:
          response.reasoning_summary_text.done
          response.reasoning_summary_part.done
          response.output_item.done (item.type == "reasoning")
        """
        if not self.reasoning_started or not self.current_reasoning_id:
            return []
        rs_id = self.current_reasoning_id
        text = self.accumulated_reasoning
        summary_part = {"type": "summary_text", "text": text}
        reasoning_item = {
            "type": "reasoning",
            "id": rs_id,
            "summary": [summary_part],
        }
        if self.upstream_encrypted_content:
            reasoning_item["encrypted_content"] = self.upstream_encrypted_content
        events = [
            sse_event(
                {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": rs_id,
                    "output_index": self.output_index,
                    "summary_index": self.summary_index,
                    "text": text,
                }
            ),
            sse_event(
                {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": rs_id,
                    "output_index": self.output_index,
                    "summary_index": self.summary_index,
                    "part": summary_part,
                }
            ),
            sse_event(
                {
                    "type": "response.output_item.done",
                    "output_index": self.output_index,
                    "item": reasoning_item,
                }
            ),
        ]
        self.output_items.append(reasoning_item)
        self.reasoning_item_count += 1
        if advance_index:
            self.output_index += 1
        self.reasoning_started = False
        self.current_reasoning_id = None
        self.upstream_encrypted_content = None
        self.accumulated_reasoning = ""
        self.summary_index = 0
        return events


async def stream_response(
    model_router: Router,
    request: InternalResponsesRequest,
    request_id: str,
    start_time: float,
) -> AsyncGenerator[str, None]:
    """Stream Responses API response."""
    try:
        stream, provider_name = await model_router.responses_completion_stream(request)
        response_id = generate_id("resp")
        created_at = int(time.time())

        logger.debug(
            "Stream started: req_id=%s, resp_id=%s, provider=%s",
            request_id,
            response_id,
            provider_name,
        )

        # Base response object with all required fields (matching OpenAI spec)
        base_response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": request.model,
            "error": None,
            "incomplete_details": None,
        }

        state = _StreamMessageState()
        final_usage = None
        stream_completed = False

        # response.created
        yield sse_event(
            {
                "type": "response.created",
                "response": {
                    **base_response,
                    "status": "in_progress",
                    "output": [],
                },
            }
        )

        # response.in_progress
        yield sse_event(
            {
                "type": "response.in_progress",
                "response": {
                    **base_response,
                    "status": "in_progress",
                    "output": [],
                },
            }
        )

        async for chunk in stream:
            # Handle reasoning summary text deltas. gpt-5.x and other thinking
            # models stream chain-of-thought via these events; if we don't
            # forward them, Codex sees only the final user-visible message and
            # the model appears to "stop without doing anything" because all
            # of its planning tokens vanished. We open a separate
            # ``reasoning`` output item so messages and tool_calls keep their
            # own item indices.
            if chunk.thinking:
                if state.message_started:
                    for evt in state.close_open_message():
                        yield evt
                if not state.reasoning_started:
                    # Prefer the upstream reasoning item id (e.g. ``rs_…``)
                    # supplied on the delta event. Copilot signs the
                    # encrypted_content blob against this id, so emitting a
                    # locally-generated id would mismatch the blob and 400
                    # the next turn.
                    state.current_reasoning_id = chunk.thinking_id or generate_id("rs")
                    state.reasoning_started = True
                    state.summary_index = 0
                    yield sse_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": state.output_index,
                            "item": {
                                "type": "reasoning",
                                "id": state.current_reasoning_id,
                                "summary": [],
                            },
                        }
                    )
                    yield sse_event(
                        {
                            "type": "response.reasoning_summary_part.added",
                            "item_id": state.current_reasoning_id,
                            "output_index": state.output_index,
                            "summary_index": state.summary_index,
                            "part": {"type": "summary_text", "text": ""},
                        }
                    )
                state.accumulated_reasoning += chunk.thinking
                yield sse_event(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": state.current_reasoning_id,
                        "output_index": state.output_index,
                        "summary_index": state.summary_index,
                        "delta": chunk.thinking,
                    }
                )

            # The upstream id may also arrive separately (e.g. on
            # output_item.done before any text deltas arrive). Capture it
            # so close_open_reasoning emits the correct (id, blob) pair.
            if chunk.thinking_id and state.reasoning_started:
                if state.current_reasoning_id != chunk.thinking_id:
                    state.current_reasoning_id = chunk.thinking_id

            if chunk.thinking_signature:
                state.upstream_encrypted_content = chunk.thinking_signature
                for evt in state.close_open_reasoning():
                    yield evt

            # Handle text content
            if chunk.content:
                # Close any open reasoning before switching to a message item
                # so the output indices stay monotonic and Codex can replay
                # the items in order.
                for evt in state.close_open_reasoning():
                    yield evt
                if not state.message_started:
                    state.current_message_id = generate_id("msg")
                    state.message_started = True

                    # Note: content starts as empty array, matching OpenAI spec
                    yield sse_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": state.output_index,
                            "item": {
                                "type": "message",
                                "id": state.current_message_id,
                                "role": "assistant",
                                "content": [],
                                "status": "in_progress",
                            },
                        }
                    )

                    yield sse_event(
                        {
                            "type": "response.content_part.added",
                            "item_id": state.current_message_id,
                            "output_index": state.output_index,
                            "content_index": state.content_index,
                            "part": make_text_content(""),
                        }
                    )

                state.accumulated_content += chunk.content

                yield sse_event(
                    {
                        "type": "response.output_text.delta",
                        "item_id": state.current_message_id,
                        "output_index": state.output_index,
                        "content_index": state.content_index,
                        "delta": chunk.content,
                    }
                )

            # Handle complete tool call
            if chunk.tool_call:
                tc = chunk.tool_call
                for evt in state.close_open_reasoning():
                    yield evt
                for evt in state.close_open_message():
                    yield evt

                if tc.kind == "custom":
                    # Custom tool call (e.g. apply_patch) — free-form text input,
                    # not JSON arguments. Round-trip as custom_tool_call so codex
                    # parses ``input`` as raw text.
                    ctc_id = generate_id("ctc")
                    ctc_item = {
                        "type": "custom_tool_call",
                        "id": ctc_id,
                        "call_id": tc.call_id,
                        "name": tc.name,
                        "input": tc.arguments,
                        "status": "completed",
                    }
                    yield sse_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": state.output_index,
                            "item": {
                                **ctc_item,
                                "input": "",
                                "status": "in_progress",
                            },
                        }
                    )
                    yield sse_event(
                        {
                            "type": "response.custom_tool_call_input.delta",
                            "item_id": ctc_id,
                            "output_index": state.output_index,
                            "delta": tc.arguments,
                        }
                    )
                    yield sse_event(
                        {
                            "type": "response.custom_tool_call_input.done",
                            "item_id": ctc_id,
                            "output_index": state.output_index,
                            "input": tc.arguments,
                        }
                    )
                    yield sse_event(
                        {
                            "type": "response.output_item.done",
                            "output_index": state.output_index,
                            "item": ctc_item,
                        }
                    )
                    state.output_items.append(ctc_item)
                    state.output_index += 1
                elif tc.kind == "tool_search":
                    # Codex's MCP tool-discovery dispatcher matches on
                    # ResponseItem::ToolSearchCall (codex-rs/core/src/tools/
                    # router.rs). Wrapping this as a function_call(name=
                    # "tool_search") makes the dispatcher silently abort the
                    # call (the registry has no function tool of that name)
                    # and Codex writes ``output: 'aborted'`` to the conversation
                    # — the model retries forever (v0.3.5/v0.3.6 bug).
                    # Codex only requires output_item.done with the full
                    # item; arguments must be a dict, not a JSON string.
                    try:
                        args_obj = json.loads(tc.arguments) if tc.arguments else {}
                    except (TypeError, ValueError):
                        args_obj = {}
                    tsc_item = {
                        "type": "tool_search_call",
                        "call_id": tc.call_id,
                        "execution": "client",
                        "status": "completed",
                        "arguments": args_obj,
                    }
                    yield sse_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": state.output_index,
                            "item": {
                                **tsc_item,
                                "status": "in_progress",
                                "arguments": {},
                            },
                        }
                    )
                    yield sse_event(
                        {
                            "type": "response.output_item.done",
                            "output_index": state.output_index,
                            "item": tsc_item,
                        }
                    )
                    state.output_items.append(tsc_item)
                    state.output_index += 1
                else:
                    fc_id = generate_id("fc")
                    fc_item = make_function_call_item(
                        fc_id,
                        tc.call_id,
                        tc.name,
                        tc.arguments,
                        namespace=tc.namespace,
                    )

                    yield sse_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": state.output_index,
                            "item": make_function_call_item(
                                fc_id,
                                tc.call_id,
                                tc.name,
                                "",
                                "in_progress",
                                namespace=tc.namespace,
                            ),
                        }
                    )

                    yield sse_event(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": fc_id,
                            "output_index": state.output_index,
                            "delta": tc.arguments,
                        }
                    )

                    yield sse_event(
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": fc_id,
                            "output_index": state.output_index,
                            "arguments": tc.arguments,
                        }
                    )

                    yield sse_event(
                        {
                            "type": "response.output_item.done",
                            "output_index": state.output_index,
                            "item": fc_item,
                        }
                    )

                    state.output_items.append(fc_item)
                    state.output_index += 1

            if chunk.usage:
                final_usage = chunk.usage

            if chunk.finish_reason:
                stream_completed = True

                for evt in state.close_open_reasoning():
                    yield evt
                for evt in state.close_open_message(advance_index=False):
                    yield evt

                yield sse_event(
                    {
                        "type": "response.completed",
                        "response": {
                            **base_response,
                            "status": "completed",
                            "output": state.output_items,
                            "usage": make_usage(final_usage),
                        },
                    }
                )

        if not stream_completed:
            logger.warning(
                "Stream ended without finish_reason: req_id=%s model=%s output_items=%d "
                "accumulated_text_len=%d message_started=%s — emitting fallback completion",
                request_id,
                request.model,
                len(state.output_items),
                len(state.accumulated_content),
                state.message_started,
            )

            for evt in state.close_open_reasoning():
                yield evt
            for evt in state.close_open_message(advance_index=False):
                yield evt

            yield sse_event(
                {
                    "type": "response.completed",
                    "response": {
                        **base_response,
                        "status": "completed",
                        "output": state.output_items,
                        "usage": make_usage(final_usage),
                    },
                }
            )

        elapsed_ms = (time.time() - start_time) * 1000
        function_call_count = sum(
            1 for item in state.output_items if item.get("type") == "function_call"
        )
        message_count = sum(1 for item in state.output_items if item.get("type") == "message")
        reasoning_count = sum(1 for item in state.output_items if item.get("type") == "reasoning")
        logger.info(
            "Stream completed: req_id=%s model=%s elapsed=%.1fms output_items=%d "
            "(messages=%d, function_calls=%d, reasoning=%d) text_len=%d "
            "reasoning_text_len=%d stream_completed=%s usage=%s",
            request_id,
            request.model,
            elapsed_ms,
            len(state.output_items),
            message_count,
            function_call_count,
            reasoning_count,
            len(state.accumulated_content),
            sum(
                len(part.get("text", ""))
                for item in state.output_items
                if item.get("type") == "reasoning"
                for part in item.get("summary", [])
            ),
            stream_completed,
            final_usage,
        )

        # NOTE: Do NOT send "data: [DONE]\n\n" - agent-maestro doesn't send it
        # for Responses API

    except ProviderError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Stream failed: req_id=%s, elapsed=%.1fms, error=%s",
            request_id,
            elapsed_ms,
            e,
        )
        # Send response.failed event matching OpenAI spec
        yield sse_event(
            {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "failed",
                    "created_at": created_at,
                    "model": request.model,
                    "output": [],
                    "error": {
                        "code": "server_error",
                        "message": str(e),
                    },
                    "incomplete_details": None,
                },
            }
        )
    except asyncio.CancelledError:
        logger.info("Responses stream cancelled by client: req_id=%s", request_id)
        raise
    except Exception:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Unexpected error in responses stream: req_id=%s, elapsed=%.1fms",
            request_id,
            elapsed_ms,
            exc_info=True,
        )
        fallback_id = generate_id("resp")
        yield sse_event(
            {
                "type": "response.failed",
                "response": {
                    "id": fallback_id,
                    "object": "response",
                    "status": "failed",
                    "created_at": int(time.time()),
                    "model": request.model,
                    "output": [],
                    "error": {
                        "code": "server_error",
                        "message": "Internal server error",
                    },
                    "incomplete_details": None,
                },
            }
        )


def _close_message_events(
    msg_id: str, output_index: int, content_index: int, text: str
) -> list[str]:
    """Generate events to close a message output item."""
    return [
        sse_event(
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": content_index,
                "text": text,
            }
        ),
        sse_event(
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": make_text_content(text),
            }
        ),
        sse_event(
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": make_message_item(msg_id, text),
            }
        ),
    ]
