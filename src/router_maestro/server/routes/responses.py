"""Responses API route for Codex models."""

import asyncio
import json
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from router_maestro.providers import (
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesStreamChunk,
    ResponseStatus,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
    client_cancelled_outcome,
    exception_outcome,
    resolve_terminal_outcome,
    unexpected_eof_outcome,
)
from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.routing import Router, get_router
from router_maestro.routing.model_ref import qualify_model_id
from router_maestro.server.protocols import client_error_response, unrepresented_option_error
from router_maestro.server.schemas import (
    ResponsesReasoningConfig,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.utils import get_logger
from router_maestro.utils.async_iterators import close_async_iterator

logger = get_logger("server.routes.responses")

router = APIRouter()


def _reasoning_validation_error(error: ValidationError) -> RequestOptionError:
    """Map strict Responses reasoning failures to one stable OpenAI parameter."""
    first = error.errors()[0]
    location = first.get("loc", ())
    if location == ("effort",):
        parameter = "reasoning_effort"
    elif location:
        parameter = f"reasoning.{'.'.join(str(part) for part in location)}"
    else:
        parameter = "reasoning"
    return RequestOptionError(
        f"Invalid request option '{parameter}'",
        parameter=parameter,
    )


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


def make_refusal_content(refusal: str) -> dict[str, Any]:
    """Create a refusal content block."""
    return {"type": "refusal", "refusal": refusal}


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


def _make_message_item_from_parts(
    msg_id: str,
    content: list[dict[str, Any]],
    status: str = "completed",
) -> dict[str, Any]:
    """Create a message output item from typed content parts."""
    return {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "content": content,
        "status": status,
    }


def make_message_item(
    msg_id: str,
    text: str | None,
    status: str = "completed",
    *,
    refusal: str | None = None,
) -> dict[str, Any]:
    """Create message output item."""
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append(make_text_content(text))
    if refusal is not None:
        content.append(make_refusal_content(refusal))
    return _make_message_item_from_parts(msg_id, content, status)


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


def make_custom_tool_call_item(
    ctc_id: str,
    call_id: str,
    name: str,
    input_text: str,
    status: str = "completed",
) -> dict[str, Any]:
    """Create custom_tool_call output item."""
    return {
        "type": "custom_tool_call",
        "id": ctc_id,
        "call_id": call_id,
        "name": name,
        "input": input_text,
        "status": status,
    }


def parse_tool_search_arguments(arguments: str) -> dict[str, Any]:
    """Parse tool_search arguments into the dict shape clients expect."""
    try:
        parsed = json.loads(arguments) if arguments else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def make_tool_search_call_item(call_id: str, arguments: str) -> dict[str, Any]:
    """Create tool_search_call output item."""
    return {
        "type": "tool_search_call",
        "call_id": call_id,
        "execution": "client",
        "status": "completed",
        "arguments": parse_tool_search_arguments(arguments),
    }


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
    if error := unrepresented_option_error(request):
        return client_error_response(error, "openai")

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
        top_p=request.top_p,
        metadata=request.metadata,
        service_tier=request.service_tier,
    )

    if request.reasoning is not None:
        try:
            reasoning = ResponsesReasoningConfig.model_validate(request.reasoning)
        except ValidationError as error:
            return client_error_response(_reasoning_validation_error(error), "openai")
        internal_request.reasoning_effort = reasoning.effort

    if request.stream:
        try:
            prepared_plan = await model_router.prepare_responses_completion_stream(internal_request)
        except ProviderError as e:
            if isinstance(e, RequestOptionError):
                return client_error_response(e, "openai")
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        return sse_streaming_response(
            stream_response(
                model_router,
                internal_request,
                request_id,
                start_time,
                prepared_plan=prepared_plan,
            ),
        )

    try:
        response, provider_name = await model_router.responses_completion(internal_request)

        usage = None
        if response.usage:
            usage = ResponsesUsage(
                input_tokens=response.usage.get("input_tokens", 0),
                output_tokens=response.usage.get("output_tokens", 0),
                total_tokens=response.usage.get("total_tokens", 0),
                input_tokens_details=response.usage.get("input_tokens_details"),
                output_tokens_details=response.usage.get("output_tokens_details"),
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

        if response.content or response.refusal:
            message_id = generate_id("msg")
            output.append(
                make_message_item(
                    message_id,
                    response.content or None,
                    refusal=response.refusal,
                )
            )

        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.kind == "custom":
                    output.append(
                        make_custom_tool_call_item(
                            generate_id("ctc"),
                            tc.call_id,
                            tc.name,
                            tc.arguments,
                        )
                    )
                elif tc.kind == "tool_search":
                    output.append(make_tool_search_call_item(tc.call_id, tc.arguments))
                else:
                    output.append(
                        make_function_call_item(
                            generate_id("fc"),
                            tc.call_id,
                            tc.name,
                            tc.arguments,
                            namespace=tc.namespace,
                        )
                    )

        outcome = resolve_terminal_outcome(response.terminal_outcome, None)
        if outcome is None:
            outcome = TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.COMPLETED,
            )
        elif outcome.transport is not TransportTermination.EXPLICIT_TERMINAL:
            outcome = TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.FAILED,
                error=TerminalError(
                    code="upstream_protocol_error",
                    message="Provider returned an invalid non-stream terminal outcome",
                ),
            )
        error = (
            {"code": outcome.error.code, "message": outcome.error.message}
            if outcome.error is not None
            else None
        )

        return ResponsesResponse.model_construct(
            id=response_id,
            object="response",
            model=(
                response.selected_model.qualified_id
                if response.selected_model is not None
                else qualify_model_id(provider_name, response.model)
            ),
            status=outcome.response_status.value,
            output=output,
            usage=usage,
            error=error,
            incomplete_details=outcome.incomplete_details,
        )
    except ProviderError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Responses request failed: req_id=%s, elapsed=%.1fms, error=%s",
            request_id,
            elapsed_ms,
            e,
        )
        if isinstance(e, RequestOptionError):
            return client_error_response(e, "openai")
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e


@dataclass
class _StreamMessageState:
    """Tracks the open message during Responses API streaming."""

    output_items: list[dict[str, Any]] = field(default_factory=list)
    output_index: int = 0
    content_index: int = 0
    current_message_id: str | None = None
    current_source_content_index: int | None = None
    accumulated_content: str = ""
    accumulated_refusal: str = ""
    message_started: bool = False
    current_part_type: str | None = None
    message_content_parts: list[dict[str, Any]] = field(default_factory=list)
    # Reasoning summary state. Deltas remain buffered until the upstream
    # identity arrives or the item must close. Only then may we emit the
    # added/delta/done sequence under one stable id. A local id is generated
    # only at a boundary when the upstream never supplied one.
    # ``upstream_encrypted_content`` is the verifiable blob from
    # ``output_item.done.item.encrypted_content``; it is signed against the
    # upstream id, so the (id, blob) pair MUST round-trip together — pairing
    # the blob with a local id 400s the next turn with ``Encrypted content
    # could not be decrypted``.
    current_reasoning_id: str | None = None
    pending_reasoning_id: str | None = None
    upstream_encrypted_content: str | None = None
    reasoning_fragments: dict[int, list[str]] = field(default_factory=dict)
    reasoning_started: bool = False
    reasoning_events_started: bool = False
    reasoning_emission_offsets: dict[int, int] = field(default_factory=dict)
    dirty_reasoning_indices: list[int] = field(default_factory=list)
    _dirty_reasoning_index_set: set[int] = field(default_factory=set)
    flush_reasoning_scan_count: int = 0
    emitted_reasoning_parts: set[int] = field(default_factory=set)
    max_reasoning_summary_index: int | None = None
    visible_reasoning_summary_indices: dict[int, int] = field(default_factory=dict)
    reasoning_item_count: int = 0

    def begin_reasoning(self) -> None:
        """Open logical reasoning state without committing a wire identity."""
        if self.reasoning_started:
            return
        self.reasoning_started = True

    def bind_reasoning_id(self, reasoning_id: str) -> None:
        """Bind one upstream identity before any reasoning event is emitted."""
        self.begin_reasoning()
        if self.pending_reasoning_id is not None and self.pending_reasoning_id != reasoning_id:
            raise ProviderError(
                "Responses reasoning item changed identity",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if self.current_reasoning_id is not None and self.current_reasoning_id != reasoning_id:
            raise ProviderError(
                "Responses reasoning item changed identity",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        self.current_reasoning_id = reasoning_id
        self.pending_reasoning_id = None

    def latch_reasoning_id(self, reasoning_id: str) -> None:
        """Remember provenance identity without opening a visible reasoning item."""
        if self.current_reasoning_id is not None and self.current_reasoning_id != reasoning_id:
            raise ProviderError(
                "Responses reasoning item changed identity",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if self.pending_reasoning_id is not None and self.pending_reasoning_id != reasoning_id:
            raise ProviderError(
                "Responses reasoning item changed identity",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        self.pending_reasoning_id = reasoning_id

    def append_reasoning(self, delta: str, summary_index: int | None) -> None:
        """Accumulate one delta and defer it until the item identity is stable."""
        self.begin_reasoning()
        if self.pending_reasoning_id is not None:
            self.bind_reasoning_id(self.pending_reasoning_id)
        index = 0 if summary_index is None else summary_index
        self.bind_reasoning_summary_index(index)
        if index not in self.visible_reasoning_summary_indices:
            self.visible_reasoning_summary_indices[index] = len(
                self.visible_reasoning_summary_indices
            )
        self.reasoning_fragments.setdefault(index, []).append(delta)
        if index not in self._dirty_reasoning_index_set:
            self._dirty_reasoning_index_set.add(index)
            self.dirty_reasoning_indices.append(index)

    def bind_reasoning_summary_index(self, summary_index: int | None) -> None:
        """Validate provenance without opening an empty reasoning output item."""
        index = 0 if summary_index is None else summary_index
        maximum = self.max_reasoning_summary_index
        if maximum is None and index != 0:
            raise ProviderError(
                "Responses reasoning summary_index must start at zero",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if maximum is not None and index > maximum + 1:
            raise ProviderError(
                "Responses reasoning summary_index contains a gap",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if maximum is None or index > maximum:
            self.max_reasoning_summary_index = index

    def start_reasoning_events(self, *, allow_local_id: bool = False) -> list[str]:
        """Emit added events and buffered deltas once one stable id is available."""
        if not self.reasoning_started or self.reasoning_events_started:
            return []
        if self.current_reasoning_id is None:
            if not allow_local_id:
                return []
            self.current_reasoning_id = generate_id("rs")

        rs_id = self.current_reasoning_id
        self.reasoning_events_started = True
        events = [
            sse_event(
                {
                    "type": "response.output_item.added",
                    "output_index": self.output_index,
                    "item": {"type": "reasoning", "id": rs_id, "summary": []},
                }
            ),
        ]
        dirty_indices = self.dirty_reasoning_indices
        self.dirty_reasoning_indices = []
        self._dirty_reasoning_index_set = set()
        for source_index in dirty_indices:
            deltas = self.reasoning_fragments[source_index]
            summary_index = self.visible_reasoning_summary_indices[source_index]
            events.append(
                sse_event(
                    {
                        "type": "response.reasoning_summary_part.added",
                        "item_id": rs_id,
                        "output_index": self.output_index,
                        "summary_index": summary_index,
                        "part": {"type": "summary_text", "text": ""},
                    }
                )
            )
            self.emitted_reasoning_parts.add(source_index)
            self.reasoning_emission_offsets[source_index] = len(deltas)
            events.extend(
                sse_event(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": rs_id,
                        "output_index": self.output_index,
                        "summary_index": summary_index,
                        "delta": delta,
                    }
                )
                for delta in deltas
            )
        return events

    def flush_reasoning_events(self) -> list[str]:
        """Emit deltas received after the reasoning item acquired its id."""
        if not self.reasoning_events_started or self.current_reasoning_id is None:
            return []
        events: list[str] = []
        dirty_indices = self.dirty_reasoning_indices
        self.dirty_reasoning_indices = []
        self._dirty_reasoning_index_set = set()
        for source_index in dirty_indices:
            self.flush_reasoning_scan_count += 1
            deltas = self.reasoning_fragments[source_index]
            summary_index = self.visible_reasoning_summary_indices[source_index]
            if source_index not in self.emitted_reasoning_parts:
                events.append(
                    sse_event(
                        {
                            "type": "response.reasoning_summary_part.added",
                            "item_id": self.current_reasoning_id,
                            "output_index": self.output_index,
                            "summary_index": summary_index,
                            "part": {"type": "summary_text", "text": ""},
                        }
                    )
                )
                self.emitted_reasoning_parts.add(source_index)
            offset = self.reasoning_emission_offsets.get(source_index, 0)
            events.extend(
                sse_event(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": self.current_reasoning_id,
                        "output_index": self.output_index,
                        "summary_index": summary_index,
                        "delta": delta,
                    }
                )
                for delta in deltas[offset:]
            )
            self.reasoning_emission_offsets[source_index] = len(deltas)
        return events

    def start_message_part(self, part_type: str) -> list[str]:
        """Open one typed Responses content part, closing a different active part."""
        if part_type not in {"output_text", "refusal"}:
            raise ValueError(f"unsupported Responses content part type: {part_type}")

        events: list[str] = []
        if not self.message_started:
            self.current_message_id = generate_id("msg")
            self.message_started = True
            events.append(
                sse_event(
                    {
                        "type": "response.output_item.added",
                        "output_index": self.output_index,
                        "item": {
                            "type": "message",
                            "id": self.current_message_id,
                            "role": "assistant",
                            "content": [],
                            "status": "in_progress",
                        },
                    }
                )
            )

        if self.current_part_type == part_type:
            return events
        events.extend(self.close_open_content_part())
        self.current_part_type = part_type
        empty_part = (
            make_text_content("") if part_type == "output_text" else make_refusal_content("")
        )
        events.append(
            sse_event(
                {
                    "type": "response.content_part.added",
                    "item_id": self.current_message_id,
                    "output_index": self.output_index,
                    "content_index": self.content_index,
                    "part": empty_part,
                }
            )
        )
        return events

    def bind_source_content_index(self, source_index: int | None) -> list[str]:
        """Honor native content-part boundaries within one indexed message."""
        if source_index is None:
            return []
        if self.current_source_content_index is None:
            if source_index != 0:
                raise ProviderError(
                    "Responses message content_index must start at zero",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                )
            self.current_source_content_index = source_index
            return []
        if source_index < self.current_source_content_index:
            raise ProviderError(
                "Responses message content_index moved backwards",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if source_index == self.current_source_content_index:
            return []
        if source_index != self.current_source_content_index + 1:
            raise ProviderError(
                "Responses message content_index contains a gap",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        events = self.close_open_content_part()
        self.current_source_content_index = source_index
        return events

    def bind_empty_source_content_index(self, source_index: int | None) -> None:
        """Validate provenance without opening an empty message output item."""
        if source_index is None:
            return
        current = self.current_source_content_index
        if current is None and source_index != 0:
            raise ProviderError(
                "Responses message content_index must start at zero",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if current is not None:
            if source_index < current:
                raise ProviderError(
                    "Responses message content_index moved backwards",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                )
            if source_index > current + 1:
                raise ProviderError(
                    "Responses message content_index contains a gap",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                )
        self.current_source_content_index = source_index

    def close_open_content_part(self) -> list[str]:
        """Close the active output_text or refusal part without closing its message."""
        if self.current_part_type is None or self.current_message_id is None:
            return []

        if self.current_part_type == "output_text":
            value = self.accumulated_content
            part = make_text_content(value)
            done = {
                "type": "response.output_text.done",
                "item_id": self.current_message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": value,
            }
            self.accumulated_content = ""
        else:
            value = self.accumulated_refusal
            part = make_refusal_content(value)
            done = {
                "type": "response.refusal.done",
                "item_id": self.current_message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "refusal": value,
            }
            self.accumulated_refusal = ""

        events = [
            sse_event(done),
            sse_event(
                {
                    "type": "response.content_part.done",
                    "item_id": self.current_message_id,
                    "output_index": self.output_index,
                    "content_index": self.content_index,
                    "part": part,
                }
            ),
        ]
        self.message_content_parts.append(part)
        self.content_index += 1
        self.current_part_type = None
        return events

    def close_open_message(self, advance_index: bool = True) -> list[str]:
        """Close the currently open message.

        Returns SSE events to yield. Does nothing if no message is open.
        """
        if not self.message_started or not self.current_message_id:
            return []
        events = self.close_open_content_part()
        item = _make_message_item_from_parts(
            self.current_message_id,
            list(self.message_content_parts),
        )
        events.append(
            sse_event(
                {
                    "type": "response.output_item.done",
                    "output_index": self.output_index,
                    "item": item,
                }
            )
        )
        self.output_items.append(item)
        if advance_index:
            self.output_index += 1
        self.message_started = False
        self.current_message_id = None
        self.accumulated_content = ""
        self.accumulated_refusal = ""
        self.content_index = 0
        self.current_part_type = None
        self.message_content_parts = []
        self.current_source_content_index = None
        return events

    def close_open_reasoning(self, advance_index: bool = True) -> list[str]:
        """Close any open reasoning summary block + reasoning item.

        Emits the OpenAI Responses-API event sequence:
          response.reasoning_summary_text.done
          response.reasoning_summary_part.done
          response.output_item.done (item.type == "reasoning")
        """
        if not self.reasoning_started:
            return []
        events = self.start_reasoning_events(allow_local_id=True)
        if self.current_reasoning_id is None:
            raise RuntimeError("reasoning id was not established")
        rs_id = self.current_reasoning_id
        indexed_summary_parts = [
            (
                self.visible_reasoning_summary_indices[source_index],
                {"type": "summary_text", "text": text},
            )
            for source_index, fragments in sorted(self.reasoning_fragments.items())
            for text in ["".join(fragments)]
        ]
        reasoning_item = {
            "type": "reasoning",
            "id": rs_id,
            "summary": [part for _, part in indexed_summary_parts],
        }
        if self.upstream_encrypted_content:
            reasoning_item["encrypted_content"] = self.upstream_encrypted_content
        for summary_index, summary_part in indexed_summary_parts:
            events.extend(
                [
                    sse_event(
                        {
                            "type": "response.reasoning_summary_text.done",
                            "item_id": rs_id,
                            "output_index": self.output_index,
                            "summary_index": summary_index,
                            "text": summary_part["text"],
                        }
                    ),
                    sse_event(
                        {
                            "type": "response.reasoning_summary_part.done",
                            "item_id": rs_id,
                            "output_index": self.output_index,
                            "summary_index": summary_index,
                            "part": summary_part,
                        }
                    ),
                ]
            )
        events.append(
            sse_event(
                {
                    "type": "response.output_item.done",
                    "output_index": self.output_index,
                    "item": reasoning_item,
                }
            )
        )
        self.output_items.append(reasoning_item)
        self.reasoning_item_count += 1
        if advance_index:
            self.output_index += 1
        self.reasoning_started = False
        self.reasoning_events_started = False
        self.current_reasoning_id = None
        self.pending_reasoning_id = None
        self.upstream_encrypted_content = None
        self.reasoning_fragments = {}
        self.reasoning_emission_offsets = {}
        self.dirty_reasoning_indices = []
        self._dirty_reasoning_index_set = set()
        self.flush_reasoning_scan_count = 0
        self.emitted_reasoning_parts = set()
        self.max_reasoning_summary_index = None
        self.visible_reasoning_summary_indices = {}
        return events

    def finish_empty_reasoning_item(self) -> None:
        """Clear marker-only provenance at its indexed item boundary."""
        if self.reasoning_started:
            return
        self.pending_reasoning_id = None
        self.max_reasoning_summary_index = None

    def finish_empty_message_item(self) -> None:
        """Clear marker-only content provenance at its indexed item boundary."""
        if self.message_started:
            return
        self.current_source_content_index = None


@dataclass
class _IndexedOutputBucket:
    """Canonical chunks belonging to one upstream Responses output item."""

    item_type: str
    chunks: list[ResponsesStreamChunk] = field(default_factory=list)
    done: bool = False
    buffered_chunk_count: int = 0
    buffered_payload_bytes: int = 0


@dataclass
class _IndexedOutputScheduler:
    """Release interleaved native output items in source ``output_index`` order."""

    buckets: dict[int, _IndexedOutputBucket] = field(default_factory=dict)
    next_output_index: int = 0
    saw_indexed_item: bool = False
    max_future_buckets: int = 64
    max_future_chunks: int = 4096
    max_future_payload_bytes: int = 8 * 1024 * 1024
    future_chunk_count: int = 0
    future_payload_bytes: int = 0

    @staticmethod
    def _protocol_error(message: str) -> ProviderError:
        return ProviderError(
            message,
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        )

    def add(self, chunk: ResponsesStreamChunk) -> list[ResponsesStreamChunk]:
        """Release the active item immediately and buffer only future items."""
        output_index = chunk.output_index
        if output_index is None:
            return [chunk]
        self.saw_indexed_item = True
        if output_index < 0:
            raise self._protocol_error("Responses output_index must be non-negative")
        if output_index < self.next_output_index:
            raise self._protocol_error("Responses output item changed after completion")
        if output_index - self.next_output_index > self.max_future_buckets:
            raise self._protocol_error("Responses future output index window limit exceeded")
        item_type = chunk.output_item_type
        if not item_type:
            raise self._protocol_error("Responses indexed chunk is missing its output item type")

        bucket = self.buckets.get(output_index)
        if bucket is None:
            bucket = _IndexedOutputBucket(item_type=item_type)
            self.buckets[output_index] = bucket
        elif bucket.item_type != item_type:
            raise self._protocol_error("Responses output index changed item type")
        if bucket.done:
            raise self._protocol_error("Responses output item received data after completion")

        bucket.chunks.append(chunk)
        bucket.done = chunk.output_item_done
        if output_index > self.next_output_index:
            payload_size = self._chunk_payload_size(chunk)
            bucket.buffered_chunk_count += 1
            bucket.buffered_payload_bytes += payload_size
            self.future_chunk_count += 1
            self.future_payload_bytes += payload_size
            if self._future_bucket_count() > self.max_future_buckets:
                raise self._protocol_error("Responses future output item buffer limit exceeded")
            if self.future_chunk_count > self.max_future_chunks:
                raise self._protocol_error("Responses future output chunk limit exceeded")
            if self.future_payload_bytes > self.max_future_payload_bytes:
                raise self._protocol_error("Responses future output payload limit exceeded")
        return self._drain_available()

    def finalize(self) -> list[ResponsesStreamChunk]:
        """Close and drain every buffered item at explicit terminal or EOF."""
        if not self.buckets:
            return []
        highest_index = max(self.buckets)
        missing = [
            index
            for index in range(self.next_output_index, highest_index + 1)
            if index not in self.buckets
        ]
        if missing:
            raise self._protocol_error(f"Responses output item sequence has gaps: {missing}")
        for bucket in self.buckets.values():
            if not bucket.done:
                if bucket.chunks:
                    bucket.chunks[-1] = replace(bucket.chunks[-1], output_item_done=True)
                else:
                    bucket.chunks.append(
                        ResponsesStreamChunk(
                            content="",
                            output_index=self.next_output_index,
                            output_item_type=bucket.item_type,
                            output_item_done=True,
                        )
                    )
                bucket.done = True
        return self._drain_available()

    def _drain_available(self) -> list[ResponsesStreamChunk]:
        ready: list[ResponsesStreamChunk] = []
        while True:
            bucket = self.buckets.get(self.next_output_index)
            if bucket is None:
                break
            ready.extend(bucket.chunks)
            self.future_chunk_count -= bucket.buffered_chunk_count
            bucket.buffered_chunk_count = 0
            bucket.chunks = []
            self.future_payload_bytes -= bucket.buffered_payload_bytes
            bucket.buffered_payload_bytes = 0
            if not bucket.done:
                break
            del self.buckets[self.next_output_index]
            self.next_output_index += 1
        return ready

    def _future_bucket_count(self) -> int:
        return sum(index > self.next_output_index for index in self.buckets)

    @staticmethod
    def _chunk_payload_size(chunk: ResponsesStreamChunk) -> int:
        """Approximate retained payload bytes without serializing arbitrary objects."""
        size = len(chunk.content.encode("utf-8"))
        for value in (chunk.refusal, chunk.thinking, chunk.thinking_id, chunk.thinking_signature):
            if value:
                size += len(value.encode("utf-8"))
        if chunk.tool_call is not None:
            for value in (
                chunk.tool_call.call_id,
                chunk.tool_call.name,
                chunk.tool_call.arguments,
                chunk.tool_call.namespace,
            ):
                if value:
                    size += len(value.encode("utf-8"))
        if chunk.usage is not None:
            size += len(json.dumps(chunk.usage, separators=(",", ":"), default=str).encode("utf-8"))
        return size


async def stream_response(
    model_router: Router,
    request: InternalResponsesRequest,
    request_id: str,
    start_time: float,
    *,
    prepared_plan=None,
) -> AsyncGenerator[str, None]:
    """Stream Responses API response."""
    # Generate these before the try so the except handlers can always reference
    # them even if responses_completion_stream() raises before returning.
    response_id = generate_id("resp")
    created_at = int(time.time())
    response_model = request.model
    pipeline = None
    stream = None
    terminal_outcome: TerminalOutcome | None = None
    try:
        if prepared_plan is None:
            stream, provider_name = await model_router.responses_completion_stream(request)
        else:
            stream, provider_name = await model_router.responses_completion_stream(
                request,
                prepared_plan=prepared_plan,
            )
        selected_model = getattr(stream, "selected_model", None)
        response_model = (
            selected_model.qualified_id
            if selected_model is not None
            else qualify_model_id(provider_name, request.model)
        )

        # Unified pipeline: guards + audit
        from router_maestro.pipeline import RequestPipeline

        pipeline = RequestPipeline.create(
            request_id=request_id,
            model=request.model,
        )

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
            "model": response_model,
            "error": None,
            "incomplete_details": None,
        }

        state = _StreamMessageState()
        final_usage = None
        deferred_chunks = []
        chunk_queue = deque()
        indexed_outputs = _IndexedOutputScheduler()
        saw_unindexed_output = False
        stream_iterator = stream.__aiter__()
        stream_exhausted = False

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

        while chunk_queue or not stream_exhausted:
            if chunk_queue:
                chunk, feed_pipeline = chunk_queue.popleft()
            else:
                try:
                    chunk = await stream_iterator.__anext__()
                except StopAsyncIteration:
                    stream_exhausted = True
                    indexed_ready = indexed_outputs.finalize()
                    if indexed_ready:
                        chunk_queue.extend((pending, False) for pending in indexed_ready)
                        continue
                    if state.reasoning_started and deferred_chunks:
                        for evt in state.close_open_reasoning():
                            yield evt
                        chunk_queue.extend((pending, False) for pending in deferred_chunks)
                        deferred_chunks = []
                        continue
                    break
                feed_pipeline = True

            # Feed through guards
            abort_reason = pipeline.feed_stream(chunk) if feed_pipeline else None
            if abort_reason:
                terminal_outcome = exception_outcome(abort_reason, code="overloaded")
                logger.warning("Responses stream aborted: %s", abort_reason)
                yield sse_event(
                    {
                        "type": "response.failed",
                        "response": {
                            **base_response,
                            "status": "failed",
                            "output": [],
                            "error": {
                                "type": "server_error",
                                "message": "Overloaded: please retry",
                            },
                        },
                    }
                )
                pipeline.finish(
                    wire_status=200,
                    outcome=terminal_outcome,
                    body_summary=abort_reason,
                )
                return

            has_output_payload = bool(
                chunk.content
                or chunk.refusal
                or chunk.tool_call
                or chunk.thinking
                or chunk.thinking_id
                or chunk.thinking_signature
                or chunk.output_item_type
                or chunk.output_item_done
            )
            if feed_pipeline and chunk.output_index is not None:
                if saw_unindexed_output:
                    raise _IndexedOutputScheduler._protocol_error(
                        "Responses stream mixed indexed and unindexed output items"
                    )
                chunk_is_terminal = (
                    chunk.terminal_outcome is not None or chunk.finish_reason is not None
                )
                control_chunk = None
                indexed_chunk = chunk
                if chunk_is_terminal:
                    control_chunk = replace(
                        chunk,
                        content="",
                        refusal=None,
                        tool_call=None,
                        thinking=None,
                        thinking_id=None,
                        thinking_signature=None,
                        output_index=None,
                        content_index=None,
                        reasoning_summary_index=None,
                        provenance_only=False,
                        output_item_type=None,
                        output_item_done=False,
                    )
                    indexed_chunk = replace(
                        chunk,
                        usage=None,
                        finish_reason=None,
                        terminal_outcome=None,
                    )
                indexed_ready = indexed_outputs.add(indexed_chunk)
                if control_chunk is not None:
                    indexed_ready.extend(indexed_outputs.finalize())
                chunk_queue.extend((pending, False) for pending in indexed_ready)
                if control_chunk is not None:
                    chunk_queue.append((control_chunk, False))
                continue

            if feed_pipeline and chunk.output_index is None and has_output_payload:
                if indexed_outputs.saw_indexed_item:
                    raise _IndexedOutputScheduler._protocol_error(
                        "Responses stream mixed indexed and unindexed output items"
                    )
                saw_unindexed_output = True

            if feed_pipeline and indexed_outputs.saw_indexed_item:
                chunk_is_terminal = (
                    chunk.terminal_outcome is not None or chunk.finish_reason is not None
                )
                if chunk_is_terminal:
                    indexed_ready = indexed_outputs.finalize()
                    if indexed_ready:
                        chunk_queue.extend((pending, False) for pending in indexed_ready)
                        chunk_queue.append((chunk, False))
                        continue

            chunk_is_terminal = (
                chunk.terminal_outcome is not None or chunk.finish_reason is not None
            )
            if chunk.provenance_only:
                if chunk.output_item_type == "reasoning":
                    state.bind_reasoning_summary_index(chunk.reasoning_summary_index)
                    if chunk.thinking_id:
                        state.latch_reasoning_id(chunk.thinking_id)
                elif chunk.output_item_type == "message":
                    state.bind_empty_source_content_index(chunk.content_index)
                if chunk.output_item_done:
                    if chunk.output_item_type == "reasoning":
                        for evt in state.close_open_reasoning():
                            yield evt
                        state.finish_empty_reasoning_item()
                    elif chunk.output_item_type == "message":
                        for evt in state.close_open_message():
                            yield evt
                        state.finish_empty_message_item()
                if not chunk_is_terminal and chunk.usage is None:
                    continue
                chunk = replace(
                    chunk,
                    thinking_id=None,
                    output_index=None,
                    content_index=None,
                    reasoning_summary_index=None,
                    provenance_only=False,
                    output_item_type=None,
                    output_item_done=False,
                )
            chunk_has_reasoning = bool(
                chunk.thinking or chunk.thinking_id or chunk.thinking_signature
            )
            has_non_reasoning_payload = bool(
                chunk.content
                or chunk.refusal
                or chunk.tool_call
                or chunk.usage
                or chunk_is_terminal
            )
            # A reasoning item is not complete merely because its upstream id
            # is known: Copilot may deliver the encrypted blob on a later
            # output_item.done event. Keep any following canonical payload in
            # order until the item is closed by that blob or by terminal/EOF.
            # Include reasoning carried by this same chunk because deferral is
            # decided before the chunk's reasoning fields are applied below.
            reasoning_pending = state.reasoning_started or chunk_has_reasoning
            defer_non_reasoning = reasoning_pending and has_non_reasoning_payload
            if defer_non_reasoning and has_non_reasoning_payload:
                deferred_chunks.append(
                    replace(
                        chunk,
                        thinking=None,
                        thinking_id=None,
                        thinking_signature=None,
                    )
                )
                chunk = replace(
                    chunk,
                    content="",
                    refusal=None,
                    tool_call=None,
                    usage=None,
                    finish_reason=None,
                    terminal_outcome=None,
                )

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
                state.append_reasoning(chunk.thinking, chunk.reasoning_summary_index)

            # The upstream id may also arrive separately (e.g. on
            # output_item.done before any text deltas arrive). Capture it
            # so close_open_reasoning emits the correct (id, blob) pair.
            if chunk.thinking_id:
                if state.message_started and not state.reasoning_started:
                    for evt in state.close_open_message():
                        yield evt
                state.bind_reasoning_id(chunk.thinking_id)

            for evt in state.start_reasoning_events():
                yield evt
            for evt in state.flush_reasoning_events():
                yield evt

            if chunk.thinking_signature:
                if state.current_reasoning_id is None:
                    raise ProviderError(
                        "Reasoning signature is missing its upstream item id",
                        status_code=502,
                        retryable=True,
                        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                    )
                state.upstream_encrypted_content = chunk.thinking_signature

            indexed_reasoning_done = (
                chunk.output_item_done and chunk.output_item_type == "reasoning"
            )
            if chunk.thinking_signature or indexed_reasoning_done:
                for evt in state.close_open_reasoning():
                    yield evt
                if indexed_reasoning_done:
                    state.finish_empty_reasoning_item()

            if chunk_is_terminal and deferred_chunks and state.reasoning_started:
                for evt in state.close_open_reasoning():
                    yield evt

            # Handle text content
            if chunk.content:
                # Close any open reasoning before switching to a message item
                # so the output indices stay monotonic and Codex can replay
                # the items in order.
                for evt in state.close_open_reasoning():
                    yield evt
                for evt in state.bind_source_content_index(chunk.content_index):
                    yield evt
                for evt in state.start_message_part("output_text"):
                    yield evt

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

            if chunk.refusal:
                for evt in state.close_open_reasoning():
                    yield evt
                for evt in state.bind_source_content_index(chunk.content_index):
                    yield evt
                for evt in state.start_message_part("refusal"):
                    yield evt
                state.accumulated_refusal += chunk.refusal
                yield sse_event(
                    {
                        "type": "response.refusal.delta",
                        "item_id": state.current_message_id,
                        "output_index": state.output_index,
                        "content_index": state.content_index,
                        "delta": chunk.refusal,
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
                    tsc_item = {
                        "type": "tool_search_call",
                        "call_id": tc.call_id,
                        "execution": "client",
                        "status": "completed",
                        "arguments": parse_tool_search_arguments(tc.arguments),
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

            if chunk.output_item_done and chunk.output_item_type == "message":
                for evt in state.close_open_message():
                    yield evt
                state.finish_empty_message_item()

            if chunk.usage:
                final_usage = chunk.usage

            chunk_outcome = resolve_terminal_outcome(
                chunk.terminal_outcome,
                chunk.finish_reason,
            )
            if chunk_outcome is not None:
                terminal_outcome = chunk_outcome
                for evt in state.close_open_reasoning():
                    yield evt
                for evt in state.close_open_message(advance_index=False):
                    yield evt

                yield _terminal_response_event(
                    chunk_outcome,
                    base_response=base_response,
                    output=state.output_items,
                    usage=make_usage(final_usage),
                )
                pipeline.finish(
                    wire_status=200,
                    outcome=chunk_outcome,
                    body_summary=(
                        chunk_outcome.error.message if chunk_outcome.error is not None else None
                    ),
                )
                return

            if deferred_chunks and not state.reasoning_started:
                chunk_queue.extend((pending, False) for pending in deferred_chunks)
                deferred_chunks = []

        terminal_outcome = unexpected_eof_outcome()
        logger.warning(
            "Stream ended without explicit terminal: req_id=%s model=%s output_items=%d "
            "accumulated_text_len=%d message_started=%s",
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

        yield _terminal_response_event(
            terminal_outcome,
            base_response=base_response,
            output=state.output_items,
            usage=make_usage(final_usage),
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "Stream terminated: req_id=%s model=%s elapsed=%.1fms output_items=%d "
            "transport=%s response_status=%s usage=%s",
            request_id,
            request.model,
            elapsed_ms,
            len(state.output_items),
            terminal_outcome.transport.value,
            terminal_outcome.response_status.value,
            final_usage,
        )
        pipeline.finish(
            wire_status=200,
            outcome=terminal_outcome,
            body_summary=terminal_outcome.error.message,
        )

    except ProviderError as e:
        outcome_code = (
            "upstream_protocol_error"
            if e.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
            else "provider_error"
        )
        wire_error_code = (
            "upstream_protocol_error"
            if e.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
            else "server_error"
        )
        terminal_outcome = exception_outcome(str(e), code=outcome_code)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Stream failed: req_id=%s, elapsed=%.1fms, error=%s",
            request_id,
            elapsed_ms,
            e,
        )
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary=str(e),
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
                    "model": response_model,
                    "output": [],
                    "error": {
                        "code": wire_error_code,
                        "message": str(e),
                    },
                    "incomplete_details": None,
                },
            }
        )
    except asyncio.CancelledError:
        if pipeline is not None:
            pipeline.finish(wire_status=200, outcome=client_cancelled_outcome())
        logger.info("Responses stream cancelled by client: req_id=%s", request_id)
        raise
    except Exception:
        terminal_outcome = exception_outcome("Internal server error", code="server_error")
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Unexpected error in responses stream: req_id=%s, elapsed=%.1fms",
            request_id,
            elapsed_ms,
            exc_info=True,
        )
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary="Internal server error",
            )
        # Reuse the stream's response_id/created_at (hoisted above the try) so
        # clients correlating events by id see a consistent response object.
        yield sse_event(
            {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "failed",
                    "created_at": created_at,
                    "model": response_model,
                    "output": [],
                    "error": {
                        "code": "server_error",
                        "message": "Internal server error",
                    },
                    "incomplete_details": None,
                },
            }
        )
    finally:
        await close_async_iterator(stream)


def _terminal_response_event(
    outcome: TerminalOutcome,
    *,
    base_response: dict[str, Any],
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
) -> str:
    """Encode one canonical terminal outcome as a Responses SSE event."""
    if outcome.transport is TransportTermination.UNEXPECTED_EOF:
        event_type = "response.incomplete"
        status = "incomplete"
        incomplete_details = {"reason": "unexpected_eof"}
        error = None
    elif outcome.response_status is ResponseStatus.COMPLETED:
        event_type = "response.completed"
        status = "completed"
        incomplete_details = None
        error = None
    elif outcome.response_status is ResponseStatus.INCOMPLETE:
        event_type = "response.incomplete"
        status = "incomplete"
        incomplete_details = outcome.incomplete_details
        error = None
    elif outcome.response_status is ResponseStatus.CANCELLED:
        event_type = "response.failed"
        status = "cancelled"
        incomplete_details = None
        error = (
            {"code": outcome.error.code, "message": outcome.error.message}
            if outcome.error is not None
            else None
        )
    else:
        event_type = "response.failed"
        status = "failed"
        incomplete_details = outcome.incomplete_details
        error = (
            {"code": outcome.error.code, "message": outcome.error.message}
            if outcome.error is not None
            else None
        )

    return sse_event(
        {
            "type": event_type,
            "response": {
                **base_response,
                "status": status,
                "output": output,
                "usage": usage,
                "error": error,
                "incomplete_details": incomplete_details,
            },
        }
    )
