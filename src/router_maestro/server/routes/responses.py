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
                items.append(
                    {
                        "type": "function_call",
                        "id": item.get("id"),
                        "call_id": item.get("call_id"),
                        "name": item.get("name"),
                        "arguments": item.get("arguments", "{}"),
                        "status": item.get("status", "completed"),
                    }
                )

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

    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
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
    fc_id: str, call_id: str, name: str, arguments: str, status: str = "completed"
) -> dict[str, Any]:
    """Create function_call output item."""
    return {
        "type": "function_call",
        "id": fc_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


@router.post("/api/openai/v1/responses")
async def create_response(request: ResponsesRequest):
    """Handle Responses API requests (for Codex models)."""
    request_id = generate_id("req")
    start_time = time.time()

    logger.info(
        "Received responses request: req_id=%s, model=%s, stream=%s, has_tools=%s",
        request_id,
        request.model,
        request.stream,
        request.tools is not None,
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

        if response.content:
            message_id = generate_id("msg")
            output.append(make_message_item(message_id, response.content))

        if response.tool_calls:
            for tc in response.tool_calls:
                fc_id = generate_id("fc")
                output.append(make_function_call_item(fc_id, tc.call_id, tc.name, tc.arguments))

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
            # Handle text content
            if chunk.content:
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

            # Handle tool call delta
            if chunk.tool_call_delta:
                delta = chunk.tool_call_delta
                for evt in state.close_open_message():
                    yield evt

                yield sse_event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": delta.get("item_id", ""),
                        "output_index": delta.get("output_index", state.output_index),
                        "delta": delta.get("delta", ""),
                    }
                )

            # Handle complete tool call
            if chunk.tool_call:
                tc = chunk.tool_call
                for evt in state.close_open_message():
                    yield evt

                fc_id = generate_id("fc")
                fc_item = make_function_call_item(fc_id, tc.call_id, tc.name, tc.arguments)

                yield sse_event(
                    {
                        "type": "response.output_item.added",
                        "output_index": state.output_index,
                        "item": make_function_call_item(
                            fc_id, tc.call_id, tc.name, "", "in_progress"
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
            logger.warning("Stream ended without finish_reason, sending completion events")

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
        logger.info(
            "Stream completed: req_id=%s, elapsed=%.1fms, output_items=%d",
            request_id,
            elapsed_ms,
            len(state.output_items),
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
