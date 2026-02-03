"""Responses API route for Codex models."""

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from router_maestro.providers import ProviderError
from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.routing import Router, get_router
from router_maestro.server.schemas import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
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
    logger.info(
        "Received responses request: model=%s, stream=%s, has_tools=%s",
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
        return StreamingResponse(
            stream_response(model_router, internal_request),
            media_type="text/event-stream",
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
        logger.error("Responses request failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=str(e))


async def stream_response(
    model_router: Router, request: InternalResponsesRequest
) -> AsyncGenerator[str, None]:
    """Stream Responses API response."""
    try:
        stream, provider_name = await model_router.responses_completion_stream(request)
        response_id = generate_id("resp")

        output_items: list[dict[str, Any]] = []
        output_index = 0
        content_index = 0

        current_message_id: str | None = None
        accumulated_content = ""
        message_started = False

        final_usage = None
        stream_completed = False

        # response.created
        yield sse_event(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
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
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "output": [],
                },
            }
        )

        async for chunk in stream:
            # Handle text content
            if chunk.content:
                if not message_started:
                    current_message_id = generate_id("msg")
                    message_started = True

                    yield sse_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": make_message_item(current_message_id, "", "in_progress"),
                        }
                    )

                    yield sse_event(
                        {
                            "type": "response.content_part.added",
                            "item_id": current_message_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "part": make_text_content(""),
                        }
                    )

                accumulated_content += chunk.content

                yield sse_event(
                    {
                        "type": "response.output_text.delta",
                        "item_id": current_message_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "delta": chunk.content,
                    }
                )

            # Handle tool call delta
            if chunk.tool_call_delta:
                delta = chunk.tool_call_delta
                if message_started and current_message_id:
                    # Close current message
                    for evt in _close_message_events(
                        current_message_id,
                        output_index,
                        content_index,
                        accumulated_content,
                    ):
                        yield evt
                    output_items.append(make_message_item(current_message_id, accumulated_content))
                    output_index += 1
                    message_started = False
                    current_message_id = None

                yield sse_event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": delta.get("item_id", ""),
                        "output_index": delta.get("output_index", output_index),
                        "delta": delta.get("delta", ""),
                    }
                )

            # Handle complete tool call
            if chunk.tool_call:
                tc = chunk.tool_call
                if message_started and current_message_id:
                    for evt in _close_message_events(
                        current_message_id,
                        output_index,
                        content_index,
                        accumulated_content,
                    ):
                        yield evt
                    output_items.append(make_message_item(current_message_id, accumulated_content))
                    output_index += 1
                    message_started = False
                    current_message_id = None

                fc_id = generate_id("fc")
                fc_item = make_function_call_item(fc_id, tc.call_id, tc.name, tc.arguments)

                yield sse_event(
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": make_function_call_item(
                            fc_id, tc.call_id, tc.name, "", "in_progress"
                        ),
                    }
                )

                yield sse_event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": fc_id,
                        "output_index": output_index,
                        "delta": tc.arguments,
                    }
                )

                yield sse_event(
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": fc_id,
                        "output_index": output_index,
                        "arguments": tc.arguments,
                    }
                )

                yield sse_event(
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": fc_item,
                    }
                )

                output_items.append(fc_item)
                output_index += 1

            if chunk.usage:
                final_usage = chunk.usage

            if chunk.finish_reason:
                stream_completed = True

                if message_started and current_message_id:
                    for evt in _close_message_events(
                        current_message_id,
                        output_index,
                        content_index,
                        accumulated_content,
                    ):
                        yield evt
                    output_items.append(make_message_item(current_message_id, accumulated_content))

                yield sse_event(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "status": "completed",
                            "model": request.model,
                            "output": output_items,
                            "usage": final_usage,
                        },
                    }
                )

        if not stream_completed:
            logger.warning("Stream ended without finish_reason, sending completion events")

            if message_started and current_message_id:
                for evt in _close_message_events(
                    current_message_id,
                    output_index,
                    content_index,
                    accumulated_content,
                ):
                    yield evt
                output_items.append(make_message_item(current_message_id, accumulated_content))

            yield sse_event(
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "model": request.model,
                        "output": output_items,
                        "usage": final_usage,
                    },
                }
            )

        yield "data: [DONE]\n\n"

    except ProviderError as e:
        error_data = {"error": {"message": str(e), "type": "provider_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"


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
