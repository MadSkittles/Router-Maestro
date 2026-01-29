"""Anthropic Messages API compatible route."""

import json
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from router_maestro.providers import ChatRequest, ProviderError
from router_maestro.routing import Router, get_router
from router_maestro.server.schemas.anthropic import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamState,
    AnthropicTextBlock,
    AnthropicUsage,
)
from router_maestro.server.translation import (
    translate_anthropic_to_openai,
    translate_openai_chunk_to_anthropic_events,
)
from router_maestro.utils import (
    estimate_tokens_from_char_count,
    get_logger,
    map_openai_stop_reason_to_anthropic,
)
from router_maestro.utils.tokens import AnthropicStopReason

logger = get_logger("server.routes.anthropic")

router = APIRouter()


@router.post("/v1/messages")
@router.post("/api/anthropic/v1/messages")
async def messages(request: AnthropicMessagesRequest):
    """Handle Anthropic Messages API requests."""
    logger.info(
        "Received Anthropic messages request: model=%s, stream=%s",
        request.model,
        request.stream,
    )
    model_router = get_router()

    # Translate Anthropic request to OpenAI format
    chat_request = translate_anthropic_to_openai(request)

    if request.stream:
        # Estimate input tokens for context display
        estimated_tokens = _estimate_input_tokens(request)
        return StreamingResponse(
            stream_response(model_router, chat_request, request.model, estimated_tokens),
            media_type="text/event-stream",
        )

    try:
        response, provider_name = await model_router.chat_completion(chat_request)

        # Build Anthropic response
        content = []
        if response.content:
            content.append(AnthropicTextBlock(type="text", text=response.content))

        usage = AnthropicUsage(
            input_tokens=response.usage.get("prompt_tokens", 0) if response.usage else 0,
            output_tokens=response.usage.get("completion_tokens", 0) if response.usage else 0,
        )

        # Map finish reason
        stop_reason = _map_finish_reason(response.finish_reason)

        return AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            type="message",
            role="assistant",
            content=content,
            model=response.model,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=usage,
        )
    except ProviderError as e:
        logger.error("Anthropic messages request failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/v1/messages/count_tokens")
@router.post("/api/anthropic/v1/messages/count_tokens")
async def count_tokens(request: AnthropicCountTokensRequest):
    """Count tokens for a messages request.

    This is a simplified implementation that estimates tokens.
    Since we're proxying to various providers, we can't get exact counts
    without making an actual request.
    """
    total_chars = 0

    # Count system prompt
    if request.system:
        if isinstance(request.system, str):
            total_chars += len(request.system)
        else:
            for block in request.system:
                total_chars += len(block.text)

    # Count messages
    for msg in request.messages:
        content = msg.content
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                elif hasattr(block, "text"):
                    total_chars += len(block.text)  # type: ignore[union-attr]

    return {"input_tokens": estimate_tokens_from_char_count(total_chars)}


def _map_finish_reason(reason: str | None) -> AnthropicStopReason | None:
    """Map OpenAI finish reason to Anthropic stop reason."""
    return map_openai_stop_reason_to_anthropic(reason)


def _estimate_input_tokens(request: AnthropicMessagesRequest) -> int:
    """Estimate input tokens from request content.

    Uses a rough approximation of ~4 characters per token for English text.
    This provides an estimate for context display before actual usage is known.
    """
    total_chars = 0

    # Count system prompt
    if request.system:
        if isinstance(request.system, str):
            total_chars += len(request.system)
        else:
            for block in request.system:
                if hasattr(block, "text"):
                    total_chars += len(block.text)

    # Count messages
    for msg in request.messages:
        content = msg.content
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str):
                            total_chars += len(tool_content)
                        elif isinstance(tool_content, list):
                            for tc in tool_content:
                                if isinstance(tc, dict) and tc.get("type") == "text":
                                    total_chars += len(tc.get("text", ""))
                elif hasattr(block, "text"):
                    total_chars += len(block.text)  # type: ignore[union-attr]

    # Count tools definitions if present
    if request.tools:
        for tool in request.tools:
            if hasattr(tool, "name"):
                total_chars += len(tool.name)
            if hasattr(tool, "description") and tool.description:
                total_chars += len(tool.description)
            if hasattr(tool, "input_schema"):
                # Rough estimate for schema
                import json

                try:
                    schema_str = json.dumps(tool.input_schema)
                    total_chars += len(schema_str)
                except Exception:
                    pass

    return estimate_tokens_from_char_count(total_chars)


async def stream_response(
    model_router: Router,
    request: ChatRequest,
    original_model: str,
    estimated_input_tokens: int = 0,
) -> AsyncGenerator[str, None]:
    """Stream Anthropic Messages API response."""
    try:
        stream, provider_name = await model_router.chat_completion_stream(request)
        response_id = f"msg_{uuid.uuid4().hex[:24]}"

        state = AnthropicStreamState(estimated_input_tokens=estimated_input_tokens)

        async for chunk in stream:
            # Build OpenAI-style chunk for translation
            openai_chunk = {
                "id": response_id,
                "choices": [
                    {
                        "delta": {
                            "content": chunk.content if chunk.content else None,
                            "tool_calls": chunk.tool_calls,
                        },
                        "finish_reason": chunk.finish_reason,
                    }
                ],
                "usage": chunk.usage,  # Pass through usage info
            }

            events = translate_openai_chunk_to_anthropic_events(openai_chunk, state, original_model)

            for event in events:
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    except ProviderError as e:
        error_event = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": str(e),
            },
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
