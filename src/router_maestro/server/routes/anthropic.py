"""Anthropic Messages API compatible route."""

import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from router_maestro.providers import ChatRequest, ProviderError
from router_maestro.routing import Router, get_router
from router_maestro.server.schemas.anthropic import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicModelInfo,
    AnthropicModelList,
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


def _create_test_response() -> AnthropicMessagesResponse:
    """Create a mock response for test model."""
    return AnthropicMessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        content=[AnthropicTextBlock(type="text", text="This is a test response from Router-Maestro.")],
        model="test",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=10, output_tokens=10),
    )


async def _stream_test_response() -> AsyncGenerator[str, None]:
    """Stream a mock test response."""
    response_id = f"msg_{uuid.uuid4().hex[:24]}"

    # message_start event
    yield f'event: message_start\ndata: {json.dumps({"type": "message_start", "message": {"id": response_id, "type": "message", "role": "assistant", "content": [], "model": "test", "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 0}}})}\n\n'

    # content_block_start event
    yield f'event: content_block_start\ndata: {json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})}\n\n'

    # content_block_delta event
    yield f'event: content_block_delta\ndata: {json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "This is a test response from Router-Maestro."}})}\n\n'

    # content_block_stop event
    yield f'event: content_block_stop\ndata: {json.dumps({"type": "content_block_stop", "index": 0})}\n\n'

    # message_delta event
    yield f'event: message_delta\ndata: {json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 10}})}\n\n'

    # message_stop event
    yield f'event: message_stop\ndata: {json.dumps({"type": "message_stop"})}\n\n'


@router.post("/v1/messages")
@router.post("/api/anthropic/v1/messages")
async def messages(request: AnthropicMessagesRequest):
    """Handle Anthropic Messages API requests."""
    logger.info(
        "Received Anthropic messages request: model=%s, stream=%s",
        request.model,
        request.stream,
    )

    # Handle test model
    if request.model == "test":
        if request.stream:
            return StreamingResponse(
                _stream_test_response(),
                media_type="text/event-stream",
            )
        return _create_test_response()

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


def _generate_display_name(model_id: str) -> str:
    """Generate a human-readable display name from model ID.

    Transforms model IDs like 'github-copilot/claude-sonnet-4' into
    'Claude Sonnet 4 (github-copilot)'.
    """
    if "/" in model_id:
        provider, model_name = model_id.split("/", 1)
    else:
        provider = ""
        model_name = model_id

    # Capitalize words and handle common patterns
    words = model_name.replace("-", " ").replace("_", " ").split()
    display_words = []
    for word in words:
        # Keep version numbers as-is
        if word.replace(".", "").isdigit():
            display_words.append(word)
        else:
            display_words.append(word.capitalize())

    display_name = " ".join(display_words)
    if provider:
        display_name = f"{display_name} ({provider})"

    return display_name


@router.get("/api/anthropic/v1/models")
async def list_models(
    limit: int = 20,
    after_id: str | None = None,
    before_id: str | None = None,
) -> AnthropicModelList:
    """List available models in Anthropic format.

    Args:
        limit: Maximum number of models to return (default 20)
        after_id: Return models after this ID (for forward pagination)
        before_id: Return models before this ID (for backward pagination)
    """
    model_router = get_router()
    models = await model_router.list_models()

    # Generate ISO 8601 timestamp for created_at
    # Using current time since actual creation dates aren't tracked
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Convert to Anthropic format
    anthropic_models = [
        AnthropicModelInfo(
            id=model.id,
            created_at=created_at,
            display_name=_generate_display_name(model.id),
            type="model",
        )
        for model in models
    ]

    # Handle pagination
    start_idx = 0
    if after_id:
        for i, model in enumerate(anthropic_models):
            if model.id == after_id:
                start_idx = i + 1
                break

    end_idx = len(anthropic_models)
    if before_id:
        for i, model in enumerate(anthropic_models):
            if model.id == before_id:
                end_idx = i
                break

    # Apply limit
    paginated = anthropic_models[start_idx : min(start_idx + limit, end_idx)]

    first_id = paginated[0].id if paginated else None
    last_id = paginated[-1].id if paginated else None
    has_more = (start_idx + limit) < end_idx

    return AnthropicModelList(
        data=paginated,
        first_id=first_id,
        last_id=last_id,
        has_more=has_more,
    )
