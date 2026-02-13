"""Anthropic Messages API compatible route."""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from router_maestro.providers import AnthropicProvider, ChatRequest, ProviderError
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
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.server.translation import (
    translate_anthropic_to_openai,
    translate_openai_chunk_to_anthropic_events,
)
from router_maestro.utils import (
    count_anthropic_request_tokens,
    estimate_anthropic_request_tokens,
    get_logger,
    map_openai_stop_reason_to_anthropic,
)
from router_maestro.utils.token_config import (
    count_tokens_via_anthropic_api,
    get_config_for_provider,
)
from router_maestro.utils.tokens import AnthropicStopReason

logger = get_logger("server.routes.anthropic")

router = APIRouter()


TEST_RESPONSE_TEXT = "This is a test response from Router-Maestro."


async def _resolve_provider_name(model: str) -> str | None:
    """Resolve which provider will handle a model.

    Returns the provider name (e.g. ``"github-copilot"``, ``"anthropic"``)
    or ``None`` if the model cannot be resolved.
    """
    model_router = get_router()
    try:
        provider_name, _actual_model, _provider = await model_router._resolve_provider(model)
        return provider_name
    except ProviderError:
        return None


def _create_test_response() -> AnthropicMessagesResponse:
    """Create a mock response for test model."""
    return AnthropicMessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        content=[AnthropicTextBlock(type="text", text=TEST_RESPONSE_TEXT)],
        model="test",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=10, output_tokens=10),
    )


async def _stream_test_response() -> AsyncGenerator[str, None]:
    """Stream a mock test response."""
    response_id = f"msg_{uuid.uuid4().hex[:24]}"

    # message_start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "test",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # content_block_start event
    block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"

    # content_block_delta event
    block_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": TEST_RESPONSE_TEXT},
    }
    yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"

    # content_block_stop event
    block_stop = {"type": "content_block_stop", "index": 0}
    yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"

    # message_delta event
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 10},
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    # message_stop event
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


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
            return sse_streaming_response(_stream_test_response())
        return _create_test_response()

    model_router = get_router()

    # Translate Anthropic request to OpenAI format
    chat_request = translate_anthropic_to_openai(request)

    if request.stream:
        # Resolve provider for accurate token estimation
        provider_name = await _resolve_provider_name(request.model)
        estimated_tokens = _estimate_input_tokens(request, provider_name)
        return sse_streaming_response(
            stream_response(model_router, chat_request, request.model, estimated_tokens),
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

    Resolves which provider will handle the model and selects the appropriate
    token-counting configuration. For native Anthropic, attempts an upstream
    API call for an exact count before falling back to local estimation.
    """
    provider_name = await _resolve_provider_name(request.model)
    config = get_config_for_provider(provider_name)

    # For native Anthropic, try the upstream count_tokens API first
    if provider_name == "anthropic":
        model_router = get_router()
        provider = model_router.providers.get("anthropic")
        if isinstance(provider, AnthropicProvider) and provider.is_authenticated():
            try:
                # Serialise messages/tools to plain dicts for the API call
                msgs = [
                    m if isinstance(m, dict) else m.model_dump(exclude_none=True)
                    for m in request.messages
                ]
                system = request.system
                if system is not None and not isinstance(system, str):
                    system = [
                        b if isinstance(b, dict) else b.model_dump(exclude_none=True)
                        for b in system
                    ]
                tools_dicts = None
                if request.tools:
                    tools_dicts = [
                        t if isinstance(t, dict) else t.model_dump(exclude_none=True)
                        for t in request.tools
                    ]
                # Strip provider prefix from model name for upstream API
                model_name = request.model
                if "/" in model_name:
                    model_name = model_name.split("/", 1)[1]

                exact_tokens = await count_tokens_via_anthropic_api(
                    base_url=provider.base_url,
                    api_key=provider._get_api_key(),
                    model=model_name,
                    messages=msgs,
                    system=system,
                    tools=tools_dicts,
                )
                return {"input_tokens": exact_tokens}
            except Exception:
                logger.warning(
                    "Anthropic upstream count_tokens failed, falling back to local estimation",
                    exc_info=True,
                )

    input_tokens = count_anthropic_request_tokens(
        system=request.system,
        messages=request.messages,
        tools=request.tools,
        model=request.model,
        config=config,
    )
    return {"input_tokens": input_tokens}


def _map_finish_reason(reason: str | None) -> AnthropicStopReason | None:
    """Map OpenAI finish reason to Anthropic stop reason."""
    return map_openai_stop_reason_to_anthropic(reason)


def _estimate_input_tokens(
    request: AnthropicMessagesRequest,
    provider_name: str | None = None,
) -> int:
    """Estimate input tokens from request content.

    Uses the centralized token estimation function with provider-aware
    configuration for accurate estimates.
    """
    config = get_config_for_provider(provider_name)
    return estimate_anthropic_request_tokens(
        system=request.system,
        messages=request.messages,
        tools=request.tools,
        model=request.model,
        config=config,
    )


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
        yield _sse_error_event(str(e))
    except asyncio.CancelledError:
        logger.info("Anthropic stream cancelled by client")
        raise
    except Exception:
        logger.error("Unexpected error in Anthropic stream", exc_info=True)
        yield _sse_error_event("Internal server error")


def _sse_error_event(message: str) -> str:
    """Format an Anthropic SSE error event."""
    error_event = {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message,
        },
    }
    return f"event: error\ndata: {json.dumps(error_event)}\n\n"


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
