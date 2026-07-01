"""Anthropic Messages API beta route — native passthrough to Copilot.

For Claude models routed via GitHub Copilot, forwards requests directly to
Copilot's native ``/v1/messages`` endpoint without Anthropic→OpenAI→Anthropic
translation.  Non-Claude or non-Copilot models fall back to the standard
translation-based handler transparently.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse

from router_maestro.providers import ProviderError
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.routing import get_router
from router_maestro.server.routes.anthropic import (
    ANTHROPIC_PING_FRAME,
)
from router_maestro.server.routes.anthropic import (
    messages as standard_messages,
)
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.utils import get_logger
from router_maestro.utils.context_window import resolve_thinking_budget

logger = get_logger("server.routes.anthropic_beta")

router = APIRouter()

COPILOT_MESSAGES_PATH = "/v1/messages"
COPILOT_COUNT_TOKENS_PATH = "/v1/messages/count_tokens"

_STRIP_RESPONSE_KEYS = frozenset({"copilot_usage", "stop_details"})
_STRIP_STREAM_MESSAGE_STOP_KEYS = frozenset({"copilot_usage", "amazon-bedrock-invocationMetrics"})


def _is_native_eligible(provider_name: str, actual_model: str) -> bool:
    """Whether this model can use the native Copilot Anthropic endpoint."""
    if provider_name != "github-copilot":
        return False
    bare = actual_model.split("/", 1)[-1].lower()
    return bare.startswith("claude-")


def _strip_response(data: dict) -> dict:
    """Remove Copilot-internal fields from a non-streaming response."""
    for key in _STRIP_RESPONSE_KEYS:
        data.pop(key, None)
    msg = data.get("message")
    if isinstance(msg, dict):
        for key in _STRIP_RESPONSE_KEYS:
            msg.pop(key, None)
    return data


def _apply_thinking_budget_native(body: dict, actual_model: str) -> dict:
    """Apply server-side thinking budget config to the raw Anthropic body.

    Modifies ``body["thinking"]`` in-place if the server config specifies a
    budget and the client hasn't already set one.
    """
    from router_maestro.config import load_priorities_config

    priorities = load_priorities_config()
    thinking_config = priorities.thinking

    client_thinking = body.get("thinking")
    client_budget = None
    client_type = None
    if isinstance(client_thinking, dict):
        client_budget = client_thinking.get("budget_tokens")
        client_type = client_thinking.get("type")

    model_router = get_router()
    model_info = None
    if hasattr(model_router, "_models_cache"):
        cache_entry = model_router._models_cache.get(actual_model)
        if cache_entry:
            _, model_info = cache_entry

    supports_thinking = model_info.supports_thinking if model_info else True
    max_output = (model_info.max_output_tokens or 16384) if model_info else 16384

    budget, thinking_type = resolve_thinking_budget(
        client_budget=client_budget,
        client_thinking_type=client_type,
        model_id=actual_model,
        max_output_tokens=max_output,
        thinking_config=thinking_config,
        supports_thinking=supports_thinking,
    )

    if budget != client_budget or thinking_type != client_type:
        if budget is not None and thinking_type in ("enabled", "adaptive"):
            body["thinking"] = {"type": thinking_type, "budget_tokens": budget}
        elif thinking_type == "disabled" or budget is None:
            body.pop("thinking", None)

    return body


async def _resolve_model(model: str) -> tuple[str, str, CopilotProvider | None]:
    """Resolve model to (provider_name, actual_model_id, provider_if_copilot).

    Returns (provider_name, actual_model, None) for non-Copilot providers.
    Raises HTTPException(404) if the model can't be resolved at all.
    """
    model_router = get_router()
    try:
        provider_name, actual_model, provider = await model_router._resolve_provider(model)
    except ProviderError as e:
        raise HTTPException(status_code=e.status_code or 404, detail=str(e))
    if isinstance(provider, CopilotProvider):
        return provider_name, actual_model, provider
    return provider_name, actual_model, None


@router.post("/api/anthropic/beta/v1/messages")
async def beta_messages(raw_request: FastAPIRequest):
    """Handle Anthropic Messages API requests via native passthrough or fallback."""
    body_bytes = await raw_request.body()
    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="'model' field is required")

    stream = body.get("stream", False)

    # Resolve provider and check native eligibility
    provider_name, actual_model, copilot_provider = await _resolve_model(model)

    if not _is_native_eligible(provider_name, actual_model) or copilot_provider is None:
        logger.info(
            "Beta route falling back to standard path: model=%s, provider=%s",
            model,
            provider_name,
        )
        # Delegate to the standard translation-based handler
        return await standard_messages(
            request=await _parse_as_anthropic_request(body_bytes),
            raw_request=raw_request,
        )

    logger.info(
        "Beta route using native passthrough: model=%s -> %s, stream=%s",
        model,
        actual_model,
        stream,
    )

    # Apply server-side thinking budget
    body = _apply_thinking_budget_native(body, actual_model)

    # Replace model with the resolved catalog name
    body["model"] = actual_model

    # Copilot's native endpoint rejects temperature + top_p together;
    # drop top_p when both are present (temperature takes priority).
    if "temperature" in body and "top_p" in body:
        del body["top_p"]

    # Ensure Copilot token is fresh
    await copilot_provider.ensure_token()

    if stream:
        return sse_streaming_response(
            _stream_passthrough(copilot_provider, body),
            keepalive_frame=ANTHROPIC_PING_FRAME,
        )

    # Non-streaming passthrough
    try:
        response = await copilot_provider._send_with_auth_retry(
            "POST",
            COPILOT_MESSAGES_PATH,
            json=body,
        )
    except ProviderError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))

    if response.status_code >= 400:
        # Forward upstream error verbatim (already Anthropic format)
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": {"type": "api_error", "message": response.text}}
        return JSONResponse(content=error_body, status_code=response.status_code)

    data = response.json()
    _strip_response(data)
    return JSONResponse(content=data)


@router.post("/api/anthropic/beta/v1/messages/count_tokens")
async def beta_count_tokens(raw_request: FastAPIRequest):
    """Count tokens via native passthrough or fallback."""
    body_bytes = await raw_request.body()
    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="'model' field is required")

    provider_name, actual_model, copilot_provider = await _resolve_model(model)

    if not _is_native_eligible(provider_name, actual_model) or copilot_provider is None:
        # Fallback to standard count_tokens handler
        from router_maestro.server.routes.anthropic import count_tokens
        from router_maestro.server.schemas.anthropic import AnthropicCountTokensRequest

        request = AnthropicCountTokensRequest.model_validate(body)
        return await count_tokens(request)

    body["model"] = actual_model
    await copilot_provider.ensure_token()

    try:
        response = await copilot_provider._send_with_auth_retry(
            "POST",
            COPILOT_COUNT_TOKENS_PATH,
            json=body,
        )
    except ProviderError as e:
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))

    if response.status_code >= 400:
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": {"type": "api_error", "message": response.text}}
        return JSONResponse(content=error_body, status_code=response.status_code)

    return JSONResponse(content=response.json())


async def _stream_passthrough(
    provider: CopilotProvider,
    payload: dict,
) -> AsyncGenerator[str, None]:
    """Stream SSE events from Copilot's native Anthropic endpoint.

    Filters out copilot-internal events and strips internal metadata from
    ``message_stop`` events before yielding to the client.
    """
    try:
        async with provider._stream_with_auth_retry(
            COPILOT_MESSAGES_PATH,
            json=payload,
            headers_kwargs={},
        ) as response:
            if response.status_code >= 400:
                error_text = (await response.aread()).decode()
                msg = f"Upstream error ({response.status_code}): {error_text[:200]}"
                yield _sse_error_event(msg)
                return

            current_event: str | None = None
            data_buffer: str = ""

            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]
                    # Don't yield event line yet — wait for data to decide
                elif line.startswith("data: "):
                    data_buffer = line[6:]
                elif line == "":
                    # End of SSE frame — process and yield
                    if current_event and data_buffer:
                        cleaned = _clean_stream_frame(current_event, data_buffer)
                        if cleaned is not None:
                            yield f"event: {current_event}\ndata: {cleaned}\n\n"
                    elif current_event and not data_buffer:
                        yield f"event: {current_event}\ndata: \n\n"
                    current_event = None
                    data_buffer = ""

    except ProviderError as e:
        yield _sse_error_event(str(e))
    except asyncio.CancelledError:
        logger.info("Beta anthropic stream cancelled by client")
        raise
    except Exception:
        logger.error("Unexpected error in beta anthropic stream", exc_info=True)
        yield _sse_error_event("Internal server error")


def _clean_stream_frame(event_type: str, data_str: str) -> str | None:
    """Clean a single SSE frame's data payload.

    Returns the cleaned data string, or None to suppress the event entirely.
    """
    # Filter out copilot-internal events
    if event_type == "copilot_usage":
        return None

    # For message_start, strip copilot fields from nested message
    if event_type == "message_start":
        try:
            data = json.loads(data_str)
            msg = data.get("message")
            if isinstance(msg, dict):
                for key in _STRIP_RESPONSE_KEYS:
                    msg.pop(key, None)
            return json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            return data_str

    # For message_stop, strip bedrock metrics
    if event_type == "message_stop":
        try:
            data = json.loads(data_str)
            for key in _STRIP_STREAM_MESSAGE_STOP_KEYS:
                data.pop(key, None)
            return json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            return data_str

    return data_str


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


async def _parse_as_anthropic_request(body_bytes: bytes):
    """Parse raw body bytes into an AnthropicMessagesRequest for fallback."""
    from router_maestro.server.schemas.anthropic import AnthropicMessagesRequest

    body = json.loads(body_bytes)
    return AnthropicMessagesRequest.model_validate(body)
