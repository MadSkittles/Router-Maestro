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
from router_maestro.utils.reasoning import VALID_EFFORTS, pick_closest_effort

logger = get_logger("server.routes.anthropic_beta")

router = APIRouter()

COPILOT_MESSAGES_PATH = "/v1/messages"
COPILOT_COUNT_TOKENS_PATH = "/v1/messages/count_tokens"

_STRIP_RESPONSE_KEYS = frozenset({"copilot_usage", "stop_details"})
_STRIP_STREAM_MESSAGE_STOP_KEYS = frozenset({"copilot_usage", "amazon-bedrock-invocationMetrics"})
_THINKING_TYPES = frozenset({"enabled", "adaptive", "disabled"})

# Fields the Copilot native Anthropic endpoint accepts. Anything not in this
# set is stripped before forwarding — Copilot returns 400 on unknown fields.
_COPILOT_ACCEPTED_FIELDS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "stream",
        "system",
        "thinking",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "metadata",
        "output_config",
    }
)


def _is_native_eligible(provider_name: str, actual_model: str) -> bool:
    """Whether this model can use the native Copilot Anthropic endpoint."""
    if provider_name != "github-copilot":
        return False
    bare = actual_model.split("/", 1)[-1].lower()
    return bare.startswith("claude-")


def _sanitize_output_config(body: dict) -> str | None:
    """Keep only a valid effort supported by Copilot's native endpoint."""
    output_config = body.get("output_config")
    if not isinstance(output_config, dict):
        body.pop("output_config", None)
        return None

    effort = output_config.get("effort")
    if effort not in VALID_EFFORTS:
        body.pop("output_config", None)
        return None

    body["output_config"] = {"effort": effort}
    return effort


def _invalid_request(message: str) -> JSONResponse:
    """Return an Anthropic-native invalid request response."""
    return JSONResponse(
        status_code=400,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": message,
            },
        },
    )


def _validate_native_thinking(body: dict) -> JSONResponse | None:
    """Validate raw token limits before native thinking-budget resolution."""
    max_tokens = body.get("max_tokens")
    if "max_tokens" in body and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        return _invalid_request("max_tokens must be a positive integer")

    if "thinking" not in body:
        return None

    thinking = body["thinking"]
    if not isinstance(thinking, dict):
        return _invalid_request("thinking must be an object")

    thinking_type = thinking.get("type")
    if not isinstance(thinking_type, str) or thinking_type not in _THINKING_TYPES:
        return _invalid_request("thinking.type must be enabled, adaptive, or disabled")

    if "budget_tokens" not in thinking:
        return None

    budget = thinking["budget_tokens"]
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        return _invalid_request("thinking.budget_tokens must be a positive integer")

    if max_tokens is not None and budget >= max_tokens:
        return _invalid_request("thinking.budget_tokens must be less than max_tokens")

    return None


def _strip_history_thinking_blocks(body: dict) -> None:
    """Remove thinking blocks from assistant messages in conversation history.

    Called as a retry fallback when Copilot rejects a signature it didn't
    produce.  Stripping is safe — the model doesn't need prior thinking
    to generate a new response.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        filtered = [b for b in content if not (isinstance(b, dict) and b.get("type") == "thinking")]
        if len(filtered) != len(content):
            msg["content"] = filtered


def _is_signature_error(response_text: str) -> bool:
    """Check if an upstream 400 is a thinking signature validation error."""
    lower = response_text.lower()
    return "signature" in lower and "thinking" in lower


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
    """Apply effort precedence or the server budget fallback to a raw body.

    Explicit ``output_config.effort`` removes an adaptive thinking budget.
    Manual enabled thinking retains a normalized budget, using the server fallback
    when the client omits it. Without effort, budget resolution is unchanged.
    """
    effort = _sanitize_output_config(body)
    client_thinking = body.get("thinking")

    model_router = None
    model_info = None
    if effort is not None:
        model_router = get_router()
        if hasattr(model_router, "_models_cache"):
            cache_entry = model_router._models_cache.get(actual_model)
            if cache_entry:
                _, model_info = cache_entry
        if model_info is not None and model_info.reasoning_effort_values is not None:
            mapped_effort = pick_closest_effort(effort, model_info.reasoning_effort_values)
            if mapped_effort is None:
                body.pop("output_config", None)
                effort = None
            else:
                body["output_config"] = {"effort": mapped_effort}
                effort = mapped_effort

    if isinstance(client_thinking, dict) and client_thinking.get("type") == "adaptive":
        thinking = dict(client_thinking)
        thinking.pop("budget_tokens", None)
        body["thinking"] = thinking
        return body

    if effort is not None:
        if not (
            isinstance(client_thinking, dict)
            and client_thinking.get("type") in ("enabled", "disabled")
        ):
            return body

    from router_maestro.config import load_priorities_config

    priorities = load_priorities_config()
    thinking_config = priorities.thinking

    client_budget = None
    client_type = None
    if isinstance(client_thinking, dict):
        client_budget = client_thinking.get("budget_tokens")
        client_type = client_thinking.get("type")

    if model_router is None:
        model_router = get_router()
    if model_info is None and hasattr(model_router, "_models_cache"):
        cache_entry = model_router._models_cache.get(actual_model)
        if cache_entry:
            _, model_info = cache_entry

    supports_thinking = model_info.supports_thinking if model_info else True
    max_output = (model_info.max_output_tokens or 16384) if model_info else 16384
    request_max_output = body.get("max_tokens")
    if isinstance(request_max_output, int):
        max_output = min(max_output, request_max_output)

    budget, thinking_type = resolve_thinking_budget(
        client_budget=client_budget,
        client_thinking_type=client_type,
        model_id=actual_model,
        max_output_tokens=max_output,
        thinking_config=thinking_config,
        supports_thinking=supports_thinking,
    )

    if thinking_type == "adaptive":
        adaptive = dict(client_thinking) if isinstance(client_thinking, dict) else {}
        adaptive["type"] = "adaptive"
        adaptive.pop("budget_tokens", None)
        body["thinking"] = adaptive
        if budget != client_budget or thinking_type != client_type:
            logger.debug(
                "Thinking budget adjusted: %s/%s -> %s/%s for model=%s",
                client_type,
                client_budget,
                thinking_type,
                budget,
                actual_model,
            )
    elif thinking_type == "enabled" and budget is not None:
        enabled = dict(client_thinking) if isinstance(client_thinking, dict) else {}
        enabled["type"] = "enabled"
        enabled["budget_tokens"] = budget
        body["thinking"] = enabled
        if budget != client_budget or thinking_type != client_type:
            logger.debug(
                "Thinking budget adjusted: %s/%s -> %s/%s for model=%s",
                client_type,
                client_budget,
                thinking_type,
                budget,
                actual_model,
            )
    else:
        body.pop("thinking", None)
        if client_type is not None:
            logger.debug(
                "Thinking removed: client had %s/%s, resolved to %s for model=%s",
                client_type,
                client_budget,
                thinking_type,
                actual_model,
            )

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
    thinking = body.get("thinking")

    # Debug: log all top-level keys to identify fields Copilot might reject
    logger.debug("Beta request body keys: %s", sorted(body.keys()))
    logger.info(
        "Received beta Anthropic request: model=%s, stream=%s, max_tokens=%s, "
        "thinking=%s, output_config=%s",
        model,
        stream,
        body.get("max_tokens"),
        thinking,
        body.get("output_config"),
    )

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

    validation_error = _validate_native_thinking(body)
    if validation_error is not None:
        return validation_error

    # Sanitize before the budget helper so unsupported output_config siblings
    # cannot survive even if the helper is replaced by an integration hook.
    _sanitize_output_config(body)

    # Apply budget fallback only when effort is absent.
    body = _apply_thinking_budget_native(body, actual_model)

    # Replace model with the resolved catalog name
    body["model"] = actual_model

    # Copilot's native endpoint rejects temperature + top_p together;
    # drop top_p when both are present (temperature takes priority).
    if "temperature" in body and "top_p" in body:
        logger.debug("Stripping top_p (temperature also present)")
        del body["top_p"]

    # Strip fields the Copilot native endpoint doesn't accept.
    unknown = set(body.keys()) - _COPILOT_ACCEPTED_FIELDS
    if unknown:
        logger.debug("Stripping unknown fields before passthrough: %s", unknown)
        for key in unknown:
            del body[key]

    # Ensure Copilot token is fresh
    await copilot_provider.ensure_token()

    if stream:
        return sse_streaming_response(
            _stream_passthrough(copilot_provider, body),
            keepalive_frame=ANTHROPIC_PING_FRAME,
        )

    # Non-streaming passthrough — try-forward with signature retry
    try:
        response = await copilot_provider._send_with_auth_retry(
            "POST",
            COPILOT_MESSAGES_PATH,
            json=body,
        )
    except ProviderError as e:
        logger.error("Beta passthrough request failed: %s", e)
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))

    # If Copilot rejects a thinking signature, strip history thinking blocks
    # and retry once — this handles the standard→beta route transition case.
    if response.status_code == 400 and _is_signature_error(response.text):
        logger.info("Signature rejected by upstream, stripping thinking blocks and retrying")
        _strip_history_thinking_blocks(body)
        try:
            response = await copilot_provider._send_with_auth_retry(
                "POST",
                COPILOT_MESSAGES_PATH,
                json=body,
            )
        except ProviderError as e:
            logger.error("Beta passthrough retry failed: %s", e)
            raise HTTPException(status_code=e.status_code or 502, detail=str(e))

    if response.status_code >= 400:
        logger.warning(
            "Upstream returned %d for model=%s: %s",
            response.status_code,
            actual_model,
            response.text[:200],
        )
        # Forward upstream error verbatim (already Anthropic format)
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": {"type": "api_error", "message": response.text}}
        return JSONResponse(content=error_body, status_code=response.status_code)

    data = response.json()
    _strip_response(data)

    usage = data.get("usage", {})
    content = data.get("content", [])
    logger.info(
        "Beta passthrough response: model=%s, stop_reason=%s, "
        "blocks=%d, input_tokens=%s, output_tokens=%s",
        data.get("model"),
        data.get("stop_reason"),
        len(content),
        usage.get("input_tokens"),
        usage.get("output_tokens"),
    )

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
        logger.info(
            "Beta count_tokens falling back to standard: model=%s, provider=%s",
            model,
            provider_name,
        )
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
        logger.error("Beta count_tokens failed: %s", e)
        raise HTTPException(status_code=e.status_code or 502, detail=str(e))

    if response.status_code >= 400:
        logger.warning(
            "Beta count_tokens upstream %d for model=%s",
            response.status_code,
            actual_model,
        )
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": {"type": "api_error", "message": response.text}}
        return JSONResponse(content=error_body, status_code=response.status_code)

    result = response.json()
    logger.debug(
        "Beta count_tokens: model=%s, input_tokens=%s",
        actual_model,
        result.get("input_tokens"),
    )
    return JSONResponse(content=result)


async def _stream_passthrough(
    provider: CopilotProvider,
    payload: dict,
) -> AsyncGenerator[str, None]:
    """Stream SSE events from Copilot's native Anthropic endpoint.

    Filters out copilot-internal events and strips internal metadata from
    ``message_stop`` events before yielding to the client.

    On a signature validation error (400), strips history thinking blocks
    and retries once before surfacing the error.
    """
    # Build leak guard with tool names from the payload
    from router_maestro.pipeline.leak_guard import RawFrameLeakGuard

    tool_names: set[str] | None = None
    tools = payload.get("tools")
    if tools:
        tool_names = {t.get("name", "") for t in tools if t.get("name")} or None
    leak_guard = RawFrameLeakGuard(allowed_tool_names=tool_names)

    try:
        async with provider._stream_with_auth_retry(
            COPILOT_MESSAGES_PATH,
            json=payload,
            headers_kwargs={},
        ) as response:
            if response.status_code == 400:
                error_text = (await response.aread()).decode()
                if _is_signature_error(error_text):
                    logger.info(
                        "Stream: signature rejected, stripping thinking blocks and retrying"
                    )
                    _strip_history_thinking_blocks(payload)
                    # Fall through to retry below
                else:
                    msg = f"Upstream error ({response.status_code}): {error_text[:200]}"
                    yield _sse_error_event(msg)
                    return
            elif response.status_code >= 400:
                error_text = (await response.aread()).decode()
                msg = f"Upstream error ({response.status_code}): {error_text[:200]}"
                yield _sse_error_event(msg)
                return
            else:
                # Success — stream events
                async for frame in _iter_sse_frames(response, leak_guard=leak_guard):
                    yield frame
                return

        # Retry after stripping thinking blocks (only reached on signature error)
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

            async for frame in _iter_sse_frames(response, leak_guard=leak_guard):
                yield frame

    except ProviderError as e:
        yield _sse_error_event(str(e))
    except asyncio.CancelledError:
        logger.info("Beta anthropic stream cancelled by client")
        raise
    except Exception:
        logger.error("Unexpected error in beta anthropic stream", exc_info=True)
        yield _sse_error_event("Internal server error")


async def _iter_sse_frames(response, *, leak_guard=None) -> AsyncGenerator[str, None]:
    """Iterate SSE frames from a response, filtering copilot-internal events."""
    current_event: str | None = None
    data_buffer: str = ""

    async for line in response.aiter_lines():
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            data_buffer = line[6:]
        elif line == "":
            if current_event and data_buffer:
                # Check leak guard before yielding
                if leak_guard:
                    abort_reason = leak_guard.feed_frame(current_event, data_buffer)
                    if abort_reason:
                        error_event = {
                            "type": "error",
                            "error": {
                                "type": "overloaded",
                                "message": "Overloaded: please retry this request",
                            },
                        }
                        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                        return

                cleaned = _clean_stream_frame(current_event, data_buffer)
                if cleaned is not None:
                    yield f"event: {current_event}\ndata: {cleaned}\n\n"
            elif current_event and not data_buffer:
                yield f"event: {current_event}\ndata: \n\n"
            current_event = None
            data_buffer = ""


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
