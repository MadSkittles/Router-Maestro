"""Anthropic Messages API beta route — native passthrough to Copilot.

For Claude models routed via GitHub Copilot, forwards requests directly to
Copilot's native ``/v1/messages`` endpoint without Anthropic→OpenAI→Anthropic
translation.  Non-Claude or non-Copilot models fall back to the standard
translation-based handler transparently.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, replace

from fastapi import APIRouter, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from router_maestro.providers import (
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponseStatus,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
    classify_upstream_status,
    client_cancelled_outcome,
    exception_outcome,
    unexpected_eof_outcome,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.routing import get_router
from router_maestro.routing.capabilities import CapabilitySupport, Operation, RequestFeatures
from router_maestro.routing.model_ref import qualify_model_id
from router_maestro.routing.route_plan import NoCompatibleRouteError, RouteCandidate, RoutePlan
from router_maestro.routing.router import Router
from router_maestro.server.protocols.errors import (
    client_error_response,
    postcommit_error_data,
    protocol_error_response,
)
from router_maestro.server.routes.anthropic import (
    ANTHROPIC_PING_FRAME,
)
from router_maestro.server.routes.anthropic import (
    messages as standard_messages,
)
from router_maestro.server.streaming import parse_sse_frame, sse_streaming_response
from router_maestro.utils import get_logger
from router_maestro.utils.async_iterators import close_async_iterator
from router_maestro.utils.context_window import resolve_thinking_budget
from router_maestro.utils.reasoning import VALID_EFFORTS, pick_closest_effort

logger = get_logger("server.routes.anthropic_beta")

router = APIRouter()

COPILOT_MESSAGES_PATH = "/v1/messages"

_STRIP_RESPONSE_KEYS = frozenset({"copilot_usage", "stop_details"})
_STRIP_STREAM_MESSAGE_STOP_KEYS = frozenset({"copilot_usage", "amazon-bedrock-invocationMetrics"})
_THINKING_TYPES = frozenset({"enabled", "adaptive", "disabled"})
_ANTHROPIC_SSE_EVENTS = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
        "error",
    }
)


@dataclass(frozen=True, slots=True)
class _ResolvedModel:
    provider_name: str
    actual_model: str
    copilot_provider: CopilotProvider | None


@dataclass(frozen=True, slots=True)
class _NativeModelResolution:
    model: _ResolvedModel
    support: CapabilitySupport
    plan: RoutePlan | None = None


class _NativeUpstreamStatusError(ProviderError):
    """Typed native status failure retaining its response only for route encoding."""

    def __init__(self, response, *, provider: str, model: str) -> None:
        status_code = response.status_code
        kind, retryable = classify_upstream_status(status_code)
        super().__init__(
            f"Native Anthropic upstream returned {status_code}",
            status_code=status_code,
            retryable=retryable,
            kind=kind,
            upstream_status_code=status_code,
            provider=provider,
            model=model,
        )
        self.response = response


class _NativeUnexpectedEOFError(ProviderError):
    """Native Anthropic transport ended cleanly before a terminal event."""


def _raise_for_native_status(response, candidate: RouteCandidate) -> None:
    """Normalize a native non-success status without retaining body in the ledger."""
    if response.status_code >= 400:
        raise _NativeUpstreamStatusError(
            response,
            provider=candidate.model.provider,
            model=candidate.model.upstream_id,
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


def _validate_beta_request_options(body: dict) -> None:
    """Validate representable beta Messages options before transport selection.

    Options that Router-Maestro does not model as typed fields (for example
    ``context_management`` sent by Claude Code) are NOT rejected: a transparent
    proxy forwards or ignores unknown options rather than returning a 400. The
    native passthrough path strips anything outside the Copilot provider's
    outbound contract (``CopilotOutboundContract.forwardable_fields``)
    before forwarding, so unknown top-level keys are ignored, not echoed upstream.

    Only ``output_config`` is validated here, because Router-Maestro actively
    consumes it to shape the native reasoning-effort payload; a malformed value
    cannot be represented and is reported instead of silently mishandled.
    """
    if "output_config" in body:
        output_config = body["output_config"]
        if not isinstance(output_config, dict):
            raise RequestOptionError(
                "output_config must be an object",
                parameter="output_config",
            )

        unsupported_fields = sorted(set(output_config) - {"effort"})
        if unsupported_fields:
            parameter = f"output_config.{unsupported_fields[0]}"
            raise RequestOptionError(
                f"{parameter} is not supported by the native Anthropic transport",
                parameter=parameter,
            )

        effort = output_config.get("effort")
        if effort not in VALID_EFFORTS:
            raise RequestOptionError(
                "output_config.effort must be one of " + ", ".join(VALID_EFFORTS),
                parameter="output_config.effort",
            )


def _validate_native_request_options(body: dict) -> None:
    """Reject explicit options that the native Copilot transport cannot preserve."""
    _validate_beta_request_options(body)

    if "temperature" in body and "top_p" in body:
        raise RequestOptionError(
            "top_p cannot be combined with temperature on the native Anthropic transport",
            parameter="top_p",
        )


def _resolve_native_effort(
    effort: str,
    allowed: tuple[str, ...] | list[str] | None,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Resolve a native effort exactly or downward, preserving unknown catalogs.

    The native passthrough resolves ``output_config.effort`` with its own
    catalog-or-passthrough rule (distinct error surface and no family fallback),
    so it is intentionally NOT routed through
    ``CopilotOutboundContract.resolve_reasoning`` (the chat/responses orchestration).
    Both share the ``pick_closest_effort`` primitive as the single source of the
    downgrade math.
    """
    if allowed is None:
        return effort
    mapped_effort = pick_closest_effort(effort, list(allowed))
    if mapped_effort is None:
        raise RequestOptionError(
            "output_config.effort has no supported tier at or below the requested tier",
            parameter="output_config.effort",
            provider=provider,
            model=model,
        )
    return mapped_effort


def _validate_native_candidate_options(body: dict, candidate: RouteCandidate) -> None:
    """Validate model-specific native options without starting transport work."""
    output_config = body.get("output_config")
    if not isinstance(output_config, dict):
        return
    effort = output_config.get("effort")
    if not isinstance(effort, str) or effort not in VALID_EFFORTS:
        return
    _resolve_native_effort(
        effort,
        candidate.capabilities.reasoning_effort_values,
        provider=candidate.model.provider,
        model=candidate.model.upstream_id,
    )


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


def _validate_explicit_requested_features(plan: RoutePlan | None) -> JSONResponse | None:
    """Reject an explicit model's known feature mismatch independently of transport."""
    if plan is None or not plan.explicit:
        return None
    candidate = plan.primary
    unsupported = next(
        (
            feature
            for feature in plan.features.required()
            if candidate.capabilities.feature(feature) is CapabilitySupport.UNSUPPORTED
        ),
        None,
    )
    if unsupported is None:
        return None
    return _invalid_request(
        f"Model '{candidate.model.qualified_id}' does not support "
        f"the requested {unsupported.value} feature"
    )


def _validate_native_thinking(
    body: dict,
    *,
    validate_max_tokens: bool = True,
) -> JSONResponse | None:
    """Validate raw thinking before either native execution or standard fallback."""
    max_tokens = body.get("max_tokens")
    max_tokens_is_integer = isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
    if (
        validate_max_tokens
        and "max_tokens" in body
        and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0)
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

    if max_tokens_is_integer and budget >= max_tokens:
        return _invalid_request("thinking.budget_tokens must be less than max_tokens")

    return None


def _validate_canonical_thinking_budget(body: dict, max_tokens: int) -> JSONResponse | None:
    """Validate a strict raw budget against the schema-coerced token limit."""
    thinking = body.get("thinking")
    if not isinstance(thinking, dict) or "budget_tokens" not in thinking:
        return None
    budget = thinking["budget_tokens"]
    if isinstance(budget, int) and not isinstance(budget, bool) and budget >= max_tokens:
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


def _is_token_count(value) -> bool:
    """Whether a protocol token count is an exact non-negative integer."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_nullable_string(value, field: str) -> None:
    """Validate a nullable protocol string without echoing its value."""
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")


def _validate_usage(usage: dict, *, required_fields: tuple[str, ...] = ()) -> None:
    """Validate stable token-count fields while allowing future usage metadata."""
    token_fields = {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    }
    for field in required_fields:
        if field not in usage:
            raise TypeError(f"usage {field} is required")
    for field in token_fields & usage.keys():
        if not _is_token_count(usage[field]):
            raise TypeError(f"usage {field} must be a non-negative integer")


def _validate_content_block(block: object) -> None:
    """Validate the stable fields of known Anthropic response content blocks."""
    if not isinstance(block, dict):
        raise TypeError("content block must be an object")
    block_type = block.get("type")
    if not isinstance(block_type, str) or not block_type:
        raise TypeError("content block type must be a non-empty string")
    if block_type == "text":
        if not isinstance(block.get("text"), str):
            raise TypeError("text block text must be a string")
    elif block_type == "thinking":
        if not isinstance(block.get("thinking"), str):
            raise TypeError("thinking block thinking must be a string")
        _validate_nullable_string(block.get("signature"), "thinking signature")
    elif block_type == "redacted_thinking":
        if not isinstance(block.get("data"), str):
            raise TypeError("redacted thinking data must be a string")
    elif block_type == "tool_use" and (
        not isinstance(block.get("id"), str)
        or not isinstance(block.get("name"), str)
        or not isinstance(block.get("input"), dict)
    ):
        raise TypeError("tool use block is malformed")


def _parse_native_message_response(response, *, provider: str, model: str) -> dict:
    """Parse and minimally validate a successful native Messages response."""
    try:
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("native Messages response must be an object")
        required = {
            "id": str,
            "type": str,
            "role": str,
            "content": list,
            "model": str,
            "usage": dict,
        }
        for field, expected_type in required.items():
            if not isinstance(data.get(field), expected_type):
                raise TypeError(f"native Messages response has invalid {field}")
        if data["type"] != "message" or data["role"] != "assistant":
            raise ValueError("native Messages response has invalid discriminator")
        for block in data["content"]:
            _validate_content_block(block)
        _validate_usage(
            data["usage"],
            required_fields=("input_tokens", "output_tokens"),
        )
        _validate_nullable_string(data.get("stop_reason"), "stop_reason")
        _validate_nullable_string(data.get("stop_sequence"), "stop_sequence")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ProviderError(
            "Native Anthropic upstream returned a malformed response",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            upstream_status_code=200,
            provider=provider,
            model=model,
            cause=error,
        ) from error
    return data


def _apply_thinking_budget_native(
    body: dict,
    actual_model: str,
    reasoning_effort_values: tuple[str, ...] | list[str] | None = None,
) -> dict:
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
        effort = _resolve_native_effort(
            effort,
            reasoning_effort_values,
            provider="github-copilot",
            model=actual_model,
        )
        body["output_config"] = {"effort": effort}

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

    from router_maestro.runtime import get_current_request_context

    context = get_current_request_context()
    if context is not None:
        priorities = context.config
    else:
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


async def _resolve_model(model: str) -> _ResolvedModel:
    """Resolve model to (provider_name, actual_model_id, provider_if_copilot).

    Returns (provider_name, actual_model, None) for non-Copilot providers.
    Raises ProviderError if the model cannot be resolved.
    """
    model_router = get_router()
    provider_name, actual_model, provider = await model_router._resolve_provider(model)
    if isinstance(provider, CopilotProvider):
        return _ResolvedModel(provider_name, actual_model, provider)
    return _ResolvedModel(provider_name, actual_model, None)


async def _resolve_native_model(
    model: str,
    features: RequestFeatures,
) -> _NativeModelResolution:
    """Resolve native Anthropic planning while preserving safe standard fallback."""
    try:
        plan = await get_router().plan_route(
            model,
            Operation.NATIVE_ANTHROPIC,
            features,
        )
    except NoCompatibleRouteError:
        return _NativeModelResolution(
            model=await _resolve_model(model),
            support=CapabilitySupport.UNSUPPORTED,
        )
    except ProviderError:
        raise

    candidate = plan.primary
    provider = candidate.provider
    return _NativeModelResolution(
        model=_ResolvedModel(
            provider_name=candidate.model.provider,
            actual_model=candidate.model.upstream_id,
            copilot_provider=(provider if isinstance(provider, CopilotProvider) else None),
        ),
        support=candidate.support,
        plan=plan,
    )


def _native_payload_for_candidate(body: dict, candidate: RouteCandidate) -> dict:
    """Build one candidate-owned native Anthropic payload."""
    payload = deepcopy(body)
    payload = _apply_thinking_budget_native(
        payload,
        candidate.model.upstream_id,
        candidate.capabilities.reasoning_effort_values,
    )
    payload["model"] = candidate.model.upstream_id
    return payload


async def _send_native_nonstream(candidate: RouteCandidate, body: dict):
    """Send one native Anthropic attempt without moving wire concerns into Router."""
    provider = candidate.provider
    if not isinstance(provider, CopilotProvider):
        raise ProviderError(
            "Provider does not support native Anthropic transport",
            status_code=501,
            retryable=False,
            kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
            provider=candidate.model.provider,
            model=candidate.model.upstream_id,
        )
    payload = _native_payload_for_candidate(body, candidate)
    await provider.ensure_token()
    response = await provider._send_with_auth_retry(
        "POST",
        COPILOT_MESSAGES_PATH,
        json=payload,
        model=candidate.model.upstream_id,
    )
    if response.status_code == 400 and _is_signature_error(response.text):
        logger.info("Signature rejected by upstream, stripping thinking blocks and retrying")
        _strip_history_thinking_blocks(payload)
        response = await provider._send_with_auth_retry(
            "POST",
            COPILOT_MESSAGES_PATH,
            json=payload,
            model=candidate.model.upstream_id,
        )
    _raise_for_native_status(response, candidate)
    data = _parse_native_message_response(
        response,
        provider=candidate.model.provider,
        model=candidate.model.upstream_id,
    )
    # The frozen candidate owns public identity. Upstream ``model`` strings do
    # not carry enough provenance to distinguish an already-qualified ID from
    # a raw namespaced ID beginning with the provider name.
    data["model"] = candidate.model.qualified_id
    return response, data


async def _stream_native_candidate(
    candidate: RouteCandidate,
    body: dict,
) -> AsyncGenerator[str, None]:
    """Open one candidate's native stream and surface pre-commit typed failures."""
    provider = candidate.provider
    if not isinstance(provider, CopilotProvider):
        raise ProviderError(
            "Provider does not support native Anthropic transport",
            status_code=501,
            retryable=False,
            kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
            provider=candidate.model.provider,
            model=candidate.model.upstream_id,
        )
    payload = _native_payload_for_candidate(body, candidate)
    await provider.ensure_token()
    stream = _stream_passthrough(provider, payload, raise_provider_errors=True)
    try:
        try:
            async for frame in stream:
                yield _qualify_native_stream_frame(frame, candidate.model.qualified_id)
        except ProviderError as error:
            if error.kind is not ProviderFailureKind.UPSTREAM_PROTOCOL or error.provider:
                raise
            error_type = (
                _NativeUnexpectedEOFError
                if isinstance(error, _NativeUnexpectedEOFError)
                else ProviderError
            )
            raise error_type(
                error.safe_message,
                status_code=error.status_code,
                retryable=error.retryable,
                kind=error.kind,
                upstream_status_code=error.upstream_status_code,
                provider=candidate.model.provider,
                model=candidate.model.upstream_id,
                cause=error.cause,
            ) from error
    finally:
        await close_async_iterator(stream)


async def _beta_planned_native_stream(
    plan: RoutePlan,
    body: dict,
    *,
    pipeline=None,
) -> AsyncGenerator[str, None]:
    """Prime and stream a native beta plan under the SSE keepalive wrapper.

    ``execute_plan_stream`` primes the upstream by awaiting Copilot's first
    frame, which for a large context can take minutes. Running that await here
    — after the eager ping and inside the generator driven by
    ``resilient_sse_generator`` — keeps keepalive pings flowing during the
    priming window, so the client's streaming idle watchdog never starves.

    A stream-open failure therefore surfaces as an SSE ``error`` event rather
    than an HTTP status, mirroring the standard Anthropic streaming path (the
    response headers have already committed by the time priming runs).
    """
    yield ANTHROPIC_PING_FRAME
    try:
        selected_stream, _used_provider = await Router.execute_plan_stream(
            plan,
            lambda candidate: _stream_native_candidate(candidate, body),
            log_prefix="Beta native",
        )
    except ProviderError as error:
        logger.error(
            "beta_native_stream_open_failed kind=%s retryable=%s",
            error.kind.value,
            str(error.retryable).lower(),
        )
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=exception_outcome(error.safe_message, code="provider_error"),
                body_summary=error.safe_message,
            )
        yield _sse_error_event(error)
        return
    except asyncio.CancelledError:
        # Client disconnected while priming the upstream — the very window this
        # eager-ping restructure protects. Finalize the ledger as cancelled so
        # the outcome matches an in-stream disconnect (``_encode_native_stream_errors``
        # handles the same case once frames are flowing).
        if pipeline is not None:
            pipeline.finish(wire_status=200, outcome=client_cancelled_outcome())
        raise
    except Exception:
        # An unexpected priming failure (routing/normalization bug, provider
        # hook) would otherwise reach ``resilient_sse_generator``, which logs
        # and swallows it — leaving the client with a lone ping + EOF and the
        # ledger unfinished. Mirror the standard path's generic handler.
        logger.error("Unexpected error opening beta native stream", exc_info=True)
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=exception_outcome("Internal server error", code="server_error"),
                body_summary="Internal server error",
            )
        yield _sse_error_event(RuntimeError("Internal server error"))
        return
    # Own the delegated iterator so its cleanup (upstream close, cancellation
    # outcome) runs deterministically when this generator is closed, rather
    # than being deferred to async-generator finalization.
    encoded = _encode_native_stream_errors(selected_stream, pipeline=pipeline)
    try:
        async for frame in encoded:
            yield frame
    finally:
        await close_async_iterator(encoded)


async def _encode_native_stream_errors(
    stream: AsyncIterator[str],
    *,
    pipeline=None,
) -> AsyncGenerator[str, None]:
    """Apply raw stream guards and preserve native semantic terminal state."""
    terminal_outcome: TerminalOutcome | None = None
    response_status = ResponseStatus.COMPLETED
    try:
        async for frame in stream:
            event_type, data = parse_sse_frame(frame)
            if pipeline is not None:
                abort_reason = _feed_native_frame_guards(
                    pipeline,
                    event_type,
                    data,
                    frame,
                )
                if abort_reason is not None:
                    terminal_outcome = exception_outcome(abort_reason, code="overloaded")
                    pipeline.finish(
                        wire_status=200,
                        outcome=terminal_outcome,
                        body_summary=abort_reason,
                    )
                    yield _native_overload_event()
                    return
            if event_type == "message_delta" and isinstance(data, dict):
                delta = data.get("delta")
                if isinstance(delta, dict):
                    stop_reason = delta.get("stop_reason")
                    if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
                        response_status = ResponseStatus.INCOMPLETE
                    elif stop_reason is not None:
                        response_status = ResponseStatus.COMPLETED
            frame_outcome = _native_frame_outcome(
                event_type,
                data,
                response_status=response_status,
            )
            if frame_outcome is not None:
                terminal_outcome = frame_outcome
            yield frame
            if terminal_outcome is not None:
                if pipeline is not None:
                    pipeline.finish(wire_status=200, outcome=terminal_outcome)
                return
        terminal_outcome = unexpected_eof_outcome()
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary=terminal_outcome.error.message,
            )
        yield _sse_error_event(
            ProviderError(
                terminal_outcome.error.message,
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        )
    except _NativeUnexpectedEOFError as error:
        terminal_outcome = unexpected_eof_outcome()
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary=terminal_outcome.error.message,
            )
        yield _sse_error_event(error)
    except ProviderError as error:
        terminal_outcome = exception_outcome(error.safe_message, code="provider_error")
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary=error.safe_message,
            )
        yield _sse_error_event(error)
    except asyncio.CancelledError:
        if pipeline is not None:
            pipeline.finish(wire_status=200, outcome=client_cancelled_outcome())
        raise
    except Exception:
        terminal_outcome = exception_outcome("Internal server error", code="server_error")
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary="Internal server error",
            )
        logger.error("Unexpected error in beta anthropic stream", exc_info=True)
        yield _sse_error_event("Internal server error")
    finally:
        await close_async_iterator(stream)


def _feed_native_frame_guards(
    pipeline,
    event_type: str | None,
    data: dict | None,
    wire_frame: str,
) -> str | None:
    """Project payload-bearing native events onto the shared guard contract."""
    for chunk in _native_guard_chunks(event_type, data, wire_frame):
        abort_reason = pipeline.feed_stream(chunk)
        if abort_reason is not None:
            return abort_reason
    return None


def _native_guard_chunks(
    event_type: str | None,
    data: dict | None,
    wire_frame: str,
):
    """Project one raw frame into visible fields plus exact wire-volume payload."""
    from router_maestro.providers.base import ChatStreamChunk

    if data is None:
        return []

    if not _native_frame_has_guard_payload(event_type, data):
        return []

    if event_type == "message_start":
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return [ChatStreamChunk(content="", opaque_payload=wire_frame)]
        chunks = [
            chunk
            for block in content
            if isinstance(block, dict)
            for chunk in _native_content_block_guard_chunks(block)
        ]
        return [_combine_native_guard_chunks(chunks, wire_frame)]

    if event_type == "content_block_start":
        block = data.get("content_block")
        chunks = _native_content_block_guard_chunks(block) if isinstance(block, dict) else []
        return [_combine_native_guard_chunks(chunks, wire_frame)]

    if event_type != "content_block_delta":
        return [ChatStreamChunk(content="", opaque_payload=wire_frame)]
    delta = data.get("delta")
    if not isinstance(delta, dict):
        return [ChatStreamChunk(content="", opaque_payload=wire_frame)]
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        return [ChatStreamChunk(content=delta.get("text", ""), opaque_payload=wire_frame)]
    if delta_type == "thinking_delta":
        return [
            ChatStreamChunk(
                content="",
                thinking=delta.get("thinking", ""),
                opaque_payload=wire_frame,
            )
        ]
    if delta_type == "signature_delta":
        return [
            ChatStreamChunk(
                content="",
                thinking_signature=delta.get("signature", ""),
                opaque_payload=wire_frame,
            )
        ]
    if delta_type == "input_json_delta":
        return [
            ChatStreamChunk(
                content="",
                tool_calls=[{"function": {"arguments": delta.get("partial_json", "")}}],
                opaque_payload=wire_frame,
            )
        ]
    return [ChatStreamChunk(content="", opaque_payload=wire_frame)]


def _native_content_block_guard_chunks(block: dict):
    """Project one initial native content block without scanning opaque payloads."""
    from router_maestro.providers.base import ChatStreamChunk

    block_type = block.get("type")
    if block_type == "text":
        return [ChatStreamChunk(content=block.get("text", ""))]
    if block_type == "thinking":
        return [
            ChatStreamChunk(
                content="",
                thinking=block.get("thinking", ""),
                thinking_signature=block.get("signature"),
            )
        ]
    if block_type == "redacted_thinking":
        return [ChatStreamChunk(content="", thinking_signature=block.get("data", ""))]
    if block_type == "tool_use":
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            return []
        return [
            ChatStreamChunk(
                content="",
                tool_calls=[
                    {
                        "function": {
                            "arguments": json.dumps(
                                tool_input,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        }
                    }
                ],
            )
        ]
    return []


def _combine_native_guard_chunks(chunks, wire_frame: str):
    """Collapse initial block projections so one emitted frame is counted once."""
    from router_maestro.providers.base import ChatStreamChunk

    if not chunks:
        return ChatStreamChunk(content="", opaque_payload=wire_frame)
    return ChatStreamChunk(
        content="".join(chunk.content for chunk in chunks if chunk.content),
        thinking="".join(chunk.thinking for chunk in chunks if chunk.thinking) or None,
        thinking_signature="".join(
            chunk.thinking_signature for chunk in chunks if chunk.thinking_signature
        )
        or None,
        tool_calls=[tool_call for chunk in chunks for tool_call in (chunk.tool_calls or [])]
        or None,
        opaque_payload=wire_frame,
    )


def _native_frame_has_guard_payload(event_type: str | None, data: dict) -> bool:
    """Select emitted frames that carry output or forward-compatible extensions."""
    if event_type in {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "message_delta",
        "error",
    }:
        return True
    baseline_fields = {
        "content_block_stop": {"type", "index"},
        "message_stop": {"type"},
        "ping": {"type"},
    }
    baseline = baseline_fields.get(event_type)
    return baseline is not None and bool(set(data) - baseline)


def _native_frame_outcome(
    event_type: str | None,
    data: dict | None,
    *,
    response_status: ResponseStatus,
) -> TerminalOutcome | None:
    """Map a native Anthropic terminal event onto the shared semantic outcome."""
    if event_type == "error":
        error = data.get("error") if isinstance(data, dict) else None
        error_type = error.get("type") if isinstance(error, dict) else "api_error"
        message = error.get("message") if isinstance(error, dict) else "Upstream response failed"
        return TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.FAILED,
            error=TerminalError(code=str(error_type), message=str(message)),
        )
    if event_type != "message_stop":
        return None
    return TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=response_status,
        incomplete_details=(
            {"reason": "max_output_tokens"}
            if response_status is ResponseStatus.INCOMPLETE
            else None
        ),
    )


def _native_overload_event() -> str:
    return _sse_error_event(
        ProviderError(
            "Overloaded: please retry this request",
            status_code=529,
            retryable=True,
            kind=ProviderFailureKind.RATE_LIMIT,
        )
    )


def _qualify_native_stream_frame(frame: str, qualified_model: str) -> str:
    """Rewrite one native message_start frame with the selected public model ID."""
    event_type: str | None = None
    data: dict | None = None
    for line in frame.splitlines():
        if line.startswith("event: "):
            event_type = line.removeprefix("event: ")
        elif line.startswith("data: "):
            try:
                parsed = json.loads(line.removeprefix("data: "))
            except json.JSONDecodeError:
                return frame
            if isinstance(parsed, dict):
                data = parsed
    if event_type != "message_start" or data is None:
        return frame
    message = data.get("message")
    if not isinstance(message, dict):
        return frame
    message["model"] = qualified_model
    return f"event: message_start\ndata: {json.dumps(data)}\n\n"


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

    try:
        _validate_beta_request_options(body)
    except RequestOptionError as error:
        return client_error_response(error, "anthropic")

    stream = body.get("stream", False)
    thinking = body.get("thinking")
    thinking_type = thinking.get("type") if isinstance(thinking, dict) else None
    thinking_budget = thinking.get("budget_tokens") if isinstance(thinking, dict) else None
    output_config = body.get("output_config")
    output_effort = output_config.get("effort") if isinstance(output_config, dict) else None
    features = RequestFeatures.for_anthropic_native(body)

    # Debug: log all top-level keys to identify fields Copilot might reject
    logger.debug("Beta request body keys: %s", sorted(body.keys()))
    logger.info(
        "Received beta Anthropic request: model=%s, stream=%s, max_tokens=%s, "
        "thinking_type=%s, thinking_budget=%s, output_effort=%s",
        model,
        stream,
        body.get("max_tokens"),
        thinking_type,
        thinking_budget,
        output_effort,
    )

    # Resolve provider and check native eligibility
    try:
        native_resolution = await _resolve_native_model(model, features)
    except ProviderError as error:
        return protocol_error_response(error, "anthropic")
    provider_name = native_resolution.model.provider_name
    actual_model = native_resolution.model.actual_model
    copilot_provider = native_resolution.model.copilot_provider
    native_support = native_resolution.support
    operation_support = (
        native_resolution.plan.primary.capabilities.operation(Operation.NATIVE_ANTHROPIC)
        if native_resolution.plan is not None
        else native_support
    )

    uses_native_transport = (
        copilot_provider is not None and native_support is not CapabilitySupport.UNSUPPORTED
    )
    validation_error = _validate_native_thinking(
        body,
        validate_max_tokens=uses_native_transport,
    )
    if validation_error is not None:
        return validation_error

    if copilot_provider is None or operation_support is CapabilitySupport.UNSUPPORTED:
        feature_error = _validate_explicit_requested_features(native_resolution.plan)
        if feature_error is not None:
            return feature_error
        logger.info(
            "Beta route falling back to standard path: model=%s, provider=%s",
            model,
            provider_name,
        )
        # The standard handler adapts an explicit selection through a translated
        # transport while preserving that model; this is not model fallback.
        parsed_request = await _parse_as_anthropic_request(body_bytes)
        if isinstance(parsed_request, JSONResponse):
            return parsed_request
        canonical_error = _validate_canonical_thinking_budget(body, parsed_request.max_tokens)
        if canonical_error is not None:
            return canonical_error
        return await standard_messages(
            request=parsed_request,
            raw_request=raw_request,
        )

    if native_support is CapabilitySupport.UNSUPPORTED:
        try:
            Router._validate_plan_primary(native_resolution.plan)
        except ProviderError as error:
            return _invalid_request(error.safe_message)

    try:
        _validate_native_request_options(body)
        if native_resolution.plan is not None:
            validated_plan = Router.prevalidate_plan(
                native_resolution.plan,
                lambda candidate: _validate_native_candidate_options(body, candidate),
            )
            native_resolution = replace(native_resolution, plan=validated_plan)
    except RequestOptionError as error:
        return client_error_response(error, "anthropic")

    logger.info(
        "Beta route using native passthrough: model=%s -> %s, stream=%s",
        model,
        actual_model,
        stream,
    )

    # Sanitize before the budget helper so unsupported output_config siblings
    # cannot survive even if the helper is replaced by an integration hook.
    _sanitize_output_config(body)

    # Plan execution normalizes a candidate-owned copy for each model. The
    # legacy plan-less compatibility path still normalizes the resolved model once.
    if native_resolution.plan is None:
        try:
            body = _apply_thinking_budget_native(body, actual_model)
        except RequestOptionError as error:
            return client_error_response(error, "anthropic")
        body["model"] = actual_model

    # Defense in depth for fields introduced by internal transformations. Raw
    # client options have already passed the shallow beta option gate above.
    forwardable = copilot_provider.outbound_contract.forwardable_fields(Operation.NATIVE_ANTHROPIC)
    if forwardable is not None:
        unknown = set(body.keys()) - forwardable
        if unknown:
            logger.debug("Stripping unknown fields before passthrough: %s", unknown)
            for key in unknown:
                del body[key]

    # Legacy test/integration hooks may still return a resolution without a plan.
    # Real native routing always executes the immutable plan below.
    if native_resolution.plan is None:
        await copilot_provider.ensure_token()

    if stream:
        from router_maestro.pipeline import RequestPipeline

        tool_names = {
            tool.get("name", "")
            for tool in body.get("tools", [])
            if isinstance(tool, dict) and tool.get("name")
        } or None
        from router_maestro.runtime import get_current_request_context

        context = get_current_request_context()
        pipeline = RequestPipeline.create(
            request_id=(context.request_id if context is not None else f"beta-{actual_model}"),
            model=model,
            tool_names=tool_names,
        )
        if native_resolution.plan is not None:
            return sse_streaming_response(
                _beta_planned_native_stream(
                    native_resolution.plan,
                    body,
                    pipeline=pipeline,
                ),
                keepalive_frame=ANTHROPIC_PING_FRAME,
            )
        return sse_streaming_response(
            _encode_native_stream_errors(
                _stream_passthrough(
                    copilot_provider,
                    body,
                    raise_provider_errors=True,
                ),
                pipeline=pipeline,
            ),
            keepalive_frame=ANTHROPIC_PING_FRAME,
        )

    # Non-streaming passthrough — consume the frozen native RoutePlan when present.
    if native_resolution.plan is not None:
        try:
            result, used_provider = await Router.execute_plan_nonstream(
                native_resolution.plan,
                lambda candidate: _send_native_nonstream(candidate, body),
            )
            response, data = result
            model_is_qualified = True
        except ProviderError as e:
            logger.error(
                "beta_native_request_failed kind=%s retryable=%s",
                e.kind.value,
                str(e.retryable).lower(),
            )
            if isinstance(e, _NativeUpstreamStatusError):
                try:
                    error_body = e.response.json()
                except Exception:
                    error_body = {"error": {"type": "api_error", "message": e.response.text}}
                return JSONResponse(content=error_body, status_code=e.status_code)
            raise HTTPException(status_code=e.status_code or 502, detail=str(e))
    else:
        try:
            response = await copilot_provider._send_with_auth_retry(
                "POST",
                COPILOT_MESSAGES_PATH,
                json=body,
                model=body.get("model"),
            )
        except ProviderError as e:
            logger.error(
                "beta_native_request_failed kind=%s retryable=%s",
                e.kind.value,
                str(e.retryable).lower(),
            )
            raise HTTPException(status_code=e.status_code or 502, detail=str(e))

        # Preserve the legacy signature retry for plan-less test/integration hooks.
        if response.status_code == 400 and _is_signature_error(response.text):
            logger.info("Signature rejected by upstream, stripping thinking blocks and retrying")
            _strip_history_thinking_blocks(body)
            try:
                response = await copilot_provider._send_with_auth_retry(
                    "POST",
                    COPILOT_MESSAGES_PATH,
                    json=body,
                    model=body.get("model"),
                )
            except ProviderError as e:
                logger.error(
                    "beta_native_retry_failed kind=%s retryable=%s",
                    e.kind.value,
                    str(e.retryable).lower(),
                )
                raise HTTPException(status_code=e.status_code or 502, detail=str(e))

    if response.status_code >= 400:
        logger.warning(
            "native_upstream_status status=%d model=%s body_bytes=%d",
            response.status_code,
            actual_model,
            len(response.text.encode("utf-8", errors="replace")),
        )
        # Forward upstream error verbatim (already Anthropic format)
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": {"type": "api_error", "message": response.text}}
        return JSONResponse(content=error_body, status_code=response.status_code)

    if native_resolution.plan is None:
        try:
            data = _parse_native_message_response(
                response,
                provider=provider_name,
                model=actual_model,
            )
        except ProviderError as error:
            raise HTTPException(status_code=502, detail=error.safe_message) from error
        used_provider = provider_name
        model_is_qualified = False
    if not model_is_qualified:
        data["model"] = qualify_model_id(used_provider, data["model"])
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

    try:
        resolution = await _resolve_model(model)
    except ProviderError as error:
        return protocol_error_response(error, "anthropic")
    provider_name = resolution.provider_name
    actual_model = resolution.actual_model
    copilot_provider = resolution.copilot_provider

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

    try:
        body["model"] = actual_model
        input_tokens = await copilot_provider.count_native_anthropic_tokens(
            body,
            model=actual_model,
        )
    except ProviderError as e:
        logger.error(
            "beta_count_tokens_failed kind=%s retryable=%s",
            e.kind.value,
            str(e.retryable).lower(),
        )
        return protocol_error_response(e, "anthropic")

    logger.debug(
        "Beta count_tokens: model=%s, input_tokens=%s",
        actual_model,
        input_tokens,
    )
    return JSONResponse(content={"input_tokens": input_tokens})


async def _stream_passthrough(
    provider: CopilotProvider,
    payload: dict,
    *,
    raise_provider_errors: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream SSE events from Copilot's native Anthropic endpoint.

    Filters out copilot-internal events and strips internal metadata from
    ``message_stop`` events before yielding to the client.

    On a signature validation error (400), strips history thinking blocks
    and retries once before surfacing the error.
    """
    try:
        async with provider._stream_with_auth_retry(
            COPILOT_MESSAGES_PATH,
            json=payload,
            headers_kwargs={},
            model=payload.get("model"),
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
                    if raise_provider_errors:
                        raise _NativeUpstreamStatusError(
                            response,
                            provider=provider.name,
                            model=payload.get("model", ""),
                        )
                    msg = f"Upstream error ({response.status_code}): {error_text[:200]}"
                    yield _sse_error_event(msg)
                    return
            elif response.status_code >= 400:
                error_text = (await response.aread()).decode()
                if raise_provider_errors:
                    raise _NativeUpstreamStatusError(
                        response,
                        provider=provider.name,
                        model=payload.get("model", ""),
                    )
                msg = f"Upstream error ({response.status_code}): {error_text[:200]}"
                yield _sse_error_event(msg)
                return
            else:
                # Success — stream events
                async for frame in _iter_sse_frames(response):
                    yield frame
                return

        # Retry after stripping thinking blocks (only reached on signature error)
        async with provider._stream_with_auth_retry(
            COPILOT_MESSAGES_PATH,
            json=payload,
            headers_kwargs={},
            model=payload.get("model"),
        ) as response:
            if response.status_code >= 400:
                error_text = (await response.aread()).decode()
                if raise_provider_errors:
                    raise _NativeUpstreamStatusError(
                        response,
                        provider=provider.name,
                        model=payload.get("model", ""),
                    )
                msg = f"Upstream error ({response.status_code}): {error_text[:200]}"
                yield _sse_error_event(msg)
                return

            async for frame in _iter_sse_frames(response):
                yield frame

    except _NativeUnexpectedEOFError:
        raise
    except ProviderError as e:
        if raise_provider_errors:
            raise
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
    terminal_received = False

    async for line in response.aiter_lines():
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            data_buffer = line[6:]
        elif line == "":
            if current_event and data_buffer:
                cleaned = _clean_stream_frame(current_event, data_buffer)
                if cleaned is None:
                    current_event = None
                    data_buffer = ""
                    continue

                # Check leak guard before yielding
                if leak_guard:
                    abort_reason = leak_guard.feed_frame(current_event, cleaned)
                    if abort_reason:
                        terminal_received = True
                        yield _sse_error_event(
                            ProviderError(
                                "Overloaded: please retry this request",
                                status_code=529,
                                retryable=True,
                                kind=ProviderFailureKind.RATE_LIMIT,
                            )
                        )
                        return

                if current_event in {"message_stop", "error"}:
                    terminal_received = True
                yield f"event: {current_event}\ndata: {cleaned}\n\n"
                if terminal_received:
                    return
            elif current_event and not data_buffer:
                _raise_native_stream_protocol_error(ValueError("SSE event is missing data"))
            current_event = None
            data_buffer = ""

    if current_event is not None or data_buffer:
        _raise_native_stream_protocol_error(ValueError("truncated upstream SSE frame"))
    if not terminal_received:
        _raise_native_stream_unexpected_eof(
            ValueError("upstream SSE ended before a terminal event")
        )


def _raise_native_stream_protocol_error(cause: BaseException) -> None:
    """Raise a safe retryable failure for malformed native Anthropic SSE."""
    raise ProviderError(
        "Native Anthropic upstream returned a malformed stream",
        status_code=502,
        retryable=True,
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        upstream_status_code=200,
        cause=cause,
    ) from cause


def _raise_native_stream_unexpected_eof(cause: BaseException) -> None:
    """Raise the typed signal for clean EOF before a native terminal event."""
    raise _NativeUnexpectedEOFError(
        "Native Anthropic upstream returned a malformed stream",
        status_code=502,
        retryable=True,
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        upstream_status_code=200,
        cause=cause,
    ) from cause


def _validate_native_stream_event_shape(event_type: str, data: dict) -> None:
    """Validate only the stable minimum shape of a known Anthropic SSE event."""

    def valid_index(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    if event_type == "message_start":
        message = data.get("message")
        if not isinstance(message, dict):
            _raise_native_stream_protocol_error(ValueError("invalid message_start message"))
        if message.get("type") != "message":
            _raise_native_stream_protocol_error(ValueError("invalid message_start type"))
        if not isinstance(message.get("id"), str) or not message["id"]:
            _raise_native_stream_protocol_error(ValueError("invalid message_start id"))
        if not isinstance(message.get("model"), str) or not message["model"]:
            _raise_native_stream_protocol_error(ValueError("invalid message_start model"))
        if message.get("role") != "assistant":
            _raise_native_stream_protocol_error(ValueError("invalid message_start role"))
        if not isinstance(message.get("content"), list):
            _raise_native_stream_protocol_error(ValueError("invalid message_start content"))
        usage = message.get("usage")
        if not isinstance(usage, dict):
            _raise_native_stream_protocol_error(ValueError("invalid message_start usage"))
        try:
            for block in message["content"]:
                _validate_content_block(block)
            _validate_usage(usage, required_fields=("input_tokens",))
            _validate_nullable_string(message.get("stop_reason"), "message_start stop_reason")
            _validate_nullable_string(message.get("stop_sequence"), "message_start stop_sequence")
        except (KeyError, TypeError, ValueError) as error:
            _raise_native_stream_protocol_error(error)
    elif event_type == "content_block_start":
        if not valid_index(data.get("index")):
            _raise_native_stream_protocol_error(ValueError("invalid content_block_start"))
        try:
            _validate_content_block(data.get("content_block"))
        except (TypeError, ValueError) as error:
            _raise_native_stream_protocol_error(error)
    elif event_type == "content_block_delta":
        delta = data.get("delta")
        if not valid_index(data.get("index")) or not isinstance(delta, dict):
            _raise_native_stream_protocol_error(ValueError("invalid content_block_delta"))
        delta_type = delta.get("type")
        if not isinstance(delta_type, str) or not delta_type:
            _raise_native_stream_protocol_error(ValueError("invalid content delta type"))
        field_by_type = {
            "text_delta": "text",
            "thinking_delta": "thinking",
            "signature_delta": "signature",
            "input_json_delta": "partial_json",
        }
        field = field_by_type.get(delta_type)
        if field is not None and not isinstance(delta.get(field), str):
            _raise_native_stream_protocol_error(TypeError(f"{delta_type} {field} must be a string"))
    elif event_type == "content_block_stop":
        if not valid_index(data.get("index")):
            _raise_native_stream_protocol_error(ValueError("invalid content_block_stop"))
    elif event_type == "message_delta":
        delta = data.get("delta")
        usage = data.get("usage")
        if not isinstance(delta, dict) or not isinstance(usage, dict):
            _raise_native_stream_protocol_error(ValueError("invalid message_delta"))
        try:
            _validate_nullable_string(delta.get("stop_reason"), "message_delta stop_reason")
            _validate_nullable_string(delta.get("stop_sequence"), "message_delta stop_sequence")
            _validate_usage(usage, required_fields=("output_tokens",))
        except (KeyError, TypeError, ValueError) as error:
            _raise_native_stream_protocol_error(error)
    elif event_type == "error":
        error = data.get("error")
        if (
            not isinstance(error, dict)
            or not isinstance(error.get("type"), str)
            or not error["type"]
            or not isinstance(error.get("message"), str)
        ):
            _raise_native_stream_protocol_error(ValueError("invalid error event"))


def _clean_stream_frame(event_type: str, data_str: str) -> str | None:
    """Clean a single SSE frame's data payload.

    Returns the cleaned data string, or None to suppress the event entirely.
    """
    # Filter out copilot-internal events
    if event_type == "copilot_usage":
        return None

    if event_type not in _ANTHROPIC_SSE_EVENTS:
        _raise_native_stream_protocol_error(ValueError("unknown Anthropic SSE event"))

    try:
        data = json.loads(data_str)
    except (json.JSONDecodeError, TypeError) as error:
        _raise_native_stream_protocol_error(error)
    if not isinstance(data, dict) or data.get("type") != event_type:
        _raise_native_stream_protocol_error(ValueError("Anthropic SSE discriminator mismatch"))
    _validate_native_stream_event_shape(event_type, data)

    # For message_start, strip copilot fields from nested message
    if event_type == "message_start":
        msg = data.get("message")
        for key in _STRIP_RESPONSE_KEYS:
            msg.pop(key, None)
        return json.dumps(data)

    # For message_stop, strip bedrock metrics
    if event_type == "message_stop":
        for key in _STRIP_STREAM_MESSAGE_STOP_KEYS:
            data.pop(key, None)
        return json.dumps(data)

    return json.dumps(data)


def _sse_error_event(error: Exception | str) -> str:
    """Format an Anthropic SSE error event."""
    error_event = (
        postcommit_error_data(error, "anthropic")
        if isinstance(error, Exception)
        else {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": error,
            },
        }
    )
    return f"event: error\ndata: {json.dumps(error_event)}\n\n"


async def _parse_as_anthropic_request(body_bytes: bytes):
    """Parse raw body bytes into an AnthropicMessagesRequest for fallback."""
    from router_maestro.server.schemas.anthropic import AnthropicMessagesRequest

    body = json.loads(body_bytes)
    try:
        return AnthropicMessagesRequest.model_validate(body)
    except ValidationError as error:
        errors = error.errors(include_input=False, include_url=False)
        if not errors:
            return _invalid_request("Invalid request body")
        first = errors[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = str(first.get("msg", "Invalid value"))
        return _invalid_request(f"{location}: {message}" if location else message)
