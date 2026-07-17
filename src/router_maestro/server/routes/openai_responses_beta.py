"""OpenAI Responses API beta route — native passthrough to Copilot.

For GPT-5 family models routed via GitHub Copilot, forwards Responses API
requests directly to Copilot's native ``/responses`` endpoint without the
parse-normalize-rebuild cycle the standard ``/api/openai/v1/responses`` route
performs.  This preserves fidelity-relevant fields Router-Maestro does not model
(``include``, ``previous_response_id``, ``prompt_cache_key``, ``text``, the
encrypted-reasoning round-trip).  Non-eligible models fall back to the standard
translated handler transparently.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass

from fastapi import APIRouter
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from router_maestro.providers import (
    ProviderError,
    ProviderFailureKind,
    ResponseStatus,
    TerminalOutcome,
    TransportTermination,
    classify_upstream_status,
    client_cancelled_outcome,
    exception_outcome,
    unexpected_eof_outcome,
)
from router_maestro.providers.copilot import COPILOT_RESPONSES_PATH, CopilotProvider
from router_maestro.providers.copilot_support.responses_codec import CopilotResponsesCodec
from router_maestro.routing import get_router
from router_maestro.routing.capabilities import CapabilitySupport, Operation, RequestFeatures
from router_maestro.routing.route_plan import NoCompatibleRouteError, RouteCandidate, RoutePlan
from router_maestro.routing.router import Router
from router_maestro.server.protocols.errors import (
    postcommit_error_data,
    protocol_error_response,
)
from router_maestro.server.routes.responses import create_response
from router_maestro.server.schemas import ResponsesRequest
from router_maestro.server.streaming import parse_sse_frame, sse_streaming_response
from router_maestro.utils import get_logger
from router_maestro.utils.async_iterators import close_async_iterator

logger = get_logger("server.routes.openai_responses_beta")

router = APIRouter()

# Copilot-internal SSE events that must not reach the client.
_STRIP_STREAM_EVENTS = frozenset({"copilot_usage"})

# Terminal Responses SSE events carrying a nested ``response`` snapshot. Mirrors
# the terminal set in ``CopilotResponsesCodec.decode_stream``; ``response.done``
# is included because some upstream gateways use it as the terminal event name
# instead of ``response.completed``.
_TERMINAL_STREAM_EVENTS = frozenset(
    {
        "response.done",
        "response.completed",
        "response.incomplete",
        "response.failed",
        "response.cancelled",
    }
)


@dataclass(frozen=True, slots=True)
class _ResponsesResolution:
    """Outcome of native Responses planning with safe standard fallback."""

    provider_name: str
    actual_model: str
    copilot_provider: CopilotProvider | None
    support: CapabilitySupport
    plan: RoutePlan | None = None


def _invalid_request(message: str) -> JSONResponse:
    """Return an OpenAI-native invalid request response."""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


def _operation_for(stream: bool) -> Operation:
    return Operation.RESPONSES_STREAM if stream else Operation.RESPONSES


async def _resolve_responses_model(
    model: str,
    operation: Operation,
    features: RequestFeatures,
) -> _ResponsesResolution:
    """Plan a native Responses route, preserving a safe standard fallback."""
    try:
        plan = await get_router().plan_route(model, operation, features)
    except NoCompatibleRouteError:
        provider_name, actual_model, provider = await get_router()._resolve_provider(model)
        return _ResponsesResolution(
            provider_name=provider_name,
            actual_model=actual_model,
            copilot_provider=(provider if isinstance(provider, CopilotProvider) else None),
            support=CapabilitySupport.UNSUPPORTED,
        )

    candidate = plan.primary
    provider = candidate.provider
    return _ResponsesResolution(
        provider_name=candidate.model.provider,
        actual_model=candidate.model.upstream_id,
        copilot_provider=(provider if isinstance(provider, CopilotProvider) else None),
        support=candidate.support,
        plan=plan,
    )


def _strip_forbidden_fields(body: dict, copilot_provider: CopilotProvider, operation: Operation):
    """Remove top-level fields GHC's /responses rejects before forwarding."""
    forwardable = copilot_provider.outbound_contract.forwardable_fields(operation)
    if forwardable is None:
        return
    unknown = set(body.keys()) - forwardable
    if unknown:
        logger.debug("Stripping unknown fields before passthrough: %s", sorted(unknown))
        for key in unknown:
            del body[key]


def _strip_response(data: dict) -> dict:
    """Remove Copilot-internal fields from a non-streaming Responses payload."""
    data.pop("copilot_usage", None)
    return data


class _ResponsesUpstreamStatusError(ProviderError):
    """Typed native status failure retaining its response only for encoding."""

    def __init__(self, response, *, provider: str, model: str) -> None:
        status_code = response.status_code
        kind, retryable = classify_upstream_status(status_code)
        super().__init__(
            f"Native Responses upstream returned {status_code}",
            status_code=status_code,
            retryable=retryable,
            kind=kind,
            upstream_status_code=status_code,
            provider=provider,
            model=model,
        )
        self.response = response


class _ResponsesUnexpectedEOFError(ProviderError):
    """Native Responses transport ended cleanly before a terminal event."""


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


def _parse_native_response(response, *, provider: str, model: str) -> dict:
    """Parse and minimally validate a successful native Responses payload."""
    try:
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("native Responses payload must be an object")
        CopilotResponsesCodec.validate_usage(data.get("usage"))
        has_deliverable = CopilotResponsesCodec.validate_output(data)
        outcome = CopilotResponsesCodec.terminal_outcome(data)
        if (
            outcome.transport is TransportTermination.EXCEPTION
            and outcome.error is not None
            and outcome.error.code == "upstream_protocol_error"
        ):
            raise ValueError(outcome.error.message)
        if outcome.response_status is ResponseStatus.COMPLETED and not has_deliverable:
            raise ValueError("completed Responses response contains no deliverable output")
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ProviderError(
            "Native Responses upstream returned a malformed response",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            upstream_status_code=200,
            provider=provider,
            model=model,
            cause=error,
        ) from error
    return data


async def _send_native_nonstream(candidate: RouteCandidate, body: dict):
    """Send one native Responses attempt via the candidate's Copilot provider."""
    provider = candidate.provider
    if not isinstance(provider, CopilotProvider):
        raise ProviderError(
            "Provider does not support native Responses transport",
            status_code=501,
            retryable=False,
            kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
            provider=candidate.model.provider,
            model=candidate.model.upstream_id,
        )
    payload = dict(body)
    payload["model"] = candidate.model.upstream_id
    payload["stream"] = False
    await provider.ensure_token()
    response = await provider._send_with_auth_retry(
        "POST",
        COPILOT_RESPONSES_PATH,
        json=payload,
        headers_kwargs={
            "vision_request": provider._responses_input_has_vision(payload.get("input")),
            "response_input": payload.get("input"),
        },
        model=candidate.model.upstream_id,
    )
    if response.status_code >= 400:
        raise _ResponsesUpstreamStatusError(
            response,
            provider=candidate.model.provider,
            model=candidate.model.upstream_id,
        )
    data = _parse_native_response(
        response,
        provider=candidate.model.provider,
        model=candidate.model.upstream_id,
    )
    data["model"] = candidate.model.qualified_id
    return data


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def _stream_native_candidate(
    candidate: RouteCandidate,
    body: dict,
) -> AsyncGenerator[str, None]:
    """Open one candidate's native Responses stream and pump raw SSE frames."""
    provider = candidate.provider
    if not isinstance(provider, CopilotProvider):
        raise ProviderError(
            "Provider does not support native Responses transport",
            status_code=501,
            retryable=False,
            kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
            provider=candidate.model.provider,
            model=candidate.model.upstream_id,
        )
    payload = dict(body)
    payload["model"] = candidate.model.upstream_id
    payload["stream"] = True
    await provider.ensure_token()
    stream = _stream_passthrough(provider, payload, candidate)
    try:
        async for frame in stream:
            yield frame
    finally:
        await close_async_iterator(stream)


async def _stream_passthrough(
    provider: CopilotProvider,
    payload: dict,
    candidate: RouteCandidate,
) -> AsyncGenerator[str, None]:
    """Stream raw SSE frames from Copilot's native /responses endpoint."""
    async with provider._stream_with_auth_retry(
        COPILOT_RESPONSES_PATH,
        json=payload,
        headers_kwargs={
            "vision_request": provider._responses_input_has_vision(payload.get("input")),
            "response_input": payload.get("input"),
        },
        model=candidate.model.upstream_id,
    ) as response:
        if response.status_code >= 400:
            await response.aread()
            raise _ResponsesUpstreamStatusError(
                response,
                provider=candidate.model.provider,
                model=candidate.model.upstream_id,
            )
        async for frame in _iter_sse_frames(response):
            yield frame


async def _iter_sse_frames(response) -> AsyncGenerator[str, None]:
    """Iterate SSE frames, filtering Copilot-internal events."""
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
                if current_event not in _STRIP_STREAM_EVENTS:
                    if current_event in _TERMINAL_STREAM_EVENTS:
                        terminal_received = True
                    yield f"event: {current_event}\ndata: {data_buffer}\n\n"
            current_event = None
            data_buffer = ""

    if not terminal_received:
        _raise_native_stream_unexpected_eof(
            ValueError("upstream Responses SSE ended before a terminal event")
        )


def _raise_native_stream_unexpected_eof(cause: BaseException) -> None:
    """Raise the typed signal for a clean EOF before a native terminal event."""
    raise _ResponsesUnexpectedEOFError(
        "Native Responses upstream returned a malformed stream",
        status_code=502,
        retryable=True,
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        upstream_status_code=200,
        cause=cause,
    ) from cause


def _guard_chunk(event_type: str | None, data: dict | None, wire_frame: str):
    """Project one raw Responses frame onto the shared guard contract.

    Exposes the visible text/thinking/tool-argument content so the leak and
    runaway guards can inspect generation, while ``opaque_payload`` carries the
    exact wire bytes for the volume circuit-breaker.
    """
    from router_maestro.providers.base import ChatStreamChunk

    if data is None:
        return ChatStreamChunk(content="", opaque_payload=wire_frame)
    if event_type == "response.output_text.delta":
        delta = data.get("delta")
        return ChatStreamChunk(
            content=delta if isinstance(delta, str) else "",
            opaque_payload=wire_frame,
        )
    if event_type == "response.reasoning_summary_text.delta":
        delta = data.get("delta")
        return ChatStreamChunk(
            content="",
            thinking=delta if isinstance(delta, str) else "",
            opaque_payload=wire_frame,
        )
    if event_type == "response.function_call_arguments.delta":
        delta = data.get("delta")
        return ChatStreamChunk(
            content="",
            tool_calls=[{"function": {"arguments": delta if isinstance(delta, str) else ""}}],
            opaque_payload=wire_frame,
        )
    return ChatStreamChunk(content="", opaque_payload=wire_frame)


def _frame_terminal_outcome(event_type: str | None, data: dict | None) -> TerminalOutcome | None:
    """Map a native Responses terminal SSE event onto the shared outcome."""
    if event_type not in _TERMINAL_STREAM_EVENTS or not isinstance(data, dict):
        return None
    terminal_response = data.get("response")
    if not isinstance(terminal_response, dict):
        return None
    return CopilotResponsesCodec.terminal_outcome(terminal_response)


def _sse_error_event(error: Exception | str) -> str:
    """Format an OpenAI Responses SSE error event."""
    if isinstance(error, Exception):
        payload = postcommit_error_data(error, "openai_responses")
    else:
        payload = {"code": "server_error", "message": error}
    event = {"type": "error", **payload}
    return f"event: error\ndata: {json.dumps(event)}\n\n"


async def _encode_native_stream_errors(
    stream: AsyncIterator[str],
    *,
    pipeline=None,
) -> AsyncGenerator[str, None]:
    """Apply the shared guard chain and preserve native terminal semantics."""
    terminal_outcome: TerminalOutcome | None = None
    try:
        async for frame in stream:
            event_type, data = parse_sse_frame(frame)
            if pipeline is not None:
                abort_reason = pipeline.feed_stream(_guard_chunk(event_type, data, frame))
                if abort_reason is not None:
                    terminal_outcome = exception_outcome(abort_reason, code="overloaded")
                    pipeline.finish(
                        wire_status=200,
                        outcome=terminal_outcome,
                        body_summary=abort_reason,
                    )
                    yield _sse_error_event(
                        ProviderError(
                            "Overloaded: please retry this request",
                            status_code=529,
                            retryable=True,
                            kind=ProviderFailureKind.RATE_LIMIT,
                        )
                    )
                    return
            frame_outcome = _frame_terminal_outcome(event_type, data)
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
    except _ResponsesUnexpectedEOFError as error:
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
        logger.error("Unexpected error in beta Responses stream", exc_info=True)
        yield _sse_error_event("Internal server error")
    finally:
        await close_async_iterator(stream)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


async def _fallback_to_standard(body: dict) -> JSONResponse:
    """Delegate to the standard translated Responses handler."""
    try:
        request = ResponsesRequest.model_validate(body)
    except ValidationError as error:
        errors = error.errors(include_input=False, include_url=False)
        if not errors:
            return _invalid_request("Invalid request body")
        first = errors[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = str(first.get("msg", "Invalid value"))
        return _invalid_request(f"{location}: {message}" if location else message)
    return await create_response(request)


@router.post("/api/openai/beta/v1/responses")
async def beta_responses(raw_request: FastAPIRequest):
    """Handle Responses API requests via native passthrough or fallback."""
    body_bytes = await raw_request.body()
    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _invalid_request("Invalid JSON body")
    if not isinstance(body, dict):
        return _invalid_request("Request body must be a JSON object")

    model = body.get("model")
    if not model:
        return _invalid_request("'model' field is required")

    stream = bool(body.get("stream", False))
    operation = _operation_for(stream)
    features = RequestFeatures.for_responses_native(body)

    logger.info(
        "Received beta Responses request: model=%s, stream=%s, has_tools=%s",
        model,
        stream,
        bool(body.get("tools")),
    )

    try:
        resolution = await _resolve_responses_model(model, operation, features)
    except ProviderError as error:
        return protocol_error_response(error, "openai_responses")

    copilot_provider = resolution.copilot_provider
    uses_native = (
        copilot_provider is not None and resolution.support is not CapabilitySupport.UNSUPPORTED
    )

    if not uses_native:
        logger.info(
            "Beta Responses falling back to standard path: model=%s, provider=%s",
            model,
            resolution.provider_name,
        )
        return await _fallback_to_standard(body)

    _strip_forbidden_fields(body, copilot_provider, operation)
    body["model"] = resolution.actual_model
    plan = resolution.plan

    logger.info(
        "Beta Responses using native passthrough: model=%s -> %s, stream=%s",
        model,
        resolution.actual_model,
        stream,
    )

    if stream:
        from router_maestro.pipeline import RequestPipeline
        from router_maestro.runtime import get_current_request_context

        context = get_current_request_context()
        tool_names = {
            tool.get("name", "")
            for tool in body.get("tools", [])
            if isinstance(tool, dict) and tool.get("name")
        } or None
        pipeline = RequestPipeline.create(
            request_id=(context.request_id if context is not None else f"beta-{model}"),
            model=model,
            tool_names=tool_names,
        )
        try:
            selected_stream, _used_provider = await Router.execute_plan_stream(
                plan,
                lambda candidate: _stream_native_candidate(candidate, body),
                log_prefix="Beta Responses",
            )
        except ProviderError as error:
            logger.error(
                "beta_responses_stream_open_failed kind=%s retryable=%s",
                error.kind.value,
                str(error.retryable).lower(),
            )
            return protocol_error_response(error, "openai_responses")
        return sse_streaming_response(
            _encode_native_stream_errors(selected_stream, pipeline=pipeline),
        )

    try:
        data, _used_provider = await Router.execute_plan_nonstream(
            plan,
            lambda candidate: _send_native_nonstream(candidate, body),
        )
    except ProviderError as error:
        logger.error(
            "beta_responses_request_failed kind=%s retryable=%s",
            error.kind.value,
            str(error.retryable).lower(),
        )
        if isinstance(error, _ResponsesUpstreamStatusError):
            try:
                error_body = error.response.json()
            except Exception:
                error_body = {"error": {"type": "api_error", "message": error.response.text}}
            return JSONResponse(content=error_body, status_code=error.status_code)
        return protocol_error_response(error, "openai_responses")

    _strip_response(data)
    usage = data.get("usage") or {}
    logger.info(
        "Beta Responses passthrough response: model=%s, status=%s, "
        "output_items=%d, input_tokens=%s, output_tokens=%s",
        data.get("model"),
        data.get("status"),
        len(data.get("output", [])),
        usage.get("input_tokens"),
        usage.get("output_tokens"),
    )
    return JSONResponse(content=data)
