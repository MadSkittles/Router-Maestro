"""Anthropic Messages API compatible route."""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse

from router_maestro.providers import (
    AnthropicProvider,
    ChatRequest,
    ChatStreamChunk,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponseStatus,
    TerminalOutcome,
    client_cancelled_outcome,
    exception_outcome,
    finish_reason_for_outcome,
    resolve_terminal_outcome,
    unexpected_eof_outcome,
)
from router_maestro.routing import Router, get_router
from router_maestro.routing.capabilities import CapabilitySupport, Feature
from router_maestro.routing.model_ref import catalog_model_public_id, qualify_model_id
from router_maestro.routing.route_plan import RouteCandidate
from router_maestro.server.dependencies import get_app_router
from router_maestro.server.protocols import client_error_response
from router_maestro.server.protocols.anthropic_reducer import (
    AnthropicReducer,
    AnthropicStreamProtocolError,
    build_anthropic_response,
)
from router_maestro.server.protocols.errors import postcommit_error_data, protocol_error_response
from router_maestro.server.routes._outcomes import record_chat_response_outcome
from router_maestro.server.schemas.anthropic import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicModelInfo,
    AnthropicModelList,
    AnthropicTextBlock,
    AnthropicUsage,
)
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.server.translation import translate_anthropic_to_openai
from router_maestro.utils import count_anthropic_request_tokens, get_logger
from router_maestro.utils.async_iterators import close_async_iterator
from router_maestro.utils.context_window import resolve_thinking_budget
from router_maestro.utils.token_config import (
    count_tokens_via_anthropic_api,
    get_config_for_provider,
)

logger = get_logger("server.routes.anthropic")

router = APIRouter()


TEST_RESPONSE_TEXT = "This is a test response from Router-Maestro."

# Real Anthropic protocol ping event used as the SSE keepalive frame for
# this route. SSE comments (the shared default) don't reset Claude Code's
# "time since last protocol event" timer, which fires at ~15s and cancels
# the stream while a slow upstream (e.g. claude-opus-4.7-1m) is still
# producing its first content token.
ANTHROPIC_PING_FRAME = f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"


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


async def _apply_thinking_budget(
    model_router: Router,
    chat_request: ChatRequest,
    original_model: str,
    *,
    candidate: RouteCandidate | None = None,
) -> ChatRequest:
    """Resolve thinking budget from server config and return updated request.

    Uses chat_request.model (the translated/normalized name) for cache
    lookups since translate_anthropic_to_openai may strip date suffixes.
    Falls back to original_model for config key matching.

    Returns the original request unchanged if no adjustment is needed. Explicit
    reasoning effort replaces adaptive budgets. Manual enabled thinking retains
    a normalized budget because the Anthropic wire format requires one.
    """
    if chat_request.reasoning_effort is not None and chat_request.thinking_type == "adaptive":
        return chat_request.with_thinking(
            thinking_budget=None,
            thinking_type=chat_request.thinking_type,
        )
    if chat_request.reasoning_effort is not None and chat_request.thinking_type != "enabled":
        return chat_request
    if chat_request.thinking_type == "disabled":
        if chat_request.thinking_budget is None:
            return chat_request
        return chat_request.with_thinking(
            thinking_budget=None,
            thinking_type=chat_request.thinking_type,
        )

    from router_maestro.runtime import get_current_request_context

    context = get_current_request_context()
    if context is not None:
        priorities = context.config
    else:
        from router_maestro.config import load_priorities_config

        priorities = load_priorities_config()
    thinking_config = priorities.thinking

    translated_model = chat_request.model
    if candidate is not None:
        supports_thinking = (
            candidate.capabilities.feature(Feature.REASONING) is CapabilitySupport.SUPPORTED
        )
        max_output = candidate.capabilities.max_output_tokens or 16384
        config_model = candidate.model.qualified_id
    else:
        model_info = await model_router.get_model_info(translated_model)
        if model_info is None:
            model_info = await model_router.get_model_info(original_model)
        supports_thinking = model_info.supports_thinking if model_info else False
        max_output = (model_info.max_output_tokens or 16384) if model_info else 16384
        config_model = translated_model
    if chat_request.max_tokens is not None:
        max_output = min(max_output, chat_request.max_tokens)

    # Use translated model for config lookup (matches cache keys)
    budget, thinking_type = resolve_thinking_budget(
        client_budget=chat_request.thinking_budget,
        client_thinking_type=chat_request.thinking_type,
        model_id=config_model,
        max_output_tokens=max_output,
        thinking_config=thinking_config,
        supports_thinking=supports_thinking,
    )

    if budget != chat_request.thinking_budget or thinking_type != chat_request.thinking_type:
        return chat_request.with_thinking(thinking_budget=budget, thinking_type=thinking_type)

    return chat_request


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


async def _stream_test_response() -> AsyncGenerator[str]:
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


# Anthropic beta header that lets clients append ``{"role": "system", ...}``
# entries directly into ``messages[]`` mid-conversation, preserving prompt
# cache locality instead of rebuilding the top-level ``system`` field.
# Claude Code 2.1.x defaults this beta to ON for ``claude-opus-4-8``.
_MID_CONV_SYSTEM_BETA = "mid-conversation-system-2026-04-07"


def _has_mid_conv_system_beta(raw_request: FastAPIRequest) -> bool:
    """Check if the request opts into the mid-conversation-system beta."""
    header = raw_request.headers.get("anthropic-beta", "")
    return _MID_CONV_SYSTEM_BETA in header.lower()


async def _raw_body_has_inline_system(raw_request: FastAPIRequest) -> bool:
    """Return True iff the original request body had ``role="system"`` in messages.

    The Pydantic ``before`` validator hoists inline system messages out of the
    array so the rest of the schema can validate, which means by the time the
    route runs we've lost the original shape. Re-reading the raw body lets us
    tell whether Claude Code actually sent the mid-conversation-system payload
    so we can return the 400 it expects, vs. a non-Claude-Code client where
    the silent hoist is the right behavior.

    Body bytes are cached by Starlette, so this is a no-op second read.
    """
    try:
        body = await raw_request.body()
        if not body:
            return False
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    return any(isinstance(m, dict) and m.get("role") == "system" for m in messages)


async def _raw_body_thinking(raw_request: FastAPIRequest) -> dict | None:
    """Return the original request ``thinking`` object, if it was a JSON object."""
    try:
        body = await raw_request.body()
        if not body:
            return None
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    thinking = payload.get("thinking")
    return thinking if isinstance(thinking, dict) else None


def _mid_conv_system_rejection_response() -> JSONResponse:
    """Return a 400 that triggers Claude Code's mid-conv-system fallback.

    Claude Code's ``Mk8`` error classifier (Claude Code 2.1.x) accepts this
    request as a beta-not-supported signal when the response message
    contains both the beta header string and ``"anthropic-beta"``. Hitting
    that path makes Claude Code:

    1. Strip the ``mid-conversation-system-2026-04-07`` header for the rest
       of the session (sticky until ``/clear`` or ``/compact``).
    2. Rewrite the inline system messages into ``<system-reminder>`` blocks
       inside ``user`` messages, preserving mid-conversation position so the
       prompt cache stays warm.
    3. Retry transparently — the user sees no error.

    Returning the hoisted payload with a 200 instead would silently work but
    waste cache: Claude Code would keep sending the beta every turn, and we
    would keep flattening it into a top-level ``system`` rewrite that
    invalidates the cache prefix on every message.
    """
    message = (
        f"This model does not support the `anthropic-beta: {_MID_CONV_SYSTEM_BETA}` "
        'header through Router-Maestro. Inline `role: "system"` messages were '
        "rejected; the client should drop the beta header and fall back to "
        "rebuilding the top-level system prompt or to <system-reminder> blocks."
    )
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


@router.post("/v1/messages")
@router.post("/api/anthropic/v1/messages")
async def messages(request: AnthropicMessagesRequest, raw_request: FastAPIRequest):
    """Handle Anthropic Messages API requests."""
    parsed_thinking = request.thinking.model_dump(exclude_none=True) if request.thinking else None
    effort = request.output_config.effort if request.output_config else None
    raw_thinking = await _raw_body_thinking(raw_request)
    logger.info(
        "Received Anthropic messages request: model=%s, stream=%s, max_tokens=%s, "
        "thinking=%s, raw_thinking=%s, effort=%s",
        request.model,
        request.stream,
        request.max_tokens,
        parsed_thinking,
        raw_thinking,
        effort,
    )

    # Handle test model
    if request.model == "test":
        if request.stream:
            return sse_streaming_response(
                _stream_test_response(),
                keepalive_frame=ANTHROPIC_PING_FRAME,
            )
        return _create_test_response()

    # Claude Code 2.1.x enables `mid-conversation-system-2026-04-07` by
    # default for opus-4-8 and sends `role:"system"` blocks inline in
    # `messages`. The schema validator silently hoists them so generic
    # OpenAI-shaped clients (Cline, Aider, …) still work, but Claude Code
    # itself has a richer fallback path that preserves prompt-cache
    # locality. When we see the beta header AND the original payload had
    # inline system messages, return a 400 that matches Claude Code's
    # `Mk8` detector so it strips the beta and retries with
    # <system-reminder> blocks. See binary analysis in PR description.
    if _has_mid_conv_system_beta(raw_request) and await _raw_body_has_inline_system(raw_request):
        logger.info(
            "Rejecting mid-conversation-system beta for model=%s so the "
            "client falls back to <system-reminder> blocks (preserves "
            "prompt cache)",
            request.model,
        )
        return _mid_conv_system_rejection_response()

    model_router = get_router()

    # Translate Anthropic request to OpenAI format
    chat_request = translate_anthropic_to_openai(request)

    try:
        route_plan = await model_router.plan_chat_completion(
            chat_request,
            stream=request.stream,
        )
        candidate_requests = {
            candidate.model: await _apply_thinking_budget(
                model_router,
                chat_request,
                request.model,
                candidate=candidate,
            )
            for candidate in (route_plan.primary, *route_plan.prevalidation_fallbacks)
        }
        chat_request = candidate_requests[route_plan.primary.model]
        prepared_plan = model_router.prepare_planned_chat_completion(
            route_plan,
            chat_request,
            candidate_requests=candidate_requests,
        )
    except ProviderError as e:
        if isinstance(e, RequestOptionError):
            return client_error_response(e, "anthropic")
        return protocol_error_response(e, "anthropic")

    if request.stream:
        return sse_streaming_response(
            stream_response(
                model_router,
                chat_request,
                request.model,
                token_estimation_request=request,
                prepared_plan=prepared_plan,
            ),
            keepalive_frame=ANTHROPIC_PING_FRAME,
        )

    try:
        response, provider_name = await model_router.chat_completion(
            chat_request,
            prepared_plan=prepared_plan,
        )
        response_model = (
            response.selected_model.qualified_id
            if response.selected_model is not None
            else qualify_model_id(provider_name, response.model)
        )
        downstream_response = build_anthropic_response(
            response,
            response_id=f"msg_{uuid.uuid4().hex[:24]}",
            model=response_model,
        )
        record_chat_response_outcome(response)

        logger.info(
            "Upstream response from %s: content_len=%s, tool_calls=%s, finish_reason=%s",
            provider_name,
            len(response.content) if response.content else 0,
            len(response.tool_calls) if response.tool_calls else 0,
            response.finish_reason,
        )
        return downstream_response
    except ProviderError as e:
        logger.error(
            "Anthropic messages request failed: model=%s, status=%s, error=%s",
            request.model,
            e.status_code,
            e,
        )
        if isinstance(e, RequestOptionError):
            return client_error_response(e, "anthropic")
        return protocol_error_response(e, "anthropic")


@router.post("/v1/messages/count_tokens")
@router.post("/api/anthropic/v1/messages/count_tokens")
async def count_tokens(request: AnthropicCountTokensRequest):
    """Count tokens for a messages request.

    Resolves which provider will handle the model and selects the appropriate
    token-counting configuration. For native Anthropic, attempts an upstream
    API call for an exact count before falling back to local estimation.
    """
    logger.debug("count_tokens request: model=%s, provider=%s", request.model, None)
    provider_name = await _resolve_provider_name(request.model)
    logger.debug("count_tokens resolved provider=%s for model=%s", provider_name, request.model)
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
                    "Anthropic upstream count_tokens failed for model=%s, "
                    "falling back to local estimation",
                    request.model,
                    exc_info=True,
                )

    input_tokens = count_anthropic_request_tokens(
        system=request.system,
        messages=request.messages,
        tools=request.tools,
        model=request.model,
        config=config,
    )
    logger.debug(
        "count_tokens result: model=%s, provider=%s, input_tokens=%d",
        request.model,
        provider_name,
        input_tokens,
    )
    return {"input_tokens": input_tokens}


def _estimate_input_tokens(
    request: AnthropicMessagesRequest,
    provider_name: str | None = None,
) -> int:
    """Estimate input tokens from request content.

    Uses the centralized token estimation function with provider-aware
    configuration for accurate estimates.
    """
    config = get_config_for_provider(provider_name)
    return count_anthropic_request_tokens(
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
    *,
    token_estimation_request: AnthropicMessagesRequest | None = None,
    prepared_plan=None,
) -> AsyncGenerator[str]:
    """Stream Anthropic Messages API response."""
    response_id = f"msg_{uuid.uuid4().hex[:24]}"
    reducer: AnthropicReducer | None = None

    # Emit a protocol ping before opening the upstream stream. The shared SSE
    # wrapper continues emitting pings while Router primes the first chunk, so
    # slow models do not hit Claude Code's first-byte timeout. ``message_start``
    # waits until Router has selected the actual provider/model after fallback.
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    pipeline = None
    stream = None
    terminal_outcome: TerminalOutcome | None = None
    try:
        if prepared_plan is None:
            stream, provider_name = await model_router.chat_completion_stream(request)
        else:
            stream, provider_name = await model_router.chat_completion_stream(
                request,
                prepared_plan=prepared_plan,
            )
        selected_model = getattr(stream, "selected_model", None)
        if token_estimation_request is not None:
            selected_provider = (
                selected_model.provider if selected_model is not None else provider_name
            )
            estimated_input_tokens = _estimate_input_tokens(
                token_estimation_request,
                selected_provider,
            )
        response_model = (
            selected_model.qualified_id
            if selected_model is not None
            else qualify_model_id(provider_name, original_model)
        )
        reducer = AnthropicReducer(
            response_id=response_id,
            model=response_model,
            estimated_input_tokens=estimated_input_tokens,
        )
        for start_event in reducer.start():
            yield f"event: {start_event['type']}\ndata: {json.dumps(start_event)}\n\n"

        # Track state needed to recover tool calls that the upstream model
        # leaked as <invoke> XML *text* instead of structured tool_calls (a
        # github-copilot/claude-* quirk under long contexts). Without recovery
        # the client (e.g. Claude Code) receives the raw XML in a text block
        # with stop_reason=end_turn, shows it verbatim, and the agent stalls
        # because the intended tool never runs.
        allowed_tool_names: set[str] | None = None
        if request.tools:
            allowed_tool_names = {
                (t.get("function") or {}).get("name", "")
                for t in request.tools
                if (t.get("function") or {}).get("name")
            } or None
        saw_real_tool_call = False

        # Unified pipeline: guards + audit in one object
        from router_maestro.pipeline import RequestPipeline

        pipeline = RequestPipeline.create(
            request_id=response_id,
            model=request.model,
            tool_names=allowed_tool_names,
        )

        async for chunk in stream:
            # Feed through guards (leak detection + runaway)
            abort_reason = pipeline.feed_stream(chunk)
            if abort_reason:
                terminal_outcome = exception_outcome(abort_reason, code="overloaded")
                logger.warning("Stream aborted: %s", abort_reason)
                yield _sse_error_event(
                    ProviderError(
                        "Overloaded: please retry this request",
                        status_code=529,
                        kind=ProviderFailureKind.RATE_LIMIT,
                    )
                )
                pipeline.finish(
                    wire_status=200,
                    outcome=terminal_outcome,
                    body_summary=abort_reason,
                )
                return

            if chunk.tool_calls:
                saw_real_tool_call = True

            chunk_outcome = resolve_terminal_outcome(
                chunk.terminal_outcome,
                chunk.finish_reason,
            )
            if chunk_outcome is not None:
                terminal_outcome = chunk_outcome
            recovered_tool_calls = chunk.tool_calls
            finish_reason = (
                finish_reason_for_outcome(chunk_outcome)
                if chunk_outcome is not None
                else chunk.finish_reason
            )
            if finish_reason and not saw_real_tool_call:
                leaked = pipeline.check_invoke_at_finish()
                if leaked:
                    recovered_tool_calls = leaked
                    finish_reason = "tool_calls"
                    if (
                        chunk_outcome is not None
                        and chunk_outcome.response_status is ResponseStatus.COMPLETED
                    ):
                        chunk_outcome = replace(
                            chunk_outcome,
                            finish_reason="tool_calls",
                        )
                        terminal_outcome = chunk_outcome

            if chunk_outcome is not None and chunk_outcome.response_status not in {
                ResponseStatus.COMPLETED,
                ResponseStatus.INCOMPLETE,
            }:
                error = chunk_outcome.error
                message = error.message if error is not None else "Upstream response failed"
                yield _sse_error_event(
                    ProviderError(message, status_code=502, kind=ProviderFailureKind.UNKNOWN)
                )
                pipeline.finish(
                    wire_status=200,
                    outcome=chunk_outcome,
                    body_summary=message,
                )
                return

            try:
                events = reducer.reduce(
                    ChatStreamChunk(
                        content=chunk.content,
                        refusal=chunk.refusal,
                        finish_reason=finish_reason,
                        usage=chunk.usage,
                        tool_calls=recovered_tool_calls,
                        thinking=chunk.thinking,
                        thinking_signature=chunk.thinking_signature,
                        thinking_id=chunk.thinking_id,
                        terminal_outcome=chunk_outcome,
                    )
                )
            except AnthropicStreamProtocolError:
                terminal_outcome = exception_outcome(
                    "Invalid tool call from upstream",
                    code="upstream_protocol_error",
                )
                logger.warning("Invalid Anthropic tool stream from upstream", exc_info=True)
                yield _sse_error_event(
                    ProviderError(
                        "Invalid tool call from upstream",
                        status_code=502,
                        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                    )
                )
                pipeline.finish(
                    wire_status=200,
                    outcome=terminal_outcome,
                    body_summary="Invalid tool call from upstream",
                )
                return

            for event in events:
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

            if chunk_outcome is not None:
                pipeline.finish(wire_status=200, outcome=chunk_outcome)
                return

        terminal_outcome = unexpected_eof_outcome()
        yield _sse_error_event(
            ProviderError(
                terminal_outcome.error.code,
                status_code=502,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        )
        pipeline.finish(
            wire_status=200,
            outcome=terminal_outcome,
            body_summary=terminal_outcome.error.message,
        )

    except ProviderError as e:
        terminal_outcome = exception_outcome(str(e), code="provider_error")
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary=str(e),
            )
        yield _sse_error_event(e)
    except asyncio.CancelledError:
        if pipeline is not None:
            pipeline.finish(wire_status=200, outcome=client_cancelled_outcome())
        logger.info("Anthropic stream cancelled by client")
        raise
    except Exception:
        terminal_outcome = exception_outcome("Internal server error", code="server_error")
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary="Internal server error",
            )
        logger.error("Unexpected error in Anthropic stream", exc_info=True)
        yield _sse_error_event(RuntimeError("Internal server error"))
    finally:
        await close_async_iterator(stream)


def _sse_error_event(error: Exception) -> str:
    """Format an Anthropic SSE error event."""
    error_event = postcommit_error_data(error, "anthropic")
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
    model_router: Router = Depends(get_app_router),
) -> AnthropicModelList:
    """List available models in Anthropic format.

    Args:
        limit: Maximum number of models to return (default 20)
        after_id: Return models after this ID (for forward pagination)
        before_id: Return models before this ID (for backward pagination)
    """
    models = await model_router.list_models()

    # Generate ISO 8601 timestamp for created_at
    # Using current time since actual creation dates aren't tracked
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Convert to Anthropic format
    anthropic_models = [
        AnthropicModelInfo(
            id=catalog_model_public_id(
                model.provider,
                model.id,
                id_is_qualified=model.id_is_qualified,
            ),
            created_at=created_at,
            display_name=_generate_display_name(
                catalog_model_public_id(
                    model.provider,
                    model.id,
                    id_is_qualified=model.id_is_qualified,
                )
            ),
            type="model",
            max_prompt_tokens=model.max_prompt_tokens,
            max_output_tokens=model.max_output_tokens,
            max_context_window_tokens=model.max_context_window_tokens,
            supports_thinking=model.supports_thinking or None,
            supports_vision=model.supports_vision or None,
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
