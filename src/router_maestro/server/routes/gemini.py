"""Gemini API-compatible routes."""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import replace

from fastapi import APIRouter

from router_maestro.providers import (
    ChatRequest,
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
from router_maestro.routing import get_router
from router_maestro.routing.model_ref import qualify_model_id
from router_maestro.server.protocols import client_error_response
from router_maestro.server.protocols.errors import postcommit_error_data, protocol_error_response
from router_maestro.server.routes._outcomes import record_chat_response_outcome
from router_maestro.server.schemas.gemini import (
    GeminiCandidate,
    GeminiContent,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiPart,
    GeminiStreamState,
    GeminiUsageMetadata,
)
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.server.translation_gemini import (
    translate_gemini_to_openai,
    translate_openai_chunk_to_gemini,
    translate_openai_to_gemini,
)
from router_maestro.utils import get_logger
from router_maestro.utils.async_iterators import close_async_iterator

logger = get_logger("server.routes.gemini")

router = APIRouter()


TEST_RESPONSE_TEXT = "This is a test response from Router-Maestro."


def _extract_model_from_path(model_with_method: str) -> str:
    """Extract model name from the path parameter ``{model}:method``."""
    if ":" in model_with_method:
        return model_with_method.split(":")[0]
    return model_with_method


def _estimate_input_tokens(request: GeminiGenerateContentRequest) -> int:
    """Rough token estimate based on serialised request size.

    Uses ~4 chars per token as a simple heuristic, consistent with the
    approach in the TypeScript reference implementation.
    """
    try:
        text = request.model_dump_json(exclude_none=True)
        return max(1, len(text) // 4)
    except Exception:
        return 0


# ============================================================================
# POST /api/gemini/v1beta/models/{model}:generateContent
# ============================================================================


@router.post("/api/gemini/v1beta/models/{model_method:path}:generateContent")
async def generate_content(
    model_method: str,
    request: GeminiGenerateContentRequest,
):
    """Handle Gemini generateContent (non-streaming) requests."""
    model = _extract_model_from_path(model_method)
    logger.info("Received Gemini generateContent request: model=%s", model)

    # Handle test model
    if model == "test":
        return _create_test_response(model).model_dump(by_alias=True, exclude_none=True)

    model_router = get_router()
    chat_request = translate_gemini_to_openai(request, model)

    try:
        response, provider_name = await model_router.chat_completion(chat_request)
        estimated_tokens = _estimate_input_tokens(request)
        downstream_response = translate_openai_to_gemini(
            response,
            (
                response.selected_model.qualified_id
                if response.selected_model is not None
                else qualify_model_id(provider_name, response.model)
            ),
            input_tokens=estimated_tokens,
        ).model_dump(by_alias=True, exclude_none=True)
        record_chat_response_outcome(response)
        return downstream_response
    except ProviderError as e:
        logger.error("Gemini generateContent request failed: %s", e)
        if isinstance(e, RequestOptionError):
            return client_error_response(e, "gemini")
        return protocol_error_response(e, "gemini")


# ============================================================================
# POST /api/gemini/v1beta/models/{model}:streamGenerateContent
# ============================================================================


@router.post("/api/gemini/v1beta/models/{model_method:path}:streamGenerateContent")
async def stream_generate_content(
    model_method: str,
    request: GeminiGenerateContentRequest,
):
    """Handle Gemini streamGenerateContent (streaming) requests."""
    model = _extract_model_from_path(model_method)
    logger.info("Received Gemini streamGenerateContent request: model=%s", model)

    # Handle test model
    if model == "test":
        return sse_streaming_response(_stream_test_response(model))

    model_router = get_router()
    # logger.debug("Raw Gemini request: %s", request.model_dump_json(exclude_none=True))
    chat_request = translate_gemini_to_openai(request, model)
    # logger.debug(
    #     "Translated ChatRequest: model=%s, messages=%d, tools=%s, max_tokens=%s",
    #     chat_request.model,
    #     len(chat_request.messages),
    #     len(chat_request.tools) if chat_request.tools else 0,
    #     chat_request.max_tokens,
    # )
    # Enable streaming, preserving every other ChatRequest field so reasoning
    # metadata survives into the provider call.
    chat_request = replace(chat_request, stream=True, extra={})

    estimated_tokens = _estimate_input_tokens(request)
    stream = None
    try:
        prepared_plan = await model_router.prepare_chat_completion_stream(chat_request)
        stream, provider_name = await model_router.chat_completion_stream(
            chat_request,
            prepared_plan=prepared_plan,
        )
    except ProviderError as e:
        if isinstance(e, RequestOptionError):
            return client_error_response(e, "gemini")
        return protocol_error_response(e, "gemini")
    try:
        return sse_streaming_response(
            _stream_response(
                model_router,
                chat_request,
                model,
                estimated_tokens,
                opened_stream=stream,
                opened_provider_name=provider_name,
            ),
        )
    except Exception:
        await close_async_iterator(stream)
        raise


# ============================================================================
# POST /api/gemini/v1beta/models/{model}:countTokens
# ============================================================================


@router.post("/api/gemini/v1beta/models/{model_method:path}:countTokens")
async def count_tokens(
    model_method: str,
    request: GeminiGenerateContentRequest,
):
    """Handle Gemini countTokens requests."""
    model = _extract_model_from_path(model_method)
    logger.info("Received Gemini countTokens request: model=%s", model)

    total_tokens = _estimate_input_tokens(request)
    return {"totalTokens": total_tokens}


# ============================================================================
# Test helpers
# ============================================================================


def _create_test_response(model: str) -> GeminiGenerateContentResponse:
    """Create a mock non-streaming test response."""
    return GeminiGenerateContentResponse(
        candidates=[
            GeminiCandidate(
                content=GeminiContent(
                    parts=[GeminiPart(text=TEST_RESPONSE_TEXT)],
                    role="model",
                ),
                finish_reason="STOP",
                index=0,
            )
        ],
        usage_metadata=GeminiUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=10,
            total_token_count=20,
        ),
        model_version=model,
    )


async def _stream_test_response(
    model: str,
) -> AsyncGenerator[str, None]:
    """Stream a mock test response in Gemini SSE format."""
    # Content chunk
    content_chunk = GeminiGenerateContentResponse(
        candidates=[
            GeminiCandidate(
                content=GeminiContent(
                    parts=[GeminiPart(text=TEST_RESPONSE_TEXT)],
                    role="model",
                ),
                index=0,
            )
        ],
    )
    yield (f"data: {content_chunk.model_dump_json(exclude_none=True, by_alias=True)}\r\n\r\n")

    # Final chunk with finish reason and usage
    final_chunk = GeminiGenerateContentResponse(
        candidates=[
            GeminiCandidate(
                finish_reason="STOP",
                index=0,
            )
        ],
        usage_metadata=GeminiUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=10,
            total_token_count=20,
        ),
        model_version=model,
    )
    yield (f"data: {final_chunk.model_dump_json(exclude_none=True, by_alias=True)}\r\n\r\n")


# ============================================================================
# Streaming
# ============================================================================


async def _stream_response(
    model_router,
    request: ChatRequest,
    original_model: str,
    estimated_input_tokens: int = 0,
    *,
    prepared_plan=None,
    opened_stream=None,
    opened_provider_name: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream Gemini-format SSE response from the upstream provider."""
    pipeline = None
    stream = opened_stream
    terminal_outcome: TerminalOutcome | None = None
    try:
        if stream is not None:
            provider_name = opened_provider_name
        elif prepared_plan is None:
            stream, provider_name = await model_router.chat_completion_stream(request)
        else:
            stream, provider_name = await model_router.chat_completion_stream(
                request,
                prepared_plan=prepared_plan,
            )
        if provider_name is None:
            raise RuntimeError("Opened Gemini stream is missing provider identity")
        selected_model = getattr(stream, "selected_model", None)
        response_model = (
            selected_model.qualified_id
            if selected_model is not None
            else qualify_model_id(provider_name, original_model)
        )

        # Unified pipeline: guards + audit
        from router_maestro.pipeline import RequestPipeline

        pipeline = RequestPipeline.create(
            request_id=f"gemini-{original_model}",
            model=request.model,
        )

        state = GeminiStreamState(estimated_input_tokens=estimated_input_tokens)

        async for chunk in stream:
            abort_reason = pipeline.feed_stream(chunk)
            if abort_reason:
                terminal_outcome = exception_outcome(abort_reason, code="overloaded")
                logger.warning("Gemini stream aborted: %s", abort_reason)
                yield _sse_error_data(
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

            chunk_outcome = resolve_terminal_outcome(
                chunk.terminal_outcome,
                chunk.finish_reason,
            )
            if chunk_outcome is not None:
                terminal_outcome = chunk_outcome
            if chunk_outcome is not None and chunk_outcome.response_status not in {
                ResponseStatus.COMPLETED,
                ResponseStatus.INCOMPLETE,
            }:
                error = chunk_outcome.error
                message = error.message if error is not None else "Upstream response failed"
                yield _sse_error_data(
                    ProviderError(message, status_code=502, kind=ProviderFailureKind.UNKNOWN)
                )
                pipeline.finish(
                    wire_status=200,
                    outcome=chunk_outcome,
                    body_summary=message,
                )
                return

            openai_chunk = {
                "choices": [
                    {
                        "delta": {
                            "content": (chunk.content or chunk.refusal or None),
                            "tool_calls": chunk.tool_calls,
                        },
                        "finish_reason": (
                            finish_reason_for_outcome(chunk_outcome)
                            if chunk_outcome is not None
                            else chunk.finish_reason
                        ),
                    }
                ],
                "usage": chunk.usage,
            }

            gemini_event = translate_openai_chunk_to_gemini(openai_chunk, state, response_model)
            if gemini_event is not None:
                yield (
                    "data: "
                    f"{gemini_event.model_dump_json(exclude_none=True, by_alias=True)}"
                    "\r\n\r\n"
                )

            if chunk_outcome is not None:
                pipeline.finish(wire_status=200, outcome=chunk_outcome)
                return

        terminal_outcome = unexpected_eof_outcome()
        yield _sse_error_data(
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
        yield _sse_error_data(e)
    except asyncio.CancelledError:
        if pipeline is not None:
            pipeline.finish(wire_status=200, outcome=client_cancelled_outcome())
        logger.info("Gemini stream cancelled by client")
        raise
    except Exception:
        terminal_outcome = exception_outcome("Internal server error", code="server_error")
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary="Internal server error",
            )
        logger.error("Unexpected error in Gemini stream", exc_info=True)
        yield _sse_error_data(RuntimeError("Internal server error"))
    finally:
        await close_async_iterator(stream)


def _sse_error_data(error: Exception) -> str:
    """Format a Gemini SSE error event."""
    return f"data: {json.dumps(postcommit_error_data(error, 'gemini'))}\r\n\r\n"
