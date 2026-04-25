"""Gemini API-compatible routes."""

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException

from router_maestro.providers import ChatRequest, ProviderError
from router_maestro.routing import get_router
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
from router_maestro.utils.responses_bridge import (
    is_experimental_responses_enabled,
)

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


def _maybe_enable_responses_api(request: ChatRequest, model: str) -> ChatRequest:
    """Opt this ChatRequest into Copilot /responses when the env flag is on.

    The Copilot provider is the authoritative gate
    (``should_use_responses_for_chat``): it checks the resolved provider name
    and model eligibility and falls through to ``/chat/completions`` when
    either fails. Setting the flag here is a no-op for non-Copilot or
    non-GPT-5 backends — we don't pre-filter on the raw path model so
    aliases/fuzzy matches resolve through the router instead of the entry
    route.
    """
    if not is_experimental_responses_enabled():
        return request
    return ChatRequest(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        thinking_budget=request.thinking_budget,
        thinking_type=request.thinking_type,
        reasoning_effort=request.reasoning_effort,
        use_responses_api=True,
        extra=request.extra,
    )


# ============================================================================
# POST /api/gemini/v1beta/models/{model}:generateContent
# ============================================================================


@router.post("/api/gemini/v1beta/models/{model_method}:generateContent")
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
    chat_request = _maybe_enable_responses_api(chat_request, model)

    try:
        response, _provider_name = await model_router.chat_completion(chat_request)
        estimated_tokens = _estimate_input_tokens(request)
        return translate_openai_to_gemini(
            response, model, input_tokens=estimated_tokens
        ).model_dump(by_alias=True, exclude_none=True)
    except ProviderError as e:
        logger.error("Gemini generateContent request failed: %s", e)
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": {
                    "code": e.status_code,
                    "message": str(e),
                    "status": "INTERNAL",
                }
            },
        )


# ============================================================================
# POST /api/gemini/v1beta/models/{model}:streamGenerateContent
# ============================================================================


@router.post("/api/gemini/v1beta/models/{model_method}:streamGenerateContent")
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
    # metadata and the experimental flag survive into the provider call.
    chat_request = ChatRequest(
        model=chat_request.model,
        messages=chat_request.messages,
        temperature=chat_request.temperature,
        max_tokens=chat_request.max_tokens,
        stream=True,
        tools=chat_request.tools,
        tool_choice=chat_request.tool_choice,
        thinking_budget=chat_request.thinking_budget,
        thinking_type=chat_request.thinking_type,
        reasoning_effort=chat_request.reasoning_effort,
        use_responses_api=chat_request.use_responses_api,
        extra=chat_request.extra,
    )
    chat_request = _maybe_enable_responses_api(chat_request, model)

    estimated_tokens = _estimate_input_tokens(request)
    return sse_streaming_response(
        _stream_response(model_router, chat_request, model, estimated_tokens),
    )


# ============================================================================
# POST /api/gemini/v1beta/models/{model}:countTokens
# ============================================================================


@router.post("/api/gemini/v1beta/models/{model_method}:countTokens")
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
) -> AsyncGenerator[str, None]:
    """Stream Gemini-format SSE response from the upstream provider."""
    try:
        stream, _provider_name = await model_router.chat_completion_stream(request)

        state = GeminiStreamState(estimated_input_tokens=estimated_input_tokens)

        async for chunk in stream:
            openai_chunk = {
                "choices": [
                    {
                        "delta": {
                            "content": (chunk.content if chunk.content else None),
                            "tool_calls": chunk.tool_calls,
                        },
                        "finish_reason": chunk.finish_reason,
                    }
                ],
                "usage": chunk.usage,
            }

            gemini_event = translate_openai_chunk_to_gemini(openai_chunk, state, original_model)
            if gemini_event is not None:
                yield (
                    "data: "
                    f"{gemini_event.model_dump_json(exclude_none=True, by_alias=True)}"
                    "\r\n\r\n"
                )

    except ProviderError as e:
        yield _sse_error_data(str(e))
    except asyncio.CancelledError:
        logger.info("Gemini stream cancelled by client")
        raise
    except Exception:
        logger.error("Unexpected error in Gemini stream", exc_info=True)
        yield _sse_error_data("Internal server error")


def _sse_error_data(message: str) -> str:
    """Format a Gemini SSE error event."""
    error_response = GeminiGenerateContentResponse(
        candidates=[
            GeminiCandidate(
                content=GeminiContent(
                    parts=[GeminiPart(text=message)],
                    role="model",
                ),
                finish_reason="OTHER",
                index=0,
            )
        ],
    )
    return f"data: {error_response.model_dump_json(exclude_none=True, by_alias=True)}\r\n\r\n"
