"""Chat completions route."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException

from router_maestro.providers import (
    ChatRequest,
    Message,
    ProviderError,
    ResponseStatus,
    TerminalOutcome,
    client_cancelled_outcome,
    exception_outcome,
    finish_reason_for_outcome,
    resolve_terminal_outcome,
    unexpected_eof_outcome,
)
from router_maestro.routing import Router, get_router
from router_maestro.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ChatMessageToolCall,
)
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.utils import get_logger
from router_maestro.utils.async_iterators import close_async_iterator
from router_maestro.utils.reasoning import VALID_EFFORTS, effort_to_budget

logger = get_logger("server.routes.chat")

router = APIRouter()


def make_chat_usage(raw_usage: dict | None) -> ChatCompletionUsage | None:
    """Create OpenAI chat usage while preserving upstream detail fields."""
    if not raw_usage:
        return None

    return ChatCompletionUsage(
        prompt_tokens=raw_usage.get("prompt_tokens", 0),
        completion_tokens=raw_usage.get("completion_tokens", 0),
        total_tokens=raw_usage.get("total_tokens", 0),
        prompt_tokens_details=raw_usage.get("prompt_tokens_details"),
        completion_tokens_details=raw_usage.get("completion_tokens_details"),
    )


@router.post("/api/openai/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    logger.info(
        "Received chat completion request: model=%s, stream=%s",
        request.model,
        request.stream,
    )
    model_router = get_router()

    # Convert to internal format
    messages = []
    for m in request.messages:
        tool_calls_raw = None
        if m.tool_calls:
            tool_calls_raw = [tc.model_dump() for tc in m.tool_calls]
        messages.append(
            Message(
                role=m.role,
                content=m.content,
                tool_call_id=m.tool_call_id,
                tool_calls=tool_calls_raw,
            )
        )

    extra = {
        key: value
        for key, value in {
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop,
            "user": request.user,
        }.items()
        if value is not None
    }

    chat_request = ChatRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        extra=extra,
    )

    # Reasoning / thinking passthrough.
    # Prefer OpenAI-style ``reasoning_effort``; also accept Anthropic-style
    # ``thinking`` for SDKs that forward it via the OpenAI endpoint.
    effort = (request.reasoning_effort or "").lower() or None
    if effort and effort in VALID_EFFORTS:
        chat_request.reasoning_effort = effort
        chat_request.thinking_budget = effort_to_budget(effort)
        chat_request.thinking_type = "enabled"
    elif request.thinking:
        t_type = request.thinking.get("type")
        t_budget = request.thinking.get("budget_tokens")
        if t_type:
            chat_request.thinking_type = t_type
        if isinstance(t_budget, int):
            chat_request.thinking_budget = t_budget

    if request.stream:
        return sse_streaming_response(stream_response(model_router, chat_request))

    try:
        response, provider_name = await model_router.chat_completion(chat_request)

        usage = make_chat_usage(response.usage)

        # Build response message with optional tool_calls
        response_tool_calls = None
        if response.tool_calls:
            response_tool_calls = [ChatMessageToolCall(**tc) for tc in response.tool_calls]

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=response_tool_calls,
                    ),
                    finish_reason=response.finish_reason,
                )
            ],
            usage=usage,
        )
    except ProviderError as e:
        logger.error("Chat completion request failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=str(e))


async def stream_response(model_router: Router, request: ChatRequest) -> AsyncGenerator[str, None]:
    """Stream chat completion response."""
    pipeline = None
    stream = None
    terminal_outcome: TerminalOutcome | None = None
    terminal_usage: ChatCompletionUsage | None = None
    try:
        stream, provider_name = await model_router.chat_completion_stream(request)
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Unified pipeline: guards + audit
        from router_maestro.pipeline import RequestPipeline

        tool_names: set[str] | None = None
        if request.tools:
            tool_names = {
                (t.get("function") or {}).get("name", "")
                for t in request.tools
                if (t.get("function") or {}).get("name")
            } or None
        pipeline = RequestPipeline.create(
            request_id=response_id,
            model=request.model,
            tool_names=tool_names,
        )

        # Send initial chunk with role
        initial_chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        async for chunk in stream:
            if terminal_outcome is not None:
                if _is_usage_only_chunk(chunk):
                    terminal_usage = make_chat_usage(chunk.usage)
                    continue
                terminal_outcome = exception_outcome(
                    "Chunk received after explicit terminal",
                    code="upstream_protocol_error",
                )
                yield _stream_error(
                    terminal_outcome.error.message,
                    terminal_outcome.error.code,
                )
                pipeline.finish(
                    wire_status=200,
                    outcome=terminal_outcome,
                    body_summary=terminal_outcome.error.message,
                )
                return

            abort_reason = pipeline.feed_stream(chunk)
            if abort_reason:
                terminal_outcome = exception_outcome(abort_reason, code="overloaded")
                error_data = {
                    "error": {"message": "Server overloaded, retry", "type": "overloaded"}
                }
                yield f"data: {json.dumps(error_data)}\n\n"
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
            usage = make_chat_usage(chunk.usage)
            if chunk.content or chunk.tool_calls:
                chunk_response = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(
                                content=chunk.content if chunk.content else None,
                                tool_calls=chunk.tool_calls,
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None if chunk_outcome is not None else usage,
                )
                yield f"data: {chunk_response.model_dump_json()}\n\n"
            elif usage and chunk_outcome is None:
                usage_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=request.model,
                    choices=[],
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

            if chunk_outcome is not None:
                if chunk_outcome.response_status not in {
                    ResponseStatus.COMPLETED,
                    ResponseStatus.INCOMPLETE,
                }:
                    error = chunk_outcome.error
                    error_data = {
                        "error": {
                            "message": error.message if error else "Upstream response failed",
                            "type": error.code if error else "provider_error",
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    pipeline.finish(
                        wire_status=200,
                        outcome=chunk_outcome,
                        body_summary=error.message if error else None,
                    )
                    return
                terminal_outcome = chunk_outcome
                terminal_usage = usage

        if terminal_outcome is None:
            terminal_outcome = unexpected_eof_outcome()
            yield _stream_error(
                terminal_outcome.error.message,
                terminal_outcome.error.code,
            )
        else:
            final_chunk = ChatCompletionChunk(
                id=response_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason=finish_reason_for_outcome(terminal_outcome),
                    )
                ],
                usage=terminal_usage,
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        pipeline.finish(wire_status=200, outcome=terminal_outcome)

    except ProviderError as e:
        terminal_outcome = exception_outcome(str(e), code="provider_error")
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary=str(e),
            )
        yield _stream_error(str(e), "provider_error")
    except asyncio.CancelledError:
        if pipeline is not None:
            pipeline.finish(wire_status=200, outcome=client_cancelled_outcome())
        logger.info("Chat stream cancelled by client")
        raise
    except Exception:
        terminal_outcome = exception_outcome("Internal server error", code="server_error")
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary="Internal server error",
            )
        logger.error("Unexpected error in chat stream", exc_info=True)
        yield _stream_error("Internal server error", "server_error")
    finally:
        await close_async_iterator(stream)


def _stream_error(message: str, error_type: str) -> str:
    """Format an OpenAI Chat stream error without a success sentinel."""
    return f"data: {json.dumps({'error': {'message': message, 'type': error_type}})}\n\n"


def _is_usage_only_chunk(chunk) -> bool:
    """Return whether a post-terminal chunk carries usage and nothing else."""
    return bool(chunk.usage) and not any(
        (
            chunk.content,
            chunk.finish_reason,
            chunk.tool_calls,
            chunk.thinking,
            chunk.thinking_signature,
            chunk.thinking_id,
            chunk.terminal_outcome,
        )
    )
