"""Chat completions route."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from pydantic import ValidationError

from router_maestro.providers import (
    ChatRequest,
    Message,
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
from router_maestro.routing.model_ref import qualify_model_id
from router_maestro.server.protocols import client_error_response, unrepresented_option_error
from router_maestro.server.protocols.errors import postcommit_error_data, protocol_error_response
from router_maestro.server.routes._outcomes import record_chat_response_outcome
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
from router_maestro.server.schemas.openai import (
    ChatCompletionStreamOptions,
    OpenAIThinkingConfig,
)
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.utils import get_logger
from router_maestro.utils.async_iterators import close_async_iterator
from router_maestro.utils.reasoning import VALID_EFFORTS, effort_to_budget

logger = get_logger("server.routes.chat")

router = APIRouter()


def _thinking_validation_error(error: ValidationError) -> RequestOptionError:
    """Map strict thinking-schema failures to one stable OpenAI parameter."""
    first = error.errors()[0]
    location = first.get("loc", ())
    parameter = "thinking"
    if location:
        parameter = f"thinking.{'.'.join(str(part) for part in location)}"
    return RequestOptionError(
        f"Invalid request option '{parameter}'",
        parameter=parameter,
    )


def _stream_options_validation_error(error: ValidationError) -> RequestOptionError:
    """Map strict stream-options failures to one stable OpenAI parameter."""
    first = error.errors()[0]
    location = first.get("loc", ())
    parameter = "stream_options"
    if location:
        parameter = f"stream_options.{'.'.join(str(part) for part in location)}"
    return RequestOptionError(
        f"Invalid request option '{parameter}'",
        parameter=parameter,
    )


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
    stream_options = None
    if "stream_options" in request.model_fields_set:
        if request.stream_options is None:
            return client_error_response(
                RequestOptionError(
                    "Invalid request option 'stream_options'",
                    parameter="stream_options",
                ),
                "openai",
            )
        try:
            stream_options = ChatCompletionStreamOptions.model_validate(request.stream_options)
        except ValidationError as error:
            return client_error_response(
                _stream_options_validation_error(error),
                "openai",
            )
        request.stream_options = stream_options

    if error := unrepresented_option_error(request):
        return client_error_response(error, "openai")
    if stream_options is not None and not request.stream:
        return client_error_response(
            RequestOptionError(
                "Invalid request option 'stream_options'",
                parameter="stream_options",
            ),
            "openai",
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
                refusal=m.refusal,
            )
        )

    chat_request = ChatRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        stop=request.stop,
        user=request.user,
        metadata=request.metadata,
        service_tier=request.service_tier,
    )

    # Reasoning / thinking passthrough.
    # Prefer OpenAI-style ``reasoning_effort``; also accept Anthropic-style
    # ``thinking`` for SDKs that forward it via the OpenAI endpoint.
    thinking = None
    if "thinking" in request.model_fields_set and request.thinking is None:
        return client_error_response(
            RequestOptionError(
                "Invalid request option 'thinking'",
                parameter="thinking",
            ),
            "openai",
        )
    if request.thinking is not None:
        try:
            thinking = OpenAIThinkingConfig.model_validate(request.thinking)
        except ValidationError as error:
            return client_error_response(_thinking_validation_error(error), "openai")

    effort = (request.reasoning_effort or "").lower() or None
    if effort and effort not in VALID_EFFORTS:
        return client_error_response(
            RequestOptionError(
                f"Unsupported reasoning_effort '{request.reasoning_effort}'",
                parameter="reasoning_effort",
            ),
            "openai",
        )
    if effort:
        chat_request.reasoning_effort = effort
        chat_request.thinking_budget = effort_to_budget(effort)
        chat_request.thinking_type = "enabled"
    elif thinking is not None:
        chat_request.thinking_type = thinking.type
        chat_request.thinking_budget = (
            None if thinking.type == "disabled" else thinking.budget_tokens
        )

    if request.stream:
        stream = None
        try:
            prepared_plan = await model_router.prepare_chat_completion_stream(chat_request)
            stream, provider_name = await model_router.chat_completion_stream(
                chat_request,
                prepared_plan=prepared_plan,
            )
        except ProviderError as e:
            if isinstance(e, RequestOptionError):
                return client_error_response(e, "openai")
            return protocol_error_response(e, "openai_chat")
        try:
            return sse_streaming_response(
                stream_response(
                    model_router,
                    chat_request,
                    include_usage=(
                        stream_options.include_usage if stream_options is not None else None
                    ),
                    opened_stream=stream,
                    opened_provider_name=provider_name,
                )
            )
        except Exception:
            await close_async_iterator(stream)
            raise

    try:
        response, provider_name = await model_router.chat_completion(chat_request)

        usage = make_chat_usage(response.usage)

        # Build response message with optional tool_calls
        response_tool_calls = None
        if response.tool_calls:
            response_tool_calls = [ChatMessageToolCall(**tc) for tc in response.tool_calls]

        downstream_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=(
                response.selected_model.qualified_id
                if response.selected_model is not None
                else qualify_model_id(provider_name, response.model)
            ),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response.content,
                        refusal=response.refusal,
                        tool_calls=response_tool_calls,
                    ),
                    finish_reason=response.finish_reason,
                )
            ],
            usage=usage,
        )
        record_chat_response_outcome(response)
        return downstream_response
    except ProviderError as e:
        logger.error("Chat completion request failed: %s", e)
        if isinstance(e, RequestOptionError):
            return client_error_response(e, "openai")
        return protocol_error_response(e, "openai_chat")


async def stream_response(
    model_router: Router,
    request: ChatRequest,
    *,
    include_usage: bool | None = None,
    prepared_plan=None,
    opened_stream=None,
    opened_provider_name: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream chat completion response with an optional explicit usage policy.

    ``None`` preserves Router-Maestro's legacy wire behavior. Explicit booleans
    implement OpenAI ``stream_options.include_usage`` without changing what the
    provider is asked to collect.
    """
    pipeline = None
    stream = opened_stream
    terminal_outcome: TerminalOutcome | None = None
    terminal_usage: ChatCompletionUsage | None = None
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
            raise RuntimeError("Opened Chat stream is missing provider identity")
        selected_model = getattr(stream, "selected_model", None)
        response_model = (
            selected_model.qualified_id
            if selected_model is not None
            else qualify_model_id(provider_name, request.model)
        )
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
            model=response_model,
            tool_names=tool_names,
        )

        # Send initial chunk with role
        initial_chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=response_model,
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
                error_data = postcommit_error_data(
                    ProviderError(
                        "Server overloaded, retry",
                        status_code=529,
                        kind=ProviderFailureKind.RATE_LIMIT,
                    ),
                    "openai_chat",
                )
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
            if include_usage is not None and usage is not None:
                terminal_usage = usage
            if chunk.content or chunk.refusal or chunk.tool_calls:
                chunk_response = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=response_model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(
                                content=chunk.content if chunk.content else None,
                                refusal=chunk.refusal,
                                tool_calls=chunk.tool_calls,
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=(
                        None if include_usage is not None or chunk_outcome is not None else usage
                    ),
                )
                yield f"data: {chunk_response.model_dump_json()}\n\n"
            elif usage and chunk_outcome is None and include_usage is None:
                usage_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=response_model,
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
                if include_usage is None:
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
                model=response_model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason=finish_reason_for_outcome(terminal_outcome),
                    )
                ],
                usage=terminal_usage if include_usage is None else None,
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            if include_usage and terminal_usage is not None:
                usage_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=response_model,
                    choices=[],
                    usage=terminal_usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"
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
        yield f"data: {json.dumps(postcommit_error_data(e, 'openai_chat'))}\n\n"
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
            chunk.refusal,
            chunk.finish_reason,
            chunk.tool_calls,
            chunk.thinking,
            chunk.thinking_signature,
            chunk.thinking_id,
            chunk.terminal_outcome,
        )
    )
