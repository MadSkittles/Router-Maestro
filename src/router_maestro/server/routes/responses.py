"""Responses API route for Codex models."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from router_maestro.providers import (
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    TerminalOutcome,
    client_cancelled_outcome,
    exception_outcome,
)
from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.routing import Router, get_router
from router_maestro.routing.model_ref import qualify_model_id
from router_maestro.server.protocols import (
    ResponsesReducer,
    build_nonstream_snapshot,
    client_error_response,
    responses_reducer,
    unrepresented_option_error,
)
from router_maestro.server.schemas import (
    ResponsesReasoningConfig,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from router_maestro.server.streaming import sse_streaming_response
from router_maestro.utils import get_logger
from router_maestro.utils.async_iterators import close_async_iterator

logger = get_logger("server.routes.responses")

# Compatibility exports for existing helper callers. Construction remains
# authoritative in the pure reducer module.
generate_id = responses_reducer.generate_id
make_function_call_item = responses_reducer.make_function_call_item
make_message_item = responses_reducer.make_message_item
make_text_content = responses_reducer.make_text_content
make_usage = responses_reducer.make_usage

router = APIRouter()


def _reasoning_validation_error(error: ValidationError) -> RequestOptionError:
    """Map strict Responses reasoning failures to one stable OpenAI parameter."""
    first = error.errors()[0]
    location = first.get("loc", ())
    if location == ("effort",):
        parameter = "reasoning_effort"
    elif location:
        parameter = f"reasoning.{'.'.join(str(part) for part in location)}"
    else:
        parameter = "reasoning"
    return RequestOptionError(
        f"Invalid request option '{parameter}'",
        parameter=parameter,
    )


def sse_event(data: dict[str, Any]) -> str:
    """Format data as SSE event with event type field."""
    event_type = data.get("type", "")
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def extract_text_from_content(content: str | list[Any]) -> str:
    """Extract text from content which can be a string or list of content blocks."""
    if isinstance(content, str):
        return content

    texts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") in ("input_text", "output_text"):
                texts.append(block.get("text", ""))
            elif "text" in block:
                texts.append(block.get("text", ""))
        elif hasattr(block, "text"):
            texts.append(block.text)
    return "".join(texts)


def convert_content_to_serializable(content: Any) -> Any:
    """Convert content to JSON-serializable format.

    Handles Pydantic models and nested structures.
    """
    if isinstance(content, str):
        return content
    if hasattr(content, "model_dump"):
        return content.model_dump(exclude_none=True)
    if isinstance(content, list):
        return [convert_content_to_serializable(item) for item in content]
    if isinstance(content, dict):
        return {k: convert_content_to_serializable(v) for k, v in content.items()}
    return content


def convert_input_to_internal(
    input_data: str | list[Any],
) -> str | list[dict[str, Any]]:
    """Convert the incoming input format to internal format.

    Preserves the original content format (string or array) as the upstream
    Copilot API accepts both formats. Converts Pydantic models to dicts.
    """
    if isinstance(input_data, str):
        return input_data

    items = []
    for item in input_data:
        if isinstance(item, dict):
            item_type = item.get("type", "message")

            if item_type == "message" or (item_type is None and "role" in item):
                role = item.get("role", "user")
                content = item.get("content", "")
                # Convert content to serializable format
                content = convert_content_to_serializable(content)
                items.append({"type": "message", "role": role, "content": content})

            elif item_type == "function_call":
                fc_item: dict[str, Any] = {
                    "type": "function_call",
                    "id": item.get("id"),
                    "call_id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments", "{}"),
                    "status": item.get("status", "completed"),
                }
                # Preserve MCP namespace verbatim. Copilot CAPI rejects the
                # next turn with ``Missing namespace for function_call 'X'``
                # if a previously-namespaced call is round-tripped without it.
                if item.get("namespace") is not None:
                    fc_item["namespace"] = item["namespace"]
                items.append(fc_item)

            elif item_type == "function_call_output":
                output = item.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.get("call_id"),
                        "output": output,
                    }
                )

            elif item_type == "reasoning":
                # Echoed back from a prior turn — preserve the full shape so
                # Copilot can correlate chain-of-thought across turns. Mirrors
                # vscode-copilot-chat responsesApi.ts:216-230 (extractThinkingData).
                #
                # Codex CLI's ``ResponseItem::Reasoning`` marks ``id`` as
                # ``#[serde(default, skip_serializing)]`` (see openai/codex
                # codex-rs/protocol/src/models.rs), so it NEVER sends the id
                # back on round-trip. Copilot CAPI signs ``encrypted_content``
                # against the upstream id and rejects (id, blob) pairs that
                # don't match. Without an id, the blob is unverifiable, so we
                # MUST strip it — otherwise Copilot 400s with ``Encrypted
                # content could not be decrypted``. (See openai/codex#17541
                # and the parallel litellm bug BerriAI/litellm#22189.)
                reasoning_item: dict[str, Any] = {
                    "type": "reasoning",
                    "id": item.get("id"),
                    "summary": item.get("summary", []) or [],
                }
                if item.get("encrypted_content") and item.get("id"):
                    reasoning_item["encrypted_content"] = item["encrypted_content"]
                items.append(reasoning_item)
            else:
                items.append(convert_content_to_serializable(item))

        elif hasattr(item, "model_dump"):
            # Pydantic model - convert to dict
            items.append(item.model_dump(exclude_none=True))

        elif hasattr(item, "role") and hasattr(item, "content"):
            # Object with role and content attributes
            content = convert_content_to_serializable(item.content)
            items.append({"type": "message", "role": item.role, "content": content})

    return items


def convert_tools_to_internal(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert tools to internal format."""
    if not tools:
        return None
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            result.append(tool)
        elif hasattr(tool, "model_dump"):
            result.append(tool.model_dump(exclude_none=True))
        else:
            result.append(dict(tool))
    return result


def convert_tool_choice_to_internal(
    tool_choice: str | Any | None,
) -> str | dict[str, Any] | None:
    """Convert tool_choice to internal format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        return tool_choice
    if hasattr(tool_choice, "model_dump"):
        return tool_choice.model_dump(exclude_none=True)
    return dict(tool_choice)


@router.post("/api/openai/v1/responses")
async def create_response(request: ResponsesRequest):
    """Handle Responses API requests (for Codex models)."""
    request_id = generate_id("req")
    start_time = time.time()

    logger.info(
        "Received responses request: req_id=%s, model=%s, stream=%s, has_tools=%s, reasoning=%s",
        request_id,
        request.model,
        request.stream,
        request.tools is not None,
        request.reasoning,
    )
    if error := unrepresented_option_error(request):
        return client_error_response(error, "openai")

    model_router = get_router()

    input_value = convert_input_to_internal(request.input)

    internal_request = InternalResponsesRequest(
        model=request.model,
        input=input_value,
        stream=request.stream,
        instructions=request.instructions,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        tools=convert_tools_to_internal(request.tools),
        tool_choice=convert_tool_choice_to_internal(request.tool_choice),
        parallel_tool_calls=request.parallel_tool_calls,
        top_p=request.top_p,
        metadata=request.metadata,
        service_tier=request.service_tier,
    )

    if request.reasoning is not None:
        try:
            reasoning = ResponsesReasoningConfig.model_validate(request.reasoning)
        except ValidationError as error:
            return client_error_response(_reasoning_validation_error(error), "openai")
        internal_request.reasoning_effort = reasoning.effort

    if request.stream:
        try:
            prepared_plan = await model_router.prepare_responses_completion_stream(internal_request)
        except ProviderError as e:
            if isinstance(e, RequestOptionError):
                return client_error_response(e, "openai")
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        return sse_streaming_response(
            stream_response(
                model_router,
                internal_request,
                request_id,
                start_time,
                prepared_plan=prepared_plan,
            ),
        )

    try:
        response, provider_name = await model_router.responses_completion(internal_request)
        snapshot = build_nonstream_snapshot(
            response,
            response_id=generate_id("resp"),
            model=(
                response.selected_model.qualified_id
                if response.selected_model is not None
                else qualify_model_id(provider_name, response.model)
            ),
        )
        payload = snapshot.response
        usage = (
            ResponsesUsage.model_validate(payload["usage"])
            if payload["usage"] is not None
            else None
        )
        return ResponsesResponse.model_construct(**{**payload, "usage": usage})
    except ProviderError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Responses request failed: req_id=%s, elapsed=%.1fms, error=%s",
            request_id,
            elapsed_ms,
            e,
        )
        if isinstance(e, RequestOptionError):
            return client_error_response(e, "openai")
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e


async def stream_response(
    model_router: Router,
    request: InternalResponsesRequest,
    request_id: str,
    start_time: float,
    *,
    prepared_plan=None,
) -> AsyncGenerator[str, None]:
    """Orchestrate a provider stream through the pure Responses reducer."""
    response_id = generate_id("resp")
    created_at = int(time.time())
    response_model = request.model
    pipeline = None
    stream = None
    terminal_outcome: TerminalOutcome | None = None
    reducer = ResponsesReducer(
        {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": response_model,
            "error": None,
            "incomplete_details": None,
        }
    )
    try:
        if prepared_plan is None:
            stream, provider_name = await model_router.responses_completion_stream(request)
        else:
            stream, provider_name = await model_router.responses_completion_stream(
                request,
                prepared_plan=prepared_plan,
            )
        selected_model = getattr(stream, "selected_model", None)
        response_model = (
            selected_model.qualified_id
            if selected_model is not None
            else qualify_model_id(provider_name, request.model)
        )
        reducer.bind_model(response_model)

        from router_maestro.pipeline import RequestPipeline

        pipeline = RequestPipeline.create(request_id=request_id, model=request.model)
        for event in reducer.start().events:
            yield sse_event(event)

        async for chunk in stream:
            abort_reason = pipeline.feed_stream(chunk)
            if abort_reason:
                terminal_outcome = exception_outcome(abort_reason, code="overloaded")
                logger.warning("Responses stream aborted: %s", abort_reason)
                result = reducer.terminate(
                    terminal_outcome,
                    wire_error={
                        "type": "server_error",
                        "message": "Overloaded: please retry",
                    },
                )
                for event in result.events:
                    yield sse_event(event)
                pipeline.finish(
                    wire_status=200,
                    outcome=terminal_outcome,
                    body_summary=abort_reason,
                )
                return

            result = reducer.feed(chunk)
            for event in result.events:
                yield sse_event(event)
            if result.snapshot is not None:
                terminal_outcome = result.snapshot.outcome
                pipeline.finish(
                    wire_status=200,
                    outcome=terminal_outcome,
                    body_summary=(
                        terminal_outcome.error.message
                        if terminal_outcome.error is not None
                        else None
                    ),
                )
                return

        result = reducer.finish()
        for event in result.events:
            yield sse_event(event)
        if result.snapshot is None:
            raise RuntimeError("Responses reducer did not produce an EOF snapshot")
        terminal_outcome = result.snapshot.outcome
        logger.warning(
            "Stream ended without explicit terminal: req_id=%s model=%s output_items=%d",
            request_id,
            request.model,
            len(result.snapshot.response["output"]),
        )
        pipeline.finish(
            wire_status=200,
            outcome=terminal_outcome,
            body_summary=(
                terminal_outcome.error.message if terminal_outcome.error is not None else None
            ),
        )

    except ProviderError as error:
        outcome_code = (
            "upstream_protocol_error"
            if error.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
            else "provider_error"
        )
        wire_error_code = (
            "upstream_protocol_error"
            if error.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
            else "server_error"
        )
        terminal_outcome = exception_outcome(str(error), code=outcome_code)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Stream failed: req_id=%s, elapsed=%.1fms, error=%s",
            request_id,
            elapsed_ms,
            error,
        )
        result = reducer.terminate(
            terminal_outcome,
            wire_error={"code": wire_error_code, "message": str(error)},
        )
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary=str(error),
            )
        for event in result.events:
            yield sse_event(event)
    except asyncio.CancelledError:
        if pipeline is not None:
            pipeline.finish(wire_status=200, outcome=client_cancelled_outcome())
        logger.info("Responses stream cancelled by client: req_id=%s", request_id)
        raise
    except Exception:
        terminal_outcome = exception_outcome("Internal server error", code="server_error")
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Unexpected error in responses stream: req_id=%s, elapsed=%.1fms",
            request_id,
            elapsed_ms,
            exc_info=True,
        )
        result = reducer.terminate(
            terminal_outcome,
            wire_error={"code": "server_error", "message": "Internal server error"},
        )
        if pipeline is not None:
            pipeline.finish(
                wire_status=200,
                outcome=terminal_outcome,
                body_summary="Internal server error",
            )
        for event in result.events:
            yield sse_event(event)
    finally:
        await close_async_iterator(stream)
