"""Bridges between semantic IR and the provider layer's legacy Chat DTOs."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from router_maestro.protocols._tool_result_projection import project_tool_result_output
from router_maestro.protocols._wire import reject, thaw_json
from router_maestro.protocols.models import (
    FileContent,
    FrozenJsonValue,
    ImageContent,
    MessageRole,
    OpaqueState,
    ReasoningSummary,
    RefusalContent,
    SemanticEvent,
    SemanticEventType,
    SemanticMessage,
    SemanticRequest,
    SemanticResponse,
    TerminalMetadata,
    TextContent,
    ToolCall,
    ToolResult,
    Usage,
    WireProtocol,
)

if TYPE_CHECKING:
    from router_maestro.providers.base import ChatRequest, ChatResponse, ChatStreamChunk

_PROTOCOL = WireProtocol.OPENAI_CHAT


def semantic_request_to_legacy_chat(request: SemanticRequest) -> ChatRequest:
    """Project portable semantic request fields into the legacy Chat DTO."""
    from router_maestro.providers.base import ChatRequest, Message

    if request.parallel_tool_calls is not None:
        reject(
            _PROTOCOL,
            "parallel_tool_calls",
            "legacy Chat request DTO cannot preserve this option",
        )

    messages = []
    for index, item in enumerate(request.input):
        semantic = _coerce_message(item, parameter=f"input[{index}]")
        messages.append(_legacy_message(semantic, parameter=f"input[{index}]", message_cls=Message))
    tools = []
    for tool in request.tools:
        function: dict[str, Any] = {
            "name": tool.name,
            "parameters": thaw_json(tool.input_schema),
        }
        if tool.description is not None:
            function["description"] = tool.description
        if tool.strict is not None:
            function["strict"] = tool.strict
        tools.append({"type": "function", "function": function})
    thinking_type = None
    reasoning_effort = None
    thinking_budget = None
    if request.reasoning is not None:
        thinking_budget = request.reasoning.budget_tokens
        if request.reasoning.effort == "adaptive":
            thinking_type = "adaptive"
        else:
            reasoning_effort = request.reasoning.effort
            if request.reasoning.enabled is True:
                thinking_type = "enabled"
            elif request.reasoning.enabled is False:
                thinking_type = "disabled"
    return ChatRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=tools or None,
        tool_choice=_legacy_tool_choice(request),
        thinking_budget=thinking_budget,
        thinking_type=thinking_type,
        reasoning_effort=reasoning_effort,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        stop=list(request.stop_sequences) or None,
        user=request.user,
        top_k=request.top_k,
        metadata=thaw_json(request.metadata) or None,
        service_tier=request.service_tier,
        candidate_count=request.candidate_count,
        response_mime_type=request.response_mime_type,
        output_format=(
            thaw_json(request.structured_output) if request.structured_output is not None else None
        ),
        provider_extensions=thaw_json(request.provider_extensions),
    )


def _coerce_message(item: object, *, parameter: str) -> SemanticMessage:
    if isinstance(item, SemanticMessage):
        return item
    if isinstance(item, ToolCall):
        return SemanticMessage(role=MessageRole.ASSISTANT, content=(item,))
    if isinstance(item, ToolResult):
        return SemanticMessage(role=MessageRole.TOOL, content=(item,))
    if isinstance(item, TextContent | ImageContent | FileContent):
        return SemanticMessage(role=MessageRole.USER, content=(item,))
    if isinstance(item, ReasoningSummary | RefusalContent):
        return SemanticMessage(role=MessageRole.ASSISTANT, content=(item,))
    reject(_PROTOCOL, parameter, f"unsupported semantic item {type(item).__name__}")


def _legacy_message(
    message: SemanticMessage,
    *,
    parameter: str,
    message_cls: type,
) -> Any:
    if message.item_id is not None or message.status is not None:
        reject(_PROTOCOL, parameter, "legacy Chat messages cannot carry item metadata")
    if message.role is MessageRole.TOOL:
        if len(message.content) != 1 or not isinstance(message.content[0], ToolResult):
            reject(_PROTOCOL, parameter, "tool role requires exactly one ToolResult")
        result = message.content[0]
        if result.item_id is not None:
            reject(_PROTOCOL, parameter, "legacy Chat tool results lack item IDs")
        content = project_tool_result_output(
            _legacy_tool_result_content(result, parameter=parameter),
            is_error=result.is_error,
        )
        return message_cls(
            role="tool",
            content=content,
            tool_call_id=result.call_id,
        )

    blocks: list[dict[str, Any]] = []
    tool_calls = []
    refusal = None
    for index, part in enumerate(message.content):
        path = f"{parameter}.content[{index}]"
        if isinstance(part, TextContent):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            if message.role not in {MessageRole.USER, MessageRole.SYSTEM}:
                reject(_PROTOCOL, path, "legacy Chat images require user/system role")
            blocks.append(_legacy_image(part, parameter=path))
        elif isinstance(part, FileContent):
            if message.role not in {MessageRole.USER, MessageRole.SYSTEM}:
                reject(_PROTOCOL, path, "legacy Chat files require user/system role")
            blocks.append(_legacy_file(part, parameter=path))
        elif isinstance(part, ToolCall):
            if message.role is not MessageRole.ASSISTANT:
                reject(_PROTOCOL, path, "legacy Chat tool calls require assistant role")
            if part.item_id not in {None, part.call_id}:
                reject(_PROTOCOL, path, "legacy Chat cannot preserve a distinct tool item ID")
            if part.kind != "function" or part.namespace is not None:
                reject(_PROTOCOL, path, "legacy Chat supports unnamespaced function calls only")
            if part.opaque_state is not None:
                reject(_PROTOCOL, path, "legacy Chat cannot carry opaque tool state")
            tool_calls.append(
                {
                    "id": part.call_id,
                    "type": "function",
                    "function": {
                        "name": part.name,
                        "arguments": json.dumps(
                            thaw_json(part.arguments),
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    },
                }
            )
        elif isinstance(part, RefusalContent):
            if message.role is not MessageRole.ASSISTANT or refusal is not None:
                reject(_PROTOCOL, path, "legacy Chat supports one assistant refusal")
            refusal = part.refusal
        elif isinstance(part, ReasoningSummary):
            reject(_PROTOCOL, path, "legacy Chat request history cannot carry reasoning items")
        elif isinstance(part, ToolResult):
            reject(_PROTOCOL, path, "tool results require a tool-role semantic message")
        else:  # pragma: no cover - closed semantic union
            reject(_PROTOCOL, path, f"unsupported content {type(part).__name__}")
    if not blocks:
        content: str | list | None = None
    elif len(blocks) == 1 and blocks[0].get("type") == "text":
        content = blocks[0]["text"]
    else:
        content = blocks
    return message_cls(
        role=message.role.value,
        content=content,
        tool_calls=tool_calls or None,
        refusal=refusal,
    )


def _legacy_image(image: ImageContent, *, parameter: str) -> dict[str, Any]:
    if isinstance(image.source, bytes):
        if image.media_type is None:
            reject(_PROTOCOL, f"{parameter}.media_type", "binary image requires MIME type")
        url = f"data:{image.media_type};base64,{base64.b64encode(image.source).decode('ascii')}"
    elif image.source_kind in {"base64", "inline_data"}:
        if image.media_type is None:
            reject(_PROTOCOL, f"{parameter}.media_type", "base64 image requires MIME type")
        url = f"data:{image.media_type};base64,{image.source}"
    else:
        url = image.source
    image_url: dict[str, Any] = {"url": url}
    if image.detail is not None:
        image_url["detail"] = image.detail
    return {"type": "image_url", "image_url": image_url}


def _legacy_file(file: FileContent, *, parameter: str) -> dict[str, Any]:
    source_kind = file.source_kind
    if source_kind is None:
        source_kind = "base64" if isinstance(file.source, bytes) else None
    if source_kind not in {"base64", "inline_data", "url", "file_uri", "text", "content"}:
        reject(_PROTOCOL, f"{parameter}.source_kind", "legacy file source is ambiguous")
    if isinstance(file.source, bytes):
        data = base64.b64encode(file.source).decode("ascii")
    else:
        data = file.source
    anthropic_kind = {
        "inline_data": "base64",
        "file_uri": "url",
    }.get(source_kind, source_kind)
    source: dict[str, Any] = {"type": anthropic_kind}
    field = (
        "url" if anthropic_kind == "url" else "content" if anthropic_kind == "content" else "data"
    )
    source[field] = data
    if file.media_type is not None:
        source["media_type"] = file.media_type
    payload: dict[str, Any] = {"type": "document", "source": source}
    if file.filename is not None:
        payload["title"] = file.filename
    return payload


def _legacy_tool_result_content(result: ToolResult, *, parameter: str) -> str:
    if result.content and result.structured_content is not None:
        reject(_PROTOCOL, parameter, "cannot combine text and structured tool output")
    texts = []
    for index, part in enumerate(result.content):
        if not isinstance(part, TextContent):
            reject(
                _PROTOCOL,
                f"{parameter}.content[{index}]",
                "legacy Chat tool results support text only",
            )
        texts.append(part.text)
    if result.structured_content is not None:
        return json.dumps(thaw_json(result.structured_content), ensure_ascii=False)
    return "".join(texts)


def _legacy_tool_choice(request: SemanticRequest) -> str | dict[str, Any] | None:
    choice = request.tool_choice
    if choice is None:
        return None
    if choice.mode in {"auto", "none", "required"}:
        if choice.name is not None:
            reject(_PROTOCOL, "tool_choice.name", "named choice requires function mode")
        return choice.mode
    if choice.mode == "function" and choice.name:
        return {"type": "function", "function": {"name": choice.name}}
    reject(_PROTOCOL, "tool_choice.mode", f"unsupported mode {choice.mode!r}")


def semantic_response_from_legacy_chat(
    response: ChatResponse,
    *,
    response_id: str,
    origin_protocol: WireProtocol = WireProtocol.OPENAI_CHAT,
    origin_provider: str | None = None,
    origin_binding: str | None = None,
) -> SemanticResponse:
    """Convert a provider Chat response into semantic IR without wire re-encoding."""
    parts: list[Any] = []
    if response.content is not None:
        parts.append(TextContent(response.content))
    if response.refusal is not None:
        parts.append(RefusalContent(response.refusal))
    if response.thinking is not None or response.thinking_signature is not None:
        opaque = _legacy_opaque_state(
            signature=response.thinking_signature,
            opaque_payload=None,
            item_id=response.thinking_id,
            model=response.model,
            origin_protocol=origin_protocol,
            origin_provider=origin_provider,
            origin_binding=origin_binding,
        )
        parts.append(ReasoningSummary(response.thinking or "", opaque_state=opaque))
    for index, raw_call in enumerate(response.tool_calls or []):
        parts.append(_legacy_tool_call(raw_call, parameter=f"response.tool_calls[{index}]"))
    terminal = _legacy_terminal(response.finish_reason, response.terminal_outcome)
    return SemanticResponse(
        id=response_id,
        model=response.model,
        output=(SemanticMessage(role=MessageRole.ASSISTANT, content=tuple(parts)),),
        usage=_legacy_usage(response.usage),
        terminal=terminal,
    )


def _legacy_tool_call(value: object, *, parameter: str) -> ToolCall:
    if not isinstance(value, Mapping):
        reject(_PROTOCOL, parameter, "tool call must be an object")
    if value.get("type", "function") != "function":
        reject(_PROTOCOL, f"{parameter}.type", "legacy Chat supports function calls only")
    function = value.get("function")
    if not isinstance(function, Mapping):
        reject(_PROTOCOL, f"{parameter}.function", "must be an object")
    call_id = value.get("id")
    name = function.get("name")
    if not isinstance(call_id, str) or not call_id:
        reject(_PROTOCOL, f"{parameter}.id", "must be a non-empty string")
    if not isinstance(name, str) or not name:
        reject(_PROTOCOL, f"{parameter}.function.name", "must be a non-empty string")
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            reject(_PROTOCOL, f"{parameter}.function.arguments", "must be valid JSON")
    if not isinstance(arguments, Mapping):
        reject(_PROTOCOL, f"{parameter}.function.arguments", "must decode to an object")
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _legacy_usage(value: object) -> Usage | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        reject(_PROTOCOL, "usage", "must be an object")
    return Usage(
        input_tokens=_token_count(value.get("prompt_tokens"), "usage.prompt_tokens"),
        output_tokens=_token_count(value.get("completion_tokens"), "usage.completion_tokens"),
        total_tokens=_token_count(value.get("total_tokens"), "usage.total_tokens"),
        cached_input_tokens=_nested_token_count(
            value,
            "prompt_tokens_details",
            "cached_tokens",
        ),
        reasoning_tokens=_nested_token_count(
            value,
            "completion_tokens_details",
            "reasoning_tokens",
        ),
    )


def _token_count(value: object, parameter: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        reject(_PROTOCOL, parameter, "must be a non-negative integer")
    return value


def _nested_token_count(
    usage: Mapping[str, Any],
    details_field: str,
    value_field: str,
) -> int | None:
    details = usage.get(details_field)
    if details is None:
        return None
    if not isinstance(details, Mapping):
        reject(_PROTOCOL, f"usage.{details_field}", "must be an object")
    return _token_count(details.get(value_field), f"usage.{details_field}.{value_field}")


def _legacy_terminal(finish_reason: str | None, outcome: object) -> TerminalMetadata:
    if outcome is None:
        status = "incomplete" if finish_reason in {"length", "content_filter"} else "completed"
        return TerminalMetadata(finish_reason=finish_reason, response_status=status)
    response_status = getattr(outcome, "response_status", None)
    status = getattr(response_status, "value", response_status)
    canonical_finish = getattr(outcome, "finish_reason", None) or finish_reason
    error = getattr(outcome, "error", None)
    return TerminalMetadata(
        finish_reason=canonical_finish,
        error_code=getattr(error, "code", None),
        error_message=getattr(error, "message", None),
        response_status=status,
    )


def semantic_events_from_legacy_chat_chunk(
    chunk: ChatStreamChunk,
    *,
    response_id: str | None = None,
    model: str | None = None,
    origin_protocol: WireProtocol = WireProtocol.OPENAI_CHAT,
    origin_provider: str | None = None,
    origin_binding: str | None = None,
    sequence_start: int | None = None,
) -> tuple[SemanticEvent, ...]:
    """Convert one legacy Chat chunk into ordered semantic stream events."""
    events: list[SemanticEvent] = []
    sequence = sequence_start

    def add(event_type: SemanticEventType, **kwargs: Any) -> None:
        nonlocal sequence
        events.append(
            SemanticEvent(
                type=event_type,
                sequence=sequence,
                response_id=response_id,
                **kwargs,
            )
        )
        if sequence is not None:
            sequence += 1

    if chunk.thinking:
        add(SemanticEventType.REASONING_DELTA, delta=chunk.thinking)
    if chunk.thinking_signature is not None or chunk.opaque_payload is not None:
        if model is None:
            reject(_PROTOCOL, "chunk.model", "opaque reasoning requires model context")
        opaque = _legacy_opaque_state(
            signature=chunk.thinking_signature,
            opaque_payload=chunk.opaque_payload,
            item_id=chunk.thinking_id,
            model=model,
            origin_protocol=origin_protocol,
            origin_provider=origin_provider,
            origin_binding=origin_binding,
        )
        add(
            SemanticEventType.OUTPUT_ITEM,
            item_id=chunk.thinking_id,
            item=ReasoningSummary("", opaque_state=opaque),
        )
    if chunk.content:
        add(SemanticEventType.TEXT_DELTA, delta=chunk.content)
    if chunk.refusal:
        add(SemanticEventType.OUTPUT_ITEM, item=RefusalContent(chunk.refusal))
    for raw_call in chunk.tool_calls or []:
        if not isinstance(raw_call, Mapping):
            reject(_PROTOCOL, "chunk.tool_calls", "tool call delta must be an object")
        function = raw_call.get("function") or {}
        if not isinstance(function, Mapping):
            reject(_PROTOCOL, "chunk.tool_calls.function", "must be an object")
        call_id = raw_call.get("id")
        name = function.get("name")
        tool_index = raw_call.get("index")
        if call_id is not None and (not isinstance(call_id, str) or not call_id):
            reject(_PROTOCOL, "chunk.tool_calls.id", "must be a non-empty string")
        if name is not None and (not isinstance(name, str) or not name):
            reject(_PROTOCOL, "chunk.tool_calls.function.name", "must be non-empty text")
        if call_id is not None and name is not None:
            add(
                SemanticEventType.OUTPUT_ITEM,
                item_id=call_id,
                item=ToolCall(call_id=call_id, name=name),
                metadata={"tool_index": tool_index},
            )
        arguments = function.get("arguments")
        if arguments:
            if not isinstance(arguments, str):
                reject(_PROTOCOL, "chunk.tool_calls.function.arguments", "must be text")
            add(
                SemanticEventType.TOOL_ARGUMENTS_DELTA,
                item_id=call_id,
                delta=arguments,
                metadata={"tool_index": tool_index, "name": name},
            )
    if chunk.usage is not None:
        usage = _legacy_usage(chunk.usage)
        if usage is not None:
            add(SemanticEventType.USAGE, usage=usage)
    if chunk.finish_reason is not None or chunk.terminal_outcome is not None:
        add(
            SemanticEventType.TERMINAL,
            terminal=_legacy_terminal(chunk.finish_reason, chunk.terminal_outcome),
        )
    return tuple(events)


def _legacy_opaque_state(
    *,
    signature: str | None,
    opaque_payload: str | None,
    item_id: str | None,
    model: str,
    origin_protocol: WireProtocol,
    origin_provider: str | None,
    origin_binding: str | None,
) -> OpaqueState | None:
    if signature is None and opaque_payload is None:
        return None
    if item_id is None:
        reject(_PROTOCOL, "opaque_state.item_id", "opaque state requires an item ID")
    if opaque_payload is None:
        if signature is None:
            raise AssertionError("opaque state carrier is required")
        blob: FrozenJsonValue = signature
    else:
        blob = opaque_payload
        try:
            blob = cast(FrozenJsonValue, json.loads(opaque_payload))
        except json.JSONDecodeError:
            blob = opaque_payload
    return OpaqueState(
        origin_protocol=origin_protocol,
        origin_provider=origin_provider,
        origin_model=model,
        item_id=item_id,
        blob=blob,
        origin_binding=origin_binding,
    )


__all__ = [
    "semantic_events_from_legacy_chat_chunk",
    "semantic_request_to_legacy_chat",
    "semantic_response_from_legacy_chat",
]
