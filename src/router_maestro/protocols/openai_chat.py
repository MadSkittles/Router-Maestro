"""Concrete OpenAI Chat Completions protocol runtime."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
from typing import Any, cast

from router_maestro.protocols._tool_namespace import (
    decode_namespaced_tool_name,
    encode_namespaced_tool_name,
)
from router_maestro.protocols._tool_result_projection import (
    ToolResultProjectionError,
    project_tool_result_output,
    unproject_tool_result_output,
)
from router_maestro.protocols.models import (
    FileContent,
    ImageContent,
    MessageRole,
    OpaqueState,
    ReasoningConfig,
    ReasoningSummary,
    RefusalContent,
    RequestManifest,
    SemanticEvent,
    SemanticEventType,
    SemanticMessage,
    SemanticRequest,
    SemanticResponse,
    TerminalMetadata,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResult,
    WireProtocol,
)
from router_maestro.protocols.openai_common import (
    decode_reject,
    decode_tool_choice,
    decode_usage,
    encode_arguments,
    encode_stream_usage,
    encode_tool_choice,
    encode_usage,
    has_typed_block,
    optional_bool,
    optional_int,
    optional_number,
    optional_string,
    parse_arguments,
    reject,
    reject_unknown_keys,
    require_list,
    require_mapping,
    require_string,
    terminal_event_values,
    terminal_outcome_from_event,
    thaw_json,
)
from router_maestro.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    resolve_terminal_outcome,
)

_PROTOCOL = WireProtocol.OPENAI_CHAT
_NON_SUCCESS_TERMINAL_ERRORS = {
    "failed": ("upstream_error", "Upstream response failed"),
    "cancelled": ("upstream_cancelled", "Upstream response was cancelled"),
    "unknown": ("upstream_status_unknown", "Upstream response ended with an unknown status"),
}
_NON_SUCCESS_TRANSPORT_STATUSES = {
    "exception": "failed",
    "client_cancelled": "cancelled",
    "unexpected_eof": "unknown",
}
_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "stream",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "thinking",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "user",
        "metadata",
        "service_tier",
        "response_format",
        # This controls the ingress Chat stream envelope rather than model
        # generation. Identity attempts preserve it verbatim; cross-protocol
        # decoding validates it here and consumes it at the route encoder.
        "stream_options",
    }
)


def _terminal_status_error(terminal: TerminalMetadata) -> tuple[str, str] | None:
    """Return a safe Chat error when a terminal cannot mean successful completion."""
    status = terminal.response_status
    if status is None:
        status = _NON_SUCCESS_TRANSPORT_STATUSES.get(terminal.transport_termination or "")
    if status in _NON_SUCCESS_TERMINAL_ERRORS:
        return _NON_SUCCESS_TERMINAL_ERRORS[status]
    if status not in {None, "completed", "incomplete"}:
        reject(_PROTOCOL, "response.terminal.response_status", f"unsupported value {status!r}")
    return None


class OpenAIChatRuntime:
    """Strict Chat wire codec used only when semantic conversion is needed."""

    protocol = _PROTOCOL

    def __init__(
        self,
        *,
        origin_provider: str | None = None,
        origin_binding: str | None = None,
        default_model: str | None = None,
        allow_reasoning_opaque: bool = False,
    ) -> None:
        self.origin_provider = origin_provider
        self.origin_binding = origin_binding
        self.default_model = default_model
        self.allow_reasoning_opaque = allow_reasoning_opaque
        self._stream_decoder: ContextVar[OpenAIChatStreamDecoder | None] = ContextVar(
            f"openai_chat_stream_decoder_{id(self)}",
            default=None,
        )
        self._stream_encoder: ContextVar[OpenAIChatStreamEncoder | None] = ContextVar(
            f"openai_chat_stream_encoder_{id(self)}",
            default=None,
        )

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        messages = payload.get("messages")
        return RequestManifest(
            protocol=self.protocol,
            model=payload.get("model") if isinstance(payload.get("model"), str) else None,
            stream=payload.get("stream") is True,
            tools=bool(payload.get("tools")),
            images=has_typed_block(messages, {"image_url", "input_image"}),
            files=has_typed_block(messages, {"file", "input_file"}),
            reasoning=bool(payload.get("reasoning_effort") or payload.get("thinking")),
            parallel_tools=payload.get("parallel_tool_calls") is True,
        )

    async def decode_request(self, payload: Mapping[str, Any]) -> SemanticRequest:
        return _decode_request(payload)

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        return _encode_request(
            request,
            target_provider=self.origin_provider,
            target_binding=self.origin_binding,
            allow_reasoning_opaque=self.allow_reasoning_opaque,
        )

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        return _decode_response(
            payload,
            origin_provider=self.origin_provider,
            origin_binding=self.origin_binding,
            allow_reasoning_opaque=self.allow_reasoning_opaque,
        )

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        return _encode_response(response)

    def new_stream_decoder(self, *, sequence_start: int = 0) -> OpenAIChatStreamDecoder:
        return OpenAIChatStreamDecoder(
            origin_provider=self.origin_provider,
            origin_binding=self.origin_binding,
            default_model=self.default_model,
            sequence_start=sequence_start,
            allow_reasoning_opaque=self.allow_reasoning_opaque,
        )

    def new_stream_encoder(
        self,
        *,
        model: str | None = None,
        response_id: str | None = None,
    ) -> OpenAIChatStreamEncoder:
        return OpenAIChatStreamEncoder(model=model, response_id=response_id)

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        decoder = self._stream_decoder.get()
        payload_id = payload.get("id")
        if decoder is None or (
            decoder.terminal and isinstance(payload_id, str) and payload_id != decoder.response_id
        ):
            decoder = self.new_stream_decoder()
            self._stream_decoder.set(decoder)
        return decoder.decode(payload)

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        encoder = self._stream_encoder.get()
        if encoder is None or (
            encoder.terminal and event.type is SemanticEventType.RESPONSE_STARTED
        ):
            model = event.metadata.get("model")
            encoder = self.new_stream_encoder(
                model=model if isinstance(model, str) else None,
                response_id=event.response_id,
            )
            self._stream_encoder.set(encoder)
        return encoder.encode(event)


class OpenAIChatStreamDecoder:
    """Stateful decoder for one Chat Completions chunk sequence."""

    def __init__(
        self,
        *,
        origin_provider: str | None = None,
        origin_binding: str | None = None,
        default_model: str | None = None,
        sequence_start: int = 0,
        allow_reasoning_opaque: bool = False,
    ) -> None:
        self.origin_provider = origin_provider
        self.origin_binding = origin_binding
        self.allow_reasoning_opaque = allow_reasoning_opaque
        self._sequence = sequence_start
        self._started = False
        self._terminal = False
        self._allow_post_terminal_usage = False
        self._response_id: str | None = None
        self._model = default_model
        self._allow_missing_identity = default_model is not None
        self._tool_calls: dict[int, dict[str, str]] = {}

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def response_id(self) -> str | None:
        return self._response_id

    def decode(self, payload: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        frame = require_mapping(payload, protocol=_PROTOCOL, parameter="stream")
        reject_unknown_keys(
            frame,
            frozenset(
                {
                    "id",
                    "object",
                    "created",
                    "model",
                    "choices",
                    "usage",
                    "system_fingerprint",
                    "service_tier",
                    "error",
                }
            ),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        if self._terminal:
            return self._decode_post_terminal_usage(frame)
        if frame.get("error") is not None:
            return self._decode_error(frame)

        response_id, model = self._decode_identity(frame)

        specs: list[tuple[SemanticEventType, dict[str, Any]]] = []
        if not self._started:
            self._started = True
            metadata: dict[str, Any] = {}
            if model is not None:
                metadata["model"] = model
            if "object" in frame:
                metadata["object"] = require_string(
                    frame.get("object"), protocol=_PROTOCOL, parameter="stream.object"
                )
            if "created" in frame:
                metadata["created"] = optional_int(
                    frame.get("created"), protocol=_PROTOCOL, parameter="stream.created"
                )
            specs.append((SemanticEventType.RESPONSE_STARTED, {"metadata": metadata}))

        choices = require_list(
            frame.get("choices", []), protocol=_PROTOCOL, parameter="stream.choices"
        )
        if len(choices) > 1:
            decode_reject(_PROTOCOL, "stream.choices", "at most one choice is supported")
        terminal: TerminalMetadata | None = None
        if choices:
            choice = require_mapping(choices[0], protocol=_PROTOCOL, parameter="stream.choices[0]")
            reject_unknown_keys(
                choice,
                frozenset({"index", "delta", "finish_reason", "logprobs"}),
                protocol=_PROTOCOL,
                parameter="stream.choices[0]",
            )
            if (
                optional_int(
                    choice.get("index", 0),
                    protocol=_PROTOCOL,
                    parameter="stream.choices[0].index",
                )
                != 0
            ):
                decode_reject(_PROTOCOL, "stream.choices[0].index", "must be zero")
            if choice.get("logprobs") is not None:
                decode_reject(_PROTOCOL, "stream.choices[0].logprobs", "is not modeled")
            specs.extend(self._decode_delta(choice.get("delta", {})))
            finish = optional_string(
                choice.get("finish_reason"),
                protocol=_PROTOCOL,
                parameter="stream.choices[0].finish_reason",
            )
            if finish is not None:
                status = "incomplete" if finish in {"length", "content_filter"} else "completed"
                terminal = TerminalMetadata(
                    finish_reason=finish,
                    response_status=status,
                    transport_termination="explicit_terminal",
                    incomplete_details=(
                        {"reason": "max_output_tokens"}
                        if finish == "length"
                        else {"reason": "content_filter"}
                        if finish == "content_filter"
                        else None
                    ),
                )

        if frame.get("usage") is not None:
            specs.append(
                (
                    SemanticEventType.USAGE,
                    {
                        "usage": decode_usage(
                            frame["usage"],
                            protocol=_PROTOCOL,
                            input_field="prompt_tokens",
                            output_field="completion_tokens",
                            input_details_field="prompt_tokens_details",
                            output_details_field="completion_tokens_details",
                            top_level_reasoning_field="reasoning_tokens",
                        )
                    },
                )
            )
        if terminal is not None:
            specs.append((SemanticEventType.TERMINAL, {"terminal": terminal}))
            self._terminal = True
            # OpenAI's include_usage contract permits one usage-only tail only
            # when the finish chunk did not already carry the final snapshot.
            self._allow_post_terminal_usage = frame.get("usage") is None
        return self._events(specs)

    def _decode_identity(self, frame: Mapping[str, Any]) -> tuple[str | None, str | None]:
        if "id" not in frame and self._allow_missing_identity:
            response_id = self._response_id
        else:
            response_id = require_string(
                frame.get("id"),
                protocol=_PROTOCOL,
                parameter="stream.id",
                allow_empty=False,
            )
        if "model" not in frame and self._allow_missing_identity:
            model = self._model
        else:
            model = require_string(
                frame.get("model"),
                protocol=_PROTOCOL,
                parameter="stream.model",
                allow_empty=False,
            )
        if response_id is not None:
            if self._response_id is not None and response_id != self._response_id:
                decode_reject(_PROTOCOL, "stream.id", "response ID changed within stream")
            self._response_id = response_id
        if model is not None:
            if self._model is not None and model != self._model:
                decode_reject(_PROTOCOL, "stream.model", "model changed within stream")
            self._model = model
        return response_id, model

    def _decode_post_terminal_usage(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        if (
            not self._allow_post_terminal_usage
            or frame.get("choices") != []
            or not isinstance(frame.get("usage"), Mapping)
            or "error" in frame
        ):
            decode_reject(_PROTOCOL, "stream", "frame arrived after terminal event")
        self._decode_identity(frame)
        usage = decode_usage(
            frame["usage"],
            protocol=_PROTOCOL,
            input_field="prompt_tokens",
            output_field="completion_tokens",
            input_details_field="prompt_tokens_details",
            output_details_field="completion_tokens_details",
            top_level_reasoning_field="reasoning_tokens",
        )
        if usage is None:  # pragma: no cover - the mapping guard makes this unreachable
            decode_reject(_PROTOCOL, "stream.usage", "usage-only frame requires usage")
        self._allow_post_terminal_usage = False
        return self._events([(SemanticEventType.USAGE, {"usage": usage})])

    def finish_eof(self) -> tuple[SemanticEvent, ...]:
        if self._terminal:
            self._allow_post_terminal_usage = False
            return ()
        terminal = TerminalMetadata(
            error_code="unexpected_eof",
            error_message="Upstream stream ended before a finish_reason",
            response_status="unknown",
            transport_termination="unexpected_eof",
        )
        self._terminal = True
        return self._events(
            [
                (SemanticEventType.ERROR, {"terminal": terminal}),
                (SemanticEventType.TERMINAL, {"terminal": terminal}),
            ]
        )

    def _decode_delta(
        self,
        value: object,
    ) -> list[tuple[SemanticEventType, dict[str, Any]]]:
        delta = require_mapping(value, protocol=_PROTOCOL, parameter="stream.choices[0].delta")
        reject_unknown_keys(
            delta,
            frozenset(
                {
                    "role",
                    "content",
                    "refusal",
                    "reasoning",
                    "reasoning_content",
                    "reasoning_text",
                    "tool_calls",
                    "thinking_id",
                    "thinking_signature",
                    "reasoning_opaque",
                }
            ),
            protocol=_PROTOCOL,
            parameter="stream.choices[0].delta",
        )
        role = delta.get("role")
        if role not in {None, "assistant"}:
            decode_reject(_PROTOCOL, "stream.choices[0].delta.role", "must be assistant")
        specs: list[tuple[SemanticEventType, dict[str, Any]]] = []
        reasoning_fields = {
            key: delta.get(key)
            for key in ("reasoning_content", "reasoning", "reasoning_text")
            if delta.get(key) is not None
        }
        if len(reasoning_fields) > 1:
            decode_reject(
                _PROTOCOL,
                "stream.choices[0].delta",
                "reasoning fields conflict",
            )
        reasoning = next(iter(reasoning_fields.values()), None)
        if reasoning is not None:
            specs.append(
                (
                    SemanticEventType.REASONING_DELTA,
                    {
                        "delta": require_string(
                            reasoning,
                            protocol=_PROTOCOL,
                            parameter="stream.choices[0].delta.reasoning_content",
                        )
                    },
                )
            )
        content = optional_string(
            delta.get("content"),
            protocol=_PROTOCOL,
            parameter="stream.choices[0].delta.content",
        )
        if content:
            specs.append((SemanticEventType.TEXT_DELTA, {"delta": content}))
        refusal = optional_string(
            delta.get("refusal"),
            protocol=_PROTOCOL,
            parameter="stream.choices[0].delta.refusal",
        )
        if refusal:
            specs.append((SemanticEventType.OUTPUT_ITEM, {"item": RefusalContent(refusal)}))
        signature = optional_string(
            delta.get("thinking_signature"),
            protocol=_PROTOCOL,
            parameter="stream.choices[0].delta.thinking_signature",
        )
        reasoning_opaque = optional_string(
            delta.get("reasoning_opaque"),
            protocol=_PROTOCOL,
            parameter="stream.choices[0].delta.reasoning_opaque",
        )
        if reasoning_opaque is not None:
            if not self.allow_reasoning_opaque:
                decode_reject(
                    _PROTOCOL,
                    "stream.choices[0].delta.reasoning_opaque",
                    "provider-private reasoning is unavailable on this binding",
                )
            if signature is not None:
                decode_reject(
                    _PROTOCOL,
                    "stream.choices[0].delta",
                    "reasoning_opaque conflicts with thinking_signature",
                )
            signature = reasoning_opaque
        thinking_id = optional_string(
            delta.get("thinking_id"),
            protocol=_PROTOCOL,
            parameter="stream.choices[0].delta.thinking_id",
        )
        if reasoning_opaque is not None:
            thinking_id = _opaque_reasoning_item_id(reasoning_opaque)
        if signature is not None:
            if thinking_id is None or self._model is None or self.origin_provider is None:
                decode_reject(
                    _PROTOCOL,
                    "stream.choices[0].delta.thinking_signature",
                    "opaque reasoning requires thinking_id, model, and origin_provider",
                )
            specs.append(
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {
                        "item_id": thinking_id,
                        "item": ReasoningSummary(
                            "",
                            opaque_state=OpaqueState(
                                origin_protocol=_PROTOCOL,
                                origin_provider=self.origin_provider,
                                origin_model=self._model,
                                item_id=thinking_id,
                                blob=signature,
                                origin_binding=self.origin_binding,
                            ),
                        ),
                        "metadata": {
                            "output_item_type": "reasoning",
                            "output_item_done": True,
                        },
                    },
                )
            )
        elif thinking_id is not None:
            specs.append(
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {
                        "item_id": thinking_id,
                        "metadata": {
                            "output_item_type": "reasoning",
                            "output_item_done": True,
                        },
                    },
                )
            )
        tool_calls = delta.get("tool_calls")
        if tool_calls is not None:
            calls = require_list(
                tool_calls,
                protocol=_PROTOCOL,
                parameter="stream.choices[0].delta.tool_calls",
            )
            for index, raw_call in enumerate(calls):
                path = f"stream.choices[0].delta.tool_calls[{index}]"
                call = require_mapping(raw_call, protocol=_PROTOCOL, parameter=path)
                reject_unknown_keys(
                    call,
                    frozenset({"index", "id", "type", "function"}),
                    protocol=_PROTOCOL,
                    parameter=path,
                )
                call_index = optional_int(
                    call.get("index", index),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.index",
                )
                if call_index is None:  # pragma: no cover - default is the loop index
                    decode_reject(_PROTOCOL, f"{path}.index", "tool index is required")
                function = require_mapping(
                    call.get("function", {}),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.function",
                )
                reject_unknown_keys(
                    function,
                    frozenset({"name", "arguments"}),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.function",
                )
                metadata: dict[str, Any] = {}
                call_id = optional_string(
                    call.get("id"), protocol=_PROTOCOL, parameter=f"{path}.id"
                )
                name = optional_string(
                    function.get("name"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.function.name",
                )
                namespace = None
                if name is not None:
                    namespaced = decode_namespaced_tool_name(name)
                    if namespaced is not None:
                        namespace, name = namespaced
                state = self._tool_calls.setdefault(call_index, {})
                if call_id is not None:
                    state["call_id"] = call_id
                if name is not None:
                    state["name"] = name
                if namespace is not None:
                    state["namespace"] = namespace
                metadata.update(state)
                metadata["tool_index"] = call_index
                specs.append(
                    (
                        SemanticEventType.TOOL_ARGUMENTS_DELTA,
                        {
                            "item_id": state.get("call_id"),
                            "delta": optional_string(
                                function.get("arguments"),
                                protocol=_PROTOCOL,
                                parameter=f"{path}.function.arguments",
                            )
                            or "",
                            "metadata": metadata,
                        },
                    )
                )
        return specs

    def _decode_error(self, frame: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        error = require_mapping(frame.get("error"), protocol=_PROTOCOL, parameter="stream.error")
        code = error.get("code", error.get("type", "upstream_error"))
        terminal = TerminalMetadata(
            error_code=require_string(code, protocol=_PROTOCOL, parameter="stream.error.code"),
            error_message=require_string(
                error.get("message"),
                protocol=_PROTOCOL,
                parameter="stream.error.message",
            ),
            response_status="failed",
            transport_termination="explicit_terminal",
        )
        self._terminal = True
        return self._events(
            [
                (SemanticEventType.ERROR, {"terminal": terminal}),
                (SemanticEventType.TERMINAL, {"terminal": terminal}),
            ]
        )

    def _events(
        self,
        specs: list[tuple[SemanticEventType, dict[str, Any]]],
    ) -> tuple[SemanticEvent, ...]:
        events = []
        for event_type, values in specs:
            events.append(
                SemanticEvent(
                    type=event_type,
                    sequence=self._sequence,
                    response_id=self._response_id,
                    **values,
                )
            )
            self._sequence += 1
        return tuple(events)


class OpenAIChatStreamEncoder:
    """Stateful encoder for one Chat Completions chunk sequence."""

    def __init__(
        self,
        *,
        model: str | None = None,
        response_id: str | None = None,
    ) -> None:
        self.model = model
        self.response_id = response_id
        self._created = int(time.time())
        self._started = False
        self._terminal = False
        self._pending_error: TerminalMetadata | None = None

    @property
    def terminal(self) -> bool:
        return self._terminal

    def encode(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        if self._terminal:
            reject(_PROTOCOL, "event.type", "event arrived after terminal event")
        self._capture_identity(event)
        if event.type is SemanticEventType.RESPONSE_STARTED:
            return tuple(self._ensure_started())
        if event.type is SemanticEventType.ERROR:
            self._pending_error = event.terminal or TerminalMetadata(
                error_code="upstream_error",
                error_message="Upstream stream failed",
                response_status="failed",
            )
            return ()
        if event.type is SemanticEventType.TERMINAL:
            return self._encode_terminal(event)
        frames = self._ensure_started()
        if event.type is SemanticEventType.USAGE:
            if event.usage is None:
                reject(_PROTOCOL, "event.usage", "usage event requires Usage")
            frames.append(
                self._frame(
                    choices=[],
                    usage=encode_stream_usage(
                        event.usage,
                        protocol=_PROTOCOL,
                        input_field="prompt_tokens",
                        output_field="completion_tokens",
                        input_details_field="prompt_tokens_details",
                        output_details_field="completion_tokens_details",
                    ),
                )
            )
            return tuple(frames)
        for delta in self._event_deltas(event):
            frames.append(
                self._frame(
                    choices=[
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ]
                )
            )
        return tuple(frames)

    def finish_eof(self) -> tuple[Mapping[str, Any], ...]:
        if self._terminal:
            return ()
        self._terminal = True
        return (
            {
                "error": {
                    "code": "unexpected_eof",
                    "message": "Semantic event stream ended before terminal event",
                }
            },
        )

    def _capture_identity(self, event: SemanticEvent) -> None:
        metadata_model = event.metadata.get("model")
        if isinstance(metadata_model, str):
            if self.model is not None and self.model != metadata_model:
                reject(_PROTOCOL, "event.metadata.model", "model changed within stream")
            self.model = metadata_model
        if event.response_id is not None:
            if self.response_id is not None and self.response_id != event.response_id:
                reject(_PROTOCOL, "event.response_id", "response ID changed within stream")
            self.response_id = event.response_id

    def _ensure_started(self) -> list[Mapping[str, Any]]:
        if self._started:
            return []
        self._started = True
        return [
            self._frame(
                choices=[
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ]
            )
        ]

    def _event_deltas(self, event: SemanticEvent) -> list[dict[str, Any]]:
        if event.type is SemanticEventType.TEXT_DELTA:
            return [{"content": event.delta or ""}]
        if event.type is SemanticEventType.REASONING_DELTA:
            return [{"reasoning_content": event.delta or ""}]
        if event.type is SemanticEventType.TOOL_ARGUMENTS_DELTA:
            function: dict[str, Any] = {"arguments": event.delta or ""}
            name = event.metadata.get("name")
            if isinstance(name, str):
                function["name"] = name
            call: dict[str, Any] = {
                "index": event.output_index or 0,
                "function": function,
            }
            call_id = event.item_id or event.metadata.get("call_id")
            if isinstance(call_id, str):
                call.update({"id": call_id, "type": "function"})
            return [{"tool_calls": [call]}]
        if event.type is not SemanticEventType.OUTPUT_ITEM:
            reject(_PROTOCOL, "event.type", f"unsupported event {event.type.value!r}")
        item = event.item
        if item is None:
            return []
        if isinstance(item, TextContent):
            return [{"content": item.text}]
        if isinstance(item, RefusalContent):
            return [{"refusal": item.refusal}]
        if isinstance(item, ReasoningSummary):
            delta: dict[str, Any] = {"reasoning_content": item.text}
            state = item.opaque_state
            if state is not None:
                if state.origin_protocol is not _PROTOCOL or not isinstance(state.blob, str):
                    reject(
                        _PROTOCOL,
                        "event.item.opaque_state",
                        "Chat stream requires Chat-origin text opaque state",
                    )
                delta["thinking_id"] = state.item_id
                delta["thinking_signature"] = state.blob
            return [delta]
        if isinstance(item, ToolCall):
            call = _encode_tool_call(item, parameter="event.item")
            call["index"] = event.output_index or 0
            return [{"tool_calls": [call]}]
        if isinstance(item, SemanticMessage):
            if item.role is not MessageRole.ASSISTANT:
                reject(_PROTOCOL, "event.item.role", "must be assistant")
            deltas = []
            for part in item.content:
                deltas.extend(self._event_deltas(replace(event, item=part)))
            return deltas
        reject(_PROTOCOL, "event.item", f"unsupported stream item {type(item).__name__}")

    def _encode_terminal(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        terminal = (
            self._pending_error
            or event.terminal
            or TerminalMetadata(
                finish_reason="stop",
                response_status="completed",
            )
        )
        self._terminal = True
        status_error = _terminal_status_error(terminal)
        if terminal.error_code is not None or terminal.error_message is not None or status_error:
            synthesized_code, synthesized_message = status_error or (
                "upstream_error",
                "Upstream stream failed",
            )
            return (
                {
                    "error": {
                        "code": terminal.error_code or synthesized_code,
                        "message": terminal.error_message or synthesized_message,
                    }
                },
            )
        frames = self._ensure_started()
        frames.append(
            self._frame(
                choices=[
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": terminal.finish_reason or "stop",
                    }
                ]
            )
        )
        return tuple(frames)

    def _frame(
        self,
        *,
        choices: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.response_id:
            reject(_PROTOCOL, "event.response_id", "Chat stream requires a response ID")
        if not self.model:
            reject(_PROTOCOL, "event.metadata.model", "Chat stream requires a model")
        payload: dict[str, Any] = {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self.model,
            "choices": choices,
        }
        if usage is not None:
            payload["usage"] = usage
        return payload


def _opaque_reasoning_item_id(value: str) -> str:
    return f"chat_rs_{sha256(value.encode()).hexdigest()[:24]}"


def _decode_request(payload: Mapping[str, Any]) -> SemanticRequest:
    body = require_mapping(payload, protocol=_PROTOCOL, parameter="request")
    reject_unknown_keys(body, _REQUEST_FIELDS, protocol=_PROTOCOL, parameter="")
    _validate_stream_options(body.get("stream_options"))
    model = require_string(
        body.get("model"), protocol=_PROTOCOL, parameter="model", allow_empty=False
    )
    messages = require_list(body.get("messages"), protocol=_PROTOCOL, parameter="messages")
    semantic_messages = tuple(
        _decode_message(item, parameter=f"messages[{index}]") for index, item in enumerate(messages)
    )
    stop = _decode_stop(body.get("stop"))
    reasoning = _decode_reasoning(body)
    metadata = body.get("metadata") or {}
    metadata = require_mapping(metadata, protocol=_PROTOCOL, parameter="metadata")
    structured_output = body.get("response_format")
    if structured_output is not None:
        structured_output = require_mapping(
            structured_output,
            protocol=_PROTOCOL,
            parameter="response_format",
        )
    return SemanticRequest(
        model=model,
        input=semantic_messages,
        tools=_decode_tools(body.get("tools")),
        stream=optional_bool(body.get("stream", False), protocol=_PROTOCOL, parameter="stream")
        or False,
        max_output_tokens=optional_int(
            body.get("max_tokens"), protocol=_PROTOCOL, parameter="max_tokens"
        ),
        temperature=optional_number(
            body.get("temperature"), protocol=_PROTOCOL, parameter="temperature"
        ),
        top_p=optional_number(body.get("top_p"), protocol=_PROTOCOL, parameter="top_p"),
        frequency_penalty=optional_number(
            body.get("frequency_penalty"),
            protocol=_PROTOCOL,
            parameter="frequency_penalty",
        ),
        presence_penalty=optional_number(
            body.get("presence_penalty"),
            protocol=_PROTOCOL,
            parameter="presence_penalty",
        ),
        stop_sequences=stop,
        tool_choice=decode_tool_choice(
            body.get("tool_choice"), protocol=_PROTOCOL, nested_function=True
        ),
        parallel_tool_calls=optional_bool(
            body.get("parallel_tool_calls"),
            protocol=_PROTOCOL,
            parameter="parallel_tool_calls",
        ),
        reasoning=reasoning,
        structured_output=structured_output,
        user=optional_string(body.get("user"), protocol=_PROTOCOL, parameter="user"),
        service_tier=optional_string(
            body.get("service_tier"), protocol=_PROTOCOL, parameter="service_tier"
        ),
        metadata=metadata,
        explicit_fields=frozenset(body) - {"stream_options"},
    )


def _validate_stream_options(value: object) -> None:
    if value is None:
        return
    options = require_mapping(value, protocol=_PROTOCOL, parameter="stream_options")
    reject_unknown_keys(
        options,
        frozenset({"include_usage"}),
        protocol=_PROTOCOL,
        parameter="stream_options",
    )
    optional_bool(
        options.get("include_usage", False),
        protocol=_PROTOCOL,
        parameter="stream_options.include_usage",
    )


def _decode_stop(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    items = require_list(value, protocol=_PROTOCOL, parameter="stop")
    return tuple(
        require_string(item, protocol=_PROTOCOL, parameter=f"stop[{index}]")
        for index, item in enumerate(items)
    )


def _decode_reasoning(body: Mapping[str, Any]) -> ReasoningConfig | None:
    effort = optional_string(
        body.get("reasoning_effort"),
        protocol=_PROTOCOL,
        parameter="reasoning_effort",
    )
    thinking = body.get("thinking")
    if thinking is None:
        return ReasoningConfig(effort=effort) if effort is not None else None
    config = require_mapping(thinking, protocol=_PROTOCOL, parameter="thinking")
    reject_unknown_keys(
        config,
        frozenset({"type", "budget_tokens"}),
        protocol=_PROTOCOL,
        parameter="thinking",
    )
    thinking_type = require_string(
        config.get("type"), protocol=_PROTOCOL, parameter="thinking.type"
    )
    if thinking_type not in {"enabled", "adaptive", "disabled"}:
        decode_reject(_PROTOCOL, "thinking.type", f"unsupported type {thinking_type!r}")
    budget = optional_int(
        config.get("budget_tokens"),
        protocol=_PROTOCOL,
        parameter="thinking.budget_tokens",
    )
    return ReasoningConfig(
        enabled=thinking_type != "disabled",
        effort=effort,
        budget_tokens=budget,
    )


def _decode_message(
    value: object,
    *,
    parameter: str,
    origin_provider: str | None = None,
    origin_binding: str | None = None,
    origin_model: str | None = None,
    allow_reasoning_opaque: bool = False,
) -> SemanticMessage:
    message = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        message,
        frozenset(
            {
                "role",
                "content",
                "name",
                "tool_call_id",
                "tool_calls",
                "refusal",
                "reasoning",
                "reasoning_content",
                "reasoning_text",
                "thinking_id",
                "thinking_signature",
                "reasoning_opaque",
            }
        ),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    role_value = require_string(
        message.get("role"), protocol=_PROTOCOL, parameter=f"{parameter}.role"
    )
    try:
        role = MessageRole(role_value)
    except ValueError:
        decode_reject(_PROTOCOL, f"{parameter}.role", f"unsupported role {role_value!r}")
    name = optional_string(message.get("name"), protocol=_PROTOCOL, parameter=f"{parameter}.name")

    if role is MessageRole.TOOL:
        call_id = require_string(
            message.get("tool_call_id"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.tool_call_id",
            allow_empty=False,
        )
        if message.get("tool_calls") is not None or message.get("refusal") is not None:
            decode_reject(_PROTOCOL, parameter, "tool messages cannot contain calls or refusal")
        try:
            output, is_error = unproject_tool_result_output(message.get("content"))
        except ToolResultProjectionError as exc:
            decode_reject(_PROTOCOL, f"{parameter}.content", str(exc))
        content = _decode_content(output, parameter=f"{parameter}.content")
        return SemanticMessage(
            role=role,
            name=name,
            content=(ToolResult(call_id=call_id, content=content, is_error=is_error),),
        )

    content = _decode_content(message.get("content"), parameter=f"{parameter}.content")
    if message.get("tool_call_id") is not None:
        decode_reject(
            _PROTOCOL,
            f"{parameter}.tool_call_id",
            "is only valid for tool messages",
        )
    parts: list[Any] = list(content)
    refusal = optional_string(
        message.get("refusal"), protocol=_PROTOCOL, parameter=f"{parameter}.refusal"
    )
    if refusal is not None:
        if role is not MessageRole.ASSISTANT:
            decode_reject(_PROTOCOL, f"{parameter}.refusal", "is only valid for assistant messages")
        parts.append(RefusalContent(refusal))
    reasoning_fields = {
        key: message.get(key)
        for key in ("reasoning_content", "reasoning", "reasoning_text")
        if message.get(key) is not None
    }
    if len(reasoning_fields) > 1:
        decode_reject(_PROTOCOL, parameter, "reasoning fields conflict")
    reasoning = next(iter(reasoning_fields.values()), None)
    thinking_signature = optional_string(
        message.get("thinking_signature"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.thinking_signature",
    )
    reasoning_opaque = optional_string(
        message.get("reasoning_opaque"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.reasoning_opaque",
    )
    if reasoning_opaque is not None:
        if not allow_reasoning_opaque:
            decode_reject(
                _PROTOCOL,
                f"{parameter}.reasoning_opaque",
                "provider-private reasoning is unavailable on this binding",
            )
        if thinking_signature is not None:
            decode_reject(
                _PROTOCOL,
                parameter,
                "reasoning_opaque conflicts with thinking_signature",
            )
        thinking_signature = reasoning_opaque
    thinking_id = optional_string(
        message.get("thinking_id"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.thinking_id",
    )
    if reasoning_opaque is not None:
        thinking_id = _opaque_reasoning_item_id(reasoning_opaque)
    opaque_state = None
    if thinking_signature is not None:
        if (
            role is not MessageRole.ASSISTANT
            or thinking_id is None
            or origin_provider is None
            or origin_model is None
        ):
            decode_reject(
                _PROTOCOL,
                f"{parameter}.thinking_signature",
                "opaque reasoning requires assistant role and provider provenance",
            )
        opaque_state = OpaqueState(
            origin_protocol=_PROTOCOL,
            origin_provider=origin_provider,
            origin_model=origin_model,
            item_id=thinking_id,
            blob=thinking_signature,
            origin_binding=origin_binding,
        )
    elif thinking_id is not None:
        decode_reject(
            _PROTOCOL,
            f"{parameter}.thinking_id",
            "requires thinking_signature",
        )
    if reasoning is not None or opaque_state is not None:
        if role is not MessageRole.ASSISTANT:
            decode_reject(_PROTOCOL, f"{parameter}.reasoning_content", "requires assistant role")
        parts.append(
            ReasoningSummary(
                (
                    require_string(
                        reasoning,
                        protocol=_PROTOCOL,
                        parameter=f"{parameter}.reasoning_content",
                    )
                    if reasoning is not None
                    else ""
                ),
                opaque_state=opaque_state,
            )
        )
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if role is not MessageRole.ASSISTANT:
            decode_reject(_PROTOCOL, f"{parameter}.tool_calls", "requires assistant role")
        calls = require_list(tool_calls, protocol=_PROTOCOL, parameter=f"{parameter}.tool_calls")
        parts.extend(
            _decode_tool_call(item, parameter=f"{parameter}.tool_calls[{index}]")
            for index, item in enumerate(calls)
        )
    return SemanticMessage(role=role, name=name, content=tuple(parts))


def _decode_content(value: object, *, parameter: str) -> tuple[TextContent | ImageContent, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (TextContent(value),)
    blocks = require_list(value, protocol=_PROTOCOL, parameter=parameter)
    decoded: list[TextContent | ImageContent] = []
    for index, raw_block in enumerate(blocks):
        path = f"{parameter}[{index}]"
        block = require_mapping(raw_block, protocol=_PROTOCOL, parameter=path)
        block_type = require_string(block.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type")
        if block_type == "text":
            reject_unknown_keys(
                block,
                frozenset({"type", "text"}),
                protocol=_PROTOCOL,
                parameter=path,
            )
            decoded.append(
                TextContent(
                    require_string(block.get("text"), protocol=_PROTOCOL, parameter=f"{path}.text")
                )
            )
            continue
        if block_type == "image_url":
            reject_unknown_keys(
                block,
                frozenset({"type", "image_url"}),
                protocol=_PROTOCOL,
                parameter=path,
            )
            image = block.get("image_url")
            if isinstance(image, str):
                decoded.append(ImageContent(image))
                continue
            image_object = require_mapping(image, protocol=_PROTOCOL, parameter=f"{path}.image_url")
            reject_unknown_keys(
                image_object,
                frozenset({"url", "detail"}),
                protocol=_PROTOCOL,
                parameter=f"{path}.image_url",
            )
            decoded.append(
                ImageContent(
                    source=require_string(
                        image_object.get("url"),
                        protocol=_PROTOCOL,
                        parameter=f"{path}.image_url.url",
                    ),
                    detail=optional_string(
                        image_object.get("detail"),
                        protocol=_PROTOCOL,
                        parameter=f"{path}.image_url.detail",
                    ),
                )
            )
            continue
        decode_reject(_PROTOCOL, f"{path}.type", f"unsupported content type {block_type!r}")
    return tuple(decoded)


def _decode_tool_call(value: object, *, parameter: str) -> ToolCall:
    call = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        call,
        frozenset({"id", "type", "function"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    call_type = require_string(
        call.get("type", "function"), protocol=_PROTOCOL, parameter=f"{parameter}.type"
    )
    if call_type != "function":
        decode_reject(_PROTOCOL, f"{parameter}.type", "only function calls are supported")
    function = require_mapping(
        call.get("function"), protocol=_PROTOCOL, parameter=f"{parameter}.function"
    )
    reject_unknown_keys(
        function,
        frozenset({"name", "arguments"}),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.function",
    )
    raw_name = require_string(
        function.get("name"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.function.name",
        allow_empty=False,
    )
    namespaced = decode_namespaced_tool_name(raw_name)
    namespace, name = namespaced if namespaced is not None else (None, raw_name)
    return ToolCall(
        call_id=require_string(
            call.get("id"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.id",
            allow_empty=False,
        ),
        name=name,
        arguments=parse_arguments(
            function.get("arguments", "{}"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.function.arguments",
        ),
        namespace=namespace,
    )


def _decode_tools(value: object) -> tuple[ToolDefinition, ...]:
    if value is None:
        return ()
    tools = require_list(value, protocol=_PROTOCOL, parameter="tools")
    decoded = []
    for index, raw_tool in enumerate(tools):
        path = f"tools[{index}]"
        tool = require_mapping(raw_tool, protocol=_PROTOCOL, parameter=path)
        reject_unknown_keys(
            tool,
            frozenset({"type", "function"}),
            protocol=_PROTOCOL,
            parameter=path,
        )
        tool_type = require_string(tool.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type")
        if tool_type != "function":
            decode_reject(_PROTOCOL, f"{path}.type", "only function tools are supported")
        function = require_mapping(
            tool.get("function"), protocol=_PROTOCOL, parameter=f"{path}.function"
        )
        reject_unknown_keys(
            function,
            frozenset({"name", "description", "parameters", "strict"}),
            protocol=_PROTOCOL,
            parameter=f"{path}.function",
        )
        parameters = function.get("parameters", {})
        parameters = require_mapping(
            parameters, protocol=_PROTOCOL, parameter=f"{path}.function.parameters"
        )
        raw_name = require_string(
            function.get("name"),
            protocol=_PROTOCOL,
            parameter=f"{path}.function.name",
            allow_empty=False,
        )
        namespaced = decode_namespaced_tool_name(raw_name)
        namespace, name = namespaced if namespaced is not None else (None, raw_name)
        decoded.append(
            ToolDefinition(
                name=name,
                description=optional_string(
                    function.get("description"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.function.description",
                ),
                input_schema=parameters,
                strict=optional_bool(
                    function.get("strict"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.function.strict",
                ),
                namespace=namespace,
            )
        )
    return tuple(decoded)


def _encode_request(
    request: SemanticRequest,
    *,
    target_provider: str | None = None,
    target_binding: str | None = None,
    allow_reasoning_opaque: bool = False,
) -> dict[str, Any]:
    _reject_request_fields(request)
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            _encode_message(
                item,
                index=index,
                model=request.model,
                target_provider=target_provider,
                target_binding=target_binding,
                allow_reasoning_opaque=allow_reasoning_opaque,
            )
            for index, item in enumerate(request.input)
        ],
    }
    _put(payload, "stream", request.stream, request, source_name="stream", default=False)
    _put(payload, "temperature", request.temperature, request)
    _put(payload, "max_tokens", request.max_output_tokens, request, source_name="max_tokens")
    _put(payload, "top_p", request.top_p, request)
    _put(payload, "frequency_penalty", request.frequency_penalty, request)
    _put(payload, "presence_penalty", request.presence_penalty, request)
    _put(payload, "user", request.user, request)
    _put(payload, "service_tier", request.service_tier, request)
    if request.stop_sequences:
        payload["stop"] = list(request.stop_sequences)
    if request.metadata:
        payload["metadata"] = thaw_json(request.metadata)
    if request.tools:
        payload["tools"] = [_encode_tool(tool) for tool in request.tools]
    tool_choice_value = request.tool_choice
    if tool_choice_value is not None and tool_choice_value.namespace is not None:
        try:
            encoded_name = encode_namespaced_tool_name(
                tool_choice_value.namespace,
                tool_choice_value.name or "",
            )
        except ValueError as error:
            reject(_PROTOCOL, "tool_choice.namespace", str(error))
        tool_choice_value = replace(tool_choice_value, name=encoded_name, namespace=None)
    tool_choice = encode_tool_choice(
        tool_choice_value,
        protocol=_PROTOCOL,
        nested_function=True,
    )
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.reasoning is not None:
        if request.reasoning.effort is not None:
            payload["reasoning_effort"] = request.reasoning.effort
        if request.reasoning.enabled is not None or request.reasoning.budget_tokens is not None:
            thinking_type = "enabled" if request.reasoning.enabled is not False else "disabled"
            thinking: dict[str, Any] = {"type": thinking_type}
            if request.reasoning.budget_tokens is not None:
                thinking["budget_tokens"] = request.reasoning.budget_tokens
            payload["thinking"] = thinking
    if request.structured_output is not None:
        payload["response_format"] = thaw_json(request.structured_output)
    return payload


def _put(
    payload: dict[str, Any],
    name: str,
    value: object,
    request: SemanticRequest,
    *,
    source_name: str | None = None,
    default: object = None,
) -> None:
    if value != default or (source_name or name) in request.explicit_fields:
        payload[name] = value


def _reject_request_fields(request: SemanticRequest) -> None:
    if request.candidate_count not in {None, 1}:
        reject(
            _PROTOCOL,
            "candidate_count",
            "Chat Completions supports exactly one candidate",
        )
    for name, value in (
        ("top_k", request.top_k),
        ("response_mime_type", request.response_mime_type),
    ):
        if value is not None:
            reject(_PROTOCOL, name, "field is not supported by Chat Completions")
    if request.provider_extensions:
        key = sorted(request.provider_extensions)[0]
        reject(_PROTOCOL, key, "provider extension is not portable")


def _encode_message(
    item: object,
    *,
    index: int,
    model: str | None = None,
    target_provider: str | None = None,
    target_binding: str | None = None,
    allow_reasoning_opaque: bool = False,
) -> dict[str, Any]:
    path = f"input[{index}]"
    if isinstance(item, ToolCall):
        item = SemanticMessage(role=MessageRole.ASSISTANT, content=(item,))
    elif isinstance(item, ToolResult):
        item = SemanticMessage(role=MessageRole.TOOL, content=(item,))
    elif isinstance(item, TextContent | ImageContent):
        item = SemanticMessage(role=MessageRole.USER, content=(item,))
    elif isinstance(item, RefusalContent | ReasoningSummary):
        item = SemanticMessage(role=MessageRole.ASSISTANT, content=(item,))
    if not isinstance(item, SemanticMessage):
        reject(_PROTOCOL, path, f"{type(item).__name__} cannot be encoded as a chat message")
    # Responses message item IDs and lifecycle status belong to the container,
    # not its ordered role/content semantics. Chat has no corresponding fields,
    # so cross-protocol request projection intentionally consumes them here.
    if item.role is MessageRole.TOOL:
        return _encode_tool_result_message(item, parameter=path)

    text_blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    refusal = None
    reasoning_parts: list[str] = []
    reasoning_seen = False
    reasoning_opaque = None
    for part_index, part in enumerate(item.content):
        part_path = f"{path}.content[{part_index}]"
        if isinstance(part, TextContent):
            text_blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            if isinstance(part.source, bytes):
                reject(_PROTOCOL, part_path, "binary images require a data URL")
            source_kind = part.source_kind or "url"
            if source_kind in {"base64", "inline_data"}:
                if part.media_type is None:
                    reject(
                        _PROTOCOL,
                        f"{part_path}.media_type",
                        "base64 images require a media type",
                    )
                image_url = f"data:{part.media_type};base64,{part.source}"
            elif source_kind == "url":
                image_url = part.source
            else:
                reject(
                    _PROTOCOL,
                    f"{part_path}.source_kind",
                    f"unsupported kind {source_kind!r}",
                )
            image: dict[str, Any] = {"url": image_url}
            if part.detail is not None:
                image["detail"] = part.detail
            text_blocks.append({"type": "image_url", "image_url": image})
        elif isinstance(part, FileContent):
            reject(_PROTOCOL, part_path, "Chat Completions does not support file blocks")
        elif isinstance(part, RefusalContent):
            if item.role is not MessageRole.ASSISTANT or refusal is not None:
                reject(_PROTOCOL, part_path, "refusal requires one assistant content part")
            refusal = part.refusal
        elif isinstance(part, ReasoningSummary):
            if item.role is not MessageRole.ASSISTANT:
                reject(_PROTOCOL, part_path, "reasoning requires assistant role")
            reasoning_seen = True
            if part.opaque_state is not None:
                if reasoning_opaque is not None:
                    reject(_PROTOCOL, part_path, "Chat can carry only one opaque reasoning state")
                state = part.opaque_state
                if (
                    not allow_reasoning_opaque
                    or target_provider is None
                    or target_binding is None
                    or model is None
                    or state.origin_provider != target_provider
                    or state.origin_binding != target_binding
                    or state.origin_model != model
                    or state.origin_protocol is not _PROTOCOL
                    or not isinstance(state.blob, str)
                ):
                    reject(_PROTOCOL, part_path, "Chat cannot carry opaque reasoning state")
                reasoning_opaque = state.blob
            reasoning_parts.append(part.text)
        elif isinstance(part, ToolCall):
            if item.role is not MessageRole.ASSISTANT:
                reject(_PROTOCOL, part_path, "tool calls require assistant role")
            tool_calls.append(_encode_tool_call(part, parameter=part_path))
        elif isinstance(part, ToolResult):
            reject(_PROTOCOL, part_path, "tool results require tool role")
        else:  # pragma: no cover - closed semantic union
            reject(_PROTOCOL, part_path, f"unsupported content {type(part).__name__}")
    payload: dict[str, Any] = {"role": item.role.value}
    if item.name is not None:
        payload["name"] = item.name
    if len(text_blocks) == 1 and text_blocks[0]["type"] == "text":
        payload["content"] = text_blocks[0]["text"]
    elif text_blocks:
        payload["content"] = text_blocks
    else:
        payload["content"] = None
    if refusal is not None:
        payload["refusal"] = refusal
    if reasoning_seen:
        reasoning = "".join(reasoning_parts)
        if reasoning_opaque is not None:
            payload["reasoning_text"] = reasoning
            payload["reasoning_opaque"] = reasoning_opaque
        else:
            payload["reasoning_content"] = reasoning
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _encode_tool_result_message(message: SemanticMessage, *, parameter: str) -> dict[str, Any]:
    if len(message.content) != 1 or not isinstance(message.content[0], ToolResult):
        reject(_PROTOCOL, f"{parameter}.content", "tool role requires one ToolResult")
    result = message.content[0]
    if result.kind != "function":
        reject(
            _PROTOCOL,
            f"{parameter}.content.kind",
            f"unsupported tool result kind {result.kind!r}",
        )
    if result.item_id is not None:
        reject(_PROTOCOL, f"{parameter}.content.item_id", "Chat tool results lack item IDs")
    pieces = []
    for index, part in enumerate(result.content):
        if not isinstance(part, TextContent):
            reject(
                _PROTOCOL,
                f"{parameter}.content[{index}]",
                "Chat tool results support text only",
            )
        pieces.append(part.text)
    if result.structured_content is not None:
        if pieces:
            reject(
                _PROTOCOL,
                f"{parameter}.content",
                "cannot combine text and structured tool result content",
            )
        pieces.append(json.dumps(thaw_json(result.structured_content), ensure_ascii=False))
    payload: dict[str, Any] = {
        "role": "tool",
        "content": project_tool_result_output(
            "".join(pieces),
            is_error=result.is_error,
        ),
        "tool_call_id": result.call_id,
    }
    if message.name is not None:
        payload["name"] = message.name
    return payload


def _encode_tool_call(call: ToolCall, *, parameter: str) -> dict[str, Any]:
    if call.item_id is not None:
        reject(_PROTOCOL, f"{parameter}.item_id", "Chat tool calls lack item IDs")
    if call.kind != "function":
        reject(_PROTOCOL, f"{parameter}.kind", f"unsupported tool call kind {call.kind!r}")
    if call.opaque_state is not None:
        reject(_PROTOCOL, f"{parameter}.opaque_state", "Chat cannot carry opaque tool state")
    name = call.name
    if call.namespace is not None:
        try:
            name = encode_namespaced_tool_name(call.namespace, call.name)
        except ValueError as error:
            reject(_PROTOCOL, f"{parameter}.namespace", str(error))
    return {
        "id": call.call_id,
        "type": "function",
        "function": {"name": name, "arguments": encode_arguments(call.arguments)},
    }


def _encode_tool(tool: ToolDefinition) -> dict[str, Any]:
    name = tool.name
    if tool.namespace is not None:
        try:
            name = encode_namespaced_tool_name(tool.namespace, tool.name)
        except ValueError as error:
            reject(_PROTOCOL, f"tools.{tool.name}.namespace", str(error))
    function: dict[str, Any] = {
        "name": name,
        "parameters": thaw_json(tool.input_schema),
    }
    if tool.description is not None:
        function["description"] = tool.description
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def _decode_response(
    payload: Mapping[str, Any],
    *,
    origin_provider: str | None = None,
    origin_binding: str | None = None,
    allow_reasoning_opaque: bool = False,
) -> SemanticResponse:
    body = require_mapping(payload, protocol=_PROTOCOL, parameter="response")
    reject_unknown_keys(
        body,
        frozenset(
            {
                "id",
                "object",
                "created",
                "model",
                "choices",
                "usage",
                "system_fingerprint",
                "service_tier",
            }
        ),
        protocol=_PROTOCOL,
        parameter="response",
    )
    choices = require_list(body.get("choices"), protocol=_PROTOCOL, parameter="response.choices")
    if len(choices) != 1:
        decode_reject(_PROTOCOL, "response.choices", "exactly one choice is required")
    choice = require_mapping(choices[0], protocol=_PROTOCOL, parameter="response.choices[0]")
    reject_unknown_keys(
        choice,
        frozenset({"index", "message", "finish_reason", "logprobs"}),
        protocol=_PROTOCOL,
        parameter="response.choices[0]",
    )
    if choice.get("logprobs") is not None:
        decode_reject(_PROTOCOL, "response.choices[0].logprobs", "is not modeled")
    model = require_string(
        body.get("model"),
        protocol=_PROTOCOL,
        parameter="response.model",
        allow_empty=False,
    )
    message = _decode_message(
        choice.get("message"),
        parameter="response.choices[0].message",
        origin_provider=origin_provider,
        origin_binding=origin_binding,
        origin_model=model,
        allow_reasoning_opaque=allow_reasoning_opaque,
    )
    if message.role is not MessageRole.ASSISTANT:
        decode_reject(_PROTOCOL, "response.choices[0].message.role", "must be assistant")
    finish_reason = optional_string(
        choice.get("finish_reason"),
        protocol=_PROTOCOL,
        parameter="response.choices[0].finish_reason",
    )
    status = "incomplete" if finish_reason in {"length", "content_filter"} else "completed"
    metadata: dict[str, Any] = {}
    if "created" in body:
        metadata["created"] = optional_int(
            body.get("created"), protocol=_PROTOCOL, parameter="response.created"
        )
    if "object" in body:
        metadata["object"] = require_string(
            body.get("object"), protocol=_PROTOCOL, parameter="response.object"
        )
    return SemanticResponse(
        id=require_string(
            body.get("id"), protocol=_PROTOCOL, parameter="response.id", allow_empty=False
        ),
        model=model,
        output=(message,),
        usage=decode_usage(
            body.get("usage"),
            protocol=_PROTOCOL,
            input_field="prompt_tokens",
            output_field="completion_tokens",
            input_details_field="prompt_tokens_details",
            output_details_field="completion_tokens_details",
            top_level_reasoning_field="reasoning_tokens",
        ),
        terminal=TerminalMetadata(finish_reason=finish_reason, response_status=status),
        metadata=metadata,
    )


def _encode_response(response: SemanticResponse) -> dict[str, Any]:
    if response.id is None:
        reject(_PROTOCOL, "response.id", "Chat responses require an ID")
    message = _response_message(response)
    finish_reason = response.terminal.finish_reason if response.terminal is not None else None
    if finish_reason is None:
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
    if response.terminal is not None:
        if response.terminal.error_code is not None or response.terminal.error_message is not None:
            reject(_PROTOCOL, "response.terminal", "Chat responses cannot carry terminal errors")
        status_error = _terminal_status_error(response.terminal)
        if status_error is not None:
            reject(
                _PROTOCOL,
                "response.terminal.response_status",
                f"Chat responses cannot represent {response.terminal.response_status!r} status",
            )
        if response.terminal.transport_status is not None:
            reject(
                _PROTOCOL,
                "response.terminal.transport_status",
                "transport status is not a Chat response field",
            )
    allowed_metadata = {"created", "created_at", "object"}
    unknown = sorted(set(response.metadata) - allowed_metadata)
    if unknown:
        reject(_PROTOCOL, f"response.metadata.{unknown[0]}", "metadata is not portable")
    created = response.metadata.get(
        "created",
        response.metadata.get("created_at", int(time.time())),
    )
    if (
        "created" in response.metadata
        and "created_at" in response.metadata
        and response.metadata["created"] != response.metadata["created_at"]
    ):
        reject(_PROTOCOL, "response.metadata.created_at", "conflicts with created")
    if not isinstance(created, int) or isinstance(created, bool):
        source = "created" if "created" in response.metadata else "created_at"
        reject(_PROTOCOL, f"response.metadata.{source}", "must be an integer")
    source_object = response.metadata.get("object")
    if source_object is not None and source_object not in {"chat.completion", "response"}:
        reject(_PROTOCOL, "response.metadata.object", "metadata is not portable")
    return {
        "id": response.id,
        "object": "chat.completion",
        "created": created,
        "model": response.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": encode_usage(
            response.usage,
            protocol=_PROTOCOL,
            input_field="prompt_tokens",
            output_field="completion_tokens",
            input_details_field="prompt_tokens_details",
            output_details_field="completion_tokens_details",
        ),
    }


def _response_message(response: SemanticResponse) -> dict[str, Any]:
    parts = []
    for index, item in enumerate(response.output):
        if isinstance(item, SemanticMessage):
            if item.role is not MessageRole.ASSISTANT:
                reject(_PROTOCOL, f"response.output[{index}].role", "must be assistant")
            if item.name is not None:
                reject(
                    _PROTOCOL,
                    f"response.output[{index}].name",
                    "Chat response messages cannot carry a semantic name",
                )
            # Responses assigns an ID and lifecycle status to its output-message
            # container. Chat carries the same assistant content directly inside
            # a choice, so those two transport-only fields have no wire slots and
            # are intentionally projected away. Semantic content remains guarded
            # by _encode_message (for example opaque reasoning and tool namespace).
            parts.extend(item.content)
        elif isinstance(item, TextContent | RefusalContent | ReasoningSummary | ToolCall):
            parts.append(item)
        else:
            reject(
                _PROTOCOL,
                f"response.output[{index}]",
                f"{type(item).__name__} is not a Chat assistant output",
            )
    return _encode_message(
        SemanticMessage(role=MessageRole.ASSISTANT, content=tuple(parts)), index=0
    )


def chat_request_to_semantic(request: ChatRequest) -> SemanticRequest:
    """Convert the legacy provider-facing Chat DTO into semantic IR."""
    stop = request.stop_sequences if request.stop_sequences is not None else request.stop
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                **({"name": getattr(message, "name")} if hasattr(message, "name") else {}),
                **({"tool_call_id": message.tool_call_id} if message.tool_call_id else {}),
                **({"tool_calls": message.tool_calls} if message.tool_calls else {}),
                **({"refusal": message.refusal} if message.refusal is not None else {}),
            }
            for message in request.messages
        ],
        "stream": request.stream,
    }
    for key, value in {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "tools": request.tools,
        "tool_choice": request.tool_choice,
        "reasoning_effort": request.reasoning_effort,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "stop": stop,
        "user": request.user,
        "metadata": request.metadata,
        "service_tier": request.service_tier,
    }.items():
        if value is not None:
            payload[key] = value
    if request.thinking_type is not None or request.thinking_budget is not None:
        payload["thinking"] = {
            "type": request.thinking_type or "enabled",
            **(
                {"budget_tokens": request.thinking_budget}
                if request.thinking_budget is not None
                else {}
            ),
        }
    semantic = _decode_request(payload)
    return replace(
        semantic,
        top_k=request.top_k,
        candidate_count=request.candidate_count,
        response_mime_type=request.response_mime_type,
        structured_output=request.output_format,
        provider_extensions=request.provider_extensions,
    )


def semantic_to_chat_request(request: SemanticRequest) -> ChatRequest:
    """Convert semantic IR into the legacy provider-facing Chat DTO."""
    messages = []
    for index, item in enumerate(request.input):
        encoded = _encode_message(item, index=index)
        if "name" in encoded:
            reject(
                _PROTOCOL,
                f"input[{index}].name",
                "legacy Chat Message cannot preserve message names",
            )
        messages.append(
            Message(
                role=encoded["role"],
                content=encoded.get("content"),
                tool_call_id=encoded.get("tool_call_id"),
                tool_calls=encoded.get("tool_calls"),
                refusal=encoded.get("refusal"),
            )
        )
    return ChatRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=[_encode_tool(tool) for tool in request.tools] or None,
        tool_choice=encode_tool_choice(
            request.tool_choice, protocol=_PROTOCOL, nested_function=True
        ),
        thinking_budget=request.reasoning.budget_tokens if request.reasoning else None,
        thinking_type=(
            "enabled"
            if request.reasoning and request.reasoning.enabled is True
            else "disabled"
            if request.reasoning and request.reasoning.enabled is False
            else None
        ),
        reasoning_effort=request.reasoning.effort if request.reasoning else None,
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


def chat_response_to_semantic(
    response: ChatResponse,
    *,
    response_id: str,
    origin_provider: str | None = None,
) -> SemanticResponse:
    """Convert a legacy Chat response, including reasoning provenance, to IR."""
    parts: list[Any] = []
    if response.content is not None:
        parts.append(TextContent(response.content))
    if response.refusal is not None:
        parts.append(RefusalContent(response.refusal))
    if response.thinking is not None or response.thinking_signature is not None:
        opaque = None
        if response.thinking_signature is not None:
            if response.thinking_id is None or origin_provider is None:
                reject(
                    _PROTOCOL,
                    "response.thinking_signature",
                    "opaque reasoning requires thinking_id and origin_provider",
                )
            opaque = OpaqueState(
                origin_protocol=_PROTOCOL,
                origin_provider=origin_provider,
                origin_model=response.model,
                item_id=response.thinking_id,
                blob=response.thinking_signature,
            )
        parts.append(ReasoningSummary(response.thinking or "", opaque_state=opaque))
    for index, raw_call in enumerate(response.tool_calls or []):
        call = _decode_tool_call(raw_call, parameter=f"response.tool_calls[{index}]")
        parts.append(call)
    raw = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": response.finish_reason,
            }
        ],
        "usage": response.usage,
    }
    semantic = _decode_response(raw)
    outcome = resolve_terminal_outcome(
        response.terminal_outcome,
        response.finish_reason,
    )
    terminal = semantic.terminal
    if outcome is not None:
        terminal, _ = terminal_event_values(outcome)
    return replace(
        semantic,
        output=(SemanticMessage(role=MessageRole.ASSISTANT, content=tuple(parts)),),
        terminal=terminal,
    )


def semantic_to_chat_response(response: SemanticResponse) -> ChatResponse:
    """Convert semantic IR into the legacy provider-facing Chat response DTO."""
    # The internal DTO has dedicated fields for the opaque reasoning pair,
    # while Chat's public message shape does not. Strip only that pair for the
    # strict wire projection, then restore it on the DTO below.
    wire_output = []
    opaque_states = []
    for item in response.output:
        if isinstance(item, SemanticMessage):
            parts = []
            for part in item.content:
                if isinstance(part, ReasoningSummary) and part.opaque_state is not None:
                    opaque_states.append(part.opaque_state)
                    parts.append(replace(part, opaque_state=None))
                else:
                    parts.append(part)
            wire_output.append(replace(item, content=tuple(parts)))
        elif isinstance(item, ReasoningSummary) and item.opaque_state is not None:
            opaque_states.append(item.opaque_state)
            wire_output.append(replace(item, opaque_state=None))
        else:
            wire_output.append(item)
    if len(opaque_states) > 1:
        reject(
            _PROTOCOL,
            "response.output",
            "legacy ChatResponse can preserve only one opaque reasoning state",
        )
    opaque = opaque_states[0] if opaque_states else None
    if opaque is not None:
        if opaque.origin_protocol is not _PROTOCOL:
            reject(
                _PROTOCOL,
                "response.output.opaque_state.origin_protocol",
                "legacy ChatResponse can preserve only Chat-origin opaque state",
            )
        if not isinstance(opaque.blob, str):
            reject(
                _PROTOCOL,
                "response.output.opaque_state.blob",
                "legacy ChatResponse requires text opaque state",
            )
    terminal_outcome = None
    if response.terminal is not None:
        terminal_outcome = terminal_outcome_from_event(
            SemanticEvent(
                type=SemanticEventType.TERMINAL,
                terminal=response.terminal,
            ),
            protocol=_PROTOCOL,
        )
    wire_terminal = response.terminal
    if wire_terminal is not None:
        wire_response_status = wire_terminal.response_status
        if terminal_outcome is not None and terminal_outcome.response_status.value in {
            "failed",
            "cancelled",
            "unknown",
        }:
            # This encoder call only extracts the legacy DTO message fields.
            # The canonical non-success state is restored below via terminal_outcome.
            wire_response_status = None
        wire_terminal = TerminalMetadata(
            finish_reason=wire_terminal.finish_reason,
            response_status=wire_response_status,
        )
    encoded = _encode_response(
        replace(
            response,
            output=tuple(wire_output),
            terminal=wire_terminal,
        )
    )
    choice = encoded["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content")
    tool_calls = message.get("tool_calls")
    legacy_finish_reason = choice["finish_reason"]
    if terminal_outcome is not None and terminal_outcome.response_status.value in {
        "failed",
        "cancelled",
        "unknown",
    }:
        legacy_finish_reason = None
    return ChatResponse(
        content=message.get("content"),
        model=response.model,
        finish_reason=cast(str, legacy_finish_reason),
        usage=encoded.get("usage"),
        tool_calls=tool_calls,
        thinking=reasoning,
        thinking_id=opaque.item_id if opaque is not None else None,
        thinking_signature=opaque.blob if opaque is not None else None,
        refusal=message.get("refusal"),
        terminal_outcome=terminal_outcome,
    )


def chat_chunk_to_semantic_events(
    chunk: ChatStreamChunk,
    *,
    sequence_start: int = 0,
    response_id: str | None = None,
    model: str | None = None,
    origin_provider: str | None = None,
) -> tuple[SemanticEvent, ...]:
    """Project one legacy Chat chunk into a deterministically ordered event batch.

    Payload events are followed by usage and then terminal/error events.
    Opaque reasoning is accepted only with complete provenance.
    """
    events: list[SemanticEvent] = []

    def emit(event_type: SemanticEventType, **values: Any) -> None:
        events.append(
            SemanticEvent(
                type=event_type,
                sequence=sequence_start + len(events),
                response_id=response_id,
                **values,
            )
        )

    if chunk.thinking:
        emit(SemanticEventType.REASONING_DELTA, delta=chunk.thinking)
    if chunk.thinking_signature is not None:
        if chunk.thinking_id is None or model is None or origin_provider is None:
            reject(
                _PROTOCOL,
                "chunk.thinking_signature",
                "opaque reasoning requires thinking_id, model, and origin_provider",
            )
        emit(
            SemanticEventType.OUTPUT_ITEM,
            item_id=chunk.thinking_id,
            item=ReasoningSummary(
                "",
                opaque_state=OpaqueState(
                    origin_protocol=_PROTOCOL,
                    origin_provider=origin_provider,
                    origin_model=model,
                    item_id=chunk.thinking_id,
                    blob=chunk.thinking_signature,
                ),
            ),
            metadata={"output_item_type": "reasoning", "output_item_done": True},
        )
    elif chunk.thinking_id is not None:
        emit(
            SemanticEventType.OUTPUT_ITEM,
            item_id=chunk.thinking_id,
            metadata={"output_item_type": "reasoning", "output_item_done": True},
        )
    if chunk.content:
        emit(SemanticEventType.TEXT_DELTA, delta=chunk.content)
    if chunk.refusal:
        emit(SemanticEventType.OUTPUT_ITEM, item=RefusalContent(chunk.refusal))
    for index, raw_call in enumerate(chunk.tool_calls or []):
        call_path = f"chunk.tool_calls[{index}]"
        call = require_mapping(raw_call, protocol=_PROTOCOL, parameter=call_path)
        reject_unknown_keys(
            call,
            frozenset({"index", "id", "type", "function"}),
            protocol=_PROTOCOL,
            parameter=call_path,
        )
        function = require_mapping(
            call.get("function", {}),
            protocol=_PROTOCOL,
            parameter=f"{call_path}.function",
        )
        reject_unknown_keys(
            function,
            frozenset({"name", "arguments"}),
            protocol=_PROTOCOL,
            parameter=f"{call_path}.function",
        )
        call_type = call.get("type")
        if call_type not in {None, "function"}:
            decode_reject(
                _PROTOCOL,
                f"{call_path}.type",
                "only function tool-call deltas are supported",
            )
        metadata: dict[str, Any] = {}
        call_id = optional_string(call.get("id"), protocol=_PROTOCOL, parameter=f"{call_path}.id")
        name = optional_string(
            function.get("name"),
            protocol=_PROTOCOL,
            parameter=f"{call_path}.function.name",
        )
        if call_id is not None:
            metadata["call_id"] = call_id
        if name is not None:
            metadata["name"] = name
        arguments = optional_string(
            function.get("arguments"),
            protocol=_PROTOCOL,
            parameter=f"{call_path}.function.arguments",
        )
        raw_index = call.get("index", index)
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            decode_reject(_PROTOCOL, f"{call_path}.index", "must be an integer")
        emit(
            SemanticEventType.TOOL_ARGUMENTS_DELTA,
            output_index=raw_index,
            delta=arguments or "",
            metadata=metadata,
        )
    if chunk.usage is not None:
        emit(
            SemanticEventType.USAGE,
            usage=decode_usage(
                chunk.usage,
                protocol=_PROTOCOL,
                input_field="prompt_tokens",
                output_field="completion_tokens",
                input_details_field="prompt_tokens_details",
                output_details_field="completion_tokens_details",
            ),
        )
    outcome = resolve_terminal_outcome(chunk.terminal_outcome, chunk.finish_reason)
    if outcome is not None:
        terminal, metadata = terminal_event_values(outcome)
        if terminal.error_code is not None or terminal.error_message is not None:
            emit(SemanticEventType.ERROR, terminal=terminal, metadata=metadata)
        emit(SemanticEventType.TERMINAL, terminal=terminal, metadata=metadata)
    return tuple(events)


def semantic_events_to_chat_chunks(
    events: tuple[SemanticEvent, ...] | list[SemanticEvent],
) -> tuple[ChatStreamChunk, ...]:
    """Project semantic events to legacy Chat chunks without reordering them."""
    event_list = tuple(events)
    chunks: list[ChatStreamChunk] = []
    terminal_indices = {
        index for index, event in enumerate(event_list) if event.type is SemanticEventType.TERMINAL
    }
    for index, event in enumerate(event_list):
        if event.type is SemanticEventType.RESPONSE_STARTED:
            continue
        if event.type is SemanticEventType.TEXT_DELTA:
            chunks.append(ChatStreamChunk(content=event.delta or ""))
        elif event.type is SemanticEventType.REASONING_DELTA:
            chunks.append(ChatStreamChunk(content="", thinking=event.delta or ""))
        elif event.type is SemanticEventType.TOOL_ARGUMENTS_DELTA:
            call_delta: dict[str, Any] = {"index": event.output_index or 0}
            call_id = event.metadata.get("call_id")
            if isinstance(call_id, str):
                call_delta.update({"id": call_id, "type": "function"})
            function: dict[str, Any] = {"arguments": event.delta or ""}
            name = event.metadata.get("name")
            if isinstance(name, str):
                function["name"] = name
            call_delta["function"] = function
            chunks.append(ChatStreamChunk(content="", tool_calls=[call_delta]))
        elif event.type is SemanticEventType.OUTPUT_ITEM:
            chunks.extend(_chat_chunks_for_item(event))
        elif event.type is SemanticEventType.USAGE:
            if event.usage is None:
                reject(_PROTOCOL, "event.usage", "usage event requires Usage")
            chunks.append(
                ChatStreamChunk(
                    content="",
                    usage=encode_stream_usage(
                        event.usage,
                        protocol=_PROTOCOL,
                        input_field="prompt_tokens",
                        output_field="completion_tokens",
                        input_details_field="prompt_tokens_details",
                        output_details_field="completion_tokens_details",
                    ),
                )
            )
        elif event.type is SemanticEventType.ERROR:
            if not any(terminal_index > index for terminal_index in terminal_indices):
                chunks.append(
                    ChatStreamChunk(
                        content="",
                        terminal_outcome=terminal_outcome_from_event(event, protocol=_PROTOCOL),
                    )
                )
        elif event.type is SemanticEventType.TERMINAL:
            outcome = terminal_outcome_from_event(event, protocol=_PROTOCOL)
            chunks.append(
                ChatStreamChunk(
                    content="",
                    finish_reason=outcome.finish_reason,
                    terminal_outcome=outcome,
                )
            )
    return tuple(chunks)


def _chat_chunks_for_item(event: SemanticEvent) -> list[ChatStreamChunk]:
    item = event.item
    if item is None:
        return []
    if isinstance(item, TextContent):
        return [ChatStreamChunk(content=item.text)]
    if isinstance(item, RefusalContent):
        return [ChatStreamChunk(content="", refusal=item.refusal)]
    if isinstance(item, ReasoningSummary):
        state = item.opaque_state
        signature = None
        if state is not None:
            blob = state.blob
            if not isinstance(blob, str):
                reject(_PROTOCOL, "event.item.opaque_state.blob", "Chat DTO requires text state")
            signature = blob
        return [
            ChatStreamChunk(
                content="",
                thinking=item.text or None,
                thinking_id=state.item_id if state is not None else event.item_id,
                thinking_signature=signature,
            )
        ]
    if isinstance(item, ToolCall):
        call = _encode_tool_call(item, parameter="event.item")
        call["index"] = event.output_index or 0
        return [ChatStreamChunk(content="", tool_calls=[call])]
    if isinstance(item, SemanticMessage):
        if item.item_id is not None or item.status is not None or item.name is not None:
            reject(_PROTOCOL, "event.item", "Chat stream chunks cannot preserve message metadata")
        chunks = []
        for part in item.content:
            chunks.extend(_chat_chunks_for_item(replace(event, item=part)))
        return chunks
    reject(_PROTOCOL, "event.item", f"unsupported stream item {type(item).__name__}")


__all__ = [
    "OpenAIChatRuntime",
    "OpenAIChatStreamDecoder",
    "OpenAIChatStreamEncoder",
    "chat_chunk_to_semantic_events",
    "chat_request_to_semantic",
    "chat_response_to_semantic",
    "semantic_events_to_chat_chunks",
    "semantic_to_chat_request",
    "semantic_to_chat_response",
]
