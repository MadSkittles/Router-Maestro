"""Concrete Gemini generateContent protocol runtime.

Gemini carries its model and streaming mode in the endpoint rather than the
JSON body.  A dispatcher therefore supplies endpoint context through the
runtime constructor or ``decode_request_for_model``.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from router_maestro.protocols._wire import (
    decode_reject,
    is_reasoning_capsule_carrier,
    optional_bool,
    optional_int,
    optional_number,
    optional_string,
    reject,
    reject_unknown_keys,
    require_list,
    require_mapping,
    require_string,
    thaw_json,
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
    ToolChoice,
    ToolDefinition,
    ToolResult,
    Usage,
    UsageMode,
    WireProtocol,
)
from router_maestro.protocols.runtime import OpaqueStateDecodeHook, OpaqueStateEncodeHook

_PROTOCOL = WireProtocol.GEMINI
_REQUEST_FIELDS = frozenset(
    {"contents", "systemInstruction", "generationConfig", "tools", "toolConfig", "model"}
)
_CONTENT_FIELDS = frozenset({"role", "parts"})
_PART_FIELDS = frozenset(
    {
        "text",
        "functionCall",
        "functionResponse",
        "inlineData",
        "fileData",
        "thought",
        "thoughtSignature",
    }
)
_GENERATION_FIELDS = frozenset(
    {
        "temperature",
        "topP",
        "topK",
        "maxOutputTokens",
        "stopSequences",
        "candidateCount",
        "responseMimeType",
        "responseSchema",
        "responseJsonSchema",
        "frequencyPenalty",
        "presencePenalty",
        "thinkingConfig",
    }
)
_FINISH_TO_SEMANTIC = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
}
_FINISH_FROM_SEMANTIC = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",
    "content_filter": "SAFETY",
    "STOP": "STOP",
    "MAX_TOKENS": "MAX_TOKENS",
    "SAFETY": "SAFETY",
    "RECITATION": "RECITATION",
}
_NON_SUCCESS_TERMINAL_ERRORS = {
    "failed": (502, "INTERNAL", "upstream_error", "Upstream response failed"),
    "cancelled": (499, "CANCELLED", "upstream_cancelled", "Upstream response was cancelled"),
    "unknown": (
        502,
        "INTERNAL",
        "upstream_status_unknown",
        "Upstream response ended with an unknown status",
    ),
}
_NON_SUCCESS_TRANSPORT_STATUSES = {
    "exception": "failed",
    "client_cancelled": "cancelled",
    "unexpected_eof": "unknown",
}


def _terminal_status_error(terminal: TerminalMetadata) -> tuple[int, str, str, str] | None:
    """Return a safe Gemini error when a terminal cannot mean successful completion."""
    status = terminal.response_status
    if status is None:
        status = _NON_SUCCESS_TRANSPORT_STATUSES.get(terminal.transport_termination or "")
    if status in _NON_SUCCESS_TERMINAL_ERRORS:
        return _NON_SUCCESS_TERMINAL_ERRORS[status]
    if status not in {None, "completed", "incomplete"}:
        reject(_PROTOCOL, "response.terminal.response_status", f"unsupported value {status!r}")
    return None


@dataclass(slots=True)
class _PendingToolCall:
    """Small per-call buffer required by Gemini's object-valued functionCall."""

    call_id: str | None = None
    name: str | None = None
    kind: str = "function"
    seed_arguments: Mapping[str, Any] | None = None
    argument_parts: list[str] = field(default_factory=list)


class GeminiRuntime:
    """Strict Gemini wire codec for semantic conversion paths."""

    protocol = _PROTOCOL

    def __init__(
        self,
        default_model: str | None = None,
        *,
        stream: bool = False,
        origin_provider: str | None = None,
        decode_opaque_state: OpaqueStateDecodeHook | None = None,
        encode_opaque_state: OpaqueStateEncodeHook | None = None,
    ) -> None:
        self.default_model = default_model
        self.stream = stream
        self.origin_provider = origin_provider
        self.decode_opaque_state = decode_opaque_state
        self.encode_opaque_state = encode_opaque_state
        self._stream_decoder: ContextVar[GeminiStreamDecoder | None] = ContextVar(
            f"gemini_stream_decoder_{id(self)}",
            default=None,
        )
        self._stream_encoder: ContextVar[GeminiStreamEncoder | None] = ContextVar(
            f"gemini_stream_encoder_{id(self)}",
            default=None,
        )

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        model = payload.get("model")
        if not isinstance(model, str):
            model = self.default_model
        contents = payload.get("contents")
        opaque_carriers = _thought_signatures(contents)
        return RequestManifest(
            protocol=self.protocol,
            model=model,
            stream=self.stream,
            tools=bool(payload.get("tools")),
            images=_has_gemini_media(contents, image=True),
            files=_has_gemini_media(contents, image=False),
            reasoning=_has_field(payload, "thinkingConfig")
            or _has_truthy_field(payload, "thought"),
            reasoning_capsules=tuple(
                carrier for carrier in opaque_carriers if is_reasoning_capsule_carrier(carrier)
            ),
            opaque_continuation=bool(opaque_carriers),
        )

    async def decode_request(self, payload: Mapping[str, Any]) -> SemanticRequest:
        model = payload.get("model")
        if not isinstance(model, str):
            model = self.default_model
        if not model:
            decode_reject(
                _PROTOCOL,
                "model",
                "Gemini model is endpoint context; configure default_model or use "
                "decode_request_for_model",
            )
        return _decode_request(
            payload,
            model=model,
            stream=self.stream,
            origin_provider=self.origin_provider,
            decode_opaque_state=self.decode_opaque_state,
        )

    async def decode_request_for_model(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool | None = None,
    ) -> SemanticRequest:
        if not model:
            decode_reject(_PROTOCOL, "model", "endpoint model cannot be empty")
        return _decode_request(
            payload,
            model=model,
            stream=self.stream if stream is None else stream,
            origin_provider=self.origin_provider,
            decode_opaque_state=self.decode_opaque_state,
        )

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        return _encode_request(request, encode_opaque_state=self.encode_opaque_state)

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        return _decode_response(
            payload,
            fallback_model=self.default_model,
            origin_provider=self.origin_provider,
            decode_opaque_state=self.decode_opaque_state,
        )

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        return _encode_response(response, encode_opaque_state=self.encode_opaque_state)

    def new_stream_decoder(self, *, sequence_start: int = 0) -> GeminiStreamDecoder:
        """Create isolated state for one upstream Gemini response stream."""
        return GeminiStreamDecoder(
            default_model=self.default_model,
            origin_provider=self.origin_provider,
            decode_opaque_state=self.decode_opaque_state,
            sequence_start=sequence_start,
        )

    def new_stream_encoder(self, *, model: str | None = None) -> GeminiStreamEncoder:
        """Create isolated state for one downstream Gemini response stream."""
        return GeminiStreamEncoder(
            model=model or self.default_model,
            encode_opaque_state=self.encode_opaque_state,
        )

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        """Convenience delegate for one stream per async context."""
        decoder = self._stream_decoder.get()
        if decoder is None or (decoder.terminal and "candidates" in payload):
            decoder = self.new_stream_decoder()
            self._stream_decoder.set(decoder)
        return decoder.decode(payload)

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        """Convenience delegate for one stream per async context."""
        encoder = self._stream_encoder.get()
        if encoder is None or (
            encoder.terminal and event.type is SemanticEventType.RESPONSE_STARTED
        ):
            metadata_model = event.metadata.get("model")
            encoder = self.new_stream_encoder(
                model=metadata_model if isinstance(metadata_model, str) else None
            )
            self._stream_encoder.set(encoder)
        return encoder.encode(event)


def _has_truthy_field(value: object, field: str) -> bool:
    if isinstance(value, Mapping):
        return value.get(field) is True or any(
            _has_truthy_field(item, field) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_has_truthy_field(item, field) for item in value)
    return False


def _has_field(value: object, field: str) -> bool:
    if isinstance(value, Mapping):
        return (field in value and value.get(field) is not None) or any(
            _has_field(item, field) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_has_field(item, field) for item in value)
    return False


def _thought_signatures(value: object) -> tuple[str, ...]:
    """Collect Gemini thought signatures without interpreting their contents."""
    signatures: list[str] = []

    def visit(candidate: object) -> None:
        if isinstance(candidate, Mapping):
            signature = candidate.get("thoughtSignature")
            if isinstance(signature, str):
                signatures.append(signature)
            for nested in candidate.values():
                visit(nested)
        elif isinstance(candidate, list | tuple):
            for nested in candidate:
                visit(nested)

    visit(value)
    return tuple(signatures)


def _has_gemini_media(value: object, *, image: bool) -> bool:
    """Classify Gemini inline/file data by MIME type during cheap inspection."""
    if isinstance(value, Mapping):
        for field in ("inlineData", "fileData"):
            media = value.get(field)
            if not isinstance(media, Mapping):
                continue
            media_type = media.get("mimeType")
            is_image = isinstance(media_type, str) and media_type.startswith("image/")
            if is_image is image:
                return True
        return any(_has_gemini_media(nested, image=image) for nested in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_has_gemini_media(nested, image=image) for nested in value)
    return False


class GeminiStreamDecoder:
    """Stateful decoder for exactly one Gemini streaming response."""

    def __init__(
        self,
        *,
        default_model: str | None = None,
        origin_provider: str | None = None,
        decode_opaque_state: OpaqueStateDecodeHook | None = None,
        sequence_start: int = 0,
    ) -> None:
        self.model = default_model
        self.origin_provider = origin_provider
        self.decode_opaque_state = decode_opaque_state
        self._sequence = sequence_start
        self._started = False
        self._terminal = False

    @property
    def terminal(self) -> bool:
        return self._terminal

    def decode(self, payload: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        frame = require_mapping(payload, protocol=_PROTOCOL, parameter="stream")
        if self._terminal:
            decode_reject(_PROTOCOL, "stream", "frame arrived after terminal event")
        if "error" in frame:
            return self._decode_error(frame)
        reject_unknown_keys(
            frame,
            frozenset({"candidates", "usageMetadata", "modelVersion", "promptFeedback"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        model_value = frame.get("modelVersion")
        if model_value is not None:
            model = require_string(
                model_value,
                protocol=_PROTOCOL,
                parameter="stream.modelVersion",
                allow_empty=False,
            )
            if self.model is not None and self.model != model:
                decode_reject(_PROTOCOL, "stream.modelVersion", "model changed within stream")
            self.model = model

        specs: list[tuple[SemanticEventType, dict[str, Any]]] = []
        if not self._started:
            self._started = True
            metadata = {"model": self.model} if self.model is not None else {}
            specs.append((SemanticEventType.RESPONSE_STARTED, {"metadata": metadata}))

        prompt_feedback, terminal = _decode_prompt_feedback(
            frame.get("promptFeedback"),
            parameter="stream.promptFeedback",
        )
        if frame.get("candidates") is not None:
            candidates = require_list(
                frame.get("candidates"), protocol=_PROTOCOL, parameter="stream.candidates"
            )
            if not candidates and terminal is not None:
                candidates = ()
            if len(candidates) != 1:
                if candidates or terminal is None:
                    decode_reject(
                        _PROTOCOL,
                        "stream.candidates",
                        "exactly one candidate is required",
                    )
            else:
                if terminal is not None:
                    decode_reject(
                        _PROTOCOL,
                        "stream.promptFeedback",
                        "blocked prompt cannot include an output candidate",
                    )
                candidate = require_mapping(
                    candidates[0], protocol=_PROTOCOL, parameter="stream.candidates[0]"
                )
                reject_unknown_keys(
                    candidate,
                    frozenset({"content", "finishReason", "index"}),
                    protocol=_PROTOCOL,
                    parameter="stream.candidates[0]",
                )
                index = optional_int(
                    candidate.get("index", 0),
                    protocol=_PROTOCOL,
                    parameter="stream.candidates[0].index",
                )
                if index != 0:
                    decode_reject(_PROTOCOL, "stream.candidates[0].index", "must be zero")
                if candidate.get("content") is not None:
                    if not self.model:
                        decode_reject(
                            _PROTOCOL,
                            "stream.modelVersion",
                            "reasoning-safe stream decoding requires model context",
                        )
                    messages = _decode_content(
                        candidate["content"],
                        parameter="stream.candidates[0].content",
                        content_index=0,
                        model=self.model,
                        prior_calls={},
                        origin_provider=self.origin_provider,
                        decode_opaque_state=self.decode_opaque_state,
                    )
                    if len(messages) != 1 or messages[0].role is not MessageRole.ASSISTANT:
                        decode_reject(
                            _PROTOCOL,
                            "stream.candidates[0].content",
                            "candidate content must be one model message",
                        )
                    for part in messages[0].content:
                        specs.extend(self._part_events(part, output_index=0))
                finish = optional_string(
                    candidate.get("finishReason"),
                    protocol=_PROTOCOL,
                    parameter="stream.candidates[0].finishReason",
                )
                if finish is not None:
                    semantic_finish = _FINISH_TO_SEMANTIC.get(finish)
                    if semantic_finish is None:
                        decode_reject(
                            _PROTOCOL,
                            "stream.candidates[0].finishReason",
                            f"unsupported finish reason {finish!r}",
                        )
                    terminal = TerminalMetadata(
                        finish_reason=semantic_finish,
                        response_status=(
                            "incomplete"
                            if semantic_finish in {"length", "content_filter"}
                            else "completed"
                        ),
                    )
        if frame.get("usageMetadata") is not None:
            usage = _decode_usage(frame["usageMetadata"])
            if usage is not None:
                specs.append((SemanticEventType.USAGE, {"usage": usage}))
        if terminal is not None:
            metadata = (
                {"gemini_prompt_feedback": prompt_feedback} if prompt_feedback is not None else {}
            )
            specs.append((SemanticEventType.TERMINAL, {"terminal": terminal, "metadata": metadata}))
            self._terminal = True
        return self._events(*specs)

    def finish_eof(self) -> tuple[SemanticEvent, ...]:
        """Convert EOF without finishReason into one safe terminal pair."""
        if self._terminal:
            return ()
        terminal = TerminalMetadata(
            error_code="unexpected_eof",
            error_message="Upstream stream ended before finishReason",
            response_status="unknown",
        )
        self._terminal = True
        return self._events(
            (SemanticEventType.ERROR, {"terminal": terminal}),
            (SemanticEventType.TERMINAL, {"terminal": terminal}),
            common_metadata={"transport_termination": "unexpected_eof"},
        )

    def _part_events(
        self,
        part: object,
        *,
        output_index: int,
    ) -> list[tuple[SemanticEventType, dict[str, Any]]]:
        if isinstance(part, TextContent):
            return [
                (SemanticEventType.TEXT_DELTA, {"output_index": output_index, "delta": part.text})
            ]
        if isinstance(part, ReasoningSummary):
            events: list[tuple[SemanticEventType, dict[str, Any]]] = []
            if part.text:
                events.append(
                    (
                        SemanticEventType.REASONING_DELTA,
                        {"output_index": output_index, "delta": part.text},
                    )
                )
            if part.opaque_state is not None:
                events.append(
                    (
                        SemanticEventType.OUTPUT_ITEM,
                        {
                            "output_index": output_index,
                            "item_id": part.opaque_state.item_id,
                            # The text was already emitted as a reasoning delta.
                            # This item carries only the opaque continuation so a
                            # target encoder cannot duplicate visible reasoning.
                            "item": ReasoningSummary("", opaque_state=part.opaque_state),
                            "metadata": {"output_item_type": "reasoning"},
                        },
                    )
                )
            return events
        if isinstance(part, ToolCall):
            return [
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {
                        "output_index": output_index,
                        "item_id": part.call_id,
                        "item": part,
                        "metadata": {"output_item_type": "function_call"},
                    },
                )
            ]
        if isinstance(part, ImageContent | FileContent):
            return [
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {"output_index": output_index, "item": part},
                )
            ]
        decode_reject(_PROTOCOL, "stream.candidates[0].content", "unsupported output part")

    def _decode_error(self, frame: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        reject_unknown_keys(
            frame,
            frozenset({"error"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        error = require_mapping(frame.get("error"), protocol=_PROTOCOL, parameter="stream.error")
        code_value = error.get("status", error.get("code", "upstream_error"))
        code = str(code_value)
        message = require_string(
            error.get("message"),
            protocol=_PROTOCOL,
            parameter="stream.error.message",
        )
        terminal = TerminalMetadata(
            error_code=code,
            error_message=message,
            response_status="failed",
        )
        self._terminal = True
        return self._events(
            (SemanticEventType.ERROR, {"terminal": terminal}),
            (SemanticEventType.TERMINAL, {"terminal": terminal}),
            common_metadata={"transport_termination": "exception"},
        )

    def _events(
        self,
        *specs: tuple[SemanticEventType, dict[str, Any]],
        common_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[SemanticEvent, ...]:
        events = []
        for event_type, values in specs:
            metadata = dict(common_metadata or {})
            metadata.update(values.pop("metadata", {}))
            events.append(
                SemanticEvent(
                    type=event_type,
                    sequence=self._sequence,
                    metadata=metadata,
                    **values,
                )
            )
            self._sequence += 1
        return tuple(events)


class GeminiStreamEncoder:
    """Stateful encoder for exactly one Gemini streaming response."""

    def __init__(
        self,
        *,
        model: str | None = None,
        encode_opaque_state: OpaqueStateEncodeHook | None = None,
    ) -> None:
        self.model = model
        self.encode_opaque_state = encode_opaque_state
        self._terminal = False
        self._pending_error: TerminalMetadata | None = None
        self._pending_tool_calls: dict[int, _PendingToolCall] = {}
        self._flushed_tool_calls: set[int] = set()

    @property
    def terminal(self) -> bool:
        return self._terminal

    def encode(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        if self._terminal:
            reject(_PROTOCOL, "event.type", "event arrived after terminal event")
        metadata_model = event.metadata.get("model")
        if isinstance(metadata_model, str):
            if self.model is not None and self.model != metadata_model:
                reject(_PROTOCOL, "event.metadata.model", "model changed within stream")
            self.model = metadata_model
        if event.type is SemanticEventType.RESPONSE_STARTED:
            return ()
        if event.type is SemanticEventType.ERROR:
            self._pending_error = event.terminal or TerminalMetadata(
                error_code="upstream_error",
                error_message="Upstream stream failed",
                response_status="failed",
            )
            return ()
        if event.type is SemanticEventType.TERMINAL:
            return self._encode_terminal(event)
        if event.type is SemanticEventType.TOOL_ARGUMENTS_DELTA:
            self._capture_tool_delta(event)
            return ()
        if event.type is SemanticEventType.OUTPUT_ITEM and event.item is None:
            # Other runtimes expose output-item start/stop bookkeeping as semantic
            # lifecycle events. A tool-call completion is the one exception: it
            # closes the argument buffer that Gemini must expose as one JSON object.
            if event.metadata.get("output_item_done") is True and event.metadata.get(
                "output_item_type"
            ) in {"function_call", "custom_tool_call", "tool_use"}:
                return self._flush_tool_calls(self._tool_index(event))
            return ()
        if event.type is SemanticEventType.OUTPUT_ITEM and isinstance(event.item, ToolCall):
            self._capture_tool_item(event, event.item)
            if event.metadata.get("output_item_done") is True:
                return self._flush_tool_calls(self._tool_index(event))
            return ()
        payload: dict[str, Any] = {}
        self._put_model(payload)
        if event.type is SemanticEventType.USAGE:
            if event.usage is None:
                reject(_PROTOCOL, "event.usage", "usage event requires Usage")
            usage = _encode_usage(event.usage)
            if usage is not None:
                payload["usageMetadata"] = usage
            return (*self._flush_tool_calls(), payload)
        else:
            part = self._encode_part(event)
            payload["candidates"] = [{"content": {"role": "model", "parts": [part]}, "index": 0}]
        return (payload,)

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

    def _encode_part(self, event: SemanticEvent) -> dict[str, Any]:
        if event.output_index not in {None, 0}:
            reject(_PROTOCOL, "event.output_index", "Gemini runtime supports candidate zero")
        if event.type is SemanticEventType.TEXT_DELTA:
            return {"text": event.delta or ""}
        if event.type is SemanticEventType.REASONING_DELTA:
            return {"text": event.delta or "", "thought": True}
        if event.type is not SemanticEventType.OUTPUT_ITEM or event.item is None:
            reject(_PROTOCOL, "event.type", f"unsupported event {event.type.value!r}")
        item = event.item
        if isinstance(item, TextContent):
            return {"text": item.text}
        if isinstance(item, ReasoningSummary):
            return _encode_reasoning_part(
                item,
                parameter="event.item",
                model=self._required_model(),
                encode_opaque_state=self.encode_opaque_state,
            )
        if isinstance(item, ToolCall):
            return _encode_function_call(
                item,
                parameter="event.item",
                model=self._required_model(),
                encode_opaque_state=self.encode_opaque_state,
            )
        if isinstance(item, ImageContent | FileContent):
            return _encode_data_part(item, parameter="event.item")
        if isinstance(item, RefusalContent):
            reject(_PROTOCOL, "event.item", "Gemini has no distinct refusal part")
        reject(_PROTOCOL, "event.item", f"unsupported output {type(item).__name__}")

    def _encode_terminal(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        terminal = (
            self._pending_error
            or event.terminal
            or TerminalMetadata(
                finish_reason="stop",
                response_status="completed",
            )
        )
        prefix: tuple[Mapping[str, Any], ...] = ()
        status_error = _terminal_status_error(terminal)
        if terminal.error_code is not None or terminal.error_message is not None:
            unexpected_eof = terminal.error_code == "unexpected_eof"
            payload: dict[str, Any] = {
                "error": {
                    "code": 502 if unexpected_eof else terminal.error_code or "upstream_error",
                    "message": terminal.error_message or "Upstream stream failed",
                    **(
                        {
                            "status": "INTERNAL",
                            "details": [{"reason": "unexpected_eof"}],
                        }
                        if unexpected_eof
                        else {}
                    ),
                }
            }
        elif status_error is not None:
            code, status, reason, message = status_error
            payload = {
                "error": {
                    "code": code,
                    "message": message,
                    "status": status,
                    "details": [{"reason": reason}],
                }
            }
        else:
            prefix = self._flush_tool_calls()
            finish = _FINISH_FROM_SEMANTIC.get(terminal.finish_reason or "stop")
            if finish is None:
                reject(
                    _PROTOCOL,
                    "event.terminal.finish_reason",
                    f"unsupported reason {terminal.finish_reason!r}",
                )
            payload = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": []},
                        "finishReason": finish,
                        "index": 0,
                    }
                ]
            }
            self._put_model(payload)
        self._terminal = True
        return (*prefix, payload)

    def _put_model(self, payload: dict[str, Any]) -> None:
        # Gemini responses identify the model per response object. In an SSE
        # stream each data frame is independently shaped like a response, so the
        # model belongs on usage and terminal frames as well as content frames.
        payload["modelVersion"] = self._required_model()

    def _capture_tool_item(self, event: SemanticEvent, item: ToolCall) -> None:
        index = self._tool_index(event)
        pending = self._pending_tool_call(index)
        pending.call_id = item.call_id or event.item_id or pending.call_id
        pending.name = item.name or pending.name
        pending.kind = item.kind
        pending.seed_arguments = item.arguments

    def _capture_tool_delta(self, event: SemanticEvent) -> None:
        index = self._tool_index(event)
        pending = self._pending_tool_call(index)
        call_id = event.metadata.get("call_id")
        if not isinstance(call_id, str):
            call_id = event.item_id if isinstance(event.item_id, str) else None
        name = event.metadata.get("name")
        if call_id:
            if pending.call_id is not None and pending.call_id != call_id:
                reject(_PROTOCOL, "event.item_id", "tool call ID changed within stream")
            pending.call_id = call_id
        if isinstance(name, str) and name:
            if pending.name is not None and pending.name != name:
                reject(_PROTOCOL, "event.metadata.name", "tool name changed within stream")
            pending.name = name
        output_item_type = event.metadata.get("output_item_type")
        if output_item_type == "custom_tool_call":
            pending.kind = "custom"
        elif output_item_type not in {None, "function_call", "tool_use"}:
            reject(
                _PROTOCOL,
                "event.metadata.output_item_type",
                f"unsupported tool call type {output_item_type!r}",
            )
        pending.argument_parts.append(event.delta or "")

    def _pending_tool_call(self, index: int) -> _PendingToolCall:
        if index in self._flushed_tool_calls:
            reject(_PROTOCOL, "event.output_index", "tool event arrived after completion")
        return self._pending_tool_calls.setdefault(index, _PendingToolCall())

    def _flush_tool_calls(
        self,
        index: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        indices = (
            [index]
            if index is not None and index in self._pending_tool_calls
            else sorted(self._pending_tool_calls)
            if index is None
            else []
        )
        if not indices:
            return ()

        parts = []
        for tool_index in indices:
            pending = self._pending_tool_calls.pop(tool_index)
            if pending.kind != "function":
                reject(
                    _PROTOCOL,
                    f"event.tool_calls[{tool_index}].kind",
                    "Gemini supports function calls only",
                )
            if not pending.name:
                reject(
                    _PROTOCOL,
                    f"event.tool_calls[{tool_index}].name",
                    "tool call name is required",
                )
            arguments = self._tool_arguments(pending, tool_index)
            parts.append(
                _encode_function_call(
                    ToolCall(
                        call_id=pending.call_id or "",
                        name=pending.name,
                        arguments=arguments,
                    ),
                    parameter=f"event.tool_calls[{tool_index}]",
                    model=self._required_model(),
                    encode_opaque_state=self.encode_opaque_state,
                )
            )
            self._flushed_tool_calls.add(tool_index)

        payload: dict[str, Any] = {
            "candidates": [{"content": {"role": "model", "parts": parts}, "index": 0}]
        }
        self._put_model(payload)
        return (payload,)

    @staticmethod
    def _tool_arguments(
        pending: _PendingToolCall,
        index: int,
    ) -> Mapping[str, Any]:
        if not pending.argument_parts:
            return pending.seed_arguments or {}
        raw_arguments = "".join(pending.argument_parts)
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            reject(
                _PROTOCOL,
                f"event.tool_calls[{index}].arguments",
                f"must contain valid JSON ({error.msg})",
            )
        if not isinstance(arguments, Mapping):
            reject(
                _PROTOCOL,
                f"event.tool_calls[{index}].arguments",
                "must contain a JSON object",
            )
        seeded = pending.seed_arguments
        if seeded and dict(seeded) != dict(arguments):
            reject(
                _PROTOCOL,
                f"event.tool_calls[{index}].arguments",
                "completed arguments do not match streamed deltas",
            )
        return arguments

    @staticmethod
    def _tool_index(event: SemanticEvent) -> int:
        index = event.output_index if event.output_index is not None else 0
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            reject(_PROTOCOL, "event.output_index", "tool output index must be non-negative")
        return index

    def _required_model(self) -> str:
        if not self.model:
            reject(_PROTOCOL, "event.metadata.model", "Gemini stream requires a model")
        return self.model


def _decode_request(
    payload: Mapping[str, Any],
    *,
    model: str,
    stream: bool,
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
) -> SemanticRequest:
    body = require_mapping(payload, protocol=_PROTOCOL, parameter="request")
    reject_unknown_keys(body, _REQUEST_FIELDS, protocol=_PROTOCOL, parameter="")
    if "model" in body:
        injected_model = require_string(
            body.get("model"), protocol=_PROTOCOL, parameter="model", allow_empty=False
        )
        if injected_model != model:
            decode_reject(_PROTOCOL, "model", "body model conflicts with endpoint model")

    messages: list[SemanticMessage] = []
    system_instruction = body.get("systemInstruction")
    if system_instruction is not None:
        messages.append(_decode_system_instruction(system_instruction))

    prior_calls: dict[str, list[ToolCall]] = {}
    contents = body.get("contents")
    if contents is not None:
        raw_contents = require_list(contents, protocol=_PROTOCOL, parameter="contents")
        for content_index, raw_content in enumerate(raw_contents):
            decoded = _decode_content(
                raw_content,
                parameter=f"contents[{content_index}]",
                content_index=content_index,
                model=model,
                prior_calls=prior_calls,
                origin_provider=origin_provider,
                decode_opaque_state=decode_opaque_state,
            )
            messages.extend(decoded)

    generation = _decode_generation_config(body.get("generationConfig"))
    tool_choice = _decode_tool_config(body.get("toolConfig"))
    return SemanticRequest(
        model=model,
        input=tuple(messages),
        tools=_decode_tools(body.get("tools")),
        stream=stream,
        max_output_tokens=generation["max_output_tokens"],
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        top_k=generation["top_k"],
        candidate_count=generation["candidate_count"],
        frequency_penalty=generation["frequency_penalty"],
        presence_penalty=generation["presence_penalty"],
        stop_sequences=generation["stop_sequences"],
        tool_choice=tool_choice,
        reasoning=generation["reasoning"],
        structured_output=generation["structured_output"],
        response_mime_type=generation["response_mime_type"],
        explicit_fields=frozenset(
            set(body) | {f"generationConfig.{key}" for key in generation["explicit_fields"]}
        ),
    )


def _decode_system_instruction(value: object) -> SemanticMessage:
    content = require_mapping(value, protocol=_PROTOCOL, parameter="systemInstruction")
    reject_unknown_keys(content, _CONTENT_FIELDS, protocol=_PROTOCOL, parameter="systemInstruction")
    role = optional_string(
        content.get("role"), protocol=_PROTOCOL, parameter="systemInstruction.role"
    )
    if role not in {None, "system", "user"}:
        decode_reject(_PROTOCOL, "systemInstruction.role", "unsupported system role")
    parts = require_list(
        content.get("parts"), protocol=_PROTOCOL, parameter="systemInstruction.parts"
    )
    decoded = []
    for index, raw_part in enumerate(parts):
        path = f"systemInstruction.parts[{index}]"
        part = require_mapping(raw_part, protocol=_PROTOCOL, parameter=path)
        reject_unknown_keys(part, _PART_FIELDS, protocol=_PROTOCOL, parameter=path)
        present = _part_payload_fields(part)
        if (
            present != ["text"]
            or part.get("thought") is not None
            or part.get("thoughtSignature") is not None
        ):
            decode_reject(_PROTOCOL, path, "system instructions support plain text parts only")
        decoded.append(
            TextContent(
                require_string(part.get("text"), protocol=_PROTOCOL, parameter=f"{path}.text")
            )
        )
    return SemanticMessage(role=MessageRole.SYSTEM, content=tuple(decoded))


def _decode_content(
    value: object,
    *,
    parameter: str,
    content_index: int,
    model: str,
    prior_calls: dict[str, list[ToolCall]],
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
) -> tuple[SemanticMessage, ...]:
    content = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(content, _CONTENT_FIELDS, protocol=_PROTOCOL, parameter=parameter)
    role_value = optional_string(
        content.get("role"), protocol=_PROTOCOL, parameter=f"{parameter}.role"
    )
    role_value = role_value or "user"
    if role_value not in {"user", "model"}:
        decode_reject(_PROTOCOL, f"{parameter}.role", "must be user or model")
    role = MessageRole.USER if role_value == "user" else MessageRole.ASSISTANT
    raw_parts = require_list(
        content.get("parts"), protocol=_PROTOCOL, parameter=f"{parameter}.parts"
    )
    messages: list[SemanticMessage] = []
    pending: list[Any] = []

    def flush() -> None:
        if pending:
            messages.append(SemanticMessage(role=role, content=tuple(pending)))
            pending.clear()

    for part_index, raw_part in enumerate(raw_parts):
        path = f"{parameter}.parts[{part_index}]"
        part = require_mapping(raw_part, protocol=_PROTOCOL, parameter=path)
        reject_unknown_keys(part, _PART_FIELDS, protocol=_PROTOCOL, parameter=path)
        payload_fields = _part_payload_fields(part)
        if len(payload_fields) != 1:
            decode_reject(
                _PROTOCOL,
                path,
                "part must contain exactly one of text, functionCall, functionResponse, "
                "inlineData, or fileData",
            )
        payload_field = payload_fields[0]
        thought = optional_bool(
            part.get("thought"), protocol=_PROTOCOL, parameter=f"{path}.thought"
        )
        signature = optional_string(
            part.get("thoughtSignature"),
            protocol=_PROTOCOL,
            parameter=f"{path}.thoughtSignature",
        )
        item_id = f"gemini-part-{content_index}-{part_index}"
        if payload_field == "text":
            text = require_string(part.get("text"), protocol=_PROTOCOL, parameter=f"{path}.text")
            if thought:
                if role is not MessageRole.ASSISTANT:
                    decode_reject(_PROTOCOL, path, "thought text requires model role")
                pending.append(
                    ReasoningSummary(
                        text=text,
                        opaque_state=_opaque_state(
                            signature,
                            item_id=item_id,
                            model=model,
                            origin_provider=origin_provider,
                            decode_opaque_state=decode_opaque_state,
                            parameter=path,
                        ),
                    )
                )
            else:
                if signature is not None:
                    decode_reject(
                        _PROTOCOL,
                        f"{path}.thoughtSignature",
                        "signature requires a thought or functionCall part",
                    )
                pending.append(TextContent(text))
        elif payload_field == "functionCall":
            if role is not MessageRole.ASSISTANT:
                decode_reject(_PROTOCOL, path, "functionCall requires model role")
            call = _decode_function_call(
                part.get("functionCall"),
                parameter=f"{path}.functionCall",
                generated_id=f"gemini-call-{content_index}-{part_index}",
                opaque_state=_opaque_state(
                    signature,
                    item_id=item_id,
                    model=model,
                    origin_provider=origin_provider,
                    decode_opaque_state=decode_opaque_state,
                    parameter=path,
                ),
            )
            if thought is not None:
                decode_reject(_PROTOCOL, f"{path}.thought", "is not valid on functionCall")
            pending.append(call)
            prior_calls.setdefault(call.name, []).append(call)
        elif payload_field == "functionResponse":
            if role is not MessageRole.USER:
                decode_reject(_PROTOCOL, path, "functionResponse requires user role")
            if thought is not None or signature is not None:
                decode_reject(_PROTOCOL, path, "functionResponse cannot carry thought metadata")
            flush()
            name, result = _decode_function_response(
                part.get("functionResponse"),
                parameter=f"{path}.functionResponse",
                prior_calls=prior_calls,
            )
            messages.append(SemanticMessage(role=MessageRole.TOOL, name=name, content=(result,)))
        elif payload_field == "inlineData":
            if thought is not None or signature is not None:
                decode_reject(_PROTOCOL, path, "inlineData cannot carry thought metadata")
            pending.append(
                _decode_inline_data(part.get("inlineData"), parameter=f"{path}.inlineData")
            )
        else:
            if thought is not None or signature is not None:
                decode_reject(_PROTOCOL, path, "fileData cannot carry thought metadata")
            pending.append(_decode_file_data(part.get("fileData"), parameter=f"{path}.fileData"))
    flush()
    if not messages:
        messages.append(SemanticMessage(role=role, content=()))
    return tuple(messages)


def _part_payload_fields(part: Mapping[str, Any]) -> list[str]:
    return [
        name
        for name in ("text", "functionCall", "functionResponse", "inlineData", "fileData")
        if part.get(name) is not None
    ]


def _opaque_state(
    signature: str | None,
    *,
    item_id: str,
    model: str,
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
    parameter: str,
) -> OpaqueState | None:
    if signature is None:
        return None
    if is_reasoning_capsule_carrier(signature):
        if decode_opaque_state is None:
            decode_reject(
                _PROTOCOL,
                parameter,
                "Router-Maestro reasoning capsule requires decoder context",
            )
        try:
            state = decode_opaque_state(
                signature,
                protocol=_PROTOCOL,
                model=model,
                item_id=item_id,
            )
        except ValueError:
            decode_reject(_PROTOCOL, parameter, "invalid Router-Maestro reasoning capsule")
        if not isinstance(state, OpaqueState):
            decode_reject(_PROTOCOL, parameter, "capsule decoder returned invalid state")
        return state
    return OpaqueState(
        origin_protocol=_PROTOCOL,
        origin_provider=origin_provider,
        origin_model=model,
        item_id=item_id,
        blob=signature,
    )


def _decode_function_call(
    value: object,
    *,
    parameter: str,
    generated_id: str,
    opaque_state: OpaqueState | None,
) -> ToolCall:
    call = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        call,
        frozenset({"id", "name", "args"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    explicit_id = optional_string(call.get("id"), protocol=_PROTOCOL, parameter=f"{parameter}.id")
    arguments = call.get("args", {})
    arguments = require_mapping(arguments, protocol=_PROTOCOL, parameter=f"{parameter}.args")
    return ToolCall(
        call_id=explicit_id or generated_id,
        item_id=explicit_id,
        name=require_string(
            call.get("name"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.name",
            allow_empty=False,
        ),
        arguments=arguments,
        opaque_state=opaque_state,
    )


def _decode_function_response(
    value: object,
    *,
    parameter: str,
    prior_calls: dict[str, list[ToolCall]],
) -> tuple[str, ToolResult]:
    response = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        response,
        frozenset({"id", "name", "response"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    name = require_string(
        response.get("name"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.name",
        allow_empty=False,
    )
    call_id = optional_string(response.get("id"), protocol=_PROTOCOL, parameter=f"{parameter}.id")
    if call_id is None:
        matches = prior_calls.get(name, [])
        if len(matches) != 1:
            decode_reject(
                _PROTOCOL,
                f"{parameter}.id",
                "missing id requires exactly one prior matching functionCall; "
                f"found {len(matches)}",
            )
        call_id = matches.pop().call_id
    else:
        matches = prior_calls.get(name, [])
        for index, call in enumerate(matches):
            if call.call_id == call_id:
                matches.pop(index)
                break
    structured = require_mapping(
        response.get("response", {}),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.response",
    )
    is_error = set(structured) == {"error"}
    return (
        name,
        ToolResult(
            call_id=call_id,
            structured_content=structured,
            is_error=is_error,
        ),
    )


def _decode_inline_data(value: object, *, parameter: str) -> ImageContent | FileContent:
    data = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        data,
        frozenset({"mimeType", "data", "displayName"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    media_type = require_string(
        data.get("mimeType"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.mimeType",
        allow_empty=False,
    )
    source = require_string(data.get("data"), protocol=_PROTOCOL, parameter=f"{parameter}.data")
    filename = optional_string(
        data.get("displayName"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.displayName",
    )
    if media_type.startswith("image/"):
        if filename is not None:
            decode_reject(_PROTOCOL, f"{parameter}.displayName", "image name is not modeled")
        return ImageContent(
            source=source,
            media_type=media_type,
            source_kind="inline_data",
        )
    return FileContent(
        source=source,
        filename=filename,
        media_type=media_type,
        source_kind="inline_data",
    )


def _decode_file_data(value: object, *, parameter: str) -> ImageContent | FileContent:
    data = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        data,
        frozenset({"mimeType", "fileUri", "displayName"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    media_type = optional_string(
        data.get("mimeType"), protocol=_PROTOCOL, parameter=f"{parameter}.mimeType"
    )
    source = require_string(
        data.get("fileUri"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.fileUri",
        allow_empty=False,
    )
    filename = optional_string(
        data.get("displayName"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.displayName",
    )
    if media_type is not None and media_type.startswith("image/"):
        if filename is not None:
            decode_reject(_PROTOCOL, f"{parameter}.displayName", "image name is not modeled")
        return ImageContent(
            source=source,
            media_type=media_type,
            source_kind="file_uri",
        )
    return FileContent(
        source=source,
        filename=filename,
        media_type=media_type,
        source_kind="file_uri",
    )


def _decode_tools(value: object) -> tuple[ToolDefinition, ...]:
    if value is None:
        return ()
    raw_tools = require_list(value, protocol=_PROTOCOL, parameter="tools")
    result = []
    for tool_index, raw_tool in enumerate(raw_tools):
        path = f"tools[{tool_index}]"
        tool = require_mapping(raw_tool, protocol=_PROTOCOL, parameter=path)
        reject_unknown_keys(
            tool,
            frozenset({"functionDeclarations"}),
            protocol=_PROTOCOL,
            parameter=path,
        )
        declarations = require_list(
            tool.get("functionDeclarations"),
            protocol=_PROTOCOL,
            parameter=f"{path}.functionDeclarations",
        )
        if not declarations:
            decode_reject(
                _PROTOCOL,
                f"{path}.functionDeclarations",
                "must contain at least one function",
            )
        for declaration_index, raw_declaration in enumerate(declarations):
            declaration_path = f"{path}.functionDeclarations[{declaration_index}]"
            declaration = require_mapping(
                raw_declaration, protocol=_PROTOCOL, parameter=declaration_path
            )
            reject_unknown_keys(
                declaration,
                frozenset({"name", "description", "parameters", "parametersJsonSchema"}),
                protocol=_PROTOCOL,
                parameter=declaration_path,
            )
            parameters = declaration.get("parameters")
            json_schema = declaration.get("parametersJsonSchema")
            if parameters is not None and json_schema is not None:
                decode_reject(
                    _PROTOCOL,
                    declaration_path,
                    "parameters and parametersJsonSchema are mutually exclusive",
                )
            schema = parameters if parameters is not None else json_schema
            if schema is None:
                schema = {"type": "object", "properties": {}}
            schema = require_mapping(
                schema, protocol=_PROTOCOL, parameter=f"{declaration_path}.parameters"
            )
            result.append(
                ToolDefinition(
                    name=require_string(
                        declaration.get("name"),
                        protocol=_PROTOCOL,
                        parameter=f"{declaration_path}.name",
                        allow_empty=False,
                    ),
                    description=optional_string(
                        declaration.get("description"),
                        protocol=_PROTOCOL,
                        parameter=f"{declaration_path}.description",
                    ),
                    input_schema=_normalize_schema_types(schema),
                )
            )
    return tuple(result)


def _normalize_schema_types(value: object) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key == "type" and isinstance(item, str):
                if item.upper() == "TYPE_UNSPECIFIED":
                    continue
                result[key] = item.lower()
            elif key == "properties" and isinstance(item, Mapping):
                result[key] = {
                    name: _normalize_schema_types(schema) for name, schema in item.items()
                }
            elif key in {"items", "additionalProperties", "not"}:
                result[key] = _normalize_schema_types(item)
            elif key in {"anyOf", "oneOf", "allOf"} and isinstance(item, list | tuple):
                result[key] = [_normalize_schema_types(schema) for schema in item]
            else:
                result[key] = thaw_json(item)
        return result
    if isinstance(value, list | tuple):
        return [_normalize_schema_types(item) for item in value]
    return value


def _decode_tool_config(value: object) -> ToolChoice | None:
    if value is None:
        return None
    config = require_mapping(value, protocol=_PROTOCOL, parameter="toolConfig")
    reject_unknown_keys(
        config,
        frozenset({"functionCallingConfig"}),
        protocol=_PROTOCOL,
        parameter="toolConfig",
    )
    calling = require_mapping(
        config.get("functionCallingConfig"),
        protocol=_PROTOCOL,
        parameter="toolConfig.functionCallingConfig",
    )
    reject_unknown_keys(
        calling,
        frozenset({"mode", "allowedFunctionNames"}),
        protocol=_PROTOCOL,
        parameter="toolConfig.functionCallingConfig",
    )
    mode = require_string(
        calling.get("mode"),
        protocol=_PROTOCOL,
        parameter="toolConfig.functionCallingConfig.mode",
    ).upper()
    allowed = _decode_string_list(
        calling.get("allowedFunctionNames"),
        "toolConfig.functionCallingConfig.allowedFunctionNames",
    )
    if mode == "VALIDATED":
        decode_reject(
            _PROTOCOL,
            "toolConfig.functionCallingConfig.mode",
            "VALIDATED has no exact cross-protocol mapping",
        )
    if mode == "AUTO":
        if allowed:
            decode_reject(
                _PROTOCOL,
                "toolConfig.functionCallingConfig.allowedFunctionNames",
                f"cannot preserve an allowlist with mode {mode}",
            )
        return ToolChoice("auto")
    if mode == "NONE":
        if allowed:
            decode_reject(
                _PROTOCOL,
                "toolConfig.functionCallingConfig.allowedFunctionNames",
                "cannot combine an allowlist with NONE",
            )
        return ToolChoice("none")
    if mode == "ANY":
        if len(allowed) == 1:
            return ToolChoice("function", name=allowed[0])
        if not allowed:
            return ToolChoice("required")
        decode_reject(
            _PROTOCOL,
            "toolConfig.functionCallingConfig.allowedFunctionNames",
            "semantic tool choice cannot preserve multiple allowed names",
        )
    decode_reject(
        _PROTOCOL,
        "toolConfig.functionCallingConfig.mode",
        f"unsupported mode {mode!r}",
    )


def _decode_generation_config(value: object) -> dict[str, Any]:
    result: dict[str, Any] = {
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_output_tokens": None,
        "stop_sequences": (),
        "candidate_count": None,
        "response_mime_type": None,
        "frequency_penalty": None,
        "presence_penalty": None,
        "reasoning": None,
        "structured_output": None,
        "explicit_fields": frozenset(),
    }
    if value is None:
        return result
    config = require_mapping(value, protocol=_PROTOCOL, parameter="generationConfig")
    reject_unknown_keys(
        config, _GENERATION_FIELDS, protocol=_PROTOCOL, parameter="generationConfig"
    )
    result.update(
        temperature=optional_number(
            config.get("temperature"),
            protocol=_PROTOCOL,
            parameter="generationConfig.temperature",
        ),
        top_p=optional_number(
            config.get("topP"), protocol=_PROTOCOL, parameter="generationConfig.topP"
        ),
        top_k=optional_int(
            config.get("topK"), protocol=_PROTOCOL, parameter="generationConfig.topK"
        ),
        max_output_tokens=optional_int(
            config.get("maxOutputTokens"),
            protocol=_PROTOCOL,
            parameter="generationConfig.maxOutputTokens",
        ),
        stop_sequences=_decode_string_list(
            config.get("stopSequences"), "generationConfig.stopSequences"
        ),
        candidate_count=optional_int(
            config.get("candidateCount"),
            protocol=_PROTOCOL,
            parameter="generationConfig.candidateCount",
        ),
        response_mime_type=optional_string(
            config.get("responseMimeType"),
            protocol=_PROTOCOL,
            parameter="generationConfig.responseMimeType",
        ),
        frequency_penalty=optional_number(
            config.get("frequencyPenalty"),
            protocol=_PROTOCOL,
            parameter="generationConfig.frequencyPenalty",
        ),
        presence_penalty=optional_number(
            config.get("presencePenalty"),
            protocol=_PROTOCOL,
            parameter="generationConfig.presencePenalty",
        ),
        reasoning=_decode_thinking_config(config.get("thinkingConfig")),
        explicit_fields=frozenset(config),
    )
    if result["candidate_count"] not in {None, 1}:
        decode_reject(
            _PROTOCOL,
            "generationConfig.candidateCount",
            "must equal 1",
        )
    response_schema = config.get("responseSchema")
    response_json_schema = config.get("responseJsonSchema")
    if response_schema is not None and response_json_schema is not None:
        decode_reject(
            _PROTOCOL,
            "generationConfig",
            "responseSchema and responseJsonSchema are mutually exclusive",
        )
    schema = response_json_schema if response_json_schema is not None else response_schema
    if schema is not None:
        schema = require_mapping(
            schema,
            protocol=_PROTOCOL,
            parameter=(
                "generationConfig.responseJsonSchema"
                if response_json_schema is not None
                else "generationConfig.responseSchema"
            ),
        )
        result["structured_output"] = {
            "type": "json_schema",
            "schema": _normalize_schema_types(schema),
        }
    for name in ("top_k", "max_output_tokens", "candidate_count"):
        number = result[name]
        if number is not None and number < 0:
            field = {
                "top_k": "topK",
                "max_output_tokens": "maxOutputTokens",
                "candidate_count": "candidateCount",
            }[name]
            decode_reject(_PROTOCOL, f"generationConfig.{field}", "cannot be negative")
    return result


def _decode_thinking_config(value: object) -> ReasoningConfig | None:
    if value is None:
        return None
    config = require_mapping(value, protocol=_PROTOCOL, parameter="generationConfig.thinkingConfig")
    reject_unknown_keys(
        config,
        frozenset({"thinkingBudget", "includeThoughts"}),
        protocol=_PROTOCOL,
        parameter="generationConfig.thinkingConfig",
    )
    budget = optional_int(
        config.get("thinkingBudget"),
        protocol=_PROTOCOL,
        parameter="generationConfig.thinkingConfig.thinkingBudget",
    )
    include = optional_bool(
        config.get("includeThoughts"),
        protocol=_PROTOCOL,
        parameter="generationConfig.thinkingConfig.includeThoughts",
    )
    if budget is not None and budget < -1:
        decode_reject(
            _PROTOCOL,
            "generationConfig.thinkingConfig.thinkingBudget",
            "must be -1, 0, or positive",
        )
    if budget == 0:
        if include is True:
            decode_reject(
                _PROTOCOL,
                "generationConfig.thinkingConfig",
                "disabled thinking cannot include thoughts",
            )
        return ReasoningConfig(enabled=False)
    if budget == -1:
        return ReasoningConfig(enabled=True, effort="adaptive")
    return ReasoningConfig(
        enabled=True if include is True or budget is not None else None,
        budget_tokens=budget,
    )


def _decode_string_list(value: object, parameter: str) -> tuple[str, ...]:
    if value is None:
        return ()
    items = require_list(value, protocol=_PROTOCOL, parameter=parameter)
    return tuple(
        require_string(item, protocol=_PROTOCOL, parameter=f"{parameter}[{index}]")
        for index, item in enumerate(items)
    )


def _encode_request(
    request: SemanticRequest,
    *,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    if request.provider_extensions:
        key = sorted(request.provider_extensions)[0]
        reject(_PROTOCOL, key, "provider extension is not portable")
    for name, value in (
        ("parallel_tool_calls", request.parallel_tool_calls),
        ("user", request.user),
        ("service_tier", request.service_tier),
    ):
        if value is not None:
            reject(_PROTOCOL, name, "field is not supported by Gemini")
    if request.metadata:
        key = sorted(request.metadata)[0]
        reject(_PROTOCOL, f"metadata.{key}", "Gemini requests lack metadata")

    system_messages: list[SemanticMessage] = []
    conversation: list[SemanticMessage | object] = []
    seen_conversation = False
    for index, item in enumerate(request.input):
        if isinstance(item, SemanticMessage) and item.role is MessageRole.SYSTEM:
            if seen_conversation:
                reject(
                    _PROTOCOL,
                    f"input[{index}]",
                    "Gemini system instructions must precede conversation content",
                )
            system_messages.append(item)
        else:
            seen_conversation = True
            conversation.append(item)

    call_names = _collect_call_names(conversation)
    payload: dict[str, Any] = {
        "contents": [
            _encode_content(
                item,
                parameter=f"input[{index}]",
                model=request.model,
                call_names=call_names,
                encode_opaque_state=encode_opaque_state,
            )
            for index, item in enumerate(conversation)
        ]
    }
    if system_messages:
        payload["systemInstruction"] = _encode_system_instruction(system_messages)
    if request.tools or "tools" in request.explicit_fields:
        payload["tools"] = [_encode_tools(request.tools)] if request.tools else []
    tool_config = _encode_tool_config(request.tool_choice)
    if tool_config is not None:
        payload["toolConfig"] = tool_config
    generation = _encode_generation_config(request)
    if generation or "generationConfig" in request.explicit_fields:
        payload["generationConfig"] = generation
    return payload


def _collect_call_names(items: list[SemanticMessage | object]) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in items:
        parts = item.content if isinstance(item, SemanticMessage) else (item,)
        for part in parts:
            if not isinstance(part, ToolCall):
                continue
            existing = names.get(part.call_id)
            if existing is not None and existing != part.name:
                reject(
                    _PROTOCOL,
                    "input",
                    f"call id {part.call_id!r} is reused by multiple function names",
                )
            names[part.call_id] = part.name
    return names


def _encode_system_instruction(messages: list[SemanticMessage]) -> dict[str, Any]:
    parts = []
    for message_index, message in enumerate(messages):
        if message.name is not None or message.item_id is not None or message.status is not None:
            reject(
                _PROTOCOL,
                f"systemInstruction[{message_index}]",
                "system message metadata is not supported",
            )
        for part_index, part in enumerate(message.content):
            if not isinstance(part, TextContent):
                reject(
                    _PROTOCOL,
                    f"systemInstruction[{message_index}].content[{part_index}]",
                    "system instructions support text only",
                )
            parts.append({"text": part.text})
    return {"parts": parts}


def _encode_content(
    item: object,
    *,
    parameter: str,
    model: str,
    call_names: Mapping[str, str],
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    if isinstance(item, ToolCall):
        item = SemanticMessage(role=MessageRole.ASSISTANT, content=(item,))
    elif isinstance(item, ToolResult):
        item = SemanticMessage(role=MessageRole.TOOL, content=(item,))
    elif isinstance(item, TextContent | ImageContent | FileContent):
        item = SemanticMessage(role=MessageRole.USER, content=(item,))
    elif isinstance(item, ReasoningSummary | RefusalContent):
        item = SemanticMessage(role=MessageRole.ASSISTANT, content=(item,))
    if not isinstance(item, SemanticMessage):
        reject(_PROTOCOL, parameter, f"unsupported input item {type(item).__name__}")
    if item.item_id is not None or item.status is not None:
        reject(_PROTOCOL, parameter, "Gemini contents cannot carry item metadata")
    if item.role is MessageRole.SYSTEM:
        reject(_PROTOCOL, parameter, "system instructions must use systemInstruction")
    role = "model" if item.role is MessageRole.ASSISTANT else "user"
    parts = []
    for index, part in enumerate(item.content):
        path = f"{parameter}.content[{index}]"
        if isinstance(part, TextContent):
            parts.append({"text": part.text})
        elif isinstance(part, ImageContent | FileContent):
            if role != "user":
                reject(_PROTOCOL, path, "binary/file input requires user role")
            parts.append(_encode_data_part(part, parameter=path))
        elif isinstance(part, ToolCall):
            if role != "model":
                reject(_PROTOCOL, path, "functionCall requires assistant role")
            parts.append(
                _encode_function_call(
                    part,
                    parameter=path,
                    model=model,
                    encode_opaque_state=encode_opaque_state,
                )
            )
        elif isinstance(part, ToolResult):
            if role != "user":
                reject(_PROTOCOL, path, "functionResponse requires tool/user role")
            name = item.name or call_names.get(part.call_id)
            if not name:
                reject(
                    _PROTOCOL,
                    f"{path}.name",
                    "function response needs message.name or a prior matching call",
                )
            parts.append(_encode_function_response(part, name=name, parameter=path))
        elif isinstance(part, ReasoningSummary):
            if role != "model":
                reject(_PROTOCOL, path, "thought requires assistant role")
            parts.append(
                _encode_reasoning_part(
                    part,
                    parameter=path,
                    model=model,
                    encode_opaque_state=encode_opaque_state,
                )
            )
        elif isinstance(part, RefusalContent):
            reject(_PROTOCOL, path, "Gemini has no distinct refusal content block")
        else:  # pragma: no cover - closed semantic union
            reject(_PROTOCOL, path, f"unsupported content {type(part).__name__}")
    result: dict[str, Any] = {"role": role, "parts": parts}
    return result


def _encode_data_part(
    part: ImageContent | FileContent,
    *,
    parameter: str,
) -> dict[str, Any]:
    source_kind = part.source_kind
    if source_kind is None:
        if isinstance(part.source, bytes):
            source_kind = "inline_data"
        else:
            reject(_PROTOCOL, f"{parameter}.source_kind", "string data source is ambiguous")
    display_name = part.filename if isinstance(part, FileContent) else None
    if source_kind in {"inline_data", "base64"}:
        if part.media_type is None:
            reject(_PROTOCOL, f"{parameter}.media_type", "inlineData requires a MIME type")
        data = (
            base64.b64encode(part.source).decode("ascii")
            if isinstance(part.source, bytes)
            else part.source
        )
        inline: dict[str, Any] = {"mimeType": part.media_type, "data": data}
        if display_name is not None:
            inline["displayName"] = display_name
        return {"inlineData": inline}
    if source_kind in {"file_uri", "url"}:
        if isinstance(part.source, bytes):
            reject(_PROTOCOL, parameter, "fileData URI must be text")
        file_data: dict[str, Any] = {"fileUri": part.source}
        if part.media_type is not None:
            file_data["mimeType"] = part.media_type
        if display_name is not None:
            file_data["displayName"] = display_name
        return {"fileData": file_data}
    reject(_PROTOCOL, f"{parameter}.source_kind", f"unsupported kind {source_kind!r}")


def _encode_function_call(
    call: ToolCall,
    *,
    parameter: str,
    model: str,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    if call.kind != "function":
        reject(_PROTOCOL, f"{parameter}.kind", "Gemini supports function calls only")
    if call.namespace is not None:
        reject(_PROTOCOL, f"{parameter}.namespace", "Gemini calls lack namespaces")
    function_call: dict[str, Any] = {
        "name": call.name,
        "args": thaw_json(call.arguments),
    }
    if call.call_id:
        function_call["id"] = call.call_id
    payload: dict[str, Any] = {"functionCall": function_call}
    if call.opaque_state is not None:
        payload["thoughtSignature"] = _encode_opaque_blob(
            call.opaque_state,
            parameter=parameter,
            model=model,
            encode_opaque_state=encode_opaque_state,
        )
    return payload


def _encode_function_response(
    result: ToolResult,
    *,
    name: str,
    parameter: str,
) -> dict[str, Any]:
    if result.kind != "function":
        reject(
            _PROTOCOL,
            f"{parameter}.kind",
            f"unsupported tool result kind {result.kind!r}",
        )
    if result.namespace is not None:
        reject(
            _PROTOCOL,
            f"{parameter}.namespace",
            "Gemini function responses lack namespaces",
        )
    if result.content and result.structured_content is not None:
        reject(_PROTOCOL, parameter, "cannot combine content and structured function output")
    if result.content:
        texts = []
        for index, part in enumerate(result.content):
            if not isinstance(part, TextContent):
                reject(
                    _PROTOCOL,
                    f"{parameter}.content[{index}]",
                    "Gemini function responses support structured/text data only",
                )
            texts.append(part.text)
        response: dict[str, Any] = {"result": "".join(texts)}
    elif result.structured_content is None:
        response = {}
    else:
        raw = thaw_json(result.structured_content)
        if not isinstance(raw, Mapping):
            response = {"result": raw}
        else:
            response = dict(raw)
    if result.is_error and set(response) != {"error"}:
        response = {"error": response}
    return {
        "functionResponse": {
            "id": result.call_id,
            "name": name,
            "response": response,
        }
    }


def _encode_reasoning_part(
    reasoning: ReasoningSummary,
    *,
    parameter: str,
    model: str,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": reasoning.text, "thought": True}
    if reasoning.opaque_state is not None:
        payload["thoughtSignature"] = _encode_opaque_blob(
            reasoning.opaque_state,
            parameter=parameter,
            model=model,
            encode_opaque_state=encode_opaque_state,
        )
    return payload


def _encode_opaque_blob(
    state: OpaqueState,
    *,
    parameter: str,
    model: str,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> str:
    if state.origin_protocol is _PROTOCOL and isinstance(state.blob, str):
        if state.origin_model != model:
            reject(_PROTOCOL, parameter, "opaque state model provenance does not match")
        return state.blob
    if encode_opaque_state is None:
        reject(_PROTOCOL, parameter, "foreign opaque state requires capsule encoder context")
    try:
        capsule = encode_opaque_state(
            state,
            protocol=_PROTOCOL,
            model=state.origin_model,
            item_id=state.item_id,
        )
    except ValueError:
        reject(_PROTOCOL, parameter, "opaque state could not be sealed")
    if not isinstance(capsule, str) or not capsule.startswith("rmr1."):
        reject(_PROTOCOL, parameter, "capsule encoder returned invalid state")
    return capsule


def _encode_tools(tools: tuple[ToolDefinition, ...]) -> dict[str, Any]:
    declarations = []
    for tool in tools:
        declaration: dict[str, Any] = {
            "name": tool.name,
            "parametersJsonSchema": thaw_json(tool.input_schema),
        }
        if tool.description is not None:
            declaration["description"] = tool.description
        if tool.strict is not None:
            reject(_PROTOCOL, f"tools.{tool.name}.strict", "Gemini tools lack strict mode")
        declarations.append(declaration)
    return {"functionDeclarations": declarations}


def _encode_tool_config(choice: ToolChoice | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    calling: dict[str, Any]
    if choice.mode == "auto":
        calling = {"mode": "AUTO"}
    elif choice.mode == "none":
        calling = {"mode": "NONE"}
    elif choice.mode == "required":
        calling = {"mode": "ANY"}
    elif choice.mode == "function" and choice.name:
        calling = {"mode": "ANY", "allowedFunctionNames": [choice.name]}
    else:
        reject(_PROTOCOL, "tool_choice.mode", f"unsupported mode {choice.mode!r}")
    return {"functionCallingConfig": calling}


def _encode_generation_config(request: SemanticRequest) -> dict[str, Any]:
    config: dict[str, Any] = {}
    fields = (
        ("temperature", "temperature", request.temperature),
        ("topP", "top_p", request.top_p),
        ("topK", "top_k", request.top_k),
        ("maxOutputTokens", "max_output_tokens", request.max_output_tokens),
        ("candidateCount", "candidate_count", request.candidate_count),
        ("responseMimeType", "response_mime_type", request.response_mime_type),
        ("frequencyPenalty", "frequency_penalty", request.frequency_penalty),
        ("presencePenalty", "presence_penalty", request.presence_penalty),
    )
    for wire_name, semantic_name, value in fields:
        if value is not None or f"generationConfig.{wire_name}" in request.explicit_fields:
            config[wire_name] = value
    if request.stop_sequences or "generationConfig.stopSequences" in request.explicit_fields:
        config["stopSequences"] = list(request.stop_sequences)
    if request.structured_output is not None:
        config["responseJsonSchema"] = _gemini_output_schema(request.structured_output)
    if request.reasoning is not None:
        config["thinkingConfig"] = _encode_thinking_config(request.reasoning)
    return config


def _encode_thinking_config(reasoning: ReasoningConfig) -> dict[str, Any]:
    if reasoning.effort not in {None, "adaptive"}:
        reject(_PROTOCOL, "reasoning.effort", "Gemini has no reasoning effort tier")
    if reasoning.enabled is False:
        if reasoning.budget_tokens is not None:
            reject(_PROTOCOL, "reasoning.budget_tokens", "disabled reasoning has no budget")
        return {"thinkingBudget": 0, "includeThoughts": False}
    config: dict[str, Any] = {}
    if reasoning.effort == "adaptive":
        if reasoning.budget_tokens is not None:
            reject(_PROTOCOL, "reasoning.budget_tokens", "adaptive reasoning has no budget")
        config["thinkingBudget"] = -1
    elif reasoning.budget_tokens is not None:
        config["thinkingBudget"] = reasoning.budget_tokens
    if reasoning.enabled is True:
        config["includeThoughts"] = True
    return config


def _gemini_output_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = thaw_json(value)
    if "format" in raw and isinstance(raw["format"], Mapping):
        raw = dict(raw["format"])
    if raw.get("type") == "json_schema" and isinstance(raw.get("json_schema"), Mapping):
        raw = dict(raw["json_schema"])
    if raw.get("type") == "json_schema" and isinstance(raw.get("schema"), Mapping):
        return dict(raw["schema"])
    if isinstance(raw.get("schema"), Mapping):
        return dict(raw["schema"])
    if raw.get("type") in {"object", "array", "string", "number", "integer", "boolean"}:
        return raw
    reject(_PROTOCOL, "structured_output", "unsupported structured output shape")


def _decode_response(
    payload: Mapping[str, Any],
    *,
    fallback_model: str | None,
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
) -> SemanticResponse:
    body = require_mapping(payload, protocol=_PROTOCOL, parameter="response")
    reject_unknown_keys(
        body,
        frozenset({"candidates", "usageMetadata", "modelVersion", "promptFeedback"}),
        protocol=_PROTOCOL,
        parameter="response",
    )
    model_value = body.get("modelVersion")
    model = (
        optional_string(model_value, protocol=_PROTOCOL, parameter="response.modelVersion")
        if model_value is not None
        else fallback_model
    )
    if not model:
        decode_reject(
            _PROTOCOL,
            "response.modelVersion",
            "response lacks modelVersion and runtime has no default_model",
        )
    prompt_feedback, prompt_terminal = _decode_prompt_feedback(
        body.get("promptFeedback"),
        parameter="response.promptFeedback",
    )
    candidates_value = body.get("candidates")
    if candidates_value is None and prompt_terminal is not None:
        candidates: list[Any] | tuple[Any, ...] = ()
    else:
        candidates = require_list(
            candidates_value,
            protocol=_PROTOCOL,
            parameter="response.candidates",
        )
    if not candidates and prompt_terminal is not None:
        return SemanticResponse(
            model=model,
            output=(),
            usage=_decode_usage(body.get("usageMetadata")),
            terminal=prompt_terminal,
            metadata={"gemini_prompt_feedback": prompt_feedback or {}},
        )
    if len(candidates) != 1:
        decode_reject(_PROTOCOL, "response.candidates", "exactly one candidate is required")
    if prompt_terminal is not None:
        decode_reject(
            _PROTOCOL,
            "response.promptFeedback",
            "blocked prompt cannot include an output candidate",
        )
    candidate = require_mapping(
        candidates[0], protocol=_PROTOCOL, parameter="response.candidates[0]"
    )
    reject_unknown_keys(
        candidate,
        frozenset({"content", "finishReason", "index"}),
        protocol=_PROTOCOL,
        parameter="response.candidates[0]",
    )
    index = optional_int(
        candidate.get("index", 0),
        protocol=_PROTOCOL,
        parameter="response.candidates[0].index",
    )
    if index != 0:
        decode_reject(_PROTOCOL, "response.candidates[0].index", "must be zero")
    content = require_mapping(
        candidate.get("content"),
        protocol=_PROTOCOL,
        parameter="response.candidates[0].content",
    )
    messages = _decode_content(
        content,
        parameter="response.candidates[0].content",
        content_index=0,
        model=model,
        prior_calls={},
        origin_provider=origin_provider,
        decode_opaque_state=decode_opaque_state,
    )
    if len(messages) != 1 or messages[0].role is not MessageRole.ASSISTANT:
        decode_reject(
            _PROTOCOL,
            "response.candidates[0].content",
            "candidate content must be one model message",
        )
    finish = optional_string(
        candidate.get("finishReason"),
        protocol=_PROTOCOL,
        parameter="response.candidates[0].finishReason",
    )
    semantic_finish = None
    if finish is not None:
        semantic_finish = _FINISH_TO_SEMANTIC.get(finish)
        if semantic_finish is None:
            decode_reject(
                _PROTOCOL,
                "response.candidates[0].finishReason",
                f"unsupported finish reason {finish!r}",
            )
    return SemanticResponse(
        model=model,
        output=messages,
        usage=_decode_usage(body.get("usageMetadata")),
        terminal=TerminalMetadata(
            finish_reason=semantic_finish,
            response_status=(
                "incomplete" if semantic_finish in {"length", "content_filter"} else "completed"
            ),
        ),
        metadata=(
            {"gemini_prompt_feedback": prompt_feedback} if prompt_feedback is not None else {}
        ),
    )


def _decode_prompt_feedback(
    value: object,
    *,
    parameter: str,
) -> tuple[dict[str, Any] | None, TerminalMetadata | None]:
    if value is None:
        return None, None
    feedback = thaw_json(require_mapping(value, protocol=_PROTOCOL, parameter=parameter))
    block_reason_value = feedback.get("blockReason")
    if block_reason_value is None:
        return feedback, None
    require_string(
        block_reason_value,
        protocol=_PROTOCOL,
        parameter=f"{parameter}.blockReason",
        allow_empty=False,
    )
    if "blockReasonMessage" in feedback:
        optional_string(
            feedback.get("blockReasonMessage"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.blockReasonMessage",
        )
    return feedback, TerminalMetadata(
        finish_reason="content_filter",
        response_status="incomplete",
        incomplete_details={"reason": "content_filter"},
        transport_termination="explicit_terminal",
    )


def _decode_usage(value: object) -> Usage | None:
    if value is None:
        return None
    usage = require_mapping(value, protocol=_PROTOCOL, parameter="response.usageMetadata")
    reject_unknown_keys(
        usage,
        frozenset(
            {
                "promptTokenCount",
                "candidatesTokenCount",
                "totalTokenCount",
                "cachedContentTokenCount",
                "thoughtsTokenCount",
            }
        ),
        protocol=_PROTOCOL,
        parameter="response.usageMetadata",
    )
    return Usage(
        input_tokens=optional_int(
            usage.get("promptTokenCount"),
            protocol=_PROTOCOL,
            parameter="response.usageMetadata.promptTokenCount",
        ),
        output_tokens=optional_int(
            usage.get("candidatesTokenCount"),
            protocol=_PROTOCOL,
            parameter="response.usageMetadata.candidatesTokenCount",
        ),
        total_tokens=optional_int(
            usage.get("totalTokenCount"),
            protocol=_PROTOCOL,
            parameter="response.usageMetadata.totalTokenCount",
        ),
        cached_input_tokens=optional_int(
            usage.get("cachedContentTokenCount"),
            protocol=_PROTOCOL,
            parameter="response.usageMetadata.cachedContentTokenCount",
        ),
        reasoning_tokens=optional_int(
            usage.get("thoughtsTokenCount"),
            protocol=_PROTOCOL,
            parameter="response.usageMetadata.thoughtsTokenCount",
        ),
    )


def _encode_response(
    response: SemanticResponse,
    *,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    parts = []
    for item_index, item in enumerate(response.output):
        content = item.content if isinstance(item, SemanticMessage) else (item,)
        if isinstance(item, SemanticMessage) and item.role is not MessageRole.ASSISTANT:
            reject(_PROTOCOL, f"response.output[{item_index}].role", "must be assistant")
        for part_index, part in enumerate(content):
            path = f"response.output[{item_index}].content[{part_index}]"
            if isinstance(part, TextContent):
                parts.append({"text": part.text})
            elif isinstance(part, ImageContent | FileContent):
                parts.append(_encode_data_part(part, parameter=path))
            elif isinstance(part, ToolCall):
                parts.append(
                    _encode_function_call(
                        part,
                        parameter=path,
                        model=response.model,
                        encode_opaque_state=encode_opaque_state,
                    )
                )
            elif isinstance(part, ReasoningSummary):
                parts.append(
                    _encode_reasoning_part(
                        part,
                        parameter=path,
                        model=response.model,
                        encode_opaque_state=encode_opaque_state,
                    )
                )
            elif isinstance(part, RefusalContent):
                reject(_PROTOCOL, path, "Gemini has no distinct refusal content block")
            else:
                reject(_PROTOCOL, path, f"unsupported response output {type(part).__name__}")

    finish = None
    if response.terminal is not None:
        terminal = response.terminal
        if terminal.stop_sequence is not None:
            reject(_PROTOCOL, "response.terminal.stop_sequence", "Gemini lacks stop sequence")
        if terminal.error_code is not None or terminal.error_message is not None:
            reject(_PROTOCOL, "response.terminal", "Gemini candidate cannot carry an error")
        status_error = _terminal_status_error(terminal)
        if status_error is not None:
            reject(
                _PROTOCOL,
                "response.terminal.response_status",
                f"Gemini candidates cannot represent {terminal.response_status!r} status",
            )
        if terminal.transport_status is not None:
            reject(
                _PROTOCOL,
                "response.terminal.transport_status",
                "transport status is not a candidate field",
            )
        if terminal.finish_reason is not None:
            finish = _FINISH_FROM_SEMANTIC.get(terminal.finish_reason)
            if finish is None:
                reject(
                    _PROTOCOL,
                    "response.terminal.finish_reason",
                    f"unsupported reason {terminal.finish_reason!r}",
                )
    if finish is None:
        finish = "STOP"
    candidate: dict[str, Any] = {
        "content": {"role": "model", "parts": parts},
        "finishReason": finish,
        "index": 0,
    }
    payload: dict[str, Any] = {
        "candidates": [candidate],
        "modelVersion": response.model,
    }
    usage = _encode_usage(response.usage)
    if usage is not None:
        payload["usageMetadata"] = usage
    projectable_metadata = {"created", "created_at", "object"}
    unknown_metadata = sorted(set(response.metadata) - projectable_metadata)
    if unknown_metadata:
        key = unknown_metadata[0]
        reject(_PROTOCOL, f"response.metadata.{key}", "metadata is not portable")
    created = response.metadata.get("created")
    created_at = response.metadata.get("created_at")
    if created is not None and (not isinstance(created, int) or isinstance(created, bool)):
        reject(_PROTOCOL, "response.metadata.created", "must be an integer")
    if created_at is not None and (
        not isinstance(created_at, int | float) or isinstance(created_at, bool)
    ):
        reject(_PROTOCOL, "response.metadata.created_at", "must be a number")
    if created is not None and created_at is not None and created != created_at:
        reject(_PROTOCOL, "response.metadata.created_at", "conflicts with created")
    source_object = response.metadata.get("object")
    if source_object is not None and source_object not in {"chat.completion", "response"}:
        reject(_PROTOCOL, "response.metadata.object", "metadata is not portable")
    return payload


def _encode_usage(usage: Usage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    if usage.mode is not UsageMode.SNAPSHOT:
        reject(_PROTOCOL, "response.usage.mode", "non-stream usage must be a snapshot")
    payload = {}
    for name, value in (
        ("promptTokenCount", usage.input_tokens),
        ("candidatesTokenCount", usage.output_tokens),
        ("totalTokenCount", usage.total_tokens),
        ("cachedContentTokenCount", usage.cached_input_tokens),
        ("thoughtsTokenCount", usage.reasoning_tokens),
    ):
        if value is not None:
            payload[name] = value
    return payload


__all__ = ["GeminiRuntime", "GeminiStreamDecoder", "GeminiStreamEncoder"]
