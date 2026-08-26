"""Concrete Anthropic Messages protocol runtime.

The runtime is intentionally used only on semantic conversion paths.  Native
Anthropic-to-Anthropic dispatch keeps the original payload in ``RequestEnvelope``
and never calls this codec.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
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

_PROTOCOL = WireProtocol.ANTHROPIC_MESSAGES
_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "system",
        "metadata",
        "stop_sequences",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "tools",
        "tool_choice",
        "thinking",
        "context_management",
        "service_tier",
        "output_config",
    }
)
_STOP_TO_SEMANTIC = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
_STOP_FROM_SEMANTIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "tool_use": "tool_use",
    "refusal": "refusal",
}
_NON_SUCCESS_TERMINAL_MESSAGES = {
    "failed": "Upstream response failed",
    "cancelled": "Upstream response was cancelled",
    "unknown": "Upstream response ended with an unknown status",
}
_NON_SUCCESS_TRANSPORT_STATUSES = {
    "exception": "failed",
    "client_cancelled": "cancelled",
    "unexpected_eof": "unknown",
}
_RESPONSE_METADATA_FIELDS = frozenset(
    {
        "cache_creation_input_tokens",
        "service_tier",
        # Source-protocol envelope metadata has no Anthropic response slot and
        # carries no generated semantics. Keep this closed set explicit.
        "created",
        "created_at",
        "object",
    }
)


def _terminal_status_error(
    terminal: TerminalMetadata,
    *,
    parameter: str,
) -> tuple[str, str] | None:
    """Return a safe Anthropic error for a terminal that is not successful output."""
    status = terminal.response_status
    if status is None:
        status = _NON_SUCCESS_TRANSPORT_STATUSES.get(terminal.transport_termination or "")
    if status in _NON_SUCCESS_TERMINAL_MESSAGES:
        return "api_error", _NON_SUCCESS_TERMINAL_MESSAGES[status]
    if status not in {None, "completed", "incomplete"}:
        reject(_PROTOCOL, parameter, f"unsupported value {status!r}")
    return None


class AnthropicMessagesRuntime:
    """Strict Messages wire codec used only when semantic conversion is needed."""

    protocol = _PROTOCOL

    def __init__(
        self,
        *,
        origin_provider: str | None = None,
        decode_opaque_state: OpaqueStateDecodeHook | None = None,
        encode_opaque_state: OpaqueStateEncodeHook | None = None,
    ) -> None:
        self.origin_provider = origin_provider
        self.decode_opaque_state = decode_opaque_state
        self.encode_opaque_state = encode_opaque_state
        self._stream_decoder: ContextVar[AnthropicStreamDecoder | None] = ContextVar(
            f"anthropic_stream_decoder_{id(self)}",
            default=None,
        )
        self._stream_encoder: ContextVar[AnthropicStreamEncoder | None] = ContextVar(
            f"anthropic_stream_encoder_{id(self)}",
            default=None,
        )

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        messages = payload.get("messages")
        tool_choice = payload.get("tool_choice")
        images, files, message_reasoning, opaque_carriers = _inspect_message_features(messages)
        return RequestManifest(
            protocol=self.protocol,
            model=payload.get("model") if isinstance(payload.get("model"), str) else None,
            stream=payload.get("stream") is True,
            tools=bool(payload.get("tools")),
            images=images,
            files=files,
            reasoning=bool(payload.get("thinking")) or message_reasoning,
            parallel_tools=isinstance(tool_choice, Mapping)
            and tool_choice.get("disable_parallel_tool_use") is False,
            reasoning_capsules=tuple(
                carrier for carrier in opaque_carriers if is_reasoning_capsule_carrier(carrier)
            ),
            opaque_continuation=bool(opaque_carriers),
        )

    async def decode_request(self, payload: Mapping[str, Any]) -> SemanticRequest:
        return _decode_request(
            payload,
            origin_provider=self.origin_provider,
            decode_opaque_state=self.decode_opaque_state,
        )

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        return _encode_request(request, encode_opaque_state=self.encode_opaque_state)

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        return _decode_response(
            payload,
            origin_provider=self.origin_provider,
            decode_opaque_state=self.decode_opaque_state,
        )

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        return _encode_response(response, encode_opaque_state=self.encode_opaque_state)

    def new_stream_decoder(self, *, sequence_start: int = 0) -> AnthropicStreamDecoder:
        """Create isolated state for one upstream Messages stream."""
        return AnthropicStreamDecoder(
            origin_provider=self.origin_provider,
            decode_opaque_state=self.decode_opaque_state,
            sequence_start=sequence_start,
        )

    def new_stream_encoder(
        self,
        *,
        model: str | None = None,
        response_id: str | None = None,
    ) -> AnthropicStreamEncoder:
        """Create isolated state for one downstream Messages stream."""
        return AnthropicStreamEncoder(
            model=model,
            response_id=response_id,
            encode_opaque_state=self.encode_opaque_state,
        )

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        """Convenience delegate for one stream per async context.

        Dispatchers that interleave streams in one task should use
        :meth:`new_stream_decoder` and retain the returned instance explicitly.
        """
        decoder = self._stream_decoder.get()
        if decoder is None or (decoder.terminal and payload.get("type") == "message_start"):
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
            model = event.metadata.get("model")
            encoder = self.new_stream_encoder(
                model=model if isinstance(model, str) else None,
                response_id=event.response_id,
            )
            self._stream_encoder.set(encoder)
        return encoder.encode(event)


def _inspect_message_features(value: object) -> tuple[bool, bool, bool, tuple[str, ...]]:
    """Inspect Messages content once without validating or decoding it."""
    images = False
    files = False
    reasoning = False
    carriers: list[str] = []

    def visit(candidate: object) -> None:
        nonlocal images, files, reasoning
        if isinstance(candidate, Mapping):
            block_type = candidate.get("type")
            if block_type == "image":
                images = True
            elif block_type == "document":
                files = True
            elif block_type in {"thinking", "redacted_thinking"}:
                reasoning = True
            field = (
                "signature"
                if block_type == "thinking"
                else "data"
                if block_type == "redacted_thinking"
                else None
            )
            carrier = candidate.get(field) if field is not None else None
            if isinstance(carrier, str) and carrier:
                carriers.append(carrier)
            for nested in candidate.values():
                visit(nested)
        elif isinstance(candidate, list | tuple):
            for nested in candidate:
                visit(nested)

    visit(value)
    return images, files, reasoning, tuple(carriers)


@dataclass(slots=True)
class _AnthropicDecodeBlock:
    block_type: str
    item_id: str | None = None
    name: str | None = None
    signature_parts: list[str] = field(default_factory=list)
    raw_block: dict[str, Any] = field(default_factory=dict)
    opaque_emitted: bool = False


class AnthropicStreamDecoder:
    """Stateful decoder for exactly one Anthropic SSE data-frame sequence."""

    def __init__(
        self,
        *,
        origin_provider: str | None = None,
        decode_opaque_state: OpaqueStateDecodeHook | None = None,
        sequence_start: int = 0,
    ) -> None:
        self.origin_provider = origin_provider
        self.decode_opaque_state = decode_opaque_state
        self._sequence = sequence_start
        self._started = False
        self._terminal = False
        self._response_id: str | None = None
        self._model: str | None = None
        self._blocks: dict[int, _AnthropicDecodeBlock] = {}
        self._pending_terminal: TerminalMetadata | None = None

    @property
    def terminal(self) -> bool:
        return self._terminal

    def decode(self, payload: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        frame = require_mapping(payload, protocol=_PROTOCOL, parameter="stream")
        frame_type = require_string(
            frame.get("type"), protocol=_PROTOCOL, parameter="stream.type", allow_empty=False
        )
        if self._terminal:
            decode_reject(_PROTOCOL, "stream.type", "frame arrived after terminal event")
        if frame_type == "ping":
            return ()
        if frame_type == "message_start":
            return self._decode_message_start(frame)
        if frame_type == "error":
            return self._decode_error(frame)
        if not self._started:
            decode_reject(_PROTOCOL, "stream.type", "message_start must be the first data frame")
        if frame_type == "content_block_start":
            return self._decode_content_start(frame)
        if frame_type == "content_block_delta":
            return self._decode_content_delta(frame)
        if frame_type == "content_block_stop":
            return self._decode_content_stop(frame)
        if frame_type == "message_delta":
            return self._decode_message_delta(frame)
        if frame_type == "message_stop":
            return self._decode_message_stop(frame)
        decode_reject(_PROTOCOL, "stream.type", f"unsupported event {frame_type!r}")

    def finish_eof(self) -> tuple[SemanticEvent, ...]:
        """Convert transport EOF before message_stop into one safe terminal pair."""
        if self._terminal:
            return ()
        terminal = TerminalMetadata(
            error_code="unexpected_eof",
            error_message="Upstream stream ended before message_stop",
            response_status="unknown",
        )
        self._terminal = True
        return self._events(
            (SemanticEventType.ERROR, {"terminal": terminal}),
            (SemanticEventType.TERMINAL, {"terminal": terminal}),
            common_metadata={"transport_termination": "unexpected_eof"},
        )

    def _decode_message_start(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        if self._started:
            decode_reject(_PROTOCOL, "stream.type", "duplicate message_start")
        reject_unknown_keys(
            frame,
            frozenset({"type", "message"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        message = require_mapping(
            frame.get("message"), protocol=_PROTOCOL, parameter="stream.message"
        )
        reject_unknown_keys(
            message,
            frozenset(
                {"id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "usage"}
            ),
            protocol=_PROTOCOL,
            parameter="stream.message",
        )
        if message.get("type", "message") != "message":
            decode_reject(_PROTOCOL, "stream.message.type", "must be message")
        if message.get("role", "assistant") != "assistant":
            decode_reject(_PROTOCOL, "stream.message.role", "must be assistant")
        content = message.get("content", [])
        if content not in (None, []):
            decode_reject(_PROTOCOL, "stream.message.content", "must start empty")
        self._response_id = require_string(
            message.get("id"),
            protocol=_PROTOCOL,
            parameter="stream.message.id",
            allow_empty=False,
        )
        self._model = require_string(
            message.get("model"),
            protocol=_PROTOCOL,
            parameter="stream.message.model",
            allow_empty=False,
        )
        self._started = True
        specs: list[tuple[SemanticEventType, dict[str, Any]]] = [
            (
                SemanticEventType.RESPONSE_STARTED,
                {"metadata": {"model": self._model}},
            )
        ]
        if message.get("usage") is not None:
            specs.append(
                (SemanticEventType.USAGE, {"usage": _decode_stream_usage(message["usage"])})
            )
        return self._events(*specs)

    def _decode_content_start(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        reject_unknown_keys(
            frame,
            frozenset({"type", "index", "content_block"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        index = _stream_index(frame.get("index"), "stream.index")
        if index in self._blocks:
            decode_reject(_PROTOCOL, "stream.index", "content block is already open")
        block = require_mapping(
            frame.get("content_block"),
            protocol=_PROTOCOL,
            parameter="stream.content_block",
        )
        block_type = require_string(
            block.get("type"),
            protocol=_PROTOCOL,
            parameter="stream.content_block.type",
            allow_empty=False,
        )
        item_id = f"anthropic-thinking-{index}"
        state = _AnthropicDecodeBlock(block_type=block_type, raw_block=dict(block))
        specs: list[tuple[SemanticEventType, dict[str, Any]]] = []
        if block_type == "text":
            text = _decode_text_block(block, parameter="stream.content_block").text
            specs.append(
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {"output_index": index, "metadata": {"output_item_type": "text"}},
                )
            )
            if text:
                specs.append((SemanticEventType.TEXT_DELTA, {"output_index": index, "delta": text}))
        elif block_type == "tool_use":
            call = _decode_tool_use(block, parameter="stream.content_block")
            state.item_id = call.call_id
            state.name = call.name
            specs.append(
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {"output_index": index, "item_id": call.call_id, "item": call},
                )
            )
        elif block_type in {"thinking", "redacted_thinking"}:
            state.item_id = item_id
            carrier = block.get("signature") if block_type == "thinking" else block.get("data")
            if isinstance(carrier, str) and carrier:
                state.signature_parts.append(carrier)
            text = block.get("thinking") if block_type == "thinking" else ""
            if text is not None and not isinstance(text, str):
                decode_reject(_PROTOCOL, "stream.content_block.thinking", "must be a string")
            specs.append(
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {
                        "output_index": index,
                        "item_id": item_id,
                        "metadata": {"output_item_type": "reasoning"},
                    },
                )
            )
            if text:
                specs.append(
                    (
                        SemanticEventType.REASONING_DELTA,
                        {"output_index": index, "item_id": item_id, "delta": text},
                    )
                )
            if block_type == "redacted_thinking" and state.signature_parts:
                reasoning = _decode_reasoning_block(
                    block,
                    parameter="stream.content_block",
                    model=self._required_model(),
                    item_id=item_id,
                    origin_provider=self.origin_provider,
                    decode_opaque_state=self.decode_opaque_state,
                )
                state.opaque_emitted = True
                specs.append(
                    (
                        SemanticEventType.OUTPUT_ITEM,
                        {"output_index": index, "item_id": item_id, "item": reasoning},
                    )
                )
        else:
            decode_reject(
                _PROTOCOL,
                "stream.content_block.type",
                f"unsupported block {block_type!r}",
            )
        self._blocks[index] = state
        return self._events(*specs)

    def _decode_content_delta(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        reject_unknown_keys(
            frame,
            frozenset({"type", "index", "delta"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        index = _stream_index(frame.get("index"), "stream.index")
        state = self._blocks.get(index)
        if state is None:
            decode_reject(_PROTOCOL, "stream.index", "content delta has no open block")
        delta = require_mapping(frame.get("delta"), protocol=_PROTOCOL, parameter="stream.delta")
        delta_type = require_string(
            delta.get("type"),
            protocol=_PROTOCOL,
            parameter="stream.delta.type",
            allow_empty=False,
        )
        if delta_type == "text_delta" and state.block_type == "text":
            value = require_string(
                delta.get("text"), protocol=_PROTOCOL, parameter="stream.delta.text"
            )
            return self._events(
                (SemanticEventType.TEXT_DELTA, {"output_index": index, "delta": value})
            )
        if delta_type == "thinking_delta" and state.block_type == "thinking":
            value = require_string(
                delta.get("thinking"),
                protocol=_PROTOCOL,
                parameter="stream.delta.thinking",
            )
            return self._events(
                (
                    SemanticEventType.REASONING_DELTA,
                    {"output_index": index, "item_id": state.item_id, "delta": value},
                )
            )
        if delta_type == "signature_delta" and state.block_type == "thinking":
            value = require_string(
                delta.get("signature"),
                protocol=_PROTOCOL,
                parameter="stream.delta.signature",
            )
            state.signature_parts.append(value)
            return ()
        if delta_type == "input_json_delta" and state.block_type == "tool_use":
            value = require_string(
                delta.get("partial_json"),
                protocol=_PROTOCOL,
                parameter="stream.delta.partial_json",
            )
            return self._events(
                (
                    SemanticEventType.TOOL_ARGUMENTS_DELTA,
                    {
                        "output_index": index,
                        "item_id": state.item_id,
                        "delta": value,
                        "metadata": {"name": state.name},
                    },
                )
            )
        decode_reject(
            _PROTOCOL,
            "stream.delta.type",
            f"{delta_type!r} is invalid for {state.block_type!r}",
        )

    def _decode_content_stop(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        reject_unknown_keys(
            frame,
            frozenset({"type", "index"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        index = _stream_index(frame.get("index"), "stream.index")
        state = self._blocks.pop(index, None)
        if state is None:
            decode_reject(_PROTOCOL, "stream.index", "content stop has no open block")
        specs: list[tuple[SemanticEventType, dict[str, Any]]] = []
        if (
            state.block_type in {"thinking", "redacted_thinking"}
            and state.signature_parts
            and not state.opaque_emitted
        ):
            signature = "".join(state.signature_parts)
            raw = dict(state.raw_block)
            if state.block_type == "thinking":
                raw["signature"] = signature
                raw.setdefault("thinking", "")
            else:
                raw["data"] = signature
            opaque = _decode_opaque_block(
                raw,
                signature=signature,
                parameter="stream.content_block",
                model=self._required_model(),
                item_id=state.item_id or f"anthropic-thinking-{index}",
                origin_provider=self.origin_provider,
                decode_opaque_state=self.decode_opaque_state,
            )
            specs.append(
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {
                        "output_index": index,
                        "item_id": opaque.item_id,
                        "item": ReasoningSummary("", opaque_state=opaque),
                    },
                )
            )
        specs.append(
            (
                SemanticEventType.OUTPUT_ITEM,
                {
                    "output_index": index,
                    "item_id": state.item_id,
                    "metadata": {
                        "output_item_type": state.block_type,
                        "output_item_done": True,
                    },
                },
            )
        )
        return self._events(*specs)

    def _decode_message_delta(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        reject_unknown_keys(
            frame,
            frozenset({"type", "delta", "usage"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        delta = require_mapping(
            frame.get("delta", {}), protocol=_PROTOCOL, parameter="stream.delta"
        )
        reject_unknown_keys(
            delta,
            frozenset({"stop_reason", "stop_sequence"}),
            protocol=_PROTOCOL,
            parameter="stream.delta",
        )
        stop_reason = optional_string(
            delta.get("stop_reason"),
            protocol=_PROTOCOL,
            parameter="stream.delta.stop_reason",
        )
        finish_reason = None
        if stop_reason is not None:
            finish_reason = _STOP_TO_SEMANTIC.get(stop_reason)
            if finish_reason is None:
                decode_reject(
                    _PROTOCOL,
                    "stream.delta.stop_reason",
                    f"unsupported reason {stop_reason!r}",
                )
        stop_sequence = optional_string(
            delta.get("stop_sequence"),
            protocol=_PROTOCOL,
            parameter="stream.delta.stop_sequence",
        )
        if stop_reason is not None or stop_sequence is not None:
            self._pending_terminal = TerminalMetadata(
                finish_reason=finish_reason,
                stop_sequence=stop_sequence,
                response_status=(
                    "incomplete" if finish_reason in {"length", "content_filter"} else "completed"
                ),
            )
        if frame.get("usage") is None:
            return ()
        return self._events(
            (SemanticEventType.USAGE, {"usage": _decode_stream_usage(frame["usage"])})
        )

    def _decode_message_stop(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        reject_unknown_keys(
            frame,
            frozenset({"type"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        if self._blocks:
            decode_reject(_PROTOCOL, "stream.type", "message_stop arrived with open blocks")
        terminal = self._pending_terminal or TerminalMetadata(
            finish_reason="stop",
            response_status="completed",
        )
        self._terminal = True
        return self._events((SemanticEventType.TERMINAL, {"terminal": terminal}))

    def _decode_error(self, frame: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        reject_unknown_keys(
            frame,
            frozenset({"type", "error"}),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        error = require_mapping(frame.get("error"), protocol=_PROTOCOL, parameter="stream.error")
        code = require_string(
            error.get("type", "upstream_error"),
            protocol=_PROTOCOL,
            parameter="stream.error.type",
            allow_empty=False,
        )
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

    def _required_model(self) -> str:
        if self._model is None:  # pragma: no cover - guarded by message_start
            decode_reject(_PROTOCOL, "stream.message.model", "model context is unavailable")
        return self._model

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
                    response_id=self._response_id,
                    metadata=metadata,
                    **values,
                )
            )
            self._sequence += 1
        return tuple(events)


@dataclass(slots=True)
class _AnthropicEncodeBlock:
    block_type: str
    item_id: str | None = None
    source_item_id: str | None = None
    name: str | None = None
    streamed_tool_arguments: bool = False
    tool_arguments_started: bool = False


class AnthropicStreamEncoder:
    """Stateful encoder for exactly one Anthropic SSE data-frame sequence."""

    def __init__(
        self,
        *,
        model: str | None = None,
        response_id: str | None = None,
        encode_opaque_state: OpaqueStateEncodeHook | None = None,
    ) -> None:
        self.model = model
        self.response_id = response_id
        self.encode_opaque_state = encode_opaque_state
        self._started = False
        self._terminal = False
        self._blocks: dict[int, _AnthropicEncodeBlock] = {}
        self._next_index = 0
        self._pending_error: TerminalMetadata | None = None
        self._pending_usage: Usage | None = None
        self._pending_usage_metadata: Mapping[str, Any] = {}
        self._saw_refusal = False

    @property
    def terminal(self) -> bool:
        return self._terminal

    def encode(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        if self._terminal:
            reject(_PROTOCOL, "event.type", "event arrived after terminal event")
        if event.type is SemanticEventType.RESPONSE_STARTED:
            return self._encode_started(event)
        if event.type is SemanticEventType.ERROR:
            self._pending_error = event.terminal or TerminalMetadata(
                error_code="upstream_error",
                error_message="Upstream stream failed",
                response_status="failed",
            )
            return ()
        if event.type is SemanticEventType.TERMINAL:
            return self._encode_terminal(event)
        frames = self._ensure_started(event)
        if event.type is SemanticEventType.TEXT_DELTA:
            index = self._event_index(event)
            frames.extend(self._ensure_block(index, "text"))
            frames.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": event.delta or ""},
                }
            )
        elif event.type is SemanticEventType.REASONING_DELTA:
            index = self._event_index(event)
            frames.extend(self._ensure_block(index, "thinking", item_id=event.item_id))
            frames.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": event.delta or ""},
                }
            )
        elif event.type is SemanticEventType.TOOL_ARGUMENTS_DELTA:
            index = self._event_index(event)
            existing = self._blocks.get(index)
            source_item_id = event.item_id
            call_id = event.metadata.get("call_id")
            name = event.metadata.get("name")
            if existing is not None:
                if existing.block_type != "tool_use":
                    reject(_PROTOCOL, "event.output_index", "output block changed type")
                if (
                    source_item_id is not None
                    and existing.source_item_id is not None
                    and source_item_id != existing.source_item_id
                ):
                    reject(_PROTOCOL, "event.item_id", "tool delta changed item ID")
                if call_id is not None and call_id != existing.item_id:
                    reject(_PROTOCOL, "event.metadata.call_id", "tool delta changed call ID")
                if name is not None and name != existing.name:
                    reject(_PROTOCOL, "event.metadata.name", "tool delta changed name")
                if existing.source_item_id is None:
                    existing.source_item_id = source_item_id
                call_id = existing.item_id
                name = existing.name
            elif call_id is None:
                # Protocols such as Anthropic expose one identifier for both the
                # streamed item and tool call. Responses exposes a distinct
                # ``fc_*`` item ID and carries the callable ``call_*`` ID in
                # metadata, which must win whenever it is available.
                call_id = source_item_id
            if not isinstance(call_id, str) or not call_id:
                reject(_PROTOCOL, "event.metadata.call_id", "tool delta requires a call ID")
            if not isinstance(name, str) or not name:
                reject(_PROTOCOL, "event.metadata.name", "tool delta requires a name")
            frames.extend(
                self._ensure_block(
                    index,
                    "tool_use",
                    item_id=call_id,
                    source_item_id=source_item_id,
                    name=name,
                )
            )
            block = self._blocks[index]
            block.streamed_tool_arguments = True
            delta = event.delta or ""
            if block.tool_arguments_started or delta.strip():
                block.tool_arguments_started = True
                frames.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": delta},
                    }
                )
        elif event.type is SemanticEventType.OUTPUT_ITEM:
            frames.extend(self._encode_output_item(event))
        elif event.type is SemanticEventType.USAGE:
            if event.usage is None:
                reject(_PROTOCOL, "event.usage", "usage event requires Usage")
            self._pending_usage = event.usage
            self._pending_usage_metadata = event.metadata
        else:
            reject(_PROTOCOL, "event.type", f"unsupported event {event.type.value!r}")
        return tuple(frames)

    def finish_eof(self) -> tuple[Mapping[str, Any], ...]:
        """Terminate an unfinished downstream stream with one Anthropic error frame."""
        if self._terminal:
            return ()
        self._terminal = True
        return (
            {
                "type": "error",
                "error": {
                    "type": "unexpected_eof",
                    "message": "Semantic event stream ended before terminal event",
                },
            },
        )

    def _encode_started(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        if self._started:
            reject(_PROTOCOL, "event.type", "duplicate response_started")
        metadata_model = event.metadata.get("model")
        if isinstance(metadata_model, str):
            self.model = metadata_model
        if event.response_id is not None:
            self.response_id = event.response_id
        return tuple(self._ensure_started(event))

    def _ensure_started(self, event: SemanticEvent) -> list[Mapping[str, Any]]:
        if self._started:
            return []
        if event.response_id is not None:
            self.response_id = event.response_id
        metadata_model = event.metadata.get("model")
        if isinstance(metadata_model, str):
            self.model = metadata_model
        if not self.model:
            reject(_PROTOCOL, "event.metadata.model", "Messages stream requires a model")
        if not self.response_id:
            reject(_PROTOCOL, "event.response_id", "Messages stream requires a response ID")
        self._started = True
        return [
            {
                "type": "message_start",
                "message": {
                    "id": self.response_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
        ]

    def _ensure_block(
        self,
        index: int,
        block_type: str,
        *,
        item_id: str | None = None,
        source_item_id: str | None = None,
        name: str | None = None,
    ) -> list[Mapping[str, Any]]:
        existing = self._blocks.get(index)
        if existing is not None:
            if existing.block_type != block_type:
                reject(_PROTOCOL, "event.output_index", "output block changed type")
            return []
        if block_type == "text":
            block: dict[str, Any] = {"type": "text", "text": ""}
        elif block_type == "thinking":
            block = {"type": "thinking", "thinking": ""}
        elif block_type == "tool_use":
            if not item_id or not name:
                reject(_PROTOCOL, "event.item", "tool block requires ID and name")
            block = {"type": "tool_use", "id": item_id, "name": name, "input": {}}
        else:  # pragma: no cover - private callers use the closed set above
            reject(_PROTOCOL, "event.item", f"unsupported block type {block_type!r}")
        self._blocks[index] = _AnthropicEncodeBlock(
            block_type,
            item_id=item_id,
            source_item_id=source_item_id,
            name=name,
        )
        return [{"type": "content_block_start", "index": index, "content_block": block}]

    def _encode_output_item(self, event: SemanticEvent) -> list[Mapping[str, Any]]:
        item = event.item
        index = self._event_index(event)
        frames: list[Mapping[str, Any]] = []
        if item is None:
            if event.metadata.get("output_item_done") is True:
                frames.extend(self._close_block(index))
            return frames
        if isinstance(item, TextContent | RefusalContent):
            if isinstance(item, RefusalContent):
                self._saw_refusal = True
                text = item.refusal
            else:
                text = item.text
            frames.extend(self._ensure_block(index, "text"))
            if text:
                frames.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    }
                )
        elif isinstance(item, ToolCall):
            block = _encode_tool_call(item, parameter="event.item")
            source_item_id = event.item_id or item.item_id
            if event.item_id is not None and item.item_id is not None:
                if event.item_id != item.item_id:
                    reject(_PROTOCOL, "event.item_id", "tool output item ID changed")
            existing = self._blocks.get(index)
            if existing is None:
                self._blocks[index] = _AnthropicEncodeBlock(
                    "tool_use",
                    item_id=item.call_id,
                    source_item_id=source_item_id,
                    name=item.name,
                )
                frames.append(
                    {"type": "content_block_start", "index": index, "content_block": block}
                )
            else:
                if existing.block_type != "tool_use":
                    reject(_PROTOCOL, "event.output_index", "output block changed type")
                if existing.item_id != item.call_id:
                    reject(_PROTOCOL, "event.item.call_id", "tool output changed call ID")
                if existing.name != item.name:
                    reject(_PROTOCOL, "event.item.name", "tool output changed name")
                if (
                    source_item_id is not None
                    and existing.source_item_id is not None
                    and source_item_id != existing.source_item_id
                ):
                    reject(_PROTOCOL, "event.item_id", "tool output item ID changed")
                if existing.source_item_id is None:
                    existing.source_item_id = source_item_id
        elif isinstance(item, ReasoningSummary):
            raw = _encode_reasoning_block(
                item,
                parameter="event.item",
                model=self._required_model(),
                encode_opaque_state=self.encode_opaque_state,
            )
            existing = self._blocks.get(index)
            if existing is None:
                item_id = (
                    item.opaque_state.item_id if item.opaque_state is not None else event.item_id
                )
                if raw["type"] == "thinking":
                    # Claude Code persists continuation state from
                    # signature_delta. A complete Responses reasoning item can
                    # first appear only at output_item.done, so project that
                    # snapshot through the normal incremental lifecycle.
                    frames.extend(self._ensure_block(index, "thinking", item_id=item_id))
                    if item.text:
                        frames.append(
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": item.text,
                                },
                            }
                        )
                    signature = raw.get("signature")
                    if isinstance(signature, str):
                        frames.append(
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "signature_delta",
                                    "signature": signature,
                                },
                            }
                        )
                else:
                    self._blocks[index] = _AnthropicEncodeBlock(
                        raw["type"],
                        item_id=item_id,
                    )
                    frames.append(
                        {"type": "content_block_start", "index": index, "content_block": raw}
                    )
            elif existing.block_type != "thinking":
                reject(_PROTOCOL, "event.item", "reasoning state conflicts with open block")
            else:
                if raw["type"] == "thinking":
                    signature = raw.get("signature")
                elif raw["type"] == "redacted_thinking":
                    # The final opaque state event intentionally carries no
                    # duplicate summary text. If visible thinking deltas have
                    # already opened this block, its capsule belongs in the
                    # thinking signature rather than a second redacted block.
                    signature = raw.get("data")
                else:
                    reject(_PROTOCOL, "event.item", "reasoning state conflicts with open block")
                if isinstance(signature, str):
                    frames.append(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "signature_delta", "signature": signature},
                        }
                    )
        else:
            reject(_PROTOCOL, "event.item", f"unsupported output {type(item).__name__}")
        if event.metadata.get("output_item_done") is True:
            frames.extend(self._close_block(index))
        return frames

    def _encode_terminal(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        frames = self._ensure_started(event)
        for index in sorted(self._blocks):
            frames.extend(self._close_block(index))
        terminal = (
            self._pending_error
            or event.terminal
            or TerminalMetadata(
                finish_reason="content_filter" if self._saw_refusal else "stop",
                response_status="completed",
            )
        )
        status_error = _terminal_status_error(
            terminal,
            parameter="event.terminal.response_status",
        )
        if terminal.error_code is not None or terminal.error_message is not None or status_error:
            synthesized_type, synthesized_message = status_error or (
                "upstream_error",
                "Upstream stream failed",
            )
            frames.append(
                {
                    "type": "error",
                    "error": {
                        "type": terminal.error_code or synthesized_type,
                        "message": terminal.error_message or synthesized_message,
                    },
                }
            )
        else:
            if self._saw_refusal:
                stop_reason = "refusal"
                if terminal.finish_reason is not None:
                    projected_reason = _STOP_FROM_SEMANTIC.get(terminal.finish_reason)
                    if projected_reason is None:
                        reject(
                            _PROTOCOL,
                            "event.terminal.finish_reason",
                            f"unsupported reason {terminal.finish_reason!r}",
                        )
                    if projected_reason != "refusal":
                        reject(
                            _PROTOCOL,
                            "event.terminal.finish_reason",
                            "conflicts with refusal output",
                        )
                if terminal.stop_sequence is not None:
                    reject(
                        _PROTOCOL,
                        "event.terminal.stop_sequence",
                        "conflicts with refusal output",
                    )
            else:
                stop_reason = _STOP_FROM_SEMANTIC.get(terminal.finish_reason or "stop")
                if stop_reason is None:
                    reject(
                        _PROTOCOL,
                        "event.terminal.finish_reason",
                        f"unsupported reason {terminal.finish_reason!r}",
                    )
                if terminal.stop_sequence is not None:
                    stop_reason = "stop_sequence"
            usage = (
                _encode_stream_usage(
                    self._pending_usage,
                    self._pending_usage_metadata,
                )
                if self._pending_usage is not None
                else {"output_tokens": 0}
            )
            frames.extend(
                [
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": terminal.stop_sequence,
                        },
                        "usage": usage,
                    },
                    {"type": "message_stop"},
                ]
            )
        self._terminal = True
        return tuple(frames)

    def _close_block(self, index: int) -> list[Mapping[str, Any]]:
        block = self._blocks.pop(index, None)
        if block is None:
            return []
        frames: list[Mapping[str, Any]] = []
        if (
            block.block_type == "tool_use"
            and block.streamed_tool_arguments
            and not block.tool_arguments_started
        ):
            frames.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                }
            )
        frames.append({"type": "content_block_stop", "index": index})
        return frames

    def _event_index(self, event: SemanticEvent) -> int:
        if event.output_index is not None:
            if event.output_index < 0:
                reject(_PROTOCOL, "event.output_index", "cannot be negative")
            self._next_index = max(self._next_index, event.output_index + 1)
            return event.output_index
        if event.item_id is not None:
            for index, block in self._blocks.items():
                if event.item_id in {block.item_id, block.source_item_id}:
                    return index
        index = self._next_index
        self._next_index += 1
        return index

    def _required_model(self) -> str:
        if not self.model:
            reject(_PROTOCOL, "event.metadata.model", "Messages stream requires a model")
        return self.model


def _stream_index(value: object, parameter: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        decode_reject(_PROTOCOL, parameter, "must be a non-negative integer")
    return value


def _decode_stream_usage(value: object) -> Usage:
    usage = require_mapping(value, protocol=_PROTOCOL, parameter="stream.usage")
    reject_unknown_keys(
        usage,
        frozenset(
            {
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "service_tier",
            }
        ),
        protocol=_PROTOCOL,
        parameter="stream.usage",
    )
    input_tokens = optional_int(
        usage.get("input_tokens"), protocol=_PROTOCOL, parameter="stream.usage.input_tokens"
    )
    output_tokens = optional_int(
        usage.get("output_tokens"),
        protocol=_PROTOCOL,
        parameter="stream.usage.output_tokens",
    )
    for name, count in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        if count is not None and count < 0:
            decode_reject(_PROTOCOL, f"stream.usage.{name}", "cannot be negative")
    return Usage(
        mode=UsageMode.SNAPSHOT,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        ),
        cached_input_tokens=optional_int(
            usage.get("cache_read_input_tokens"),
            protocol=_PROTOCOL,
            parameter="stream.usage.cache_read_input_tokens",
        ),
    )


def _encode_stream_usage(usage: Usage, metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if usage.input_tokens is not None:
        payload["input_tokens"] = usage.input_tokens
    if usage.output_tokens is not None:
        payload["output_tokens"] = usage.output_tokens
    if usage.cached_input_tokens is not None:
        payload["cache_read_input_tokens"] = usage.cached_input_tokens
    for key in ("cache_creation_input_tokens", "service_tier"):
        if key in metadata:
            payload[key] = metadata[key]
    return payload


def _decode_request(
    payload: Mapping[str, Any],
    *,
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
) -> SemanticRequest:
    body = require_mapping(payload, protocol=_PROTOCOL, parameter="request")
    reject_unknown_keys(body, _REQUEST_FIELDS, protocol=_PROTOCOL, parameter="")
    context_management = _decode_context_management(body.get("context_management"))
    model = require_string(
        body.get("model"), protocol=_PROTOCOL, parameter="model", allow_empty=False
    )
    raw_messages = require_list(body.get("messages"), protocol=_PROTOCOL, parameter="messages")
    messages: list[SemanticMessage] = []
    system = body.get("system")
    if system is not None:
        messages.append(_decode_system(system))
    for index, raw_message in enumerate(raw_messages):
        messages.extend(
            _decode_message(
                raw_message,
                parameter=f"messages[{index}]",
                model=model,
                message_index=index,
                origin_provider=origin_provider,
                decode_opaque_state=decode_opaque_state,
            )
        )

    max_tokens = optional_int(body.get("max_tokens"), protocol=_PROTOCOL, parameter="max_tokens")
    if max_tokens is None:
        decode_reject(_PROTOCOL, "max_tokens", "is required")
    if max_tokens < 0:
        decode_reject(_PROTOCOL, "max_tokens", "cannot be negative")

    output_config = body.get("output_config")
    effort = None
    structured_output = None
    if output_config is not None:
        config = require_mapping(output_config, protocol=_PROTOCOL, parameter="output_config")
        reject_unknown_keys(
            config,
            frozenset({"effort", "format"}),
            protocol=_PROTOCOL,
            parameter="output_config",
        )
        effort = optional_string(
            config.get("effort"),
            protocol=_PROTOCOL,
            parameter="output_config.effort",
        )
        if config.get("format") is not None:
            structured_output = require_mapping(
                config.get("format"),
                protocol=_PROTOCOL,
                parameter="output_config.format",
            )

    reasoning = _decode_reasoning(body.get("thinking"), effort=effort)
    tool_choice, parallel_tool_calls = _decode_tool_choice(body.get("tool_choice"))
    metadata = body.get("metadata") or {}
    metadata = require_mapping(metadata, protocol=_PROTOCOL, parameter="metadata")
    return SemanticRequest(
        model=model,
        input=tuple(messages),
        tools=_decode_tools(body.get("tools")),
        stream=optional_bool(body.get("stream", False), protocol=_PROTOCOL, parameter="stream")
        or False,
        max_output_tokens=max_tokens,
        temperature=optional_number(
            body.get("temperature"), protocol=_PROTOCOL, parameter="temperature"
        ),
        top_p=optional_number(body.get("top_p"), protocol=_PROTOCOL, parameter="top_p"),
        top_k=optional_int(body.get("top_k"), protocol=_PROTOCOL, parameter="top_k"),
        stop_sequences=_decode_string_list(body.get("stop_sequences"), "stop_sequences"),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        reasoning=reasoning,
        structured_output=structured_output,
        service_tier=optional_string(
            body.get("service_tier"), protocol=_PROTOCOL, parameter="service_tier"
        ),
        metadata=metadata,
        provider_extensions=(
            {"context_management": context_management} if context_management is not None else {}
        ),
        explicit_fields=frozenset(body),
    )


def _decode_context_management(value: object) -> dict[str, Any] | None:
    """Normalize exact no-ops and preserve active edits for target rejection.

    Active context editing is provider-side stateful behavior and has no exact
    Chat or Responses equivalent.  Claude Code currently sends
    ``clear_thinking_20251015`` with ``keep: \"all\"`` even when routing to a
    Responses-only model.  That edit explicitly preserves every thinking turn,
    so consuming it without emitting an outbound field is lossless.  Other
    structurally valid edits are retained as a provider extension so target
    runtimes reject them as unrepresentable without overriding an earlier
    retryable native-transport failure.
    """
    if value is None:
        return None
    config = require_mapping(
        value,
        protocol=_PROTOCOL,
        parameter="context_management",
    )
    if "edits" not in config:
        return None if not config else thaw_json(config)
    edits = require_list(
        config["edits"],
        protocol=_PROTOCOL,
        parameter="context_management.edits",
    )
    exact_noop = set(config) == {"edits"}
    for index, raw_edit in enumerate(edits):
        parameter = f"context_management.edits[{index}]"
        edit = require_mapping(raw_edit, protocol=_PROTOCOL, parameter=parameter)
        edit_type = require_string(
            edit.get("type"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.type",
            allow_empty=False,
        )
        _validate_context_management_edit(edit, edit_type=edit_type, parameter=parameter)
        exact_noop = exact_noop and (
            set(edit) == {"type", "keep"}
            and edit_type == "clear_thinking_20251015"
            and _is_context_management_keep_all(edit.get("keep"))
        )
    if exact_noop:
        return None
    return thaw_json(config)


def _validate_context_management_edit(
    edit: Mapping[str, Any],
    *,
    edit_type: str,
    parameter: str,
) -> None:
    """Validate fields whose Anthropic wire schema is known.

    Unknown edit types and extra members remain opaque so a native Messages
    binding can forward future protocol additions unchanged. Known members of
    the two published edit variants must still be structurally valid; otherwise
    the request is malformed at ingress rather than merely unrepresentable by a
    later target protocol.
    """
    if edit_type == "clear_thinking_20251015":
        if "keep" in edit:
            _validate_clear_thinking_keep(edit["keep"], parameter=f"{parameter}.keep")
        return
    if edit_type != "clear_tool_uses_20250919":
        return

    if "trigger" in edit:
        _validate_context_management_count_selector(
            edit["trigger"],
            parameter=f"{parameter}.trigger",
            allowed_types=frozenset({"input_tokens", "tool_uses"}),
        )
    if "keep" in edit:
        _validate_context_management_count_selector(
            edit["keep"],
            parameter=f"{parameter}.keep",
            allowed_types=frozenset({"tool_uses"}),
        )
    if "clear_at_least" in edit and edit["clear_at_least"] is not None:
        _validate_context_management_count_selector(
            edit["clear_at_least"],
            parameter=f"{parameter}.clear_at_least",
            allowed_types=frozenset({"input_tokens"}),
        )
    if "exclude_tools" in edit and edit["exclude_tools"] is not None:
        tools = require_list(
            edit["exclude_tools"],
            protocol=_PROTOCOL,
            parameter=f"{parameter}.exclude_tools",
        )
        for index, tool_name in enumerate(tools):
            require_string(
                tool_name,
                protocol=_PROTOCOL,
                parameter=f"{parameter}.exclude_tools[{index}]",
                allow_empty=False,
            )
    if "clear_tool_inputs" in edit:
        _validate_clear_tool_inputs(
            edit["clear_tool_inputs"],
            parameter=f"{parameter}.clear_tool_inputs",
        )


def _is_context_management_keep_all(value: object) -> bool:
    return value == "all" or (
        isinstance(value, Mapping) and set(value) == {"type"} and value.get("type") == "all"
    )


def _validate_clear_thinking_keep(value: object, *, parameter: str) -> None:
    if isinstance(value, str):
        if value != "all":
            decode_reject(_PROTOCOL, parameter, 'must be "all" or an object')
        return
    _validate_context_management_count_selector(
        value,
        parameter=parameter,
        allowed_types=frozenset({"all", "thinking_turns"}),
    )


def _validate_context_management_count_selector(
    value: object,
    *,
    parameter: str,
    allowed_types: frozenset[str],
) -> None:
    selector = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    selector_type = require_string(
        selector.get("type"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.type",
        allow_empty=False,
    )
    if selector_type not in allowed_types:
        expected = ", ".join(sorted(allowed_types))
        decode_reject(_PROTOCOL, f"{parameter}.type", f"must be one of: {expected}")
    if selector_type == "all":
        return
    count = selector.get("value")
    if not isinstance(count, int) or isinstance(count, bool):
        decode_reject(_PROTOCOL, f"{parameter}.value", "must be an integer")


def _validate_clear_tool_inputs(value: object, *, parameter: str) -> None:
    if value is None or isinstance(value, bool):
        return
    tools = require_list(value, protocol=_PROTOCOL, parameter=parameter)
    for index, tool_name in enumerate(tools):
        require_string(
            tool_name,
            protocol=_PROTOCOL,
            parameter=f"{parameter}[{index}]",
            allow_empty=False,
        )


def _decode_system(value: object) -> SemanticMessage:
    if isinstance(value, str):
        return SemanticMessage(role=MessageRole.SYSTEM, content=(TextContent(value),))
    blocks = require_list(value, protocol=_PROTOCOL, parameter="system")
    content = tuple(
        _decode_text_block(raw, parameter=f"system[{index}]") for index, raw in enumerate(blocks)
    )
    return SemanticMessage(role=MessageRole.SYSTEM, content=content)


def _decode_message(
    value: object,
    *,
    parameter: str,
    model: str,
    message_index: int,
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
) -> tuple[SemanticMessage, ...]:
    message = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        message,
        frozenset({"role", "content"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    role_value = require_string(
        message.get("role"), protocol=_PROTOCOL, parameter=f"{parameter}.role"
    )
    if role_value not in {"user", "assistant"}:
        decode_reject(_PROTOCOL, f"{parameter}.role", "must be user or assistant")
    role = MessageRole.USER if role_value == "user" else MessageRole.ASSISTANT
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        return (SemanticMessage(role=role, content=(TextContent(raw_content),)),)
    blocks = require_list(raw_content, protocol=_PROTOCOL, parameter=f"{parameter}.content")

    result: list[SemanticMessage] = []
    pending: list[Any] = []

    def flush() -> None:
        if pending:
            result.append(SemanticMessage(role=role, content=tuple(pending)))
            pending.clear()

    for block_index, raw_block in enumerate(blocks):
        path = f"{parameter}.content[{block_index}]"
        block = require_mapping(raw_block, protocol=_PROTOCOL, parameter=path)
        block_type = require_string(block.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type")
        if block_type == "text":
            pending.append(_decode_text_block(block, parameter=path))
        elif block_type == "image":
            if role is not MessageRole.USER:
                decode_reject(_PROTOCOL, path, "image blocks require user role")
            pending.append(_decode_image(block, parameter=path))
        elif block_type == "document":
            if role is not MessageRole.USER:
                decode_reject(_PROTOCOL, path, "document blocks require user role")
            pending.append(_decode_document(block, parameter=path))
        elif block_type == "tool_use":
            if role is not MessageRole.ASSISTANT:
                decode_reject(_PROTOCOL, path, "tool_use blocks require assistant role")
            pending.append(_decode_tool_use(block, parameter=path))
        elif block_type == "tool_result":
            if role is not MessageRole.USER:
                decode_reject(_PROTOCOL, path, "tool_result blocks require user role")
            flush()
            result.append(
                SemanticMessage(
                    role=MessageRole.TOOL,
                    content=(_decode_tool_result(block, parameter=path),),
                )
            )
        elif block_type in {"thinking", "redacted_thinking"}:
            if role is not MessageRole.ASSISTANT:
                decode_reject(_PROTOCOL, path, "thinking blocks require assistant role")
            pending.append(
                _decode_reasoning_block(
                    block,
                    parameter=path,
                    model=model,
                    item_id=f"anthropic-thinking-{message_index}-{block_index}",
                    origin_provider=origin_provider,
                    decode_opaque_state=decode_opaque_state,
                )
            )
        else:
            decode_reject(_PROTOCOL, f"{path}.type", f"unsupported block {block_type!r}")
    flush()
    if not result:
        result.append(SemanticMessage(role=role, content=()))
    return tuple(result)


def _decode_text_block(value: object, *, parameter: str) -> TextContent:
    block = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        block,
        frozenset({"type", "text", "cache_control"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    _consume_cache_control(block, parameter=parameter)
    block_type = require_string(
        block.get("type", "text"), protocol=_PROTOCOL, parameter=f"{parameter}.type"
    )
    if block_type != "text":
        decode_reject(_PROTOCOL, f"{parameter}.type", "must be text")
    return TextContent(
        require_string(block.get("text"), protocol=_PROTOCOL, parameter=f"{parameter}.text")
    )


def _decode_image(value: object, *, parameter: str) -> ImageContent:
    block = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        block,
        frozenset({"type", "source", "cache_control"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    _consume_cache_control(block, parameter=parameter)
    source = require_mapping(
        block.get("source"), protocol=_PROTOCOL, parameter=f"{parameter}.source"
    )
    reject_unknown_keys(
        source,
        frozenset({"type", "media_type", "data"}),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.source",
    )
    source_type = require_string(
        source.get("type"), protocol=_PROTOCOL, parameter=f"{parameter}.source.type"
    )
    if source_type != "base64":
        decode_reject(_PROTOCOL, f"{parameter}.source.type", "only base64 is supported")
    return ImageContent(
        source=require_string(
            source.get("data"), protocol=_PROTOCOL, parameter=f"{parameter}.source.data"
        ),
        media_type=require_string(
            source.get("media_type"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.source.media_type",
        ),
        source_kind="base64",
    )


def _decode_document(value: object, *, parameter: str) -> FileContent:
    block = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        block,
        frozenset({"type", "source", "title", "context", "citations", "cache_control"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    _consume_cache_control(block, parameter=parameter)
    for unsupported in ("context", "citations"):
        if block.get(unsupported) is not None:
            decode_reject(_PROTOCOL, f"{parameter}.{unsupported}", "field is not modeled")
    source = require_mapping(
        block.get("source"), protocol=_PROTOCOL, parameter=f"{parameter}.source"
    )
    reject_unknown_keys(
        source,
        frozenset({"type", "media_type", "data", "url", "content"}),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.source",
    )
    source_type = require_string(
        source.get("type"), protocol=_PROTOCOL, parameter=f"{parameter}.source.type"
    )
    if source_type == "base64":
        raw_source = require_string(
            source.get("data"), protocol=_PROTOCOL, parameter=f"{parameter}.source.data"
        )
    elif source_type == "url":
        raw_source = require_string(
            source.get("url"), protocol=_PROTOCOL, parameter=f"{parameter}.source.url"
        )
    elif source_type in {"text", "content"}:
        raw_source = source.get("data", source.get("content"))
        raw_source = require_string(
            raw_source,
            protocol=_PROTOCOL,
            parameter=f"{parameter}.source.{source_type}",
        )
    else:
        decode_reject(
            _PROTOCOL,
            f"{parameter}.source.type",
            f"unsupported document source {source_type!r}",
        )
    return FileContent(
        source=raw_source,
        filename=optional_string(
            block.get("title"), protocol=_PROTOCOL, parameter=f"{parameter}.title"
        ),
        media_type=optional_string(
            source.get("media_type"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.source.media_type",
        ),
        source_kind=source_type,
    )


def _decode_tool_use(value: object, *, parameter: str) -> ToolCall:
    block = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        block,
        frozenset({"type", "id", "name", "input", "cache_control"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    _consume_cache_control(block, parameter=parameter)
    call_id = require_string(
        block.get("id"), protocol=_PROTOCOL, parameter=f"{parameter}.id", allow_empty=False
    )
    arguments = require_mapping(
        block.get("input"), protocol=_PROTOCOL, parameter=f"{parameter}.input"
    )
    return ToolCall(
        call_id=call_id,
        name=require_string(
            block.get("name"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.name",
            allow_empty=False,
        ),
        arguments=arguments,
    )


def _decode_tool_result(value: object, *, parameter: str) -> ToolResult:
    block = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    reject_unknown_keys(
        block,
        frozenset({"type", "tool_use_id", "content", "is_error", "cache_control"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    _consume_cache_control(block, parameter=parameter)
    raw_content = block.get("content", "")
    if isinstance(raw_content, str):
        content = (TextContent(raw_content),)
    else:
        blocks = require_list(raw_content, protocol=_PROTOCOL, parameter=f"{parameter}.content")
        decoded = []
        for index, raw_part in enumerate(blocks):
            path = f"{parameter}.content[{index}]"
            part = require_mapping(raw_part, protocol=_PROTOCOL, parameter=path)
            part_type = require_string(
                part.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type"
            )
            if part_type == "text":
                decoded.append(_decode_text_block(part, parameter=path))
            elif part_type == "image":
                decoded.append(_decode_image(part, parameter=path))
            elif part_type == "document":
                decoded.append(_decode_document(part, parameter=path))
            else:
                decode_reject(_PROTOCOL, f"{path}.type", f"unsupported block {part_type!r}")
        content = tuple(decoded)
    return ToolResult(
        call_id=require_string(
            block.get("tool_use_id"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.tool_use_id",
            allow_empty=False,
        ),
        content=content,
        is_error=optional_bool(
            block.get("is_error", False),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.is_error",
        )
        or False,
    )


def _decode_reasoning_block(
    value: object,
    *,
    parameter: str,
    model: str,
    item_id: str,
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
) -> ReasoningSummary:
    block = require_mapping(value, protocol=_PROTOCOL, parameter=parameter)
    block_type = require_string(
        block.get("type"), protocol=_PROTOCOL, parameter=f"{parameter}.type"
    )
    if block_type == "thinking":
        text = require_string(
            block.get("thinking"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.thinking",
        )
        signature = optional_string(
            block.get("signature"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.signature",
        )
        if signature == "":
            decode_reject(
                _PROTOCOL,
                f"{parameter}.signature",
                "Invalid signature in thinking block",
            )
    elif block_type == "redacted_thinking":
        text = ""
        signature = require_string(
            block.get("data"), protocol=_PROTOCOL, parameter=f"{parameter}.data"
        )
    else:  # pragma: no cover - guarded by the caller
        decode_reject(_PROTOCOL, f"{parameter}.type", "unsupported reasoning block")
    opaque = None
    if signature is not None:
        opaque = _decode_opaque_block(
            block,
            signature=signature,
            parameter=parameter,
            model=model,
            item_id=item_id,
            origin_provider=origin_provider,
            decode_opaque_state=decode_opaque_state,
        )
    return ReasoningSummary(text=text, opaque_state=opaque)


def _decode_opaque_block(
    block: Mapping[str, Any],
    *,
    signature: str,
    parameter: str,
    model: str,
    item_id: str,
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
) -> OpaqueState:
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
        blob=block,
    )


def _decode_tools(value: object) -> tuple[ToolDefinition, ...]:
    if value is None:
        return ()
    raw_tools = require_list(value, protocol=_PROTOCOL, parameter="tools")
    tools = []
    for index, raw_tool in enumerate(raw_tools):
        path = f"tools[{index}]"
        tool = require_mapping(raw_tool, protocol=_PROTOCOL, parameter=path)
        reject_unknown_keys(
            tool,
            frozenset({"name", "description", "input_schema", "strict", "cache_control"}),
            protocol=_PROTOCOL,
            parameter=path,
        )
        _consume_cache_control(tool, parameter=path)
        schema = tool.get("input_schema") or {"type": "object", "properties": {}}
        schema = require_mapping(schema, protocol=_PROTOCOL, parameter=f"{path}.input_schema")
        tools.append(
            ToolDefinition(
                name=require_string(
                    tool.get("name"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.name",
                    allow_empty=False,
                ),
                description=optional_string(
                    tool.get("description"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.description",
                ),
                input_schema=schema,
                strict=optional_bool(
                    tool.get("strict"), protocol=_PROTOCOL, parameter=f"{path}.strict"
                ),
            )
        )
    return tuple(tools)


def _consume_cache_control(value: Mapping[str, Any], *, parameter: str) -> None:
    """Consume Anthropic's standard ephemeral cache hint on semantic paths.

    The hint changes provider cache placement, not model-visible request
    semantics. Chat and Responses providers own their own automatic prompt
    caching, so the exact standard hint is an explicit advisory no-op when the
    request crosses protocols. Native Messages identity traffic still forwards
    the original object unchanged.
    """
    if "cache_control" not in value:
        return
    path = f"{parameter}.cache_control"
    raw_cache_control = value["cache_control"]
    if raw_cache_control is None:
        return
    cache_control = require_mapping(raw_cache_control, protocol=_PROTOCOL, parameter=path)
    cache_type = require_string(
        cache_control.get("type"),
        protocol=_PROTOCOL,
        parameter=f"{path}.type",
        allow_empty=False,
    )
    if "ttl" in cache_control:
        ttl = require_string(
            cache_control["ttl"],
            protocol=_PROTOCOL,
            parameter=f"{path}.ttl",
            allow_empty=False,
        )
        if ttl not in {"5m", "1h"}:
            decode_reject(_PROTOCOL, f"{path}.ttl", 'must be "5m" or "1h"')
    unknown = sorted(set(cache_control) - {"type"})
    if unknown:
        reject(
            _PROTOCOL,
            f"{path}.{unknown[0]}",
            "cache option is not portable across protocols",
        )
    if cache_type != "ephemeral":
        reject(
            _PROTOCOL,
            f"{path}.type",
            "only the standard ephemeral cache hint is portable across protocols",
        )


def _decode_tool_choice(value: object) -> tuple[ToolChoice | None, bool | None]:
    if value is None:
        return None, None
    choice = require_mapping(value, protocol=_PROTOCOL, parameter="tool_choice")
    reject_unknown_keys(
        choice,
        frozenset({"type", "name", "disable_parallel_tool_use"}),
        protocol=_PROTOCOL,
        parameter="tool_choice",
    )
    choice_type = require_string(
        choice.get("type"), protocol=_PROTOCOL, parameter="tool_choice.type"
    )
    name = optional_string(choice.get("name"), protocol=_PROTOCOL, parameter="tool_choice.name")
    if choice_type == "auto":
        semantic = ToolChoice("auto")
    elif choice_type == "any":
        semantic = ToolChoice("required")
    elif choice_type == "none":
        semantic = ToolChoice("none")
    elif choice_type == "tool":
        if not name:
            decode_reject(_PROTOCOL, "tool_choice.name", "is required for tool choice")
        semantic = ToolChoice("function", name=name)
    else:
        decode_reject(_PROTOCOL, "tool_choice.type", f"unsupported type {choice_type!r}")
    if choice_type != "tool" and name is not None:
        decode_reject(_PROTOCOL, "tool_choice.name", "is only valid for type tool")
    disable_parallel = optional_bool(
        choice.get("disable_parallel_tool_use"),
        protocol=_PROTOCOL,
        parameter="tool_choice.disable_parallel_tool_use",
    )
    return semantic, None if disable_parallel is None else not disable_parallel


def _decode_reasoning(value: object, *, effort: str | None) -> ReasoningConfig | None:
    if value is None and effort is None:
        return None
    enabled = None
    budget = None
    if value is not None:
        thinking = require_mapping(value, protocol=_PROTOCOL, parameter="thinking")
        reject_unknown_keys(
            thinking,
            frozenset({"type", "budget_tokens", "display"}),
            protocol=_PROTOCOL,
            parameter="thinking",
        )
        display = optional_string(
            thinking.get("display"),
            protocol=_PROTOCOL,
            parameter="thinking.display",
        )
        if display not in {None, "omitted"}:
            decode_reject(_PROTOCOL, "thinking.display", "only omitted display is supported")
        thinking_type = require_string(
            thinking.get("type", "enabled"),
            protocol=_PROTOCOL,
            parameter="thinking.type",
        )
        if thinking_type not in {"enabled", "adaptive", "disabled"}:
            decode_reject(_PROTOCOL, "thinking.type", f"unsupported type {thinking_type!r}")
        enabled = thinking_type != "disabled"
        budget = optional_int(
            thinking.get("budget_tokens"),
            protocol=_PROTOCOL,
            parameter="thinking.budget_tokens",
        )
        if budget is not None and budget < 0:
            decode_reject(_PROTOCOL, "thinking.budget_tokens", "cannot be negative")
        if thinking_type == "adaptive" and budget is not None:
            decode_reject(_PROTOCOL, "thinking.budget_tokens", "adaptive thinking has no budget")
    return ReasoningConfig(enabled=enabled, effort=effort, budget_tokens=budget)


def _decode_string_list(value: object, parameter: str) -> tuple[str, ...]:
    if value is None:
        return ()
    values = require_list(value, protocol=_PROTOCOL, parameter=parameter)
    return tuple(
        require_string(item, protocol=_PROTOCOL, parameter=f"{parameter}[{index}]")
        for index, item in enumerate(values)
    )


def _encode_request(
    request: SemanticRequest,
    *,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    if request.provider_extensions:
        key = sorted(request.provider_extensions)[0]
        reject(_PROTOCOL, key, "provider extension is not portable")
    if request.candidate_count not in {None, 1}:
        reject(
            _PROTOCOL,
            "candidate_count",
            "Anthropic Messages supports exactly one candidate",
        )
    for name, value in (
        ("frequency_penalty", request.frequency_penalty),
        ("presence_penalty", request.presence_penalty),
        ("response_mime_type", request.response_mime_type),
        ("user", request.user),
    ):
        if value is not None:
            reject(_PROTOCOL, name, "field is not supported by Anthropic Messages")
    if request.max_output_tokens is None:
        reject(_PROTOCOL, "max_output_tokens", "Anthropic Messages requires max_tokens")

    system_messages: list[SemanticMessage] = []
    messages: list[dict[str, Any]] = []
    seen_conversation = False
    for index, item in enumerate(request.input):
        if isinstance(item, SemanticMessage) and item.role is MessageRole.SYSTEM:
            if seen_conversation:
                reject(
                    _PROTOCOL,
                    f"input[{index}]",
                    "Anthropic system instructions must precede conversation messages",
                )
            system_messages.append(item)
            continue
        seen_conversation = True
        messages.append(
            _encode_message(
                item,
                parameter=f"input[{index}]",
                model=request.model,
                encode_opaque_state=encode_opaque_state,
            )
        )

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_output_tokens,
    }
    if system_messages:
        payload["system"] = _encode_system(system_messages)
    _put(payload, "stream", request.stream, request, default=False)
    _put(payload, "temperature", request.temperature, request)
    _put(payload, "top_p", request.top_p, request)
    _put(payload, "top_k", request.top_k, request)
    _put(payload, "service_tier", request.service_tier, request)
    if request.stop_sequences or "stop_sequences" in request.explicit_fields:
        payload["stop_sequences"] = list(request.stop_sequences)
    if request.metadata or "metadata" in request.explicit_fields:
        payload["metadata"] = thaw_json(request.metadata)
    if request.tools or "tools" in request.explicit_fields:
        payload["tools"] = [_encode_tool(tool) for tool in request.tools]
    choice = _encode_tool_choice(request.tool_choice, request.parallel_tool_calls)
    if choice is not None:
        payload["tool_choice"] = choice
    if request.reasoning is not None:
        thinking, effort = _encode_reasoning(request.reasoning)
        if thinking is not None:
            payload["thinking"] = thinking
    else:
        effort = None
    if request.structured_output is not None or effort is not None:
        output_config: dict[str, Any] = {}
        if effort is not None:
            output_config["effort"] = effort
        if request.structured_output is not None:
            output_config["format"] = _anthropic_output_format(request.structured_output)
        payload["output_config"] = output_config
    return payload


def _put(
    payload: dict[str, Any],
    name: str,
    value: object,
    request: SemanticRequest,
    *,
    default: object = None,
) -> None:
    if value != default or name in request.explicit_fields:
        payload[name] = value


def _encode_system(messages: list[SemanticMessage]) -> list[dict[str, Any]]:
    blocks = []
    for message_index, message in enumerate(messages):
        if message.name is not None or message.item_id is not None or message.status is not None:
            reject(
                _PROTOCOL,
                f"system[{message_index}]",
                "system message metadata is not supported",
            )
        for part_index, part in enumerate(message.content):
            if not isinstance(part, TextContent):
                reject(
                    _PROTOCOL,
                    f"system[{message_index}].content[{part_index}]",
                    "system instructions support text only",
                )
            blocks.append({"type": "text", "text": part.text})
    return blocks


def _encode_message(
    item: object,
    *,
    parameter: str,
    model: str,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    if isinstance(item, ToolCall):
        item = SemanticMessage(role=MessageRole.ASSISTANT, content=(item,))
    elif isinstance(item, ToolResult):
        item = SemanticMessage(role=MessageRole.TOOL, content=(item,))
    elif isinstance(item, TextContent | ImageContent | FileContent):
        item = SemanticMessage(role=MessageRole.USER, content=(item,))
    elif isinstance(item, RefusalContent | ReasoningSummary):
        item = SemanticMessage(role=MessageRole.ASSISTANT, content=(item,))
    if not isinstance(item, SemanticMessage):
        reject(_PROTOCOL, parameter, f"unsupported item {type(item).__name__}")
    if item.name is not None or item.item_id is not None or item.status is not None:
        reject(_PROTOCOL, parameter, "Anthropic messages cannot carry message metadata")
    if item.role is MessageRole.SYSTEM:
        reject(_PROTOCOL, parameter, "system instructions must use top-level system")
    role = "user" if item.role in {MessageRole.USER, MessageRole.TOOL} else "assistant"
    blocks = []
    for index, part in enumerate(item.content):
        path = f"{parameter}.content[{index}]"
        if isinstance(part, TextContent):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            if role != "user":
                reject(_PROTOCOL, path, "image blocks require user role")
            blocks.append(_encode_image(part, parameter=path))
        elif isinstance(part, FileContent):
            if role != "user":
                reject(_PROTOCOL, path, "document blocks require user role")
            blocks.append(_encode_document(part, parameter=path))
        elif isinstance(part, ToolCall):
            if role != "assistant":
                reject(_PROTOCOL, path, "tool calls require assistant role")
            blocks.append(_encode_tool_call(part, parameter=path))
        elif isinstance(part, ToolResult):
            if role != "user":
                reject(_PROTOCOL, path, "tool results require user/tool role")
            blocks.append(_encode_tool_result(part, parameter=path))
        elif isinstance(part, ReasoningSummary):
            if role != "assistant":
                reject(_PROTOCOL, path, "reasoning requires assistant role")
            blocks.append(
                _encode_reasoning_block(
                    part,
                    parameter=path,
                    model=model,
                    encode_opaque_state=encode_opaque_state,
                )
            )
        elif isinstance(part, RefusalContent):
            reject(
                _PROTOCOL,
                path,
                "Anthropic Messages request history has no refusal content carrier",
            )
        else:  # pragma: no cover - closed semantic union
            reject(_PROTOCOL, path, f"unsupported content {type(part).__name__}")
    return {"role": role, "content": blocks}


def _encode_image(image: ImageContent, *, parameter: str) -> dict[str, Any]:
    source_kind = image.source_kind
    if source_kind is None:
        if isinstance(image.source, bytes):
            source_kind = "base64"
        else:
            reject(_PROTOCOL, f"{parameter}.source_kind", "string image source is ambiguous")
    if source_kind != "base64":
        reject(_PROTOCOL, f"{parameter}.source_kind", "Anthropic images require base64")
    if image.media_type is None:
        reject(_PROTOCOL, f"{parameter}.media_type", "is required for base64 images")
    data = (
        base64.b64encode(image.source).decode("ascii")
        if isinstance(image.source, bytes)
        else image.source
    )
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": image.media_type, "data": data},
    }


def _encode_document(document: FileContent, *, parameter: str) -> dict[str, Any]:
    source_kind = document.source_kind
    if source_kind is None:
        if isinstance(document.source, bytes):
            source_kind = "base64"
        else:
            reject(_PROTOCOL, f"{parameter}.source_kind", "string file source is ambiguous")
    if source_kind not in {"base64", "url", "text", "content"}:
        reject(_PROTOCOL, f"{parameter}.source_kind", f"unsupported kind {source_kind!r}")
    if isinstance(document.source, bytes):
        if source_kind != "base64":
            reject(_PROTOCOL, parameter, "binary document source requires base64 kind")
        source_value = base64.b64encode(document.source).decode("ascii")
    else:
        source_value = document.source
    source: dict[str, Any] = {"type": source_kind}
    if source_kind == "url":
        source["url"] = source_value
    elif source_kind == "content":
        source["content"] = source_value
    else:
        source["data"] = source_value
    if document.media_type is not None:
        source["media_type"] = document.media_type
    payload: dict[str, Any] = {"type": "document", "source": source}
    if document.filename is not None:
        payload["title"] = document.filename
    return payload


def _encode_tool_call(call: ToolCall, *, parameter: str) -> dict[str, Any]:
    if call.kind != "function":
        reject(_PROTOCOL, f"{parameter}.kind", "Anthropic supports function tools only")
    if call.namespace is not None:
        reject(_PROTOCOL, f"{parameter}.namespace", "Anthropic tools lack namespaces")
    if call.opaque_state is not None:
        reject(_PROTOCOL, f"{parameter}.opaque_state", "tool calls cannot carry opaque state")
    return {
        "type": "tool_use",
        "id": call.call_id,
        "name": call.name,
        "input": thaw_json(call.arguments),
    }


def _encode_tool_result(result: ToolResult, *, parameter: str) -> dict[str, Any]:
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
            "Anthropic tools lack namespaces",
        )
    blocks = []
    for index, part in enumerate(result.content):
        path = f"{parameter}.content[{index}]"
        if isinstance(part, TextContent):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            blocks.append(_encode_image(part, parameter=path))
        elif isinstance(part, FileContent):
            blocks.append(_encode_document(part, parameter=path))
        else:
            reject(_PROTOCOL, path, f"unsupported tool result content {type(part).__name__}")
    if result.structured_content is not None:
        if blocks:
            reject(_PROTOCOL, parameter, "cannot combine blocks and structured tool content")
        blocks.append(
            {
                "type": "text",
                "text": json.dumps(thaw_json(result.structured_content), ensure_ascii=False),
            }
        )
    payload: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": blocks,
    }
    if result.is_error:
        payload["is_error"] = True
    return payload


def _encode_reasoning_block(
    reasoning: ReasoningSummary,
    *,
    parameter: str,
    model: str,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "thinking", "thinking": reasoning.text}
    if reasoning.opaque_state is not None:
        state = reasoning.opaque_state
        if state.origin_protocol is _PROTOCOL and isinstance(state.blob, Mapping):
            if state.origin_model != model:
                reject(_PROTOCOL, parameter, "opaque reasoning model provenance does not match")
            raw = thaw_json(state.blob)
            raw_type = raw.get("type")
            if raw_type not in {"thinking", "redacted_thinking"}:
                reject(_PROTOCOL, parameter, "stored reasoning block has an invalid type")
            return raw
        if state.origin_protocol is _PROTOCOL and isinstance(state.blob, str):
            payload["signature"] = state.blob
            return payload
        if encode_opaque_state is None:
            reject(
                _PROTOCOL,
                parameter,
                "foreign opaque reasoning requires capsule encoder context",
            )
        try:
            capsule = encode_opaque_state(
                state,
                protocol=_PROTOCOL,
                model=state.origin_model,
                item_id=state.item_id,
            )
        except ValueError:
            reject(_PROTOCOL, parameter, "opaque reasoning could not be sealed")
        if not isinstance(capsule, str) or not capsule.startswith("rmr1."):
            reject(_PROTOCOL, parameter, "capsule encoder returned invalid state")
        if reasoning.text:
            payload["signature"] = capsule
        else:
            payload = {"type": "redacted_thinking", "data": capsule}
    return payload


def _encode_tool(tool: ToolDefinition) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": tool.name,
        "input_schema": thaw_json(tool.input_schema),
    }
    if tool.description is not None:
        payload["description"] = tool.description
    if tool.strict is not None:
        payload["strict"] = tool.strict
    return payload


def _encode_tool_choice(
    choice: ToolChoice | None,
    parallel_tool_calls: bool | None,
) -> dict[str, Any] | None:
    if choice is None and parallel_tool_calls is None:
        return None
    payload: dict[str, Any]
    if choice is None or choice.mode == "auto":
        payload = {"type": "auto"}
    elif choice.mode == "required":
        payload = {"type": "any"}
    elif choice.mode == "none":
        payload = {"type": "none"}
    elif choice.mode == "function" and choice.name:
        payload = {"type": "tool", "name": choice.name}
    else:
        reject(_PROTOCOL, "tool_choice.mode", f"unsupported mode {choice.mode!r}")
    if parallel_tool_calls is not None:
        payload["disable_parallel_tool_use"] = not parallel_tool_calls
    return payload


def _encode_reasoning(
    reasoning: ReasoningConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    thinking: dict[str, Any] | None = None
    if reasoning.enabled is not None or reasoning.budget_tokens is not None:
        thinking_type = "disabled" if reasoning.enabled is False else "enabled"
        thinking = {"type": thinking_type}
        if reasoning.budget_tokens is not None:
            if reasoning.enabled is False:
                reject(_PROTOCOL, "reasoning.budget_tokens", "disabled reasoning has no budget")
            thinking["budget_tokens"] = reasoning.budget_tokens
    return thinking, reasoning.effort


def _anthropic_output_format(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = thaw_json(value)
    if "format" in raw and isinstance(raw["format"], Mapping):
        raw = dict(raw["format"])
    if raw.get("type") == "json_schema" and isinstance(raw.get("json_schema"), Mapping):
        json_schema = raw["json_schema"]
        schema = json_schema.get("schema")
        if not isinstance(schema, Mapping):
            reject(_PROTOCOL, "structured_output.json_schema.schema", "must be an object")
        return {"type": "json_schema", "schema": dict(schema)}
    if raw.get("type") == "json_schema" and isinstance(raw.get("schema"), Mapping):
        return raw
    if isinstance(raw.get("schema"), Mapping):
        return {"type": "json_schema", "schema": dict(raw["schema"])}
    if raw.get("type") in {"object", "array", "string", "number", "integer", "boolean"}:
        return {"type": "json_schema", "schema": raw}
    reject(_PROTOCOL, "structured_output", "unsupported structured output shape")


def _decode_response(
    payload: Mapping[str, Any],
    *,
    origin_provider: str | None,
    decode_opaque_state: OpaqueStateDecodeHook | None,
) -> SemanticResponse:
    body = require_mapping(payload, protocol=_PROTOCOL, parameter="response")
    reject_unknown_keys(
        body,
        frozenset(
            {"id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "usage"}
        ),
        protocol=_PROTOCOL,
        parameter="response",
    )
    if body.get("type", "message") != "message":
        decode_reject(_PROTOCOL, "response.type", "must be message")
    if body.get("role", "assistant") != "assistant":
        decode_reject(_PROTOCOL, "response.role", "must be assistant")
    model = require_string(
        body.get("model"), protocol=_PROTOCOL, parameter="response.model", allow_empty=False
    )
    messages = _decode_message(
        {"role": "assistant", "content": body.get("content")},
        parameter="response",
        model=model,
        message_index=0,
        origin_provider=origin_provider,
        decode_opaque_state=decode_opaque_state,
    )
    if len(messages) != 1:
        decode_reject(_PROTOCOL, "response.content", "assistant response cannot split messages")
    stop_reason = optional_string(
        body.get("stop_reason"), protocol=_PROTOCOL, parameter="response.stop_reason"
    )
    finish_reason = None
    if stop_reason is not None:
        finish_reason = _STOP_TO_SEMANTIC.get(stop_reason)
        if finish_reason is None:
            decode_reject(
                _PROTOCOL,
                "response.stop_reason",
                f"cannot project stop reason {stop_reason!r}",
            )
    stop_sequence = optional_string(
        body.get("stop_sequence"),
        protocol=_PROTOCOL,
        parameter="response.stop_sequence",
    )
    usage, usage_metadata = _decode_usage(body.get("usage"))
    return SemanticResponse(
        id=require_string(
            body.get("id"), protocol=_PROTOCOL, parameter="response.id", allow_empty=False
        ),
        model=model,
        output=messages,
        usage=usage,
        terminal=TerminalMetadata(
            finish_reason=finish_reason,
            stop_sequence=stop_sequence,
            response_status=(
                "incomplete" if finish_reason in {"length", "content_filter"} else "completed"
            ),
        ),
        metadata=usage_metadata,
    )


def _decode_usage(value: object) -> tuple[Usage, dict[str, Any]]:
    usage = require_mapping(value, protocol=_PROTOCOL, parameter="response.usage")
    reject_unknown_keys(
        usage,
        frozenset(
            {
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "service_tier",
            }
        ),
        protocol=_PROTOCOL,
        parameter="response.usage",
    )
    input_tokens = optional_int(
        usage.get("input_tokens"),
        protocol=_PROTOCOL,
        parameter="response.usage.input_tokens",
    )
    output_tokens = optional_int(
        usage.get("output_tokens"),
        protocol=_PROTOCOL,
        parameter="response.usage.output_tokens",
    )
    if input_tokens is None or output_tokens is None:
        decode_reject(_PROTOCOL, "response.usage", "input_tokens and output_tokens are required")
    metadata = {}
    for key in ("cache_creation_input_tokens", "service_tier"):
        if usage.get(key) is not None:
            metadata[key] = usage[key]
    return (
        Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_input_tokens=optional_int(
                usage.get("cache_read_input_tokens"),
                protocol=_PROTOCOL,
                parameter="response.usage.cache_read_input_tokens",
            ),
        ),
        metadata,
    )


def _encode_response(
    response: SemanticResponse,
    *,
    encode_opaque_state: OpaqueStateEncodeHook | None,
) -> dict[str, Any]:
    if response.id is None:
        reject(_PROTOCOL, "response.id", "Anthropic responses require an ID")
    blocks = []
    has_refusal = False
    for item_index, item in enumerate(response.output):
        parts = item.content if isinstance(item, SemanticMessage) else (item,)
        if isinstance(item, SemanticMessage) and item.role is not MessageRole.ASSISTANT:
            reject(_PROTOCOL, f"response.output[{item_index}].role", "must be assistant")
        for part_index, part in enumerate(parts):
            path = f"response.output[{item_index}].content[{part_index}]"
            if isinstance(part, TextContent):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, RefusalContent):
                has_refusal = True
                blocks.append({"type": "text", "text": part.refusal})
            elif isinstance(part, ToolCall):
                blocks.append(_encode_tool_call(part, parameter=path))
            elif isinstance(part, ReasoningSummary):
                blocks.append(
                    _encode_reasoning_block(
                        part,
                        parameter=path,
                        model=response.model,
                        encode_opaque_state=encode_opaque_state,
                    )
                )
            else:
                reject(_PROTOCOL, path, f"unsupported assistant output {type(part).__name__}")

    stop_reason = None
    stop_sequence = None
    if response.terminal is not None:
        terminal = response.terminal
        if terminal.error_code is not None or terminal.error_message is not None:
            reject(_PROTOCOL, "response.terminal", "Messages response cannot carry an error")
        status_error = _terminal_status_error(
            terminal,
            parameter="response.terminal.response_status",
        )
        if status_error is not None:
            reject(
                _PROTOCOL,
                "response.terminal.response_status",
                f"Messages responses cannot represent {terminal.response_status!r} status",
            )
        if terminal.transport_status is not None:
            reject(
                _PROTOCOL,
                "response.terminal.transport_status",
                "transport status is not a Messages field",
            )
        if terminal.finish_reason is not None:
            stop_reason = _STOP_FROM_SEMANTIC.get(terminal.finish_reason)
            if stop_reason is None:
                reject(
                    _PROTOCOL,
                    "response.terminal.finish_reason",
                    f"unsupported reason {terminal.finish_reason!r}",
                )
            if has_refusal and stop_reason != "refusal":
                reject(
                    _PROTOCOL,
                    "response.terminal.finish_reason",
                    "conflicts with refusal output",
                )
        stop_sequence = terminal.stop_sequence
    if has_refusal:
        stop_reason = "refusal"
    elif stop_reason is None:
        stop_reason = (
            "tool_use" if any(block["type"] == "tool_use" for block in blocks) else "end_turn"
        )
    if stop_sequence is not None:
        if stop_reason not in {"end_turn", "stop_sequence"}:
            reject(_PROTOCOL, "response.terminal.stop_sequence", "conflicts with finish reason")
        stop_reason = "stop_sequence"

    usage = _encode_usage(response.usage, response.metadata)
    unknown_metadata = sorted(set(response.metadata) - _RESPONSE_METADATA_FIELDS)
    if unknown_metadata:
        reject(
            _PROTOCOL,
            f"response.metadata.{unknown_metadata[0]}",
            "metadata is not portable",
        )
    return {
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": response.model,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": usage,
    }


def _encode_usage(
    usage: Usage | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if usage is None:
        reject(_PROTOCOL, "response.usage", "Anthropic responses require usage")
    if usage.mode is not UsageMode.SNAPSHOT:
        reject(_PROTOCOL, "response.usage.mode", "non-stream usage must be a snapshot")
    if usage.input_tokens is None or usage.output_tokens is None:
        reject(_PROTOCOL, "response.usage", "input and output tokens are required")
    payload: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    if usage.cached_input_tokens is not None:
        payload["cache_read_input_tokens"] = usage.cached_input_tokens
    if metadata.get("cache_creation_input_tokens") is not None:
        payload["cache_creation_input_tokens"] = metadata["cache_creation_input_tokens"]
    if metadata.get("service_tier") is not None:
        payload["service_tier"] = metadata["service_tier"]
    return payload


__all__ = [
    "AnthropicMessagesRuntime",
    "AnthropicStreamDecoder",
    "AnthropicStreamEncoder",
]
