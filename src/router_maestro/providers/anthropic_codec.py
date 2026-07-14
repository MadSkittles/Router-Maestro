"""Decode native Anthropic response objects into canonical Chat values."""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from router_maestro.providers.base import ChatResponse, ChatStreamChunk

CanonicalStopReason = Literal["stop", "length", "tool_calls"]
KNOWN_BLOCK_TYPES = frozenset({"text", "thinking", "redacted_thinking", "tool_use"})
POST_TERMINAL_FORBIDDEN_EVENTS = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
    }
)


class AnthropicCodecError(ValueError):
    """A native Anthropic response cannot be represented canonically."""


class AnthropicStopCause(StrEnum):
    """Lossless native Anthropic terminal causes understood by this codec."""

    END_TURN = "end_turn"
    STOP_SEQUENCE = "stop_sequence"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


@dataclass(slots=True)
class _StreamBlockState:
    """Lifecycle and canonical projection metadata for one native content block."""

    block_type: str
    known: bool
    open: bool = True
    tool_index: int | None = None
    has_initial_tool_input: bool = False


def parse_stop_cause(native: str) -> AnthropicStopCause:
    """Parse a known native terminal cause without applying Chat policy."""
    try:
        return AnthropicStopCause(native)
    except ValueError as exc:
        raise AnthropicCodecError(f"unsupported stop_reason: {native!r}") from exc


def project_stop_cause(cause: AnthropicStopCause) -> CanonicalStopReason:
    """Project a native terminal cause into the current canonical Chat contract."""
    mapping: dict[AnthropicStopCause, CanonicalStopReason] = {
        AnthropicStopCause.END_TURN: "stop",
        AnthropicStopCause.STOP_SEQUENCE: "stop",
        AnthropicStopCause.MAX_TOKENS: "length",
        AnthropicStopCause.TOOL_USE: "tool_calls",
    }
    try:
        return mapping[cause]
    except KeyError as exc:
        raise AnthropicCodecError(
            f"stop_reason {cause.value!r} cannot be projected to canonical Chat"
        ) from exc


def canonical_stop_reason(native: str | None) -> CanonicalStopReason | None:
    """Map one explicit native terminal cause without guessing unknown values."""
    if native is None:
        return None
    return project_stop_cause(parse_stop_cause(native))


def _required_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnthropicCodecError(f"{label} must be an object")
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AnthropicCodecError(f"{label} must be a string")
    return value


def _token_count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AnthropicCodecError(f"{label} must be a non-negative integer")
    return value


def _usage(value: object, *, label: str) -> tuple[int, int]:
    if value is None:
        return 0, 0
    usage = _required_object(value, f"{label} usage")
    input_tokens = _token_count(usage.get("input_tokens", 0), f"{label} usage input_tokens")
    output_tokens = _token_count(usage.get("output_tokens", 0), f"{label} usage output_tokens")
    return input_tokens, output_tokens


def _chat_usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def decode_message_response(
    data: object,
    *,
    fallback_model: str,
    include_reasoning: bool,
) -> ChatResponse:
    """Decode one complete native Messages response."""
    message = _required_object(data, "messages response")
    blocks = message.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise AnthropicCodecError("messages content must be a non-empty list")

    model = message.get("model", fallback_model)
    if not isinstance(model, str):
        raise AnthropicCodecError("response model must be a string")

    # Compatibility: older Anthropic-compatible endpoints omit stop_reason on
    # otherwise complete responses. The previous provider treated omission as
    # canonical ``stop``; explicit null remains non-terminal and is rejected.
    native_stop = message.get("stop_reason", "end_turn")
    if native_stop is not None and not isinstance(native_stop, str):
        raise AnthropicCodecError("response stop_reason must be a string or null")
    if native_stop is None:
        raise AnthropicCodecError("messages response stop_reason cannot be null")
    finish_reason = canonical_stop_reason(native_stop)
    assert finish_reason is not None

    input_tokens, output_tokens = _usage(message.get("usage"), label="messages response")
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    thinking_parts: list[str] = []
    thinking_signature: str | None = None
    reasoning_block_count = 0
    signature_owner_seen = False

    for raw_block in blocks:
        block = _required_object(raw_block, "content block")
        block_type = _required_string(block.get("type"), "content block type")
        if block_type == "text":
            text_parts.append(_required_string(block.get("text"), "text content block text"))
        elif block_type == "tool_use":
            tool_input = block.get("input", {})
            if not isinstance(tool_input, dict):
                raise AnthropicCodecError("tool use input must be an object")
            tool_id = _required_string(block.get("id"), "tool use id")
            name = _required_string(block.get("name"), "tool use name")
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(tool_input),
                    },
                }
            )
        elif block_type == "thinking":
            thinking = _required_string(block.get("thinking"), "thinking block thinking")
            signature = block.get("signature")
            if signature is not None and not isinstance(signature, str):
                raise AnthropicCodecError("thinking signature must be a string or null")
            if include_reasoning:
                has_signature = bool(signature)
                if reasoning_block_count and (signature_owner_seen or has_signature):
                    raise AnthropicCodecError(
                        "thinking signature cannot represent multiple reasoning blocks"
                    )
                reasoning_block_count += 1
                signature_owner_seen = signature_owner_seen or has_signature
                thinking_parts.append(thinking)
                if signature:
                    thinking_signature = signature
        elif block_type == "redacted_thinking":
            redacted = _required_string(block.get("data"), "redacted thinking data")
            if include_reasoning:
                has_signature = bool(redacted)
                if reasoning_block_count and (signature_owner_seen or has_signature):
                    raise AnthropicCodecError(
                        "thinking signature cannot represent multiple reasoning blocks"
                    )
                reasoning_block_count += 1
                signature_owner_seen = signature_owner_seen or has_signature
                if redacted:
                    thinking_signature = redacted

    content = "".join(text_parts)
    thinking = "".join(thinking_parts)
    if not content and not tool_calls and not thinking and not thinking_signature:
        raise AnthropicCodecError("messages response contains no deliverable output")

    return ChatResponse(
        content=content or None,
        model=model,
        finish_reason=finish_reason,
        usage=_chat_usage(input_tokens, output_tokens),
        tool_calls=tool_calls or None,
        thinking=thinking or None,
        thinking_signature=thinking_signature,
    )


@dataclass(slots=True)
class AnthropicStreamDecoder:
    """Stateful reducer from native Anthropic events to canonical Chat chunks."""

    include_reasoning: bool
    prompt_tokens: int = 0
    blocks: dict[int, _StreamBlockState] = field(default_factory=dict)
    reasoning_block_indices: set[int] = field(default_factory=set)
    signature_owner_index: int | None = None
    next_tool_index: int = 0
    saw_deliverable: bool = False
    saw_explicit_terminal: bool = False

    def decode_event(self, data: object) -> list[ChatStreamChunk]:
        """Validate and decode one parsed Anthropic SSE event object."""
        event = _required_object(data, "stream event")
        event_type = _required_string(event.get("type"), "stream event type")
        if self.saw_explicit_terminal and event_type in POST_TERMINAL_FORBIDDEN_EVENTS:
            raise AnthropicCodecError(f"{event_type} arrived after explicit terminal")
        if event_type == "message_start":
            self._decode_message_start(event)
            return []
        if event_type == "content_block_start":
            return self._decode_content_block_start(event)
        if event_type == "content_block_delta":
            return self._decode_content_block_delta(event)
        if event_type == "content_block_stop":
            self._decode_content_block_stop(event)
            return []
        if event_type == "message_delta":
            return [self._decode_message_delta(event)]
        return []

    def finalize(self) -> None:
        """Validate that normal transport EOF followed a complete native stream."""
        if not self.saw_deliverable:
            raise AnthropicCodecError("stream contained no deliverable output")
        if not self.saw_explicit_terminal:
            raise AnthropicCodecError("stream ended without an explicit terminal stop_reason")

    @staticmethod
    def _content_index(event: dict[str, Any]) -> int:
        index = event.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise AnthropicCodecError("content block index must be an integer")
        return index

    def _decode_message_start(self, event: dict[str, Any]) -> None:
        message = _required_object(event.get("message"), "message_start message")
        usage = _required_object(message.get("usage", {}), "message_start usage")
        self.prompt_tokens = _token_count(
            usage.get("input_tokens", 0), "message_start input_tokens"
        )

    def _decode_content_block_start(self, event: dict[str, Any]) -> list[ChatStreamChunk]:
        index = self._content_index(event)
        block = _required_object(event.get("content_block"), "content block")
        block_type = _required_string(block.get("type"), "content block type")
        existing = self.blocks.get(index)
        known = block_type in KNOWN_BLOCK_TYPES
        if existing is not None:
            if existing.known or known:
                raise AnthropicCodecError("duplicate content block index")
            return []
        if not known:
            self.blocks[index] = _StreamBlockState(block_type=block_type, known=False)
            return []

        if block_type == "tool_use":
            tool_id = _required_string(block.get("id"), "tool use id")
            name = _required_string(block.get("name"), "tool use name")
            tool_input = block.get("input", {})
            if not isinstance(tool_input, dict):
                raise AnthropicCodecError("tool use input must be an object")
            tool_index = self.next_tool_index
            arguments = json.dumps(tool_input) if tool_input else ""
            self.blocks[index] = _StreamBlockState(
                block_type=block_type,
                known=True,
                tool_index=tool_index,
                has_initial_tool_input=bool(tool_input),
            )
            self.next_tool_index += 1
            self.saw_deliverable = True
            return [
                ChatStreamChunk(
                    content="",
                    tool_calls=[
                        {
                            "index": tool_index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                )
            ]
        if block_type == "text":
            text = _required_string(block.get("text", ""), "text block text")
            self.blocks[index] = _StreamBlockState(block_type=block_type, known=True)
            if text:
                self.saw_deliverable = True
                return [ChatStreamChunk(content=text)]
            return []
        if block_type == "thinking":
            thinking = _required_string(block.get("thinking", ""), "thinking block thinking")
            signature = _required_string(block.get("signature", ""), "thinking block signature")
            self._validate_reasoning_signature_owner(index, bool(signature))
            self.blocks[index] = _StreamBlockState(block_type=block_type, known=True)
            self._record_reasoning_block(index, bool(signature))
            if self.include_reasoning and (thinking or signature):
                self.saw_deliverable = True
                return [
                    ChatStreamChunk(
                        content="",
                        thinking=thinking or None,
                        thinking_signature=signature or None,
                    )
                ]
            return []
        if block_type == "redacted_thinking":
            redacted = _required_string(block.get("data"), "redacted thinking data")
            self._validate_reasoning_signature_owner(index, bool(redacted))
            self.blocks[index] = _StreamBlockState(block_type=block_type, known=True)
            self._record_reasoning_block(index, bool(redacted))
            if self.include_reasoning and redacted:
                self.saw_deliverable = True
                return [ChatStreamChunk(content="", thinking_signature=redacted)]
            return []
        raise AssertionError(f"unhandled known block type: {block_type}")

    def _decode_content_block_delta(self, event: dict[str, Any]) -> list[ChatStreamChunk]:
        index = self._content_index(event)
        delta = _required_object(event.get("delta"), "content block delta")
        delta_type = _required_string(delta.get("type"), "content delta type")
        field_by_type = {
            "text_delta": "text",
            "thinking_delta": "thinking",
            "signature_delta": "signature",
            "input_json_delta": "partial_json",
        }
        field_name = field_by_type.get(delta_type)
        if field_name is None:
            return []
        block = self.blocks.get(index)
        if block is None:
            raise AnthropicCodecError("known content delta arrived before content block start")
        if not block.known:
            return []
        if not block.open:
            raise AnthropicCodecError("known content delta arrived for a closed content block")
        expected_block_type = {
            "text_delta": "text",
            "thinking_delta": "thinking",
            "signature_delta": "thinking",
            "input_json_delta": "tool_use",
        }[delta_type]
        if block.block_type != expected_block_type:
            raise AnthropicCodecError(
                f"{delta_type} is incompatible with {block.block_type} content block"
            )
        value = _required_string(delta.get(field_name), f"{delta_type} {field_name}")
        if delta_type == "text_delta":
            if value:
                self.saw_deliverable = True
            return [ChatStreamChunk(content=value)]
        if delta_type == "thinking_delta":
            if self.include_reasoning and value:
                self.saw_deliverable = True
            return (
                [ChatStreamChunk(content="", thinking=value or None)]
                if self.include_reasoning
                else []
            )
        if delta_type == "signature_delta":
            self._validate_reasoning_signature_owner(index, bool(value))
            if self.include_reasoning and value:
                self.signature_owner_index = index
            if self.include_reasoning and value:
                self.saw_deliverable = True
            return (
                [ChatStreamChunk(content="", thinking_signature=value or None)]
                if self.include_reasoning
                else []
            )
        tool_index = block.tool_index
        assert tool_index is not None
        if block.has_initial_tool_input:
            raise AnthropicCodecError(
                "tool block cannot stream partial JSON after non-empty initial input"
            )
        if value:
            self.saw_deliverable = True
        return [
            ChatStreamChunk(
                content="",
                tool_calls=[
                    {
                        "index": tool_index,
                        "function": {"arguments": value},
                    }
                ],
            )
        ]

    def _decode_content_block_stop(self, event: dict[str, Any]) -> None:
        index = self._content_index(event)
        block = self.blocks.get(index)
        if block is None:
            raise AnthropicCodecError("content block stop arrived before content block start")
        if not block.known:
            block.open = False
            return
        if not block.open:
            raise AnthropicCodecError("duplicate content block stop")
        block.open = False

    def _validate_reasoning_signature_owner(self, index: int, has_signature: bool) -> None:
        if not self.include_reasoning:
            return
        other_reasoning_exists = bool(self.reasoning_block_indices - {index})
        other_signature_owner = (
            self.signature_owner_index is not None and self.signature_owner_index != index
        )
        if other_signature_owner or (has_signature and other_reasoning_exists):
            raise AnthropicCodecError(
                "thinking signature cannot represent multiple reasoning blocks"
            )

    def _record_reasoning_block(self, index: int, has_signature: bool) -> None:
        if not self.include_reasoning:
            return
        self.reasoning_block_indices.add(index)
        if has_signature:
            self.signature_owner_index = index

    def _decode_message_delta(self, event: dict[str, Any]) -> ChatStreamChunk:
        delta = _required_object(event.get("delta"), "message delta")
        native_stop = delta.get("stop_reason")
        if native_stop is not None and not isinstance(native_stop, str):
            raise AnthropicCodecError("message stop_reason must be a string or null")
        usage = _required_object(event.get("usage", {}), "message_delta usage")
        output_tokens = _token_count(usage.get("output_tokens", 0), "message_delta output_tokens")
        finish_reason = canonical_stop_reason(native_stop) if native_stop is not None else None
        if finish_reason is not None:
            if not self.saw_deliverable:
                raise AnthropicCodecError("stream terminal preceded all deliverable output")
            self.saw_explicit_terminal = True
        return ChatStreamChunk(
            content="",
            finish_reason=finish_reason,
            usage=_chat_usage(self.prompt_tokens, output_tokens),
        )
