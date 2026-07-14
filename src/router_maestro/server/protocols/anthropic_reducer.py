"""Reduce canonical Chat responses and chunks to downstream Anthropic values."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from router_maestro.providers.base import (
    ChatResponse,
    ChatStreamChunk,
    ProviderError,
    ProviderFailureKind,
    ResponseStatus,
    finish_reason_for_outcome,
    resolve_terminal_outcome,
)
from router_maestro.server.schemas.anthropic import (
    AnthropicAssistantContentBlock,
    AnthropicMessagesResponse,
    AnthropicStreamState,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolCallAccumulator,
    AnthropicToolUseBlock,
    AnthropicUsage,
)
from router_maestro.utils.tokens import map_openai_stop_reason_to_anthropic

Event = dict[str, Any]


class AnthropicStreamProtocolError(ProviderError):
    """A canonical Chat stream cannot be represented as legal Anthropic SSE."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        )


@dataclass(frozen=True, slots=True)
class Text:
    """One downstream text content block."""

    text: str


@dataclass(frozen=True, slots=True)
class Reasoning:
    """One downstream thinking block and its opaque signature metadata."""

    text: str
    signature: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One complete downstream tool-use block."""

    tool_id: str
    name: str
    input: dict[str, Any]
    arguments_json: str


ContentPart = Text | Reasoning | ToolCall


def _validate_canonical_scalars(value: ChatResponse | ChatStreamChunk) -> None:
    for field_name in ("content", "refusal", "thinking", "thinking_signature"):
        field_value = getattr(value, field_name)
        if field_value is not None and not isinstance(field_value, str):
            raise AnthropicStreamProtocolError(f"{field_name} must be a string or null")


def _validated_usage(usage: object) -> dict[str, Any]:
    if usage is None:
        return {}
    if not isinstance(usage, dict):
        raise AnthropicStreamProtocolError("usage must be an object or null")
    for field_name in ("prompt_tokens", "completion_tokens"):
        value = usage.get(field_name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AnthropicStreamProtocolError(f"usage {field_name} must be a non-negative integer")
    return usage


def _anthropic_stop_reason(finish_reason: str | None) -> str | None:
    if finish_reason not in {None, "stop", "length", "tool_calls", "content_filter"}:
        raise AnthropicStreamProtocolError(
            f"canonical stream has non-success terminal reason: {finish_reason}"
        )
    return map_openai_stop_reason_to_anthropic(finish_reason)


def _parse_tool_call(tool_call: dict[str, Any]) -> ToolCall:
    if not isinstance(tool_call, dict):
        raise AnthropicStreamProtocolError("tool call must be an object")
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise AnthropicStreamProtocolError("tool call function must be an object")
    tool_id = tool_call.get("id")
    if not isinstance(tool_id, str):
        raise AnthropicStreamProtocolError("tool call id must be a string")
    if not tool_id:
        raise AnthropicStreamProtocolError("tool call missing id")
    name = function.get("name")
    if name is not None and not isinstance(name, str):
        raise AnthropicStreamProtocolError("tool call name must be a string")
    if not name:
        raise AnthropicStreamProtocolError("tool call missing name")
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        arguments_json = arguments
        try:
            parsed = json.loads(arguments) if arguments else {}
        except (json.JSONDecodeError, TypeError) as exc:
            raise AnthropicStreamProtocolError(
                "tool call arguments must be a valid JSON object"
            ) from exc
    elif isinstance(arguments, dict):
        parsed = arguments
        arguments_json = json.dumps(arguments)
    else:
        raise AnthropicStreamProtocolError("tool call arguments must be a string or object")
    if not isinstance(parsed, dict):
        raise AnthropicStreamProtocolError("tool call arguments must be a JSON object")
    return ToolCall(
        tool_id=tool_id,
        name=name,
        input=parsed,
        arguments_json=arguments_json,
    )


def _response_parts(response: ChatResponse) -> list[ContentPart]:
    parts: list[ContentPart] = []
    if response.thinking or response.thinking_signature:
        parts.append(Reasoning(response.thinking or "", response.thinking_signature))
    if response.content:
        parts.append(Text(response.content))
    if response.refusal:
        parts.append(Text(response.refusal))
    parts.extend(_parse_tool_call(tool_call) for tool_call in response.tool_calls or [])
    return parts


def _block_dict(part: ContentPart, *, streaming: bool = False) -> dict[str, Any]:
    if isinstance(part, Reasoning):
        block: dict[str, Any] = {"type": "thinking", "thinking": "" if streaming else part.text}
        if not streaming and part.signature is not None:
            block["signature"] = part.signature
        return block
    if isinstance(part, Text):
        return {"type": "text", "text": "" if streaming else part.text}
    return {
        "type": "tool_use",
        "id": part.tool_id,
        "name": part.name,
        "input": {} if streaming else part.input,
    }


def _schema_block(part: ContentPart) -> AnthropicAssistantContentBlock:
    data = _block_dict(part)
    if isinstance(part, Reasoning):
        return AnthropicThinkingBlock(**data)
    if isinstance(part, Text):
        return AnthropicTextBlock(**data)
    return AnthropicToolUseBlock(**data)


def build_anthropic_response(
    response: ChatResponse,
    *,
    response_id: str,
    model: str,
) -> AnthropicMessagesResponse:
    """Build one non-stream response through the shared content rules."""
    _validate_canonical_scalars(response)
    usage = _validated_usage(response.usage)
    return AnthropicMessagesResponse(
        id=response_id,
        type="message",
        role="assistant",
        content=[_schema_block(part) for part in _response_parts(response)],
        model=model,
        stop_reason=_anthropic_stop_reason(response.finish_reason),
        stop_sequence=None,
        usage=AnthropicUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        ),
    )


class AnthropicReducer:
    """Stateful reducer from canonical Chat chunks to Anthropic event objects."""

    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        estimated_input_tokens: int = 0,
        state: AnthropicStreamState | None = None,
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.state = state or AnthropicStreamState(estimated_input_tokens=estimated_input_tokens)

    def start(self) -> list[Event]:
        """Emit the idempotent downstream message_start event."""
        if self.state.message_start_sent:
            return []
        input_tokens = 0
        if self.state.last_usage:
            input_tokens = self.state.last_usage.get("prompt_tokens", 0)
        elif self.state.estimated_input_tokens:
            input_tokens = self.state.estimated_input_tokens
        self.state.message_start_sent = True
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
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 1,
                        "cache_creation_input_tokens": None,
                        "cache_read_input_tokens": None,
                        "server_tool_use": None,
                        "service_tier": "standard",
                    },
                },
            }
        ]

    def reduce(self, chunk: ChatStreamChunk) -> list[Event]:
        """Reduce one canonical Chat chunk to zero or more wire-event objects."""
        if self.state.message_complete:
            return []
        _validate_canonical_scalars(chunk)
        events = self.start()
        self.observe_usage(chunk.usage)
        outcome = resolve_terminal_outcome(chunk.terminal_outcome, chunk.finish_reason)
        finish_reason = None
        if outcome is not None:
            if outcome.response_status not in {
                ResponseStatus.COMPLETED,
                ResponseStatus.INCOMPLETE,
            }:
                raise AnthropicStreamProtocolError(
                    "canonical stream has a non-success terminal outcome"
                )
            finish_reason = finish_reason_for_outcome(outcome)
        self._emit_reasoning(events, chunk)

        text_parts = [text for text in (chunk.content, chunk.refusal) if text]
        if text_parts:
            self._close_thinking(events)
            for text in text_parts:
                self._emit_text(events, text)

        for tool_call in chunk.tool_calls or []:
            self._accumulate_tool_call(tool_call)

        if finish_reason:
            self._emit_terminal(events, chunk, finish_reason)
        return events

    def observe_usage(self, usage: dict[str, Any] | None) -> None:
        """Track a canonical usage snapshot without opening a content stream."""
        validated = _validated_usage(usage)
        if not validated:
            return
        usage = validated
        self.state.last_usage = usage
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0:
            self.state.accumulated_completion_tokens = max(
                self.state.accumulated_completion_tokens, completion_tokens
            )
        prompt_tokens = usage.get("prompt_tokens", 0)
        if prompt_tokens > 0:
            self.state.accumulated_prompt_tokens = max(
                self.state.accumulated_prompt_tokens, prompt_tokens
            )
        if usage.get("completion_tokens_details"):
            self.state.completion_tokens_details = usage["completion_tokens_details"]
        if usage.get("prompt_tokens_details"):
            self.state.prompt_tokens_details = usage["prompt_tokens_details"]

    def _emit_reasoning(self, events: list[Event], chunk: ChatStreamChunk) -> None:
        if not chunk.thinking and not chunk.thinking_signature:
            return
        if not self.state.thinking_block_open:
            self._close_open_block(events, advance=True)
            events.append(
                {
                    "type": "content_block_start",
                    "index": self.state.content_block_index,
                    "content_block": _block_dict(Reasoning(""), streaming=True),
                }
            )
            self.state.thinking_block_open = True
            self.state.content_block_open = True
        if chunk.thinking:
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.state.content_block_index,
                    "delta": {"type": "thinking_delta", "thinking": chunk.thinking},
                }
            )
        if chunk.thinking_signature:
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.state.content_block_index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": chunk.thinking_signature,
                    },
                }
            )

    def _close_thinking(self, events: list[Event]) -> None:
        if not self.state.thinking_block_open:
            return
        self._close_open_block(events, advance=True)

    def _emit_text(self, events: list[Event], text: str) -> None:
        if not self.state.content_block_open:
            events.append(
                {
                    "type": "content_block_start",
                    "index": self.state.content_block_index,
                    "content_block": _block_dict(Text(""), streaming=True),
                }
            )
            self.state.content_block_open = True
        events.append(
            {
                "type": "content_block_delta",
                "index": self.state.content_block_index,
                "delta": {"type": "text_delta", "text": text},
            }
        )

    def _close_open_block(self, events: list[Event], *, advance: bool) -> None:
        if not self.state.content_block_open:
            return
        events.append({"type": "content_block_stop", "index": self.state.content_block_index})
        self.state.content_block_open = False
        self.state.thinking_block_open = False
        if advance:
            self.state.content_block_index += 1

    def _emit_terminal(
        self,
        events: list[Event],
        chunk: ChatStreamChunk,
        finish_reason: str,
    ) -> None:
        tool_calls = self._validated_tool_calls()
        self._close_open_block(events, advance=bool(tool_calls))

        for tool_call in tool_calls:
            index = self.state.content_block_index
            events.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": _block_dict(tool_call, streaming=True),
                    },
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_call.arguments_json,
                        },
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
            self.state.content_block_index += 1

        usage = chunk.usage or self.state.last_usage or {}
        prompt_tokens = self.state.accumulated_prompt_tokens or usage.get("prompt_tokens", 0)
        completion_tokens = self.state.accumulated_completion_tokens or usage.get(
            "completion_tokens", 0
        )
        displayed_input = prompt_tokens if prompt_tokens > 0 else self.state.estimated_input_tokens
        events.extend(
            [
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _anthropic_stop_reason(finish_reason),
                        "stop_sequence": None,
                    },
                    "usage": {
                        "input_tokens": displayed_input,
                        "output_tokens": completion_tokens,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "server_tool_use": None,
                    },
                },
                {"type": "message_stop"},
            ]
        )
        self.state.message_complete = True

    def _accumulate_tool_call(self, tool_call: dict[str, Any]) -> None:
        if not isinstance(tool_call, dict):
            raise AnthropicStreamProtocolError("tool call delta must be an object")
        upstream_index = tool_call.get("index")
        tool_id = tool_call.get("id")
        function = tool_call.get("function")
        if function is None:
            function = {}
        if not isinstance(function, dict):
            raise AnthropicStreamProtocolError("tool call function must be an object")
        name = function.get("name")
        arguments = function.get("arguments")

        if upstream_index is None and not tool_id:
            raise AnthropicStreamProtocolError("tool call delta missing both index and id")
        if upstream_index is not None and (
            isinstance(upstream_index, bool)
            or not isinstance(upstream_index, int)
            or upstream_index < 0
        ):
            raise AnthropicStreamProtocolError("tool call index must be a non-negative integer")
        for field_name, value in (
            ("id", tool_id),
            ("name", name),
            ("arguments", arguments),
        ):
            if value is not None and not isinstance(value, str):
                raise AnthropicStreamProtocolError(f"tool call {field_name} must be a string")

        by_index = (
            next(
                (call for call in self.state.tool_calls if call.upstream_index == upstream_index),
                None,
            )
            if upstream_index is not None
            else None
        )
        by_id = (
            next(
                (call for call in self.state.tool_calls if call.tool_id == tool_id),
                None,
            )
            if tool_id
            else None
        )
        if by_index is not None and by_id is not None and by_index is not by_id:
            raise AnthropicStreamProtocolError("tool call has conflicting index and id")
        call = by_index or by_id
        if call is None:
            call = AnthropicToolCallAccumulator(
                upstream_index=upstream_index,
                tool_id=tool_id or None,
                name=name or None,
                arrival_ordinal=self.state.next_tool_arrival_ordinal,
            )
            self.state.next_tool_arrival_ordinal += 1
            self.state.tool_calls.append(call)
        else:
            if upstream_index is not None:
                if call.upstream_index is not None and call.upstream_index != upstream_index:
                    raise AnthropicStreamProtocolError("tool call has conflicting index")
                call.upstream_index = upstream_index
            if tool_id:
                if call.tool_id is not None and call.tool_id != tool_id:
                    raise AnthropicStreamProtocolError("tool call has conflicting id")
                call.tool_id = tool_id
        if name:
            if call.name is not None and call.name != name:
                raise AnthropicStreamProtocolError("tool call has conflicting name")
            call.name = name
        if arguments:
            call.argument_fragments.append(arguments)

    def _validated_tool_calls(self) -> list[ToolCall]:
        for call in self.state.tool_calls:
            if not call.tool_id:
                raise AnthropicStreamProtocolError("tool call missing id")
            if not call.name:
                raise AnthropicStreamProtocolError("tool call missing name")
            arguments = "".join(call.argument_fragments)
            try:
                parsed = json.loads(arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise AnthropicStreamProtocolError(
                    "tool call arguments must be a valid JSON object"
                ) from exc
            if not isinstance(parsed, dict):
                raise AnthropicStreamProtocolError("tool call arguments must be a JSON object")

        explicitly_indexed = sorted(
            (call for call in self.state.tool_calls if call.upstream_index is not None),
            key=lambda call: call.upstream_index,
        )
        indexless = sorted(
            (call for call in self.state.tool_calls if call.upstream_index is None),
            key=lambda call: call.arrival_ordinal,
        )
        return [
            _parse_tool_call(
                {
                    "id": call.tool_id,
                    "function": {
                        "name": call.name,
                        "arguments": "".join(call.argument_fragments),
                    },
                }
            )
            for call in explicitly_indexed + indexless
        ]
