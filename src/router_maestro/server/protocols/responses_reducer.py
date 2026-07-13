"""Pure state reduction for canonical Responses provider chunks."""

from __future__ import annotations

import copy
import json
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from router_maestro.providers.base import (
    ProviderError,
    ProviderFailureKind,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
    resolve_terminal_outcome,
    unexpected_eof_outcome,
)

Event = dict[str, Any]
IdFactory = Callable[[str], str]


def generate_id(prefix: str) -> str:
    """Generate one Responses-compatible local identifier."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def make_text_content(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": []}


def make_refusal_content(refusal: str) -> dict[str, Any]:
    return {"type": "refusal", "refusal": refusal}


def make_usage(
    raw_usage: dict[str, Any] | None,
    *,
    streaming: bool = True,
) -> dict[str, Any] | None:
    """Normalize usage through one shared stream/non-stream rule source."""
    if not raw_usage:
        return None
    input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("output_tokens", 0)
    input_details = raw_usage.get("input_tokens_details") or {}
    output_details = raw_usage.get("output_tokens_details") or {}
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            input_tokens + output_tokens if streaming else raw_usage.get("total_tokens", 0)
        ),
    }
    if streaming or raw_usage.get("input_tokens_details") is not None:
        usage["input_tokens_details"] = {"cached_tokens": input_details.get("cached_tokens", 0)}
    if streaming or raw_usage.get("output_tokens_details") is not None:
        usage["output_tokens_details"] = {
            "reasoning_tokens": output_details.get("reasoning_tokens", 0)
        }
    return usage


def make_message_item(
    msg_id: str,
    text: str | None,
    status: str = "completed",
    *,
    refusal: str | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append(make_text_content(text))
    if refusal is not None:
        content.append(make_refusal_content(refusal))
    return make_message_item_from_parts(msg_id, content, status)


def make_message_item_from_parts(
    msg_id: str,
    content: list[dict[str, Any]],
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "content": content,
        "status": status,
    }


def make_function_call_item(
    fc_id: str,
    call_id: str,
    name: str,
    arguments: str,
    status: str = "completed",
    namespace: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "function_call",
        "id": fc_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }
    if namespace is not None:
        item["namespace"] = namespace
    return item


def make_custom_tool_call_item(
    item_id: str,
    call_id: str,
    name: str,
    input_text: str,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "type": "custom_tool_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "input": input_text,
        "status": status,
    }


def parse_tool_search_arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments) if arguments else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def make_tool_search_call_item(call_id: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "tool_search_call",
        "call_id": call_id,
        "execution": "client",
        "status": "completed",
        "arguments": parse_tool_search_arguments(arguments),
    }


def make_reasoning_item(
    item_id: str,
    summary: str | list[dict[str, Any]] | None,
    signature: str | None = None,
) -> dict[str, Any]:
    parts = (
        [{"type": "summary_text", "text": summary}]
        if isinstance(summary, str) and summary
        else list(summary or [])
    )
    item: dict[str, Any] = {"type": "reasoning", "id": item_id, "summary": parts}
    if signature:
        item["encrypted_content"] = signature
    return item


def validate_reasoning_signature(item_id: str | None, signature: str | None) -> None:
    """Require the upstream identity against which encrypted reasoning was signed."""
    if signature is not None and item_id is None:
        raise _protocol_error("Reasoning signature is missing its upstream item id")


def make_tool_call_item(
    tool_call: ResponsesToolCall,
    *,
    id_factory: IdFactory = generate_id,
) -> dict[str, Any]:
    if tool_call.kind == "custom":
        return make_custom_tool_call_item(
            id_factory("ctc"),
            tool_call.call_id,
            tool_call.name,
            tool_call.arguments,
        )
    if tool_call.kind == "tool_search":
        return make_tool_search_call_item(tool_call.call_id, tool_call.arguments)
    return make_function_call_item(
        id_factory("fc"),
        tool_call.call_id,
        tool_call.name,
        tool_call.arguments,
        namespace=tool_call.namespace,
    )


def build_output_items(
    response: ResponsesResponse,
    *,
    id_factory: IdFactory = generate_id,
) -> list[dict[str, Any]]:
    """Build output items shared by streaming and non-streaming responses."""
    output: list[dict[str, Any]] = []
    validate_reasoning_signature(response.thinking_id, response.thinking_signature)
    if response.thinking_id or response.thinking:
        output.append(
            make_reasoning_item(
                response.thinking_id or id_factory("rs"),
                response.thinking,
                response.thinking_signature,
            )
        )
    if response.content or response.refusal:
        output.append(
            make_message_item(
                id_factory("msg"),
                response.content or None,
                refusal=response.refusal,
            )
        )
    output.extend(
        make_tool_call_item(tool_call, id_factory=id_factory)
        for tool_call in response.tool_calls or []
    )
    return output


def terminal_response_payload(
    outcome: TerminalOutcome,
    *,
    base_response: dict[str, Any],
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    wire_error: dict[str, Any] | None = None,
) -> Event:
    if outcome.transport is TransportTermination.UNEXPECTED_EOF:
        event_type = "response.incomplete"
        status = "incomplete"
        incomplete_details = {"reason": "unexpected_eof"}
        error = None
    elif outcome.response_status is ResponseStatus.COMPLETED:
        event_type = "response.completed"
        status = "completed"
        incomplete_details = None
        error = None
    elif outcome.response_status is ResponseStatus.INCOMPLETE:
        event_type = "response.incomplete"
        status = "incomplete"
        incomplete_details = outcome.incomplete_details
        error = None
    else:
        event_type = "response.failed"
        status = "cancelled" if outcome.response_status is ResponseStatus.CANCELLED else "failed"
        incomplete_details = outcome.incomplete_details
        error = (
            {"code": outcome.error.code, "message": outcome.error.message}
            if outcome.error is not None
            else None
        )
    if wire_error is not None:
        error = wire_error
    return {
        "type": event_type,
        "response": {
            **base_response,
            "status": status,
            "output": list(output),
            "usage": dict(usage) if usage is not None else None,
            "error": error,
            "incomplete_details": incomplete_details,
        },
    }


def build_nonstream_snapshot(
    response: ResponsesResponse,
    *,
    response_id: str,
    model: str,
    id_factory: IdFactory = generate_id,
) -> ResponsesSnapshot:
    """Build a non-stream response through the same item and terminal rules."""
    outcome = resolve_terminal_outcome(response.terminal_outcome, None)
    if outcome is None:
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.COMPLETED,
        )
    elif outcome.transport is not TransportTermination.EXPLICIT_TERMINAL:
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.FAILED,
            error=TerminalError(
                code="upstream_protocol_error",
                message="Provider returned an invalid non-stream terminal outcome",
            ),
        )
    base_response = {"id": response_id, "object": "response", "model": model}
    try:
        output = build_output_items(response, id_factory=id_factory)
    except ProviderError as error:
        if error.kind is not ProviderFailureKind.UPSTREAM_PROTOCOL:
            raise
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.FAILED,
            error=TerminalError(code="upstream_protocol_error", message=str(error)),
        )
        output = []
    terminal = terminal_response_payload(
        outcome,
        base_response=base_response,
        output=output,
        usage=make_usage(response.usage, streaming=False),
    )
    return ResponsesSnapshot(outcome=outcome, response=terminal["response"])


@dataclass(frozen=True, slots=True)
class ResponsesSnapshot:
    outcome: TerminalOutcome
    response: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReducerResult:
    events: tuple[Event, ...] = ()
    snapshot: ResponsesSnapshot | None = None


def _protocol_error(message: str) -> ProviderError:
    return ProviderError(
        message,
        status_code=502,
        retryable=True,
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
    )


def _event(data: Event) -> Event:
    """Mark one typed wire-event payload without encoding a transport frame."""
    return data


@dataclass
class _StreamMessageState:
    """Tracks the open message during Responses API streaming."""

    id_factory: IdFactory = generate_id
    output_items: list[dict[str, Any]] = field(default_factory=list)
    output_index: int = 0
    content_index: int = 0
    current_message_id: str | None = None
    current_source_content_index: int | None = None
    accumulated_content: str = ""
    accumulated_refusal: str = ""
    message_started: bool = False
    current_part_type: str | None = None
    message_content_parts: list[dict[str, Any]] = field(default_factory=list)
    # Reasoning summary state. Deltas remain buffered until the upstream
    # identity arrives or the item must close. Only then may we emit the
    # added/delta/done sequence under one stable id. A local id is generated
    # only at a boundary when the upstream never supplied one.
    # ``upstream_encrypted_content`` is the verifiable blob from
    # ``output_item.done.item.encrypted_content``; it is signed against the
    # upstream id, so the (id, blob) pair MUST round-trip together — pairing
    # the blob with a local id 400s the next turn with ``Encrypted content
    # could not be decrypted``.
    current_reasoning_id: str | None = None
    pending_reasoning_id: str | None = None
    upstream_encrypted_content: str | None = None
    reasoning_fragments: dict[int, list[str]] = field(default_factory=dict)
    reasoning_started: bool = False
    reasoning_events_started: bool = False
    reasoning_emission_offsets: dict[int, int] = field(default_factory=dict)
    dirty_reasoning_indices: list[int] = field(default_factory=list)
    _dirty_reasoning_index_set: set[int] = field(default_factory=set)
    flush_reasoning_scan_count: int = 0
    emitted_reasoning_parts: set[int] = field(default_factory=set)
    max_reasoning_summary_index: int | None = None
    visible_reasoning_summary_indices: dict[int, int] = field(default_factory=dict)
    reasoning_item_count: int = 0

    def begin_reasoning(self) -> None:
        """Open logical reasoning state without committing a wire identity."""
        if self.reasoning_started:
            return
        self.reasoning_started = True

    def bind_reasoning_id(self, reasoning_id: str) -> None:
        """Bind one upstream identity before any reasoning event is emitted."""
        self.begin_reasoning()
        if self.pending_reasoning_id is not None and self.pending_reasoning_id != reasoning_id:
            raise ProviderError(
                "Responses reasoning item changed identity",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if self.current_reasoning_id is not None and self.current_reasoning_id != reasoning_id:
            raise ProviderError(
                "Responses reasoning item changed identity",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        self.current_reasoning_id = reasoning_id
        self.pending_reasoning_id = None

    def latch_reasoning_id(self, reasoning_id: str) -> None:
        """Remember provenance identity without opening a visible reasoning item."""
        if self.current_reasoning_id is not None and self.current_reasoning_id != reasoning_id:
            raise ProviderError(
                "Responses reasoning item changed identity",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if self.pending_reasoning_id is not None and self.pending_reasoning_id != reasoning_id:
            raise ProviderError(
                "Responses reasoning item changed identity",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        self.pending_reasoning_id = reasoning_id

    def append_reasoning(self, delta: str, summary_index: int | None) -> None:
        """Accumulate one delta and defer it until the item identity is stable."""
        self.begin_reasoning()
        if self.pending_reasoning_id is not None:
            self.bind_reasoning_id(self.pending_reasoning_id)
        index = 0 if summary_index is None else summary_index
        self.bind_reasoning_summary_index(index)
        if index not in self.visible_reasoning_summary_indices:
            self.visible_reasoning_summary_indices[index] = len(
                self.visible_reasoning_summary_indices
            )
        self.reasoning_fragments.setdefault(index, []).append(delta)
        if index not in self._dirty_reasoning_index_set:
            self._dirty_reasoning_index_set.add(index)
            self.dirty_reasoning_indices.append(index)

    def bind_reasoning_summary_index(self, summary_index: int | None) -> None:
        """Validate provenance without opening an empty reasoning output item."""
        index = 0 if summary_index is None else summary_index
        maximum = self.max_reasoning_summary_index
        if maximum is None and index != 0:
            raise ProviderError(
                "Responses reasoning summary_index must start at zero",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if maximum is not None and index > maximum + 1:
            raise ProviderError(
                "Responses reasoning summary_index contains a gap",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if maximum is None or index > maximum:
            self.max_reasoning_summary_index = index

    def start_reasoning_events(self, *, allow_local_id: bool = False) -> list[Event]:
        """Emit added events and buffered deltas once one stable id is available."""
        if not self.reasoning_started or self.reasoning_events_started:
            return []
        if self.current_reasoning_id is None:
            if not allow_local_id:
                return []
            self.current_reasoning_id = self.id_factory("rs")

        rs_id = self.current_reasoning_id
        self.reasoning_events_started = True
        events = [
            _event(
                {
                    "type": "response.output_item.added",
                    "output_index": self.output_index,
                    "item": {"type": "reasoning", "id": rs_id, "summary": []},
                }
            ),
        ]
        dirty_indices = self.dirty_reasoning_indices
        self.dirty_reasoning_indices = []
        self._dirty_reasoning_index_set = set()
        for source_index in dirty_indices:
            deltas = self.reasoning_fragments[source_index]
            summary_index = self.visible_reasoning_summary_indices[source_index]
            events.append(
                _event(
                    {
                        "type": "response.reasoning_summary_part.added",
                        "item_id": rs_id,
                        "output_index": self.output_index,
                        "summary_index": summary_index,
                        "part": {"type": "summary_text", "text": ""},
                    }
                )
            )
            self.emitted_reasoning_parts.add(source_index)
            self.reasoning_emission_offsets[source_index] = len(deltas)
            events.extend(
                _event(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": rs_id,
                        "output_index": self.output_index,
                        "summary_index": summary_index,
                        "delta": delta,
                    }
                )
                for delta in deltas
            )
        return events

    def flush_reasoning_events(self) -> list[Event]:
        """Emit deltas received after the reasoning item acquired its id."""
        if not self.reasoning_events_started or self.current_reasoning_id is None:
            return []
        events: list[Event] = []
        dirty_indices = self.dirty_reasoning_indices
        self.dirty_reasoning_indices = []
        self._dirty_reasoning_index_set = set()
        for source_index in dirty_indices:
            self.flush_reasoning_scan_count += 1
            deltas = self.reasoning_fragments[source_index]
            summary_index = self.visible_reasoning_summary_indices[source_index]
            if source_index not in self.emitted_reasoning_parts:
                events.append(
                    _event(
                        {
                            "type": "response.reasoning_summary_part.added",
                            "item_id": self.current_reasoning_id,
                            "output_index": self.output_index,
                            "summary_index": summary_index,
                            "part": {"type": "summary_text", "text": ""},
                        }
                    )
                )
                self.emitted_reasoning_parts.add(source_index)
            offset = self.reasoning_emission_offsets.get(source_index, 0)
            events.extend(
                _event(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": self.current_reasoning_id,
                        "output_index": self.output_index,
                        "summary_index": summary_index,
                        "delta": delta,
                    }
                )
                for delta in deltas[offset:]
            )
            self.reasoning_emission_offsets[source_index] = len(deltas)
        return events

    def start_message_part(self, part_type: str) -> list[Event]:
        """Open one typed Responses content part, closing a different active part."""
        if part_type not in {"output_text", "refusal"}:
            raise ValueError(f"unsupported Responses content part type: {part_type}")

        events: list[Event] = []
        if not self.message_started:
            self.current_message_id = self.id_factory("msg")
            self.message_started = True
            events.append(
                _event(
                    {
                        "type": "response.output_item.added",
                        "output_index": self.output_index,
                        "item": {
                            "type": "message",
                            "id": self.current_message_id,
                            "role": "assistant",
                            "content": [],
                            "status": "in_progress",
                        },
                    }
                )
            )

        if self.current_part_type == part_type:
            return events
        events.extend(self.close_open_content_part())
        self.current_part_type = part_type
        empty_part = (
            make_text_content("") if part_type == "output_text" else make_refusal_content("")
        )
        events.append(
            _event(
                {
                    "type": "response.content_part.added",
                    "item_id": self.current_message_id,
                    "output_index": self.output_index,
                    "content_index": self.content_index,
                    "part": empty_part,
                }
            )
        )
        return events

    def bind_source_content_index(self, source_index: int | None) -> list[Event]:
        """Honor native content-part boundaries within one indexed message."""
        if source_index is None:
            return []
        if self.current_source_content_index is None:
            if source_index != 0:
                raise ProviderError(
                    "Responses message content_index must start at zero",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                )
            self.current_source_content_index = source_index
            return []
        if source_index < self.current_source_content_index:
            raise ProviderError(
                "Responses message content_index moved backwards",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if source_index == self.current_source_content_index:
            return []
        if source_index != self.current_source_content_index + 1:
            raise ProviderError(
                "Responses message content_index contains a gap",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        events = self.close_open_content_part()
        self.current_source_content_index = source_index
        return events

    def bind_empty_source_content_index(self, source_index: int | None) -> None:
        """Validate provenance without opening an empty message output item."""
        if source_index is None:
            return
        current = self.current_source_content_index
        if current is None and source_index != 0:
            raise ProviderError(
                "Responses message content_index must start at zero",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        if current is not None:
            if source_index < current:
                raise ProviderError(
                    "Responses message content_index moved backwards",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                )
            if source_index > current + 1:
                raise ProviderError(
                    "Responses message content_index contains a gap",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                )
        self.current_source_content_index = source_index

    def close_open_content_part(self) -> list[Event]:
        """Close the active output_text or refusal part without closing its message."""
        if self.current_part_type is None or self.current_message_id is None:
            return []

        if self.current_part_type == "output_text":
            value = self.accumulated_content
            part = make_text_content(value)
            done = {
                "type": "response.output_text.done",
                "item_id": self.current_message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": value,
            }
            self.accumulated_content = ""
        else:
            value = self.accumulated_refusal
            part = make_refusal_content(value)
            done = {
                "type": "response.refusal.done",
                "item_id": self.current_message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "refusal": value,
            }
            self.accumulated_refusal = ""

        events = [
            _event(done),
            _event(
                {
                    "type": "response.content_part.done",
                    "item_id": self.current_message_id,
                    "output_index": self.output_index,
                    "content_index": self.content_index,
                    "part": part,
                }
            ),
        ]
        self.message_content_parts.append(part)
        self.content_index += 1
        self.current_part_type = None
        return events

    def close_open_message(self, advance_index: bool = True) -> list[Event]:
        """Close the currently open message.

        Returns SSE events to yield. Does nothing if no message is open.
        """
        if not self.message_started or not self.current_message_id:
            return []
        events = self.close_open_content_part()
        item = make_message_item_from_parts(
            self.current_message_id,
            list(self.message_content_parts),
        )
        events.append(
            _event(
                {
                    "type": "response.output_item.done",
                    "output_index": self.output_index,
                    "item": item,
                }
            )
        )
        self.output_items.append(item)
        if advance_index:
            self.output_index += 1
        self.message_started = False
        self.current_message_id = None
        self.accumulated_content = ""
        self.accumulated_refusal = ""
        self.content_index = 0
        self.current_part_type = None
        self.message_content_parts = []
        self.current_source_content_index = None
        return events

    def close_open_reasoning(self, advance_index: bool = True) -> list[Event]:
        """Close any open reasoning summary block + reasoning item.

        Emits the OpenAI Responses-API event sequence:
          response.reasoning_summary_text.done
          response.reasoning_summary_part.done
          response.output_item.done (item.type == "reasoning")
        """
        if not self.reasoning_started:
            return []
        events = self.start_reasoning_events(allow_local_id=True)
        if self.current_reasoning_id is None:
            raise RuntimeError("reasoning id was not established")
        rs_id = self.current_reasoning_id
        indexed_summary_parts = [
            (
                self.visible_reasoning_summary_indices[source_index],
                {"type": "summary_text", "text": text},
            )
            for source_index, fragments in sorted(self.reasoning_fragments.items())
            for text in ["".join(fragments)]
        ]
        reasoning_item = make_reasoning_item(
            rs_id,
            [part for _, part in indexed_summary_parts],
            self.upstream_encrypted_content,
        )
        for summary_index, summary_part in indexed_summary_parts:
            events.extend(
                [
                    _event(
                        {
                            "type": "response.reasoning_summary_text.done",
                            "item_id": rs_id,
                            "output_index": self.output_index,
                            "summary_index": summary_index,
                            "text": summary_part["text"],
                        }
                    ),
                    _event(
                        {
                            "type": "response.reasoning_summary_part.done",
                            "item_id": rs_id,
                            "output_index": self.output_index,
                            "summary_index": summary_index,
                            "part": summary_part,
                        }
                    ),
                ]
            )
        events.append(
            _event(
                {
                    "type": "response.output_item.done",
                    "output_index": self.output_index,
                    "item": reasoning_item,
                }
            )
        )
        self.output_items.append(reasoning_item)
        self.reasoning_item_count += 1
        if advance_index:
            self.output_index += 1
        self.reasoning_started = False
        self.reasoning_events_started = False
        self.current_reasoning_id = None
        self.pending_reasoning_id = None
        self.upstream_encrypted_content = None
        self.reasoning_fragments = {}
        self.reasoning_emission_offsets = {}
        self.dirty_reasoning_indices = []
        self._dirty_reasoning_index_set = set()
        self.flush_reasoning_scan_count = 0
        self.emitted_reasoning_parts = set()
        self.max_reasoning_summary_index = None
        self.visible_reasoning_summary_indices = {}
        return events

    def finish_empty_reasoning_item(self) -> None:
        """Clear marker-only provenance at its indexed item boundary."""
        if self.reasoning_started:
            return
        self.pending_reasoning_id = None
        self.max_reasoning_summary_index = None

    def finish_empty_message_item(self) -> None:
        """Clear marker-only content provenance at its indexed item boundary."""
        if self.message_started:
            return
        self.current_source_content_index = None


@dataclass
class _IndexedOutputBucket:
    """Canonical chunks belonging to one upstream Responses output item."""

    item_type: str
    chunks: list[ResponsesStreamChunk] = field(default_factory=list)
    done: bool = False
    buffered_chunk_count: int = 0
    buffered_payload_bytes: int = 0


@dataclass
class _IndexedOutputScheduler:
    """Release interleaved native output items in source ``output_index`` order."""

    buckets: dict[int, _IndexedOutputBucket] = field(default_factory=dict)
    next_output_index: int = 0
    saw_indexed_item: bool = False
    max_future_buckets: int = 64
    max_future_chunks: int = 4096
    max_future_payload_bytes: int = 8 * 1024 * 1024
    future_chunk_count: int = 0
    future_payload_bytes: int = 0

    @staticmethod
    def _protocol_error(message: str) -> ProviderError:
        return ProviderError(
            message,
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        )

    def add(self, chunk: ResponsesStreamChunk) -> list[ResponsesStreamChunk]:
        """Release the active item immediately and buffer only future items."""
        output_index = chunk.output_index
        if output_index is None:
            return [chunk]
        self.saw_indexed_item = True
        if output_index < 0:
            raise self._protocol_error("Responses output_index must be non-negative")
        if output_index < self.next_output_index:
            raise self._protocol_error("Responses output item changed after completion")
        if output_index - self.next_output_index > self.max_future_buckets:
            raise self._protocol_error("Responses future output index window limit exceeded")
        item_type = chunk.output_item_type
        if not item_type:
            raise self._protocol_error("Responses indexed chunk is missing its output item type")

        bucket = self.buckets.get(output_index)
        if bucket is None:
            bucket = _IndexedOutputBucket(item_type=item_type)
            self.buckets[output_index] = bucket
        elif bucket.item_type != item_type:
            raise self._protocol_error("Responses output index changed item type")
        if bucket.done:
            raise self._protocol_error("Responses output item received data after completion")

        bucket.chunks.append(chunk)
        bucket.done = chunk.output_item_done
        if output_index > self.next_output_index:
            payload_size = self._chunk_payload_size(chunk)
            bucket.buffered_chunk_count += 1
            bucket.buffered_payload_bytes += payload_size
            self.future_chunk_count += 1
            self.future_payload_bytes += payload_size
            if self._future_bucket_count() > self.max_future_buckets:
                raise self._protocol_error("Responses future output item buffer limit exceeded")
            if self.future_chunk_count > self.max_future_chunks:
                raise self._protocol_error("Responses future output chunk limit exceeded")
            if self.future_payload_bytes > self.max_future_payload_bytes:
                raise self._protocol_error("Responses future output payload limit exceeded")
        return self._drain_available()

    def finalize(self) -> list[ResponsesStreamChunk]:
        """Close and drain every buffered item at explicit terminal or EOF."""
        if not self.buckets:
            return []
        highest_index = max(self.buckets)
        missing = [
            index
            for index in range(self.next_output_index, highest_index + 1)
            if index not in self.buckets
        ]
        if missing:
            raise self._protocol_error(f"Responses output item sequence has gaps: {missing}")
        for bucket in self.buckets.values():
            if not bucket.done:
                if bucket.chunks:
                    bucket.chunks[-1] = replace(bucket.chunks[-1], output_item_done=True)
                else:
                    bucket.chunks.append(
                        ResponsesStreamChunk(
                            content="",
                            output_index=self.next_output_index,
                            output_item_type=bucket.item_type,
                            output_item_done=True,
                        )
                    )
                bucket.done = True
        return self._drain_available()

    def finalize_contiguous(self) -> list[ResponsesStreamChunk]:
        """Drain the safe indexed prefix and discard future buckets beyond a gap."""
        ready: list[ResponsesStreamChunk] = []
        while (bucket := self.buckets.get(self.next_output_index)) is not None:
            if not bucket.done:
                if bucket.chunks:
                    bucket.chunks[-1] = replace(bucket.chunks[-1], output_item_done=True)
                else:
                    bucket.chunks.append(
                        ResponsesStreamChunk(
                            content="",
                            output_index=self.next_output_index,
                            output_item_type=bucket.item_type,
                            output_item_done=True,
                        )
                    )
                bucket.done = True
            ready.extend(self._drain_available())
        self.discard_pending()
        return ready

    def discard_pending(self) -> None:
        """Drop buffered indexed suffix state and reset its accounting."""
        self.buckets.clear()
        self.future_chunk_count = 0
        self.future_payload_bytes = 0

    def _drain_available(self) -> list[ResponsesStreamChunk]:
        ready: list[ResponsesStreamChunk] = []
        while True:
            bucket = self.buckets.get(self.next_output_index)
            if bucket is None:
                break
            ready.extend(bucket.chunks)
            self.future_chunk_count -= bucket.buffered_chunk_count
            bucket.buffered_chunk_count = 0
            bucket.chunks = []
            self.future_payload_bytes -= bucket.buffered_payload_bytes
            bucket.buffered_payload_bytes = 0
            if not bucket.done:
                break
            del self.buckets[self.next_output_index]
            self.next_output_index += 1
        return ready

    def _future_bucket_count(self) -> int:
        return sum(index > self.next_output_index for index in self.buckets)

    @staticmethod
    def _chunk_payload_size(chunk: ResponsesStreamChunk) -> int:
        """Approximate retained payload bytes without serializing arbitrary objects."""
        size = len(chunk.content.encode("utf-8"))
        for value in (chunk.refusal, chunk.thinking, chunk.thinking_id, chunk.thinking_signature):
            if value:
                size += len(value.encode("utf-8"))
        if chunk.tool_call is not None:
            for value in (
                chunk.tool_call.call_id,
                chunk.tool_call.name,
                chunk.tool_call.arguments,
                chunk.tool_call.namespace,
            ):
                if value:
                    size += len(value.encode("utf-8"))
        if chunk.usage is not None:
            size += len(json.dumps(chunk.usage, separators=(",", ":"), default=str).encode("utf-8"))
        return size


@dataclass
class ResponsesReducer:
    """Reduce canonical provider chunks into typed Responses event payloads."""

    base_response: dict[str, Any]
    id_factory: IdFactory = generate_id
    state: _StreamMessageState = field(init=False)
    indexed_outputs: _IndexedOutputScheduler = field(default_factory=_IndexedOutputScheduler)
    final_usage: dict[str, Any] | None = None
    snapshot: ResponsesSnapshot | None = None
    saw_unindexed_output: bool = False
    deferred_chunks: list[ResponsesStreamChunk] = field(default_factory=list)
    _started: bool = False

    def __post_init__(self) -> None:
        self.base_response = dict(self.base_response)
        self.state = _StreamMessageState(id_factory=self.id_factory)

    def bind_model(self, model: str) -> None:
        """Bind the router-selected model before lifecycle events are emitted."""
        if self._started:
            raise RuntimeError("Responses reducer model cannot change after start")
        self.base_response["model"] = model

    def start(self) -> ReducerResult:
        """Emit the two initial lifecycle events exactly once."""
        if self._started:
            return ReducerResult()
        self._started = True
        response = {
            **self.base_response,
            "status": "in_progress",
            "output": [],
        }
        return ReducerResult(
            events=(
                {"type": "response.created", "response": response},
                {"type": "response.in_progress", "response": dict(response)},
            )
        )

    def feed(
        self,
        chunk: ResponsesStreamChunk,
        *,
        _source: bool = True,
    ) -> ReducerResult:
        """Consume one provider chunk and return every newly visible event."""
        if self.snapshot is not None:
            raise RuntimeError("Responses reducer received data after terminal state")
        events: list[Event] = []
        queue: deque[tuple[ResponsesStreamChunk, bool]] = deque([(chunk, _source)])
        while queue and self.snapshot is None:
            current, is_source = queue.popleft()
            if is_source and current.output_index is not None:
                if self.saw_unindexed_output:
                    raise _protocol_error(
                        "Responses stream mixed indexed and unindexed output items"
                    )
                terminal = self._is_terminal(current)
                control = None
                indexed = current
                if terminal:
                    control = replace(
                        current,
                        content="",
                        refusal=None,
                        tool_call=None,
                        thinking=None,
                        thinking_id=None,
                        thinking_signature=None,
                        output_index=None,
                        content_index=None,
                        reasoning_summary_index=None,
                        provenance_only=False,
                        output_item_type=None,
                        output_item_done=False,
                    )
                    indexed = replace(
                        current,
                        usage=None,
                        finish_reason=None,
                        terminal_outcome=None,
                    )
                ready = self.indexed_outputs.add(indexed)
                if control is not None:
                    ready.extend(self.indexed_outputs.finalize())
                queue.extend((pending, False) for pending in ready)
                if control is not None:
                    queue.append((control, False))
                continue

            has_output = self._has_output_payload(current)
            if is_source and current.output_index is None and has_output:
                if self.indexed_outputs.saw_indexed_item:
                    raise _protocol_error(
                        "Responses stream mixed indexed and unindexed output items"
                    )
                self.saw_unindexed_output = True
            if is_source and self.indexed_outputs.saw_indexed_item and self._is_terminal(current):
                ready = self.indexed_outputs.finalize()
                if ready:
                    queue.extend((pending, False) for pending in ready)
                    queue.append((current, False))
                    continue

            events.extend(self._reduce_ready_chunk(current, queue))
        return ReducerResult(events=tuple(events), snapshot=self.snapshot)

    def finish(self, outcome: TerminalOutcome | None = None) -> ReducerResult:
        """Drain indexed/deferred state and synthesize EOF when no terminal arrived."""
        if self.snapshot is not None:
            return ReducerResult(snapshot=self.snapshot)
        events: list[Event] = []
        ready = self.indexed_outputs.finalize()
        for chunk in ready:
            step = self.feed(chunk, _source=False)
            events.extend(step.events)
            if step.snapshot is not None:
                return ReducerResult(tuple(events), step.snapshot)
        if self.state.reasoning_started and self.deferred_chunks:
            events.extend(self.state.close_open_reasoning())
            deferred = self.deferred_chunks
            self.deferred_chunks = []
            for chunk in deferred:
                step = self.feed(chunk, _source=False)
                events.extend(step.events)
                if step.snapshot is not None:
                    return ReducerResult(tuple(events), step.snapshot)
        outcome = outcome or unexpected_eof_outcome()
        events.extend(self._close_and_terminal(outcome))
        return ReducerResult(tuple(events), self.snapshot)

    def terminate(
        self,
        outcome: TerminalOutcome,
        *,
        wire_error: dict[str, Any] | None = None,
    ) -> ReducerResult:
        """Close partial state and emit one typed abnormal terminal event."""
        if self.snapshot is not None:
            return ReducerResult(snapshot=self.snapshot)
        events: list[Event] = []
        try:
            events.extend(self._drain_indexed_for_termination())
        except Exception:
            self.indexed_outputs.discard_pending()
        try:
            events.extend(self._drain_deferred_for_termination())
        except Exception:
            self.deferred_chunks = []
        try:
            events.extend(self._close_and_terminal(outcome, wire_error=wire_error))
        except Exception:
            terminal = terminal_response_payload(
                outcome,
                base_response=self.base_response,
                output=self.state.output_items,
                usage=make_usage(self.final_usage),
                wire_error=wire_error,
            )
            self._set_snapshot(outcome, terminal)
            events.append(terminal)
        return ReducerResult(tuple(events), self.snapshot)

    def _drain_indexed_for_termination(self) -> list[Event]:
        """Materialize the safe indexed prefix without accepting its terminal state."""
        events: list[Event] = []
        try:
            ready = self.indexed_outputs.finalize_contiguous()
        except Exception:
            self.indexed_outputs.discard_pending()
            return events
        for chunk in ready:
            payload = replace(chunk, finish_reason=None, terminal_outcome=None)
            replayed = self._try_termination_payload(payload)
            if replayed is None:
                break
            events.extend(replayed)
        return events

    def _drain_deferred_for_termination(self) -> list[Event]:
        """Materialize accepted deferred payload without accepting its terminal state."""
        events = self.state.close_open_reasoning()
        while self.deferred_chunks:
            deferred = self.deferred_chunks
            self.deferred_chunks = []
            for chunk in deferred:
                payload = replace(chunk, finish_reason=None, terminal_outcome=None)
                replayed = self._try_termination_payload(payload)
                if replayed is None:
                    self.deferred_chunks = []
                    return events
                events.extend(replayed)
            if self.state.reasoning_started:
                events.extend(self.state.close_open_reasoning())
        return events

    def _try_termination_payload(self, payload: ResponsesStreamChunk) -> tuple[Event, ...] | None:
        """Replay one pending payload transactionally; return None when it is unsafe."""
        state_before = copy.deepcopy(self.state)
        usage_before = copy.deepcopy(self.final_usage)
        deferred_before = list(self.deferred_chunks)
        try:
            step = self.feed(payload, _source=False)
            if step.snapshot is not None:
                raise RuntimeError("Termination payload produced a terminal snapshot")
        except Exception:
            self.state = state_before
            self.final_usage = usage_before
            self.deferred_chunks = deferred_before
            return None
        return step.events

    @staticmethod
    def _is_terminal(chunk: ResponsesStreamChunk) -> bool:
        return chunk.terminal_outcome is not None or chunk.finish_reason is not None

    @staticmethod
    def _has_output_payload(chunk: ResponsesStreamChunk) -> bool:
        return bool(
            chunk.content
            or chunk.refusal
            or chunk.tool_call
            or chunk.thinking
            or chunk.thinking_id
            or chunk.thinking_signature
            or chunk.output_item_type
            or chunk.output_item_done
        )

    def _reduce_ready_chunk(
        self,
        chunk: ResponsesStreamChunk,
        queue: deque[tuple[ResponsesStreamChunk, bool]],
    ) -> list[Event]:
        events: list[Event] = []
        chunk_is_terminal = self._is_terminal(chunk)
        if chunk.provenance_only:
            if chunk.output_item_type == "reasoning":
                self.state.bind_reasoning_summary_index(chunk.reasoning_summary_index)
                if chunk.thinking_id:
                    self.state.latch_reasoning_id(chunk.thinking_id)
            elif chunk.output_item_type == "message":
                self.state.bind_empty_source_content_index(chunk.content_index)
            if chunk.output_item_done:
                if chunk.output_item_type == "reasoning":
                    events.extend(self.state.close_open_reasoning())
                    self.state.finish_empty_reasoning_item()
                elif chunk.output_item_type == "message":
                    events.extend(self.state.close_open_message())
                    self.state.finish_empty_message_item()
            if not chunk_is_terminal and chunk.usage is None:
                return events
            chunk = replace(
                chunk,
                thinking_id=None,
                output_index=None,
                content_index=None,
                reasoning_summary_index=None,
                provenance_only=False,
                output_item_type=None,
                output_item_done=False,
            )

        chunk_has_reasoning = bool(chunk.thinking or chunk.thinking_id or chunk.thinking_signature)
        has_non_reasoning = bool(
            chunk.content or chunk.refusal or chunk.tool_call or chunk.usage or chunk_is_terminal
        )
        if (self.state.reasoning_started or chunk_has_reasoning) and has_non_reasoning:
            self.deferred_chunks.append(
                replace(
                    chunk,
                    thinking=None,
                    thinking_id=None,
                    thinking_signature=None,
                )
            )
            chunk = replace(
                chunk,
                content="",
                refusal=None,
                tool_call=None,
                usage=None,
                finish_reason=None,
                terminal_outcome=None,
            )

        if chunk.thinking:
            if self.state.message_started:
                events.extend(self.state.close_open_message())
            self.state.append_reasoning(chunk.thinking, chunk.reasoning_summary_index)
        if chunk.thinking_id:
            if self.state.message_started and not self.state.reasoning_started:
                events.extend(self.state.close_open_message())
            self.state.bind_reasoning_id(chunk.thinking_id)
        events.extend(self.state.start_reasoning_events())
        events.extend(self.state.flush_reasoning_events())

        if chunk.thinking_signature:
            validate_reasoning_signature(
                self.state.current_reasoning_id,
                chunk.thinking_signature,
            )
            self.state.upstream_encrypted_content = chunk.thinking_signature
        indexed_reasoning_done = chunk.output_item_done and chunk.output_item_type == "reasoning"
        if chunk.thinking_signature or indexed_reasoning_done:
            events.extend(self.state.close_open_reasoning())
            if indexed_reasoning_done:
                self.state.finish_empty_reasoning_item()
        if chunk_is_terminal and self.deferred_chunks and self.state.reasoning_started:
            events.extend(self.state.close_open_reasoning())

        if chunk.content:
            events.extend(self.state.close_open_reasoning())
            events.extend(self.state.bind_source_content_index(chunk.content_index))
            events.extend(self.state.start_message_part("output_text"))
            self.state.accumulated_content += chunk.content
            events.append(
                {
                    "type": "response.output_text.delta",
                    "item_id": self.state.current_message_id,
                    "output_index": self.state.output_index,
                    "content_index": self.state.content_index,
                    "delta": chunk.content,
                }
            )
        if chunk.refusal:
            events.extend(self.state.close_open_reasoning())
            events.extend(self.state.bind_source_content_index(chunk.content_index))
            events.extend(self.state.start_message_part("refusal"))
            self.state.accumulated_refusal += chunk.refusal
            events.append(
                {
                    "type": "response.refusal.delta",
                    "item_id": self.state.current_message_id,
                    "output_index": self.state.output_index,
                    "content_index": self.state.content_index,
                    "delta": chunk.refusal,
                }
            )
        if chunk.tool_call:
            events.extend(self.state.close_open_reasoning())
            events.extend(self.state.close_open_message())
            events.extend(self._tool_events(chunk.tool_call))
        if chunk.output_item_done and chunk.output_item_type == "message":
            events.extend(self.state.close_open_message())
            self.state.finish_empty_message_item()
        if chunk.usage:
            self.final_usage = chunk.usage

        chunk_outcome = resolve_terminal_outcome(chunk.terminal_outcome, chunk.finish_reason)
        if chunk_outcome is not None:
            events.extend(self._close_and_terminal(chunk_outcome))
            return events
        if self.deferred_chunks and not self.state.reasoning_started:
            deferred = self.deferred_chunks
            self.deferred_chunks = []
            queue.extend((pending, False) for pending in deferred)
        return events

    def _tool_events(self, tool_call: ResponsesToolCall) -> list[Event]:
        index = self.state.output_index
        item = make_tool_call_item(tool_call, id_factory=self.id_factory)
        if tool_call.kind == "custom":
            item_id = item["id"]
            events = [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "input": "", "status": "in_progress"},
                },
                {
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "delta": tool_call.arguments,
                },
                {
                    "type": "response.custom_tool_call_input.done",
                    "item_id": item_id,
                    "output_index": index,
                    "input": tool_call.arguments,
                },
                {"type": "response.output_item.done", "output_index": index, "item": item},
            ]
        elif tool_call.kind == "tool_search":
            events = [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "status": "in_progress", "arguments": {}},
                },
                {"type": "response.output_item.done", "output_index": index, "item": item},
            ]
        else:
            item_id = item["id"]
            events = [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": make_function_call_item(
                        item_id,
                        tool_call.call_id,
                        tool_call.name,
                        "",
                        "in_progress",
                        namespace=tool_call.namespace,
                    ),
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "delta": tool_call.arguments,
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item_id,
                    "output_index": index,
                    "arguments": tool_call.arguments,
                },
                {"type": "response.output_item.done", "output_index": index, "item": item},
            ]
        self.state.output_items.append(item)
        self.state.output_index += 1
        return events

    def _close_and_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        wire_error: dict[str, Any] | None = None,
    ) -> list[Event]:
        events = self.state.close_open_reasoning()
        events.extend(self.state.close_open_message(advance_index=False))
        terminal = terminal_response_payload(
            outcome,
            base_response=self.base_response,
            output=self.state.output_items,
            usage=make_usage(self.final_usage),
            wire_error=wire_error,
        )
        self._set_snapshot(outcome, terminal)
        events.append(terminal)
        return events

    def _set_snapshot(self, outcome: TerminalOutcome, terminal: Event) -> None:
        response = {
            **terminal["response"],
            "output": list(terminal["response"]["output"]),
        }
        self.snapshot = ResponsesSnapshot(outcome=outcome, response=response)
