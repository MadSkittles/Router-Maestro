"""Tests for Copilot /responses streaming event handlers.

These tests exercise the `responses_completion_stream` parser against
synthetic SSE payloads produced via `httpx.MockTransport`. The goal is to
pin down handling of event types that have caused regressions in the field
(custom_tool_call from v0.3.4, tool_search_call from v0.3.5).
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from router_maestro.providers import CopilotProvider
from router_maestro.providers.base import (
    ProviderError,
    ProviderFailureKind,
    ResponsesRequest,
    ResponsesStreamChunk,
    ResponseStatus,
    TerminalError,
    TransportTermination,
)
from router_maestro.server.routes.responses import stream_response


def _sse_lines(events: list[dict]) -> bytes:
    """Encode a list of events as SSE `data: …\\n\\n` lines."""
    parts = []
    for evt in events:
        parts.append(f"data: {json.dumps(evt)}\n\n")
    return "".join(parts).encode("utf-8")


def _make_provider_with_stream(body: bytes) -> CopilotProvider:
    """Build a CopilotProvider whose HTTP client returns ``body`` on POST.

    Bypasses ``ensure_token`` and the GitHub OAuth flow by stubbing
    ``_get_headers`` and the token-refresh hook.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    provider = CopilotProvider()
    provider._client = httpx.AsyncClient(transport=transport)
    # Skip token refresh entirely — tests run offline.
    provider.ensure_token = _noop  # type: ignore[method-assign]
    provider._get_headers = lambda *args, **kwargs: {"authorization": "Bearer test"}  # type: ignore[method-assign]
    return provider


async def _noop() -> None:
    return None


async def _collect(provider: CopilotProvider, model: str = "gpt-5.5") -> list:
    chunks: list[ResponsesStreamChunk] = []
    async for chunk in provider.responses_completion_stream(
        ResponsesRequest(model=model, input="hi", stream=True)
    ):
        chunks.append(chunk)
    return chunks


def _parse_sse(events: list[str]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                parsed.append(json.loads(line.removeprefix("data: ")))
    return parsed


async def _route_provider_chunks(provider: CopilotProvider) -> list[dict[str, Any]]:
    class ProviderStreamRouter:
        async def responses_completion_stream(
            self,
            request: ResponsesRequest,
            fallback: bool = True,
        ) -> tuple[AsyncIterator[ResponsesStreamChunk], str]:
            return provider.responses_completion_stream(request), "github-copilot"

    raw_events = [
        event
        async for event in stream_response(
            ProviderStreamRouter(),  # type: ignore[arg-type]
            ResponsesRequest(model="gpt-5.5", input="hi", stream=True),
            request_id="req-provider-route",
            start_time=0.0,
        )
    ]
    return _parse_sse(raw_events)


class TestTerminalInvariant:
    """Transport closure must not synthesize a semantic Responses terminal."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            _sse_lines(
                [
                    {
                        "type": "response.output_text.delta",
                        "item_id": "msg-1",
                        "delta": "partial",
                    }
                ]
            ),
            _sse_lines(
                [
                    {
                        "type": "response.output_text.delta",
                        "item_id": "msg-1",
                        "delta": "partial",
                    }
                ]
            )
            + b"data: [DONE]\n\n",
        ],
        ids=["raw-eof", "done-sentinel"],
    )
    async def test_eof_without_response_terminal_does_not_emit_finish(self, body: bytes):
        provider = _make_provider_with_stream(body)

        chunks = await _collect(provider)

        assert "".join(chunk.content for chunk in chunks) == "partial"
        assert not any(chunk.finish_reason for chunk in chunks)


class TestOutputTextSnapshotRecovery:
    @staticmethod
    def _text_event(event_type: str, text: str, output_index: int, content_index: int):
        field = "delta" if event_type.endswith(".delta") else "text"
        return {
            "type": event_type,
            "item_id": f"msg-{output_index}",
            "output_index": output_index,
            "content_index": content_index,
            field: text,
        }

    @pytest.mark.asyncio
    async def test_done_only_snapshot_emits_complete_text(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    self._text_event("response.output_text.done", "hello", 0, 0),
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        chunks = await _collect(provider)

        assert "".join(chunk.content for chunk in chunks) == "hello"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("delta", "snapshot", "expected"),
        [
            pytest.param("hello", "hello", "hello", id="matching-full-delta"),
            pytest.param("he", "hello", "hello", id="partial-delta-suffix"),
        ],
    )
    async def test_done_snapshot_completes_without_duplicate(self, delta, snapshot, expected):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    self._text_event("response.output_text.delta", delta, 0, 0),
                    self._text_event("response.output_text.done", snapshot, 0, 0),
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        chunks = await _collect(provider)

        assert "".join(chunk.content for chunk in chunks) == expected

    @pytest.mark.asyncio
    async def test_multiple_parts_have_independent_snapshots_and_duplicate_done_is_idempotent(self):
        first_done = self._text_event("response.output_text.done", "one", 0, 0)
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    self._text_event("response.output_text.delta", "o", 0, 0),
                    first_done,
                    first_done,
                    self._text_event("response.output_text.done", "two", 0, 1),
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        chunks = await _collect(provider)

        assert [chunk.content for chunk in chunks if chunk.content] == ["o", "ne", "two"]

    @pytest.mark.asyncio
    async def test_inconsistent_done_snapshot_is_typed_protocol_failure(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    self._text_event("response.output_text.delta", "hello", 0, 0),
                    self._text_event("response.output_text.done", "goodbye", 0, 0),
                ]
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502
        assert exc_info.value.status_code == 502
        assert exc_info.value.provider == "github-copilot"
        assert exc_info.value.model == "gpt-5.5"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event",
        [
            pytest.param(
                {
                    "type": "response.output_text.done",
                    "text": 7,
                    "output_index": 0,
                    "content_index": 0,
                },
                id="done-text-non-string",
            ),
            pytest.param(
                {
                    "type": "response.output_text.done",
                    "text": "hello",
                    "output_index": True,
                    "content_index": 0,
                },
                id="done-output-index-bool",
            ),
            pytest.param(
                {
                    "type": "response.output_text.delta",
                    "delta": "hello",
                    "output_index": 0,
                    "content_index": "0",
                },
                id="delta-content-index-non-integer",
            ),
        ],
    )
    async def test_text_part_fields_are_typed_protocol_contracts(self, event):
        provider = _make_provider_with_stream(_sse_lines([event]))

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL


class TestRefusalSnapshotRecovery:
    @staticmethod
    def _refusal_event(
        event_type: str, refusal: str, output_index: int, content_index: int
    ) -> dict:
        field = "delta" if event_type.endswith(".delta") else "refusal"
        return {
            "type": event_type,
            "item_id": f"msg-{output_index}",
            "output_index": output_index,
            "content_index": content_index,
            field: refusal,
        }

    @pytest.mark.asyncio
    async def test_done_snapshot_completes_partial_delta_without_duplicate(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    self._refusal_event("response.refusal.delta", "I can", 0, 0),
                    self._refusal_event("response.refusal.done", "I cannot", 0, 0),
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        chunks = await _collect(provider)

        assert [chunk.refusal for chunk in chunks if chunk.refusal] == ["I can", "not"]

    @pytest.mark.asyncio
    async def test_parts_have_independent_done_snapshots_and_duplicates_are_idempotent(self):
        first_done = self._refusal_event("response.refusal.done", "one", 0, 0)
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    first_done,
                    first_done,
                    self._refusal_event("response.refusal.done", "two", 0, 1),
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        chunks = await _collect(provider)

        assert [chunk.refusal for chunk in chunks if chunk.refusal] == ["one", "two"]

    @pytest.mark.asyncio
    async def test_conflicting_done_snapshot_is_typed_protocol_failure(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    self._refusal_event("response.refusal.done", "safe", 0, 0),
                    self._refusal_event("response.refusal.done", "unsafe", 0, 0),
                ]
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL


class TestNativeTerminalStatus:
    """Every Responses terminal envelope preserves its inner response semantics."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outer_type", "inner", "expected_status"),
        [
            ("response.done", {"status": "completed"}, ResponseStatus.COMPLETED),
            (
                "response.completed",
                {
                    "status": "incomplete",
                    "incomplete_details": {
                        "reason": "max_output_tokens",
                        "vendor": {"limit": 12},
                    },
                },
                ResponseStatus.INCOMPLETE,
            ),
            (
                "response.incomplete",
                {
                    "status": "incomplete",
                    "incomplete_details": {
                        "reason": "content_filter",
                        "vendor": "safety",
                    },
                },
                ResponseStatus.INCOMPLETE,
            ),
            (
                "response.incomplete",
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "vendor_limit", "raw": [1, 2]},
                },
                ResponseStatus.INCOMPLETE,
            ),
            (
                "response.failed",
                {
                    "status": "failed",
                    "error": {"code": "upstream_failed", "message": "boom"},
                },
                ResponseStatus.FAILED,
            ),
            (
                "response.failed",
                {"status": "cancelled", "error": None},
                ResponseStatus.CANCELLED,
            ),
        ],
        ids=[
            "done-completed",
            "outer-completed-inner-incomplete",
            "incomplete-content-filter",
            "incomplete-unknown-details",
            "failed",
            "cancelled",
        ],
    )
    async def test_terminal_envelopes_emit_one_canonical_chunk(
        self, outer_type: str, inner: dict, expected_status: ResponseStatus
    ):
        usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
        response = {**inner, "usage": usage}
        provider = _make_provider_with_stream(
            _sse_lines([{"type": outer_type, "response": response}])
        )

        chunks = await _collect(provider)

        assert len(chunks) == 1
        terminal = chunks[0]
        assert terminal.finish_reason is None
        assert terminal.usage == usage
        assert terminal.terminal_outcome is not None
        assert terminal.terminal_outcome.transport is TransportTermination.EXPLICIT_TERMINAL
        assert terminal.terminal_outcome.response_status is expected_status
        assert terminal.terminal_outcome.incomplete_details == inner.get("incomplete_details")
        raw_error = inner.get("error")
        if raw_error is None:
            assert terminal.terminal_outcome.error is None
        else:
            assert terminal.terminal_outcome.error == TerminalError(
                code=raw_error["code"], message=raw_error["message"]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            {"status": "unknown"},
            {"status": "in_progress"},
            {"status": None},
            {"status": []},
            {
                "status": "completed",
                "error": {"code": "conflict", "message": "must not coexist"},
            },
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "error": {"code": "conflict", "message": "must not coexist"},
            },
            {
                "status": "completed",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            {"status": "failed", "error": "not-an-object"},
            {"status": "incomplete", "incomplete_details": "not-an-object"},
        ],
        ids=[
            "unknown",
            "in-progress",
            "null",
            "non-string",
            "completed-with-error",
            "incomplete-with-error",
            "completed-with-details",
            "malformed-error",
            "malformed-details",
        ],
    )
    async def test_streaming_malformed_terminal_is_protocol_failure(self, response: dict):
        provider = _make_provider_with_stream(
            _sse_lines([{"type": "response.completed", "response": response}])
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502
        assert exc_info.value.upstream_status_code == 200
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [None, [], ["DO_NOT_LEAK"]],
        ids=["null", "list", "list-secret"],
    )
    async def test_streaming_terminal_response_must_be_object(self, response):
        provider = _make_provider_with_stream(
            _sse_lines([{"type": "response.completed", "response": response}])
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert "DO_NOT_LEAK" not in str(exc_info.value)


class TestToolSearchCallForwarding:
    """gpt-5.x's `tool_search_call` must surface as a regular function_call."""

    @pytest.mark.asyncio
    async def test_tool_search_call_emitted_as_function_call(self):
        events = [
            {"type": "response.created", "response": {}},
            {"type": "response.in_progress", "response": {}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "tool_search_call",
                    "execution": "client",
                    "call_id": "call_abc123",
                    "status": "in_progress",
                    "arguments": {},
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "tool_search_call",
                    "execution": "client",
                    "call_id": "call_abc123",
                    "status": "completed",
                    "arguments": {"query": "writing files", "limit": 8},
                },
            },
            {
                "type": "response.completed",
                "response": {"status": "completed", "usage": None},
            },
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)
        tool_chunks = [c for c in chunks if c.tool_call is not None]

        assert len(tool_chunks) == 1, f"expected one tool_call chunk, got {chunks}"
        tc = tool_chunks[0].tool_call
        assert tc is not None
        assert tc.name == "tool_search"
        assert tc.kind == "tool_search"
        assert tc.is_custom is False
        assert tc.call_id == "call_abc123"
        # Arguments must be a JSON string (function_call shape downstream).
        assert json.loads(tc.arguments) == {"query": "writing files", "limit": 8}

    @pytest.mark.asyncio
    async def test_tool_search_call_with_string_arguments_passes_through(self):
        # Hypothetical: if Copilot ever serializes arguments as a string,
        # we must not double-encode.
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "tool_search_call",
                    "execution": "client",
                    "call_id": "call_xyz",
                    "arguments": '{"query": "x"}',
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)
        tool_chunks = [c for c in chunks if c.tool_call is not None]

        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_call
        assert tc is not None
        assert tc.kind == "tool_search"
        assert tc.arguments == '{"query": "x"}'

    @pytest.mark.asyncio
    async def test_provider_preserves_generic_source_item_provenance(self):
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-zero",
                "output_index": 0,
                "summary_index": 0,
                "delta": "plan",
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg-one",
                "output_index": 1,
                "content_index": 0,
                "delta": "answer",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs-zero",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "message",
                    "id": "msg-one",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        canonical = [chunk for chunk in chunks if chunk.output_index is not None]
        assert [chunk.output_index for chunk in canonical] == [0, 1, 0, 1]
        assert [chunk.output_item_type for chunk in canonical] == [
            "reasoning",
            "message",
            "reasoning",
            "message",
        ]
        assert [chunk.output_item_done for chunk in canonical] == [False, False, True, True]
        assert canonical[1].content_index == 0

    @pytest.mark.asyncio
    async def test_provider_populates_provenance_for_every_supported_item_type(self):
        events = [
            {
                "type": "response.refusal.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "denied",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg-refusal",
                    "content": [{"type": "refusal", "refusal": "denied"}],
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "call_id": "call-function",
                    "name": "lookup",
                    "arguments": "{}",
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {
                    "type": "custom_tool_call",
                    "call_id": "call-custom",
                    "name": "apply_patch",
                    "input": "patch",
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 3,
                "item": {
                    "type": "tool_search_call",
                    "call_id": "call-search",
                    "arguments": {"query": "tools"},
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        payloads = [chunk for chunk in chunks if chunk.output_index is not None]
        assert [
            (
                chunk.output_index,
                chunk.output_item_type,
                chunk.content_index,
                chunk.output_item_done,
            )
            for chunk in payloads
        ] == [
            (0, "message", 0, False),
            (0, "message", None, True),
            (1, "function_call", None, True),
            (2, "custom_tool_call", None, True),
            (3, "tool_search_call", None, True),
        ]
        assert [chunk.tool_call.kind for chunk in payloads if chunk.tool_call] == [
            "function",
            "custom",
            "tool_search",
        ]

    @staticmethod
    def _completed_item(item_type: str) -> dict[str, Any]:
        if item_type == "message":
            return {
                "type": "message",
                "id": "msg-duplicate",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer"}],
            }
        if item_type == "function_call":
            return {
                "type": "function_call",
                "call_id": "call-function",
                "name": "lookup",
                "arguments": '{"key":"value"}',
                "namespace": "tools",
            }
        if item_type == "custom_tool_call":
            return {
                "type": "custom_tool_call",
                "call_id": "call-custom",
                "name": "apply_patch",
                "input": "patch",
            }
        return {
            "type": "tool_search_call",
            "call_id": "call-search",
            "arguments": {"query": "tools", "limit": 8},
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "item_type",
        ["message", "function_call", "custom_tool_call", "tool_search_call"],
    )
    async def test_identical_completed_output_item_is_idempotent(self, item_type):
        item = self._completed_item(item_type)
        done = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        }
        provider = _make_provider_with_stream(
            _sse_lines(
                [done, done, {"type": "response.completed", "response": {"status": "completed"}}]
            )
        )

        chunks = await _collect(provider)

        completed = [chunk for chunk in chunks if chunk.output_item_done]
        assert len(completed) == 1
        assert completed[0].output_item_type == item_type
        assert chunks[-1].terminal_outcome is not None
        assert chunks[-1].terminal_outcome.response_status is ResponseStatus.COMPLETED

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("item_type", "field", "conflicting_value"),
        [
            ("message", "id", "msg-conflict"),
            ("function_call", "arguments", '{"key":"other"}'),
            ("custom_tool_call", "input", "different patch"),
            ("tool_search_call", "arguments", {"query": "different"}),
        ],
    )
    async def test_conflicting_completed_output_item_is_protocol_failure(
        self, item_type, field, conflicting_value
    ):
        first = self._completed_item(item_type)
        second = {**first, field: conflicting_value}
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {"type": "response.output_item.done", "output_index": 0, "item": first},
                    {"type": "response.output_item.done", "output_index": 0, "item": second},
                ]
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_completed_output_index_cannot_change_between_reasoning_and_message(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-first",
                            "summary": [{"type": "summary_text", "text": "plan"}],
                        },
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": self._completed_item("message"),
                    },
                ]
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("added_type", "done_type"),
        [
            ("reasoning", "message"),
            ("function_call", "custom_tool_call"),
            ("custom_tool_call", "function_call"),
            ("tool_search_call", "message"),
            ("message", "function_call"),
        ],
    )
    async def test_output_item_added_type_must_match_done_type(self, added_type, done_type):
        added_item = self._completed_item(added_type)
        done_item = self._completed_item(done_type)
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": added_item,
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": done_item,
                    },
                ]
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("typed_event", "done_type"),
        [
            (
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "",
                },
                "reasoning",
            ),
            (
                {
                    "type": "response.refusal.done",
                    "output_index": 0,
                    "content_index": 0,
                    "refusal": "",
                },
                "reasoning",
            ),
            (
                {
                    "type": "response.reasoning_summary_text.done",
                    "output_index": 0,
                    "summary_index": 0,
                    "text": "",
                },
                "message",
            ),
        ],
    )
    async def test_typed_event_item_type_must_match_done_type(self, typed_event, done_type):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    typed_event,
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": self._completed_item(done_type),
                    },
                ]
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event",
        [
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "{}",
            },
            {
                "type": "response.custom_tool_call_input.done",
                "output_index": 0,
                "input": "patch",
            },
        ],
    )
    async def test_tool_payload_event_without_added_item_is_protocol_failure(self, event):
        provider = _make_provider_with_stream(_sse_lines([event]))

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL


class TestEmptyTypedEventProvenance:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_event_type", ["delta", "done"])
    async def test_empty_reasoning_event_item_id_does_not_bind_identity(self, empty_event_type):
        later_summary_index = 0 if empty_event_type == "delta" else 1
        empty_event = {
            "type": f"response.reasoning_summary_text.{empty_event_type}",
            "item_id": "rs-A",
            "output_index": 0,
            "summary_index": 0,
            "delta" if empty_event_type == "delta" else "text": "",
        }
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    empty_event,
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "rs-B",
                        "output_index": 0,
                        "summary_index": later_summary_index,
                        "delta": "thought",
                    },
                ]
            )
        )

        chunks = await _collect(provider)

        assert [chunk.thinking for chunk in chunks if chunk.thinking] == ["thought"]
        assert not any(chunk.thinking_id for chunk in chunks)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_event_type", ["delta", "done"])
    async def test_empty_summary_item_id_is_not_used_as_terminal_identity(self, empty_event_type):
        later_summary_index = 0 if empty_event_type == "delta" else 1
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": f"response.reasoning_summary_text.{empty_event_type}",
                        "item_id": "rs-empty-first",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta" if empty_event_type == "delta" else "text": "",
                    },
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "output_index": 0,
                        "summary_index": later_summary_index,
                        "delta": "thought",
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        wire_events = await _route_provider_chunks(provider)

        completed = next(event for event in wire_events if event["type"] == "response.completed")
        output = completed["response"]["output"]
        assert len(output) == 1
        assert output[0]["type"] == "reasoning"
        assert output[0]["id"].startswith("rs-")
        assert output[0]["id"] != "rs-empty-first"
        assert output[0]["summary"] == [{"type": "summary_text", "text": "thought"}]

    @pytest.mark.asyncio
    async def test_final_reasoning_identity_pairs_with_buffered_summary_and_blob(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "rs-empty-first",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": "",
                    },
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": "thought",
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-empty-first",
                            "summary": [{"type": "summary_text", "text": "thought"}],
                            "encrypted_content": "ENC_EMPTY_FIRST",
                        },
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        wire_events = await _route_provider_chunks(provider)

        completed = next(event for event in wire_events if event["type"] == "response.completed")
        assert completed["response"]["output"] == [
            {
                "type": "reasoning",
                "id": "rs-empty-first",
                "summary": [{"type": "summary_text", "text": "thought"}],
                "encrypted_content": "ENC_EMPTY_FIRST",
            }
        ]

    @pytest.mark.asyncio
    async def test_duplicate_empty_reasoning_identity_does_not_create_visible_output(self):
        event = {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs-empty-only",
            "output_index": 0,
            "summary_index": 0,
            "delta": "",
        }
        provider = _make_provider_with_stream(
            _sse_lines(
                [event, event, {"type": "response.completed", "response": {"status": "completed"}}]
            )
        )

        wire_events = await _route_provider_chunks(provider)

        completed = next(event for event in wire_events if event["type"] == "response.completed")
        assert completed["response"]["output"] == []

    @pytest.mark.asyncio
    async def test_empty_only_reasoning_item_does_not_leak_identity_to_next_item(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "rs-empty",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": "",
                    },
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "rs-next",
                        "output_index": 1,
                        "summary_index": 0,
                        "delta": "next",
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        wire_events = await _route_provider_chunks(provider)

        completed = next(event for event in wire_events if event["type"] == "response.completed")
        output = completed["response"]["output"]
        assert len(output) == 1
        assert output[0]["type"] == "reasoning"
        assert output[0]["id"].startswith("rs-")
        assert output[0]["id"] != "rs-next"
        assert output[0]["summary"] == [{"type": "summary_text", "text": "next"}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "eof"])
    async def test_future_message_marker_done_boundary_keeps_messages_separate(self, terminal):
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-gate",
                "output_index": 0,
                "summary_index": 0,
                "delta": "gate",
            },
            {
                "type": "response.output_text.delta",
                "output_index": 1,
                "content_index": 0,
                "delta": "A",
            },
            {
                "type": "response.output_text.delta",
                "output_index": 1,
                "content_index": 0,
                "delta": "",
            },
            {
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {
                    "type": "message",
                    "id": "msg-B",
                    "content": [{"type": "output_text", "text": "B"}],
                },
            },
        ]
        if terminal:
            events.append({"type": "response.completed", "response": {"status": "completed"}})
        provider = _make_provider_with_stream(_sse_lines(events))

        wire_events = await _route_provider_chunks(provider)

        terminal_event = next(
            event
            for event in wire_events
            if event["type"] in {"response.completed", "response.incomplete"}
        )
        output = terminal_event["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", "message", "message"]
        assert [item["content"][0]["text"] for item in output[1:]] == ["A", "B"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "eof"])
    async def test_future_reasoning_marker_done_boundary_keeps_reasoning_items_separate(
        self, terminal
    ):
        events = [
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "gate",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-A",
                "output_index": 1,
                "summary_index": 0,
                "delta": "A",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-A",
                "output_index": 1,
                "summary_index": 0,
                "delta": "",
            },
            {
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {
                    "type": "reasoning",
                    "id": "rs-B",
                    "summary": [{"type": "summary_text", "text": "B"}],
                },
            },
        ]
        if terminal:
            events.append({"type": "response.completed", "response": {"status": "completed"}})
        provider = _make_provider_with_stream(_sse_lines(events))

        wire_events = await _route_provider_chunks(provider)

        terminal_event = next(
            event
            for event in wire_events
            if event["type"] in {"response.completed", "response.incomplete"}
        )
        output = terminal_event["response"]["output"]
        assert [item["type"] for item in output] == ["message", "reasoning", "reasoning"]
        assert [item["summary"][0]["text"] for item in output[1:]] == ["A", "B"]
        assert output[1]["id"].startswith("rs-")
        assert output[1]["id"] != "rs-A"
        assert output[2]["id"] == "rs-B"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("event_type", "index_field", "payload_field"),
        [
            ("response.output_text.delta", "content_index", "delta"),
            ("response.output_text.done", "content_index", "text"),
            ("response.refusal.delta", "content_index", "delta"),
            ("response.refusal.done", "content_index", "refusal"),
            ("response.reasoning_summary_text.delta", "summary_index", "delta"),
            ("response.reasoning_summary_text.done", "summary_index", "text"),
        ],
    )
    @pytest.mark.parametrize("source_index", [-1, 2], ids=["negative", "initial-gap"])
    async def test_empty_typed_event_keeps_invalid_source_index_for_route_validation(
        self, event_type, index_field, payload_field, source_index
    ):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": event_type,
                        "output_index": 0,
                        index_field: source_index,
                        payload_field: "",
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        wire_events = await _route_provider_chunks(provider)

        failed = next(event for event in wire_events if event["type"] == "response.failed")
        assert failed["response"]["error"]["code"] == "upstream_protocol_error"
        assert not any(event["type"] == "response.completed" for event in wire_events)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("event_type", "index_field", "payload_field"),
        [
            ("response.output_text.done", "content_index", "text"),
            ("response.refusal.done", "content_index", "refusal"),
            ("response.reasoning_summary_text.done", "summary_index", "text"),
        ],
    )
    async def test_legal_empty_done_and_duplicate_do_not_create_visible_output(
        self, event_type, index_field, payload_field
    ):
        event = {
            "type": event_type,
            "output_index": 0,
            index_field: 0,
            payload_field: "",
        }
        provider = _make_provider_with_stream(
            _sse_lines(
                [event, event, {"type": "response.completed", "response": {"status": "completed"}}]
            )
        )

        wire_events = await _route_provider_chunks(provider)

        completed = next(event for event in wire_events if event["type"] == "response.completed")
        assert completed["response"]["output"] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("delta_type", "done_type", "index_field", "done_field", "value", "item_type"),
        [
            (
                "response.output_text.delta",
                "response.output_text.done",
                "content_index",
                "text",
                "answer",
                "message",
            ),
            (
                "response.refusal.delta",
                "response.refusal.done",
                "content_index",
                "refusal",
                "denied",
                "message",
            ),
            (
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "summary_index",
                "text",
                "thought",
                "reasoning",
            ),
        ],
    )
    async def test_empty_delta_preserves_later_nonempty_done_snapshot(
        self, delta_type, done_type, index_field, done_field, value, item_type
    ):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": delta_type,
                        "output_index": 0,
                        index_field: 0,
                        "delta": "",
                    },
                    {
                        "type": done_type,
                        "output_index": 0,
                        index_field: 0,
                        done_field: value,
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        wire_events = await _route_provider_chunks(provider)

        completed = next(event for event in wire_events if event["type"] == "response.completed")
        assert completed["response"]["output"][0]["type"] == item_type
        if item_type == "reasoning":
            assert completed["response"]["output"][0]["summary"][0]["text"] == value
        else:
            part = completed["response"]["output"][0]["content"][0]
            assert part.get("text", part.get("refusal")) == value

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("empty_type", "nonempty_type", "index_field", "payload_field", "value", "item_type"),
        [
            (
                "response.output_text.done",
                "response.output_text.done",
                "content_index",
                "text",
                "answer",
                "message",
            ),
            (
                "response.refusal.done",
                "response.refusal.done",
                "content_index",
                "refusal",
                "denied",
                "message",
            ),
            (
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_text.done",
                "summary_index",
                "text",
                "thought",
                "reasoning",
            ),
        ],
    )
    async def test_empty_part_before_nonempty_part_validates_source_order_without_visible_empty(
        self, empty_type, nonempty_type, index_field, payload_field, value, item_type
    ):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": empty_type,
                        "output_index": 0,
                        index_field: 0,
                        payload_field: "",
                    },
                    {
                        "type": nonempty_type,
                        "output_index": 0,
                        index_field: 1,
                        payload_field: value,
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        wire_events = await _route_provider_chunks(provider)

        completed = next(event for event in wire_events if event["type"] == "response.completed")
        output = completed["response"]["output"]
        assert len(output) == 1
        assert output[0]["type"] == item_type
        if item_type == "reasoning":
            assert output[0]["summary"] == [{"type": "summary_text", "text": value}]
            assert {
                event["summary_index"]
                for event in wire_events
                if event["type"].startswith("response.reasoning_summary")
            } == {0}
        else:
            part = output[0]["content"][0]
            assert part.get("text", part.get("refusal")) == value

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("done_type", "delta_type", "index_field", "done_field"),
        [
            (
                "response.output_text.done",
                "response.output_text.delta",
                "content_index",
                "text",
            ),
            (
                "response.refusal.done",
                "response.refusal.delta",
                "content_index",
                "refusal",
            ),
            (
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_text.delta",
                "summary_index",
                "text",
            ),
        ],
    )
    async def test_empty_delta_after_done_is_protocol_failure(
        self, done_type, delta_type, index_field, done_field
    ):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": done_type,
                        "output_index": 0,
                        index_field: 0,
                        done_field: "",
                    },
                    {
                        "type": delta_type,
                        "output_index": 0,
                        index_field: 0,
                        "delta": "",
                    },
                ]
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL

    @pytest.mark.asyncio
    async def test_message_done_snapshot_recovers_content_without_delta_events(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg-only-snapshot",
                            "content": [
                                {"type": "output_text", "text": "answer"},
                                {"type": "refusal", "refusal": "denied"},
                            ],
                        },
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        chunks = await _collect(provider)

        assert [(chunk.content, chunk.refusal, chunk.content_index) for chunk in chunks[:-1]] == [
            ("answer", None, 0),
            ("", "denied", 1),
            ("", None, None),
        ]
        assert chunks[-2].output_item_done is True

    @pytest.mark.asyncio
    async def test_message_done_snapshot_rejects_conflicting_content_indices(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 1,
                        "delta": "second",
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg-sparse-part",
                            "content": [{"type": "output_text", "text": "second"}],
                        },
                    },
                ]
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL

    @pytest.mark.asyncio
    async def test_cross_type_interleaving_round_trips_in_source_order(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "rs-zero",
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": "one",
                    },
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "rs-two",
                        "output_index": 2,
                        "summary_index": 0,
                        "delta": "two",
                    },
                    {
                        "type": "response.output_text.delta",
                        "item_id": "msg-one",
                        "output_index": 1,
                        "content_index": 0,
                        "delta": "answer",
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-zero",
                            "encrypted_content": "ENC_ZERO",
                            "summary": [{"type": "summary_text", "text": "one"}],
                        },
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 2,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-two",
                            "encrypted_content": "ENC_TWO",
                            "summary": [{"type": "summary_text", "text": "two"}],
                        },
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": {
                            "type": "message",
                            "id": "msg-one",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "answer"}],
                        },
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        events = await _route_provider_chunks(provider)

        completed = next(event for event in events if event.get("type") == "response.completed")
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", "message", "reasoning"]
        assert output[0]["id"] == "rs-zero"
        assert output[1]["content"][0]["text"] == "answer"
        assert output[2]["id"] == "rs-two"

    @pytest.mark.asyncio
    async def test_three_interleaved_reasoning_items_remain_atomic(self):
        events = []
        for index, name in enumerate(("zero", "one", "two")):
            events.append(
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": f"rs-{name}",
                    "output_index": index,
                    "summary_index": 0,
                    "delta": name,
                }
            )
        for index, name in ((1, "one"), (0, "zero"), (2, "two")):
            events.append(
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": {
                        "type": "reasoning",
                        "id": f"rs-{name}",
                        "encrypted_content": f"ENC_{name.upper()}",
                        "summary": [{"type": "summary_text", "text": name}],
                    },
                }
            )
        events.append({"type": "response.completed", "response": {"status": "completed"}})
        provider = _make_provider_with_stream(_sse_lines(events))

        routed = await _route_provider_chunks(provider)

        completed = next(event for event in routed if event.get("type") == "response.completed")
        assert [
            (item["id"], item["summary"][0]["text"], item["encrypted_content"])
            for item in completed["response"]["output"]
        ] == [
            ("rs-zero", "zero", "ENC_ZERO"),
            ("rs-one", "one", "ENC_ONE"),
            ("rs-two", "two", "ENC_TWO"),
        ]


class TestNamespacePreservation:
    """MCP-namespaced function_calls must preserve `namespace` end-to-end.

    Copilot CAPI rejects the next turn with
    ``Missing namespace for function_call 'X'`` if a previously-namespaced
    call is round-tripped without it (v0.3.7 → v0.3.8 bug).
    """

    @pytest.mark.asyncio
    async def test_namespace_captured_from_output_item_added(self):
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc-1",
                    "call_id": "call_kusto_1",
                    "name": "execute_query",
                    "namespace": "kusto",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": '{"query":"Heartbeat | take 5"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc-1",
                    "call_id": "call_kusto_1",
                    "name": "execute_query",
                    "namespace": "kusto",
                    "arguments": '{"query":"Heartbeat | take 5"}',
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))
        chunks = await _collect(provider)
        tool_chunks = [c for c in chunks if c.tool_call is not None]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_call
        assert tc is not None
        assert tc.name == "execute_query"
        assert tc.namespace == "kusto"
        assert tc.kind == "function"

    @pytest.mark.asyncio
    async def test_namespace_only_on_done_event(self):
        # This is the actual production wire shape: Copilot CAPI attaches
        # namespace ONLY on output_item.done, NOT on output_item.added (which
        # arrives before the model has decided which namespaced tool to invoke).
        # We must defer emission until output_item.done so the field is present.
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "c1", "name": "x"},
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": "{}",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "x",
                    "namespace": "ns1",
                    "arguments": "{}",
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))
        chunks = await _collect(provider)
        tool_chunks = [c for c in chunks if c.tool_call is not None]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_call
        assert tc is not None
        assert tc.namespace == "ns1"

    @pytest.mark.asyncio
    async def test_no_namespace_stays_none(self):
        # Regression: standard (non-MCP) function_calls don't carry namespace.
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "c1", "name": "weather"},
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": '{"city":"NYC"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "weather",
                    "arguments": '{"city":"NYC"}',
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))
        chunks = await _collect(provider)
        tool_chunks = [c for c in chunks if c.tool_call is not None]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call is not None
        assert tool_chunks[0].tool_call.namespace is None

    @pytest.mark.asyncio
    async def test_emission_deferred_until_output_item_done(self):
        # Reproduction of the v0.3.9 production bug: arguments.done fires
        # BEFORE output_item.done, and Copilot only puts namespace on
        # output_item.done. Emitting on arguments.done loses namespace and
        # the next turn 400s with ``Missing namespace for function_call 'X'``.
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call_late_ns",
                    "name": "execute_query",
                },  # no namespace yet
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": '{"query":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": '"x"}',
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "arguments": '{"query":"x"}',
            },  # still no namespace; we MUST NOT emit here
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call_late_ns",
                    "name": "execute_query",
                    "namespace": "mcp__kusto_mcp__",
                    "arguments": '{"query":"x"}',
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))
        chunks = await _collect(provider)
        tool_chunks = [c for c in chunks if c.tool_call is not None]
        assert len(tool_chunks) == 1, (
            f"expected exactly one emission deferred to output_item.done, got {len(tool_chunks)}"
        )
        tc = tool_chunks[0].tool_call
        assert tc is not None
        assert tc.name == "execute_query"
        assert tc.namespace == "mcp__kusto_mcp__"
        assert tc.arguments == '{"query":"x"}'


class TestUnknownEventNoise:
    """Benign upstream events should not trigger the unhandled-event warning."""

    @pytest.mark.asyncio
    async def test_benign_events_skipped_from_warning(self, caplog):
        events = [
            {"type": "response.created", "response": {}},
            {"type": "response.in_progress", "response": {}},
            {
                "type": "response.content_part.added",
                "item_id": "msg-1",
                "part": {"type": "output_text", "text": ""},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg-1",
                "delta": "hello",
            },
            {
                "type": "response.output_text.done",
                "item_id": "msg-1",
                "text": "hello",
            },
            {
                "type": "response.content_part.done",
                "item_id": "msg-1",
                "part": {"type": "output_text", "text": "hello"},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg-1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        with caplog.at_level(logging.WARNING, logger="providers.copilot"):
            await _collect(provider)

        warnings = [r for r in caplog.records if "unhandled event types" in r.getMessage()]
        assert warnings == [], (
            "benign upstream events leaked into the unknown-event warning: "
            f"{[w.getMessage() for w in warnings]}"
        )

    @pytest.mark.asyncio
    async def test_genuinely_unknown_event_still_warned(self, caplog):
        events = [
            {"type": "response.created", "response": {}},
            {
                "type": "response.brand_new_event_type_we_dont_know",
                "data": "stuff",
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        with caplog.at_level(logging.WARNING, logger="providers.copilot"):
            await _collect(provider)

        warnings = [r for r in caplog.records if "unhandled event types" in r.getMessage()]
        assert len(warnings) == 1
        assert "brand_new_event_type_we_dont_know" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_reasoning_summary_part_events_skipped_from_warning(self, caplog):
        """``reasoning_summary_part.added/done`` are pure structure envelopes.

        The route synthesizes its own equivalents from the
        ``reasoning_summary_text.delta`` we already consume — same pattern as
        ``content_part.*`` for messages. Without the skip, every xhigh request
        produced a noisy ``unhandled event types`` warning that drowned out
        genuinely-unknown events.
        """
        events = [
            {"type": "response.created", "response": {}},
            {
                "type": "response.reasoning_summary_part.added",
                "item_id": "rs-1",
                "summary_index": 0,
                "part": {"type": "summary_text", "text": ""},
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-1",
                "delta": "thinking...",
            },
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs-1",
                "text": "thinking...",
            },
            {
                "type": "response.reasoning_summary_part.done",
                "item_id": "rs-1",
                "summary_index": 0,
                "part": {"type": "summary_text", "text": "thinking..."},
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        with caplog.at_level(logging.WARNING, logger="providers.copilot"):
            await _collect(provider)

        warnings = [r for r in caplog.records if "unhandled event types" in r.getMessage()]
        assert warnings == [], (
            "reasoning_summary_part envelopes leaked into the unknown-event warning: "
            f"{[w.getMessage() for w in warnings]}"
        )


class TestThinkingSignatureSource:
    """The reasoning ``thinking_signature`` must be the upstream encrypted blob.

    Codex round-trips it back to Copilot as ``encrypted_content``; if we emit
    the local ``item_id`` (a short identifier, not a verifiable blob), Copilot
    400s the next turn with ``Encrypted content could not be decrypted``.
    """

    @pytest.mark.asyncio
    async def test_done_snapshot_completes_partial_reasoning_without_item_id_signature(self):
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-partial",
                "output_index": 0,
                "summary_index": 0,
                "delta": "think",
            },
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs-partial",
                "output_index": 0,
                "summary_index": 0,
                "text": "thinking",
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        assert [chunk.thinking for chunk in chunks if chunk.thinking] == ["think", "ing"]
        assert not any(chunk.thinking_signature for chunk in chunks)

    @pytest.mark.asyncio
    async def test_done_only_and_duplicate_reasoning_snapshots_are_complete_and_idempotent(self):
        done = {
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs-done-only",
            "output_index": 0,
            "summary_index": 0,
            "text": "complete thought",
        }
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    done,
                    done,
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        chunks = await _collect(provider)

        assert [chunk.thinking for chunk in chunks if chunk.thinking] == ["complete thought"]
        assert not any(chunk.thinking_signature for chunk in chunks)

    @pytest.mark.asyncio
    async def test_conflicting_reasoning_done_snapshot_is_typed_protocol_failure(self):
        events = [
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs-conflict",
                "output_index": 0,
                "summary_index": 0,
                "text": "alpha",
            },
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs-conflict",
                "output_index": 0,
                "summary_index": 0,
                "text": "omega",
            },
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL

    @pytest.mark.asyncio
    async def test_output_item_done_reconciles_each_reasoning_item_independently(self):
        encrypted_one = "ENC_ONE"
        encrypted_two = "ENC_TWO"
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-one",
                "output_index": 0,
                "summary_index": 0,
                "delta": "one",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-one",
                "output_index": 0,
                "summary_index": 1,
                "delta": "sec",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs-one",
                    "encrypted_content": encrypted_one,
                    "summary": [
                        {"type": "summary_text", "text": "one complete"},
                        {"type": "summary_text", "text": "second"},
                    ],
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "reasoning",
                    "id": "rs-two",
                    "encrypted_content": encrypted_two,
                    "summary": [{"type": "summary_text", "text": "two complete"}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        assert [chunk.thinking for chunk in chunks if chunk.thinking] == [
            "one",
            "sec",
            " complete",
            "ond",
            "two complete",
        ]
        assert [chunk.thinking_signature for chunk in chunks if chunk.thinking_signature] == [
            encrypted_one,
            encrypted_two,
        ]

    @pytest.mark.asyncio
    async def test_provider_route_preserves_interleaved_reasoning_summary_parts(self):
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-one",
                "output_index": 0,
                "summary_index": 0,
                "delta": "one",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-one",
                "output_index": 0,
                "summary_index": 1,
                "delta": "sec",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs-one",
                    "encrypted_content": "ENC_ONE",
                    "summary": [
                        {"type": "summary_text", "text": "one complete"},
                        {"type": "summary_text", "text": "second"},
                    ],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        wire_events = await _route_provider_chunks(provider)

        completed = next(event for event in wire_events if event["type"] == "response.completed")
        assert completed["response"]["output"] == [
            {
                "type": "reasoning",
                "id": "rs-one",
                "summary": [
                    {"type": "summary_text", "text": "one complete"},
                    {"type": "summary_text", "text": "second"},
                ],
                "encrypted_content": "ENC_ONE",
            }
        ]

    @pytest.mark.asyncio
    async def test_output_item_done_matches_reasoning_events_without_item_id_by_output_index(self):
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 3,
                "summary_index": 0,
                "delta": "plan",
            },
            {
                "type": "response.output_item.done",
                "output_index": 3,
                "item": {
                    "type": "reasoning",
                    "id": "rs-late-id",
                    "encrypted_content": "ENC_LATE",
                    "summary": [{"type": "summary_text", "text": "planning"}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        assert [chunk.thinking for chunk in chunks if chunk.thinking] == ["plan", "ning"]
        signature_chunks = [chunk for chunk in chunks if chunk.thinking_signature]
        assert [(chunk.thinking_id, chunk.thinking_signature) for chunk in signature_chunks] == [
            ("rs-late-id", "ENC_LATE")
        ]

    @pytest.mark.asyncio
    async def test_rotating_summary_item_ids_defer_to_final_reasoning_identity(self):
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "opaque-added",
                    "summary": [],
                },
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "opaque-delta-one",
                "output_index": 0,
                "summary_index": 0,
                "delta": "plan",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "opaque-delta-two",
                "output_index": 0,
                "summary_index": 0,
                "delta": "ning",
            },
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "opaque-done",
                "output_index": 0,
                "summary_index": 0,
                "text": "planning",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs-canonical",
                    "encrypted_content": "ENC_CANONICAL",
                    "summary": [{"type": "summary_text", "text": "planning"}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        assert "".join(chunk.thinking or "" for chunk in chunks) == "planning"
        assert [chunk.thinking_id for chunk in chunks if chunk.thinking_id] == ["rs-canonical"]
        signature_chunks = [chunk for chunk in chunks if chunk.thinking_signature]
        assert [(chunk.thinking_id, chunk.thinking_signature) for chunk in signature_chunks] == [
            ("rs-canonical", "ENC_CANONICAL")
        ]

    @pytest.mark.asyncio
    async def test_done_snapshot_matches_delta_by_output_index_despite_opaque_item_id(self):
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 4,
                "summary_index": 0,
                "delta": "think",
            },
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs-late",
                "output_index": 4,
                "summary_index": 0,
                "text": "thinking",
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        assert [chunk.thinking for chunk in chunks if chunk.thinking] == ["think", "ing"]

    @pytest.mark.asyncio
    async def test_summary_item_id_is_not_forwarded_as_reasoning_identity(self):
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-early",
                "output_index": 4,
                "summary_index": 0,
                "delta": "think",
            },
            {
                "type": "response.reasoning_summary_text.done",
                "output_index": 4,
                "summary_index": 0,
                "text": "thinking",
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        thinking_chunks = [chunk for chunk in chunks if chunk.thinking]
        assert [chunk.thinking for chunk in thinking_chunks] == ["think", "ing"]
        assert not any(chunk.thinking_id for chunk in thinking_chunks)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("identified_first", [False, True], ids=["id-gain", "id-loss"])
    async def test_duplicate_reasoning_item_with_changed_id_presence_emits_signature_once(
        self, identified_first
    ):
        item_without_id = {
            "type": "reasoning",
            "encrypted_content": "ENC_ONCE",
            "summary": [{"type": "summary_text", "text": "thought"}],
        }
        item_with_id = {**item_without_id, "id": "rs-late"}
        first_item, second_item = (
            (item_with_id, item_without_id) if identified_first else (item_without_id, item_with_id)
        )
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 5,
                "item": first_item,
            },
            {
                "type": "response.output_item.done",
                "output_index": 5,
                "item": second_item,
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        assert [chunk.thinking for chunk in chunks if chunk.thinking] == ["thought"]
        assert [chunk.thinking_signature for chunk in chunks if chunk.thinking_signature] == [
            "ENC_ONCE"
        ]

    @pytest.mark.asyncio
    async def test_reasoning_item_missing_streamed_summary_part_is_protocol_failure(self):
        events = [
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-orphan",
                "output_index": 6,
                "summary_index": 1,
                "delta": "orphan",
            },
            {
                "type": "response.output_item.done",
                "output_index": 6,
                "item": {
                    "type": "reasoning",
                    "id": "rs-orphan",
                    "encrypted_content": "ENC_ORPHAN",
                    "summary": [{"type": "summary_text", "text": "kept"}],
                },
            },
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("event_type", "field"),
        [
            ("response.reasoning_summary_text.delta", "delta"),
            ("response.reasoning_summary_text.done", "text"),
        ],
    )
    async def test_reasoning_event_cannot_add_part_after_output_item_done(self, event_type, field):
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 7,
                "item": {
                    "type": "reasoning",
                    "id": "rs-final",
                    "encrypted_content": "ENC_FINAL",
                    "summary": [{"type": "summary_text", "text": "final"}],
                },
            },
            {
                "type": event_type,
                "item_id": "rs-final",
                "output_index": 7,
                "summary_index": 1,
                field: "orphan",
            },
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL

    @pytest.mark.asyncio
    async def test_identical_reasoning_done_after_output_item_done_is_idempotent(self):
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 8,
                "item": {
                    "type": "reasoning",
                    "id": "rs-final",
                    "encrypted_content": "ENC_FINAL",
                    "summary": [{"type": "summary_text", "text": "final"}],
                },
            },
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs-final",
                "output_index": 8,
                "summary_index": 0,
                "text": "final",
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        assert [chunk.thinking for chunk in chunks if chunk.thinking] == ["final"]
        assert [chunk.thinking_signature for chunk in chunks if chunk.thinking_signature] == [
            "ENC_FINAL"
        ]

    @pytest.mark.asyncio
    async def test_signature_is_encrypted_blob_not_item_id(self):
        encrypted_blob = "ENC_BLOB_" + "X" * 200  # stand-in for the real ~2KB blob
        events = [
            {"type": "response.created", "response": {}},
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs-upstream-1",
                "delta": "thinking...",
            },
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs-upstream-1",
                "text": "thinking...",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs-upstream-1",
                    "encrypted_content": encrypted_blob,
                    "summary": [{"type": "summary_text", "text": "thinking..."}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        sigs = [c.thinking_signature for c in chunks if c.thinking_signature]
        # Exactly one signature must be emitted, and it must be the encrypted
        # blob — not the local ``item_id``. The previous behavior emitted
        # ``rs-upstream-1`` first (from summary_text.done), which the route
        # latched as the final encrypted_content and the encrypted blob
        # arriving later was silently dropped.
        assert sigs == [encrypted_blob], (
            f"expected exactly one signature == encrypted blob, got {sigs!r}"
        )

    @pytest.mark.asyncio
    async def test_final_reasoning_id_is_paired_with_encrypted_blob(self):
        # The canonical id from output_item.done must surface with the
        # encrypted blob. Copilot signs ``encrypted_content`` against this id;
        # pairing the blob with a local or summary-event id 400s the next turn
        # with ``Encrypted content could not be decrypted``.
        encrypted_blob = "ENC_BLOB_" + "Y" * 200
        events = [
            {"type": "response.created", "response": {}},
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_upstream_xyz",
                "delta": "planning...",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_upstream_xyz",
                    "encrypted_content": encrypted_blob,
                    "summary": [{"type": "summary_text", "text": "planning..."}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        thinking_ids = [c.thinking_id for c in chunks if c.thinking_id]
        # Summary-event IDs are ignored; only the final canonical pair is
        # forwarded.
        assert thinking_ids, "expected at least one chunk with thinking_id set"
        assert all(tid == "rs_upstream_xyz" for tid in thinking_ids), (
            f"expected all thinking_ids to be the upstream id, got {thinking_ids!r}"
        )

        # The signature chunk must carry the upstream id AND the blob
        # together so the route can pair them on the reasoning output item.
        sig_chunks = [c for c in chunks if c.thinking_signature]
        assert len(sig_chunks) == 1
        assert sig_chunks[0].thinking_signature == encrypted_blob
        assert sig_chunks[0].thinking_id == "rs_upstream_xyz"

    @pytest.mark.asyncio
    async def test_no_signature_when_upstream_omits_encrypted_content(self):
        # Defensive: if Copilot ever ships a reasoning item without
        # ``encrypted_content``, we MUST NOT fall back to ``item.id`` (a
        # short opaque string) — the codex path would treat it as a blob,
        # round-trip it, and earn a 400.
        events = [
            {"type": "response.created", "response": {}},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_no_blob",
                    "summary": [{"type": "summary_text", "text": "thinking..."}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        chunks = await _collect(provider)

        sigs = [c.thinking_signature for c in chunks if c.thinking_signature]
        assert sigs == [], (
            f"expected no signatures when upstream omits encrypted_content, got {sigs!r}"
        )

    @pytest.mark.asyncio
    async def test_encrypted_reasoning_without_upstream_id_is_protocol_failure(self):
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "encrypted_content": "ENC_WITHOUT_ID",
                    "summary": [{"type": "summary_text", "text": "thinking..."}],
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL

    @pytest.mark.asyncio
    async def test_summary_item_id_cannot_rescue_encrypted_item_without_canonical_id(self):
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "encrypted_content": "ENC_BUFFERED",
                    "summary": [{"type": "summary_text", "text": "thinking..."}],
                },
            },
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs-late-done",
                "output_index": 0,
                "summary_index": 0,
                "text": "thinking...",
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        provider = _make_provider_with_stream(_sse_lines(events))

        with pytest.raises(ProviderError) as exc_info:
            await _collect(provider)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert isinstance(exc_info.value.cause, TypeError)
        assert str(exc_info.value.cause) == (
            "Responses encrypted reasoning is missing its upstream id"
        )

    @pytest.mark.asyncio
    async def test_distinct_reasoning_output_items_round_trip_independently(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-one",
                            "encrypted_content": "ENC_ONE",
                            "summary": [{"type": "summary_text", "text": "one"}],
                        },
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-two",
                            "encrypted_content": "ENC_TWO",
                            "summary": [{"type": "summary_text", "text": "two"}],
                        },
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        events = await _route_provider_chunks(provider)

        completed = next(event for event in events if event.get("type") == "response.completed")
        assert completed["response"]["output"] == [
            {
                "type": "reasoning",
                "id": "rs-one",
                "summary": [{"type": "summary_text", "text": "one"}],
                "encrypted_content": "ENC_ONE",
            },
            {
                "type": "reasoning",
                "id": "rs-two",
                "summary": [{"type": "summary_text", "text": "two"}],
                "encrypted_content": "ENC_TWO",
            },
        ]

    @pytest.mark.asyncio
    async def test_distinct_reasoning_items_without_blobs_round_trip_independently(self):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-one-no-blob",
                            "summary": [{"type": "summary_text", "text": "one"}],
                        },
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-two-no-blob",
                            "summary": [{"type": "summary_text", "text": "two"}],
                        },
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        events = await _route_provider_chunks(provider)

        completed = next(event for event in events if event.get("type") == "response.completed")
        assert completed["response"]["output"] == [
            {
                "type": "reasoning",
                "id": "rs-one-no-blob",
                "summary": [{"type": "summary_text", "text": "one"}],
            },
            {
                "type": "reasoning",
                "id": "rs-two-no-blob",
                "summary": [{"type": "summary_text", "text": "two"}],
            },
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("include_delta_ids", [True, False], ids=["identified", "late-id"])
    async def test_interleaved_reasoning_items_keep_atomic_identity(self, include_delta_ids):
        provider = _make_provider_with_stream(
            _sse_lines(
                [
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "rs-one" if include_delta_ids else None,
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": "one",
                    },
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": "rs-two" if include_delta_ids else None,
                        "output_index": 1,
                        "summary_index": 0,
                        "delta": "two",
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-one",
                            "encrypted_content": "ENC_ONE",
                            "summary": [{"type": "summary_text", "text": "one"}],
                        },
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": {
                            "type": "reasoning",
                            "id": "rs-two",
                            "encrypted_content": "ENC_TWO",
                            "summary": [{"type": "summary_text", "text": "two"}],
                        },
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )

        events = await _route_provider_chunks(provider)

        completed = next(event for event in events if event.get("type") == "response.completed")
        assert completed["response"]["output"] == [
            {
                "type": "reasoning",
                "id": "rs-one",
                "summary": [{"type": "summary_text", "text": "one"}],
                "encrypted_content": "ENC_ONE",
            },
            {
                "type": "reasoning",
                "id": "rs-two",
                "summary": [{"type": "summary_text", "text": "two"}],
                "encrypted_content": "ENC_TWO",
            },
        ]
