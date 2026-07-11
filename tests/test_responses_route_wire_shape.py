"""Tests for the /responses route's wire-shape emission for tool calls.

The route translates internal ``ResponsesToolCall`` objects into the SSE
event sequence that downstream OpenAI-API-compatible clients (Codex CLI in
particular) consume. Each ``kind`` of tool call has its own wire shape; if
the wire shape is wrong, codex's tool dispatcher silently aborts the call
(this is exactly how v0.3.5/v0.3.6 broke ``tool_search``).

These tests pin each shape down end-to-end through ``stream_response``.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.providers.base import (
    ResponsesResponse as InternalResponsesResponse,
)
from router_maestro.providers.base import (
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
)
from router_maestro.server.routes.responses import create_response, stream_response
from router_maestro.server.schemas.responses import ResponsesRequest


class _StubRouter:
    """Minimal Router stub that returns a pre-built chunk stream."""

    def __init__(self, chunks: list[ResponsesStreamChunk]):
        self._chunks = chunks

    async def responses_completion_stream(
        self, request: InternalResponsesRequest, fallback: bool = True
    ) -> tuple[AsyncIterator[ResponsesStreamChunk], str]:
        async def _gen() -> AsyncIterator[ResponsesStreamChunk]:
            for c in self._chunks:
                yield c

        return _gen(), "github-copilot"


class _NonStreamStubRouter:
    """Minimal Router stub for non-streaming /responses route tests."""

    def __init__(self, response: InternalResponsesResponse):
        self._response = response

    async def responses_completion(
        self, request: InternalResponsesRequest, fallback: bool = True
    ) -> tuple[InternalResponsesResponse, str]:
        return self._response, "github-copilot"


def _parse_sse(events: list[str]) -> list[dict[str, Any]]:
    """Parse an SSE-encoded event list into JSON dicts."""
    parsed: list[dict[str, Any]] = []
    for evt in events:
        for line in evt.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: ") :]
                parsed.append(json.loads(payload))
    return parsed


async def _drive(chunks: list[ResponsesStreamChunk]) -> list[dict[str, Any]]:
    router = _StubRouter(chunks)
    req = InternalResponsesRequest(model="gpt-5.5", input="hi", stream=True)
    raw_events: list[str] = []
    async for evt in stream_response(router, req, request_id="req-test", start_time=0.0):  # type: ignore[arg-type]
        raw_events.append(evt)
    return _parse_sse(raw_events)


class TestToolSearchCallWireShape:
    """``kind="tool_search"`` must surface as a real ``tool_search_call`` item.

    Codex's MCP tool-discovery dispatcher (codex-rs/core/src/tools/router.rs)
    matches on ``ResponseItem::ToolSearchCall`` and rejects anything else —
    wrapping the call as ``function_call(name="tool_search")`` causes a
    silent abort and the model retries forever (the v0.3.5/v0.3.6 bug).
    """

    @pytest.mark.asyncio
    async def test_emits_tool_search_call_item(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call_abc123",
                        name="tool_search",
                        arguments='{"query": "writing files", "limit": 8}',
                        kind="tool_search",
                    ),
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        done = [
            e
            for e in events
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "tool_search_call"
        ]
        assert len(done) == 1, f"expected one tool_search_call done event, got {events}"

        item = done[0]["item"]
        assert item["type"] == "tool_search_call"
        assert item["call_id"] == "call_abc123"
        assert item["execution"] == "client", "codex requires execution=client"
        # Arguments must be a dict, NOT a JSON string — codex's parser does
        # ``serde_json::from_value::<SearchToolCallParams>(arguments)``.
        assert item["arguments"] == {"query": "writing files", "limit": 8}

    @pytest.mark.asyncio
    async def test_does_not_emit_function_call_arguments_events(self):
        """Regression: v0.3.5/v0.3.6 emitted function_call_arguments.delta/.done
        which polluted codex's stream parser even when it ignored the items."""
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call_xyz",
                        name="tool_search",
                        arguments='{"query": "x"}',
                        kind="tool_search",
                    ),
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        offending = [
            e
            for e in events
            if e.get("type")
            in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }
        ]
        assert offending == [], (
            "tool_search must not emit function_call_arguments events; "
            f"got: {[e['type'] for e in offending]}"
        )

        # And there must be no function_call output_item either.
        bad_items = [
            e
            for e in events
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "function_call"
        ]
        assert bad_items == [], (
            "tool_search must NOT be wrapped as a function_call item — "
            "codex's dispatcher would silently abort it."
        )

    @pytest.mark.asyncio
    async def test_invalid_json_arguments_falls_back_to_empty_dict(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call_bad",
                        name="tool_search",
                        arguments="not json",
                        kind="tool_search",
                    ),
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )
        done = next(
            e
            for e in events
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "tool_search_call"
        )
        assert done["item"]["arguments"] == {}


class TestNonStreamingToolCallWireShape:
    """Non-streaming output must use the same per-kind item types as streaming."""

    @pytest.mark.asyncio
    async def test_non_stream_response_maps_tool_call_kinds(self, monkeypatch):
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: _NonStreamStubRouter(
                InternalResponsesResponse(
                    content="",
                    model="github-copilot/gpt-5",
                    tool_calls=[
                        ResponsesToolCall(
                            call_id="call_fn",
                            name="get_weather",
                            arguments='{"location": "NYC"}',
                            kind="function",
                        ),
                        ResponsesToolCall(
                            call_id="call_custom",
                            name="apply_patch",
                            arguments="*** Begin Patch\n*** End Patch",
                            kind="custom",
                        ),
                        ResponsesToolCall(
                            call_id="call_search",
                            name="tool_search",
                            arguments='{"query": "routes", "limit": 3}',
                            kind="tool_search",
                        ),
                    ],
                )
            ),
        )

        response = await create_response(
            ResponsesRequest(
                model="github-copilot/gpt-5",
                input="hi",
                stream=False,
            )
        )

        output = [
            item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
            for item in response.output
        ]
        by_call_id = {item["call_id"]: item for item in output if "call_id" in item}
        assert by_call_id["call_fn"]["type"] == "function_call"
        assert by_call_id["call_fn"]["arguments"] == '{"location": "NYC"}'
        assert by_call_id["call_custom"] == {
            "type": "custom_tool_call",
            "id": by_call_id["call_custom"]["id"],
            "call_id": "call_custom",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
            "status": "completed",
        }
        assert by_call_id["call_search"] == {
            "type": "tool_search_call",
            "call_id": "call_search",
            "execution": "client",
            "status": "completed",
            "arguments": {"query": "routes", "limit": 3},
        }


class TestNativeNonStreamingTerminalStatus:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "details", "error"),
        [
            (ResponseStatus.COMPLETED, None, None),
            (
                ResponseStatus.INCOMPLETE,
                {"reason": "max_output_tokens", "vendor": {"limit": 3}},
                None,
            ),
            (
                ResponseStatus.INCOMPLETE,
                {"reason": "content_filter", "vendor": "safety"},
                None,
            ),
            (
                ResponseStatus.FAILED,
                None,
                TerminalError(code="upstream_failed", message="boom"),
            ),
            (ResponseStatus.CANCELLED, None, None),
        ],
    )
    async def test_preserves_terminal_status_details_error_output_and_usage(
        self, monkeypatch, status, details, error
    ):
        usage = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        internal = InternalResponsesResponse(
            content="partial",
            model="github-copilot/gpt-5",
            usage=usage,
            terminal_outcome=TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=status,
                incomplete_details=details,
                error=error,
            ),
        )
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: _NonStreamStubRouter(internal),
        )

        response = await create_response(
            ResponsesRequest(model="github-copilot/gpt-5", input="hi", stream=False)
        )

        assert response.status == status.value
        assert response.incomplete_details == details
        assert response.error == (
            {"code": error.code, "message": error.message} if error is not None else None
        )
        assert response.usage is not None
        assert response.usage.model_dump(exclude_none=True) == usage
        assert len(response.output) == 1
        output_item = response.output[0]
        assert isinstance(output_item, dict)
        assert output_item["content"][0]["text"] == "partial"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "outcome",
        [
            TerminalOutcome(
                transport=TransportTermination.UNEXPECTED_EOF,
                response_status=ResponseStatus.UNKNOWN,
                error=TerminalError(code="unexpected_eof", message="ended early"),
            ),
            TerminalOutcome(
                transport=TransportTermination.EXCEPTION,
                response_status=ResponseStatus.COMPLETED,
            ),
            TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.COMPLETED,
                error=TerminalError(code="conflict", message="must not coexist"),
            ),
        ],
        ids=["unexpected-eof", "illegal-exception-completed", "completed-with-error"],
    )
    async def test_invalid_provider_terminal_metadata_is_protocol_failed_response(
        self, monkeypatch, outcome
    ):
        internal = InternalResponsesResponse(
            content="partial",
            model="github-copilot/gpt-5",
            terminal_outcome=outcome,
        )
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: _NonStreamStubRouter(internal),
        )

        response = await create_response(
            ResponsesRequest(model="github-copilot/gpt-5", input="hi", stream=False)
        )

        assert response.status == "failed"
        assert response.incomplete_details is None
        assert response.error is not None
        assert response.error["code"] == "upstream_protocol_error"

    @pytest.mark.asyncio
    async def test_explicit_failed_provider_error_is_preserved(self, monkeypatch):
        internal = InternalResponsesResponse(
            content="partial",
            model="github-copilot/gpt-5",
            terminal_outcome=TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.FAILED,
                error=TerminalError(code="quota_exhausted", message="safe failure"),
            ),
        )
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: _NonStreamStubRouter(internal),
        )

        response = await create_response(
            ResponsesRequest(model="github-copilot/gpt-5", input="hi", stream=False)
        )

        assert response.status == "failed"
        assert response.error == {"code": "quota_exhausted", "message": "safe failure"}


class TestNativeStreamingTerminalStatus:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "details", "error", "event_type"),
        [
            (ResponseStatus.COMPLETED, None, None, "response.completed"),
            (
                ResponseStatus.INCOMPLETE,
                {"reason": "max_output_tokens", "vendor": {"limit": 3}},
                None,
                "response.incomplete",
            ),
            (
                ResponseStatus.INCOMPLETE,
                {"reason": "content_filter", "vendor": "safety"},
                None,
                "response.incomplete",
            ),
            (
                ResponseStatus.FAILED,
                None,
                TerminalError(code="upstream_failed", message="boom"),
                "response.failed",
            ),
            (ResponseStatus.FAILED, None, None, "response.failed"),
            (ResponseStatus.CANCELLED, None, None, "response.failed"),
        ],
    )
    async def test_encodes_canonical_status_without_manufacturing_success(
        self, status, details, error, event_type
    ):
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=status,
            incomplete_details=details,
            error=error,
        )

        events = await _drive(
            [
                ResponsesStreamChunk(content="partial"),
                ResponsesStreamChunk(
                    content="",
                    usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                    terminal_outcome=outcome,
                ),
            ]
        )

        terminal_events = [
            event
            for event in events
            if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}
        ]
        assert len(terminal_events) == 1
        terminal = terminal_events[0]
        assert terminal["type"] == event_type
        assert terminal["response"]["status"] == status.value
        assert terminal["response"]["incomplete_details"] == details
        assert terminal["response"]["error"] == (
            {"code": error.code, "message": error.message} if error is not None else None
        )
        assert terminal["response"]["output"][0]["content"][0]["text"] == "partial"
        assert terminal["response"]["usage"]["input_tokens"] == 2
        assert terminal["response"]["usage"]["output_tokens"] == 1
        assert terminal["response"]["usage"]["total_tokens"] == 3


class TestResponsesStreamUsageWireShape:
    """Responses streaming should preserve upstream usage detail fields."""

    @pytest.mark.asyncio
    async def test_completed_event_includes_usage_details(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="hello"),
                ResponsesStreamChunk(
                    content="",
                    finish_reason="stop",
                    usage={
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "input_tokens_details": {"cached_tokens": 60},
                        "output_tokens_details": {"reasoning_tokens": 9},
                    },
                ),
            ]
        )

        completed = next(e for e in events if e.get("type") == "response.completed")
        assert completed["response"]["usage"] == {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 60},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 9},
            "total_tokens": 120,
        }


class TestResponsesStreamItemStateIsolation:
    """Each interleaved Responses output item owns its accumulated state."""

    @staticmethod
    def _done_items(events: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
        return [
            (event["output_index"], event["item"])
            for event in events
            if event.get("type") == "response.output_item.done"
        ]

    @staticmethod
    def _completed_output(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        completed = next(event for event in events if event.get("type") == "response.completed")
        return completed["response"]["output"]

    @pytest.mark.asyncio
    async def test_text_reasoning_text_keeps_messages_independent(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="A"),
                ResponsesStreamChunk(content="", thinking="R", thinking_id="rs-test"),
                ResponsesStreamChunk(content="B"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        done_items = self._done_items(events)
        assert [index for index, _ in done_items] == [0, 1, 2]
        assert [item["type"] for _, item in done_items] == ["message", "reasoning", "message"]
        assert done_items[0][1]["content"][0]["text"] == "A"
        assert done_items[1][1]["summary"][0]["text"] == "R"
        assert done_items[2][1]["content"][0]["text"] == "B"

        output = self._completed_output(events)
        assert [item["type"] for item in output] == ["message", "reasoning", "message"]
        assert output[0]["content"][0]["text"] == "A"
        assert output[1]["summary"][0]["text"] == "R"
        assert output[2]["content"][0]["text"] == "B"

    @pytest.mark.asyncio
    async def test_text_tool_text_keeps_messages_independent(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="A"),
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call-state-isolation",
                        name="lookup",
                        arguments='{"query": "x"}',
                        kind="function",
                    ),
                ),
                ResponsesStreamChunk(content="B"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        done_items = self._done_items(events)
        assert [index for index, _ in done_items] == [0, 1, 2]
        assert [item["type"] for _, item in done_items] == [
            "message",
            "function_call",
            "message",
        ]
        assert done_items[0][1]["content"][0]["text"] == "A"
        assert done_items[2][1]["content"][0]["text"] == "B"

        output = self._completed_output(events)
        assert [item["type"] for item in output] == ["message", "function_call", "message"]
        assert output[0]["content"][0]["text"] == "A"
        assert output[2]["content"][0]["text"] == "B"

    @pytest.mark.asyncio
    async def test_reasoning_then_text_uses_separate_output_items(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", thinking="R", thinking_id="rs-test"),
                ResponsesStreamChunk(content="A"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        done_items = self._done_items(events)
        assert [index for index, _ in done_items] == [0, 1]
        assert [item["type"] for _, item in done_items] == ["reasoning", "message"]
        assert done_items[0][1]["summary"][0]["text"] == "R"
        assert done_items[1][1]["content"][0]["text"] == "A"
        assert self._completed_output(events) == [item for _, item in done_items]

    @pytest.mark.asyncio
    async def test_consecutive_text_deltas_share_one_message_item(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="A"),
                ResponsesStreamChunk(content="B"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        done_items = self._done_items(events)
        assert len(done_items) == 1
        assert done_items[0][0] == 0
        assert done_items[0][1]["type"] == "message"
        assert done_items[0][1]["content"][0]["text"] == "AB"
        assert self._completed_output(events) == [done_items[0][1]]


class TestFunctionCallWireShape:
    """Regular function calls must still emit the legacy function_call shape."""

    @pytest.mark.asyncio
    async def test_function_call_emits_function_call_item(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call_fn",
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                        kind="function",
                    ),
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        done = [
            e
            for e in events
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "function_call"
        ]
        assert len(done) == 1
        assert done[0]["item"]["name"] == "get_weather"
        assert done[0]["item"]["arguments"] == '{"location": "NYC"}'


class TestReasoningRoundTrip:
    """The reasoning output_item must preserve the upstream (id, blob) pair.

    Copilot signs ``encrypted_content`` against the upstream reasoning item
    id (e.g. ``rs_…``). When Codex round-trips the reasoning item back on
    the next turn, the (id, blob) pair has to match what Copilot emitted —
    otherwise the next request 400s with ``Encrypted content could not be
    decrypted``. The route MUST use ``chunk.thinking_id`` as the item id,
    NOT a locally-generated ``rs-…`` slug.
    """

    @pytest.mark.asyncio
    async def test_upstream_reasoning_id_used_on_output_item(self):
        upstream_id = "rs_upstream_xyz_12345"
        encrypted_blob = "ENC_BLOB_" + "Z" * 200
        events = await _drive(
            [
                # First reasoning delta — carries upstream id.
                ResponsesStreamChunk(content="", thinking="planning...", thinking_id=upstream_id),
                # Final reasoning chunk — id + encrypted blob from
                # output_item.done in the upstream stream.
                ResponsesStreamChunk(
                    content="",
                    thinking_id=upstream_id,
                    thinking_signature=encrypted_blob,
                ),
                ResponsesStreamChunk(content="hello", finish_reason="stop"),
            ]
        )

        # output_item.added for reasoning must carry the upstream id, not a
        # locally-generated ``rs-…`` slug.
        added_reasoning = [
            e
            for e in events
            if e.get("type") == "response.output_item.added"
            and e.get("item", {}).get("type") == "reasoning"
        ]
        assert len(added_reasoning) == 1
        assert added_reasoning[0]["item"]["id"] == upstream_id

        # output_item.done must pair the upstream id with the encrypted blob.
        done_reasoning = [
            e
            for e in events
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "reasoning"
        ]
        assert len(done_reasoning) == 1
        item = done_reasoning[0]["item"]
        assert item["id"] == upstream_id, (
            f"reasoning item id must be the upstream id, got {item['id']!r} "
            f"(this would 400 the next turn with 'Encrypted content could not be decrypted')"
        )
        assert item["encrypted_content"] == encrypted_blob

    @pytest.mark.asyncio
    async def test_reasoning_summary_text_events_use_upstream_id(self):
        # The summary_text.delta / .done events must reference the same
        # upstream id (codex correlates them by item_id).
        upstream_id = "rs_upstream_abc"
        events = await _drive(
            [
                ResponsesStreamChunk(content="", thinking="thinking...", thinking_id=upstream_id),
                ResponsesStreamChunk(
                    content="",
                    thinking_id=upstream_id,
                    thinking_signature="BLOB",
                ),
                ResponsesStreamChunk(content="ok", finish_reason="stop"),
            ]
        )

        for evt in events:
            if evt.get("type", "").startswith("response.reasoning_summary"):
                assert evt.get("item_id") == upstream_id, (
                    f"{evt['type']} should reference upstream id, got {evt.get('item_id')!r}"
                )


class TestFunctionCallNamespaceWireShape:
    """Namespaced function_calls must emit `namespace` field downstream
    so Codex can echo it back next turn — Copilot 400s otherwise."""

    @pytest.mark.asyncio
    async def test_namespace_emitted_in_output_item(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call_kusto_1",
                        name="execute_query",
                        arguments='{"query":"x"}',
                        kind="function",
                        namespace="kusto",
                    ),
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )
        done = [
            e
            for e in events
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "function_call"
        ]
        assert len(done) == 1
        assert done[0]["item"]["namespace"] == "kusto"

        added = [
            e
            for e in events
            if e.get("type") == "response.output_item.added"
            and e.get("item", {}).get("type") == "function_call"
        ]
        assert len(added) == 1
        assert added[0]["item"]["namespace"] == "kusto"

    @pytest.mark.asyncio
    async def test_no_namespace_no_field_in_output(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="c1",
                        name="weather",
                        arguments="{}",
                        kind="function",
                    ),
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )
        done = [
            e
            for e in events
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "function_call"
        ]
        assert "namespace" not in done[0]["item"]


class TestCustomToolCallWireShape:
    """Custom tools (apply_patch) must still emit custom_tool_call shape."""

    @pytest.mark.asyncio
    async def test_custom_tool_emits_custom_tool_call_item(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call_patch",
                        name="apply_patch",
                        arguments="*** Begin Patch\n*** End Patch",
                        kind="custom",
                    ),
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )
        done = [
            e
            for e in events
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "custom_tool_call"
        ]
        assert len(done) == 1
        assert done[0]["item"]["name"] == "apply_patch"
        assert done[0]["item"]["input"] == "*** Begin Patch\n*** End Patch"
