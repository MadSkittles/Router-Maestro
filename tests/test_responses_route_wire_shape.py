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
from router_maestro.providers.base import ResponsesStreamChunk, ResponsesToolCall
from router_maestro.server.routes.responses import stream_response


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
