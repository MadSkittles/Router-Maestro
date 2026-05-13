"""Tests for Copilot /responses streaming event handlers.

These tests exercise the `responses_completion_stream` parser against
synthetic SSE payloads produced via `httpx.MockTransport`. The goal is to
pin down handling of event types that have caused regressions in the field
(custom_tool_call from v0.3.4, tool_search_call from v0.3.5).
"""

import json
import logging

import httpx
import pytest

from router_maestro.providers import CopilotProvider
from router_maestro.providers.base import ResponsesRequest, ResponsesStreamChunk


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
    provider._get_headers = lambda: {"authorization": "Bearer test"}  # type: ignore[method-assign]
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
        assert tc.arguments == '{"query": "x"}'


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
