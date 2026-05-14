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
    async def test_upstream_reasoning_id_threaded_through_chunks(self):
        # The upstream reasoning item id (e.g. ``rs_…``) must surface as
        # ``thinking_id`` so the route can use it as the reasoning item's
        # id. Copilot signs ``encrypted_content`` against this id; pairing
        # the blob with a locally-generated id 400s the next turn with
        # ``Encrypted content could not be decrypted``.
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
        # The upstream id surfaces on the first delta and again on the
        # output_item.done chunk — both must equal the upstream id (never
        # a locally-generated ``rs-…``).
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
