"""Tests for the /responses route's wire-shape emission for tool calls.

The route translates internal ``ResponsesToolCall`` objects into the SSE
event sequence that downstream OpenAI-API-compatible clients (Codex CLI in
particular) consume. Each ``kind`` of tool call has its own wire shape; if
the wire shape is wrong, codex's tool dispatcher silently aborts the call
(this is exactly how v0.3.5/v0.3.6 broke ``tool_search``).

These tests pin each shape down end-to-end through ``stream_response``.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.providers.base import (
    ProviderError,
    ProviderFailureKind,
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
)
from router_maestro.providers.base import (
    ResponsesResponse as InternalResponsesResponse,
)
from router_maestro.server.routes.responses import (
    _IndexedOutputScheduler,
    _StreamMessageState,
    create_response,
    stream_response,
)
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

    def __init__(
        self,
        response: InternalResponsesResponse,
        provider_name: str = "github-copilot",
    ):
        self._response = response
        self._provider_name = provider_name

    async def responses_completion(
        self, request: InternalResponsesRequest, fallback: bool = True
    ) -> tuple[InternalResponsesResponse, str]:
        return self._response, self._provider_name


class _CapturingResponsesRouter:
    def __init__(self):
        self.request = None

    async def responses_completion(self, request, fallback: bool = True):
        self.request = request
        return InternalResponsesResponse(content="ok", model=request.model), "test-provider"


class _CopilotResponsesPreflightRouter:
    def __init__(self):
        from unittest.mock import AsyncMock

        from router_maestro.providers.copilot import CopilotProvider

        self.provider = CopilotProvider()
        self.upstream = AsyncMock(side_effect=AssertionError("upstream must not run"))
        self.request = None
        self.nonstream_calls = 0
        self.prepare_calls = 0

    async def responses_completion(self, request, fallback: bool = True):
        self.request = request
        self.nonstream_calls += 1
        self.provider.validate_responses_request(request)
        await self.upstream()

    async def prepare_responses_completion_stream(self, request, fallback: bool = True):
        self.request = request
        self.prepare_calls += 1
        self.provider.validate_responses_request(request)
        await self.upstream()


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


class TestNonStreamingResponseModelIdentity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("upstream_model", ["shared-model", "first/shared-model"])
    async def test_qualifies_actual_provider_exactly_once(self, monkeypatch, upstream_model):
        internal = InternalResponsesResponse(
            content="ok",
            model=upstream_model,
        )
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: _NonStreamStubRouter(internal, provider_name="first"),
        )

        response = await create_response(
            ResponsesRequest(model="shared-model", input="hi", stream=False)
        )

        assert response.model == "first/shared-model"


class TestResponsesTemperaturePresence:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("temperature", "expected"),
        [
            pytest.param(None, None, id="omitted"),
            pytest.param(1.0, 1.0, id="explicit-default"),
            pytest.param(0.4, 0.4, id="explicit-custom"),
        ],
    )
    async def test_route_preserves_temperature_presence(self, monkeypatch, temperature, expected):
        capturing = _CapturingResponsesRouter()
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: capturing,
        )
        request_kwargs = {"model": "model", "input": "hi", "stream": False}
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        response = await create_response(ResponsesRequest(**request_kwargs))

        assert response.status == "completed"
        assert capturing.request.temperature == expected

    @pytest.mark.asyncio
    async def test_route_preserves_minimal_reasoning_effort(self, monkeypatch):
        capturing = _CapturingResponsesRouter()
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: capturing,
        )

        response = await create_response(
            ResponsesRequest(
                model="model",
                input="hi",
                stream=False,
                reasoning={"effort": "minimal"},
            )
        )

        assert response.status == "completed"
        assert capturing.request.reasoning_effort == "minimal"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    async def test_copilot_rejects_explicit_temperature_before_upstream(
        self,
        monkeypatch,
        stream,
    ):
        capturing = _CopilotResponsesPreflightRouter()
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: capturing,
        )

        response = await create_response(
            ResponsesRequest(
                model="gpt-5.4",
                input="hi",
                stream=stream,
                temperature=0.4,
            )
        )

        assert response.status_code == 400
        assert response.body
        body = json.loads(response.body)
        assert body["error"]["param"] == "temperature"
        capturing.upstream.assert_not_awaited()
        assert capturing.nonstream_calls == (0 if stream else 1)
        assert capturing.prepare_calls == (1 if stream else 0)


class TestNativeRefusalWireShape:
    @pytest.mark.asyncio
    async def test_nonstream_refusal_is_not_retyped_as_output_text(self, monkeypatch):
        internal = InternalResponsesResponse(
            content="",
            model="github-copilot/gpt-5",
            refusal="I cannot help",
        )
        monkeypatch.setattr(
            "router_maestro.server.routes.responses.get_router",
            lambda: _NonStreamStubRouter(internal),
        )

        response = await create_response(
            ResponsesRequest(model="github-copilot/gpt-5", input="hi", stream=False)
        )

        item = response.output[0]
        assert isinstance(item, dict)
        assert item["content"] == [{"type": "refusal", "refusal": "I cannot help"}]

    @pytest.mark.asyncio
    async def test_stream_refusal_uses_refusal_events_and_part(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", refusal="I cannot "),
                ResponsesStreamChunk(content="", refusal="help"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        assert [e["delta"] for e in events if e.get("type") == "response.refusal.delta"] == [
            "I cannot ",
            "help",
        ]
        assert not any(e.get("type") == "response.output_text.delta" for e in events)
        completed = next(e for e in events if e.get("type") == "response.completed")
        assert completed["response"]["output"][0]["content"] == [
            {"type": "refusal", "refusal": "I cannot help"}
        ]

    @pytest.mark.asyncio
    async def test_stream_text_refusal_text_keeps_typed_parts_isolated(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="before"),
                ResponsesStreamChunk(content="", refusal="denied"),
                ResponsesStreamChunk(content="after"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        deltas = [
            (event["type"], event["content_index"], event["delta"])
            for event in events
            if event.get("type") in {"response.output_text.delta", "response.refusal.delta"}
        ]
        assert deltas == [
            ("response.output_text.delta", 0, "before"),
            ("response.refusal.delta", 1, "denied"),
            ("response.output_text.delta", 2, "after"),
        ]

        completed = next(event for event in events if event.get("type") == "response.completed")
        assert completed["response"]["output"] == [
            {
                "type": "message",
                "id": completed["response"]["output"][0]["id"],
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "before", "annotations": []},
                    {"type": "refusal", "refusal": "denied"},
                    {"type": "output_text", "text": "after", "annotations": []},
                ],
                "status": "completed",
            }
        ]

    @pytest.mark.asyncio
    async def test_stream_refusal_text_keeps_accumulators_separate(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", refusal="no"),
                ResponsesStreamChunk(content="yes"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        completed = next(event for event in events if event.get("type") == "response.completed")
        assert completed["response"]["output"][0]["content"] == [
            {"type": "refusal", "refusal": "no"},
            {"type": "output_text", "text": "yes", "annotations": []},
        ]

    @pytest.mark.asyncio
    async def test_stream_refusal_then_reasoning_uses_distinct_output_items(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", refusal="no"),
                ResponsesStreamChunk(content="", thinking="because", thinking_id="rs-1"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        completed = next(event for event in events if event.get("type") == "response.completed")
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["message", "reasoning"]
        assert output[0]["content"] == [{"type": "refusal", "refusal": "no"}]
        assert output[1] == {
            "type": "reasoning",
            "id": "rs-1",
            "summary": [{"type": "summary_text", "text": "because"}],
        }

    @pytest.mark.asyncio
    async def test_stream_refusal_then_tool_uses_distinct_output_items(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", refusal="no"),
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call-1",
                        name="lookup",
                        arguments='{"query":"safe"}',
                    ),
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        completed = next(event for event in events if event.get("type") == "response.completed")
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["message", "function_call"]
        assert output[0]["content"] == [{"type": "refusal", "refusal": "no"}]
        assert output[1]["call_id"] == "call-1"


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

    @pytest.mark.asyncio
    async def test_indexed_nested_completion_preserves_source_output_order(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    thinking="A",
                    thinking_id="rs-a",
                    output_index=0,
                    output_item_type="reasoning",
                ),
                ResponsesStreamChunk(
                    content="",
                    thinking="B",
                    thinking_id="rs-b",
                    output_index=1,
                    output_item_type="reasoning",
                ),
                ResponsesStreamChunk(
                    content="",
                    thinking="C",
                    thinking_id="rs-c",
                    output_index=2,
                    output_item_type="reasoning",
                ),
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-b",
                    thinking_signature="ENC_B",
                    output_index=1,
                    output_item_type="reasoning",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(
                    content="answer",
                    output_index=3,
                    content_index=0,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-a",
                    thinking_signature="ENC_A",
                    output_index=0,
                    output_item_type="reasoning",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-c",
                    thinking_signature="ENC_C",
                    output_index=2,
                    output_item_type="reasoning",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        output = self._completed_output(events)
        assert [item["type"] for item in output] == [
            "reasoning",
            "reasoning",
            "reasoning",
            "message",
        ]
        assert [item.get("id") for item in output[:3]] == ["rs-a", "rs-b", "rs-c"]
        assert output[3]["content"] == [
            {"type": "output_text", "text": "answer", "annotations": []}
        ]

    @pytest.mark.asyncio
    async def test_active_indexed_delta_is_visible_before_item_done(self):
        release_done = asyncio.Event()

        class GatedRouter:
            async def responses_completion_stream(self, request, fallback=True):
                async def chunks():
                    yield ResponsesStreamChunk(
                        content="first",
                        output_index=0,
                        content_index=0,
                        output_item_type="message",
                    )
                    await release_done.wait()
                    yield ResponsesStreamChunk(
                        content="",
                        output_index=0,
                        content_index=0,
                        output_item_type="message",
                        output_item_done=True,
                    )
                    yield ResponsesStreamChunk(content="", finish_reason="stop")

                return chunks(), "github-copilot"

        request = InternalResponsesRequest(model="gpt-5.5", input="hi", stream=True)
        response = stream_response(GatedRouter(), request, "req-gated", 0.0)  # type: ignore[arg-type]
        try:
            while True:
                raw_event = await asyncio.wait_for(response.__anext__(), 1.0)
                events = _parse_sse([raw_event])
                if any(
                    event.get("type") == "response.output_text.delta"
                    and event.get("delta") == "first"
                    for event in events
                ):
                    break
        finally:
            release_done.set()
            await response.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content_indices",
        [(0, 2), (0, -1)],
        ids=["gap", "backwards"],
    )
    async def test_invalid_content_index_provenance_is_typed_protocol_failure(
        self, content_indices
    ):
        first, second = content_indices
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="A",
                    output_index=0,
                    content_index=first,
                    output_item_type="message",
                ),
                ResponsesStreamChunk(
                    content="B",
                    output_index=0,
                    content_index=second,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["error"]["code"] == "upstream_protocol_error"
        assert failed["response"]["error"]["message"] in {
            "Responses message content_index contains a gap",
            "Responses message content_index moved backwards",
        }
        assert "Internal server error" not in failed["response"]["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content_index", [-1, 2], ids=["negative", "initial-gap"])
    async def test_invalid_initial_content_index_is_typed_protocol_failure(self, content_index):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="answer",
                    output_index=0,
                    content_index=content_index,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["error"] == {
            "code": "upstream_protocol_error",
            "message": "Responses message content_index must start at zero",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("summary_index", [-1, 2], ids=["negative", "initial-gap"])
    async def test_invalid_initial_reasoning_summary_index_is_typed_protocol_failure(
        self, summary_index
    ):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    thinking="thought",
                    thinking_id="rs-invalid-summary-index",
                    reasoning_summary_index=summary_index,
                    output_index=0,
                    output_item_type="reasoning",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["error"] == {
            "code": "upstream_protocol_error",
            "message": "Responses reasoning summary_index must start at zero",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "legacy_first", [True, False], ids=["legacy-indexed", "indexed-legacy"]
    )
    async def test_mixed_indexed_and_unindexed_output_is_typed_protocol_failure(self, legacy_first):
        legacy = ResponsesStreamChunk(content="legacy")
        indexed = ResponsesStreamChunk(
            content="indexed",
            output_index=0,
            content_index=0,
            output_item_type="message",
            output_item_done=True,
        )
        payload = [legacy, indexed] if legacy_first else [indexed, legacy]

        events = await _drive([*payload, ResponsesStreamChunk(content="", finish_reason="stop")])

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["error"] == {
            "code": "upstream_protocol_error",
            "message": "Responses stream mixed indexed and unindexed output items",
        }

    def test_future_buffer_bucket_limit_is_typed_protocol_failure(self):
        scheduler = _IndexedOutputScheduler(max_future_buckets=1)
        scheduler.add(
            ResponsesStreamChunk(
                content="one",
                output_index=1,
                output_item_type="message",
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            scheduler.add(
                ResponsesStreamChunk(
                    content="two",
                    output_index=2,
                    output_item_type="message",
                )
            )

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502

    def test_future_buffer_payload_limit_is_typed_protocol_failure(self):
        scheduler = _IndexedOutputScheduler(max_future_payload_bytes=4)

        with pytest.raises(ProviderError) as exc_info:
            scheduler.add(
                ResponsesStreamChunk(
                    content="12345",
                    output_index=1,
                    output_item_type="message",
                )
            )

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502

    def test_future_buffer_chunk_limit_counts_empty_payloads(self):
        scheduler = _IndexedOutputScheduler(max_future_chunks=2)
        empty = ResponsesStreamChunk(
            content="",
            output_index=1,
            output_item_type="message",
        )
        assert scheduler.add(empty) == []
        assert scheduler.add(empty) == []

        with pytest.raises(ProviderError) as exc_info:
            scheduler.add(empty)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502

    def test_future_output_index_window_is_bounded(self):
        scheduler = _IndexedOutputScheduler(max_future_buckets=2)

        with pytest.raises(ProviderError) as exc_info:
            scheduler.add(
                ResponsesStreamChunk(
                    content="tiny",
                    output_index=10**9,
                    output_item_type="message",
                )
            )

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502

    def test_active_scheduler_does_not_retain_emitted_payload(self):
        scheduler = _IndexedOutputScheduler()

        for index in range(100):
            chunk = ResponsesStreamChunk(
                content=f"delta-{index}",
                output_index=0,
                output_item_type="message",
            )
            assert scheduler.add(chunk) == [chunk]
            assert scheduler.buckets[0].chunks == []
            assert scheduler.future_payload_bytes == 0

    def test_activating_future_bucket_releases_its_buffer_accounting(self):
        scheduler = _IndexedOutputScheduler()
        future = ResponsesStreamChunk(
            content="future",
            output_index=1,
            output_item_type="message",
        )
        assert scheduler.add(future) == []
        assert scheduler.future_payload_bytes > 0

        current_done = ResponsesStreamChunk(
            content="",
            output_index=0,
            output_item_type="reasoning",
            output_item_done=True,
        )
        assert scheduler.add(current_done) == [current_done, future]
        assert scheduler.next_output_index == 1
        assert scheduler.buckets[1].chunks == []
        assert scheduler.future_payload_bytes == 0

    def test_finalize_synthesizes_done_without_replaying_active_payload(self):
        scheduler = _IndexedOutputScheduler()
        delta = ResponsesStreamChunk(
            content="only-once",
            output_index=0,
            output_item_type="message",
        )
        assert scheduler.add(delta) == [delta]

        final = scheduler.finalize()

        assert len(final) == 1
        assert final[0].content == ""
        assert final[0].output_index == 0
        assert final[0].output_item_type == "message"
        assert final[0].output_item_done is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload_kind", ["text", "refusal", "tool"])
    async def test_indexed_eof_drains_every_item_and_usage(self, payload_kind):
        if payload_kind == "text":
            payload = ResponsesStreamChunk(
                content="answer",
                output_index=2,
                content_index=0,
                output_item_type="message",
            )
            expected_type = "message"
        elif payload_kind == "refusal":
            payload = ResponsesStreamChunk(
                content="",
                refusal="denied",
                output_index=2,
                content_index=0,
                output_item_type="message",
            )
            expected_type = "message"
        else:
            payload = ResponsesStreamChunk(
                content="",
                tool_call=ResponsesToolCall(
                    call_id="call-eof",
                    name="lookup",
                    arguments="{}",
                ),
                output_index=2,
                output_item_type="function_call",
            )
            expected_type = "function_call"

        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    thinking="one",
                    thinking_id="rs-one",
                    output_index=0,
                    output_item_type="reasoning",
                ),
                ResponsesStreamChunk(
                    content="",
                    thinking="two",
                    thinking_id="rs-two",
                    output_index=1,
                    output_item_type="reasoning",
                ),
                payload,
                ResponsesStreamChunk(
                    content="",
                    usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
                ),
            ]
        )

        terminal = next(event for event in events if event.get("type") == "response.incomplete")
        output = terminal["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", "reasoning", expected_type]
        assert [item["summary"][0]["text"] for item in output[:2]] == ["one", "two"]
        assert terminal["response"]["usage"] == {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 7,
        }

    @pytest.mark.asyncio
    async def test_nonmonotonic_index_arrival_is_emitted_in_source_order(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="later",
                    output_index=1,
                    content_index=0,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(
                    content="",
                    thinking="first",
                    thinking_id="rs-first",
                    output_index=0,
                    output_item_type="reasoning",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        output = self._completed_output(events)
        assert [item["type"] for item in output] == ["reasoning", "message"]
        assert output[0]["summary"][0]["text"] == "first"
        assert output[1]["content"][0]["text"] == "later"

    @pytest.mark.asyncio
    async def test_scheduler_feeds_each_original_chunk_to_pipeline_once(self, monkeypatch):
        class PipelineSpy:
            instance = None

            def __init__(self):
                self.fed = []
                PipelineSpy.instance = self

            @classmethod
            def create(cls, **_kwargs):
                return cls()

            def feed_stream(self, chunk):
                self.fed.append(chunk)
                return None

            def finish(self, **_kwargs):
                return None

        monkeypatch.setattr("router_maestro.pipeline.RequestPipeline", PipelineSpy)
        chunks = [
            ResponsesStreamChunk(
                content="later",
                output_index=1,
                content_index=0,
                output_item_type="message",
                output_item_done=True,
            ),
            ResponsesStreamChunk(
                content="",
                thinking="first",
                thinking_id="rs-first",
                output_index=0,
                output_item_type="reasoning",
                output_item_done=True,
            ),
            ResponsesStreamChunk(content="", finish_reason="stop"),
        ]

        await _drive(chunks)

        assert PipelineSpy.instance is not None
        assert PipelineSpy.instance.fed == chunks

    @pytest.mark.asyncio
    async def test_provenance_marker_scheduler_requeue_feeds_pipeline_once(self, monkeypatch):
        class PipelineSpy:
            instance = None

            def __init__(self):
                self.fed = []
                PipelineSpy.instance = self

            @classmethod
            def create(cls, **_kwargs):
                return cls()

            def feed_stream(self, chunk):
                self.fed.append(chunk)
                return None

            def finish(self, **_kwargs):
                return None

        monkeypatch.setattr("router_maestro.pipeline.RequestPipeline", PipelineSpy)
        chunks = [
            ResponsesStreamChunk(
                content="first",
                output_index=0,
                content_index=0,
                output_item_type="message",
            ),
            ResponsesStreamChunk(
                content="",
                output_index=1,
                content_index=0,
                provenance_only=True,
                output_item_type="message",
            ),
            ResponsesStreamChunk(content="", finish_reason="stop"),
        ]

        await _drive(chunks)

        assert PipelineSpy.instance is not None
        assert PipelineSpy.instance.fed == chunks

    @pytest.mark.asyncio
    @pytest.mark.parametrize("marker_kind", ["text", "refusal"])
    @pytest.mark.parametrize("synthetic_done", [False, True], ids=["explicit", "synthetic"])
    @pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "eof"])
    @pytest.mark.parametrize("next_index", [0, 1], ids=["valid-zero", "invalid-one"])
    async def test_empty_message_marker_done_resets_content_provenance(
        self, marker_kind, synthetic_done, terminal, next_index
    ):
        marker = ResponsesStreamChunk(
            content="",
            refusal=None,
            output_index=0,
            content_index=0,
            provenance_only=True,
            output_item_type="message",
            output_item_done=not synthetic_done,
        )
        chunks = [
            marker,
            ResponsesStreamChunk(
                content="next" if marker_kind == "text" else "",
                refusal="next" if marker_kind == "refusal" else None,
                output_index=1,
                content_index=next_index,
                output_item_type="message",
                output_item_done=True,
            ),
        ]
        if terminal:
            chunks.append(ResponsesStreamChunk(content="", finish_reason="stop"))

        events = await _drive(chunks)

        if next_index == 1:
            failed = next(event for event in events if event.get("type") == "response.failed")
            assert failed["response"]["error"] == {
                "code": "upstream_protocol_error",
                "message": "Responses message content_index must start at zero",
            }
            return
        terminal_event = next(
            event
            for event in events
            if event["type"] in {"response.completed", "response.incomplete"}
        )
        assert len(terminal_event["response"]["output"]) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("marker_type", ["message", "refusal", "reasoning"])
    @pytest.mark.parametrize("with_done", [False, True], ids=["open", "done"])
    @pytest.mark.parametrize(
        ("control", "expected_type", "expected_status"),
        [
            ("finish", "response.completed", "completed"),
            ("incomplete", "response.incomplete", "incomplete"),
            ("failed", "response.failed", "failed"),
        ],
    )
    async def test_provenance_marker_preserves_terminal_control(
        self, marker_type, with_done, control, expected_type, expected_status
    ):
        if control == "finish":
            finish_reason = "stop"
            outcome = None
        else:
            status = ResponseStatus.INCOMPLETE if control == "incomplete" else ResponseStatus.FAILED
            finish_reason = None
            outcome = TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=status,
                incomplete_details={"reason": "max_output_tokens"}
                if status is ResponseStatus.INCOMPLETE
                else None,
                error=TerminalError(code="upstream_failed", message="failed")
                if status is ResponseStatus.FAILED
                else None,
            )
        chunk = ResponsesStreamChunk(
            content="",
            refusal=None,
            thinking_id="rs-marker" if marker_type == "reasoning" else None,
            output_index=0,
            content_index=0 if marker_type != "reasoning" else None,
            reasoning_summary_index=0 if marker_type == "reasoning" else None,
            provenance_only=True,
            output_item_type="reasoning" if marker_type == "reasoning" else "message",
            output_item_done=with_done,
            finish_reason=finish_reason,
            terminal_outcome=outcome,
            usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )

        events = await _drive([chunk])

        terminal = next(event for event in events if event.get("type") == expected_type)
        assert terminal["response"]["status"] == expected_status
        assert terminal["response"]["output"] == []
        assert terminal["response"]["usage"]["total_tokens"] == 3
        assert not any(event.get("type") == "response.incomplete" for event in events) or (
            expected_type == "response.incomplete"
        )

    @pytest.mark.asyncio
    async def test_future_provenance_marker_terminal_drains_active_bucket_first(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="active",
                    output_index=0,
                    content_index=0,
                    output_item_type="message",
                ),
                ResponsesStreamChunk(
                    content="",
                    output_index=1,
                    content_index=0,
                    provenance_only=True,
                    output_item_type="message",
                    finish_reason="stop",
                    usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                ),
            ]
        )

        completed = next(event for event in events if event.get("type") == "response.completed")
        assert completed["response"]["output"][0]["content"][0]["text"] == "active"
        assert completed["response"]["usage"]["total_tokens"] == 2

    def test_reasoning_state_keeps_one_authoritative_fragment_store(self):
        state = _StreamMessageState()

        state.append_reasoning("first", 0)
        state.append_reasoning("second", 0)
        state.append_reasoning("third", 1)

        assert state.reasoning_fragments == {0: ["first", "second"], 1: ["third"]}
        assert state.reasoning_emission_offsets == {}
        assert not hasattr(state, "reasoning_parts")
        assert not hasattr(state, "pending_reasoning_deltas")

    def test_reasoning_flush_visits_only_dirty_summary_indices(self):
        state = _StreamMessageState()
        state.bind_reasoning_id("rs-dirty")
        for summary_index in range(1000):
            state.append_reasoning(f"part-{summary_index}", summary_index)
        state.start_reasoning_events()
        state.flush_reasoning_scan_count = 0

        state.append_reasoning("new", 731)
        events = state.flush_reasoning_events()

        assert state.flush_reasoning_scan_count == 1
        assert len(events) == 1
        assert '"summary_index": 731' in events[0]
        assert state.dirty_reasoning_indices == []
        assert not hasattr(state, "dirty_reasoning_fragments")

    def test_reasoning_summary_index_validation_uses_constant_size_frontier(self):
        state = _StreamMessageState()

        for summary_index in range(50_000):
            state.bind_reasoning_summary_index(summary_index)

        assert state.max_reasoning_summary_index == 49_999
        assert not hasattr(state, "seen_reasoning_summary_indices")

    @pytest.mark.asyncio
    async def test_late_id_reasoning_over_runaway_limit_fails_before_completion(self, monkeypatch):
        from router_maestro.config.priorities import PrioritiesConfig
        from router_maestro.pipeline.request_pipeline import RequestPipeline
        from router_maestro.pipeline.runaway_guard import RunawayGuard

        class SmallPipelineFactory:
            instance = None

            @classmethod
            def create(cls, *, request_id, model, tool_names=None):
                cls.instance = RequestPipeline(
                    request_id=request_id,
                    guards=[RunawayGuard(max_bytes=5, max_deltas=50_000)],
                    leak_guard=None,
                    audit=None,
                    config=PrioritiesConfig(),
                )
                return cls.instance

        monkeypatch.setattr("router_maestro.pipeline.RequestPipeline", SmallPipelineFactory)

        events = await _drive(
            [
                ResponsesStreamChunk(content="", thinking="猫"),
                ResponsesStreamChunk(content="", thinking="猫"),
                ResponsesStreamChunk(content="", thinking_id="rs-too-late"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["status"] == "failed"
        assert not any(event.get("type") == "response.completed" for event in events)
        assert SmallPipelineFactory.instance is not None
        outcome = SmallPipelineFactory.instance._outcome
        assert outcome is not None
        assert outcome.response_status is ResponseStatus.FAILED
        assert outcome.error is not None
        assert outcome.error.code == "overloaded"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("future_kind", ["message", "refusal", "reasoning", "tool"])
    @pytest.mark.parametrize(
        ("status", "terminal_type"),
        [
            (ResponseStatus.COMPLETED, "response.completed"),
            (ResponseStatus.INCOMPLETE, "response.incomplete"),
            (ResponseStatus.FAILED, "response.failed"),
        ],
    )
    async def test_active_index_terminal_is_emitted_after_future_payload(
        self, future_kind, status, terminal_type, monkeypatch
    ):
        class PipelineSpy:
            instance = None

            def __init__(self):
                self.fed = []
                self.outcome = None
                PipelineSpy.instance = self

            @classmethod
            def create(cls, **_kwargs):
                return cls()

            def feed_stream(self, chunk):
                self.fed.append(chunk)
                return None

            def finish(self, *, wire_status, outcome, body_summary=None):
                self.outcome = outcome

        monkeypatch.setattr("router_maestro.pipeline.RequestPipeline", PipelineSpy)
        if future_kind == "message":
            future = ResponsesStreamChunk(
                content="future",
                output_index=1,
                content_index=0,
                output_item_type="message",
                output_item_done=True,
            )
        elif future_kind == "refusal":
            future = ResponsesStreamChunk(
                content="",
                refusal="future",
                output_index=1,
                content_index=0,
                output_item_type="message",
                output_item_done=True,
            )
        elif future_kind == "reasoning":
            future = ResponsesStreamChunk(
                content="",
                thinking="future",
                thinking_id="rs-future",
                reasoning_summary_index=0,
                output_index=1,
                output_item_type="reasoning",
                output_item_done=True,
            )
        else:
            future = ResponsesStreamChunk(
                content="",
                tool_call=ResponsesToolCall(call_id="call-future", name="lookup", arguments="{}"),
                output_index=1,
                output_item_type="function_call",
                output_item_done=True,
            )
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=status,
            incomplete_details={"reason": "max_output_tokens"}
            if status is ResponseStatus.INCOMPLETE
            else None,
            error=TerminalError(code="failed", message="failed")
            if status is ResponseStatus.FAILED
            else None,
        )
        active_terminal = ResponsesStreamChunk(
            content="active",
            output_index=0,
            content_index=0,
            output_item_type="message",
            output_item_done=True,
            usage={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
            terminal_outcome=outcome,
        )
        chunks = [future, active_terminal]

        events = await _drive(chunks)

        terminal_position = next(
            i for i, event in enumerate(events) if event["type"] == terminal_type
        )
        done_positions = [
            i for i, event in enumerate(events) if event["type"] == "response.output_item.done"
        ]
        assert done_positions and max(done_positions) < terminal_position
        terminal = events[terminal_position]
        assert len(terminal["response"]["output"]) == 2
        assert terminal["response"]["usage"]["total_tokens"] == 4
        assert PipelineSpy.instance is not None
        assert PipelineSpy.instance.fed == chunks
        assert PipelineSpy.instance.outcome == outcome

    @pytest.mark.asyncio
    async def test_active_reasoning_blob_over_runaway_limit_fails_not_completes(self, monkeypatch):
        from router_maestro.config.priorities import PrioritiesConfig
        from router_maestro.pipeline.request_pipeline import RequestPipeline
        from router_maestro.pipeline.runaway_guard import RunawayGuard

        class SmallPipelineFactory:
            @classmethod
            def create(cls, *, request_id, model, tool_names=None):
                return RequestPipeline(
                    request_id=request_id,
                    guards=[RunawayGuard(max_bytes=5, max_deltas=50_000)],
                    leak_guard=None,
                    audit=None,
                    config=PrioritiesConfig(),
                )

        monkeypatch.setattr("router_maestro.pipeline.RequestPipeline", SmallPipelineFactory)
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-A",
                    thinking_signature="ENCODED",
                    output_index=0,
                    output_item_type="reasoning",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["status"] == "failed"
        assert not any(event.get("type") == "response.completed" for event in events)

    @pytest.mark.asyncio
    async def test_adjacent_indexed_messages_remain_distinct_items(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="A",
                    output_index=0,
                    content_index=0,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(
                    content="B",
                    output_index=1,
                    content_index=0,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        output = self._completed_output(events)
        assert [item["content"][0]["text"] for item in output] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_indexed_message_preserves_distinct_content_parts_of_same_type(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="A",
                    output_index=0,
                    content_index=0,
                    output_item_type="message",
                ),
                ResponsesStreamChunk(
                    content="B",
                    output_index=0,
                    content_index=1,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        output = self._completed_output(events)
        assert output[0]["content"] == [
            {"type": "output_text", "text": "A", "annotations": []},
            {"type": "output_text", "text": "B", "annotations": []},
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks",
        [
            [
                ResponsesStreamChunk(
                    content="A",
                    output_index=0,
                    content_index=0,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(
                    content="late",
                    output_index=0,
                    content_index=0,
                    output_item_type="message",
                ),
            ],
            [
                ResponsesStreamChunk(
                    content="A",
                    output_index=0,
                    content_index=0,
                    output_item_type="message",
                ),
                ResponsesStreamChunk(
                    content="B",
                    output_index=0,
                    content_index=0,
                    output_item_type="reasoning",
                ),
            ],
            [
                ResponsesStreamChunk(
                    content="B",
                    output_index=1,
                    content_index=0,
                    output_item_type="message",
                    output_item_done=True,
                )
            ],
        ],
        ids=["data-after-done", "conflicting-type", "source-gap"],
    )
    async def test_malformed_indexed_sequences_are_typed_protocol_failures(self, chunks):
        events = await _drive([*chunks, ResponsesStreamChunk(content="", finish_reason="stop")])

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["status"] == "failed"
        assert "Internal server error" not in failed["response"]["error"]["message"]
        assert not any(event.get("type") == "response.completed" for event in events)

    @pytest.mark.asyncio
    async def test_terminal_is_last_after_indexed_buckets_are_drained(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="later",
                    output_index=1,
                    content_index=0,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(
                    content="first",
                    output_index=0,
                    content_index=0,
                    output_item_type="message",
                    output_item_done=True,
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        assert events[-1]["type"] == "response.completed"
        assert [
            event["output_index"]
            for event in events
            if event.get("type") == "response.output_item.done"
        ] == [0, 1]


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

    @pytest.mark.asyncio
    async def test_reasoning_id_change_is_typed_protocol_failure(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", thinking="plan", thinking_id="rs-one"),
                ResponsesStreamChunk(content="", thinking_id="rs-two"),
            ]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["error"] == {
            "code": "upstream_protocol_error",
            "message": "Responses reasoning item changed identity",
        }

    @pytest.mark.asyncio
    async def test_reasoning_signature_without_id_is_typed_protocol_failure(self):
        events = await _drive(
            [ResponsesStreamChunk(content="", thinking_signature="ENC_WITHOUT_ID")]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["error"] == {
            "code": "upstream_protocol_error",
            "message": "Reasoning signature is missing its upstream item id",
        }

    @pytest.mark.asyncio
    async def test_reasoning_marker_id_must_match_visible_item(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", thinking="plan", thinking_id="rs-A"),
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-B",
                    reasoning_summary_index=0,
                    provenance_only=True,
                    output_item_type="reasoning",
                ),
            ]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["error"] == {
            "code": "upstream_protocol_error",
            "message": "Responses reasoning item changed identity",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("with_blob", [False, True], ids=["id-only", "late-blob"])
    async def test_pending_reasoning_marker_id_must_match_done_chunk(self, with_blob):
        done = ResponsesStreamChunk(
            content="",
            thinking_id="rs-B",
            thinking_signature="ENC_B" if with_blob else None,
            output_item_type="reasoning",
            output_item_done=True,
        )
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-A",
                    reasoning_summary_index=0,
                    provenance_only=True,
                    output_item_type="reasoning",
                ),
                done,
            ]
        )

        failed = next(event for event in events if event.get("type") == "response.failed")
        assert failed["response"]["error"] == {
            "code": "upstream_protocol_error",
            "message": "Responses reasoning item changed identity",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "eof"])
    @pytest.mark.parametrize("repeat_id", [False, True], ids=["no-id", "same-id"])
    async def test_pending_reasoning_marker_id_promotes_to_visible_item(self, terminal, repeat_id):
        chunks = [
            ResponsesStreamChunk(
                content="",
                thinking_id="rs-A",
                reasoning_summary_index=0,
                provenance_only=True,
                output_item_type="reasoning",
            ),
            ResponsesStreamChunk(
                content="",
                thinking="thought",
                thinking_id="rs-A" if repeat_id else None,
                reasoning_summary_index=0,
                output_item_type="reasoning",
            ),
        ]
        if terminal:
            chunks.append(ResponsesStreamChunk(content="", finish_reason="stop"))
        events = await _drive(chunks)

        terminal_event = next(
            event
            for event in events
            if event["type"] in {"response.completed", "response.incomplete"}
        )
        assert terminal_event["response"]["output"] == [
            {
                "type": "reasoning",
                "id": "rs-A",
                "summary": [{"type": "summary_text", "text": "thought"}],
            }
        ]

    @pytest.mark.asyncio
    async def test_late_upstream_id_is_used_for_every_buffered_reasoning_event(self):
        upstream_id = "rs_late_upstream"
        encrypted_blob = "ENC_LATE"
        events = await _drive(
            [
                ResponsesStreamChunk(content="", thinking="plan"),
                ResponsesStreamChunk(content="", thinking="ning"),
                ResponsesStreamChunk(
                    content="",
                    thinking_id=upstream_id,
                    thinking_signature=encrypted_blob,
                ),
                ResponsesStreamChunk(content="ok", finish_reason="stop"),
            ]
        )

        reasoning_events = [
            event
            for event in events
            if event.get("type")
            in {
                "response.output_item.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
                "response.output_item.done",
            }
            and (
                event.get("item", {}).get("type") == "reasoning"
                or event.get("type", "").startswith("response.reasoning_summary")
            )
        ]
        assert reasoning_events
        ids = [event.get("item_id", event.get("item", {}).get("id")) for event in reasoning_events]
        assert set(ids) == {upstream_id}
        assert [
            event["delta"]
            for event in reasoning_events
            if event["type"] == "response.reasoning_summary_text.delta"
        ] == ["plan", "ning"]
        done_item = next(
            event["item"]
            for event in reasoning_events
            if event["type"] == "response.output_item.done"
        )
        assert done_item["encrypted_content"] == encrypted_blob

    @pytest.mark.asyncio
    async def test_reasoning_without_upstream_id_gets_one_local_id_at_terminal(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", thinking="local"),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        added = next(
            event
            for event in events
            if event.get("type") == "response.output_item.added"
            and event.get("item", {}).get("type") == "reasoning"
        )
        local_id = added["item"]["id"]
        reasoning_ids = [
            event.get("item_id", event.get("item", {}).get("id"))
            for event in events
            if event.get("type", "").startswith("response.reasoning_summary")
            or (
                event.get("type") == "response.output_item.done"
                and event.get("item", {}).get("type") == "reasoning"
            )
        ]
        assert local_id.startswith("rs-")
        assert set(reasoning_ids) == {local_id}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("boundary_chunk", "boundary_type", "expected_value"),
        [
            pytest.param(ResponsesStreamChunk(content="answer"), "message", "answer", id="text"),
            pytest.param(
                ResponsesStreamChunk(content="", refusal="denied"),
                "message",
                "denied",
                id="refusal",
            ),
            pytest.param(
                ResponsesStreamChunk(
                    content="",
                    tool_call=ResponsesToolCall(
                        call_id="call-late-id",
                        name="lookup",
                        arguments='{"query":"x"}',
                    ),
                ),
                "function_call",
                "call-late-id",
                id="tool",
            ),
        ],
    )
    async def test_unidentified_reasoning_survives_boundary_until_identity_arrives(
        self, boundary_chunk, boundary_type, expected_value
    ):
        events = await _drive(
            [
                ResponsesStreamChunk(content="", thinking="plan"),
                boundary_chunk,
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-upstream-after-boundary",
                    thinking_signature="ENC_AFTER_BOUNDARY",
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        completed = next(event for event in events if event.get("type") == "response.completed")
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", boundary_type]
        assert output[0] == {
            "type": "reasoning",
            "id": "rs-upstream-after-boundary",
            "summary": [{"type": "summary_text", "text": "plan"}],
            "encrypted_content": "ENC_AFTER_BOUNDARY",
        }
        if boundary_chunk.content:
            assert output[1]["content"] == [
                {"type": "output_text", "text": expected_value, "annotations": []}
            ]
        elif boundary_chunk.refusal:
            assert output[1]["content"] == [{"type": "refusal", "refusal": expected_value}]
        else:
            assert output[1]["call_id"] == expected_value

    @pytest.mark.asyncio
    async def test_unidentified_reasoning_sharing_boundary_chunk_waits_for_identity(self):
        events = await _drive(
            [
                ResponsesStreamChunk(content="answer", thinking="plan"),
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-upstream-after-combined-chunk",
                    thinking_signature="ENC_AFTER_COMBINED_CHUNK",
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        completed = next(event for event in events if event.get("type") == "response.completed")
        assert completed["response"]["output"] == [
            {
                "type": "reasoning",
                "id": "rs-upstream-after-combined-chunk",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "encrypted_content": "ENC_AFTER_COMBINED_CHUNK",
            },
            {
                "type": "message",
                "id": completed["response"]["output"][1]["id"],
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                "status": "completed",
            },
        ]

    @pytest.mark.asyncio
    async def test_reasoning_with_known_id_waits_for_late_blob_across_boundary(self):
        events = await _drive(
            [
                ResponsesStreamChunk(
                    content="",
                    thinking="plan",
                    thinking_id="rs-known-before-boundary",
                ),
                ResponsesStreamChunk(content="answer"),
                ResponsesStreamChunk(
                    content="",
                    thinking_id="rs-known-before-boundary",
                    thinking_signature="ENC_AFTER_BOUNDARY",
                ),
                ResponsesStreamChunk(content="", finish_reason="stop"),
            ]
        )

        completed = next(event for event in events if event.get("type") == "response.completed")
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", "message"]
        assert output[0] == {
            "type": "reasoning",
            "id": "rs-known-before-boundary",
            "summary": [{"type": "summary_text", "text": "plan"}],
            "encrypted_content": "ENC_AFTER_BOUNDARY",
        }
        assert output[1]["content"] == [
            {"type": "output_text", "text": "answer", "annotations": []}
        ]


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
