"""Cross-protocol invariants for streaming terminal outcomes."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from router_maestro.providers import ChatRequest, ChatStreamChunk, Message
from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.providers import base as provider_base
from router_maestro.providers.base import ProviderError, ResponsesStreamChunk
from router_maestro.server.routes import anthropic, chat, gemini, responses
from router_maestro.server.streaming import sse_streaming_response


@dataclass
class _FinishCall:
    wire_status: int | None
    outcome: Any
    body_summary: str | None


class _PipelineSpy:
    instances: list[_PipelineSpy] = []
    abort_reason: str | None = None

    def __init__(self, request_id: str, model: str, tool_names: set[str] | None = None):
        self.request_id = request_id
        self.model = model
        self.tool_names = tool_names
        self.finished: list[_FinishCall] = []
        self.__class__.instances.append(self)

    @classmethod
    def create(
        cls,
        request_id: str,
        model: str,
        tool_names: set[str] | None = None,
    ) -> _PipelineSpy:
        return cls(request_id=request_id, model=model, tool_names=tool_names)

    def feed_stream(self, chunk: Any) -> str | None:
        return self.abort_reason

    def check_invoke_at_finish(self) -> list[dict] | None:
        return None

    def finish(
        self,
        *,
        wire_status: int | None = None,
        outcome: Any = None,
        body_summary: str | None = None,
        status: int | None = None,
    ) -> None:
        # ``status`` keeps the spy able to characterize the pre-change API.
        self.finished.append(
            _FinishCall(
                wire_status=wire_status if wire_status is not None else status,
                outcome=outcome,
                body_summary=body_summary,
            )
        )


class _ChatRouter:
    def __init__(self, chunks: list[ChatStreamChunk], error: BaseException | None = None):
        self._chunks = chunks
        self._error = error

    async def chat_completion_stream(
        self, request: ChatRequest, fallback: bool = True
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        async def generate() -> AsyncIterator[ChatStreamChunk]:
            for chunk in self._chunks:
                yield chunk
            if self._error is not None:
                raise self._error

        return generate(), "stub"


class _ResponsesRouter:
    def __init__(self, chunks: list[ResponsesStreamChunk], error: BaseException | None = None):
        self._chunks = chunks
        self._error = error

    async def responses_completion_stream(
        self, request: InternalResponsesRequest, fallback: bool = True
    ) -> tuple[AsyncIterator[ResponsesStreamChunk], str]:
        async def generate() -> AsyncIterator[ResponsesStreamChunk]:
            for chunk in self._chunks:
                yield chunk
            if self._error is not None:
                raise self._error

        return generate(), "stub"


def _chat_request() -> ChatRequest:
    return ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )


def _responses_request() -> InternalResponsesRequest:
    return InternalResponsesRequest(model="github-copilot/gpt-5", input="hi", stream=True)


def _terminal_outcome(
    status: str,
    *,
    finish_reason: str | None = None,
    incomplete_details: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> Any:
    assert hasattr(provider_base, "TerminalOutcome"), "TerminalOutcome is not implemented"
    assert hasattr(provider_base, "TransportTermination")
    assert hasattr(provider_base, "ResponseStatus")
    terminal_error = None
    if error_code:
        assert hasattr(provider_base, "TerminalError")
        terminal_error = provider_base.TerminalError(
            code=error_code,
            message=error_message or error_code,
        )
    return provider_base.TerminalOutcome(
        transport=provider_base.TransportTermination.EXPLICIT_TERMINAL,
        response_status=provider_base.ResponseStatus(status),
        finish_reason=finish_reason,
        incomplete_details=incomplete_details,
        error=terminal_error,
    )


def _make_stream(
    protocol: str,
    *,
    terminal: str | None = None,
    canonical_outcome: Any = None,
    error: BaseException | None = None,
):
    if protocol == "responses":
        chunks = [ResponsesStreamChunk(content="hi")]
        if terminal is not None or canonical_outcome is not None:
            chunks.append(
                ResponsesStreamChunk(
                    content="",
                    finish_reason=terminal,
                    terminal_outcome=canonical_outcome,
                )
            )
        return responses.stream_response(
            _ResponsesRouter(chunks, error),
            _responses_request(),
            request_id="req-test",
            start_time=time.time(),
        )

    chunks = [ChatStreamChunk(content="hi")]
    if terminal is not None or canonical_outcome is not None:
        chunks.append(
            ChatStreamChunk(
                content="",
                finish_reason=terminal,
                terminal_outcome=canonical_outcome,
            )
        )
    router = _ChatRouter(chunks, error)
    request = _chat_request()
    if protocol == "chat":
        return chat.stream_response(router, request)
    if protocol == "anthropic":
        return anthropic.stream_response(router, request, "claude-sonnet-4", 1)
    if protocol == "gemini":
        return gemini._stream_response(router, request, "gemini-2.5-pro", 1)
    raise AssertionError(f"unknown protocol: {protocol}")


async def _collect(protocol: str, **kwargs: Any) -> tuple[list[str], int]:
    response = sse_streaming_response(_make_stream(protocol, **kwargs))
    events = [event async for event in response.body_iterator]
    return events, response.status_code


def _json_payloads(events: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def _event_types(events: list[str]) -> list[str]:
    return [
        line.removeprefix("event: ")
        for event in events
        for line in event.splitlines()
        if line.startswith("event: ")
    ]


def _assert_no_success_terminal(protocol: str, events: list[str]) -> None:
    if protocol == "chat":
        assert "data: [DONE]\n\n" not in events
        payloads = _json_payloads(events)
        assert not any(
            choice.get("finish_reason")
            for payload in payloads
            for choice in payload.get("choices", [])
        )
    elif protocol == "responses":
        assert "response.completed" not in _event_types(events)
    elif protocol == "anthropic":
        assert "message_stop" not in _event_types(events)
    else:
        assert not any(
            candidate.get("finishReason") == "STOP"
            for payload in _json_payloads(events)
            for candidate in payload.get("candidates", [])
        )


def _assert_error_terminal(protocol: str, events: list[str], expected: str) -> None:
    if protocol == "chat":
        errors = [payload["error"] for payload in _json_payloads(events) if "error" in payload]
        assert len(errors) == 1
        assert expected in json.dumps(errors[0])
    elif protocol == "responses":
        expected_type = "response.incomplete" if expected == "unexpected_eof" else "response.failed"
        assert _event_types(events).count(expected_type) == 1
        terminal = next(
            payload for payload in _json_payloads(events) if payload.get("type") == expected_type
        )
        if expected == "unexpected_eof":
            assert terminal["response"]["status"] == "incomplete"
            assert terminal["response"]["incomplete_details"]["reason"] == expected
        elif expected == "upstream cancelled":
            assert terminal["response"]["status"] == "cancelled"
        else:
            assert terminal["response"]["status"] == "failed"
    elif protocol == "anthropic":
        assert _event_types(events).count("error") == 1
        error = next(
            payload for payload in _json_payloads(events) if payload.get("type") == "error"
        )
        assert expected in json.dumps(error)
    else:
        errors = [payload["error"] for payload in _json_payloads(events) if "error" in payload]
        assert len(errors) == 1
        assert expected in json.dumps(errors[0])


def _assert_pipeline_outcome(
    transport: str,
    status: str,
    *,
    error_code: str | None = None,
) -> None:
    assert len(_PipelineSpy.instances) == 1
    assert len(_PipelineSpy.instances[0].finished) == 1
    call = _PipelineSpy.instances[0].finished[0]
    assert call.wire_status == 200
    assert call.outcome is not None
    assert call.outcome.transport.value == transport
    assert call.outcome.response_status.value == status
    if error_code is not None:
        assert call.outcome.error is not None
        assert call.outcome.error.code == error_code


async def _collect_chat_sequence(
    chunks: list[ChatStreamChunk],
    error: BaseException | None = None,
) -> list[str]:
    response = sse_streaming_response(
        chat.stream_response(_ChatRouter(chunks, error), _chat_request())
    )
    return [event async for event in response.body_iterator]


@pytest.fixture(autouse=True)
def pipeline_spy(monkeypatch):
    _PipelineSpy.instances.clear()
    _PipelineSpy.abort_reason = None
    monkeypatch.setattr("router_maestro.pipeline.RequestPipeline", _PipelineSpy)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_content_then_eof_is_unexpected_eof(protocol: str):
    events, status_code = await _collect(protocol)

    assert status_code == 200
    _assert_error_terminal(protocol, events, "unexpected_eof")
    _assert_no_success_terminal(protocol, events)
    _assert_pipeline_outcome("unexpected_eof", "unknown")


@pytest.mark.asyncio
async def test_anthropic_partial_tool_then_eof_does_not_flush_tool_or_success():
    chunks = [
        ChatStreamChunk(
            content="",
            tool_calls=[
                {
                    "index": 0,
                    "id": "tool-a",
                    "type": "function",
                    "function": {"name": "alpha", "arguments": '{"secret":'},
                }
            ],
        )
    ]

    response = sse_streaming_response(
        anthropic.stream_response(_ChatRouter(chunks), _chat_request(), "claude-sonnet-4", 1)
    )
    events = [event async for event in response.body_iterator]

    payloads = _json_payloads(events)
    assert not any(
        payload.get("type") == "content_block_start"
        and payload.get("content_block", {}).get("type") == "tool_use"
        for payload in payloads
    )
    assert "message_stop" not in _event_types(events)
    _assert_error_terminal("anthropic", events, "unexpected_eof")
    assert "secret" not in json.dumps(payloads)
    _assert_pipeline_outcome("unexpected_eof", "unknown")


@pytest.mark.asyncio
async def test_anthropic_malformed_tool_at_terminal_is_safe_protocol_error():
    chunks = [
        ChatStreamChunk(
            content="",
            tool_calls=[
                {
                    "index": 0,
                    "id": "tool-a",
                    "type": "function",
                    "function": {"name": "alpha", "arguments": '{"secret":'},
                }
            ],
        ),
        ChatStreamChunk(content="", finish_reason="tool_calls"),
    ]

    response = sse_streaming_response(
        anthropic.stream_response(_ChatRouter(chunks), _chat_request(), "claude-sonnet-4", 1)
    )
    events = [event async for event in response.body_iterator]

    payloads = _json_payloads(events)
    _assert_error_terminal("anthropic", events, "Invalid tool call from upstream")
    assert "message_stop" not in _event_types(events)
    assert "secret" not in json.dumps(payloads)
    _assert_pipeline_outcome(
        "exception",
        "failed",
        error_code="upstream_protocol_error",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_legacy_finish_reason_is_explicit_success(protocol: str):
    events, status_code = await _collect(protocol, terminal="stop")

    assert status_code == 200
    if protocol == "chat":
        assert events[-1] == "data: [DONE]\n\n"
        assert any(
            choice.get("finish_reason") == "stop"
            for payload in _json_payloads(events)
            for choice in payload.get("choices", [])
        )
    elif protocol == "responses":
        assert _event_types(events).count("response.completed") == 1
    elif protocol == "anthropic":
        assert _event_types(events).count("message_stop") == 1
    else:
        assert any(
            candidate.get("finishReason") == "STOP"
            for payload in _json_payloads(events)
            for candidate in payload.get("candidates", [])
        )
    _assert_pipeline_outcome("explicit_terminal", "completed")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_explicit_incomplete_is_not_unexpected_eof(protocol: str):
    outcome = _terminal_outcome(
        "incomplete",
        finish_reason="length",
        incomplete_details={"reason": "max_output_tokens"},
    )
    events, status_code = await _collect(protocol, canonical_outcome=outcome)

    assert status_code == 200
    if protocol == "chat":
        assert events[-1] == "data: [DONE]\n\n"
        assert any(
            choice.get("finish_reason") == "length"
            for payload in _json_payloads(events)
            for choice in payload.get("choices", [])
        )
    elif protocol == "responses":
        assert "response.completed" not in _event_types(events)
        assert _event_types(events).count("response.incomplete") == 1
        terminal = next(
            payload
            for payload in _json_payloads(events)
            if payload.get("type") == "response.incomplete"
        )
        assert terminal["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    elif protocol == "anthropic":
        delta = next(
            payload for payload in _json_payloads(events) if payload.get("type") == "message_delta"
        )
        assert delta["delta"]["stop_reason"] == "max_tokens"
        assert _event_types(events).count("message_stop") == 1
    else:
        assert any(
            candidate.get("finishReason") == "MAX_TOKENS"
            for payload in _json_payloads(events)
            for candidate in payload.get("candidates", [])
        )
    _assert_pipeline_outcome("explicit_terminal", "incomplete")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_explicit_failure_has_no_success_terminal(protocol: str):
    outcome = _terminal_outcome(
        "failed",
        error_code="upstream_failed",
        error_message="upstream failed",
    )
    events, status_code = await _collect(protocol, canonical_outcome=outcome)

    assert status_code == 200
    _assert_error_terminal(protocol, events, "upstream failed")
    _assert_no_success_terminal(protocol, events)
    _assert_pipeline_outcome("explicit_terminal", "failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_explicit_cancelled_has_no_success_terminal(protocol: str):
    outcome = _terminal_outcome(
        "cancelled",
        error_code="upstream_cancelled",
        error_message="upstream cancelled",
    )
    events, status_code = await _collect(protocol, canonical_outcome=outcome)

    assert status_code == 200
    _assert_error_terminal(protocol, events, "upstream cancelled")
    _assert_no_success_terminal(protocol, events)
    _assert_pipeline_outcome("explicit_terminal", "cancelled")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_explicit_unknown_has_no_success_terminal(protocol: str):
    outcome = _terminal_outcome(
        "unknown",
        error_code="unknown_status",
        error_message="unknown response status",
    )
    events, status_code = await _collect(protocol, canonical_outcome=outcome)

    assert status_code == 200
    _assert_error_terminal(protocol, events, "Illegal upstream terminal combination")
    _assert_no_success_terminal(protocol, events)
    _assert_pipeline_outcome(
        "exception",
        "failed",
        error_code="upstream_protocol_error",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_exception_is_failed_without_success_terminal(protocol: str):
    events, status_code = await _collect(protocol, error=RuntimeError("boom"))

    assert status_code == 200
    _assert_error_terminal(protocol, events, "Internal server error")
    _assert_no_success_terminal(protocol, events)
    _assert_pipeline_outcome("exception", "failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_typed_rate_limit_is_native_without_success_terminal(protocol: str):
    error = ProviderError(
        "late rate limit",
        status_code=429,
        kind=provider_base.ProviderFailureKind.RATE_LIMIT,
    )

    events, status_code = await _collect(protocol, error=error)

    assert status_code == 200
    _assert_error_terminal(protocol, events, "late rate limit")
    _assert_no_success_terminal(protocol, events)
    if protocol == "chat":
        terminal = next(
            payload["error"] for payload in _json_payloads(events) if "error" in payload
        )
        assert terminal["type"] == "rate_limit_error"
        assert terminal["code"] == "rate_limit_exceeded"
    elif protocol == "responses":
        terminal = next(
            payload
            for payload in _json_payloads(events)
            if payload.get("type") == "response.failed"
        )
        assert terminal["response"]["error"]["code"] == "rate_limit_exceeded"
    elif protocol == "anthropic":
        terminal = next(
            payload for payload in _json_payloads(events) if payload.get("type") == "error"
        )
        assert terminal["error"]["type"] == "rate_limit_error"
    else:
        terminal = next(
            payload["error"] for payload in _json_payloads(events) if "error" in payload
        )
        assert terminal["status"] == "RESOURCE_EXHAUSTED"
    _assert_pipeline_outcome("exception", "failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_guard_overload_is_native_without_success_terminal(protocol: str):
    _PipelineSpy.abort_reason = "guard overloaded"

    events, status_code = await _collect(protocol)

    assert status_code == 200
    _assert_no_success_terminal(protocol, events)
    if protocol == "chat":
        error = next(payload["error"] for payload in _json_payloads(events) if "error" in payload)
        assert error["type"] == "rate_limit_error"
        assert error["code"] == "overloaded"
    elif protocol == "responses":
        terminal = next(
            payload
            for payload in _json_payloads(events)
            if payload.get("type") == "response.failed"
        )
        assert terminal["response"]["error"]["code"] == "overloaded"
    elif protocol == "anthropic":
        terminal = next(
            payload for payload in _json_payloads(events) if payload.get("type") == "error"
        )
        assert terminal["error"]["type"] == "overloaded_error"
    else:
        terminal = next(
            payload["error"] for payload in _json_payloads(events) if "error" in payload
        )
        assert terminal["code"] == 529
    _assert_pipeline_outcome("exception", "failed", error_code="overloaded")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_client_cancellation_is_recorded_and_reraised(protocol: str):
    response = sse_streaming_response(_make_stream(protocol, error=asyncio.CancelledError()))
    events: list[str] = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    _assert_no_success_terminal(protocol, events)
    # Cancellation is a transport fact and does not produce a protocol wire error.
    if protocol == "chat":
        assert not any('"error"' in event for event in events)
    elif protocol == "responses":
        assert "response.failed" not in _event_types(events)
        assert "response.incomplete" not in _event_types(events)
    elif protocol == "anthropic":
        assert "error" not in _event_types(events)
    else:
        assert not any('"finishReason":"OTHER"' in event for event in events)
    _assert_pipeline_outcome("client_cancelled", "cancelled")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (RuntimeError("late boom"), "Internal server error"),
        (ProviderError("late provider failure", status_code=503), "late provider failure"),
    ],
    ids=["runtime-error", "provider-error"],
)
async def test_chat_late_exception_discards_pending_success(
    error: BaseException,
    expected_message: str,
):
    events = await _collect_chat_sequence(
        [
            ChatStreamChunk(content="hi"),
            ChatStreamChunk(content="", finish_reason="stop"),
        ],
        error,
    )

    _assert_no_success_terminal("chat", events)
    _assert_error_terminal("chat", events, expected_message)
    _assert_pipeline_outcome("exception", "failed")


@pytest.mark.asyncio
async def test_chat_late_cancellation_discards_pending_success_and_reraises():
    response = sse_streaming_response(
        chat.stream_response(
            _ChatRouter(
                [
                    ChatStreamChunk(content="hi"),
                    ChatStreamChunk(content="", finish_reason="stop"),
                ],
                asyncio.CancelledError(),
            ),
            _chat_request(),
        )
    )
    events: list[str] = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    _assert_no_success_terminal("chat", events)
    assert not any('"error"' in event for event in events)
    _assert_pipeline_outcome("client_cancelled", "cancelled")


@pytest.mark.asyncio
async def test_chat_terminal_then_usage_finalizes_once_at_transport_eof():
    usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    events = await _collect_chat_sequence(
        [
            ChatStreamChunk(content="hi"),
            ChatStreamChunk(content="", finish_reason="stop"),
            ChatStreamChunk(content="", usage=usage),
        ]
    )

    assert events[-1] == "data: [DONE]\n\n"
    terminal_payloads = [
        payload
        for payload in _json_payloads(events)
        if any(choice.get("finish_reason") for choice in payload.get("choices", []))
    ]
    assert len(terminal_payloads) == 1
    assert terminal_payloads[0]["choices"][0]["finish_reason"] == "stop"
    assert terminal_payloads[0]["usage"]["total_tokens"] == 6
    assert not any(payload.get("choices") == [] for payload in _json_payloads(events))
    _assert_pipeline_outcome("explicit_terminal", "completed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "late_chunk",
    [
        ChatStreamChunk(content="unexpected content"),
        ChatStreamChunk(content="", finish_reason="stop"),
    ],
    ids=["non-usage-chunk", "second-terminal"],
)
async def test_chat_rejects_non_usage_chunk_after_pending_terminal(
    late_chunk: ChatStreamChunk,
):
    events = await _collect_chat_sequence(
        [
            ChatStreamChunk(content="hi"),
            ChatStreamChunk(content="", finish_reason="stop"),
            late_chunk,
        ]
    )

    _assert_no_success_terminal("chat", events)
    _assert_error_terminal("chat", events, "after explicit terminal")
    _assert_pipeline_outcome("exception", "failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
@pytest.mark.parametrize(
    ("terminal", "canonical_outcome", "expected_wire_error"),
    [
        (
            None,
            provider_base.TerminalOutcome(
                transport=provider_base.TransportTermination.UNEXPECTED_EOF,
                response_status=provider_base.ResponseStatus.COMPLETED,
                finish_reason="stop",
            ),
            "Illegal upstream terminal combination",
        ),
        ("failed", None, "Unknown upstream finish reason"),
        (
            "tool_calls",
            provider_base.TerminalOutcome(
                transport=provider_base.TransportTermination.EXPLICIT_TERMINAL,
                response_status=provider_base.ResponseStatus.COMPLETED,
                finish_reason="stop",
            ),
            "finish reasons conflict",
        ),
        (
            None,
            provider_base.TerminalOutcome(
                transport=provider_base.TransportTermination.EXPLICIT_TERMINAL,
                response_status=provider_base.ResponseStatus.COMPLETED,
                finish_reason="stop",
                error=provider_base.TerminalError(
                    code="upstream_failed",
                    message="upstream failed",
                ),
            ),
            "Terminal error conflicts",
        ),
        (
            None,
            provider_base.TerminalOutcome(
                transport=provider_base.TransportTermination.EXPLICIT_TERMINAL,
                response_status=provider_base.ResponseStatus.INCOMPLETE,
                finish_reason="length",
                incomplete_details={"reason": "max_output_tokens"},
                error=provider_base.TerminalError(
                    code="upstream_failed",
                    message="upstream failed",
                ),
            ),
            "Terminal error conflicts",
        ),
    ],
    ids=[
        "illegal-outcome",
        "unknown-legacy-finish",
        "canonical-legacy-conflict",
        "completed-with-error",
        "incomplete-with-error",
    ],
)
async def test_invalid_terminal_data_is_one_protocol_error(
    protocol: str,
    terminal: str | None,
    canonical_outcome: Any,
    expected_wire_error: str,
):
    events, status_code = await _collect(
        protocol,
        terminal=terminal,
        canonical_outcome=canonical_outcome,
    )

    assert status_code == 200
    _assert_error_terminal(protocol, events, expected_wire_error)
    _assert_no_success_terminal(protocol, events)
    _assert_pipeline_outcome(
        "exception",
        "failed",
        error_code="upstream_protocol_error",
    )
