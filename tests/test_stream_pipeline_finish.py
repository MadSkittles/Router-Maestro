"""Regression tests for streaming routes finalizing RequestPipeline."""

import logging
from collections.abc import AsyncIterator

import pytest

from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.pipeline.request_pipeline import RequestPipeline
from router_maestro.providers import ChatRequest, ChatStreamChunk, Message
from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.providers.base import (
    ProviderError,
    ResponsesStreamChunk,
    exception_outcome,
    unexpected_eof_outcome,
)
from router_maestro.server.routes import anthropic, chat, gemini, responses


class _PipelineSpy:
    instances: list["_PipelineSpy"] = []

    def __init__(self, request_id: str, model: str, tool_names: set[str] | None = None):
        self.request_id = request_id
        self.model = model
        self.tool_names = tool_names
        self.finished: list[tuple[int | None, object | None, str | None]] = []
        _PipelineSpy.instances.append(self)

    @classmethod
    def create(
        cls,
        request_id: str,
        model: str,
        tool_names: set[str] | None = None,
    ) -> "_PipelineSpy":
        return cls(request_id=request_id, model=model, tool_names=tool_names)

    def feed_stream(self, chunk) -> str | None:
        return None

    def check_invoke_at_finish(self) -> list[dict] | None:
        return None

    def finish(
        self,
        *,
        wire_status: int | None = None,
        outcome: object | None = None,
        body_summary: str | None = None,
        status: int | None = None,
    ) -> None:
        self.finished.append(
            (wire_status if wire_status is not None else status, outcome, body_summary)
        )


class _ChatRouter:
    def __init__(self, chunks: list[ChatStreamChunk], error: ProviderError | None = None):
        self._chunks = chunks
        self._error = error

    async def chat_completion_stream(
        self, request: ChatRequest, fallback: bool = True
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        async def _gen() -> AsyncIterator[ChatStreamChunk]:
            for chunk in self._chunks:
                yield chunk
            if self._error is not None:
                raise self._error

        return _gen(), "github-copilot"


class _ResponsesRouter:
    def __init__(
        self,
        chunks: list[ResponsesStreamChunk],
        error: ProviderError | None = None,
    ):
        self._chunks = chunks
        self._error = error

    async def responses_completion_stream(
        self, request: InternalResponsesRequest, fallback: bool = True
    ) -> tuple[AsyncIterator[ResponsesStreamChunk], str]:
        async def _gen() -> AsyncIterator[ResponsesStreamChunk]:
            for chunk in self._chunks:
                yield chunk
            if self._error is not None:
                raise self._error

        return _gen(), "github-copilot"


def _chat_request() -> ChatRequest:
    return ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )


def _responses_request() -> InternalResponsesRequest:
    return InternalResponsesRequest(model="github-copilot/gpt-5", input="hi", stream=True)


@pytest.fixture(autouse=True)
def pipeline_spy(monkeypatch):
    _PipelineSpy.instances.clear()
    monkeypatch.setattr("router_maestro.pipeline.RequestPipeline", _PipelineSpy)
    return _PipelineSpy.instances


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream_func", "router", "route_request", "extra_args"),
    [
        (
            chat.stream_response,
            _ChatRouter([ChatStreamChunk(content="hi", finish_reason="stop")]),
            _chat_request(),
            (),
        ),
        (
            anthropic.stream_response,
            _ChatRouter([ChatStreamChunk(content="hi", finish_reason="stop")]),
            _chat_request(),
            ("github-copilot/claude-sonnet-4", 1),
        ),
        (
            gemini._stream_response,
            _ChatRouter([ChatStreamChunk(content="hi", finish_reason="stop")]),
            _chat_request(),
            ("github-copilot/gemini-2.5-pro", 1),
        ),
        (
            responses.stream_response,
            _ResponsesRouter([ResponsesStreamChunk(content="hi", finish_reason="stop")]),
            _responses_request(),
            ("req-test", 0.0),
        ),
    ],
)
async def test_stream_routes_finish_pipeline_on_success(
    pipeline_spy, stream_func, router, route_request, extra_args
):
    events = [event async for event in stream_func(router, route_request, *extra_args)]

    assert events
    assert len(pipeline_spy) == 1
    assert len(pipeline_spy[0].finished) == 1
    wire_status, outcome, summary = pipeline_spy[0].finished[0]
    assert wire_status == 200
    assert summary is None
    assert outcome.transport.value == "explicit_terminal"
    assert outcome.response_status.value == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream_func", "router", "route_request", "extra_args"),
    [
        (
            chat.stream_response,
            _ChatRouter(
                [ChatStreamChunk(content="hi")],
                ProviderError("upstream failed", status_code=503),
            ),
            _chat_request(),
            (),
        ),
        (
            anthropic.stream_response,
            _ChatRouter(
                [ChatStreamChunk(content="hi")],
                ProviderError("upstream failed", status_code=503),
            ),
            _chat_request(),
            ("github-copilot/claude-sonnet-4", 1),
        ),
        (
            gemini._stream_response,
            _ChatRouter(
                [ChatStreamChunk(content="hi")],
                ProviderError("upstream failed", status_code=503),
            ),
            _chat_request(),
            ("github-copilot/gemini-2.5-pro", 1),
        ),
        (
            responses.stream_response,
            _ResponsesRouter(
                [ResponsesStreamChunk(content="hi")],
                ProviderError("upstream failed", status_code=503),
            ),
            _responses_request(),
            ("req-test", 0.0),
        ),
    ],
)
async def test_stream_routes_finish_pipeline_on_provider_error_after_pipeline_exists(
    pipeline_spy, stream_func, router, route_request, extra_args
):
    events = [event async for event in stream_func(router, route_request, *extra_args)]

    assert any("upstream failed" in event for event in events)
    assert len(pipeline_spy) == 1
    assert len(pipeline_spy[0].finished) == 1
    wire_status, outcome, summary = pipeline_spy[0].finished[0]
    assert wire_status == 200
    assert summary == "upstream failed"
    assert outcome.transport.value == "exception"
    assert outcome.response_status.value == "failed"


class _AuditSpy:
    def __init__(self):
        self.outbound: list[tuple[int, str | None, object]] = []
        self.flush_count = 0

    def record_outbound(
        self,
        status: int,
        *,
        body_summary: str | None,
        outcome: object,
    ) -> None:
        self.outbound.append((status, body_summary, outcome))

    def flush(self) -> None:
        self.flush_count += 1


def _real_pipeline(audit: _AuditSpy) -> RequestPipeline:
    return RequestPipeline(
        request_id="req-test",
        guards=[],
        leak_guard=None,
        audit=audit,
        config=PrioritiesConfig.get_default(),
    )


def test_pipeline_finish_stores_first_wire_status_and_outcome():
    audit = _AuditSpy()
    pipeline = _real_pipeline(audit)
    outcome = unexpected_eof_outcome()

    pipeline.finish(wire_status=200, outcome=outcome, body_summary="first")

    assert pipeline.wire_status == 200
    assert pipeline.outcome is outcome
    assert audit.outbound == [(200, "first", outcome)]
    assert audit.flush_count == 1


def test_pipeline_finish_exact_duplicate_is_silent_noop(caplog):
    audit = _AuditSpy()
    pipeline = _real_pipeline(audit)
    outcome = unexpected_eof_outcome()

    with caplog.at_level(logging.ERROR, logger="router_maestro.pipeline"):
        pipeline.finish(wire_status=200, outcome=outcome, body_summary="first")
        pipeline.finish(wire_status=200, outcome=outcome, body_summary="second")

    assert "conflicting finalization" not in caplog.text
    assert audit.outbound == [(200, "first", outcome)]
    assert audit.flush_count == 1


@pytest.mark.parametrize("conflict", ["wire-status", "outcome"])
def test_pipeline_finish_logs_conflict_and_retains_first_finalization(
    caplog,
    monkeypatch,
    conflict: str,
):
    audit = _AuditSpy()
    pipeline = _real_pipeline(audit)
    first_outcome = unexpected_eof_outcome()
    second_outcome = exception_outcome("late failure")
    monkeypatch.setattr(logging.getLogger("router_maestro"), "propagate", True)
    monkeypatch.setattr(logging.getLogger("router_maestro.pipeline"), "propagate", True)

    pipeline.finish(wire_status=200, outcome=first_outcome, body_summary="first")
    with caplog.at_level(logging.ERROR, logger="router_maestro.pipeline"):
        pipeline.finish(
            wire_status=500 if conflict == "wire-status" else 200,
            outcome=second_outcome if conflict == "outcome" else first_outcome,
            body_summary="second",
        )

    assert "conflicting finalization" in caplog.text
    assert pipeline.wire_status == 200
    assert pipeline.outcome is first_outcome
    assert audit.outbound == [(200, "first", first_outcome)]
    assert audit.flush_count == 1
