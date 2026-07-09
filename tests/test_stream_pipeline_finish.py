"""Regression tests for streaming routes finalizing RequestPipeline."""

from collections.abc import AsyncIterator

import pytest

from router_maestro.providers import ChatRequest, ChatStreamChunk, Message
from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.providers.base import ProviderError, ResponsesStreamChunk
from router_maestro.server.routes import anthropic, chat, gemini, responses


class _PipelineSpy:
    instances: list["_PipelineSpy"] = []

    def __init__(self, request_id: str, model: str, tool_names: set[str] | None = None):
        self.request_id = request_id
        self.model = model
        self.tool_names = tool_names
        self.finished: list[tuple[int, str | None]] = []
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

    def finish(self, status: int = 200, body_summary: str | None = None) -> None:
        self.finished.append((status, body_summary))


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
    assert pipeline_spy[0].finished == [(200, None)]


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
    assert pipeline_spy[0].finished == [(503, "upstream failed")]
