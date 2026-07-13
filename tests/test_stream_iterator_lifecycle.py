"""Regression tests for closing provider iterators on stream termination."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from router_maestro.providers import ChatRequest, ChatStreamChunk, Message, ResponsesRequest
from router_maestro.providers.base import (
    ProviderError,
    ResponsesStreamChunk,
    ResponseStatus,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
)
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan
from router_maestro.routing.router import Router
from router_maestro.server.routes import anthropic, chat, gemini, responses
from router_maestro.utils import async_iterators
from router_maestro.utils.async_iterators import close_async_iterator


@dataclass
class _LifecycleStream:
    chunks: list[Any]
    block_after_chunks: bool = True
    close_count: int = 0
    waiting: asyncio.Event = field(default_factory=asyncio.Event)

    async def generate(self) -> AsyncIterator[Any]:
        try:
            for chunk in self.chunks:
                yield chunk
            if self.block_after_chunks:
                self.waiting.set()
                await asyncio.Event().wait()
        finally:
            self.close_count += 1


class _LifecycleRouter:
    """Expose a provider stream through both Router iterator wrappers."""

    def __init__(self, lifecycle: _LifecycleStream):
        self.name = "stub"
        self.lifecycle = lifecycle
        self._router = Router.__new__(Router)
        # Keep each layer alive so garbage collection cannot hide missing
        # explicit aclose() propagation.
        self.iterator_layers: list[AsyncIterator[Any]] = []

    async def ensure_token(self) -> None:
        return None

    async def _open(self) -> AsyncIterator[Any]:
        def call_stream(provider: Any, request: Any) -> AsyncIterator[Any]:
            provider_stream = self.lifecycle.generate()
            self.iterator_layers.append(provider_stream)
            return provider_stream

        stream, _provider_name = await self._router._execute_plan_stream(
            _stream_plan(("stub", "stub-model", self)),
            object(),
            True,
            lambda request, model: request,
            call_stream,
            "lifecycle",
        )
        self.iterator_layers.append(stream)
        return stream

    async def chat_completion_stream(
        self,
        request: ChatRequest,
        fallback: bool = True,
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        return await self._open(), "stub"

    async def responses_completion_stream(
        self,
        request: ResponsesRequest,
        fallback: bool = True,
    ) -> tuple[AsyncIterator[ResponsesStreamChunk], str]:
        return await self._open(), "stub"


class _PipelineStub:
    abort_reason: str | None = None

    @classmethod
    def create(
        cls,
        request_id: str,
        model: str,
        tool_names: set[str] | None = None,
    ) -> _PipelineStub:
        return cls()

    def feed_stream(self, chunk: Any) -> str | None:
        return self.abort_reason

    def check_invoke_at_finish(self) -> list[dict] | None:
        return None

    def finish(self, **kwargs: Any) -> None:
        return None


def _stream_plan(*providers: tuple[str, str, object]) -> RoutePlan:
    candidates: list[RouteCandidate] = []
    for name, model, provider in providers:
        ref = ModelRef(name, model)
        operation = Operation.CHAT_STREAM
        features = RequestFeatures()
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: CapabilitySupport.SUPPORTED},
        )
        candidates.append(
            RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            )
        )
    return RoutePlan(
        Operation.CHAT_STREAM,
        RequestFeatures(),
        candidates[0],
        tuple(candidates[1:]),
        False,
    )


@pytest.fixture(autouse=True)
def pipeline_stub(monkeypatch):
    _PipelineStub.abort_reason = None
    monkeypatch.setattr("router_maestro.pipeline.RequestPipeline", _PipelineStub)


def _chat_request() -> ChatRequest:
    return ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )


def _responses_request() -> ResponsesRequest:
    return ResponsesRequest(model="github-copilot/gpt-5", input="hi", stream=True)


def _route_stream(protocol: str, router: _LifecycleRouter):
    if protocol == "chat":
        return chat.stream_response(router, _chat_request())
    if protocol == "responses":
        return responses.stream_response(
            router,
            _responses_request(),
            request_id="req-lifecycle",
            start_time=time.time(),
        )
    if protocol == "anthropic":
        return anthropic.stream_response(router, _chat_request(), "claude-sonnet-4", 1)
    if protocol == "gemini":
        return gemini._stream_response(router, _chat_request(), "gemini-2.5-pro", 1)
    raise AssertionError(f"unknown protocol: {protocol}")


def _chunks(protocol: str, outcome: TerminalOutcome | None = None) -> list[Any]:
    chunk_type = ResponsesStreamChunk if protocol == "responses" else ChatStreamChunk
    chunks = [chunk_type(content="hello")]
    if outcome is not None:
        chunks.append(chunk_type(content="", terminal_outcome=outcome))
    return chunks


async def _consume(stream: AsyncIterator[str]) -> list[str]:
    return [event async for event in stream]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_explicit_success_closes_provider_iterator_once(protocol: str):
    outcome = TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=ResponseStatus.COMPLETED,
        finish_reason="stop",
    )
    # Chat deliberately reads through transport EOF to collect trailing usage.
    lifecycle = _LifecycleStream(
        _chunks(protocol, outcome),
        block_after_chunks=protocol != "chat",
    )
    route = _route_stream(protocol, _LifecycleRouter(lifecycle))

    await asyncio.wait_for(_consume(route), timeout=1)

    assert lifecycle.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_explicit_failure_closes_provider_iterator_once(protocol: str):
    outcome = TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=ResponseStatus.FAILED,
        error=TerminalError(code="upstream_failed", message="upstream failed"),
    )
    lifecycle = _LifecycleStream(_chunks(protocol, outcome))
    route = _route_stream(protocol, _LifecycleRouter(lifecycle))

    await asyncio.wait_for(_consume(route), timeout=1)

    assert lifecycle.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_guard_abort_closes_provider_iterator_once(protocol: str):
    _PipelineStub.abort_reason = "forced guard abort"
    lifecycle = _LifecycleStream(_chunks(protocol))
    route = _route_stream(protocol, _LifecycleRouter(lifecycle))

    await asyncio.wait_for(_consume(route), timeout=1)

    assert lifecycle.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
async def test_client_cancellation_closes_provider_iterator_once(protocol: str):
    lifecycle = _LifecycleStream(_chunks(protocol))
    route = _route_stream(protocol, _LifecycleRouter(lifecycle))
    consumer = asyncio.create_task(_consume(route))
    await asyncio.wait_for(lifecycle.waiting.wait(), timeout=1)

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert lifecycle.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_header_only_disconnect_closes_primed_provider_once(protocol: str):
    outcome = TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=ResponseStatus.COMPLETED,
        finish_reason="stop",
    )
    lifecycle = _LifecycleStream(_chunks(protocol, outcome))
    route = _route_stream(protocol, _LifecycleRouter(lifecycle))

    await anext(route)
    await route.aclose()

    assert lifecycle.close_count == 1


@pytest.mark.asyncio
async def test_never_started_primed_stream_close_closes_inner_once():
    lifecycle = _LifecycleStream(
        [ChatStreamChunk(content="primed"), ChatStreamChunk(content="remaining")]
    )
    inner = lifecycle.generate()
    first_chunk = await anext(inner)
    stream = Router.__new__(Router)._chain_first_chunk(first_chunk, inner)

    await stream.aclose()
    await stream.aclose()

    assert lifecycle.close_count == 1


@pytest.mark.asyncio
async def test_primed_stream_normal_exhaustion_closes_inner_once():
    lifecycle = _LifecycleStream(
        [ChatStreamChunk(content="primed"), ChatStreamChunk(content="remaining")],
        block_after_chunks=False,
    )
    inner = lifecycle.generate()
    first_chunk = await anext(inner)
    stream = Router.__new__(Router)._chain_first_chunk(first_chunk, inner)

    chunks = [chunk async for chunk in stream]
    await stream.aclose()

    assert [chunk.content for chunk in chunks] == ["primed", "remaining"]
    assert lifecycle.close_count == 1


@pytest.mark.asyncio
async def test_retryable_priming_failure_closes_candidate_before_fallback():
    router = Router.__new__(Router)
    primary_close_count = 0

    class _Provider:
        def __init__(self, name: str) -> None:
            self.name = name

        async def ensure_token(self) -> None:
            return None

    primary = _Provider("primary")
    secondary = _Provider("secondary")

    async def primary_stream() -> AsyncIterator[ChatStreamChunk]:
        nonlocal primary_close_count
        try:
            if False:
                yield ChatStreamChunk(content="")
            raise ProviderError("retry me", status_code=503, retryable=True)
        finally:
            primary_close_count += 1

    async def secondary_stream() -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="fallback", finish_reason="stop")

    stream, provider_name = await router._execute_plan_stream(
        _stream_plan(
            ("primary", "model", primary),
            ("secondary", "model", secondary),
        ),
        object(),
        True,
        lambda request, model: request,
        lambda provider, request: primary_stream() if provider is primary else secondary_stream(),
        "lifecycle",
    )

    assert provider_name == "secondary"
    assert primary_close_count == 1
    await stream.aclose()


@pytest.mark.asyncio
async def test_close_error_does_not_replace_stream_terminal(monkeypatch):
    warnings: list[str] = []

    class _LoggerStub:
        def warning(self, message: str, **kwargs: Any) -> None:
            warnings.append(message)

    monkeypatch.setattr(async_iterators, "logger", _LoggerStub())

    class _BrokenCloseIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    await close_async_iterator(_BrokenCloseIterator())

    assert warnings == ["Failed to close async iterator"]
