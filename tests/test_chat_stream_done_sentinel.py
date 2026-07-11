"""Regression: OpenAI Chat errors must not masquerade as successful termination.

``[DONE]`` is the successful Chat Completions stream sentinel. Once an error
frame is emitted, physical SSE EOF ends the stream without a contradictory
success sentinel.
"""

from collections.abc import AsyncIterator

import pytest

from router_maestro.providers import ChatRequest, ChatStreamChunk, Message
from router_maestro.providers.base import ProviderError
from router_maestro.server.routes.chat import stream_response


class _ProviderErrorRouter:
    """Stream that yields one chunk then raises a ProviderError mid-stream."""

    async def chat_completion_stream(
        self, request: ChatRequest, fallback: bool = True
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        async def _gen() -> AsyncIterator[ChatStreamChunk]:
            yield ChatStreamChunk(content="hi")
            raise ProviderError("upstream exploded", retryable=False)

        return _gen(), "github-copilot"


class _UnexpectedErrorRouter:
    """Stream that raises a non-ProviderError mid-stream."""

    async def chat_completion_stream(
        self, request: ChatRequest, fallback: bool = True
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        async def _gen() -> AsyncIterator[ChatStreamChunk]:
            yield ChatStreamChunk(content="hi")
            raise RuntimeError("boom")

        return _gen(), "github-copilot"


def _request() -> ChatRequest:
    return ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )


@pytest.mark.asyncio
async def test_provider_error_stream_ends_without_done():
    router = _ProviderErrorRouter()
    events = [e async for e in stream_response(router, _request())]  # type: ignore[arg-type]

    assert events[-1] != "data: [DONE]\n\n"
    assert all(event != "data: [DONE]\n\n" for event in events)
    assert any("upstream exploded" in e for e in events)


@pytest.mark.asyncio
async def test_unexpected_error_stream_ends_without_done():
    router = _UnexpectedErrorRouter()
    events = [e async for e in stream_response(router, _request())]  # type: ignore[arg-type]

    assert events[-1] != "data: [DONE]\n\n"
    assert all(event != "data: [DONE]\n\n" for event in events)
    # Internal error message is generic — no leak of the RuntimeError text.
    assert any("Internal server error" in e for e in events)
    assert not any("boom" in e for e in events)
