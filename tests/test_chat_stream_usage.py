"""Tests for OpenAI chat streaming usage propagation."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from router_maestro.providers import ChatRequest, ChatStreamChunk, CopilotProvider, Message
from router_maestro.server.routes.chat import stream_response


class _StubRouter:
    """Minimal Router stub returning a pre-built chat chunk stream."""

    def __init__(self, chunks: list[ChatStreamChunk]):
        self._chunks = chunks

    async def chat_completion_stream(
        self, request: ChatRequest, fallback: bool = True
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        async def _gen() -> AsyncIterator[ChatStreamChunk]:
            for chunk in self._chunks:
                yield chunk

        return _gen(), "github-copilot"


def _parse_chat_stream_events(raw_events: list[str]) -> list[dict[str, Any]]:
    events = []
    for raw in raw_events:
        for line in raw.splitlines():
            if line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.mark.asyncio
async def test_openai_chat_stream_emits_usage_only_chunk():
    """Provider usage-only chunks must reach OpenAI chat streaming clients."""
    router = _StubRouter(
        [
            ChatStreamChunk(content="hello"),
            ChatStreamChunk(
                content="",
                usage={
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 5},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            ),
            ChatStreamChunk(content="", finish_reason="stop"),
        ]
    )
    request = ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    raw_events = [event async for event in stream_response(router, request)]  # type: ignore[arg-type]
    events = _parse_chat_stream_events(raw_events)

    usage_events = [event for event in events if event.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["choices"] == []
    assert usage_events[0]["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 5},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }


async def _noop() -> None:
    return None


@pytest.mark.asyncio
async def test_copilot_chat_stream_requests_usage_from_upstream():
    """Copilot chat streaming should ask upstream to include usage chunks."""
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        body = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        return httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    provider = CopilotProvider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider.ensure_token = _noop  # type: ignore[method-assign]
    provider._get_headers = lambda vision_request=False: {"authorization": "Bearer test"}  # type: ignore[method-assign]

    request = ChatRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )
    chunks = [chunk async for chunk in provider.chat_completion_stream(request)]

    assert chunks[-1].finish_reason == "stop"
    assert captured_payloads[0]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_copilot_chat_stream_tool_calls_force_tool_calls_finish_reason():
    """Copilot can stream tool_calls with finish_reason=stop; normalize it."""

    def handler(_request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            b'"type":"function","function":{"name":"test_tool","arguments":"{}"}}]},'
            b'"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        )
        return httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    provider = CopilotProvider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider.ensure_token = _noop  # type: ignore[method-assign]
    provider._get_headers = lambda vision_request=False: {"authorization": "Bearer test"}  # type: ignore[method-assign]

    request = ChatRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Use the test tool")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "test_tool"}},
        stream=True,
    )
    chunks = [chunk async for chunk in provider.chat_completion_stream(request)]

    assert any(chunk.tool_calls for chunk in chunks)
    assert chunks[-1].finish_reason == "tool_calls"
