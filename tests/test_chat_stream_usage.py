"""Tests for OpenAI chat streaming usage propagation."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.providers import ChatRequest, ChatStreamChunk, CopilotProvider, Message
from router_maestro.providers.openai_compat import OpenAICompatibleProvider
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.server.routes.chat import stream_response


class _StubRouter:
    """Minimal Router stub returning a pre-built chat chunk stream."""

    def __init__(self, chunks: list[ChatStreamChunk]):
        self._chunks = chunks

    async def prepare_chat_completion_stream(self, _request: ChatRequest):
        return object()

    async def chat_completion_stream(
        self,
        request: ChatRequest,
        fallback: bool = True,
        *,
        prepared_plan=None,
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


def _usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 5},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }


@pytest.mark.asyncio
async def test_openai_compatible_chat_stream_emits_usage_only_chunk():
    """OpenAI-compatible providers must surface usage chunks with empty choices."""

    def handler(_request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":4,'
            b'"completion_tokens":2,"total_tokens":6}}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    provider = OpenAICompatibleProvider(
        name="custom",
        base_url="https://example.com/v1",
        api_key="sk-test",
    )

    provider_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "httpx.AsyncClient",
                lambda *args, **kwargs: provider_client,
            )
            chunks = [
                chunk
                async for chunk in provider.chat_completion_stream(
                    ChatRequest(
                        model="gpt-4o",
                        messages=[Message(role="user", content="hi")],
                        stream=True,
                    )
                )
            ]
    finally:
        await provider_client.aclose()

    usage_chunks = [chunk for chunk in chunks if chunk.usage]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].content == ""
    assert usage_chunks[0].finish_reason is None
    assert usage_chunks[0].usage == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }


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


@pytest.mark.asyncio
async def test_openai_chat_stream_keeps_usage_chunk_after_explicit_finish():
    """OpenAI may send include_usage data after the terminal choice chunk."""
    router = _StubRouter(
        [
            ChatStreamChunk(content="hello"),
            ChatStreamChunk(content="", finish_reason="stop"),
            ChatStreamChunk(
                content="",
                usage={
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                },
            ),
        ]
    )
    request = ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    raw_events = [event async for event in stream_response(router, request)]  # type: ignore[arg-type]
    events = _parse_chat_stream_events(raw_events)

    assert raw_events[-1] == "data: [DONE]\n\n"
    usage_events = [event for event in events if event.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["choices"][0]["finish_reason"] == "stop"
    assert usage_events[0]["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param(
            [
                ChatStreamChunk(content="hello"),
                ChatStreamChunk(content="", usage=_usage()),
                ChatStreamChunk(content="", finish_reason="stop"),
            ],
            id="usage-only-before-terminal",
        ),
        pytest.param(
            [
                ChatStreamChunk(content="hello"),
                ChatStreamChunk(content="", finish_reason="stop", usage=_usage()),
            ],
            id="usage-on-terminal",
        ),
        pytest.param(
            [
                ChatStreamChunk(content="hello"),
                ChatStreamChunk(content="", finish_reason="stop"),
                ChatStreamChunk(content="", usage=_usage()),
            ],
            id="usage-only-after-terminal",
        ),
        pytest.param(
            [
                ChatStreamChunk(content="hello", usage=_usage()),
                ChatStreamChunk(content="", finish_reason="stop"),
            ],
            id="usage-on-content-chunk",
        ),
    ],
)
async def test_openai_chat_stream_include_usage_emits_one_independent_tail_chunk(chunks):
    request = ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    raw_events = [
        event
        async for event in stream_response(  # type: ignore[arg-type]
            _StubRouter(chunks),
            request,
            include_usage=True,
        )
    ]
    events = _parse_chat_stream_events(raw_events)

    assert raw_events[-1] == "data: [DONE]\n\n"
    usage_indexes = [index for index, event in enumerate(events) if event.get("usage")]
    finish_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("choices") and event["choices"][0]["finish_reason"] == "stop"
    ]
    assert finish_indexes == [len(events) - 2]
    assert usage_indexes == [len(events) - 1]
    assert events[-1]["choices"] == []
    assert events[-1]["usage"] == _usage()
    assert all(event["usage"] is None for event in events[:-1])


@pytest.mark.asyncio
async def test_openai_chat_stream_include_usage_omits_tail_when_provider_has_no_usage():
    request = ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )
    router = _StubRouter(
        [
            ChatStreamChunk(content="hello"),
            ChatStreamChunk(content="", finish_reason="stop"),
        ]
    )

    raw_events = [
        event
        async for event in stream_response(  # type: ignore[arg-type]
            router,
            request,
            include_usage=True,
        )
    ]
    events = _parse_chat_stream_events(raw_events)

    assert raw_events[-1] == "data: [DONE]\n\n"
    assert not any(event.get("usage") for event in events)
    assert not any(event["choices"] == [] for event in events)


@pytest.mark.asyncio
async def test_openai_chat_stream_exclude_usage_suppresses_all_provider_usage():
    request = ChatRequest(
        model="github-copilot/gpt-4o",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )
    router = _StubRouter(
        [
            ChatStreamChunk(content="hello", usage=_usage()),
            ChatStreamChunk(content="", usage=_usage()),
            ChatStreamChunk(content="", finish_reason="stop", usage=_usage()),
            ChatStreamChunk(content="", usage=_usage()),
        ]
    )

    raw_events = [
        event
        async for event in stream_response(  # type: ignore[arg-type]
            router,
            request,
            include_usage=False,
        )
    ]
    events = _parse_chat_stream_events(raw_events)

    assert raw_events[-1] == "data: [DONE]\n\n"
    assert not any(event.get("usage") for event in events)
    assert not any(event["choices"] == [] for event in events)


@pytest.mark.parametrize(
    ("payload_update", "expected_usage_choices"),
    [
        pytest.param({}, [0], id="omitted-keeps-legacy-finish-usage"),
        pytest.param({"stream_options": {}}, [], id="explicit-default-excludes-usage"),
        pytest.param(
            {"stream_options": {"include_usage": True}},
            [None],
            id="explicit-include-emits-independent-usage",
        ),
        pytest.param(
            {"stream_options": {"include_usage": False}},
            [],
            id="explicit-exclude-suppresses-usage",
        ),
    ],
)
def test_openai_chat_endpoint_applies_stream_options_wire_preference(
    monkeypatch,
    payload_update,
    expected_usage_choices,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    router = _StubRouter(
        [
            ChatStreamChunk(content="hello"),
            ChatStreamChunk(content="", finish_reason="stop"),
            ChatStreamChunk(content="", usage=_usage()),
        ]
    )
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "github-copilot/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            **payload_update,
        },
    )
    events = _parse_chat_stream_events([response.text])
    usage_events = [event for event in events if event.get("usage")]

    assert response.status_code == 200
    assert [
        event["choices"][0]["index"] if event["choices"] else None for event in usage_events
    ] == expected_usage_choices


@pytest.mark.asyncio
async def test_openai_chat_stream_emits_refusal_delta_without_text_content():
    router = _StubRouter(
        [
            ChatStreamChunk(content="", refusal="I cannot help"),
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

    refusal_events = [
        event
        for event in events
        if event.get("choices") and event["choices"][0]["delta"].get("refusal")
    ]
    assert [event["choices"][0]["delta"]["refusal"] for event in refusal_events] == [
        "I cannot help"
    ]


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
    provider._get_headers = lambda *args, **kwargs: {"authorization": "Bearer test"}  # type: ignore[method-assign]

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
    provider._get_headers = lambda *args, **kwargs: {"authorization": "Bearer test"}  # type: ignore[method-assign]

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


@pytest.mark.asyncio
async def test_copilot_chat_stream_processes_all_choices():
    """Copilot can split text/tool deltas across choices in the same SSE event."""

    def handler(_request: httpx.Request) -> httpx.Response:
        body = (
            b'data: {"choices":['
            b'{"delta":{"content":"hello"},"finish_reason":null},'
            b'{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
            b'"function":{"name":"test_tool","arguments":"{}"}}]},'
            b'"finish_reason":null}'
            b"]}\n\n"
            b'data: {"choices":['
            b'{"delta":{},"finish_reason":"stop"},'
            b'{"delta":{"content":"ignored only if choice skipped"},"finish_reason":null}'
            b"]}\n\n"
        )
        return httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    provider = CopilotProvider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider.ensure_token = _noop  # type: ignore[method-assign]
    provider._get_headers = lambda *args, **kwargs: {"authorization": "Bearer test"}  # type: ignore[method-assign]

    chunks = [
        chunk
        async for chunk in provider.chat_completion_stream(
            ChatRequest(
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
                stream=True,
            )
        )
    ]

    assert any(chunk.content == "hello" for chunk in chunks)
    assert any(chunk.tool_calls for chunk in chunks)
    assert chunks[-1].finish_reason == "tool_calls"
