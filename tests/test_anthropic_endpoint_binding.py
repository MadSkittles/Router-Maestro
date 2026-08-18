"""Focused contracts for Anthropic's protocol-native Messages binding."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest

from router_maestro.protocols import WireProtocol
from router_maestro.providers.anthropic import (
    ANTHROPIC_MESSAGES_BINDING,
    AnthropicProvider,
)
from router_maestro.providers.base import ProviderError, ProviderFailureKind
from router_maestro.providers.bindings import AttemptRequestContext
from router_maestro.routing.capabilities import Operation
from router_maestro.routing.model_ref import ModelRef


def _provider() -> AnthropicProvider:
    provider = AnthropicProvider(base_url="https://anthropic.example/v1/")
    provider._get_headers = lambda: {  # type: ignore[method-assign]
        "x-api-key": "sk-test",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    return provider


def _binding(provider: AnthropicProvider):
    (binding,) = provider.bindings()
    return binding


def test_anthropic_binding_is_protocol_native_and_cached() -> None:
    provider = _provider()

    bindings = provider.bindings()
    (binding,) = bindings

    assert bindings is provider.bindings()
    assert binding.id == ANTHROPIC_MESSAGES_BINDING
    assert binding.protocol is WireProtocol.ANTHROPIC_MESSAGES
    assert binding.is_legacy is False
    assert binding.supports(Operation.NATIVE_ANTHROPIC)


@pytest.mark.asyncio
async def test_anthropic_dialect_is_copy_on_write_and_preserves_unknown_fields() -> None:
    provider = _provider()
    source = {
        "model": "anthropic/public-name",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello", "future_part": {"kept": True}}],
                "future_message": [1, 2, 3],
            }
        ],
        "max_tokens": 64,
        "stream": False,
        "future_top_level": {"kept": True},
    }
    original = deepcopy(source)

    attempt = await _binding(provider).prepare_attempt(
        model=ModelRef(provider="anthropic", upstream_id="claude-sonnet-4"),
        payload=source,
        stream=True,
    )

    assert source == original
    assert attempt.url == "https://anthropic.example/v1/messages"
    assert dict(attempt.payload) == {
        **original,
        "model": "claude-sonnet-4",
        "stream": True,
    }
    assert attempt.payload["future_top_level"] == {"kept": True}
    assert attempt.payload["messages"][0]["future_message"] == [1, 2, 3]
    assert dict(attempt.headers) == {
        "x-api-key": "sk-test",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }


@pytest.mark.asyncio
async def test_anthropic_dialect_forwards_only_opted_in_protocol_header() -> None:
    provider = _provider()
    attempt = await _binding(provider).prepare_attempt(
        model=ModelRef(provider="anthropic", upstream_id="claude-sonnet-4"),
        payload={"messages": [], "max_tokens": 1},
        stream=False,
        request_context=AttemptRequestContext(
            path="/api/anthropic/v1/messages",
            headers={
                "Anthropic-Beta": "prompt-caching-2024-07-31",
                "Authorization": "Bearer client-secret",
                "X-API-Key": "client-secret",
            },
        ),
    )

    assert attempt.headers["anthropic-beta"] == "prompt-caching-2024-07-31"
    assert attempt.headers["x-api-key"] == "sk-test"
    assert "authorization" not in {name.lower() for name in attempt.headers}


@pytest.mark.asyncio
async def test_anthropic_executor_returns_raw_response_and_preserves_extensions() -> None:
    provider = _provider()
    requests: list[httpx.Request] = []
    response_payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [{"type": "text", "text": "hello", "future_content": {"kept": True}}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 1, "future_usage": 9},
        "future_top_level": ["kept"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_payload)

    source = {
        "model": "anthropic/public-name",
        "messages": [{"role": "user", "content": "hello", "future_message": 1}],
        "max_tokens": 32,
        "future_request": {"kept": True},
    }
    attempt = await _binding(provider).prepare_attempt(
        model=ModelRef(provider="anthropic", upstream_id="claude-sonnet-4"),
        payload=source,
        stream=False,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        executor = _binding(provider).executor
        assert executor is not None
        result = await executor.execute(attempt)

    assert result == response_payload
    assert isinstance(result, dict)
    assert result["future_top_level"] == ["kept"]
    assert result["content"][0]["future_content"] == {"kept": True}
    assert len(requests) == 1
    assert requests[0].url == httpx.URL("https://anthropic.example/v1/messages")
    assert requests[0].headers["x-api-key"] == "sk-test"
    assert json.loads(requests[0].content) == {
        **source,
        "model": "claude-sonnet-4",
        "stream": False,
    }


@pytest.mark.asyncio
async def test_anthropic_executor_yields_raw_sse_dicts_without_finalizing_eof() -> None:
    provider = _provider()
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "content": [],
                "future_message": {"kept": True},
            },
            "future_frame": 1,
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello", "future_delta": "kept"},
        },
    ]
    body = "".join(
        (
            ": ping\n\n",
            *(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
        )
    ).encode()

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
    )
    attempt = await _binding(provider).prepare_attempt(
        model=ModelRef(provider="anthropic", upstream_id="claude-sonnet-4"),
        payload={
            "model": "anthropic/public-name",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
        },
        stream=True,
    )

    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        executor = _binding(provider).executor
        assert executor is not None
        frames = [frame async for frame in executor.execute_stream(attempt)]

    # The executor intentionally does not synthesize or require message_stop;
    # the dispatcher/response bridge owns canonical unexpected-EOF handling.
    assert frames == events
    assert frames[0]["future_frame"] == 1
    assert frames[0]["message"]["future_message"] == {"kept": True}
    assert frames[1]["delta"]["future_delta"] == "kept"


@pytest.mark.asyncio
async def test_anthropic_executor_filters_ping_before_first_semantic_frame() -> None:
    provider = _provider()
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4",
            "content": [],
        },
    }
    message_stop = {"type": "message_stop"}
    wire_events = [{"type": "ping"}, message_start, message_stop]
    body = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in wire_events
    ).encode()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
    )
    attempt = await _binding(provider).prepare_attempt(
        model=ModelRef(provider="anthropic", upstream_id="claude-sonnet-4"),
        payload={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
        },
        stream=True,
    )

    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        executor = _binding(provider).executor
        assert executor is not None
        frames = [frame async for frame in executor.execute_stream(attempt)]

    # The transport keepalive must not become the dispatcher's first semantic
    # frame, where it would be classified as malformed and trigger fallback.
    assert frames == [message_start, message_stop]
    assert frames[0]["type"] == "message_start"


@pytest.mark.asyncio
async def test_anthropic_executor_reuses_provider_status_classification() -> None:
    provider = _provider()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                429,
                json={"type": "error", "error": {"type": "rate_limit_error"}},
            )
        )
    )
    attempt = await _binding(provider).prepare_attempt(
        model=ModelRef(provider="anthropic", upstream_id="claude-sonnet-4"),
        payload={"messages": [], "max_tokens": 1},
        stream=False,
    )

    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        executor = _binding(provider).executor
        assert executor is not None
        with pytest.raises(ProviderError) as exc_info:
            await executor.execute(attempt)

    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
    assert exc_info.value.kind is ProviderFailureKind.RATE_LIMIT
    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.model == "claude-sonnet-4"
