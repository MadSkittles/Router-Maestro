"""Tests for AnthropicProvider native SSE streaming.

Regression coverage for the streaming handler that previously dropped tool
calls, usage, and the real finish_reason (only text deltas were handled).
"""

from unittest.mock import patch

import httpx
import pytest

from router_maestro.auth.storage import ApiKeyCredential, AuthStorage
from router_maestro.providers import ChatRequest, Message
from router_maestro.providers.anthropic import AnthropicProvider


def _sse(*events: str) -> bytes:
    return ("".join(f"data: {e}\n\n" for e in events)).encode()


def _make_provider() -> AnthropicProvider:
    provider = AnthropicProvider()
    # Inject an API key credential so _get_api_key() succeeds.
    provider.auth_manager.storage = AuthStorage()
    provider.auth_manager.storage.set("anthropic", ApiKeyCredential(key="sk-test"))
    return provider


@pytest.mark.asyncio
async def test_stream_emits_tool_calls_usage_and_finish():
    """A tool-using Anthropic stream yields tool_call deltas, usage, finish."""
    body = _sse(
        '{"type":"message_start","message":{"usage":{"input_tokens":42}}}',
        '{"type":"content_block_start","index":0,'
        '"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}',
        '{"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":"}}',
        '{"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"\\"SF\\"}"}}',
        '{"type":"content_block_stop","index":0}',
        '{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}',
        '{"type":"message_stop"}',
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    provider = _make_provider()
    request = ChatRequest(
        model="claude-sonnet-4-5",
        messages=[Message(role="user", content="weather?")],
        stream=True,
    )

    with patch(
        "httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        chunks = [c async for c in provider.chat_completion_stream(request)]

    # Tool call registered with id + name on content_block_start.
    starts = [c for c in chunks if c.tool_calls and c.tool_calls[0].get("id")]
    assert len(starts) == 1
    assert starts[0].tool_calls[0]["id"] == "toolu_1"
    assert starts[0].tool_calls[0]["function"]["name"] == "get_weather"
    assert starts[0].tool_calls[0]["index"] == 0

    # Argument fragments streamed as input_json_delta.
    arg_fragments = "".join(
        c.tool_calls[0]["function"].get("arguments", "")
        for c in chunks
        if c.tool_calls and "arguments" in c.tool_calls[0].get("function", {})
    )
    assert arg_fragments == '{"city":"SF"}'

    # finish_reason mapped tool_use -> tool_calls, with usage attached.
    finals = [c for c in chunks if c.finish_reason]
    assert len(finals) == 1
    assert finals[0].finish_reason == "tool_calls"
    assert finals[0].usage == {
        "prompt_tokens": 42,
        "completion_tokens": 7,
        "total_tokens": 49,
    }


@pytest.mark.asyncio
async def test_stream_text_and_thinking():
    """Text and thinking deltas are surfaced; end_turn maps to stop."""
    body = _sse(
        '{"type":"message_start","message":{"usage":{"input_tokens":5}}}',
        '{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}',
        '{"type":"content_block_delta","index":0,'
        '"delta":{"type":"thinking_delta","thinking":"hmm"}}',
        '{"type":"content_block_delta","index":0,'
        '"delta":{"type":"signature_delta","signature":"sig123"}}',
        '{"type":"content_block_start","index":1,"content_block":{"type":"text"}}',
        '{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}',
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}',
        '{"type":"message_stop"}',
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    provider = _make_provider()
    request = ChatRequest(
        model="claude-sonnet-4-5",
        messages=[Message(role="user", content="hi")],
        stream=True,
        thinking_type="enabled",
    )

    with patch(
        "httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        chunks = [c async for c in provider.chat_completion_stream(request)]

    assert any(c.thinking == "hmm" for c in chunks)
    assert any(c.thinking_signature == "sig123" for c in chunks)
    assert "".join(c.content for c in chunks) == "Hello"

    finals = [c for c in chunks if c.finish_reason]
    assert len(finals) == 1
    assert finals[0].finish_reason == "stop"
