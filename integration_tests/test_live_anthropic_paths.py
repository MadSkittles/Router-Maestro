"""Live Anthropic-compatible model invocation paths."""

from __future__ import annotations

import httpx

from integration_tests.conftest import (
    anthropic_compat_payload,
    anthropic_count_tokens_payload,
    anthropic_payload,
    anthropic_tool_payload,
    assert_anthropic_has_tool_use,
    assert_anthropic_usage,
    assert_http_success,
    assert_text_response,
    event_payloads,
    parse_sse_events,
)


def test_anthropic_messages_non_streaming_api_prefix(
    client: httpx.Client,
    chat_model: str,
):
    """The prefixed Anthropic Messages path should route to GHC."""
    response = client.post(
        "/api/anthropic/v1/messages",
        json=anthropic_compat_payload(chat_model),
    )
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"]
    assert data["stop_reason"] in {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
        "refusal",
    }
    text_blocks = [block for block in data["content"] if block["type"] == "text"]
    assert text_blocks, data
    assert_text_response(text_blocks[0]["text"])
    assert_anthropic_usage(data["usage"])


def test_anthropic_messages_non_streaming_root_path(
    client: httpx.Client,
    chat_model: str,
):
    """The Claude-compatible /v1/messages path should route to GHC."""
    response = client.post("/v1/messages", json=anthropic_payload(chat_model))
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert_anthropic_usage(data["usage"])


def test_anthropic_messages_streaming_api_prefix(
    client: httpx.Client,
    chat_model: str,
):
    """The prefixed Anthropic stream should emit Anthropic protocol events."""
    with client.stream(
        "POST",
        "/api/anthropic/v1/messages",
        json=anthropic_payload(chat_model, stream=True),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    payloads = event_payloads(events)

    assert "message_start" in event_names
    assert "content_block_start" in event_names
    assert "content_block_delta" in event_names
    assert "message_delta" in event_names
    assert "message_stop" in event_names
    assert any(
        payload.get("delta", {}).get("type") == "text_delta"
        for payload in payloads
        if isinstance(payload, dict)
    )
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert_anthropic_usage(message_delta["usage"])


def test_anthropic_messages_streaming_root_path(
    client: httpx.Client,
    chat_model: str,
):
    """The /v1/messages stream should emit Anthropic protocol events."""
    with client.stream(
        "POST",
        "/v1/messages",
        json=anthropic_payload(chat_model, stream=True),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    assert "message_start" in event_names
    assert "message_stop" in event_names


def test_anthropic_count_tokens_api_prefix(client: httpx.Client, chat_model: str):
    """The prefixed Anthropic count_tokens path should estimate input tokens."""
    response = client.post(
        "/api/anthropic/v1/messages/count_tokens",
        json=anthropic_count_tokens_payload(chat_model),
    )
    assert_http_success(response)
    assert response.json()["input_tokens"] > 0


def test_anthropic_count_tokens_root_path(client: httpx.Client, chat_model: str):
    """The /v1/messages/count_tokens path should estimate input tokens."""
    response = client.post(
        "/v1/messages/count_tokens",
        json=anthropic_count_tokens_payload(chat_model),
    )
    assert_http_success(response)
    assert response.json()["input_tokens"] > 0


def test_anthropic_forced_tool_call(client: httpx.Client, tool_model: str):
    """Anthropic Messages should translate OpenAI tool calls to tool_use blocks."""
    response = client.post(
        "/api/anthropic/v1/messages",
        json=anthropic_tool_payload(tool_model),
    )
    assert_http_success(response)
    data = response.json()

    assert data["stop_reason"] == "tool_use"
    assert_anthropic_has_tool_use(data, "get_weather")
    assert_anthropic_usage(data["usage"])


def test_anthropic_forced_tool_call_streaming(
    client: httpx.Client,
    tool_model: str,
):
    """Anthropic streaming should expose tool_use block events."""
    with client.stream(
        "POST",
        "/api/anthropic/v1/messages",
        json=anthropic_tool_payload(tool_model, stream=True),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    tool_start = [
        payload
        for payload in payloads
        if isinstance(payload, dict)
        and payload.get("type") == "content_block_start"
        and payload.get("content_block", {}).get("type") == "tool_use"
    ]
    assert tool_start, payloads
    assert any(
        payload.get("delta", {}).get("type") == "input_json_delta"
        for payload in payloads
        if isinstance(payload, dict)
    )
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert_anthropic_usage(message_delta["usage"])
