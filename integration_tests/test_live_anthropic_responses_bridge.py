"""Live integration tests for the Anthropic → Responses API bridge.

These tests verify that requests arriving in Anthropic Messages format
are correctly translated and routed through Copilot's /responses endpoint
for GPT-5 family models. Covers text, multimodal, tool use, and streaming.

The local server is started with ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API=1
(see conftest.py), so responses-eligible models automatically use the bridge.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from integration_tests.conftest import (
    STREAM_MODES,
    TEXT_PROMPT,
    TOOL_PROMPT,
    anthropic_weather_tool,
    assert_anthropic_has_tool_use,
    assert_anthropic_usage,
    assert_http_success,
    assert_text_response,
    event_payloads,
    parse_sse_events,
)

ENDPOINT = "/api/anthropic/v1/messages"

# A minimal valid 1x1 red PNG for multimodal tests.
_1PX_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _bridge_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Anthropic Messages payload without temperature (GPT-5 rejects it on /responses)."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
        "max_tokens": 512,
        "stream": stream,
    }


def _bridge_tool_payload(model: str, *, stream: bool = False) -> dict[str, Any]:
    """Anthropic tool payload without temperature."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": TOOL_PROMPT}],
        "max_tokens": 1024,
        "tools": [anthropic_weather_tool()],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "stream": stream,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bridge_model(anthropic_gpt5_bridge_models: list[str]) -> str:
    """Pick one responses-eligible model for the bridge tests."""
    preferred = (
        "github-copilot/gpt-5.4-mini",
        "github-copilot/gpt-5.4",
        "github-copilot/gpt-5.5",
    )
    available = set(anthropic_gpt5_bridge_models)
    for model in preferred:
        if model in available:
            return model
    return anthropic_gpt5_bridge_models[0]


# ---------------------------------------------------------------------------
# Basic text — non-streaming and streaming
# ---------------------------------------------------------------------------


def test_bridge_text_non_streaming(client: httpx.Client, bridge_model: str):
    """Plain text request through Anthropic→Responses bridge returns valid response."""
    response = client.post(ENDPOINT, json=_bridge_payload(bridge_model))
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"]
    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert text_blocks, data
    assert_text_response(text_blocks[0]["text"])
    assert_anthropic_usage(data["usage"])


def test_bridge_text_streaming(client: httpx.Client, bridge_model: str):
    """Streaming text through the bridge emits correct Anthropic protocol events."""
    with client.stream(
        "POST",
        ENDPOINT,
        json=_bridge_payload(bridge_model, stream=True),
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


# ---------------------------------------------------------------------------
# Tool use
# ---------------------------------------------------------------------------


def test_bridge_forced_tool_call(client: httpx.Client, bridge_model: str):
    """Forced tool_use through the bridge returns a valid tool_use block."""
    response = client.post(ENDPOINT, json=_bridge_tool_payload(bridge_model))
    assert_http_success(response)
    data = response.json()

    assert data["stop_reason"] == "tool_use"
    assert_anthropic_has_tool_use(data, "get_weather")
    assert_anthropic_usage(data["usage"])


def test_bridge_forced_tool_call_streaming(client: httpx.Client, bridge_model: str):
    """Streaming forced tool_use through the bridge emits tool block events."""
    with client.stream(
        "POST",
        ENDPOINT,
        json=_bridge_tool_payload(bridge_model, stream=True),
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
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert_anthropic_usage(message_delta["usage"])


def test_bridge_tool_choice_any(client: httpx.Client, bridge_model: str):
    """tool_choice=any through the bridge forces some tool call."""
    payload: dict[str, Any] = {
        "model": bridge_model,
        "messages": [{"role": "user", "content": TOOL_PROMPT}],
        "max_tokens": 1024,
        "tools": [anthropic_weather_tool()],
        "tool_choice": {"type": "any"},
    }
    response = client.post(ENDPOINT, json=payload)
    assert_http_success(response)
    data = response.json()

    assert data["stop_reason"] == "tool_use"
    assert_anthropic_has_tool_use(data, "get_weather")


# ---------------------------------------------------------------------------
# Multimodal (image)
# ---------------------------------------------------------------------------


def test_bridge_image_non_streaming(client: httpx.Client, bridge_model: str):
    """Image content in Anthropic format is bridged to Responses input_image."""
    payload: dict[str, Any] = {
        "model": bridge_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this pixel? Reply with one word."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _1PX_PNG_B64,
                        },
                    },
                ],
            }
        ],
        "max_tokens": 512,
    }
    response = client.post(ENDPOINT, json=payload)
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert text_blocks, data
    assert_text_response(text_blocks[0]["text"])
    assert_anthropic_usage(data["usage"])


def test_bridge_image_streaming(client: httpx.Client, bridge_model: str):
    """Streaming image request through the bridge returns valid events."""
    payload: dict[str, Any] = {
        "model": bridge_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this pixel? Reply with one word."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _1PX_PNG_B64,
                        },
                    },
                ],
            }
        ],
        "max_tokens": 512,
        "stream": True,
    }
    with client.stream("POST", ENDPOINT, json=payload, timeout=180.0) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    assert "message_start" in event_names
    assert "content_block_delta" in event_names
    assert "message_stop" in event_names


# ---------------------------------------------------------------------------
# System message / instructions
# ---------------------------------------------------------------------------


def test_bridge_system_message(client: httpx.Client, bridge_model: str):
    """Anthropic system field maps to Responses instructions."""
    payload: dict[str, Any] = {
        "model": bridge_model,
        "system": "Always reply with exactly the word pong.",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 512,
    }
    response = client.post(ENDPOINT, json=payload)
    assert_http_success(response)
    data = response.json()

    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert text_blocks, data
    assert "pong" in text_blocks[0]["text"].lower()


# ---------------------------------------------------------------------------
# Multi-turn conversation
# ---------------------------------------------------------------------------


def test_bridge_multi_turn(client: httpx.Client, bridge_model: str):
    """Multi-turn conversation history is correctly bridged."""
    payload: dict[str, Any] = {
        "model": bridge_model,
        "messages": [
            {"role": "user", "content": "Remember: the secret word is banana."},
            {"role": "assistant", "content": "I'll remember that the secret word is banana."},
            {"role": "user", "content": "What is the secret word? Reply with just the word."},
        ],
        "max_tokens": 512,
    }
    response = client.post(ENDPOINT, json=payload)
    assert_http_success(response)
    data = response.json()

    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert text_blocks, data
    assert "banana" in text_blocks[0]["text"].lower()


# ---------------------------------------------------------------------------
# Compatibility fields
# ---------------------------------------------------------------------------


def test_bridge_compat_fields(client: httpx.Client, responses_top_p_model: str):
    """Representable Anthropic fields (top_p and metadata) survive the bridge."""
    payload: dict[str, Any] = {
        "model": responses_top_p_model,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
        "max_tokens": 512,
        "top_p": 1,
        "metadata": {"user_id": "integration-test"},
    }
    response = client.post(ENDPOINT, json=payload)
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert text_blocks, data
    assert_text_response(text_blocks[0]["text"])


@pytest.mark.parametrize("stream", [False, True])
def test_bridge_stop_sequences_returns_native_400(
    client: httpx.Client,
    bridge_model: str,
    stream: bool,
):
    """An unrepresentable stop_sequences option is rejected before bridge I/O."""
    payload = _bridge_payload(bridge_model, stream=stream)
    payload["stop_sequences"] = ["END"]
    response = client.post(ENDPOINT, json=payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "event:" not in response.text


# ---------------------------------------------------------------------------
# Stream modes matrix (broad coverage)
# ---------------------------------------------------------------------------


def test_bridge_stream_modes_matrix(client: httpx.Client, bridge_model: str):
    """Both streaming and non-streaming produce valid Anthropic responses."""
    failures: list[str] = []

    for stream in STREAM_MODES:
        try:
            if stream:
                with client.stream(
                    "POST",
                    ENDPOINT,
                    json=_bridge_payload(bridge_model, stream=True),
                    timeout=180.0,
                ) as response:
                    assert_http_success(response)
                    events = parse_sse_events(response)
                event_names = [name for name, _ in events]
                assert "message_start" in event_names
                assert "message_stop" in event_names
            else:
                response = client.post(ENDPOINT, json=_bridge_payload(bridge_model))
                assert_http_success(response)
                data = response.json()
                assert data["type"] == "message"
                text_blocks = [b for b in data["content"] if b["type"] == "text"]
                assert text_blocks
        except (AssertionError, httpx.HTTPError) as exc:
            failures.append(f"stream={stream}: {exc}")

    assert not failures, "\n".join(failures)
