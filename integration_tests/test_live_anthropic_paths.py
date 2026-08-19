"""Live Anthropic-compatible model invocation paths."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from integration_tests.conftest import (
    anthropic_compat_payload,
    anthropic_count_tokens_payload,
    anthropic_payload,
    anthropic_reasoning_capsules,
    anthropic_thinking_replay_payload,
    anthropic_thinking_seed_payload,
    anthropic_tool_payload,
    assert_anthropic_has_tool_use,
    assert_anthropic_usage,
    assert_http_success,
    assert_text_response,
    event_payloads,
    parse_sse_events,
)


def _assistant_content_from_anthropic_stream(
    events: list[tuple[str | None, Any]],
) -> list[dict[str, Any]]:
    """Rebuild the assistant history exactly as Claude Code does from deltas."""
    blocks: dict[int, dict[str, Any]] = {}
    for _event_name, payload in events:
        if not isinstance(payload, dict):
            continue
        frame_type = payload.get("type")
        index = payload.get("index")
        if not isinstance(index, int):
            continue
        if frame_type == "content_block_start":
            block = payload.get("content_block")
            if isinstance(block, dict):
                blocks[index] = dict(block)
            continue
        if frame_type != "content_block_delta" or index not in blocks:
            continue
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            continue
        delta_type = delta.get("type")
        if delta_type == "thinking_delta":
            blocks[index]["thinking"] = blocks[index].get("thinking", "") + delta.get(
                "thinking", ""
            )
        elif delta_type == "signature_delta":
            blocks[index]["signature"] = blocks[index].get("signature", "") + delta.get(
                "signature", ""
            )
        elif delta_type == "text_delta":
            blocks[index]["text"] = blocks[index].get("text", "") + delta.get("text", "")
    return [blocks[index] for index in sorted(blocks)]


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
    assert data["model"] == chat_model
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


@pytest.mark.parametrize(
    "endpoint",
    ("/api/anthropic/v1/messages", "/api/anthropic/beta/v1/messages"),
)
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_anthropic_gpt5_rejects_unsupported_stop_sequences(
    client: httpx.Client,
    gpt5_chat_model: str,
    endpoint: str,
    stream: bool,
):
    payload = anthropic_payload(gpt5_chat_model, stream=stream)
    payload["stop_sequences"] = ["END"]

    response = client.post(endpoint, json=payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()
    assert error["type"] == "error"
    assert error["error"]["type"] == "invalid_request_error"
    assert "stop_sequences" in error["error"]["message"]


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
    assert data["model"] == chat_model
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
    message_start = next(payload for name, payload in events if name == "message_start")
    assert message_start["message"]["model"] == chat_model
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
    message_start = next(payload for name, payload in events if name == "message_start")
    assert message_start["message"]["model"] == chat_model


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

    assert data["stop_reason"] == "tool_use", data
    assert_anthropic_has_tool_use(data, "get_weather")
    assert_anthropic_usage(data["usage"])


def test_anthropic_responses_only_gpt_text(
    client: httpx.Client,
    responses_only_reasoning_model: str,
):
    """Stable Messages endpoint reaches a GPT model exposed only via Responses."""
    payload = anthropic_payload(responses_only_reasoning_model)
    payload.pop("temperature")
    payload["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
    # Responses reasoning tokens share the output budget.  A 16-token probe can
    # legitimately finish after reasoning without producing visible text.
    payload["max_tokens"] = 512
    response = client.post(
        "/api/anthropic/v1/messages",
        json=payload,
        timeout=180.0,
    )
    assert_http_success(response)
    data = response.json()

    assert data["model"] == responses_only_reasoning_model
    text = "".join(
        block.get("text", "") for block in data["content"] if block.get("type") == "text"
    )
    assert_text_response(text)
    assert_anthropic_usage(data["usage"])


def test_anthropic_responses_only_gpt_stream_with_noop_context_management(
    client: httpx.Client,
    responses_only_reasoning_model: str,
):
    """Claude Code's no-op context edit streams through Copilot Responses."""
    payload = anthropic_payload(responses_only_reasoning_model, stream=True)
    payload.pop("temperature")
    payload["max_tokens"] = 512
    payload["system"] = [
        {
            "type": "text",
            "text": "Reply with a short plain-text answer.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    payload["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Reply with exactly: context management stream ok",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    payload["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}

    with client.stream(
        "POST",
        "/api/anthropic/v1/messages?beta=true",
        json=payload,
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    assert "message_start" in event_names
    assert "message_stop" in event_names
    assert any(
        payload.get("delta", {}).get("type") == "text_delta"
        for payload in event_payloads(events)
        if isinstance(payload, dict)
    )


def test_anthropic_responses_only_gpt_stream_reasoning_replays_next_turn(
    client: httpx.Client,
    responses_only_reasoning_model: str,
):
    """Claude Code can persist a streamed capsule and replay it on the next turn."""
    seed = anthropic_thinking_seed_payload(responses_only_reasoning_model)
    seed["stream"] = True
    with client.stream(
        "POST",
        "/api/anthropic/v1/messages",
        json=seed,
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    assistant_content = _assistant_content_from_anthropic_stream(events)
    capsules = anthropic_reasoning_capsules(assistant_content)
    assert len(capsules) == 1, assistant_content
    assert not any(
        block.get("type") == "thinking" and block.get("signature") == ""
        for block in assistant_content
    ), assistant_content
    assert [name for name, _payload in events].count("message_stop") == 1

    replay = client.post(
        "/api/anthropic/v1/messages",
        json=anthropic_thinking_replay_payload(
            responses_only_reasoning_model,
            assistant_content,
        ),
        timeout=180.0,
    )
    assert_http_success(replay)
    data = replay.json()
    text = "".join(
        block.get("text", "") for block in data["content"] if block.get("type") == "text"
    )
    assert_text_response(text)
    assert_anthropic_usage(data["usage"])


def test_anthropic_responses_only_gpt_forced_tool(
    client: httpx.Client,
    responses_only_reasoning_model: str,
):
    """Claude Code-style Anthropic tools work through Copilot Responses."""
    payload = anthropic_tool_payload(responses_only_reasoning_model)
    payload.pop("temperature")
    payload["max_tokens"] = 512
    response = client.post(
        "/api/anthropic/v1/messages",
        json=payload,
        timeout=180.0,
    )
    assert_http_success(response)
    data = response.json()

    assert data["model"] == responses_only_reasoning_model
    assert data["stop_reason"] == "tool_use", data
    assert_anthropic_has_tool_use(data, "get_weather")
    assert_anthropic_usage(data["usage"])


def test_anthropic_responses_only_gpt_forced_tool_streaming(
    client: httpx.Client,
    responses_only_tool_model: str,
):
    """Responses-only Copilot tool streams survive per-event opaque item IDs."""
    payload = anthropic_tool_payload(responses_only_tool_model, stream=True)
    payload.pop("temperature")
    payload["max_tokens"] = 512
    payload["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}

    with client.stream(
        "POST",
        "/api/anthropic/v1/messages?beta=true",
        json=payload,
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    payloads = event_payloads(events)
    assert "error" not in event_names, events
    tool_starts = [
        payload
        for payload in payloads
        if isinstance(payload, dict)
        and payload.get("type") == "content_block_start"
        and payload.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1, payloads
    assert tool_starts[0]["content_block"]["name"] == "get_weather"
    assert any(
        payload.get("delta", {}).get("type") == "input_json_delta"
        for payload in payloads
        if isinstance(payload, dict)
    )
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert_anthropic_usage(message_delta["usage"])
    assert event_names.count("message_stop") == 1


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
