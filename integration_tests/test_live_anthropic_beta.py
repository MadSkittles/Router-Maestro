"""Live integration tests for the Anthropic beta passthrough endpoint.

Mirrors the coverage of ``test_live_anthropic_paths.py`` and
``test_live_reasoning_matrix.py`` against the beta endpoint to ensure
no feature gap between the standard (translation-based) and beta
(native passthrough) paths.
"""

from __future__ import annotations

from typing import Any

import httpx

from integration_tests.conftest import (
    ANTHROPIC_THINKING_BUDGETS,
    STREAM_MODES,
    anthropic_compat_payload,
    anthropic_count_tokens_payload,
    anthropic_effort_payload,
    anthropic_payload,
    anthropic_reasoning_payload,
    anthropic_tool_choice_any_payload,
    anthropic_tool_payload,
    assert_anthropic_has_tool_use,
    assert_anthropic_usage,
    assert_http_success,
    assert_text_response,
    bare_model,
    event_payloads,
    parse_sse_events,
)

BETA = "/api/anthropic/beta/v1/messages"
BETA_COUNT = "/api/anthropic/beta/v1/messages/count_tokens"


# ---------------------------------------------------------------------------
# Basic messages — mirrors test_live_anthropic_paths.py
# ---------------------------------------------------------------------------


def test_beta_non_streaming(client: httpx.Client, chat_model: str):
    """Beta endpoint returns a valid Anthropic response."""
    response = client.post(BETA, json=anthropic_payload(chat_model))
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
    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert text_blocks, data
    assert_text_response(text_blocks[0]["text"])
    assert_anthropic_usage(data["usage"])
    # Copilot-internal fields must be stripped
    assert "copilot_usage" not in data
    assert "stop_details" not in data


def test_beta_non_streaming_compat_fields(client: httpx.Client, chat_model: str):
    """Beta endpoint handles compat fields (system, top_p, stop_sequences, metadata)."""
    response = client.post(BETA, json=anthropic_compat_payload(chat_model))
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert text_blocks, data
    assert_anthropic_usage(data["usage"])


def test_beta_strips_unknown_fields(client: httpx.Client, chat_model: str):
    """Beta endpoint strips unknown fields (context_management, etc.) without error."""
    payload = anthropic_payload(chat_model)
    payload["context_management"] = {"enabled": True}
    payload["output_config"] = {"format": "text"}
    payload["service_tier"] = "standard"
    response = client.post(BETA, json=payload)
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    assert_anthropic_usage(data["usage"])


def test_beta_streaming(client: httpx.Client, chat_model: str):
    """Beta streaming emits standard Anthropic SSE event sequence."""
    with client.stream(
        "POST",
        BETA,
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
    # copilot_usage event must be filtered
    assert "copilot_usage" not in event_names
    # Must have text_delta
    assert any(
        payload.get("delta", {}).get("type") == "text_delta"
        for payload in payloads
        if isinstance(payload, dict)
    )
    message_delta = next(payload for name, payload in events if name == "message_delta")
    assert_anthropic_usage(message_delta["usage"])


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


def test_beta_count_tokens(client: httpx.Client, chat_model: str):
    """Beta count_tokens returns positive input_tokens."""
    response = client.post(
        BETA_COUNT,
        json=anthropic_count_tokens_payload(chat_model),
    )
    assert_http_success(response)
    assert response.json()["input_tokens"] > 0


# ---------------------------------------------------------------------------
# Tool use — mirrors test_live_anthropic_paths.py tool tests
# ---------------------------------------------------------------------------


def test_beta_forced_tool_call(client: httpx.Client, tool_model: str):
    """Beta endpoint supports forced tool_use blocks."""
    response = client.post(BETA, json=anthropic_tool_payload(tool_model))
    assert_http_success(response)
    data = response.json()

    assert data["stop_reason"] == "tool_use"
    assert_anthropic_has_tool_use(data, "get_weather")
    assert_anthropic_usage(data["usage"])


def test_beta_forced_tool_call_streaming(client: httpx.Client, tool_model: str):
    """Beta streaming supports tool_use block events."""
    with client.stream(
        "POST",
        BETA,
        json=anthropic_tool_payload(tool_model, stream=True),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    tool_start = [
        p
        for p in payloads
        if isinstance(p, dict)
        and p.get("type") == "content_block_start"
        and p.get("content_block", {}).get("type") == "tool_use"
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


def test_beta_tool_choice_any(client: httpx.Client, tool_model: str):
    """Beta endpoint handles tool_choice type=any."""
    response = client.post(BETA, json=anthropic_tool_choice_any_payload(tool_model))
    assert_http_success(response)
    data = response.json()

    assert data["stop_reason"] == "tool_use"
    assert_anthropic_has_tool_use(data, "get_weather")


# ---------------------------------------------------------------------------
# Thinking / reasoning — mirrors test_live_reasoning_matrix.py
# ---------------------------------------------------------------------------


def test_beta_thinking_non_streaming(client: httpx.Client, chat_model: str):
    """Beta endpoint supports extended thinking natively."""
    payload = {
        "model": chat_model,
        "messages": [{"role": "user", "content": "What is 7*8?"}],
        "max_tokens": 8000,
        "thinking": {"type": "enabled", "budget_tokens": 4000},
    }
    response = client.post(BETA, json=payload, timeout=60)
    assert_http_success(response)
    data = response.json()

    thinking_blocks = [b for b in data["content"] if b["type"] == "thinking"]
    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert thinking_blocks, f"Expected thinking block, got: {data['content']}"
    assert thinking_blocks[0].get("signature"), "Thinking block should have signature"
    assert text_blocks
    assert_anthropic_usage(data["usage"])


def test_beta_thinking_streaming(client: httpx.Client, chat_model: str):
    """Beta streaming surfaces thinking_delta and signature_delta events."""
    payload = {
        "model": chat_model,
        "messages": [{"role": "user", "content": "What is 3+5?"}],
        "max_tokens": 8000,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 4000},
    }
    with client.stream("POST", BETA, json=payload, timeout=120.0) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    payloads = event_payloads(events)

    assert "message_start" in event_names
    assert "message_stop" in event_names
    assert "copilot_usage" not in event_names

    # Must have a thinking content_block_start
    thinking_start = [
        p
        for p in payloads
        if isinstance(p, dict)
        and p.get("type") == "content_block_start"
        and p.get("content_block", {}).get("type") == "thinking"
    ]
    assert thinking_start, "Expected thinking block start in stream"

    # Must have thinking_delta events
    thinking_deltas = [
        p
        for p in payloads
        if isinstance(p, dict) and p.get("delta", {}).get("type") == "thinking_delta"
    ]
    assert thinking_deltas, "Expected thinking_delta events"

    # Must have signature_delta
    sig_deltas = [
        p
        for p in payloads
        if isinstance(p, dict) and p.get("delta", {}).get("type") == "signature_delta"
    ]
    assert sig_deltas, "Expected signature_delta events"


def test_beta_output_config_effort_precedence(
    client: httpx.Client,
    anthropic_effort_profile: tuple[str, str],
):
    """Native path should preserve effort and remove its conflicting budget."""
    model, effort = anthropic_effort_profile
    failures: list[str] = []

    for stream in STREAM_MODES:
        payload = anthropic_effort_payload(model, effort=effort, stream=stream)
        try:
            if stream:
                with client.stream("POST", BETA, json=payload, timeout=480.0) as response:
                    assert_http_success(response)
                    events = parse_sse_events(response)
                event_names = [name for name, _event in events]
                assert "message_stop" in event_names, events
                errors = [event for name, event in events if name == "error"]
                assert not errors, errors
            else:
                response = client.post(BETA, json=payload, timeout=480.0)
                assert_http_success(response)
                data = response.json()
                assert data["type"] == "message"
                assert_anthropic_usage(data["usage"])
        except AssertionError as exc:
            failures.append(f"{bare_model(model)}|effort={effort}|stream={stream}: {exc}")

    assert not failures, "\n".join(failures)


def test_beta_thinking_budget_matrix(
    client: httpx.Client,
    anthropic_thinking_models: list[str],
):
    """Beta endpoint handles thinking budget × stream mode matrix."""
    failures: list[str] = []

    for model in anthropic_thinking_models:
        for budget in ANTHROPIC_THINKING_BUDGETS:
            for stream in STREAM_MODES:
                cell = f"{bare_model(model)}|budget={budget}|stream={stream}"
                try:
                    if stream:
                        data = _post_beta_stream(client, model, budget)
                    else:
                        data = _post_beta_non_stream(client, model, budget)
                    _assert_thinking_result(data, budget)
                except AssertionError as exc:
                    failures.append(f"{cell}: {exc}")

    assert not failures, "\n".join(failures)


def test_beta_thinking_replay(client: httpx.Client, chat_model: str):
    """Multi-turn with prior thinking block triggers try-forward retry.

    The beta route first attempts to forward with the thinking block intact.
    When Copilot rejects the invalid signature, it strips thinking blocks
    and retries automatically — the client should see a successful response.
    """
    payload = {
        "model": chat_model,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Reply with just the number."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me calculate: 2+2=4",
                        "signature": "invalid-sig-from-old-route",
                    },
                    {"type": "text", "text": "4"},
                ],
            },
            {"role": "user", "content": "Now reply with exactly the word pong."},
        ],
    }
    response = client.post(BETA, json=payload, timeout=60)
    assert_http_success(response)
    data = response.json()

    text_blocks = [b for b in data["content"] if b.get("type") == "text"]
    assert text_blocks, data


def test_beta_thinking_replay_streaming(client: httpx.Client, chat_model: str):
    """Streaming with invalid signature triggers try-forward retry."""
    payload = {
        "model": chat_model,
        "max_tokens": 64,
        "stream": True,
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Reply with just the number."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "2+2=4",
                        "signature": "bad-sig",
                    },
                    {"type": "text", "text": "4"},
                ],
            },
            {"role": "user", "content": "Now reply with exactly the word pong."},
        ],
    }
    with client.stream("POST", BETA, json=payload, timeout=120.0) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    assert "message_start" in event_names
    assert "message_stop" in event_names


# ---------------------------------------------------------------------------
# Fallback path (non-Claude models)
# ---------------------------------------------------------------------------


def test_beta_fallback_non_claude_model(client: httpx.Client, copilot_models: list[str]):
    """Non-Claude model on beta endpoint falls back to standard translation."""
    chat_capable = [
        m
        for m in copilot_models
        if "claude" not in m.lower()
        and ("gpt-5-mini" in m.lower() or "gpt-5.4" == m.split("/")[-1].lower())
    ]
    if not chat_capable:
        import pytest

        pytest.skip("No chat-capable non-Claude models available")

    model = chat_capable[0]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 16,
    }
    response = client.post(BETA, json=payload, timeout=60)
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert_anthropic_usage(data["usage"])


def test_beta_fallback_non_claude_streaming(client: httpx.Client, copilot_models: list[str]):
    """Non-Claude streaming on beta endpoint falls back and produces valid SSE."""
    chat_capable = [
        m
        for m in copilot_models
        if "claude" not in m.lower()
        and ("gpt-5-mini" in m.lower() or "gpt-5.4" == m.split("/")[-1].lower())
    ]
    if not chat_capable:
        import pytest

        pytest.skip("No chat-capable non-Claude models available")

    model = chat_capable[0]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 16,
        "stream": True,
    }
    with client.stream("POST", BETA, json=payload, timeout=180.0) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    assert "message_start" in event_names
    assert "message_stop" in event_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post_beta_non_stream(
    client: httpx.Client,
    model: str,
    budget: int | None,
) -> dict[str, Any]:
    response = client.post(
        BETA,
        json=anthropic_reasoning_payload(model, budget=budget),
        timeout=240.0,
    )
    assert_http_success(response)
    return response.json()


def _post_beta_stream(
    client: httpx.Client,
    model: str,
    budget: int | None,
) -> dict[str, Any]:
    with client.stream(
        "POST",
        BETA,
        json=anthropic_reasoning_payload(model, budget=budget, stream=True),
        timeout=240.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    errors = [payload for payload in payloads if payload.get("type") == "error"]
    assert not errors, errors
    assert "message_stop" in [name for name, _payload in events]

    blocks = [
        payload.get("content_block", {})
        for payload in payloads
        if payload.get("type") == "content_block_start"
    ]
    text = "".join(
        payload.get("delta", {}).get("text", "")
        for payload in payloads
        if payload.get("delta", {}).get("type") == "text_delta"
    )
    thinking = "".join(
        payload.get("delta", {}).get("thinking", "")
        for payload in payloads
        if payload.get("delta", {}).get("type") == "thinking_delta"
    )
    message_delta = next(payload for name, payload in events if name == "message_delta")
    return {
        "content": [
            {"type": block.get("type"), "text": text, "thinking": thinking} for block in blocks
        ],
        "usage": message_delta["usage"],
        "stop_reason": message_delta["delta"].get("stop_reason"),
    }


def _assert_thinking_result(data: dict[str, Any], budget: int | None) -> None:
    """Assert thinking/reasoning result from the beta endpoint."""
    assert_anthropic_usage(data["usage"])

    # With thinking enabled, content must have at least one block
    assert data["content"], f"Empty content: {data}"

    if budget is not None:
        thinking_blocks = [b for b in data["content"] if b.get("type") == "thinking"]
        assert thinking_blocks, f"Expected thinking with budget={budget}: {data['content']}"
    else:
        # Without thinking budget, must have text
        text_blocks = [b for b in data["content"] if b.get("type") == "text"]
        assert text_blocks, f"No text block without thinking: {data['content']}"
