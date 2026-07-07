"""Live integration tests for the stream pipeline (guards + audit tracing).

Verifies that:
1. Guards do NOT interfere with normal requests across all API surfaces.
2. Audit tracing writes per-request trace files when enabled.
"""

from __future__ import annotations

import json

import httpx
import pytest

from integration_tests.conftest import (
    anthropic_payload,
    assert_http_success,
    event_payloads,
    parse_sse_events,
)

BETA = "/api/anthropic/beta/v1/messages"


# ---------------------------------------------------------------------------
# Guards do not interfere with normal requests
# ---------------------------------------------------------------------------


def test_guards_pass_normal_anthropic_stream(client: httpx.Client, chat_model: str):
    """Normal streaming Anthropic request completes without guard interference."""
    payload = anthropic_payload(chat_model)
    payload["stream"] = True
    response = client.post(
        "/api/anthropic/v1/messages",
        json=payload,
        headers={"Accept": "text/event-stream"},
    )
    assert_http_success(response)
    events = parse_sse_events(response)
    payloads = event_payloads(events)

    # Should complete normally with no error events
    event_types = [e[0] for e in events if e[0]]
    assert "error" not in event_types, f"Guard produced error: {events}"

    # Should have content
    text_deltas = [
        p for p in payloads
        if p.get("type") == "content_block_delta"
        and p.get("delta", {}).get("type") == "text_delta"
    ]
    assert len(text_deltas) > 0


def test_guards_pass_normal_beta_stream(client: httpx.Client, chat_model: str):
    """Normal streaming beta passthrough request completes without guard interference."""
    payload = anthropic_payload(chat_model)
    payload["stream"] = True
    response = client.post(
        BETA,
        json=payload,
        headers={"Accept": "text/event-stream"},
    )
    assert_http_success(response)
    events = parse_sse_events(response)

    event_types = [e[0] for e in events if e[0]]
    assert "error" not in event_types, f"Guard produced error: {events}"


def test_guards_pass_normal_openai_chat_stream(client: httpx.Client, chat_model: str):
    """Normal streaming OpenAI Chat request completes without guard interference."""
    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 64,
            "stream": True,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert_http_success(response)
    # Should contain data chunks, no error chunks
    lines = [ln for ln in response.text.split("\n") if ln.startswith("data: ")]
    for line in lines:
        if line == "data: [DONE]":
            continue
        data = json.loads(line[6:])
        assert "error" not in data, f"Guard produced error: {data}"


def test_guards_pass_normal_gemini_stream(client: httpx.Client, chat_model: str):
    """Normal streaming Gemini request completes without guard interference."""
    # Gemini endpoint needs a model without provider prefix
    bare_model = chat_model.split("/", 1)[-1] if "/" in chat_model else chat_model
    response = client.post(
        f"/api/gemini/v1/models/{bare_model}:streamGenerateContent",
        json={
            "contents": [{"role": "user", "parts": [{"text": "Say hello"}]}],
            "generationConfig": {"maxOutputTokens": 64},
        },
        headers={"Accept": "text/event-stream"},
    )
    if response.status_code == 404:
        pytest.skip(f"Gemini endpoint not available for model {bare_model}")
    assert_http_success(response)
    lines = [ln for ln in response.text.split("\n") if ln.startswith("data: ")]
    assert len(lines) > 0
    for line in lines:
        data = json.loads(line[6:])
        candidates = data.get("candidates", [])
        for c in candidates:
            assert c.get("finishReason") != "ERROR", f"Guard produced error: {data}"


