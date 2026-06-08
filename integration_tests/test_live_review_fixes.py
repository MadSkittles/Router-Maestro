"""Live coverage for the codebase-review bug fixes.

These exercise the externally observable behaviours of the fixes against the
real GitHub Copilot backend:

* ``tool_choice`` translation (Anthropic ``any`` / OpenAI ``required``) — must
  actually force a tool call rather than silently degrading to ``None``.
* Thinking-block replay must not break multi-turn Anthropic requests.
* Gemini streaming must emit a usage-bearing final event even when content and
  finish arrive close together.
* API-key auth rejects a wrong key (constant-time comparison still rejects).
"""

from __future__ import annotations

import httpx

from integration_tests.conftest import (
    anthropic_thinking_replay_payload,
    anthropic_tool_choice_any_payload,
    assert_anthropic_has_tool_use,
    assert_anthropic_usage,
    assert_gemini_usage,
    assert_http_success,
    assert_text_response,
    bare_model,
    event_payloads,
    gemini_payload,
    openai_chat_tool_required_payload,
    parse_sse_events,
)


def test_anthropic_tool_choice_any_forces_tool(client: httpx.Client, tool_model: str):
    """Anthropic ``tool_choice={"type":"any"}`` must force a tool_use block.

    Regression for ``_translate_tool_choice`` returning ``None`` for Pydantic
    objects (it now maps ``any`` -> OpenAI ``required``).
    """
    response = client.post(
        "/api/anthropic/v1/messages",
        json=anthropic_tool_choice_any_payload(tool_model),
    )
    assert_http_success(response)
    data = response.json()

    assert data["stop_reason"] == "tool_use", data
    assert_anthropic_has_tool_use(data, "get_weather")
    assert_anthropic_usage(data["usage"])


def test_openai_tool_choice_required_forces_tool(client: httpx.Client, tool_model: str):
    """OpenAI ``tool_choice="required"`` must force some tool call."""
    response = client.post(
        "/api/openai/v1/chat/completions",
        json=openai_chat_tool_required_payload(tool_model),
    )
    assert_http_success(response)
    data = response.json()

    message = data["choices"][0]["message"]
    assert message.get("tool_calls"), data
    assert data["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_thinking_replay_multiturn(client: httpx.Client, chat_model: str):
    """A history containing a prior thinking block must not break the next turn."""
    response = client.post(
        "/api/anthropic/v1/messages",
        json=anthropic_thinking_replay_payload(chat_model),
    )
    assert_http_success(response)
    data = response.json()

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    text_blocks = [block for block in data["content"] if block["type"] == "text"]
    assert text_blocks, data
    assert_text_response(text_blocks[0]["text"])
    assert_anthropic_usage(data["usage"])


def test_gemini_stream_emits_final_usage(client: httpx.Client, chat_model: str):
    """Gemini streaming must always surface a usage-bearing final event.

    Regression for the early-return that dropped the finish event when content
    and finish_reason arrived in the same upstream chunk.
    """
    with client.stream(
        "POST",
        f"/api/gemini/v1beta/models/{bare_model(chat_model)}:streamGenerateContent",
        json=gemini_payload(max_output_tokens=32),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    assert payloads, events

    # Some event carried visible text...
    assert any(
        part.get("text")
        for payload in payloads
        for candidate in payload.get("candidates", [])
        for part in (candidate.get("content") or {}).get("parts", [])
    ), payloads
    # ...and a finishReason was delivered (not dropped) with final usage.
    assert any(
        candidate.get("finishReason")
        for payload in payloads
        for candidate in payload.get("candidates", [])
    ), payloads
    final_with_usage = [p for p in payloads if p.get("usageMetadata")]
    assert final_with_usage, payloads
    assert_gemini_usage(final_with_usage[-1]["usageMetadata"])


def test_wrong_api_key_is_rejected(
    unauthenticated_client: httpx.Client,
    chat_model: str,
):
    """A wrong API key must be rejected with 401 (constant-time comparison)."""
    response = unauthenticated_client.post(
        "/api/openai/v1/chat/completions",
        headers={"Authorization": "Bearer definitely-not-the-key"},
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    assert response.status_code == 401, response.text
