"""Live reasoning and thinking matrix checks."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from integration_tests.conftest import (
    ANTHROPIC_THINKING_BUDGETS,
    OPENAI_REASONING_EFFORTS,
    STREAM_MODES,
    anthropic_effort_payload,
    anthropic_reasoning_payload,
    assert_anthropic_usage,
    assert_http_success,
    assert_openai_usage,
    assert_text_response,
    bare_model,
    event_payloads,
    openai_reasoning_payload,
    parse_sse_events,
)


def test_anthropic_claude_thinking_budget_matrix(
    client: httpx.Client,
    anthropic_thinking_models: list[str],
):
    """Claude-family Anthropic path should handle budget and stream combinations."""
    failures: list[str] = []

    for model in anthropic_thinking_models:
        for budget in ANTHROPIC_THINKING_BUDGETS:
            for stream in STREAM_MODES:
                cell = _cell_id(model, budget=budget, stream=stream)
                try:
                    payload = anthropic_reasoning_payload(model, budget=budget, stream=stream)
                    if stream:
                        data = _post_anthropic_payload_stream(client, payload)
                    else:
                        data = _post_anthropic_payload_non_stream(client, payload)
                    _assert_anthropic_reasoning_result(
                        data,
                        requested_max_tokens=payload["max_tokens"],
                        requested_thinking_budget=budget,
                    )
                except AssertionError as exc:
                    failures.append(f"{cell}: {exc}")

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("stream", STREAM_MODES)
def test_anthropic_unsupported_claude_thinking_returns_native_400(
    client: httpx.Client,
    anthropic_unsupported_thinking_model: str,
    stream: bool,
):
    payload = anthropic_reasoning_payload(
        anthropic_unsupported_thinking_model,
        budget=1024,
        stream=stream,
    )
    response = client.post("/api/anthropic/v1/messages", json=payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "event:" not in response.text


def test_anthropic_gpt5_responses_bridge_thinking_budget_matrix(
    client: httpx.Client,
    anthropic_gpt5_bridge_models: list[str],
):
    """Anthropic wire format should bridge GPT-5 models through GHC Responses."""
    failures: list[str] = []

    for model in anthropic_gpt5_bridge_models:
        for budget in ANTHROPIC_THINKING_BUDGETS:
            for stream in STREAM_MODES:
                cell = _cell_id(model, budget=budget, stream=stream)
                try:
                    payload = anthropic_reasoning_payload(model, budget=budget, stream=stream)
                    if stream:
                        data = _post_anthropic_payload_stream(client, payload)
                    else:
                        data = _post_anthropic_payload_non_stream(client, payload)
                    _assert_anthropic_reasoning_result(
                        data,
                        requested_max_tokens=payload["max_tokens"],
                        requested_thinking_budget=budget,
                    )
                except AssertionError as exc:
                    failures.append(f"{cell}: {exc}")

    assert not failures, "\n".join(failures)


def test_anthropic_enabled_budget_and_output_config_effort(
    client: httpx.Client,
    anthropic_effort_profile: tuple[str, str],
):
    """Standard Anthropic path accepts enabled thinking, budget, and effort together."""
    model, effort = anthropic_effort_profile
    failures: list[str] = []

    for stream in STREAM_MODES:
        payload = anthropic_effort_payload(model, effort=effort, stream=stream)
        try:
            if stream:
                data = _post_anthropic_payload_stream(client, payload)
            else:
                data = _post_anthropic_payload_non_stream(client, payload)
            _assert_anthropic_reasoning_result(
                data,
                requested_max_tokens=payload["max_tokens"],
                requested_thinking_budget=None,
            )
        except AssertionError as exc:
            failures.append(f"{bare_model(model)}|effort={effort}|stream={stream}: {exc}")

    assert not failures, "\n".join(failures)


def test_openai_chat_reasoning_effort_matrix(
    client: httpx.Client,
    openai_reasoning_models: list[str],
):
    """OpenAI Chat should accept reasoning_effort across models and stream modes."""
    failures: list[str] = []

    for model in openai_reasoning_models:
        for effort in OPENAI_REASONING_EFFORTS:
            for stream in STREAM_MODES:
                cell = _cell_id(model, effort=effort, stream=stream)
                try:
                    if stream:
                        data = _post_openai_stream(client, model, effort)
                    else:
                        data = _post_openai_non_stream(client, model, effort)
                    _assert_openai_reasoning_result(data)
                except AssertionError as exc:
                    failures.append(f"{cell}: {exc}")

    assert not failures, "\n".join(failures)


def _post_anthropic_payload_non_stream(
    client: httpx.Client,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(
        "/api/anthropic/v1/messages",
        json=payload,
        timeout=480.0,
    )
    assert_http_success(response)
    return response.json()


def _post_anthropic_payload_stream(
    client: httpx.Client,
    payload: dict[str, Any],
) -> dict[str, Any]:
    with client.stream(
        "POST",
        "/api/anthropic/v1/messages",
        json=payload,
        timeout=480.0,
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


def _post_openai_non_stream(
    client: httpx.Client,
    model: str,
    effort: str | None,
) -> dict[str, Any]:
    response = client.post(
        "/api/openai/v1/chat/completions",
        json=openai_reasoning_payload(model, effort=effort),
        timeout=480.0,
    )
    assert_http_success(response)
    return response.json()


def _post_openai_stream(
    client: httpx.Client,
    model: str,
    effort: str | None,
) -> dict[str, Any]:
    with client.stream(
        "POST",
        "/api/openai/v1/chat/completions",
        json=openai_reasoning_payload(model, effort=effort, stream=True),
        timeout=480.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    assert events[-1][1] == "[DONE]"
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "".join(
                        content
                        for payload in payloads
                        for choice in payload.get("choices", [])
                        for content in [choice.get("delta", {}).get("content")]
                        if isinstance(content, str)
                    ),
                },
                "finish_reason": next(
                    (
                        choice.get("finish_reason")
                        for payload in payloads
                        for choice in payload.get("choices", [])
                        if choice.get("finish_reason")
                    ),
                    None,
                ),
            }
        ],
        "usage": next(
            (payload["usage"] for payload in reversed(payloads) if payload.get("usage")),
            None,
        ),
    }


def _assert_anthropic_reasoning_result(
    data: dict[str, Any],
    *,
    requested_max_tokens: int,
    requested_thinking_budget: int | None,
) -> None:
    assert type(requested_max_tokens) is int and requested_max_tokens > 0
    assert requested_thinking_budget is None or (
        type(requested_thinking_budget) is int
        and 0 < requested_thinking_budget < requested_max_tokens
    )
    blocks = data.get("content") or []
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    thinking = "".join(
        block.get("thinking", "") for block in blocks if block.get("type") == "thinking"
    )
    stop_reason = data.get("stop_reason")
    assert stop_reason in {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
        "refusal",
    }
    usage = data.get("usage")
    assert isinstance(usage, dict)
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    assert type(input_tokens) is int and input_tokens > 0
    assert type(output_tokens) is int and output_tokens >= 0
    assert_anthropic_usage(usage)
    if not (text.strip() or thinking.strip()):
        assert stop_reason == "max_tokens", data
        assert output_tokens > 0, data


def _assert_openai_reasoning_result(data: dict[str, Any]) -> None:
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert_text_response(choice["message"].get("content"))
    assert choice.get("finish_reason") in {"stop", "length", "content_filter", None}
    if data.get("usage"):
        assert_openai_usage(data["usage"])


def _cell_id(
    model: str,
    *,
    budget: int | None = None,
    effort: str | None = None,
    stream: bool,
) -> str:
    knob = f"budget={budget}" if effort is None else f"effort={effort}"
    mode = "stream" if stream else "nonstream"
    return f"{bare_model(model)} {knob} {mode}"
