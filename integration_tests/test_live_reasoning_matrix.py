"""Live reasoning and thinking matrix checks."""

from __future__ import annotations

from typing import Any

import httpx

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
                    if stream:
                        data = _post_anthropic_stream(client, model, budget)
                    else:
                        data = _post_anthropic_non_stream(client, model, budget)
                    _assert_anthropic_reasoning_result(data)
                except AssertionError as exc:
                    failures.append(f"{cell}: {exc}")

    assert not failures, "\n".join(failures)


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
                    if stream:
                        data = _post_anthropic_stream(client, model, budget)
                    else:
                        data = _post_anthropic_non_stream(client, model, budget)
                    _assert_anthropic_reasoning_result(data)
                except AssertionError as exc:
                    failures.append(f"{cell}: {exc}")

    assert not failures, "\n".join(failures)


def test_anthropic_output_config_effort_precedence(
    client: httpx.Client,
    anthropic_effort_profile: tuple[str, str],
):
    """Standard Anthropic path should prefer explicit effort over a low budget."""
    model, effort = anthropic_effort_profile
    failures: list[str] = []

    for stream in STREAM_MODES:
        payload = anthropic_effort_payload(model, effort=effort, stream=stream)
        try:
            if stream:
                data = _post_anthropic_payload_stream(client, payload)
            else:
                response = client.post(
                    "/api/anthropic/v1/messages",
                    json=payload,
                    timeout=480.0,
                )
                assert_http_success(response)
                data = response.json()
            _assert_anthropic_reasoning_result(data)
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


def _post_anthropic_non_stream(
    client: httpx.Client,
    model: str,
    budget: int | None,
) -> dict[str, Any]:
    response = client.post(
        "/api/anthropic/v1/messages",
        json=anthropic_reasoning_payload(model, budget=budget),
        timeout=480.0,
    )
    assert_http_success(response)
    return response.json()


def _post_anthropic_stream(
    client: httpx.Client,
    model: str,
    budget: int | None,
) -> dict[str, Any]:
    return _post_anthropic_payload_stream(
        client,
        anthropic_reasoning_payload(model, budget=budget, stream=True),
    )


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


def _assert_anthropic_reasoning_result(data: dict[str, Any]) -> None:
    blocks = data.get("content") or []
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    thinking = "".join(
        block.get("thinking", "") for block in blocks if block.get("type") == "thinking"
    )
    assert text.strip() or thinking.strip(), data
    assert data.get("stop_reason") in {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
        "refusal",
    }
    assert_anthropic_usage(data["usage"])


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
