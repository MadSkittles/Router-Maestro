"""Live OpenAI-compatible model invocation paths."""

from __future__ import annotations

from typing import Any

import httpx

from integration_tests.conftest import (
    assert_http_success,
    assert_openai_usage,
    assert_response_has_function_call,
    assert_responses_usage,
    assert_text_response,
    assert_tool_call_name,
    event_payloads,
    openai_chat_tool_payload,
    openai_chat_usage_payload,
    openai_responses_payload,
    openai_responses_tool_payload,
    parse_sse_events,
)


def test_openai_chat_completion_non_streaming_returns_usage(
    client: httpx.Client,
    chat_model: str,
):
    """OpenAI Chat non-streaming should route to GHC and return usage."""
    response = client.post(
        "/api/openai/v1/chat/completions",
        json=openai_chat_usage_payload(chat_model),
    )
    assert_http_success(response)
    data = response.json()

    assert data["object"] == "chat.completion"
    assert data["model"]
    assert len(data["choices"]) == 1
    message = data["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert_text_response(message["content"])
    assert_openai_usage(data["usage"])


def test_openai_chat_completion_streaming_returns_chunks_and_done(
    client: httpx.Client,
    chat_model: str,
):
    """OpenAI Chat streaming should return SSE chunks and the [DONE] sentinel."""
    with client.stream(
        "POST",
        "/api/openai/v1/chat/completions",
        json=openai_chat_usage_payload(chat_model, stream=True),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    assert events[-1][1] == "[DONE]"
    assert any(payload.get("object") == "chat.completion.chunk" for payload in payloads)
    assert any(
        choice.get("delta", {}).get("role") == "assistant"
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    assert any(
        choice.get("delta", {}).get("content")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    usage_payloads = [payload for payload in payloads if payload.get("usage")]
    if usage_payloads:
        assert_openai_usage(usage_payloads[-1]["usage"])


def test_openai_chat_forced_tool_call(client: httpx.Client, tool_model: str):
    """OpenAI Chat should preserve forced function tool calls from GHC."""
    response = client.post(
        "/api/openai/v1/chat/completions",
        json=openai_chat_tool_payload(tool_model),
    )
    assert_http_success(response)
    data = response.json()

    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls, data
    assert_tool_call_name(tool_calls[0], "get_weather")
    assert_openai_usage(data["usage"])


def test_openai_chat_forced_tool_call_streaming(client: httpx.Client, tool_model: str):
    """OpenAI Chat stream should expose tool_call deltas and final usage."""
    with client.stream(
        "POST",
        "/api/openai/v1/chat/completions",
        json=openai_chat_tool_payload(tool_model, stream=True),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    tool_chunks = [
        tool_call
        for payload in payloads
        for choice in payload.get("choices", [])
        for tool_call in choice.get("delta", {}).get("tool_calls") or []
    ]
    assert tool_chunks, payloads
    assert any(
        choice.get("finish_reason") == "tool_calls"
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    usage_payloads = [payload for payload in payloads if payload.get("usage")]
    if usage_payloads:
        assert_openai_usage(usage_payloads[-1]["usage"])


def test_openai_responses_non_streaming_returns_output_and_usage(
    client: httpx.Client,
    responses_model: str,
):
    """OpenAI Responses non-streaming should route to GHC /responses."""
    response = client.post(
        "/api/openai/v1/responses",
        json=openai_responses_payload(responses_model),
    )
    assert_http_success(response)
    data = response.json()

    assert data["object"] == "response"
    assert data["status"] == "completed"
    message_items = [item for item in data["output"] if item.get("type") == "message"]
    reasoning_items = [item for item in data["output"] if item.get("type") == "reasoning"]
    assert message_items or reasoning_items, data
    if message_items:
        assert_text_response(_response_output_text(message_items[0]))
    assert_responses_usage(data["usage"])


def test_openai_responses_streaming_returns_lifecycle_events(
    client: httpx.Client,
    responses_model: str,
):
    """OpenAI Responses streaming should expose the core lifecycle events."""
    with client.stream(
        "POST",
        "/api/openai/v1/responses",
        json=openai_responses_payload(responses_model, stream=True),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    event_names = [name for name, _payload in events]
    payloads = event_payloads(events)

    assert "response.created" in event_names
    assert "response.in_progress" in event_names
    assert "response.completed" in event_names
    completed = next(payload for name, payload in events if name == "response.completed")
    assert completed["response"]["status"] == "completed"
    usage = completed["response"].get("usage")
    if usage:
        assert_responses_usage(usage)
    assert any(
        payload.get("type")
        in {
            "response.output_text.delta",
            "response.reasoning_summary_text.delta",
        }
        for payload in payloads
    )


def test_openai_responses_forced_tool_call(client: httpx.Client, responses_model: str):
    """OpenAI Responses should preserve forced function_call output items."""
    response = client.post(
        "/api/openai/v1/responses",
        json=openai_responses_tool_payload(responses_model),
    )
    assert_http_success(response)
    data = response.json()

    assert data["status"] == "completed"
    assert_response_has_function_call(data, "get_weather")
    assert_responses_usage(data["usage"])


def test_openai_responses_forced_tool_call_streaming(
    client: httpx.Client,
    responses_model: str,
):
    """OpenAI Responses stream should expose function-call argument events."""
    with client.stream(
        "POST",
        "/api/openai/v1/responses",
        json=openai_responses_tool_payload(responses_model, stream=True),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    function_items = [
        payload["item"]
        for payload in payloads
        if payload.get("type") == "response.output_item.done"
        and payload.get("item", {}).get("type") == "function_call"
    ]
    assert function_items, payloads
    assert any(item.get("name") == "get_weather" for item in function_items)
    completed = next(payload for name, payload in events if name == "response.completed")
    usage = completed["response"].get("usage")
    if usage:
        assert_responses_usage(usage)


def _response_output_text(message_item: dict[str, Any]) -> str:
    content = message_item.get("content", [])
    return "".join(item.get("text", "") for item in content if item.get("type") == "output_text")
