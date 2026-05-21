"""Live Gemini-compatible model invocation paths."""

from __future__ import annotations

import httpx

from integration_tests.conftest import (
    assert_gemini_has_function_call,
    assert_gemini_usage,
    assert_http_success,
    assert_text_response,
    bare_model,
    event_payloads,
    gemini_compat_payload,
    gemini_payload,
    gemini_tool_payload,
    parse_sse_events,
)


def test_gemini_generate_content(client: httpx.Client, chat_model: str):
    """Gemini generateContent should route to GHC with a bare path model."""
    response = client.post(
        f"/api/gemini/v1beta/models/{bare_model(chat_model)}:generateContent",
        json=gemini_compat_payload(),
    )
    assert_http_success(response)
    data = response.json()

    assert data["modelVersion"] == bare_model(chat_model)
    assert data["candidates"], data
    candidate = data["candidates"][0]
    assert candidate["finishReason"] in {"STOP", "MAX_TOKENS", "SAFETY", "OTHER"}
    text = "".join(
        part.get("text", "")
        for part in candidate["content"]["parts"]
        if "text" in part
    )
    assert_text_response(text)
    assert_gemini_usage(data["usageMetadata"])


def test_gemini_stream_generate_content(client: httpx.Client, chat_model: str):
    """Gemini streamGenerateContent should emit text chunks and final usage."""
    with client.stream(
        "POST",
        f"/api/gemini/v1beta/models/{bare_model(chat_model)}:streamGenerateContent",
        json=gemini_payload(),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    assert payloads, events
    assert any(
        part.get("text")
        for payload in payloads
        for candidate in payload.get("candidates", [])
        for part in (candidate.get("content") or {}).get("parts", [])
    )
    final_with_usage = [payload for payload in payloads if payload.get("usageMetadata")]
    assert final_with_usage, payloads
    assert_gemini_usage(final_with_usage[-1]["usageMetadata"])


def test_gemini_count_tokens(client: httpx.Client, chat_model: str):
    """Gemini countTokens should return a local token estimate."""
    response = client.post(
        f"/api/gemini/v1beta/models/{bare_model(chat_model)}:countTokens",
        json=gemini_payload(),
    )
    assert_http_success(response)
    assert response.json()["totalTokens"] > 0


def test_gemini_forced_tool_call(client: httpx.Client, tool_model: str):
    """Gemini generateContent should translate forced tools to functionCall."""
    response = client.post(
        f"/api/gemini/v1beta/models/{bare_model(tool_model)}:generateContent",
        json=gemini_tool_payload(),
    )
    assert_http_success(response)
    data = response.json()

    assert_gemini_has_function_call(data, "get_weather")
    assert_gemini_usage(data["usageMetadata"])


def test_gemini_forced_tool_call_streaming(client: httpx.Client, tool_model: str):
    """Gemini streaming should emit functionCall parts for forced tools."""
    with client.stream(
        "POST",
        f"/api/gemini/v1beta/models/{bare_model(tool_model)}:streamGenerateContent",
        json=gemini_tool_payload(),
        timeout=180.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    assert any(
        part.get("functionCall", {}).get("name") == "get_weather"
        for payload in payloads
        for candidate in payload.get("candidates", [])
        for part in (candidate.get("content") or {}).get("parts", [])
    ), payloads
    final_with_usage = [payload for payload in payloads if payload.get("usageMetadata")]
    assert final_with_usage, payloads
    assert_gemini_usage(final_with_usage[-1]["usageMetadata"])
