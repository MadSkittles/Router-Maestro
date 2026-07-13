"""Live Gemini-family model matrix checks."""

from __future__ import annotations

from typing import Any

import httpx

from integration_tests.conftest import (
    assert_gemini_usage,
    assert_http_success,
    assert_text_response,
    bare_model,
    event_payloads,
    gemini_model_path,
    gemini_reasoning_payload,
    parse_sse_events,
)


def test_gemini_family_generate_content_matrix(
    client: httpx.Client,
    gemini_family_models: list[str],
):
    """Every available Gemini-family Copilot model should serve Gemini API calls."""
    failures: list[str] = []

    for model in gemini_family_models:
        for stream in (False, True):
            cell = f"{bare_model(model)} {'stream' if stream else 'nonstream'}"
            try:
                if stream:
                    data = _post_gemini_stream(client, model)
                else:
                    data = _post_gemini_non_stream(client, model)
                _assert_gemini_matrix_result(data)
            except AssertionError as exc:
                failures.append(f"{cell}: {exc}")

    assert not failures, "\n".join(failures)


def _post_gemini_non_stream(client: httpx.Client, model: str) -> dict[str, Any]:
    response = client.post(
        f"/api/gemini/v1beta/models/{gemini_model_path(model)}:generateContent",
        json=gemini_reasoning_payload(),
        timeout=240.0,
    )
    assert_http_success(response)
    return response.json()


def _post_gemini_stream(client: httpx.Client, model: str) -> dict[str, Any]:
    with client.stream(
        "POST",
        f"/api/gemini/v1beta/models/{gemini_model_path(model)}:streamGenerateContent",
        json=gemini_reasoning_payload(),
        timeout=240.0,
    ) as response:
        assert_http_success(response)
        events = parse_sse_events(response)

    payloads = event_payloads(events)
    text = "".join(
        part.get("text", "")
        for payload in payloads
        for candidate in payload.get("candidates", [])
        for part in (candidate.get("content") or {}).get("parts", [])
        if "text" in part
    )
    finish = next(
        (
            candidate.get("finishReason")
            for payload in reversed(payloads)
            for candidate in payload.get("candidates", [])
            if candidate.get("finishReason")
        ),
        None,
    )
    return {
        "candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}],
        "usageMetadata": next(
            (
                payload["usageMetadata"]
                for payload in reversed(payloads)
                if payload.get("usageMetadata")
            ),
            {},
        ),
    }


def _assert_gemini_matrix_result(data: dict[str, Any]) -> None:
    candidates = data.get("candidates") or []
    assert candidates, data
    candidate = candidates[0]
    assert candidate.get("finishReason") in {"STOP", "MAX_TOKENS", "SAFETY", "OTHER", None}
    text = "".join(
        part.get("text", "")
        for part in (candidate.get("content") or {}).get("parts", [])
        if "text" in part
    )
    assert_text_response(text)
    if data.get("usageMetadata"):
        assert_gemini_usage(data["usageMetadata"])
