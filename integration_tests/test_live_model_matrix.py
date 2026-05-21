"""Broad live GHC model matrix checks through model invocation paths."""

from __future__ import annotations

import httpx

from integration_tests.conftest import (
    assert_http_success,
    assert_openai_usage,
    assert_responses_usage,
    assert_text_response,
    is_responses_eligible_model,
    model_matrix_chat_payload,
    openai_responses_payload,
)


def test_copilot_model_matrix_openai_chat(
    client: httpx.Client,
    model_matrix: list[str],
):
    """Exercise the selected Copilot model matrix through OpenAI Chat."""
    failures: list[str] = []

    for model in model_matrix:
        try:
            if is_responses_eligible_model(model):
                response = client.post(
                    "/api/openai/v1/responses",
                    json=openai_responses_payload(model),
                )
                assert_http_success(response)
                data = response.json()
                assert data["status"] == "completed"
                assert_responses_usage(data["usage"])
            else:
                response = client.post(
                    "/api/openai/v1/chat/completions",
                    json=model_matrix_chat_payload(model),
                )
                assert_http_success(response)
                data = response.json()
                choice = data["choices"][0]
                assert choice["message"]["role"] == "assistant"
                assert_text_response(choice["message"].get("content"))
                assert_openai_usage(data["usage"])
        except AssertionError as exc:
            failures.append(f"{model}: {exc}")

    assert not failures, "\n".join(failures)
