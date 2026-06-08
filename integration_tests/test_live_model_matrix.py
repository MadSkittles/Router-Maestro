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
    # Models the upstream refuses on the chosen endpoint (e.g. internal models
    # that are Responses-only but absent from RESPONSES_ELIGIBLE_MODELS). This is
    # a model-classification gap, not a Router-Maestro bug, so record + skip them
    # rather than failing — but surface them so the gap stays visible.
    unsupported: list[str] = []

    for model in model_matrix:
        try:
            if is_responses_eligible_model(model):
                response = client.post(
                    "/api/openai/v1/responses",
                    json=openai_responses_payload(model),
                )
                if _is_unsupported_api(response):
                    unsupported.append(model)
                    continue
                assert_http_success(response)
                data = response.json()
                assert data["status"] == "completed"
                assert_responses_usage(data["usage"])
            else:
                response = client.post(
                    "/api/openai/v1/chat/completions",
                    json=model_matrix_chat_payload(model),
                )
                if _is_unsupported_api(response):
                    unsupported.append(model)
                    continue
                assert_http_success(response)
                data = response.json()
                choice = data["choices"][0]
                assert choice["message"]["role"] == "assistant"
                assert_text_response(choice["message"].get("content"))
                assert_openai_usage(data["usage"])
        except AssertionError as exc:
            failures.append(f"{model}: {exc}")

    if unsupported:
        print(f"\nSkipped (upstream unsupported_api_for_model): {', '.join(unsupported)}")
    assert not failures, "\n".join(failures)


def _is_unsupported_api(response: httpx.Response) -> bool:
    """Whether upstream rejected the model on this endpoint as unsupported."""
    if response.status_code != 400:
        return False
    return "unsupported_api_for_model" in response.text
