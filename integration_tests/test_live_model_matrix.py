"""Broad live GHC model matrix checks through model invocation paths."""

from __future__ import annotations

from typing import Any

import httpx

from integration_tests.conftest import (
    LiveHttpEndpoint,
    assert_http_success,
    assert_openai_usage,
    assert_responses_usage,
    assert_text_response,
    model_matrix_chat_payload,
    openai_responses_payload,
    select_live_http_endpoint,
)
from router_maestro.providers import ModelInfo

_MAX_DIAGNOSTIC_BODY = 512


def test_copilot_model_matrix_openai_chat(
    client: httpx.Client,
    model_matrix: list[str],
    copilot_catalog: dict[str, ModelInfo],
) -> None:
    """Exercise the selected Copilot model matrix through OpenAI Chat."""
    failures: list[str] = []
    for model in model_matrix:
        try:
            endpoint = select_live_http_endpoint(model, copilot_catalog)
            if endpoint is LiveHttpEndpoint.UNSUPPORTED:
                failures.append(f"{model}: no supported HTTP endpoint declared by catalog")
                continue
            if endpoint is LiveHttpEndpoint.CHAT:
                response = _post_chat(client, model)
                if _is_unsupported_api(response):
                    raise AssertionError(
                        "catalog declared Chat support but transport rejected it: "
                        f"{_response_summary(response)}"
                    )
                attempted_endpoint = LiveHttpEndpoint.CHAT
            elif endpoint is LiveHttpEndpoint.RESPONSES:
                response = _post_responses(client, model)
                if _is_unsupported_api(response):
                    raise AssertionError(
                        "catalog declared Responses support but transport rejected it: "
                        f"{_response_summary(response)}"
                    )
                attempted_endpoint = LiveHttpEndpoint.RESPONSES
            elif endpoint is LiveHttpEndpoint.UNKNOWN:
                chat_response = _post_chat(client, model)
                if _is_unsupported_api(chat_response):
                    responses_response = _post_responses(client, model)
                    if _is_unsupported_api(responses_response):
                        raise AssertionError(
                            "both Chat and Responses explicitly unsupported; "
                            f"Chat: {_response_summary(chat_response)}; "
                            f"Responses: {_response_summary(responses_response)}"
                        )
                    response = responses_response
                    attempted_endpoint = LiveHttpEndpoint.RESPONSES
                else:
                    response = chat_response
                    attempted_endpoint = LiveHttpEndpoint.CHAT
            else:
                raise AssertionError(f"unhandled live HTTP endpoint state: {endpoint!r}")

            if attempted_endpoint is LiveHttpEndpoint.RESPONSES:
                _assert_responses_result(response, model)
            elif attempted_endpoint is LiveHttpEndpoint.CHAT:
                _assert_chat_result(response, model)
            else:
                raise AssertionError(
                    f"unhandled attempted live HTTP endpoint: {attempted_endpoint!r}"
                )
        except AssertionError as exc:
            failures.append(f"{model}: {exc}")

    assert not failures, "\n".join(failures)


def _is_unsupported_api(response: httpx.Response) -> bool:
    """Whether upstream rejected the model on this endpoint as unsupported."""
    if response.status_code != 400:
        return False
    data = _response_json_object(response)
    if data is None:
        return False
    error = data.get("error")
    return isinstance(error, dict) and error.get("code") == "unsupported_api_for_model"


def _post_chat(client: httpx.Client, model: str) -> httpx.Response:
    return client.post(
        "/api/openai/v1/chat/completions",
        json=model_matrix_chat_payload(model),
    )


def _post_responses(client: httpx.Client, model: str) -> httpx.Response:
    return client.post(
        "/api/openai/v1/responses",
        json=openai_responses_payload(model),
    )


def _assert_chat_result(response: httpx.Response, expected_model: str) -> None:
    assert_http_success(response)
    data = _required_json_object(response, "Chat")
    if "choices" not in data:
        raise AssertionError(f"Chat response missing choices: {_response_summary(response)}")
    choices = data["choices"]
    if not isinstance(choices, list):
        raise AssertionError(f"invalid Chat response choices: {_response_summary(response)}")
    if not choices:
        raise AssertionError(f"Chat response missing first choice: {_response_summary(response)}")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise AssertionError(f"invalid Chat response choice: {_response_summary(response)}")
    message = choice["message"]
    usage = data.get("usage")
    try:
        assert message.get("role") == "assistant"
        assert_text_response(message.get("content"))
        assert_openai_usage(usage)
    except (AssertionError, KeyError, TypeError) as exc:
        raise AssertionError(f"invalid Chat response: {_response_summary(response)}") from exc
    if data.get("model") != expected_model:
        raise AssertionError(
            "Chat response model identity does not match the public model invoked: "
            f"expected={expected_model!r}, actual={data.get('model')!r}; "
            f"{_response_summary(response)}"
        )


def _assert_responses_result(response: httpx.Response, expected_model: str) -> None:
    assert_http_success(response)
    data = _required_json_object(response, "Responses")
    if "status" not in data:
        raise AssertionError(f"Responses response missing status: {_response_summary(response)}")
    if "usage" not in data:
        raise AssertionError(f"Responses response missing usage: {_response_summary(response)}")
    try:
        assert data["status"] == "completed"
        assert_responses_usage(data["usage"])
    except (AssertionError, KeyError, TypeError) as exc:
        raise AssertionError(f"invalid Responses response: {_response_summary(response)}") from exc
    if data.get("model") != expected_model:
        raise AssertionError(
            "Responses model identity does not match the public model invoked: "
            f"expected={expected_model!r}, actual={data.get('model')!r}; "
            f"{_response_summary(response)}"
        )


def _required_json_object(response: httpx.Response, endpoint: str) -> dict[str, Any]:
    data = _response_json_object(response)
    if data is None:
        raise AssertionError(f"{endpoint} returned invalid JSON: {_response_summary(response)}")
    return data


def _response_json_object(response: httpx.Response) -> dict[str, Any] | None:
    _consume_response(response)
    try:
        data = response.json()
    except (ValueError, httpx.HTTPError):
        return None
    return data if isinstance(data, dict) else None


def _response_summary(response: httpx.Response) -> str:
    _consume_response(response)
    try:
        body = response.text
    except (httpx.HTTPError, RuntimeError):
        body = "<unavailable>"
    if len(body) > _MAX_DIAGNOSTIC_BODY:
        body = f"{body[:_MAX_DIAGNOSTIC_BODY]}…"
    return f"HTTP {response.status_code}, body={body!r}"


def _consume_response(response: httpx.Response) -> None:
    if response.is_stream_consumed:
        return
    try:
        response.read()
    except (httpx.HTTPError, RuntimeError):
        return
