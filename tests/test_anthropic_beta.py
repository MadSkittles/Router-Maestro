"""Contract tests for the retired Anthropic URL aliases."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.server.routes.anthropic_beta import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_beta_messages_delegates_to_stable_handler_without_app_marker() -> None:
    with patch(
        "router_maestro.server.routes.anthropic_beta.standard_messages",
        return_value={"source": "stable"},
    ) as stable_messages:
        response = _client().post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "test",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"source": "stable"}
    await_args = stable_messages.await_args
    assert await_args is not None
    request = await_args.kwargs["request"]
    raw_request = await_args.kwargs["raw_request"]
    assert request.model == "test"
    assert raw_request.url.path == "/api/anthropic/beta/v1/messages"


def test_beta_messages_preserves_stable_bare_app_compatibility() -> None:
    response = _client().post(
        "/api/anthropic/beta/v1/messages",
        json={
            "model": "test",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "test"


def test_beta_count_tokens_delegates_to_stable_handler() -> None:
    with patch(
        "router_maestro.server.routes.anthropic_beta.standard_count_tokens",
        return_value={"input_tokens": 7},
    ) as stable_count_tokens:
        response = _client().post(
            "/api/anthropic/beta/v1/messages/count_tokens",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 7}
    await_args = stable_count_tokens.await_args
    assert await_args is not None
    assert await_args.args[0].model == "test"
