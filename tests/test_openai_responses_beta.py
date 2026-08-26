"""Contract tests for the retired OpenAI Responses URL alias."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.server.routes.openai_responses_beta import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _payload() -> dict:
    return {"model": "github-copilot/gpt-test", "input": "hello"}


def test_beta_responses_delegates_to_stable_handler_without_app_marker() -> None:
    with patch(
        "router_maestro.server.routes.openai_responses_beta.standard_responses_endpoint",
        return_value={"source": "stable"},
    ) as stable_responses:
        response = _client().post("/api/openai/beta/v1/responses", json=_payload())

    assert response.status_code == 200
    assert response.json() == {"source": "stable"}
    await_args = stable_responses.await_args
    assert await_args is not None
    request = await_args.kwargs["request"]
    raw_request = await_args.kwargs["raw_request"]
    assert request.model == "github-copilot/gpt-test"
    assert raw_request.url.path == "/api/openai/beta/v1/responses"


def test_beta_responses_preserves_stable_bare_app_compatibility() -> None:
    with (
        patch("router_maestro.server.routes.responses.get_router", return_value=object()),
        patch(
            "router_maestro.server.routes.responses.create_response",
            return_value={"source": "stable-compatibility"},
        ) as create_response,
    ):
        response = _client().post("/api/openai/beta/v1/responses", json=_payload())

    assert response.status_code == 200
    assert response.json() == {"source": "stable-compatibility"}
    await_args = create_response.await_args
    assert await_args is not None
    assert await_args.args[0].model == "github-copilot/gpt-test"
