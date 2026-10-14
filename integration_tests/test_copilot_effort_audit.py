"""Recorded wire checks for disabled reasoning and stream-open error audits."""

import json
from pathlib import Path

import httpx
import pytest

from integration_tests.conftest import (
    controlled_router_server,
    scripted_upstream,
    select_recorded_request,
)
from integration_tests.test_controlled_boundaries import (
    API_KEY,
    _auth,
    _catalog_model,
    _chat_success,
    _controlled_copilot,
    _json_reply,
    _models,
    _request_payload,
    _responses_success,
    _sse_reply,
)


@pytest.mark.parametrize("effort", ["off", "none"])
@pytest.mark.parametrize("stream", [False, True])
def test_required_reasoning_floor_reaches_copilot_wire(
    tmp_path: Path, effort: str, stream: bool
) -> None:
    """A stale catalog's disabled tier must not escape to a mandatory-reasoning model."""

    def responder(request):
        if request.path == "/models":
            model = _catalog_model("gpt-6.1-sol", endpoints=["/responses"])
            model["capabilities"]["supports"]["reasoning_effort"]["values"] = [
                "none",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            ]
            return _models(model)
        if request.path == "/responses":
            payload = _request_payload(request)
            if payload.get("reasoning", {}).get("effort") != "low":
                return _json_reply(400, {"error": {"message": "Unsupported reasoning effort"}})
            reply = _responses_success(payload["model"])
            if payload.get("stream"):
                return _sse_reply({"type": "response.completed", "response": json.loads(reply[1])})
            return reply
        return _json_reply(500, {"error": "wrong operation"})

    with _controlled_copilot(tmp_path, responder, priorities=["github-copilot/gpt-6.1-sol"]) as (
        client,
        upstream,
    ):
        response = client.post(
            "/api/openai/v1/responses",
            json={
                "model": "github-copilot/gpt-6.1-sol",
                "input": "ping",
                "stream": stream,
                "reasoning": {"effort": effort},
            },
        )

    assert response.status_code == 200, response.text
    if stream:
        assert "response.completed" in response.text
    request = select_recorded_request(upstream.requests, method="POST", path="/responses")
    assert _request_payload(request)["reasoning"]["effort"] == "low"
    assert len([r for r in upstream.requests if r.path == "/responses"]) == 1


@pytest.mark.parametrize("effort", ["off", "none"])
def test_chat_disabled_reasoning_floor_reaches_copilot_wire(tmp_path: Path, effort: str) -> None:
    """The HTTP Chat route accepts both disabled aliases before provider normalization."""

    def responder(request):
        if request.path == "/models":
            return _models(_catalog_model("chat-floor", endpoints=["/chat/completions"]))
        if request.path == "/chat/completions":
            payload = _request_payload(request)
            if payload.get("reasoning_effort") != "low":
                return _json_reply(400, {"error": {"message": "Unsupported reasoning effort"}})
            return _chat_success(payload["model"])
        return _json_reply(500, {"error": "wrong operation"})

    with _controlled_copilot(tmp_path, responder, priorities=["github-copilot/chat-floor"]) as (
        client,
        upstream,
    ):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json={
                "model": "github-copilot/chat-floor",
                "messages": [{"role": "user", "content": "ping"}],
                "reasoning_effort": effort,
            },
        )

    assert response.status_code == 200, response.text
    request = select_recorded_request(upstream.requests, method="POST", path="/chat/completions")
    assert _request_payload(request)["reasoning_effort"] == "low"


def test_stream_open_400_retains_redacted_error_in_audit(tmp_path: Path) -> None:
    """The public error remains bounded while its matching audit holds the upstream cause."""
    trace_root = tmp_path / "traces"
    request_id = "req-controlled-stream-error-audit"

    def responder(request):
        if request.path == "/models":
            return _models(_catalog_model("responses-model", endpoints=["/responses"]))
        if request.path == "/responses":
            return _json_reply(
                400,
                {
                    "error": {
                        "message": "Unsupported value for reasoning.effort",
                        "code": "invalid_request_body",
                        "param": "reasoning.effort",
                        "api_key": "controlled-payload-secret",
                    }
                },
            )
        return _json_reply(500, {"error": "wrong operation"})

    with scripted_upstream(responder) as upstream:
        with controlled_router_server(
            tmp_path,
            providers={},
            priorities={
                "priorities": ["github-copilot/responses-model"],
                "audit": {"enabled": True, "trace_dir": str(trace_root)},
            },
            credentials=_auth(upstream.base_url),
            api_key=API_KEY,
        ) as server:
            response = httpx.post(
                server.base_url + "/api/openai/v1/responses",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "X-Request-ID": request_id,
                },
                json={
                    "model": "github-copilot/responses-model",
                    "input": "ping",
                    "stream": True,
                },
                timeout=15,
            )

    assert response.status_code == 400, response.text
    assert response.headers["x-request-id"] == request_id
    assert response.json()["error"]["message"] == "Copilot API error: 400"
    directory = trace_root / request_id
    records = [json.loads(p.read_text()) for p in directory.glob("upstream_resp*.json")]
    failure = next(r for r in records if r["status"] == 400)
    assert failure["body"]["error"] == {
        "message": "Unsupported value for reasoning.effort",
        "code": "invalid_request_body",
        "param": "reasoning.effort",
        "api_key": "***",
    }
    assert "stream_summary" not in failure
    assert "controlled-payload-secret" not in "".join(
        p.read_text() for p in directory.glob("*.json")
    )
    assert json.loads((directory / "outbound.json").read_text())["status"] == 400
