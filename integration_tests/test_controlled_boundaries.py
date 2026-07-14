"""Deterministic end-to-end checks against an isolated fake Copilot upstream."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from integration_tests.conftest import (
    RecordedUpstreamRequest,
    assert_exact_outbound_payload,
    controlled_router_server,
    parse_sse_events,
    scripted_upstream,
    select_recorded_request,
)

API_KEY = "controlled-router-key"
_SUCCESS_TERMINALS = {
    "openai-chat": "[DONE]",
    "openai-responses": "response.completed",
    "anthropic": "message_stop",
    "anthropic-beta": "message_stop",
    "gemini": '"finishReason"',
}


def _json_reply(status: int, payload: object) -> tuple[int, bytes, dict[str, str]]:
    return status, json.dumps(payload).encode(), {"Content-Type": "application/json"}


def _sse_reply(*events: dict[str, Any]) -> tuple[int, bytes, dict[str, str]]:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    return 200, body, {"Content-Type": "text/event-stream"}


def _chat_success(model: str) -> tuple[int, bytes, dict[str, str]]:
    return _json_reply(
        200,
        {
            "id": "chatcmpl-controlled",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )


def _responses_success(model: str) -> tuple[int, bytes, dict[str, str]]:
    return _json_reply(
        200,
        {
            "id": "resp-controlled",
            "object": "response",
            "created_at": 1,
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg-controlled",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "pong"}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        },
    )


def _models(*entries: dict[str, Any]) -> tuple[int, bytes, dict[str, str]]:
    return _json_reply(200, {"data": list(entries)})


def _catalog_model(
    model: str,
    *,
    endpoints: list[str],
    tools: bool = True,
) -> dict[str, Any]:
    return {
        "id": model,
        "name": model,
        "supported_endpoints": endpoints,
        "capabilities": {
            "type": "chat",
            "limits": {"max_output_tokens": 32768},
            "supports": {
                "tool_calls": tools,
                "thinking": True,
                "reasoning_effort": {"values": ["low", "medium", "high"]},
            },
        },
    }


def _anthropic_native_success(model: str) -> tuple[int, bytes, dict[str, str]]:
    return _json_reply(
        200,
        {
            "id": "msg-controlled",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        },
    )


def _auth(api_base: str) -> dict[str, Any]:
    return {
        "github-copilot": {
            "type": "oauth",
            "refresh": "controlled-refresh",
            "access": "controlled-access",
            "expires": int(time.time()) + 3600,
            "api_endpoint": api_base,
        }
    }


def _assert_native_open_failure(surface: str, response: httpx.Response) -> None:
    """Require the exact native envelope for one controlled stream-open failure."""
    if surface == "anthropic":
        assert response.status_code == 200
        errors = [
            payload
            for name, payload in parse_sse_events(response)
            if name == "error" and isinstance(payload, dict)
        ]
        assert errors == [
            {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "Copilot API error: 429"},
            }
        ]
        return

    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    assert "data:" not in response.text
    body = response.json()
    assert "detail" not in body
    if surface in {"openai-chat", "openai-responses"}:
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["code"] == "rate_limit_exceeded"
    elif surface == "anthropic-beta":
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"
    else:
        assert body["error"]["code"] == 429
        assert body["error"]["status"] == "RESOURCE_EXHAUSTED"
        assert body["error"]["details"] == []


def _assert_unexpected_eof_terminal(surface: str, response: httpx.Response) -> None:
    """Require one native EOF terminal and no protocol success terminal."""
    events = parse_sse_events(response)
    assert _SUCCESS_TERMINALS[surface] not in response.text

    if surface == "openai-responses":
        incomplete = [
            payload
            for name, payload in events
            if name == "response.incomplete" and isinstance(payload, dict)
        ]
        assert len(incomplete) == 1
        assert incomplete[0]["type"] == "response.incomplete"
        terminal = incomplete[0]["response"]
        assert terminal["status"] == "incomplete"
        assert terminal["incomplete_details"] == {"reason": "unexpected_eof"}
        assert terminal["error"] is None
        return

    payloads = [payload for _name, payload in events if isinstance(payload, dict)]
    if surface in {"anthropic", "anthropic-beta"}:
        errors = [payload for payload in payloads if payload.get("type") == "error"]
        assert len(errors) == 1
        assert errors[0]["type"] == "error"
        assert errors[0]["error"]["type"] == "api_error"
    elif surface == "openai-chat":
        errors = [payload["error"] for payload in payloads if "error" in payload]
        assert len(errors) == 1
        assert errors[0]["type"] == "unexpected_eof"
    else:
        errors = [payload["error"] for payload in payloads if "error" in payload]
        assert len(errors) == 1
        assert errors[0]["code"] == 502
        assert errors[0]["status"] == "INTERNAL"
        assert errors[0]["details"] == []


@contextmanager
def _controlled_copilot(
    tmp_path: Path,
    responder: Callable[[RecordedUpstreamRequest], tuple[int, bytes, dict[str, str]]],
    *,
    priorities: list[str],
) -> Iterator[tuple[httpx.Client, Any]]:
    with scripted_upstream(responder) as upstream:
        config = {
            "priorities": priorities,
            "fallback": {"strategy": "priority", "maxRetries": 3},
        }
        with controlled_router_server(
            tmp_path,
            providers={},
            priorities=config,
            credentials=_auth(upstream.base_url),
            api_key=API_KEY,
        ) as server:
            with httpx.Client(
                base_url=server.base_url,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=15,
            ) as client:
                yield client, upstream


def test_operation_routing_uses_catalog_capability_and_exact_responses_payload(tmp_path: Path):
    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(
                _catalog_model("chat-only", endpoints=["/chat/completions"]),
                _catalog_model("responses-only", endpoints=["/responses"]),
            )
        if request.path == "/responses":
            return _responses_success(request.payload["model"])
        return _json_reply(500, {"error": "wrong operation"})

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/chat-only", "github-copilot/responses-only"],
    ) as (client, upstream):
        response = client.post(
            "/api/openai/v1/responses",
            json={
                "model": "router-maestro",
                "input": "ping",
                "instructions": "answer exactly",
                "top_p": 0.7,
                "metadata": {"case": "operation-routing"},
                "service_tier": "default",
                "max_output_tokens": 17,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "github-copilot/responses-only"
    request = select_recorded_request(upstream.requests, method="POST", path="/responses")
    assert_exact_outbound_payload(
        request.payload,
        {
            "model": "responses-only",
            "input": "ping",
            "stream": False,
            "instructions": "answer exactly",
            "max_output_tokens": 17,
            "top_p": 0.7,
            "metadata": {"case": "operation-routing"},
            "service_tier": "default",
        },
    )
    assert not [r for r in upstream.requests if r.path == "/chat/completions"]


def test_capability_mismatch_is_precommit_and_never_calls_upstream(tmp_path: Path):
    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(_catalog_model("no-tools", endpoints=["/chat/completions"], tools=False))
        return _json_reply(500, {"error": "must not execute"})

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/no-tools"],
    ) as (client, upstream):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json={
                "model": "github-copilot/no-tools",
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "detail" not in response.json()
    assert not [r for r in upstream.requests if r.path != "/models"]


@pytest.mark.parametrize("stream", [False, True])
def test_explicit_responses_unsupported_temperature_is_native_400_without_completion_call(
    tmp_path: Path,
    stream: bool,
):
    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(_catalog_model("responses-model", endpoints=["/responses"]))
        return _json_reply(500, {"error": "completion must not execute"})

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/responses-model"],
    ) as (client, upstream):
        response = client.post(
            "/api/openai/v1/responses",
            json={
                "model": "github-copilot/responses-model",
                "input": "ping",
                "stream": stream,
                "temperature": 0.2,
            },
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "error": {
            "message": "GitHub Copilot Responses does not support request option 'temperature'",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "unsupported_parameter",
        }
    }
    assert [request.path for request in upstream.requests] == ["/models"]


def test_automatic_fallback_skips_feature_incompatible_candidates(tmp_path: Path):
    """Fallback execution must never attempt a candidate lacking required tools."""

    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(
                _catalog_model(
                    "primary-tools",
                    endpoints=["/chat/completions"],
                    tools=True,
                ),
                _catalog_model(
                    "no-tools",
                    endpoints=["/chat/completions"],
                    tools=False,
                ),
                _catalog_model(
                    "fallback-tools",
                    endpoints=["/chat/completions"],
                    tools=True,
                ),
            )
        if request.payload["model"] == "primary-tools":
            return _json_reply(503, {"error": {"message": "retry", "type": "server_error"}})
        if request.payload["model"] == "fallback-tools":
            return _chat_success("fallback-tools")
        return _json_reply(500, {"error": "feature-incompatible candidate was executed"})

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=[
            "github-copilot/primary-tools",
            "github-copilot/no-tools",
            "github-copilot/fallback-tools",
        ],
    ) as (client, upstream):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json={
                "model": "router-maestro",
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "github-copilot/fallback-tools"
    attempts = [
        request.payload["model"]
        for request in upstream.requests
        if request.path == "/chat/completions"
    ]
    assert attempts == ["primary-tools", "fallback-tools"]


def test_precommit_failure_falls_back_and_reports_actual_model_identity(tmp_path: Path):
    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(
                _catalog_model("primary", endpoints=["/chat/completions"]),
                _catalog_model("fallback", endpoints=["/chat/completions"]),
            )
        if request.payload["model"] == "primary":
            return _json_reply(503, {"error": {"message": "retry", "type": "server_error"}})
        return _chat_success("fallback")

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/primary", "github-copilot/fallback"],
    ) as (client, upstream):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json={"model": "router-maestro", "messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "github-copilot/fallback"
    attempts = [r.payload["model"] for r in upstream.requests if r.path == "/chat/completions"]
    assert attempts == ["primary", "fallback"]


@pytest.mark.parametrize(
    ("surface", "path", "payload", "error_path"),
    [
        (
            "openai-chat",
            "/api/openai/v1/chat/completions",
            {
                "model": "github-copilot/stream-model",
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
            "/chat/completions",
        ),
        (
            "openai-responses",
            "/api/openai/v1/responses",
            {"model": "github-copilot/stream-model", "stream": True, "input": "ping"},
            "/responses",
        ),
        (
            "anthropic",
            "/api/anthropic/v1/messages",
            {
                "model": "github-copilot/stream-model",
                "stream": True,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "ping"}],
            },
            "/chat/completions",
        ),
        (
            "anthropic-beta",
            "/api/anthropic/beta/v1/messages",
            {
                "model": "github-copilot/stream-model",
                "stream": True,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "ping"}],
            },
            "/v1/messages",
        ),
        (
            "gemini",
            "/api/gemini/v1beta/models/github-copilot/stream-model:streamGenerateContent",
            {"contents": [{"role": "user", "parts": [{"text": "ping"}]}]},
            "/chat/completions",
        ),
    ],
)
def test_stream_open_errors_use_native_precommit_or_postcommit_envelope(
    tmp_path: Path,
    surface: str,
    path: str,
    payload: dict[str, Any],
    error_path: str,
):
    endpoints = ["/chat/completions", "/responses", "/v1/messages"]

    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(_catalog_model("stream-model", endpoints=endpoints))
        return _json_reply(429, {"error": {"message": "rate limited", "type": "rate_limit_error"}})

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/stream-model"],
    ) as (client, upstream):
        response = client.post(path, json=payload)

    assert len([r for r in upstream.requests if r.path == error_path]) == 1
    _assert_native_open_failure(surface, response)


def test_stream_postcommit_failure_is_normalized_once_and_never_replays(tmp_path: Path):
    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(
                _catalog_model("primary", endpoints=["/chat/completions"]),
                _catalog_model("fallback", endpoints=["/chat/completions"]),
            )
        if request.payload["model"] == "fallback":
            return _chat_success("fallback")
        return _sse_reply(
            {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
            {"choices": [{"delta": "malformed", "finish_reason": None}]},
        )

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/primary", "github-copilot/fallback"],
    ) as (client, upstream):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json={
                "model": "router-maestro",
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 200
    events = parse_sse_events(response)
    contents = [
        choice["delta"].get("content")
        for _name, payload in events
        if isinstance(payload, dict)
        for choice in payload.get("choices", [])
        if isinstance(choice.get("delta"), dict) and choice["delta"].get("content") is not None
    ]
    assert contents == ["partial"]
    errors = [
        payload["error"]
        for _name, payload in events
        if isinstance(payload, dict) and "error" in payload
    ]
    assert errors == [
        {
            "message": "github-copilot returned a malformed upstream response",
            "type": "api_error",
            "code": "upstream_protocol_error",
        }
    ]
    assert "unexpected_eof" not in response.text
    assert "[DONE]" not in response.text
    attempts = [r.payload["model"] for r in upstream.requests if r.path == "/chat/completions"]
    assert attempts == ["primary"]


@pytest.mark.parametrize(
    ("surface", "path", "payload", "upstream_path"),
    [
        (
            "openai-chat",
            "/api/openai/v1/chat/completions",
            {
                "model": "github-copilot/eof-model",
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
            "/chat/completions",
        ),
        (
            "openai-responses",
            "/api/openai/v1/responses",
            {"model": "github-copilot/eof-model", "stream": True, "input": "ping"},
            "/responses",
        ),
        (
            "anthropic",
            "/api/anthropic/v1/messages",
            {
                "model": "github-copilot/eof-model",
                "stream": True,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "ping"}],
            },
            "/chat/completions",
        ),
        (
            "anthropic-beta",
            "/api/anthropic/beta/v1/messages",
            {
                "model": "github-copilot/eof-model",
                "stream": True,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "ping"}],
            },
            "/v1/messages",
        ),
        (
            "gemini",
            "/api/gemini/v1beta/models/github-copilot/eof-model:streamGenerateContent",
            {"contents": [{"role": "user", "parts": [{"text": "ping"}]}]},
            "/chat/completions",
        ),
    ],
)
def test_stream_unexpected_eof_is_one_native_terminal_without_success_sentinel(
    tmp_path: Path,
    surface: str,
    path: str,
    payload: dict[str, Any],
    upstream_path: str,
):
    endpoints = ["/chat/completions", "/responses", "/v1/messages"]

    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(_catalog_model("eof-model", endpoints=endpoints))
        if request.path == "/responses":
            return _sse_reply(
                {"type": "response.output_text.delta", "item_id": "msg", "delta": "partial"}
            )
        if request.path == "/v1/messages":
            return (
                200,
                b"event: message_start\ndata: "
                + json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg",
                            "type": "message",
                            "role": "assistant",
                            "model": "eof-model",
                            "content": [],
                            "usage": {"input_tokens": 1, "output_tokens": 0},
                        },
                    }
                ).encode()
                + b"\n\n",
                {"Content-Type": "text/event-stream"},
            )
        return _sse_reply({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]})

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/eof-model"],
    ) as (client, upstream):
        response = client.post(path, json=payload)

    assert response.status_code == 200, response.text
    assert len([r for r in upstream.requests if r.path == upstream_path]) == 1
    _assert_unexpected_eof_terminal(surface, response)


def test_beta_enabled_budget_effort_payload_is_exact_in_both_modes(tmp_path: Path):
    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(_catalog_model("claude-controlled", endpoints=["/v1/messages"]))
        if request.payload["stream"]:
            return (
                200,
                b"event: message_start\ndata: "
                + json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-controlled",
                            "content": [],
                            "usage": {"input_tokens": 1, "output_tokens": 0},
                        },
                    }
                ).encode()
                + b'\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n',
                {"Content-Type": "text/event-stream"},
            )
        return _anthropic_native_success("claude-controlled")

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/claude-controlled"],
    ) as (client, upstream):
        for stream in (False, True):
            payload = {
                "model": "github-copilot/claude-controlled",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4096,
                "stream": stream,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "output_config": {"effort": "high"},
            }
            response = client.post("/api/anthropic/beta/v1/messages", json=payload)
            assert response.status_code == 200, response.text

    attempts = [r for r in upstream.requests if r.path == "/v1/messages"]
    assert len(attempts) == 2
    for stream, request in zip((False, True), attempts, strict=True):
        assert_exact_outbound_payload(
            request.payload,
            {
                "model": "claude-controlled",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4096,
                "stream": stream,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "output_config": {"effort": "high"},
            },
        )


def test_standard_anthropic_enabled_budget_effort_payload_is_exact_in_both_modes(
    tmp_path: Path,
):
    def responder(request: RecordedUpstreamRequest):
        if request.path == "/models":
            return _models(_catalog_model("claude-controlled", endpoints=["/chat/completions"]))
        if request.payload["stream"]:
            return _sse_reply(
                {
                    "model": "claude-controlled",
                    "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": None}],
                },
                {
                    "model": "claude-controlled",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                },
            )
        return _chat_success("claude-controlled")

    with _controlled_copilot(
        tmp_path,
        responder,
        priorities=["github-copilot/claude-controlled"],
    ) as (client, upstream):
        for stream in (False, True):
            response = client.post(
                "/api/anthropic/v1/messages",
                json={
                    "model": "github-copilot/claude-controlled",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 4096,
                    "stream": stream,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "output_config": {"effort": "high"},
                },
            )
            assert response.status_code == 200, response.text

    attempts = [request for request in upstream.requests if request.path == "/chat/completions"]
    assert len(attempts) == 2
    for stream, request in zip((False, True), attempts, strict=True):
        expected = {
            "model": "claude-controlled",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": stream,
            "max_tokens": 4096,
            "reasoning_effort": "high",
        }
        if stream:
            expected["stream_options"] = {"include_usage": True}
        assert_exact_outbound_payload(request.payload, expected)
