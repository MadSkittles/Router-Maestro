from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.config import FallbackConfig, FallbackStrategy, PrioritiesConfig
from router_maestro.protocols import WireProtocol
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponsesToolCall,
)
from router_maestro.providers.bindings import legacy_endpoint_binding
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.runtime.reasoning_capsule import ReasoningCapsuleCodec
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.utils.cache import TTLCache


class _ResponsesOnlyProvider(BaseProvider):
    name = "github-copilot"

    def __init__(
        self,
        responses: list[ResponsesResponse] | None = None,
        stream_items: list[ResponsesStreamChunk | BaseException] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._stream_items = list(stream_items or [])
        self.requests: list[ResponsesRequest] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            operations=frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM})
        )

    def bindings(self):
        return (
            legacy_endpoint_binding(
                binding_id="copilot-openai-responses",
                protocol=WireProtocol.OPENAI_RESPONSES,
                operations=frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM}),
            ),
        )

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="gpt-responses",
                name="gpt-responses",
                provider=self.name,
                supported_endpoints=("/responses",),
            )
        ]

    def is_authenticated(self) -> bool:
        return True

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError(f"Chat transport must not be selected: {request.model}")

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        raise AssertionError(f"Chat transport must not be selected: {request.model}")
        if False:  # pragma: no cover - retain the async-generator return type
            yield ChatStreamChunk(content="")

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        self.requests.append(request)
        return self._responses.pop(0)

    async def responses_completion_stream(
        self,
        request: ResponsesRequest,
    ) -> AsyncIterator[ResponsesStreamChunk]:
        self.requests.append(request)
        for item in self._stream_items:
            if isinstance(item, BaseException):
                raise item
            yield item


def _router(provider: _ResponsesOnlyProvider) -> Router:
    model_router = Router.__new__(Router)
    model_router.providers = {provider.name: provider}
    model_router._models_cache = {}
    model_router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    model_router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    model_router._fuzzy_cache = {}
    model_router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    model_router._model_aliases = None
    model_router._managed_generation = True
    model_router._priorities_cache.set(
        PrioritiesConfig(
            priorities=["github-copilot/gpt-responses"],
            fallback=FallbackConfig(strategy=FallbackStrategy.NONE, maxRetries=0),
        )
    )
    model_router._providers_ttl.set(True)
    return model_router


def _client(provider: _ResponsesOnlyProvider) -> tuple[TestClient, Router]:
    app = FastAPI()
    app.state.reasoning_capsule_codec = ReasoningCapsuleCodec(bytes([61]) * 32)
    # The stable Chat route has no dependency on either RM beta router.
    app.include_router(chat_router)
    return TestClient(app, raise_server_exceptions=False), _router(provider)


def _payload(*, stream: bool = False) -> dict:
    return {
        "model": "github-copilot/gpt-responses",
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}],
    }


def _sse_data(text: str) -> list[dict | str]:
    events: list[dict | str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line.removeprefix("data: ")
        events.append(raw if raw == "[DONE]" else json.loads(raw))
    return events


def test_stable_chat_route_uses_responses_only_model_for_text_and_tools() -> None:
    provider = _ResponsesOnlyProvider(
        responses=[
            ResponsesResponse(
                content="hello from responses",
                model="gpt-responses",
                usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                tool_calls=[
                    ResponsesToolCall(
                        call_id="call_1",
                        name="lookup",
                        arguments='{"query":"rm"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    client, model_router = _client(provider)
    payload = _payload()
    payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]

    with patch("router_maestro.server.routes.chat.get_router", return_value=model_router):
        response = client.post("/api/openai/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "github-copilot/gpt-responses"
    assert body["choices"][0]["message"]["content"] == "hello from responses"
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"rm"}'},
        }
    ]
    assert provider.requests[0].model == "gpt-responses"
    assert provider.requests[0].tools is not None
    assert provider.requests[0].tools[0]["name"] == "lookup"


def test_stable_chat_route_streams_responses_only_model_and_done_sentinel() -> None:
    provider = _ResponsesOnlyProvider(
        stream_items=[
            ResponsesStreamChunk(content="hello"),
            ResponsesStreamChunk(
                content="",
                usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                finish_reason="stop",
            ),
        ]
    )
    client, model_router = _client(provider)

    with patch("router_maestro.server.routes.chat.get_router", return_value=model_router):
        with client.stream(
            "POST",
            "/api/openai/v1/chat/completions",
            json=_payload(stream=True),
        ) as response:
            events = _sse_data("".join(response.iter_text()))

    assert response.status_code == 200
    assert events[-1] == "[DONE]"
    frames = [event for event in events if isinstance(event, dict)]
    assert all(frame.get("model") == "github-copilot/gpt-responses" for frame in frames)
    assert any(
        choice.get("delta", {}).get("content") == "hello"
        for frame in frames
        for choice in frame.get("choices", [])
    )
    terminal = next(
        frame
        for frame in frames
        if frame.get("choices") and frame["choices"][0].get("finish_reason") == "stop"
    )
    assert terminal["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    ("include_usage", "expected_usage_positions"),
    [(True, ["usage-only"]), (False, [])],
)
def test_chat_dispatcher_preserves_explicit_stream_usage_policy(
    include_usage: bool,
    expected_usage_positions: list[str],
) -> None:
    provider = _ResponsesOnlyProvider(
        stream_items=[
            ResponsesStreamChunk(content="hello"),
            ResponsesStreamChunk(
                content="",
                usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                finish_reason="stop",
            ),
        ]
    )
    client, model_router = _client(provider)
    payload = _payload(stream=True)
    payload["stream_options"] = {"include_usage": include_usage}

    with patch("router_maestro.server.routes.chat.get_router", return_value=model_router):
        response = client.post("/api/openai/v1/chat/completions", json=payload)

    assert response.status_code == 200
    frames = [event for event in _sse_data(response.text) if isinstance(event, dict)]
    usage_positions = [
        "usage-only" if not frame["choices"] else "terminal"
        for frame in frames
        if frame.get("usage")
    ]
    assert usage_positions == expected_usage_positions


def test_chat_cross_transport_rejects_unknown_stream_option_with_exact_parameter() -> None:
    provider = _ResponsesOnlyProvider()
    client, model_router = _client(provider)
    payload = _payload(stream=True)
    payload["stream_options"] = {
        "include_usage": True,
        "future_option": {"cannot": "translate"},
    }

    with patch("router_maestro.server.routes.chat.get_router", return_value=model_router):
        response = client.post("/api/openai/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "stream_options.future_option"
    assert provider.requests == []


def test_chat_dispatcher_preframe_failure_remains_http_error() -> None:
    provider = _ResponsesOnlyProvider(
        stream_items=[
            ProviderError(
                "try later",
                status_code=429,
                retryable=False,
                kind=ProviderFailureKind.RATE_LIMIT,
            )
        ]
    )
    client, model_router = _client(provider)

    with patch("router_maestro.server.routes.chat.get_router", return_value=model_router):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json=_payload(stream=True),
        )

    assert response.status_code == 429
    assert response.json()["error"]["message"] == "try later"
    assert "data:" not in response.text


def test_chat_dispatcher_postcommit_failure_is_sse_error_without_done() -> None:
    provider = _ResponsesOnlyProvider(
        stream_items=[
            ResponsesStreamChunk(content="hello"),
            ProviderError(
                "late failure",
                status_code=429,
                retryable=False,
                kind=ProviderFailureKind.RATE_LIMIT,
            ),
        ]
    )
    client, model_router = _client(provider)

    with patch("router_maestro.server.routes.chat.get_router", return_value=model_router):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json=_payload(stream=True),
        )

    assert response.status_code == 200
    events = _sse_data(response.text)
    assert "[DONE]" not in events
    assert events[-1] == {
        "error": {
            "message": "late failure",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }
