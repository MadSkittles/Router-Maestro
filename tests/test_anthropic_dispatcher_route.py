from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
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
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponsesToolCall,
)
from router_maestro.providers.bindings import PreparedAttempt, legacy_endpoint_binding
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.runtime.reasoning_capsule import ReasoningCapsuleCodec
from router_maestro.server.routes.anthropic import router as anthropic_router
from router_maestro.server.routes.anthropic_beta import router as anthropic_beta_router
from router_maestro.utils.cache import TTLCache


class _ResponsesOnlyProvider(BaseProvider):
    name = "github-copilot"

    def __init__(self, responses: list[ResponsesResponse]) -> None:
        self._responses = list(responses)
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
        if False:
            yield ChatStreamChunk(content="")

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        self.requests.append(request)
        return self._responses.pop(0)

    async def responses_completion_stream(
        self,
        request: ResponsesRequest,
    ) -> AsyncIterator[ResponsesStreamChunk]:
        self.requests.append(request)
        yield ResponsesStreamChunk(content="hello")
        yield ResponsesStreamChunk(
            content="",
            usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            finish_reason="stop",
        )


def _router(provider: BaseProvider, *, model: str = "gpt-responses") -> Router:
    router = Router.__new__(Router)
    router.providers = {provider.name: provider}
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._model_aliases = None
    router._managed_generation = True
    router._priorities_cache.set(
        PrioritiesConfig(
            priorities=[f"github-copilot/{model}"],
            fallback=FallbackConfig(strategy=FallbackStrategy.NONE, maxRetries=0),
        )
    )
    router._providers_ttl.set(True)
    return router


def _client(
    provider: BaseProvider,
    *,
    model: str = "gpt-responses",
) -> tuple[TestClient, Router]:
    app = FastAPI()
    app.state.reasoning_capsule_codec = ReasoningCapsuleCodec(bytes([51]) * 32)
    app.include_router(anthropic_router)
    app.include_router(anthropic_beta_router)
    return TestClient(app), _router(provider, model=model)


def _payload(*, stream: bool = False) -> dict:
    return {
        "model": "github-copilot/gpt-responses",
        "max_tokens": 64,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}],
    }


class _RecordingCopilotExecutor:
    def __init__(self) -> None:
        self.attempts: list[PreparedAttempt] = []

    async def execute(self, attempt: PreparedAttempt) -> Mapping[str, Any]:
        self.attempts.append(attempt)
        if attempt.protocol is not WireProtocol.OPENAI_CHAT:
            raise AssertionError(f"incompatible transport reached I/O: {attempt.protocol.value}")
        return {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-5.4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    async def execute_stream(
        self,
        attempt: PreparedAttempt,
    ) -> AsyncIterator[Mapping[str, Any]]:
        raise AssertionError(f"unexpected stream attempt: {attempt.protocol.value}")
        if False:
            yield {}


class _TopPContractCopilotProvider(CopilotProvider):
    def __init__(self, executor: _RecordingCopilotExecutor) -> None:
        super().__init__()
        self._generation_bindings = tuple(
            replace(binding, executor=executor) for binding in super().bindings()
        )

    async def list_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        del force_refresh
        return [
            ModelInfo(
                id="gpt-5.4",
                name="gpt-5.4",
                provider=self.name,
                supported_endpoints=("/responses", "/chat/completions"),
            )
        ]

    def is_authenticated(self) -> bool:
        return True


class _ChatOnlyCopilotProvider(CopilotProvider):
    async def list_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        del force_refresh
        return [
            ModelInfo(
                id="gpt-4o",
                name="gpt-4o",
                provider=self.name,
                supported_endpoints=("/chat/completions",),
                feature_capabilities={"tools": True},
            )
        ]

    def is_authenticated(self) -> bool:
        return True


def test_stable_anthropic_gpt54_top_p_skips_responses_before_io() -> None:
    executor = _RecordingCopilotExecutor()
    provider = _TopPContractCopilotProvider(executor)
    client, model_router = _client(provider, model="gpt-5.4")
    payload = {
        "model": "github-copilot/gpt-5.4",
        "max_tokens": 64,
        "top_p": 1,
        "system": "Return concise answers.",
        "metadata": {"user_id": "router-maestro-integration"},
        "messages": [{"role": "user", "content": "Reply with exactly pong."}],
    }

    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=model_router,
    ):
        response = client.post("/api/anthropic/v1/messages", json=payload)

    assert response.status_code == 200, response.text
    assert response.json()["content"] == [{"type": "text", "text": "pong"}]
    assert [attempt.protocol for attempt in executor.attempts] == [WireProtocol.OPENAI_CHAT]
    assert executor.attempts[0].payload["top_p"] == 1


def test_stable_anthropic_tool_choice_any_decodes_copilot_chat_tool_call() -> None:
    provider = _ChatOnlyCopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-2026-08-01",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location":"Seattle"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 22,
                    "completion_tokens": 5,
                    "total_tokens": 27,
                },
                "prompt_filter_results": [{"prompt_index": 0, "private": True}],
                "copilot_usage": {"private": True},
            },
            request=httpx.Request("POST", "https://api.githubcopilot.com/chat/completions"),
        )
    )
    client, model_router = _client(provider, model="gpt-4o")
    payload = {
        "model": "github-copilot/gpt-4o",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Use the weather tool."}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
        "tool_choice": {"type": "any"},
    }

    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=model_router,
    ):
        response = client.post("/api/anthropic/v1/messages", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "call_weather",
            "name": "get_weather",
            "input": {"location": "Seattle"},
        }
    ]
    provider.ensure_token.assert_awaited()
    provider._send_with_auth_retry.assert_awaited_once()
    assert provider._send_with_auth_retry.await_args is not None
    request_kwargs = provider._send_with_auth_retry.await_args.kwargs
    assert request_kwargs["json"]["tool_choice"] == "required"
    assert request_kwargs["json"]["model"] == "gpt-4o"


def test_stable_anthropic_route_uses_responses_only_model_for_text_and_tools() -> None:
    provider = _ResponsesOnlyProvider(
        [
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
            "name": "lookup",
            "description": "Look something up",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]

    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=model_router,
    ):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "github-copilot/gpt-responses"
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0] == {"type": "text", "text": "hello from responses"}
    assert body["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup",
        "input": {"query": "rm"},
    }
    assert provider.requests[0].model == "gpt-responses"
    assert provider.requests[0].tools is not None
    assert provider.requests[0].tools[0]["name"] == "lookup"


def test_stable_anthropic_route_streams_responses_only_model() -> None:
    provider = _ResponsesOnlyProvider([])
    client, model_router = _client(provider)
    payload = _payload(stream=True)
    payload["system"] = [
        {
            "type": "text",
            "text": "Answer concisely.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    payload["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hello",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    payload["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}

    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=model_router,
    ):
        with client.stream(
            "POST",
            "/api/anthropic/v1/messages?beta=true",
            json=payload,
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: message_start" in body
    assert '"model": "github-copilot/gpt-responses"' in body
    assert '"text": "hello"' in body
    assert "event: message_stop" in body
    assert len(provider.requests) == 1
    assert provider.requests[0].provider_extensions == {}


def test_beta_anthropic_path_is_an_alias_of_the_shared_dispatcher() -> None:
    provider = _ResponsesOnlyProvider(
        [
            ResponsesResponse(
                content="alias",
                model="gpt-responses",
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                finish_reason="stop",
            )
        ]
    )
    client, model_router = _client(provider)

    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=model_router,
    ):
        response = client.post("/api/anthropic/beta/v1/messages", json=_payload())

    assert response.status_code == 200, response.text
    assert response.json()["content"] == [{"type": "text", "text": "alias"}]
    assert len(provider.requests) == 1


def test_beta_count_tokens_path_is_an_alias_of_the_stable_handler() -> None:
    app = FastAPI()
    app.state.reasoning_capsule_codec = ReasoningCapsuleCodec(bytes([51]) * 32)
    app.include_router(anthropic_beta_router)

    with patch(
        "router_maestro.server.routes.anthropic_beta.standard_count_tokens",
        return_value={"input_tokens": 7},
    ) as stable_count_tokens:
        response = TestClient(app).post(
            "/api/anthropic/beta/v1/messages/count_tokens",
            json={
                "model": "github-copilot/gpt-responses",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200, response.text
    assert response.json() == {"input_tokens": 7}
    stable_count_tokens.assert_awaited_once()


def test_anthropic_unknown_capsule_version_fails_before_provider_io() -> None:
    provider = _ResponsesOnlyProvider([])
    client, model_router = _client(provider)
    payload = _payload()
    payload["messages"] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "private summary",
                    "signature": "rmr2.unknown.payload",
                }
            ],
        },
        {"role": "user", "content": "continue"},
    ]

    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=model_router,
    ):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == "Invalid reasoning capsule"
    assert provider.requests == []


def test_anthropic_reasoning_capsule_replays_complete_responses_item() -> None:
    raw_reasoning_item = {
        "type": "reasoning",
        "id": "rs_round_one",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "inspect the repository"}],
        "encrypted_content": "provider-owned-state",
        "future_sibling": {"must_survive": True},
    }
    provider = _ResponsesOnlyProvider(
        [
            ResponsesResponse(
                content="first answer",
                model="gpt-responses",
                usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                thinking="inspect the repository",
                thinking_id="rs_round_one",
                thinking_signature="provider-owned-state",
                reasoning_item=raw_reasoning_item,
                finish_reason="stop",
            ),
            ResponsesResponse(
                content="second answer",
                model="gpt-responses",
                usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
                finish_reason="stop",
            ),
        ]
    )
    client, model_router = _client(provider)

    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=model_router,
    ):
        first = client.post("/v1/messages", json=_payload())
        first_body = first.json()
        thinking = next(block for block in first_body["content"] if block["type"] == "thinking")
        second_payload = _payload()
        second_payload["messages"] = [
            {"role": "assistant", "content": first_body["content"]},
            {"role": "user", "content": "continue"},
        ]
        second = client.post("/v1/messages", json=second_payload)

    assert first.status_code == 200, first.text
    assert thinking["signature"].startswith("rmr1.")
    assert second.status_code == 200, second.text
    assert second.json()["content"] == [{"type": "text", "text": "second answer"}]
    assert len(provider.requests) == 2
    assert isinstance(provider.requests[1].input, list)
    assert raw_reasoning_item in provider.requests[1].input
