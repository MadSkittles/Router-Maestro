"""Focused contracts for official and custom OpenAI-compatible Chat bindings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.config import FallbackConfig, FallbackStrategy, PrioritiesConfig
from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    OpenAIChatRuntime,
    RequestEnvelope,
    RequestManifest,
    SemanticEvent,
    SemanticRequest,
    SemanticResponse,
    WireProtocol,
)
from router_maestro.providers.base import (
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
)
from router_maestro.providers.bindings import OPENAI_COMPATIBLE_CHAT_BINDING
from router_maestro.providers.openai import OpenAIProvider
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.providers.openai_compat import OpenAICompatibleProvider
from router_maestro.routing.capabilities import Operation
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.runtime.reasoning_capsule import ReasoningCapsuleCodec
from router_maestro.server.dispatcher import GenerationDispatcher, LegacyProviderExecutionAdapter
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.utils.async_iterators import close_async_iterator
from router_maestro.utils.cache import TTLCache


@dataclass(frozen=True)
class _ProviderCase:
    provider: OpenAIChatProvider
    base_url: str
    headers: dict[str, str]


@pytest.fixture(params=("official", "custom"))
def provider_case(request: pytest.FixtureRequest) -> _ProviderCase:
    if request.param == "official":
        provider = OpenAIProvider(base_url="https://openai.example/v42/")
        headers = {
            "Authorization": "Bearer official-test",
            "Content-Type": "application/json",
        }
        provider._get_headers = lambda: dict(headers)  # type: ignore[method-assign]
        provider.is_authenticated = lambda: True  # type: ignore[method-assign]
        return _ProviderCase(provider, "https://openai.example/v42", headers)

    provider = OpenAICompatibleProvider(
        name="custom-test",
        base_url="https://custom.example/api/",
        api_key="custom-secret",
    )
    headers = {
        "Authorization": "Bearer custom-secret",
        "Content-Type": "application/json",
    }
    return _ProviderCase(provider, "https://custom.example/api", headers)


def _binding(provider: OpenAIChatProvider):
    (binding,) = provider.bindings()
    return binding


def _model(provider: OpenAIChatProvider) -> ModelRef:
    return ModelRef(provider=provider.name, upstream_id="upstream-model")


def _chat_response() -> dict:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 1,
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        },
    }


def _router(provider: OpenAIChatProvider) -> Router:
    provider.list_models = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            ModelInfo(
                id="upstream-model",
                name="upstream-model",
                provider=provider.name,
            )
        ]
    )
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
            priorities=[f"{provider.name}/upstream-model"],
            fallback=FallbackConfig(strategy=FallbackStrategy.NONE, maxRetries=0),
        )
    )
    router._providers_ttl.set(True)
    return router


def test_openai_chat_binding_is_protocol_native_cached_and_supports_both_operations(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider

    bindings = provider.bindings()
    (binding,) = bindings

    assert bindings is provider.bindings()
    assert binding.id == OPENAI_COMPATIBLE_CHAT_BINDING
    assert binding.protocol is WireProtocol.OPENAI_CHAT
    assert binding.is_legacy is False
    assert binding.supports(Operation.CHAT)
    assert binding.supports(Operation.CHAT_STREAM)


@pytest.mark.asyncio
async def test_openai_chat_dialect_is_copy_on_write_and_preserves_unknown_wire_fields(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider
    source = {
        "model": f"{provider.name}/public-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "future_content_field": {"kept": True},
                    }
                ],
                "future_message_field": [1, 2, 3],
            }
        ],
        "stream": False,
        "stream_options": {
            "include_usage": False,
            "future_stream_option": "kept",
        },
        "future_top_level": {"kept": True},
    }
    original = deepcopy(source)

    attempt = await _binding(provider).prepare_attempt(
        model=_model(provider),
        payload=source,
        stream=True,
    )

    assert source == original
    assert attempt.url == f"{provider_case.base_url}/chat/completions"
    assert dict(attempt.headers) == provider_case.headers
    assert dict(attempt.payload) == {
        **original,
        "model": "upstream-model",
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "future_stream_option": "kept",
        },
    }
    assert attempt.payload["future_top_level"] == {"kept": True}
    assert attempt.payload["messages"][0]["future_message_field"] == [1, 2, 3]
    assert attempt.payload["messages"][0]["content"][0]["future_content_field"] == {"kept": True}


@pytest.mark.asyncio
async def test_openai_chat_executor_sends_exact_raw_contract_and_returns_raw_json(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider
    requests: list[httpx.Request] = []
    response_payload = {
        **_chat_response(),
        "future_top_level": {"kept": True},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_payload)

    source = {
        "model": f"{provider.name}/public-model",
        "messages": [
            {
                "role": "user",
                "content": "hello",
                "future_message_field": {"kept": True},
            }
        ],
        "future_top_level": ["kept"],
    }
    attempt = await _binding(provider).prepare_attempt(
        model=_model(provider),
        payload=source,
        stream=False,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        executor = _binding(provider).executor
        assert executor is not None
        result = await executor.execute(attempt)

    assert result == response_payload
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url == httpx.URL(f"{provider_case.base_url}/chat/completions")
    assert request.headers["authorization"] == provider_case.headers["Authorization"]
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        **source,
        "model": "upstream-model",
        "stream": False,
    }


class _IdentityRuntime:
    protocol = WireProtocol.OPENAI_CHAT

    def __init__(self) -> None:
        self.decode_request = AsyncMock(
            side_effect=AssertionError("identity dispatch must not decode semantic IR")
        )

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        model = payload.get("model")
        return RequestManifest(
            protocol=self.protocol,
            model=model if isinstance(model, str) else None,
            stream=payload.get("stream") is True,
        )

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        del request
        raise AssertionError("identity dispatch must not encode semantic IR")

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        del payload
        raise AssertionError("identity dispatch must not decode a response")

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        del response
        raise AssertionError("identity dispatch must not encode a response")

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        del payload
        raise AssertionError("identity dispatch must not decode stream events")

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        del event
        raise AssertionError("identity dispatch must not encode stream events")


@pytest.mark.asyncio
async def test_openai_chat_identity_dispatch_bypasses_ir_and_legacy_dto_factory(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider
    source = {
        "model": f"{provider.name}/upstream-model",
        "messages": [
            {
                "role": "user",
                "content": "hello",
                "future_message_field": {"kept": True},
            }
        ],
        "future_top_level": {"kept": True},
    }
    runtime = _IdentityRuntime()
    envelope = RequestEnvelope(runtime, source)
    semantic_ir = AsyncMock(
        side_effect=AssertionError("identity dispatch must not call semantic_ir")
    )
    envelope.semantic_ir = semantic_ir  # type: ignore[method-assign]
    legacy_factory = Mock(side_effect=AssertionError("raw binding must not build ChatRequest"))
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=_chat_response()))
    )

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        result = await GenerationDispatcher(
            {},
            execution=LegacyProviderExecutionAdapter(chat_request_factory=legacy_factory),
        ).dispatch(_router(provider), envelope)

    assert result.value == _chat_response()
    assert result.selection.plan.binding.id == OPENAI_COMPATIBLE_CHAT_BINDING
    assert envelope.materialization_count == 0
    semantic_ir.assert_not_awaited()
    runtime.decode_request.assert_not_awaited()
    legacy_factory.assert_not_called()


def test_stable_chat_route_preserves_unknown_identity_stream_options(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        frames = [
            {
                "id": "chatcmpl_1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "upstream-model",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl_1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "upstream-model",
                "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        ]
        content = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
        content += "data: [DONE]\n\n"
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    app = FastAPI()
    app.state.reasoning_capsule_codec = ReasoningCapsuleCodec(bytes([62]) * 32)
    app.include_router(chat_router)
    client = TestClient(app, raise_server_exceptions=False)
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    payload = {
        "model": f"{provider.name}/upstream-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "future_option": {"keep": 1},
        },
    }

    with (
        patch("router_maestro.server.routes.chat.get_router", return_value=_router(provider)),
        patch(
            "router_maestro.server.routes.chat.ChatRequest",
            side_effect=AssertionError("identity route must not rebuild the legacy Chat DTO"),
        ),
        patch(
            "router_maestro.providers.openai_base.httpx.AsyncClient",
            return_value=upstream_client,
        ),
    ):
        response = client.post("/api/openai/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    assert len(requests) == 1
    outbound = json.loads(requests[0].content)
    assert outbound["stream_options"] == {
        "include_usage": True,
        "future_option": {"keep": 1},
    }


@pytest.mark.asyncio
async def test_openai_chat_binding_accepts_cross_protocol_encoded_payload(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_response())

    envelope = RequestEnvelope(
        AnthropicMessagesRuntime(origin_provider=provider.name),
        {
            "model": f"{provider.name}/upstream-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "stream": False,
        },
    )
    legacy_factory = Mock(side_effect=AssertionError("raw binding must not build ChatRequest"))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        result = await GenerationDispatcher(
            {
                WireProtocol.OPENAI_CHAT: OpenAIChatRuntime(
                    origin_provider=provider.name,
                    default_model="upstream-model",
                )
            },
            execution=LegacyProviderExecutionAdapter(chat_request_factory=legacy_factory),
        ).dispatch(_router(provider), envelope)

    assert isinstance(result.value, SemanticResponse)
    assert result.value.model == "upstream-model"
    assert envelope.materialization_count == 1
    legacy_factory.assert_not_called()
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {
        "model": "upstream-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "max_tokens": 32,
    }


class _TrackingStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.closed = False

    async def __aiter__(self):
        yield self.body

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_openai_chat_stream_yields_first_raw_frame_and_closes_all_contexts(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider
    first_frame = {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "hello",
                    "future_delta": {"kept": True},
                },
                "finish_reason": None,
            }
        ],
        "future_frame": ["kept"],
    }
    second_frame = {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "model": "upstream-model",
        "choices": [{"index": 0, "delta": {"content": " world"}}],
    }
    upstream = _TrackingStream(
        (
            ": keepalive\n\n"
            f"data: {json.dumps(first_frame)}\n\n"
            f"data: {json.dumps(second_frame)}\n\n"
            "data: [DONE]\n\n"
        ).encode()
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=upstream)

    attempt = await _binding(provider).prepare_attempt(
        model=_model(provider),
        payload={"messages": [{"role": "user", "content": "hello"}]},
        stream=True,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        executor = _binding(provider).executor
        assert executor is not None
        iterator = executor.execute_stream(attempt)
        assert await anext(iterator) == first_frame
        assert client.is_closed is False
        await close_async_iterator(iterator)

    assert upstream.closed is True
    assert client.is_closed is False
    await provider.close()
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_openai_chat_stream_does_not_emit_keepalives_empty_data_or_done(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider
    frame = {
        "id": "chatcmpl_1",
        "object": "chat.completion.chunk",
        "model": "upstream-model",
        "choices": [{"index": 0, "delta": {"content": "hello"}}],
    }
    upstream = _TrackingStream(
        (f": keepalive\n\ndata:\n\ndata: {json.dumps(frame)}\n\ndata: [DONE]\n\n").encode()
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=upstream))
    )
    attempt = await _binding(provider).prepare_attempt(
        model=_model(provider),
        payload={"messages": [{"role": "user", "content": "hello"}]},
        stream=True,
    )

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        executor = _binding(provider).executor
        assert executor is not None
        frames = [item async for item in executor.execute_stream(attempt)]

    assert frames == [frame]
    assert upstream.closed is True
    assert client.is_closed is False
    await provider.close()
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_openai_chat_executor_reuses_provider_http_error_classification(
    provider_case: _ProviderCase,
) -> None:
    provider = provider_case.provider
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                429,
                json={"error": {"message": "slow down"}},
            )
        )
    )
    attempt = await _binding(provider).prepare_attempt(
        model=_model(provider),
        payload={"messages": [{"role": "user", "content": "hello"}]},
        stream=False,
    )

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        executor = _binding(provider).executor
        assert executor is not None
        with pytest.raises(ProviderError) as exc_info:
            await executor.execute(attempt)

    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True
    assert exc_info.value.kind is ProviderFailureKind.RATE_LIMIT
    assert exc_info.value.provider == provider.name
    assert exc_info.value.model == "upstream-model"
