"""Production Responses routes wired through the protocol-aware dispatcher."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.config import FallbackConfig, FallbackStrategy, PrioritiesConfig
from router_maestro.protocols import WireProtocol
from router_maestro.protocols.openai_responses import OpenAIResponsesRuntime
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
)
from router_maestro.providers.bindings import EndpointBinding, PreparedAttempt
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.runtime.reasoning_capsule import (
    ReasoningCapsuleCodec,
    ReasoningCapsulePayload,
    serialize_opaque_state,
)
from router_maestro.server.routes.openai_responses_beta import (
    router as responses_beta_router,
)
from router_maestro.server.routes.responses import router as responses_router
from router_maestro.utils.cache import TTLCache


class _ChatOnlyProvider(BaseProvider):
    name = "chat-only"

    def __init__(
        self,
        *,
        fail_before_first_frame: bool = False,
        fail_after_first_frame: bool = False,
    ) -> None:
        self.requests: list[ChatRequest] = []
        self.fail_before_first_frame = fail_before_first_frame
        self.fail_after_first_frame = fail_after_first_frame

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM}))

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="chat-model",
                name="chat-model",
                provider=self.name,
                supported_endpoints=("/chat/completions",),
            )
        ]

    def is_authenticated(self) -> bool:
        return True

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            content="hello from chat",
            model=request.model,
            finish_reason="tool_calls",
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            tool_calls=[
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"rm"}',
                    },
                }
            ],
        )

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        self.requests.append(request)
        if self.fail_before_first_frame:
            raise ProviderError(
                "upstream unavailable before first frame",
                status_code=503,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_STATUS,
            )
        yield ChatStreamChunk(content="hello ")
        if self.fail_after_first_frame:
            raise ProviderError(
                "upstream disconnected after commit",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            )
        yield ChatStreamChunk(
            content="",
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"rm"}',
                    },
                }
            ],
        )
        yield ChatStreamChunk(
            content="",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )


class _RawResponsesDialect:
    id = "raw-responses-dialect"

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []

    async def prepare_attempt(
        self,
        *,
        binding_id: str,
        protocol: WireProtocol,
        model: ModelRef,
        payload: Mapping[str, Any],
        stream: bool,
        request_context,
    ) -> PreparedAttempt:
        del request_context
        self.received.append(deepcopy(dict(payload)))
        outbound = deepcopy(dict(payload))
        outbound["model"] = model.upstream_id
        outbound["stream"] = stream
        return PreparedAttempt(
            binding_id=binding_id,
            protocol=protocol,
            model=model,
            url="https://provider.invalid/responses",
            payload=outbound,
            stream=stream,
        )


class _RawResponsesExecutor:
    def __init__(self) -> None:
        self.attempts: list[PreparedAttempt] = []

    async def execute(self, attempt: PreparedAttempt) -> Mapping[str, Any]:
        self.attempts.append(attempt)
        return {
            "id": "resp_identity",
            "object": "response",
            "created_at": 1,
            "model": attempt.model.upstream_id,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_identity",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "identity", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "future_response_field": {"preserved": True},
        }

    def execute_stream(self, attempt: PreparedAttempt) -> AsyncIterator[Mapping[str, Any]]:
        async def frames() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp_identity",
                    "model": attempt.model.upstream_id,
                    "status": "completed",
                    "output": [],
                },
            }

        return frames()


class _RawResponsesProvider(BaseProvider):
    name = "raw-responses"

    def __init__(self) -> None:
        self.dialect = _RawResponsesDialect()
        self.executor = _RawResponsesExecutor()
        self.binding = EndpointBinding(
            id="raw-responses-binding",
            protocol=WireProtocol.OPENAI_RESPONSES,
            capabilities=ProviderCapabilities(
                operations=frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM})
            ),
            dialect=self.dialect,
            executor=self.executor,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.binding.capabilities

    def bindings(self) -> tuple[EndpointBinding, ...]:
        return (self.binding,)

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="identity-model",
                name="identity-model",
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
        if False:  # pragma: no cover - satisfy the async iterator contract
            yield ChatStreamChunk(content="")


def _router(provider: BaseProvider, model: str) -> Router:
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
            priorities=[f"{provider.name}/{model}"],
            fallback=FallbackConfig(strategy=FallbackStrategy.NONE, maxRetries=0),
        )
    )
    model_router._providers_ttl.set(True)
    return model_router


def _client() -> TestClient:
    app = FastAPI()
    app.state.reasoning_capsule_codec = ReasoningCapsuleCodec(bytes([57]) * 32)
    app.include_router(responses_router)
    app.include_router(responses_beta_router)
    return TestClient(app, raise_server_exceptions=False)


def _payload(*, model: str = "chat-only/chat-model", stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "input": "hello",
        "stream": stream,
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }


def _sse_events(body: str) -> list[dict[str, Any]]:
    events = []
    for frame in body.split("\n\n"):
        data_line = next(
            (
                line.removeprefix("data: ")
                for line in frame.splitlines()
                if line.startswith("data: ")
            ),
            None,
        )
        if data_line:
            events.append(json.loads(data_line))
    return events


def test_stable_responses_route_crosses_to_chat_for_text_and_tools() -> None:
    provider = _ChatOnlyProvider()
    model_router = _router(provider, "chat-model")

    with patch(
        "router_maestro.server.routes.responses.get_router",
        return_value=model_router,
    ):
        response = _client().post("/api/openai/v1/responses", json=_payload())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "chat-only/chat-model"
    assert [item["type"] for item in body["output"]] == ["message", "function_call"]
    assert body["output"][0]["content"][0]["text"] == "hello from chat"
    assert body["output"][1]["call_id"] == "call_1"
    assert body["output"][1]["arguments"] == '{"query":"rm"}'
    assert len(provider.requests) == 1
    assert provider.requests[0].model == "chat-model"
    assert provider.requests[0].tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]


def test_responses_summary_only_reasoning_history_crosses_to_chat() -> None:
    provider = _ChatOnlyProvider()
    model_router = _router(provider, "chat-model")
    payload = _payload()
    payload["input"] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Remember cobalt."}],
        },
        {
            "type": "reasoning",
            "id": "rs_summary",
            "summary": [{"type": "summary_text", "text": "I should remember cobalt."}],
        },
        {
            "type": "message",
            "id": "msg_ack",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ACK"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What should you remember?"}],
        },
    ]

    with patch(
        "router_maestro.server.routes.responses.get_router",
        return_value=model_router,
    ):
        response = _client().post("/api/openai/v1/responses", json=payload)

    assert response.status_code == 200, response.text
    assert len(provider.requests) == 1
    assert [message.role for message in provider.requests[0].messages] == [
        "user",
        "assistant",
        "assistant",
        "user",
    ]


def test_stable_responses_route_streams_cross_protocol_text_and_tool_events() -> None:
    provider = _ChatOnlyProvider()
    model_router = _router(provider, "chat-model")

    with patch(
        "router_maestro.server.routes.responses.get_router",
        return_value=model_router,
    ):
        with _client().stream(
            "POST",
            "/api/openai/v1/responses",
            json=_payload(stream=True),
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200, body
    events = _sse_events(body)
    event_types = [event["type"] for event in events]
    assert event_types.index("response.output_item.added") < event_types.index(
        "response.output_text.delta"
    )
    assert event_types.index("response.content_part.added") < event_types.index(
        "response.output_text.delta"
    )
    message_done_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "response.output_item.done" and event["output_index"] == 0
    )
    assert event_types.index("response.output_text.done") < message_done_index
    assert event_types.index("response.function_call_arguments.delta") < event_types.index(
        "response.function_call_arguments.done"
    )
    assert event_types[-1] == "response.completed"
    completed = events[-1]["response"]
    assert completed["model"] == "chat-only/chat-model"
    assert [item["type"] for item in completed["output"]] == ["message", "function_call"]
    assert completed["output"][0]["content"] == [{"type": "output_text", "text": "hello "}]
    assert completed["output"][1]["arguments"] == '{"query":"rm"}'


def test_beta_responses_path_is_an_alias_of_the_shared_dispatcher() -> None:
    provider = _ChatOnlyProvider()
    model_router = _router(provider, "chat-model")

    with patch(
        "router_maestro.server.routes.responses.get_router",
        return_value=model_router,
    ):
        response = _client().post("/api/openai/beta/v1/responses", json=_payload())

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "chat-only/chat-model"
    assert response.json()["output"][1]["type"] == "function_call"
    assert len(provider.requests) == 1


def test_responses_unknown_capsule_version_fails_before_provider_io() -> None:
    provider = _ChatOnlyProvider()
    model_router = _router(provider, "chat-model")
    payload = _payload()
    payload["input"] = [
        {
            "type": "reasoning",
            "id": "rs_future",
            "summary": [],
            "encrypted_content": "rmr2.unknown.payload",
        },
        {"role": "user", "content": "continue"},
    ]

    with patch(
        "router_maestro.server.routes.responses.get_router",
        return_value=model_router,
    ):
        response = _client().post("/api/openai/v1/responses", json=payload)

    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == "Invalid reasoning capsule"
    assert provider.requests == []


def test_stream_failure_before_first_frame_remains_an_http_error() -> None:
    provider = _ChatOnlyProvider(fail_before_first_frame=True)
    model_router = _router(provider, "chat-model")

    with patch(
        "router_maestro.server.routes.responses.get_router",
        return_value=model_router,
    ):
        response = _client().post(
            "/api/openai/v1/responses",
            json=_payload(stream=True),
        )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["message"] == "upstream unavailable before first frame"


def test_stream_failure_after_first_frame_is_a_typed_sse_error() -> None:
    provider = _ChatOnlyProvider(fail_after_first_frame=True)
    model_router = _router(provider, "chat-model")

    with patch(
        "router_maestro.server.routes.responses.get_router",
        return_value=model_router,
    ):
        response = _client().post(
            "/api/openai/v1/responses",
            json=_payload(stream=True),
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    assert any(event["type"] == "response.output_text.delta" for event in events)
    assert events[-1] == {
        "type": "error",
        "code": "upstream_protocol_error",
        "message": "upstream disconnected after commit",
    }


@pytest.mark.parametrize(
    ("patch_target", "path"),
    [
        (
            "router_maestro.server.routes.responses.get_router",
            "/api/openai/v1/responses",
        ),
        (
            "router_maestro.server.routes.responses.get_router",
            "/api/openai/beta/v1/responses",
        ),
    ],
    ids=["stable", "beta-alias"],
)
def test_identity_responses_preserves_raw_fields_without_decoding_ir(
    monkeypatch: pytest.MonkeyPatch,
    patch_target: str,
    path: str,
) -> None:
    provider = _RawResponsesProvider()
    model_router = _router(provider, "identity-model")

    async def fail_if_decoded(_runtime, _payload):
        raise AssertionError("Responses identity request must not materialize semantic IR")

    monkeypatch.setattr(OpenAIResponsesRuntime, "decode_request", fail_if_decoded)
    payload = {
        "model": "raw-responses/identity-model",
        "input": "hello",
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "cache-key",
        "future_option": {"nested": [1, 2, 3]},
    }

    with patch(patch_target, return_value=model_router):
        response = _client().post(path, json=payload)

    assert response.status_code == 200, response.text
    assert provider.dialect.received == [payload]
    assert dict(provider.executor.attempts[0].payload) == {
        **payload,
        "model": "identity-model",
        "stream": False,
    }
    body = response.json()
    assert body["model"] == "raw-responses/identity-model"
    assert body["future_response_field"] == {"preserved": True}


def test_identity_responses_restores_capsule_without_materializing_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RawResponsesProvider()
    model_router = _router(provider, "identity-model")
    codec = ReasoningCapsuleCodec(bytes([57]) * 32)
    raw_reasoning = {
        "type": "reasoning",
        "id": "rs_identity",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "plan"}],
        "encrypted_content": "provider-opaque-state",
        "future_reasoning_field": {"preserved": True},
    }
    capsule = codec.seal(
        ReasoningCapsulePayload(
            provider=provider.name,
            model="identity-model",
            transport=provider.binding.id,
            item_id="rs_identity",
            opaque_state=serialize_opaque_state(raw_reasoning),
        )
    )
    payload = {
        "model": "raw-responses/identity-model",
        "input": [
            {
                "type": "reasoning",
                "id": "rs_identity",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "encrypted_content": capsule,
            },
            {"role": "user", "content": "continue"},
        ],
    }

    async def fail_if_decoded(_runtime, _payload):
        raise AssertionError("Responses identity capsule replay must not materialize IR")

    monkeypatch.setattr(OpenAIResponsesRuntime, "decode_request", fail_if_decoded)
    with patch(
        "router_maestro.server.routes.responses.get_router",
        return_value=model_router,
    ):
        response = _client().post("/api/openai/v1/responses", json=payload)

    assert response.status_code == 200, response.text
    assert provider.dialect.received[0]["input"][0] == raw_reasoning
    assert provider.executor.attempts[0].payload["input"][0] == raw_reasoning
    assert payload["input"][0]["encrypted_content"] == capsule
