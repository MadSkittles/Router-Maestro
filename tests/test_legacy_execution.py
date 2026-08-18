"""Focused wire-to-legacy execution bridge contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from router_maestro.protocols import (
    MessageRole,
    RequestEnvelope,
    RequestManifest,
    SemanticEvent,
    SemanticMessage,
    SemanticRequest,
    SemanticResponse,
    TextContent,
    WireProtocol,
)
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    RequestOptionError,
)
from router_maestro.providers.bindings import legacy_endpoint_binding
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.transport_plan import FlowCandidate, TransportPlan
from router_maestro.server.legacy_execution import (
    chat_request_from_wire,
    legacy_execution_adapter,
    responses_request_from_wire,
)


class _IngressRuntime:
    protocol = WireProtocol.OPENAI_CHAT

    def __init__(self) -> None:
        self.decode_calls = 0

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        return RequestManifest(
            protocol=self.protocol,
            model=payload.get("model") if isinstance(payload.get("model"), str) else None,
            stream=payload.get("stream") is True,
        )

    async def decode_request(self, payload: Mapping[str, Any]) -> SemanticRequest:
        self.decode_calls += 1
        return SemanticRequest(
            model=str(payload["model"]),
            input=(
                SemanticMessage(
                    role=MessageRole.USER,
                    content=(TextContent("decoded"),),
                ),
            ),
        )

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        del request
        raise AssertionError("encode_request was not expected")

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        del payload
        raise AssertionError("decode_response was not expected")

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        del response
        raise AssertionError("encode_response was not expected")

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        del payload
        raise AssertionError("decode_stream_event was not expected")

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        del event
        raise AssertionError("encode_stream_event was not expected")


class _CaptureProvider(BaseProvider):
    name = "capture"

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(operations=frozenset({Operation.CHAT}))

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(content="ok", model=request.model)

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        if False:  # pragma: no cover - satisfies the abstract async iterator contract
            yield ChatStreamChunk(content="")

    async def list_models(self) -> list[ModelInfo]:
        return []

    def is_authenticated(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_identity_legacy_execution_uses_native_payload_without_semantic_ir() -> None:
    runtime = _IngressRuntime()
    envelope = RequestEnvelope(
        runtime,
        {
            "model": "capture/public-model",
            "messages": [{"role": "user", "content": "native"}],
            "future_option": {"nested": [1]},
        },
    )
    provider = _CaptureProvider()
    binding = legacy_endpoint_binding(
        binding_id="capture-chat",
        protocol=WireProtocol.OPENAI_CHAT,
        operations=frozenset({Operation.CHAT}),
    )
    plan = TransportPlan(
        model=ModelRef(provider="capture", upstream_id="upstream-model"),
        provider=provider,
        candidate=FlowCandidate.for_binding(
            source_protocol=WireProtocol.OPENAI_CHAT,
            binding=binding,
        ),
    )

    response = await legacy_execution_adapter().execute(plan, envelope.native_payload())

    assert response.content == "ok"
    assert envelope.materialization_count == 0
    assert runtime.decode_calls == 0
    assert len(provider.requests) == 1
    captured = provider.requests[0]
    assert captured.model == "upstream-model"
    assert captured.messages[0].content == "native"
    assert captured.provider_extensions == {"future_option": {"nested": [1]}}


def test_chat_unknown_wire_fields_are_snapshotted_as_provider_extensions() -> None:
    payload = {
        "model": "public-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "stream_options": {"include_usage": True},
        "future_option": {"flags": ["one"]},
    }

    request = chat_request_from_wire(payload, model="upstream-model", stream=True)
    payload["future_option"]["flags"].append("mutated")

    assert request.model == "upstream-model"
    assert request.stream is True
    assert request.provider_extensions == {
        "stream_options": {"include_usage": True},
        "future_option": {"flags": ["one"]},
    }


@pytest.mark.parametrize("parallel_tool_calls", [False, True])
def test_chat_parallel_tool_calls_fails_closed_when_dto_cannot_carry_it(
    parallel_tool_calls: bool,
) -> None:
    with pytest.raises(RequestOptionError) as raised:
        chat_request_from_wire(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "parallel_tool_calls": parallel_tool_calls,
            },
            model="upstream-model",
            stream=False,
        )

    assert raised.value.parameter == "parallel_tool_calls"


def test_responses_continuation_and_future_fields_remain_extensions() -> None:
    payload = {
        "model": "public-model",
        "input": [{"role": "user", "content": "hello"}],
        "stream": False,
        "previous_response_id": "resp_previous",
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "prompt_cache_key": "cache-key",
        "text": {"format": {"type": "json_schema", "name": "answer"}},
        "future_option": {"flags": ["one"]},
    }

    request = responses_request_from_wire(payload, model="upstream-model", stream=True)
    payload["include"].append("mutated")
    payload["future_option"]["flags"].append("mutated")

    assert request.model == "upstream-model"
    assert request.stream is True
    assert request.provider_extensions == {
        "previous_response_id": "resp_previous",
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "prompt_cache_key": "cache-key",
        "text": {"format": {"type": "json_schema", "name": "answer"}},
        "future_option": {"flags": ["one"]},
    }


def test_responses_preserves_unmodeled_reasoning_members_without_losing_effort() -> None:
    request = responses_request_from_wire(
        {
            "input": "hello",
            "reasoning": {
                "effort": "high",
                "summary": "detailed",
                "future_option": True,
            },
        },
        model="upstream-model",
        stream=False,
    )

    assert request.reasoning_effort == "high"
    assert request.provider_extensions["reasoning"] == {
        "effort": "high",
        "summary": "detailed",
        "future_option": True,
    }


@pytest.mark.parametrize("parallel_tool_calls", [False, True])
def test_responses_parallel_tool_calls_is_preserved_when_dto_supports_it(
    parallel_tool_calls: bool,
) -> None:
    request = responses_request_from_wire(
        {"input": "hello", "parallel_tool_calls": parallel_tool_calls},
        model="upstream-model",
        stream=False,
    )

    assert request.parallel_tool_calls is parallel_tool_calls
    assert "parallel_tool_calls" not in request.provider_extensions


def test_responses_rejects_non_boolean_parallel_tool_calls() -> None:
    with pytest.raises(RequestOptionError) as raised:
        responses_request_from_wire(
            {"input": "hello", "parallel_tool_calls": "yes"},
            model="upstream-model",
            stream=False,
        )

    assert raised.value.parameter == "parallel_tool_calls"
