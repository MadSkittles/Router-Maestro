"""Focused tests for provider bindings and immutable transport plans."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from router_maestro.protocols import ConversionMode, WireProtocol
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
)
from router_maestro.providers.bindings import (
    LEGACY_OPENAI_CHAT_BINDING,
    LEGACY_OPENAI_RESPONSES_BINDING,
    AttemptRequestContext,
    EndpointBinding,
    HttpExecutor,
    PreparedAttempt,
    ProviderDialect,
)
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.transport_plan import FlowCandidate, TransportPlan


class _LegacyProvider(BaseProvider):
    name = "legacy"

    def __init__(self, operations: frozenset[Operation] | None = None) -> None:
        self._operations = operations

    @property
    def capabilities(self) -> ProviderCapabilities:
        if self._operations is None:
            return super().capabilities
        return ProviderCapabilities(operations=self._operations)

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="ok", model=request.model)

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="ok", finish_reason="stop")

    async def list_models(self) -> list[ModelInfo]:
        return []

    def is_authenticated(self) -> bool:
        return True


class _Executor:
    async def execute(self, attempt: PreparedAttempt) -> object:
        return attempt

    def execute_stream(self, attempt: PreparedAttempt) -> AsyncIterator[object]:
        async def frames() -> AsyncIterator[object]:
            yield attempt

        return frames()


class _Dialect:
    id = "test-dialect"

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
        return PreparedAttempt(
            binding_id=binding_id,
            protocol=protocol,
            model=model,
            url="https://provider.example/v1/responses",
            payload=payload,
            headers={"authorization": "Bearer test"},
            stream=stream,
        )


def test_attempt_request_context_is_isolated_and_case_insensitive() -> None:
    source_headers = {"Anthropic-Beta": "feature-a"}
    source_query = {"alt": "sse"}
    context = AttemptRequestContext(
        path="/api/anthropic/v1/messages",
        query=source_query,
        headers=source_headers,
    )

    source_headers["Anthropic-Beta"] = "mutated"
    source_query["alt"] = "json"

    assert context.path == "/api/anthropic/v1/messages"
    assert dict(context.query) == {"alt": "sse"}
    assert dict(context.headers) == {"Anthropic-Beta": "feature-a"}
    assert context.header("anthropic-beta") == "feature-a"
    with pytest.raises(FrozenInstanceError):
        context.path = "/mutated"  # type: ignore[misc]


def _binding(
    protocol: WireProtocol,
    *,
    dialect: ProviderDialect | None = None,
    executor: HttpExecutor | None = None,
) -> EndpointBinding:
    operation = Operation.RESPONSES if protocol is WireProtocol.OPENAI_RESPONSES else Operation.CHAT
    return EndpointBinding(
        id=f"{protocol.value}-binding",
        protocol=protocol,
        capabilities=ProviderCapabilities(operations=frozenset({operation})),
        dialect=dialect,
        executor=executor,
    )


def test_base_provider_projects_legacy_chat_capabilities_to_a_binding() -> None:
    provider = _LegacyProvider()

    assert [(binding.id, binding.protocol) for binding in provider.bindings()] == [
        (LEGACY_OPENAI_CHAT_BINDING, WireProtocol.OPENAI_CHAT)
    ]
    assert provider.bindings()[0].capabilities.operations == frozenset(
        {Operation.CHAT, Operation.CHAT_STREAM}
    )
    assert provider.bindings()[0].is_legacy
    assert provider.transport_preferences() == (LEGACY_OPENAI_CHAT_BINDING,)


def test_base_provider_projects_only_implemented_chat_and_responses_operations() -> None:
    provider = _LegacyProvider(
        frozenset(
            {
                Operation.CHAT_STREAM,
                Operation.RESPONSES,
                Operation.RESPONSES_STREAM,
                Operation.NATIVE_ANTHROPIC,
            }
        )
    )

    chat, responses = provider.bindings()

    assert chat.id == LEGACY_OPENAI_CHAT_BINDING
    assert chat.capabilities.operations == frozenset({Operation.CHAT_STREAM})
    assert responses.id == LEGACY_OPENAI_RESPONSES_BINDING
    assert responses.capabilities.operations == frozenset(
        {Operation.RESPONSES, Operation.RESPONSES_STREAM}
    )
    assert provider.transport_preferences() == (chat.id, responses.id)


def test_endpoint_binding_requires_dialect_and_executor_as_a_pair() -> None:
    with pytest.raises(ValueError, match="both be set or both be omitted"):
        _binding(WireProtocol.OPENAI_RESPONSES, dialect=_Dialect())


@pytest.mark.asyncio
async def test_endpoint_binding_prepares_a_snapshotted_attempt() -> None:
    dialect = _Dialect()
    executor = _Executor()
    binding = _binding(
        WireProtocol.OPENAI_RESPONSES,
        dialect=dialect,
        executor=executor,
    )
    model = ModelRef(provider="legacy", upstream_id="gpt-test")
    payload = {"input": [{"type": "input_text", "text": "hello"}]}

    attempt = await binding.prepare_attempt(model=model, payload=payload, stream=True)
    payload["input"][0]["text"] = "mutated"

    assert isinstance(dialect, ProviderDialect)
    assert isinstance(executor, HttpExecutor)
    assert attempt.binding_id == binding.id
    assert attempt.protocol is WireProtocol.OPENAI_RESPONSES
    assert attempt.payload["input"][0]["text"] == "hello"
    assert attempt.stream is True
    assert attempt.method == "POST"
    with pytest.raises(TypeError):
        attempt.headers["x-new"] = "value"  # type: ignore[index]


def test_flow_candidate_selects_identity_without_semantic_ir() -> None:
    binding = _binding(WireProtocol.OPENAI_RESPONSES)

    candidate = FlowCandidate.for_binding(
        source_protocol=WireProtocol.OPENAI_RESPONSES,
        binding=binding,
    )

    assert candidate.conversion_mode is ConversionMode.IDENTITY
    assert candidate.target_protocol is WireProtocol.OPENAI_RESPONSES


def test_flow_candidate_selects_semantic_ir_only_for_cross_protocol() -> None:
    binding = _binding(WireProtocol.OPENAI_RESPONSES)

    candidate = FlowCandidate.for_binding(
        source_protocol=WireProtocol.ANTHROPIC_MESSAGES,
        binding=binding,
    )

    assert candidate.conversion_mode is ConversionMode.SEMANTIC_IR
    with pytest.raises(ValueError, match="requires semantic_ir conversion"):
        FlowCandidate(
            source_protocol=WireProtocol.ANTHROPIC_MESSAGES,
            binding=binding,
            conversion_mode=ConversionMode.IDENTITY,
        )


def test_transport_plan_freezes_candidate_metadata_without_a_payload() -> None:
    provider = _LegacyProvider()
    model = ModelRef(provider="legacy", upstream_id="gpt-test")
    candidate = FlowCandidate.for_binding(
        source_protocol=WireProtocol.ANTHROPIC_MESSAGES,
        binding=_binding(WireProtocol.OPENAI_RESPONSES),
    )
    plan = TransportPlan(model=model, provider=provider, candidate=candidate)

    assert plan.binding is candidate.binding
    assert plan.flow is candidate
    assert plan.source_protocol is WireProtocol.ANTHROPIC_MESSAGES
    assert plan.target_protocol is WireProtocol.OPENAI_RESPONSES
    assert plan.conversion_mode is ConversionMode.SEMANTIC_IR
    assert not hasattr(plan, "payload")
    with pytest.raises(FrozenInstanceError):
        plan.model = ModelRef(provider="legacy", upstream_id="other")  # type: ignore[misc]


def test_transport_plan_rejects_provider_model_mismatch() -> None:
    candidate = FlowCandidate.for_binding(
        source_protocol=WireProtocol.OPENAI_CHAT,
        binding=_binding(WireProtocol.OPENAI_CHAT),
    )

    with pytest.raises(ValueError, match="provider must match"):
        TransportPlan(
            model=ModelRef(provider="other", upstream_id="model"),
            provider=_LegacyProvider(),
            candidate=candidate,
        )
