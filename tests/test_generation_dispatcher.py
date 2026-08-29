"""Focused execution contracts for the protocol-aware generation dispatcher."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import pytest

from router_maestro.config import FallbackConfig, FallbackStrategy, PrioritiesConfig
from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    ConversionMode,
    GeminiRuntime,
    OpenAIChatRuntime,
    OpenAIResponsesRuntime,
    ProtocolRepresentabilityError,
    RepresentabilityReport,
    RequestEnvelope,
    RequestManifest,
    SemanticEvent,
    SemanticRequest,
    SemanticResponse,
    WireProtocol,
)
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
)
from router_maestro.providers.bindings import (
    EndpointBinding,
    PreparedAttempt,
    legacy_endpoint_binding,
)
from router_maestro.routing.capabilities import Feature, Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.runtime import request_context as request_context_module
from router_maestro.runtime.reasoning_capsule import (
    ReasoningCapsuleCodec,
    ReasoningCapsulePayload,
)
from router_maestro.runtime.request_context import RequestContext
from router_maestro.server.dispatcher import (
    DispatchAttemptObservation,
    DispatchAttemptOutcome,
    GenerationDispatcher,
)
from router_maestro.utils.async_iterators import close_async_iterator
from router_maestro.utils.cache import TTLCache


def _binding(protocol: WireProtocol, name: str) -> EndpointBinding:
    operations = {
        WireProtocol.ANTHROPIC_MESSAGES: frozenset({Operation.NATIVE_ANTHROPIC}),
        WireProtocol.OPENAI_CHAT: frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        WireProtocol.OPENAI_RESPONSES: frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM}),
        WireProtocol.GEMINI: frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    }[protocol]
    return legacy_endpoint_binding(
        binding_id=name,
        protocol=protocol,
        operations=operations,
    )


class _Provider(BaseProvider):
    def __init__(
        self,
        name: str,
        model: str,
        bindings: tuple[EndpointBinding, ...],
        *,
        model_info: ModelInfo | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._bindings = bindings
        self._model_info = model_info or ModelInfo(id=model, name=model, provider=name)
        self.list_models_calls = 0

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(operations=frozenset(Operation))

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="unused", model=request.model)

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="unused")

    async def list_models(self) -> list[ModelInfo]:
        self.list_models_calls += 1
        return [self._model_info]

    def is_authenticated(self) -> bool:
        return True

    def bindings(self) -> tuple[EndpointBinding, ...]:
        return self._bindings


def _router(
    providers: tuple[_Provider, ...],
    *,
    max_retries: int,
) -> Router:
    router = Router.__new__(Router)
    router.providers = {provider.name: provider for provider in providers}
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._model_aliases = None
    router._managed_generation = True
    router._priorities_cache.set(
        PrioritiesConfig(
            priorities=[f"{provider.name}/{provider.model}" for provider in providers],
            fallback=FallbackConfig(
                strategy=FallbackStrategy.PRIORITY,
                maxRetries=max_retries,
            ),
        )
    )
    router._providers_ttl.set(True)
    return router


class _Runtime:
    def __init__(
        self,
        protocol: WireProtocol,
        *,
        encode_error: BaseException | None = None,
        reasoning_capsules: tuple[str, ...] = (),
        previous_response_id: str | None = None,
        opaque_continuation: bool = False,
        tools: bool = False,
    ) -> None:
        self.protocol = protocol
        self.encode_error = encode_error
        self.reasoning_capsules = reasoning_capsules
        self.previous_response_id = previous_response_id
        self.opaque_continuation = opaque_continuation
        self.tools = tools
        self.decode_calls = 0
        self.encode_calls = 0
        self.encoded_requests: list[SemanticRequest] = []

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        model = payload.get("model")
        return RequestManifest(
            protocol=self.protocol,
            model=model if isinstance(model, str) else None,
            stream=payload.get("stream") is True,
            reasoning_capsules=self.reasoning_capsules,
            previous_response_id=self.previous_response_id,
            opaque_continuation=self.opaque_continuation,
            tools=self.tools,
        )

    async def decode_request(self, payload: Mapping[str, Any]) -> SemanticRequest:
        self.decode_calls += 1
        return SemanticRequest(
            model=str(payload["model"]),
            stream=payload.get("stream") is True,
        )

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        self.encode_calls += 1
        self.encoded_requests.append(request)
        if self.encode_error is not None:
            raise self.encode_error
        return {
            "model": request.model,
            "stream": request.stream,
            "encoded_for": self.protocol.value,
        }

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


class _Stream:
    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.closed = False

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> Any:
        if not self.items:
            raise StopAsyncIteration
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


class _Execution:
    def __init__(
        self,
        *,
        actions: Mapping[tuple[str, str], Any] | None = None,
        streams: Mapping[tuple[str, str], _Stream | BaseException] | None = None,
    ) -> None:
        self.actions = dict(actions or {})
        self.streams = dict(streams or {})
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []
        self.stream_calls: list[tuple[str, str, Mapping[str, Any]]] = []

    async def execute(self, plan, payload: Mapping[str, Any], *, request_context=None) -> Any:
        del request_context
        key = (plan.provider.name, plan.binding.id)
        self.calls.append((*key, payload))
        action = self.actions[key]
        if isinstance(action, BaseException):
            raise action
        return action

    async def open_stream(
        self,
        plan,
        payload: Mapping[str, Any],
        *,
        request_context=None,
    ) -> AsyncIterator[Any]:
        del request_context
        key = (plan.provider.name, plan.binding.id)
        self.stream_calls.append((*key, payload))
        action = self.streams[key]
        if isinstance(action, BaseException):
            raise action
        return action


class _AuditSpy:
    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []

    def record_dispatch_attempt(self, **values: Any) -> None:
        self.attempts.append(values)


class _TrackingCodec(ReasoningCapsuleCodec):
    def __init__(self, key: bytes) -> None:
        super().__init__(key)
        self.routing_calls: list[str] = []
        self.execution_calls: list[tuple[str, str, str, str]] = []

    def unseal_for_routing(self, capsule: str) -> ReasoningCapsulePayload:
        self.routing_calls.append(capsule)
        return super().unseal_for_routing(capsule)

    def unseal(
        self,
        capsule: str,
        *,
        expected_provider: str,
        expected_model: str,
        expected_transport: str,
        expected_item_id: str | None = None,
    ) -> ReasoningCapsulePayload:
        self.execution_calls.append(
            (capsule, expected_provider, expected_model, expected_transport)
        )
        return super().unseal(
            capsule,
            expected_provider=expected_provider,
            expected_model=expected_model,
            expected_transport=expected_transport,
            expected_item_id=expected_item_id,
        )


def _capsule(
    codec: ReasoningCapsuleCodec,
    *,
    provider: str,
    model: str,
    transport: str,
    item_id: str = "rs_1",
) -> str:
    return codec.seal(
        ReasoningCapsulePayload(
            provider=provider,
            model=model,
            transport=transport,
            item_id=item_id,
            opaque_state=f"opaque-{item_id}",
        )
    )


def _retryable(message: str, status_code: int = 503) -> ProviderError:
    return ProviderError(
        message,
        status_code=status_code,
        retryable=True,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )


@pytest.mark.asyncio
async def test_identity_dispatch_never_materializes_semantic_ir() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (chat,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(
        ingress,
        {
            "model": "alpha/one",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    execution = _Execution(actions={("alpha", "chat"): "ok"})

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.value == "ok"
    assert result.selection.plan.binding.id == "chat"
    assert envelope.materialization_count == 0
    assert ingress.decode_calls == 0
    assert execution.calls[0][2]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_cross_protocol_preflight_reuses_one_ir_and_encodes_once() -> None:
    class _PreflightRuntime(_Runtime):
        def __init__(self) -> None:
            super().__init__(WireProtocol.OPENAI_RESPONSES)
            self.preflight_calls = 0
            self.preflight_requests: list[SemanticRequest] = []

        async def request_representability(
            self,
            request: SemanticRequest,
        ) -> RepresentabilityReport:
            self.preflight_calls += 1
            self.preflight_requests.append(request)
            return RepresentabilityReport(representable=True)

    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (responses,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    target = _PreflightRuntime()
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution(actions={("alpha", "responses"): "ok"})

    result = await GenerationDispatcher(
        {WireProtocol.OPENAI_RESPONSES: target},
        execution=execution,
    ).dispatch(router, envelope)

    assert result.value == "ok"
    assert envelope.materialization_count == 1
    assert ingress.decode_calls == 1
    assert target.preflight_calls == 1
    assert target.encode_calls == 1
    assert target.preflight_requests[0] is target.encoded_requests[0]


@pytest.mark.asyncio
async def test_cross_protocol_preflight_rejection_skips_encode_and_provider_io() -> None:
    class _PreflightRuntime(_Runtime):
        def __init__(self) -> None:
            super().__init__(WireProtocol.OPENAI_RESPONSES)
            self.preflight_calls = 0

        def request_representability(
            self,
            request: SemanticRequest,
        ) -> RepresentabilityReport:
            self.preflight_calls += 1
            return RepresentabilityReport(
                representable=False,
                reasons=("field is unsupported",),
                parameter="top_k",
            )

    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (responses,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    target = _PreflightRuntime()
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution()

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher(
            {WireProtocol.OPENAI_RESPONSES: target},
            execution=execution,
        ).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.parameter == "top_k"
    assert isinstance(raised.value.cause, ProtocolRepresentabilityError)
    assert raised.value.cause.report.parameter == "top_k"
    assert target.preflight_calls == 1
    assert target.encode_calls == 0
    assert execution.calls == []


@pytest.mark.parametrize(
    "target_protocol", [WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_RESPONSES]
)
@pytest.mark.asyncio
async def test_anthropic_top_k_rejection_uses_ingress_parameter(
    target_protocol: WireProtocol,
) -> None:
    binding = _binding(target_protocol, target_protocol.value)
    provider = _Provider("alpha", "one", (binding,))
    router = _router((provider,), max_retries=0)
    envelope = RequestEnvelope(
        AnthropicMessagesRuntime(),
        {
            "model": "alpha/one",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "top_k": 8,
        },
    )
    runtime = (
        OpenAIChatRuntime()
        if target_protocol is WireProtocol.OPENAI_CHAT
        else OpenAIResponsesRuntime()
    )
    execution = _Execution()

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher(
            {target_protocol: runtime},
            execution=execution,
        ).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.parameter == "top_k"
    assert execution.calls == []
    assert envelope.materialization_count == 1


@pytest.mark.parametrize(
    "target_protocol", [WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_RESPONSES]
)
@pytest.mark.asyncio
async def test_gemini_top_k_rejection_uses_generation_config_parameter(
    target_protocol: WireProtocol,
) -> None:
    binding = _binding(target_protocol, target_protocol.value)
    provider = _Provider("alpha", "one", (binding,))
    router = _router((provider,), max_retries=0)
    envelope = RequestEnvelope(
        GeminiRuntime(),
        {
            "model": "alpha/one",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "generationConfig": {"topK": 8},
        },
    )
    runtime = (
        OpenAIChatRuntime()
        if target_protocol is WireProtocol.OPENAI_CHAT
        else OpenAIResponsesRuntime()
    )
    execution = _Execution()

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher(
            {target_protocol: runtime},
            execution=execution,
        ).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.parameter == "generationConfig.topK"
    assert execution.calls == []
    assert envelope.materialization_count == 1


@pytest.mark.parametrize(
    ("target_protocol", "wire_field"),
    [
        (WireProtocol.OPENAI_CHAT, "response_format"),
        (WireProtocol.OPENAI_RESPONSES, "text"),
    ],
)
@pytest.mark.asyncio
async def test_gemini_structured_output_reaches_target_wire_field(
    target_protocol: WireProtocol,
    wire_field: str,
) -> None:
    binding = _binding(target_protocol, target_protocol.value)
    provider = _Provider("alpha", "one", (binding,))
    router = _router((provider,), max_retries=0)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    envelope = RequestEnvelope(
        GeminiRuntime(),
        {
            "model": "alpha/one",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "generationConfig": {
                "responseJsonSchema": schema,
            },
        },
    )
    runtime = (
        OpenAIChatRuntime()
        if target_protocol is WireProtocol.OPENAI_CHAT
        else OpenAIResponsesRuntime()
    )
    execution = _Execution(actions={("alpha", target_protocol.value): "ok"})

    result = await GenerationDispatcher(
        {target_protocol: runtime},
        execution=execution,
    ).dispatch(router, envelope)

    assert result.value == "ok"
    expected = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": schema,
        },
    }
    if target_protocol is WireProtocol.OPENAI_RESPONSES:
        expected = {
            "format": {
                "type": "json_schema",
                "name": "response",
                "schema": schema,
            }
        }
    assert execution.calls[0][2][wire_field] == expected


@pytest.mark.parametrize(
    "target_protocol", [WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_RESPONSES]
)
@pytest.mark.asyncio
async def test_anthropic_ephemeral_tool_cache_hint_reaches_cross_protocol_provider(
    target_protocol: WireProtocol,
) -> None:
    binding = _binding(target_protocol, target_protocol.value)
    provider = _Provider("alpha", "one", (binding,))
    router = _router((provider,), max_retries=0)
    envelope = RequestEnvelope(
        AnthropicMessagesRuntime(),
        {
            "model": "alpha/one",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "lookup",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    )
    execution = _Execution(actions={("alpha", target_protocol.value): "ok"})

    result = await GenerationDispatcher(
        {
            target_protocol: (
                OpenAIChatRuntime()
                if target_protocol is WireProtocol.OPENAI_CHAT
                else OpenAIResponsesRuntime()
            )
        },
        execution=execution,
    ).dispatch(router, envelope)

    assert result.value == "ok"
    assert len(execution.calls) == 1
    outbound_tools = execution.calls[0][2]["tools"]
    assert "cache_control" not in outbound_tools[0]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_fake_gemini_native_binding_uses_identity_fast_path(stream: bool) -> None:
    terminal = {
        "modelVersion": "one",
        "candidates": [{"index": 0, "finishReason": "STOP"}],
    }

    class _Dialect:
        id = "fake-gemini-dialect"

        def __init__(self) -> None:
            self.request_contexts = []

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
            self.request_contexts.append(request_context)
            return PreparedAttempt(
                binding_id=binding_id,
                protocol=protocol,
                model=model,
                url=f"https://gemini.invalid/models/{model.upstream_id}:generateContent",
                payload=payload,
                stream=stream,
            )

    class _Executor:
        def __init__(self) -> None:
            self.attempts: list[PreparedAttempt] = []

        async def execute(self, attempt: PreparedAttempt) -> Mapping[str, Any]:
            self.attempts.append(attempt)
            return terminal

        def execute_stream(self, attempt: PreparedAttempt) -> AsyncIterator[Mapping[str, Any]]:
            self.attempts.append(attempt)

            async def frames() -> AsyncIterator[Mapping[str, Any]]:
                yield terminal

            return frames()

    executor = _Executor()
    dialect = _Dialect()
    gemini = EndpointBinding(
        id="gemini",
        protocol=WireProtocol.GEMINI,
        capabilities=ProviderCapabilities(
            operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM})
        ),
        dialect=dialect,
        executor=executor,
    )
    provider = _Provider("alpha", "one", (gemini,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.GEMINI)
    payload = {
        "model": "alpha/one",
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "stream": stream,
        "future_wire_field": {"preserved": True},
    }
    envelope = RequestEnvelope(
        ingress,
        payload,
        path="/v1beta/models/alpha~1one:generateContent",
        query={"alt": "sse"},
        headers={"x-protocol-option": "kept-as-context"},
    )
    dispatcher = GenerationDispatcher({WireProtocol.GEMINI: _Runtime(WireProtocol.GEMINI)})

    if stream:
        opened = await dispatcher.dispatch_stream(router, envelope)
        assert await anext(opened.frames) == terminal
        await close_async_iterator(opened.frames)
    else:
        result = await dispatcher.dispatch(router, envelope)
        assert result.value == terminal

    assert len(executor.attempts) == 1
    assert executor.attempts[0].payload["future_wire_field"] == {"preserved": True}
    assert executor.attempts[0].model == ModelRef("alpha", "one")
    assert len(dialect.request_contexts) == 1
    assert dialect.request_contexts[0].path == "/v1beta/models/alpha~1one:generateContent"
    assert dict(dialect.request_contexts[0].query) == {"alt": "sse"}
    assert dict(dialect.request_contexts[0].headers) == {"x-protocol-option": "kept-as-context"}
    assert envelope.materialization_count == 0
    assert ingress.decode_calls == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_explicit_feature_false_filters_model_before_provider_io(stream: bool) -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider(
        "alpha",
        "one",
        (chat,),
        model_info=ModelInfo(
            id="one",
            name="one",
            provider="alpha",
            feature_capabilities={Feature.TOOLS.value: False},
        ),
    )
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT, tools=True)
    envelope = RequestEnvelope(
        ingress,
        {
            "model": "alpha/one",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "stream": stream,
        },
    )
    execution = _Execution()
    dispatcher = GenerationDispatcher({}, execution=execution)

    with pytest.raises(ProviderError) as raised:
        if stream:
            await dispatcher.dispatch_stream(router, envelope)
        else:
            await dispatcher.dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert execution.calls == []
    assert execution.stream_calls == []
    assert envelope.materialization_count == 0
    assert ingress.decode_calls == 0


@pytest.mark.asyncio
async def test_cross_protocol_transports_share_one_ir_and_ignore_model_retry_budget() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (responses, chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    responses_runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    chat_runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution(
        actions={
            ("alpha", "responses"): _retryable("responses unavailable"),
            ("alpha", "chat"): "chat-ok",
        }
    )

    result = await GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: responses_runtime,
            WireProtocol.OPENAI_CHAT: chat_runtime,
        },
        execution=execution,
    ).dispatch(router, envelope)

    assert result.value == "chat-ok"
    assert [(provider_name, binding) for provider_name, binding, _ in execution.calls] == [
        ("alpha", "responses"),
        ("alpha", "chat"),
    ]
    assert envelope.materialization_count == 1
    assert ingress.decode_calls == 1
    assert responses_runtime.encode_calls == 1
    assert chat_runtime.encode_calls == 1
    assert responses_runtime.encoded_requests[0].model == "one"
    assert chat_runtime.encoded_requests[0].model == "one"
    assert (await envelope.semantic_ir()).model == "alpha/one"


@pytest.mark.asyncio
async def test_dispatch_exhausts_primary_transports_before_model_fallback() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    primary = _Provider("primary", "one", (responses, chat))
    fallback = _Provider("fallback", "two", (responses, chat))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    responses_runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    chat_runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(ingress, {"model": "router-maestro", "messages": []})
    execution = _Execution(
        actions={
            ("primary", "responses"): _retryable("primary responses failed"),
            ("primary", "chat"): _retryable("primary chat failed"),
            ("fallback", "responses"): "fallback-ok",
            ("fallback", "chat"): "unused",
        }
    )

    result = await GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: responses_runtime,
            WireProtocol.OPENAI_CHAT: chat_runtime,
        },
        execution=execution,
    ).dispatch(router, envelope)

    assert result.value == "fallback-ok"
    assert [(provider_name, binding) for provider_name, binding, _ in execution.calls] == [
        ("primary", "responses"),
        ("primary", "chat"),
        ("fallback", "responses"),
    ]
    assert [request.model for request in responses_runtime.encoded_requests] == ["one", "two"]
    assert [request.model for request in chat_runtime.encoded_requests] == ["one"]
    assert (await envelope.semantic_ir()).model == "router-maestro"


@pytest.mark.asyncio
async def test_malformed_raw_response_falls_back_before_model_commit() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    primary = _Provider("primary", "one", (chat,))
    fallback = _Provider("fallback", "two", (chat,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(
        ingress,
        {"model": "router-maestro", "messages": []},
    )
    valid = {"model": "two", "choices": []}
    execution = _Execution(
        actions={
            ("primary", "chat"): {"model": "one", "unexpected": "shape"},
            ("fallback", "chat"): valid,
        }
    )

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.value == valid
    assert result.selection.plan.provider is fallback
    assert [(provider, binding) for provider, binding, _ in execution.calls] == [
        ("primary", "chat"),
        ("fallback", "chat"),
    ]
    assert envelope.materialization_count == 0
    assert ingress.decode_calls == 0


@pytest.mark.asyncio
async def test_malformed_cross_response_is_fully_decoded_before_transport_commit() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (responses, chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    malformed = {
        "id": "resp_bad",
        "model": "one",
        "status": "completed",
        "output": [{"type": "future_internal_item"}],
    }
    execution = _Execution(
        actions={
            ("alpha", "responses"): malformed,
            ("alpha", "chat"): "chat-ok",
        }
    )

    result = await GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: OpenAIResponsesRuntime(
                provider_name="alpha",
                binding_id="responses",
            ),
            WireProtocol.OPENAI_CHAT: _Runtime(WireProtocol.OPENAI_CHAT),
        },
        execution=execution,
    ).dispatch(router, envelope)

    assert result.value == "chat-ok"
    assert result.selection.plan.binding.id == "chat"
    assert [(provider_name, binding) for provider_name, binding, _ in execution.calls] == [
        ("alpha", "responses"),
        ("alpha", "chat"),
    ]
    assert envelope.materialization_count == 1


@pytest.mark.asyncio
async def test_valid_cross_mapping_is_cached_as_semantic_response() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (responses,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution(
        actions={
            ("alpha", "responses"): {
                "id": "resp_1",
                "model": "one",
                "status": "completed",
                "output": [],
            }
        }
    )

    result = await GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: OpenAIResponsesRuntime(
                provider_name="alpha",
                binding_id="responses",
            )
        },
        execution=execution,
    ).dispatch(router, envelope)

    assert isinstance(result.value, SemanticResponse)
    assert result.value.id == "resp_1"
    assert result.value.model == "one"
    assert result.selection.plan.binding.id == "responses"


@pytest.mark.asyncio
async def test_dispatch_records_safe_attempt_audit_and_low_cardinality_observations() -> None:
    static_responses = legacy_endpoint_binding(
        binding_id="responses-stream-only",
        protocol=WireProtocol.OPENAI_RESPONSES,
        operations=frozenset({Operation.RESPONSES_STREAM}),
    )
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (static_responses, responses, chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution(
        actions={
            ("alpha", "responses"): _retryable("responses unavailable"),
            ("alpha", "chat"): "ok",
        }
    )
    audit = _AuditSpy()
    observations: list[DispatchAttemptObservation] = []
    context = cast(RequestContext, type("Context", (), {"audit": audit})())
    token = request_context_module._current_request_context.set(context)  # type: ignore[attr-defined]
    try:
        result = await GenerationDispatcher(
            {
                WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES),
                WireProtocol.OPENAI_CHAT: _Runtime(WireProtocol.OPENAI_CHAT),
            },
            execution=execution,
            attempt_observer=observations.append,
        ).dispatch(router, envelope)
    finally:
        request_context_module._current_request_context.reset(token)  # type: ignore[attr-defined]

    assert result.value == "ok"
    assert audit.attempts == [
        {
            "provider": "alpha",
            "model": "one",
            "binding": "responses-stream-only",
            "entry_protocol": "anthropic_messages",
            "upstream_transport": "openai_responses",
            "conversion_mode": "semantic_ir",
            "outcome": "unsupported",
            "ir_materialized": False,
        },
        {
            "provider": "alpha",
            "model": "one",
            "binding": "responses",
            "entry_protocol": "anthropic_messages",
            "upstream_transport": "openai_responses",
            "conversion_mode": "semantic_ir",
            "outcome": "retryable_failure",
            "ir_materialized": True,
        },
        {
            "provider": "alpha",
            "model": "one",
            "binding": "chat",
            "entry_protocol": "anthropic_messages",
            "upstream_transport": "openai_chat",
            "conversion_mode": "semantic_ir",
            "outcome": "selected",
            "ir_materialized": False,
        },
    ]
    assert observations == [
        DispatchAttemptObservation(
            entry_protocol=WireProtocol.ANTHROPIC_MESSAGES,
            upstream_transport=WireProtocol.OPENAI_RESPONSES,
            conversion_mode=ConversionMode.SEMANTIC_IR,
            outcome=DispatchAttemptOutcome.UNSUPPORTED,
            ir_materialized=False,
        ),
        DispatchAttemptObservation(
            entry_protocol=WireProtocol.ANTHROPIC_MESSAGES,
            upstream_transport=WireProtocol.OPENAI_RESPONSES,
            conversion_mode=ConversionMode.SEMANTIC_IR,
            outcome=DispatchAttemptOutcome.RETRYABLE_FAILURE,
            ir_materialized=True,
        ),
        DispatchAttemptObservation(
            entry_protocol=WireProtocol.ANTHROPIC_MESSAGES,
            upstream_transport=WireProtocol.OPENAI_CHAT,
            conversion_mode=ConversionMode.SEMANTIC_IR,
            outcome=DispatchAttemptOutcome.SELECTED,
            ir_materialized=False,
        ),
    ]
    assert all(
        not {"payload", "capsule", "reasoning_capsules"}.intersection(attempt)
        for attempt in audit.attempts
    )


@pytest.mark.asyncio
async def test_identity_fallback_remains_ir_free_after_cross_attempt_materializes_request() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    messages = _binding(WireProtocol.ANTHROPIC_MESSAGES, "messages")
    primary = _Provider("primary", "one", (responses,))
    fallback = _Provider("fallback", "two", (messages,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(
        ingress,
        {"model": "router-maestro", "messages": []},
    )
    observations: list[DispatchAttemptObservation] = []
    execution = _Execution(
        actions={
            ("primary", "responses"): _retryable("responses unavailable"),
            ("fallback", "messages"): "ok",
        }
    )

    result = await GenerationDispatcher(
        {WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES)},
        execution=execution,
        attempt_observer=observations.append,
    ).dispatch(router, envelope)

    assert result.value == "ok"
    assert envelope.materialization_count == 1
    assert [observation.conversion_mode for observation in observations] == [
        ConversionMode.SEMANTIC_IR,
        ConversionMode.IDENTITY,
    ]
    assert [observation.ir_materialized for observation in observations] == [True, False]


@pytest.mark.asyncio
async def test_no_representable_transport_returns_client_400() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (responses,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    representability = ProtocolRepresentabilityError(
        WireProtocol.OPENAI_RESPONSES,
        "tools[0]",
        "custom tools are unsupported",
    )
    execution = _Execution()
    observations: list[DispatchAttemptObservation] = []

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher(
            {
                WireProtocol.OPENAI_RESPONSES: _Runtime(
                    WireProtocol.OPENAI_RESPONSES,
                    encode_error=representability,
                )
            },
            execution=execution,
            attempt_observer=observations.append,
        ).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert raised.value.parameter == "tools[0]"
    assert raised.value.cause is representability
    assert execution.calls == []
    assert observations == [
        DispatchAttemptObservation(
            entry_protocol=WireProtocol.ANTHROPIC_MESSAGES,
            upstream_transport=WireProtocol.OPENAI_RESPONSES,
            conversion_mode=ConversionMode.SEMANTIC_IR,
            outcome=DispatchAttemptOutcome.UNREPRESENTABLE,
            ir_materialized=True,
        )
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_representability_error_takes_priority_over_static_unsupported(
    stream: bool,
) -> None:
    unsupported_operation = Operation.RESPONSES if stream else Operation.RESPONSES_STREAM
    unsupported = legacy_endpoint_binding(
        binding_id="responses-wrong-mode",
        protocol=WireProtocol.OPENAI_RESPONSES,
        operations=frozenset({unsupported_operation}),
    )
    unrepresentable = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    fallback_binding = _binding(WireProtocol.OPENAI_CHAT, "chat")
    primary = _Provider("primary", "one", (unsupported, unrepresentable))
    fallback = _Provider("fallback", "two", (fallback_binding,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(
        ingress,
        {"model": "router-maestro", "messages": [], "stream": stream},
    )
    representability = ProtocolRepresentabilityError(
        WireProtocol.OPENAI_RESPONSES,
        "tools[0].custom",
        "custom tools are unsupported",
    )
    fallback_stream = _Stream(["must-not-be-used"])
    execution = _Execution(
        actions={("fallback", "chat"): "must-not-be-used"},
        streams={("fallback", "chat"): fallback_stream},
    )
    dispatcher = GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: _Runtime(
                WireProtocol.OPENAI_RESPONSES,
                encode_error=representability,
            )
        },
        execution=execution,
    )

    with pytest.raises(ProviderError) as raised:
        if stream:
            await dispatcher.dispatch_stream(router, envelope)
        else:
            await dispatcher.dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert raised.value.parameter == "tools[0].custom"
    assert raised.value.cause is representability
    assert execution.calls == []
    assert execution.stream_calls == []
    assert fallback_stream.closed is False


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_only_static_unsupported_transport_returns_501(stream: bool) -> None:
    unsupported_operation = Operation.RESPONSES if stream else Operation.RESPONSES_STREAM
    unsupported = legacy_endpoint_binding(
        binding_id="responses-wrong-mode",
        protocol=WireProtocol.OPENAI_RESPONSES,
        operations=frozenset({unsupported_operation}),
    )
    provider = _Provider("alpha", "one", (unsupported,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(
        ingress,
        {"model": "alpha/one", "messages": [], "stream": stream},
    )
    dispatcher = GenerationDispatcher({}, execution=_Execution())

    with pytest.raises(ProviderError) as raised:
        if stream:
            await dispatcher.dispatch_stream(router, envelope)
        else:
            await dispatcher.dispatch(router, envelope)

    assert raised.value.status_code == 501
    assert raised.value.kind is ProviderFailureKind.UNSUPPORTED_OPERATION


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_static_rejection_does_not_trigger_model_fallback(stream: bool) -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    primary = _Provider("primary", "one", (responses,))
    fallback = _Provider("fallback", "two", (chat,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(
        ingress,
        {"model": "router-maestro", "messages": [], "stream": stream},
    )
    representability = ProtocolRepresentabilityError(
        WireProtocol.OPENAI_RESPONSES,
        "tools[0]",
        "custom tools are unsupported",
    )
    fallback_stream = _Stream(["must-not-be-used"])
    execution = _Execution(
        actions={("fallback", "chat"): "must-not-be-used"},
        streams={("fallback", "chat"): fallback_stream},
    )
    dispatcher = GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: _Runtime(
                WireProtocol.OPENAI_RESPONSES,
                encode_error=representability,
            )
        },
        execution=execution,
    )

    with pytest.raises(ProviderError) as raised:
        if stream:
            await dispatcher.dispatch_stream(router, envelope)
        else:
            await dispatcher.dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.cause is representability
    assert execution.calls == []
    assert execution.stream_calls == []
    assert fallback_stream.closed is False


@pytest.mark.asyncio
async def test_later_representability_error_does_not_override_first_upstream_failure() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    primary = _Provider("primary", "one", (chat,))
    fallback = _Provider("fallback", "two", (responses,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(
        ingress,
        {"model": "router-maestro", "messages": []},
    )
    primary_failure = _retryable("primary failed", status_code=503)
    execution = _Execution(actions={("primary", "chat"): primary_failure})

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher(
            {
                WireProtocol.OPENAI_RESPONSES: _Runtime(
                    WireProtocol.OPENAI_RESPONSES,
                    encode_error=ProtocolRepresentabilityError(
                        WireProtocol.OPENAI_RESPONSES,
                        "input",
                        "not representable",
                    ),
                )
            },
            execution=execution,
        ).dispatch(router, envelope)

    assert raised.value is primary_failure
    assert raised.value.status_code == 503
    assert [(provider_name, binding) for provider_name, binding, _ in execution.calls] == [
        ("primary", "chat")
    ]


@pytest.mark.asyncio
async def test_stream_later_representability_error_preserves_first_upstream_failure() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    primary = _Provider("primary", "one", (chat,))
    fallback = _Provider("fallback", "two", (responses,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(
        ingress,
        {"model": "router-maestro", "messages": [], "stream": True},
    )
    primary_failure = _retryable("primary stream failed", status_code=503)
    execution = _Execution(streams={("primary", "chat"): primary_failure})

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher(
            {
                WireProtocol.OPENAI_RESPONSES: _Runtime(
                    WireProtocol.OPENAI_RESPONSES,
                    encode_error=ProtocolRepresentabilityError(
                        WireProtocol.OPENAI_RESPONSES,
                        "input",
                        "not representable",
                    ),
                )
            },
            execution=execution,
        ).dispatch_stream(router, envelope)

    assert raised.value is primary_failure
    assert raised.value.status_code == 503
    assert [(provider_name, binding) for provider_name, binding, _ in execution.stream_calls] == [
        ("primary", "chat")
    ]
    assert execution.calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_anthropic_active_context_edit_does_not_override_native_failure(
    stream: bool,
) -> None:
    messages = _binding(WireProtocol.ANTHROPIC_MESSAGES, "messages")
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (messages, responses))
    router = _router((provider,), max_retries=0)
    envelope = RequestEnvelope(
        AnthropicMessagesRuntime(),
        {
            "model": "alpha/one",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "context_management": {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": 1},
                    }
                ]
            },
        },
    )
    primary_failure = _retryable("native messages failed", status_code=503)
    execution = _Execution(
        actions={("alpha", "messages"): primary_failure},
        streams={("alpha", "messages"): primary_failure},
    )
    observations: list[DispatchAttemptObservation] = []
    dispatcher = GenerationDispatcher(
        {WireProtocol.OPENAI_RESPONSES: OpenAIResponsesRuntime()},
        execution=execution,
        attempt_observer=observations.append,
    )

    with pytest.raises(ProviderError) as raised:
        if stream:
            await dispatcher.dispatch_stream(router, envelope)
        else:
            await dispatcher.dispatch(router, envelope)

    assert raised.value is primary_failure
    assert envelope.materialization_count == 1
    calls = execution.stream_calls if stream else execution.calls
    assert [(provider_name, binding) for provider_name, binding, _ in calls] == [
        ("alpha", "messages")
    ]
    assert [observation.outcome for observation in observations] == [
        DispatchAttemptOutcome.RETRYABLE_FAILURE,
        DispatchAttemptOutcome.UNREPRESENTABLE,
    ]


@pytest.mark.asyncio
async def test_provider_unsupported_operation_tries_next_transport_of_same_model() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (responses, chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    unsupported = ProviderError(
        "model rejects responses",
        status_code=501,
        retryable=False,
        kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
    )
    execution = _Execution(
        actions={
            ("alpha", "responses"): unsupported,
            ("alpha", "chat"): "ok",
        }
    )

    result = await GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES),
            WireProtocol.OPENAI_CHAT: _Runtime(WireProtocol.OPENAI_CHAT),
        },
        execution=execution,
    ).dispatch(router, envelope)

    assert result.value == "ok"
    assert [(provider_name, binding) for provider_name, binding, _ in execution.calls] == [
        ("alpha", "responses"),
        ("alpha", "chat"),
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_request_option_rejection_tries_next_transport_of_same_model(
    stream: bool,
) -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (responses, chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(
        ingress,
        {"model": "alpha/one", "messages": [], "stream": stream},
    )
    rejection = RequestOptionError(
        "Responses does not support request option 'temperature'",
        parameter="temperature",
        provider="alpha",
        model="one",
    )
    observations: list[DispatchAttemptObservation] = []
    chat_stream = _Stream(["chat-ok"])
    execution = _Execution(
        actions={
            ("alpha", "responses"): rejection,
            ("alpha", "chat"): "chat-ok",
        },
        streams={
            ("alpha", "responses"): rejection,
            ("alpha", "chat"): chat_stream,
        },
    )
    dispatcher = GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES),
            WireProtocol.OPENAI_CHAT: _Runtime(WireProtocol.OPENAI_CHAT),
        },
        execution=execution,
        attempt_observer=observations.append,
    )

    if stream:
        opened = await dispatcher.dispatch_stream(router, envelope)
        assert await anext(opened.frames) == "chat-ok"
        assert opened.selection.plan.binding.id == "chat"
        await close_async_iterator(opened.frames)
        calls = execution.stream_calls
    else:
        result = await dispatcher.dispatch(router, envelope)
        assert result.value == "chat-ok"
        assert result.selection.plan.binding.id == "chat"
        calls = execution.calls

    assert [(provider_name, binding) for provider_name, binding, _ in calls] == [
        ("alpha", "responses"),
        ("alpha", "chat"),
    ]
    assert [observation.outcome for observation in observations] == [
        DispatchAttemptOutcome.UNREPRESENTABLE,
        DispatchAttemptOutcome.SELECTED,
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_final_option_rejection_uses_ingress_parameter_path(stream: bool) -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (responses, chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(
        ingress,
        {
            "model": "alpha/one",
            "messages": [],
            "stream": stream,
            "stop_sequences": ["END"],
        },
    )
    representability = ProtocolRepresentabilityError(
        WireProtocol.OPENAI_RESPONSES,
        "stop_sequences",
        "Responses does not support stop sequences",
    )
    rejection = RequestOptionError(
        "Chat does not support request option 'stop'",
        parameter="stop",
        provider="alpha",
        model="one",
    )
    execution = _Execution(
        actions={("alpha", "chat"): rejection},
        streams={("alpha", "chat"): rejection},
    )
    dispatcher = GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: _Runtime(
                WireProtocol.OPENAI_RESPONSES,
                encode_error=representability,
            ),
            WireProtocol.OPENAI_CHAT: _Runtime(WireProtocol.OPENAI_CHAT),
        },
        execution=execution,
    )

    with pytest.raises(RequestOptionError) as raised:
        if stream:
            await dispatcher.dispatch_stream(router, envelope)
        else:
            await dispatcher.dispatch(router, envelope)

    assert raised.value.parameter == "stop_sequences"
    assert "stop_sequences" in raised.value.safe_message
    assert raised.value.cause is rejection


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_request_option_rejection_does_not_trigger_model_fallback(
    stream: bool,
) -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    primary = _Provider("primary", "one", (responses,))
    fallback = _Provider("fallback", "two", (chat,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(
        ingress,
        {"model": "router-maestro", "messages": [], "stream": stream},
    )
    rejection = RequestOptionError(
        "Responses does not support request option 'temperature'",
        parameter="temperature",
        provider="primary",
        model="one",
    )
    execution = _Execution(
        actions={("primary", "responses"): rejection},
        streams={("primary", "responses"): rejection},
    )
    dispatcher = GenerationDispatcher(
        {WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES)},
        execution=execution,
    )

    with pytest.raises(RequestOptionError) as raised:
        if stream:
            await dispatcher.dispatch_stream(router, envelope)
        else:
            await dispatcher.dispatch(router, envelope)

    assert raised.value is rejection
    assert raised.value.parameter == "temperature"
    calls = [(provider_name, binding) for provider_name, binding, _ in execution.calls]
    stream_calls = [
        (provider_name, binding) for provider_name, binding, _ in execution.stream_calls
    ]
    assert calls == ([] if stream else [("primary", "responses")])
    assert stream_calls == ([("primary", "responses")] if stream else [])


@pytest.mark.asyncio
async def test_nonfallback_provider_error_records_closed_failed_outcome() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (chat,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    fatal = ProviderError(
        "authentication failed",
        status_code=401,
        retryable=False,
        kind=ProviderFailureKind.AUTHENTICATION,
    )
    execution = _Execution(actions={("alpha", "chat"): fatal})
    observations: list[DispatchAttemptObservation] = []

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher(
            {},
            execution=execution,
            attempt_observer=observations.append,
        ).dispatch(router, envelope)

    assert raised.value is fatal
    assert observations == [
        DispatchAttemptObservation(
            entry_protocol=WireProtocol.OPENAI_CHAT,
            upstream_transport=WireProtocol.OPENAI_CHAT,
            conversion_mode=ConversionMode.IDENTITY,
            outcome=DispatchAttemptOutcome.FAILED,
            ir_materialized=False,
        )
    ]


@pytest.mark.asyncio
async def test_attempt_observer_failure_never_changes_dispatch_result() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (chat,))
    router = _router((provider,), max_retries=0)
    envelope = RequestEnvelope(
        _Runtime(WireProtocol.OPENAI_CHAT),
        {"model": "alpha/one", "messages": []},
    )

    def broken_observer(_observation: DispatchAttemptObservation) -> None:
        raise RuntimeError("metrics collector unavailable")

    result = await GenerationDispatcher(
        {},
        execution=_Execution(actions={("alpha", "chat"): "ok"}),
        attempt_observer=broken_observer,
    ).dispatch(router, envelope)

    assert result.value == "ok"


@pytest.mark.asyncio
async def test_stream_failure_before_first_frame_falls_back_and_closes_iterator() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (responses, chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(
        ingress,
        {"model": "alpha/one", "messages": [], "stream": True},
    )
    failed = _Stream([_retryable("failed before first frame")])
    selected = _Stream(["first-frame"])
    execution = _Execution(
        streams={
            ("alpha", "responses"): failed,
            ("alpha", "chat"): selected,
        }
    )
    observations: list[DispatchAttemptObservation] = []

    opened = await GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES),
            WireProtocol.OPENAI_CHAT: _Runtime(WireProtocol.OPENAI_CHAT),
        },
        execution=execution,
        attempt_observer=observations.append,
    ).dispatch_stream(router, envelope)

    assert failed.closed is True
    assert opened.selection.plan.binding.id == "chat"
    assert await anext(opened.frames) == "first-frame"
    with pytest.raises(StopAsyncIteration):
        await anext(opened.frames)
    assert selected.closed is True
    assert [observation.outcome for observation in observations] == [
        DispatchAttemptOutcome.RETRYABLE_FAILURE,
        DispatchAttemptOutcome.SELECTED,
    ]


@pytest.mark.asyncio
async def test_malformed_first_identity_frame_tries_next_transport_without_ir() -> None:
    first_chat = _binding(WireProtocol.OPENAI_CHAT, "chat-first")
    second_chat = _binding(WireProtocol.OPENAI_CHAT, "chat-second")
    provider = _Provider("alpha", "one", (first_chat, second_chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    target = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(
        ingress,
        {"model": "alpha/one", "messages": [], "stream": True},
    )
    malformed = _Stream([{"unexpected": "shape"}])
    first_frame = {
        "id": "chatcmpl_selected",
        "choices": [{"index": 0, "delta": {"content": "hello"}}],
    }
    selected = _Stream([first_frame])
    execution = _Execution(
        streams={
            ("alpha", "chat-first"): malformed,
            ("alpha", "chat-second"): selected,
        }
    )
    observations: list[DispatchAttemptObservation] = []

    opened = await GenerationDispatcher(
        {WireProtocol.OPENAI_CHAT: target},
        execution=execution,
        attempt_observer=observations.append,
    ).dispatch_stream(router, envelope)

    assert malformed.closed is True
    assert opened.selection.plan.binding.id == "chat-second"
    assert await anext(opened.frames) is first_frame
    with pytest.raises(StopAsyncIteration):
        await anext(opened.frames)
    assert selected.closed is True
    assert envelope.materialization_count == 0
    assert ingress.decode_calls == 0
    assert target.decode_calls == 0
    assert [observation.outcome for observation in observations] == [
        DispatchAttemptOutcome.RETRYABLE_FAILURE,
        DispatchAttemptOutcome.SELECTED,
    ]


@pytest.mark.asyncio
async def test_semantically_malformed_first_cross_protocol_frame_tries_next_transport() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (responses, chat))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(
        ingress,
        {"model": "alpha/one", "messages": [], "stream": True},
    )
    malformed = _Stream(
        [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_malformed",
                    "model": "one",
                    "status": "completed",
                    # The dispatcher's shallow guard accepts the response envelope,
                    # but the stateful Responses decoder must reject this field.
                    "output": "not-an-array",
                },
            }
        ]
    )
    first_frame = {
        "id": "chatcmpl_selected",
        "model": "one",
        "choices": [{"index": 0, "delta": {"content": "hello"}}],
    }
    selected = _Stream([first_frame])
    execution = _Execution(
        streams={
            ("alpha", "responses"): malformed,
            ("alpha", "chat"): selected,
        }
    )

    observations: list[DispatchAttemptObservation] = []
    opened = await GenerationDispatcher(
        {
            WireProtocol.OPENAI_RESPONSES: OpenAIResponsesRuntime(
                provider_name="alpha",
                binding_id="responses",
            ),
            WireProtocol.OPENAI_CHAT: OpenAIChatRuntime(
                origin_provider="alpha",
                default_model="one",
            ),
        },
        execution=execution,
        attempt_observer=observations.append,
    ).dispatch_stream(router, envelope)

    assert malformed.closed is True
    assert opened.selection.plan.binding.id == "chat"
    assert opened.semantic_decoder is not None
    assert opened.first_events is not None
    assert await anext(opened.frames) is first_frame
    await close_async_iterator(opened.frames)
    assert selected.closed is True
    assert envelope.materialization_count == 1
    assert ingress.decode_calls == 1
    assert [observation.outcome for observation in observations] == [
        DispatchAttemptOutcome.RETRYABLE_FAILURE,
        DispatchAttemptOutcome.SELECTED,
    ]


@pytest.mark.parametrize(
    ("protocol", "malformed", "first_frame"),
    [
        (
            WireProtocol.ANTHROPIC_MESSAGES,
            {"type": "message_delta", "delta": {}},
            {
                "type": "message_start",
                "message": {
                    "id": "msg_selected",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "one",
                },
            },
        ),
        (
            WireProtocol.ANTHROPIC_MESSAGES,
            {"type": "ping"},
            {
                "type": "message_start",
                "message": {
                    "id": "msg_selected",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "one",
                },
            },
        ),
        (
            WireProtocol.OPENAI_RESPONSES,
            {"type": "bogus"},
            {
                "type": "response.created",
                "response": {
                    "id": "resp_selected",
                    "model": "one",
                    "status": "in_progress",
                    "output": [],
                },
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_out_of_order_or_unknown_first_frame_tries_next_transport(
    protocol: WireProtocol,
    malformed: Mapping[str, Any],
    first_frame: Mapping[str, Any],
) -> None:
    first_binding = _binding(protocol, "first")
    second_binding = _binding(protocol, "second")
    provider = _Provider("alpha", "one", (first_binding, second_binding))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(protocol)
    target = _Runtime(protocol)
    envelope = RequestEnvelope(
        ingress,
        {"model": "alpha/one", "messages": [], "stream": True},
    )
    rejected = _Stream([malformed])
    selected = _Stream([first_frame])
    execution = _Execution(
        streams={
            ("alpha", "first"): rejected,
            ("alpha", "second"): selected,
        }
    )

    opened = await GenerationDispatcher(
        {protocol: target},
        execution=execution,
    ).dispatch_stream(router, envelope)

    assert rejected.closed is True
    assert opened.selection.plan.binding.id == "second"
    assert await anext(opened.frames) is first_frame
    with pytest.raises(StopAsyncIteration):
        await anext(opened.frames)
    assert selected.closed is True
    assert envelope.materialization_count == 0
    assert ingress.decode_calls == 0
    assert target.decode_calls == 0


@pytest.mark.asyncio
async def test_stream_first_frame_locks_selection_and_never_replays_fallback() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    primary = _Provider("primary", "one", (chat,))
    fallback = _Provider("fallback", "two", (chat,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(
        ingress,
        {"model": "router-maestro", "messages": [], "stream": True},
    )
    postcommit_failure = _retryable("failed after first frame")
    selected = _Stream(["first-frame", postcommit_failure])
    unused = _Stream(["fallback-frame"])
    execution = _Execution(
        streams={
            ("primary", "chat"): selected,
            ("fallback", "chat"): unused,
        }
    )

    opened = await GenerationDispatcher({}, execution=execution).dispatch_stream(router, envelope)

    assert opened.selection.plan.provider is primary
    assert await anext(opened.frames) == "first-frame"
    with pytest.raises(ProviderError) as raised:
        await anext(opened.frames)
    assert raised.value is postcommit_failure
    assert selected.closed is True
    assert unused.closed is False
    assert [(provider_name, binding) for provider_name, binding, _ in execution.stream_calls] == [
        ("primary", "chat")
    ]


@pytest.mark.asyncio
async def test_reasoning_capsule_without_codec_fails_closed_before_provider_io() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (responses,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(
        WireProtocol.ANTHROPIC_MESSAGES,
        reasoning_capsules=("rmr1.unavailable.invalid",),
    )
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution()

    with pytest.raises(ProviderError, match="^Invalid reasoning capsule$") as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert provider.list_models_calls == 0
    assert execution.calls == []
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_tampered_reasoning_capsule_fails_closed_before_provider_io() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (responses,))
    router = _router((provider,), max_retries=0)
    codec = ReasoningCapsuleCodec(bytes([1]) * 32)
    capsule = _capsule(
        codec,
        provider="alpha",
        model="one",
        transport="responses",
    )
    tampered = f"{capsule[:-1]}{'A' if capsule[-1] != 'A' else 'B'}"
    ingress = _Runtime(
        WireProtocol.ANTHROPIC_MESSAGES,
        reasoning_capsules=(tampered,),
    )
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution()

    with pytest.raises(ProviderError, match="^Invalid reasoning capsule$") as raised:
        await GenerationDispatcher(
            {},
            execution=execution,
            reasoning_capsule_codec=codec,
        ).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert provider.list_models_calls == 0
    assert execution.calls == []
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_reasoning_capsules_must_share_one_affinity_before_provider_io() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (responses, chat))
    router = _router((provider,), max_retries=0)
    codec = ReasoningCapsuleCodec(bytes([2]) * 32)
    capsules = (
        _capsule(codec, provider="alpha", model="one", transport="responses"),
        _capsule(codec, provider="alpha", model="one", transport="chat", item_id="rs_2"),
    )
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES, reasoning_capsules=capsules)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution()

    with pytest.raises(ProviderError, match="^Invalid reasoning capsule$") as raised:
        await GenerationDispatcher(
            {},
            execution=execution,
            reasoning_capsule_codec=codec,
        ).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert provider.list_models_calls == 0
    assert execution.calls == []
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_reasoning_capsules_freeze_exact_plan_and_are_rechecked_for_execution() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    primary = _Provider("primary", "one", (responses,))
    middle = _Provider("middle", "two", (responses,))
    pinned = _Provider("pinned", "three", (responses,))
    # The capsule affinity is outside the ordinary primary + one-fallback
    # window. Continuation provenance must pin it directly, not depend on the
    # configured recovery budget.
    router = _router((primary, middle, pinned), max_retries=1)
    codec = _TrackingCodec(bytes([3]) * 32)
    capsules = (
        _capsule(codec, provider="pinned", model="three", transport="responses"),
        _capsule(
            codec,
            provider="pinned",
            model="three",
            transport="responses",
            item_id="rs_2",
        ),
    )
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES, reasoning_capsules=capsules)
    responses_runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    envelope = RequestEnvelope(ingress, {"model": "router-maestro", "messages": []})
    execution = _Execution(actions={("pinned", "responses"): "ok"})

    result = await GenerationDispatcher(
        {WireProtocol.OPENAI_RESPONSES: responses_runtime},
        execution=execution,
        reasoning_capsule_codec=codec,
    ).dispatch(router, envelope)

    assert result.value == "ok"
    assert result.selection.plan.provider is pinned
    assert [(provider_name, binding) for provider_name, binding, _ in execution.calls] == [
        ("pinned", "responses")
    ]
    assert execution.calls[0][2]["model"] == "three"
    assert codec.execution_calls == [
        (capsule, "pinned", "three", "responses") for capsule in capsules
    ]
    assert codec.routing_calls == [*capsules, *capsules]
    assert (await envelope.semantic_ir()).model == "router-maestro"


@pytest.mark.asyncio
async def test_reasoning_capsule_with_unavailable_affinity_never_reaches_execution() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (responses,))
    router = _router((provider,), max_retries=0)
    codec = ReasoningCapsuleCodec(bytes([4]) * 32)
    capsule = _capsule(
        codec,
        provider="alpha",
        model="one",
        transport="missing-binding",
    )
    ingress = _Runtime(
        WireProtocol.ANTHROPIC_MESSAGES,
        reasoning_capsules=(capsule,),
    )
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution()

    with pytest.raises(ProviderError, match="^Invalid reasoning capsule$") as raised:
        await GenerationDispatcher(
            {},
            execution=execution,
            reasoning_capsule_codec=codec,
        ).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert execution.calls == []
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_opaque_continuation_never_uses_openai_chat_transport() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (chat, responses))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(
        WireProtocol.OPENAI_CHAT,
        opaque_continuation=True,
    )
    responses_runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution(actions={("alpha", "responses"): "ok"})

    result = await GenerationDispatcher(
        {WireProtocol.OPENAI_RESPONSES: responses_runtime},
        execution=execution,
    ).dispatch(router, envelope)

    assert result.value == "ok"
    assert result.selection.plan.binding.id == "responses"
    assert [(provider_name, binding) for provider_name, binding, _ in execution.calls] == [
        ("alpha", "responses")
    ]
    assert execution.calls[0][2]["model"] == "one"


@pytest.mark.asyncio
async def test_reasoning_capsule_can_pin_its_originating_chat_transport() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    provider = _Provider("alpha", "one", (chat,))
    router = _router((provider,), max_retries=0)
    codec = ReasoningCapsuleCodec(bytes([8]) * 32)
    capsule = _capsule(
        codec,
        provider="alpha",
        model="one",
        transport="chat",
    )
    ingress = _Runtime(
        WireProtocol.ANTHROPIC_MESSAGES,
        reasoning_capsules=(capsule,),
        opaque_continuation=True,
    )
    chat_runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution(actions={("alpha", "chat"): "ok"})

    result = await GenerationDispatcher(
        {WireProtocol.OPENAI_CHAT: chat_runtime},
        execution=execution,
        reasoning_capsule_codec=codec,
    ).dispatch(router, envelope)

    assert result.value == "ok"
    assert result.selection.plan.binding.id == "chat"
    assert [(provider_name, binding) for provider_name, binding, _ in execution.calls] == [
        ("alpha", "chat")
    ]


@pytest.mark.asyncio
async def test_previous_response_id_requires_primary_responses_identity_transport() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT, "chat")
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("alpha", "one", (chat, responses))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(
        WireProtocol.OPENAI_CHAT,
        previous_response_id="resp_previous",
    )
    envelope = RequestEnvelope(ingress, {"model": "alpha/one", "messages": []})
    execution = _Execution(actions={})

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher(
            {WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES)},
            execution=execution,
        ).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert execution.calls == []
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_previous_response_id_does_not_fallback_after_primary_responses_failure() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    primary = _Provider("primary", "one", (responses,))
    fallback = _Provider("fallback", "two", (responses,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(
        WireProtocol.OPENAI_RESPONSES,
        previous_response_id="resp_previous",
    )
    envelope = RequestEnvelope(ingress, {"model": "primary/one", "input": []})
    failure = _retryable("primary unavailable")
    execution = _Execution(actions={("primary", "responses"): failure})

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value is failure
    assert [(provider, binding) for provider, binding, _ in execution.calls] == [
        ("primary", "responses")
    ]
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_previous_response_id_rejects_auto_route_before_provider_io() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    primary = _Provider("primary", "one", (responses,))
    fallback = _Provider("fallback", "two", (responses,))
    router = _router((primary, fallback), max_retries=1)
    ingress = _Runtime(
        WireProtocol.OPENAI_RESPONSES,
        previous_response_id="resp_previous",
    )
    envelope = RequestEnvelope(ingress, {"model": "router-maestro", "input": []})
    execution = _Execution()

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.parameter == "previous_response_id"
    assert execution.calls == []
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_previous_response_id_rejects_unqualified_model_alias_before_provider_io() -> None:
    responses = _binding(WireProtocol.OPENAI_RESPONSES, "responses")
    provider = _Provider("primary", "one", (responses,))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(
        WireProtocol.OPENAI_RESPONSES,
        previous_response_id="resp_previous",
    )
    envelope = RequestEnvelope(ingress, {"model": "one", "input": []})
    execution = _Execution()

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.parameter == "previous_response_id"
    assert execution.calls == []
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_previous_response_id_rejects_ambiguous_native_binding_before_provider_io() -> None:
    first = _binding(WireProtocol.OPENAI_RESPONSES, "responses-primary")
    second = _binding(WireProtocol.OPENAI_RESPONSES, "responses-secondary")
    provider = _Provider("primary", "one", (first, second))
    router = _router((provider,), max_retries=0)
    ingress = _Runtime(
        WireProtocol.OPENAI_RESPONSES,
        previous_response_id="resp_previous",
    )
    envelope = RequestEnvelope(ingress, {"model": "primary/one", "input": []})
    execution = _Execution()

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.parameter == "previous_response_id"
    assert execution.calls == []
    assert envelope.materialization_count == 0
