"""Capability-aware route planning contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from router_maestro.config import PrioritiesConfig
from router_maestro.providers import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ModelInfo,
    ProviderError,
    RequestOptionError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
)
from router_maestro.providers import copilot as copilot_module
from router_maestro.providers.copilot import CopilotProvider, copilot_operation_capabilities
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    Feature,
    ModelCapabilities,
    Operation,
    ProviderCapabilities,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.utils.cache import TTLCache


class _CapabilityProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        models: list[ModelInfo],
        operations: frozenset[Operation],
        *,
        fail_retryable: bool = False,
    ) -> None:
        self._name = name
        self._models = models
        self._operations = operations
        self.fail_retryable = fail_retryable
        self.calls: list[tuple[Operation, str]] = []
        self.validations: list[tuple[Operation, str]] = []
        self.validation_requests: list[ChatRequest | ResponsesRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            operations=self._operations,
        )

    def is_authenticated(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return self._models

    def _record(self, operation: Operation, model: str) -> None:
        self.calls.append((operation, model))
        if self.fail_retryable:
            raise ProviderError("retryable", status_code=503, retryable=True)

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        self._record(Operation.CHAT, request.model)
        return ChatResponse(content="ok", model=request.model)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self._record(Operation.CHAT_STREAM, request.model)
        yield ChatStreamChunk(content="ok", finish_reason="stop")

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        self._record(Operation.RESPONSES, request.model)
        return ResponsesResponse(content="ok", model=request.model)

    async def responses_completion_stream(
        self, request: ResponsesRequest
    ) -> AsyncIterator[ResponsesStreamChunk]:
        self._record(Operation.RESPONSES_STREAM, request.model)
        yield ResponsesStreamChunk(content="ok", finish_reason="stop")

    def validate_chat_request(self, request: ChatRequest, *, stream: bool) -> None:
        operation = Operation.CHAT_STREAM if stream else Operation.CHAT
        self.validations.append((operation, request.model))
        self.validation_requests.append(request)

    def validate_responses_request(self, request: ResponsesRequest) -> None:
        operation = Operation.RESPONSES_STREAM if request.stream else Operation.RESPONSES
        self.validations.append((operation, request.model))
        self.validation_requests.append(request)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ({}, None),
        ({"supported_endpoints": []}, ()),
        ({"supported_endpoints": ("/responses", "ws:/responses")}, ("/responses", "ws:/responses")),
        ({"supported_endpoints": ["/future-endpoint"]}, ("/future-endpoint",)),
    ],
    ids=["missing", "empty", "tuple", "unknown-endpoint"],
)
def test_normalize_supported_endpoints_preserves_catalog_contract_state(model, expected) -> None:
    assert copilot_module.normalize_supported_endpoints(model) == expected


@pytest.mark.parametrize(
    "value",
    [None, "ws:/responses", {"endpoint": "/responses"}, ["/responses", 42]],
    ids=["null", "scalar", "object", "non-string-element"],
)
def test_normalize_supported_endpoints_rejects_malformed_present_values(value) -> None:
    with pytest.raises(TypeError):
        copilot_module.normalize_supported_endpoints({"supported_endpoints": value})


@pytest.mark.asyncio
async def test_endpoint_normalizer_is_shared_by_capabilities_and_catalog_materialization(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    def normalize(model):
        calls.append(model)
        return ("/responses",)

    monkeypatch.setattr(copilot_module, "normalize_supported_endpoints", normalize)
    catalog_model = {
        "id": "catalog-model",
        "supported_endpoints": None,
        "capabilities": {"type": "chat", "supports": {}},
    }

    direct = copilot_operation_capabilities(catalog_model)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [catalog_model]})

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    with patch(
        "httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        models = await provider.list_models(force_refresh=True)

    assert direct[Operation.RESPONSES] is True
    assert models[0].supported_endpoints == ("/responses",)
    assert models[0].operation_capabilities[Operation.RESPONSES] is True
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("supported_endpoints", "expected_responses"),
    [
        (["ws:/responses"], False),
        (["/responses"], True),
        (["/responses", "ws:/responses"], True),
    ],
    ids=[
        "websocket-only",
        "http-only",
        "http-and-websocket",
    ],
)
def test_copilot_http_responses_capability_requires_exact_http_endpoint(
    supported_endpoints,
    expected_responses,
) -> None:
    capabilities = copilot_operation_capabilities(
        {
            "id": "gpt-5.4",
            "supported_endpoints": supported_endpoints,
        }
    )

    assert capabilities[Operation.RESPONSES] is expected_responses
    assert capabilities[Operation.RESPONSES_STREAM] is expected_responses


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5.4", "github-copilot/gpt-5.4"],
)
def test_copilot_missing_endpoint_contract_keeps_responses_name_heuristic(model_id) -> None:
    capabilities = copilot_operation_capabilities({"id": model_id})

    assert capabilities[Operation.CHAT] is True
    assert capabilities[Operation.CHAT_STREAM] is True
    assert capabilities[Operation.RESPONSES] is True
    assert capabilities[Operation.RESPONSES_STREAM] is True


@pytest.mark.parametrize(
    "model_id",
    ["claude-sonnet-4.6", "github-copilot/claude-sonnet-4.6"],
)
def test_copilot_missing_endpoint_contract_normalizes_native_anthropic_heuristic(
    model_id,
) -> None:
    capabilities = copilot_operation_capabilities({"id": model_id})

    assert capabilities[Operation.NATIVE_ANTHROPIC] is True


def _model(
    provider: str,
    model_id: str,
    *,
    operations: dict[str, bool] | None = None,
    features: dict[str, bool] | None = None,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name=model_id,
        provider=provider,
        operation_capabilities=operations or {},
        feature_capabilities=features or {},
    )


def _router(
    providers: list[_CapabilityProvider],
    priorities: list[str],
    *,
    max_retries: int = 10,
    strategy: str = "priority",
) -> Router:
    router = Router.__new__(Router)
    router.providers = {provider.name: provider for provider in providers}
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._providers_ttl.set(True)
    for provider in providers:
        for model in provider._models:
            router._models_cache.setdefault(model.id, (provider.name, model))
            router._models_cache[f"{provider.name}/{model.id}"] = (provider.name, model)
    router._models_cache_ttl.set(True)
    router._priorities_cache.set(
        PrioritiesConfig(
            priorities=priorities,
            fallback={"strategy": strategy, "maxRetries": max_retries},
        )
    )
    return router


def test_minimal_domain_types_are_immutable_and_bind_model_identity():
    ref = ModelRef(provider="p", upstream_id="m")
    capabilities = ModelCapabilities(
        model=ref,
        operations={Operation.CHAT: CapabilitySupport.SUPPORTED},
        features={Feature.TOOLS: CapabilitySupport.UNKNOWN},
    )
    provider = type("Provider", (), {"name": "p"})()
    features = RequestFeatures(tools=True)
    candidate = RouteCandidate(
        model=ref,
        provider=provider,
        capabilities=capabilities,
        evaluated_operation=Operation.CHAT,
        evaluated_features=features,
        support=CapabilitySupport.UNKNOWN,
    )
    plan = RoutePlan(
        operation=Operation.CHAT,
        features=features,
        primary=candidate,
        fallbacks=(),
        explicit=True,
    )

    assert plan.primary.model.qualified_id == "p/m"
    assert plan.primary.capabilities.model is plan.primary.model
    assert plan.primary.capabilities.operation(Operation.CHAT) is CapabilitySupport.SUPPORTED
    assert plan.primary.capabilities.feature(Feature.TOOLS) is CapabilitySupport.UNKNOWN


def test_model_capabilities_defensively_freezes_operation_and_feature_maps():
    operation_source = {Operation.CHAT: CapabilitySupport.SUPPORTED}
    feature_source = {Feature.TOOLS: CapabilitySupport.UNKNOWN}
    ref = ModelRef("p", "m")
    capabilities = ModelCapabilities(
        model=ref,
        operations=operation_source,
        features=feature_source,
    )
    provider = type("Provider", (), {"name": "p"})()
    features = RequestFeatures()
    candidate = RouteCandidate(
        model=ref,
        provider=provider,
        capabilities=capabilities,
        evaluated_operation=Operation.CHAT,
        evaluated_features=features,
        support=capabilities.support_for(Operation.CHAT, features),
    )

    operation_source[Operation.CHAT] = CapabilitySupport.UNSUPPORTED
    feature_source[Feature.TOOLS] = CapabilitySupport.SUPPORTED

    assert candidate.capabilities.operation(Operation.CHAT) is CapabilitySupport.SUPPORTED
    assert candidate.capabilities.feature(Feature.TOOLS) is CapabilitySupport.UNKNOWN
    assert candidate.support is candidate.capabilities.support_for(
        Operation.CHAT,
        RequestFeatures(),
    )
    with pytest.raises(TypeError):
        capabilities.operations[Operation.CHAT] = CapabilitySupport.UNSUPPORTED
    with pytest.raises(TypeError):
        capabilities.features[Feature.TOOLS] = CapabilitySupport.SUPPORTED


@pytest.mark.asyncio
async def test_route_plan_freezes_catalog_reasoning_effort_values():
    provider = _CapabilityProvider(
        "p",
        [
            ModelInfo(
                id="m",
                name="m",
                provider="p",
                reasoning_effort_values=["medium", "high"],
                operation_capabilities={Operation.NATIVE_ANTHROPIC: True},
            )
        ],
        frozenset({Operation.NATIVE_ANTHROPIC}),
    )
    router = _router([provider], ["p/m"])

    plan = await router.plan_route(
        "p/m",
        Operation.NATIVE_ANTHROPIC,
        RequestFeatures(reasoning=True),
    )
    provider._models[0].reasoning_effort_values.append("low")

    assert plan.primary.capabilities.reasoning_effort_values == ("medium", "high")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({}, RequestFeatures()),
        ({"tools": []}, RequestFeatures()),
        ({"tools": [{"name": "tool"}]}, RequestFeatures(tools=True)),
        (
            {"thinking": {"type": "enabled"}},
            RequestFeatures(reasoning=True, reasoning_parameter="thinking"),
        ),
        (
            {"thinking": {"type": "adaptive"}},
            RequestFeatures(reasoning=True, reasoning_parameter="thinking"),
        ),
        ({"thinking": {"type": "disabled"}}, RequestFeatures()),
        (
            {"output_config": {"effort": "high"}},
            RequestFeatures(reasoning=True, reasoning_parameter="output_config.effort"),
        ),
        (
            {
                "thinking": {"type": "enabled"},
                "output_config": {"effort": "low"},
            },
            RequestFeatures(reasoning=True, reasoning_parameter="output_config.effort"),
        ),
        ({"output_config": {"effort": "invalid"}}, RequestFeatures()),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "abc",
                                },
                            }
                        ],
                    }
                ]
            },
            RequestFeatures(vision=True),
        ),
        (
            {"tools": [{"name": "a"}, {"name": "b"}]},
            RequestFeatures(tools=True, parallel_tools=False),
        ),
    ],
    ids=[
        "empty",
        "empty-tools",
        "tools",
        "thinking-enabled",
        "thinking-adaptive",
        "thinking-disabled",
        "output-effort",
        "effort-precedence",
        "invalid-effort-not-feature",
        "vision",
        "no-parallel-toggle",
    ],
)
def test_anthropic_native_request_features_are_derived_once_from_wire_body(body, expected):
    assert RequestFeatures.for_anthropic_native(body) == expected


def test_base_provider_declares_only_implemented_chat_transports():
    class _MinimalProvider(BaseProvider):
        async def chat_completion(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(content="ok", model=request.model)

        async def chat_completion_stream(
            self, request: ChatRequest
        ) -> AsyncIterator[ChatStreamChunk]:
            yield ChatStreamChunk(content="ok")

        async def list_models(self) -> list[ModelInfo]:
            return []

        def is_authenticated(self) -> bool:
            return True

    provider = _MinimalProvider()

    assert provider.capabilities.supports(Operation.CHAT)
    assert provider.capabilities.supports(Operation.CHAT_STREAM)
    assert not provider.capabilities.supports(Operation.RESPONSES)
    assert not provider.capabilities.supports(Operation.RESPONSES_STREAM)
    assert not provider.capabilities.supports(Operation.NATIVE_ANTHROPIC)


def test_copilot_declares_all_current_completion_transports():
    operations = CopilotProvider().capabilities.operations

    assert operations == frozenset(Operation)


@pytest.mark.asyncio
async def test_copilot_catalog_preserves_explicit_capabilities_and_unknown_keys():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "gpt-5.5",
                        "name": "GPT 5.5",
                        "capabilities": {
                            "type": "chat",
                            "supports": {
                                "vision": False,
                                "thinking": True,
                                "tool_calls": True,
                                "parallel_tool_calls": False,
                            },
                        },
                    },
                    {
                        "id": "catalog-silent",
                        "name": "Catalog Silent",
                        "capabilities": {"type": "chat", "supports": {}},
                    },
                ]
            },
            request=httpx.Request("GET", "https://api.githubcopilot.com/models"),
        )

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    with patch(
        "httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        models = await provider.list_models(force_refresh=True)

    gpt, silent = models
    assert gpt.operation_capabilities == {
        Operation.CHAT: True,
        Operation.CHAT_STREAM: True,
        Operation.RESPONSES: True,
        Operation.RESPONSES_STREAM: True,
        Operation.NATIVE_ANTHROPIC: False,
    }
    assert gpt.feature_capabilities == {
        Feature.TOOLS: True,
        Feature.VISION: False,
        Feature.REASONING: True,
        Feature.PARALLEL_TOOLS: False,
    }
    assert silent.feature_capabilities == {}


@pytest.mark.asyncio
async def test_copilot_catalog_results_are_deep_defensive_copies():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "catalog-model",
                        "name": "Catalog Model",
                        "capabilities": {
                            "type": "chat",
                            "supports": {
                                "tool_calls": True,
                                "reasoning_effort": ["low"],
                            },
                        },
                    }
                ]
            },
            request=httpx.Request("GET", "https://api.githubcopilot.com/models"),
        )

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    with patch(
        "httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        first = await provider.list_models(force_refresh=True)

    first[0].reasoning_effort_values.append("high")
    first[0].operation_capabilities[Operation.CHAT] = False
    first[0].feature_capabilities[Feature.TOOLS] = False
    second = await provider.list_models()

    assert first is not second
    assert first[0] is not second[0]
    assert second[0].reasoning_effort_values == ["low"]
    assert second[0].operation_capabilities[Operation.CHAT] is True
    assert second[0].feature_capabilities[Feature.TOOLS] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("supported_endpoints", "expected"),
    [
        (
            ["/responses", "ws:/responses"],
            {
                Operation.CHAT: False,
                Operation.CHAT_STREAM: False,
                Operation.RESPONSES: True,
                Operation.RESPONSES_STREAM: True,
            },
        ),
        (
            ["/chat/completions"],
            {
                Operation.CHAT: True,
                Operation.CHAT_STREAM: True,
                Operation.RESPONSES: False,
                Operation.RESPONSES_STREAM: False,
            },
        ),
    ],
)
async def test_copilot_catalog_supported_endpoints_are_operation_authority(
    supported_endpoints,
    expected,
):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "catalog-model",
                        "name": "Catalog Model",
                        "supported_endpoints": supported_endpoints,
                        "capabilities": {"type": "chat", "supports": {}},
                    }
                ]
            },
            request=httpx.Request("GET", "https://api.githubcopilot.com/models"),
        )

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    with patch(
        "httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        models = await provider.list_models(force_refresh=True)

    operations = models[0].operation_capabilities
    assert {operation: operations[operation] for operation in expected} == expected
    assert models[0].supported_endpoints == tuple(supported_endpoints)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("include_key", "raw_endpoints", "expected_endpoints", "expected_operations"),
    [
        (
            False,
            None,
            None,
            {
                Operation.CHAT: True,
                Operation.CHAT_STREAM: True,
                Operation.RESPONSES: True,
                Operation.RESPONSES_STREAM: True,
                Operation.NATIVE_ANTHROPIC: False,
            },
        ),
        (
            True,
            ["/chat/completions", "/responses", "/v1/messages"],
            ("/chat/completions", "/responses", "/v1/messages"),
            dict.fromkeys(Operation, True),
        ),
        (True, [], (), dict.fromkeys(Operation, False)),
    ],
    ids=["missing", "valid", "empty"],
)
async def test_copilot_catalog_materializes_endpoint_contract_state(
    include_key,
    raw_endpoints,
    expected_endpoints,
    expected_operations,
) -> None:
    catalog_model = {
        "id": "gpt-5.4",
        "name": "GPT 5.4",
        "capabilities": {"type": "chat", "supports": {}},
    }
    if include_key:
        catalog_model["supported_endpoints"] = raw_endpoints

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [catalog_model]},
            request=httpx.Request("GET", "https://api.githubcopilot.com/models"),
        )

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    with patch(
        "httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        models = await provider.list_models(force_refresh=True)

    assert models[0].supported_endpoints == expected_endpoints
    assert models[0].operation_capabilities == expected_operations


@pytest.mark.asyncio
async def test_copilot_catalog_missing_supported_endpoints_keeps_heuristic_unknown_compatibility():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "future-model",
                        "name": "Future Model",
                        "capabilities": {"type": "chat", "supports": {}},
                    }
                ]
            },
            request=httpx.Request("GET", "https://api.githubcopilot.com/models"),
        )

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    with patch(
        "httpx.AsyncClient",
        return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ):
        models = await provider.list_models(force_refresh=True)

    assert models[0].operation_capabilities == {
        Operation.CHAT: True,
        Operation.CHAT_STREAM: True,
        Operation.NATIVE_ANTHROPIC: False,
    }
    assert models[0].supported_endpoints is None


@pytest.mark.asyncio
async def test_explicit_unsupported_reasoning_is_validated_as_typed_client_option():
    provider = _CapabilityProvider(
        "github-copilot",
        [
            _model(
                "github-copilot",
                "no-reasoning",
                operations={Operation.CHAT: True},
                features={Feature.REASONING: False},
            )
        ],
        frozenset(Operation),
    )

    def reject_reasoning(request: ChatRequest, *, stream: bool) -> None:
        raise RequestOptionError(
            "reasoning is unsupported",
            provider=provider.name,
            model=request.model,
            parameter="reasoning_effort",
        )

    provider.validate_chat_request = reject_reasoning  # type: ignore[method-assign]
    router = _router([provider], [])

    with pytest.raises(RequestOptionError) as caught:
        await router.chat_completion(
            ChatRequest(
                model="github-copilot/no-reasoning",
                messages=[Message(role="user", content="hi")],
                reasoning_effort="low",
            )
        )

    assert caught.value.parameter == "reasoning_effort"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (Operation.CHAT, CapabilitySupport.SUPPORTED),
        (Operation.CHAT_STREAM, CapabilitySupport.SUPPORTED),
        (Operation.RESPONSES, CapabilitySupport.SUPPORTED),
        (Operation.RESPONSES_STREAM, CapabilitySupport.SUPPORTED),
        (Operation.NATIVE_ANTHROPIC, CapabilitySupport.UNSUPPORTED),
    ],
)
async def test_operation_capabilities_are_per_model(operation: Operation, expected):
    provider = _CapabilityProvider(
        "github-copilot",
        [
            _model(
                "github-copilot",
                "gpt-5.5",
                operations={
                    Operation.CHAT: True,
                    Operation.CHAT_STREAM: True,
                    Operation.RESPONSES: True,
                    Operation.RESPONSES_STREAM: True,
                    Operation.NATIVE_ANTHROPIC: False,
                },
            ),
            _model(
                "github-copilot",
                "claude-sonnet-4",
                operations={
                    Operation.CHAT: True,
                    Operation.CHAT_STREAM: True,
                    Operation.RESPONSES: False,
                    Operation.RESPONSES_STREAM: False,
                    Operation.NATIVE_ANTHROPIC: True,
                },
            ),
        ],
        frozenset(Operation),
    )
    router = _router(provider and [provider], [])

    plan = await router.plan_route("github-copilot/gpt-5.5", operation)

    assert plan.primary.capabilities.operation(operation) is expected


@pytest.mark.asyncio
async def test_same_provider_models_do_not_share_native_anthropic_support():
    provider = _CapabilityProvider(
        "github-copilot",
        [
            _model(
                "github-copilot",
                "claude-sonnet-4",
                operations={Operation.NATIVE_ANTHROPIC: True},
            ),
            _model(
                "github-copilot",
                "gpt-5.5",
                operations={Operation.NATIVE_ANTHROPIC: False},
            ),
        ],
        frozenset(Operation),
    )
    router = _router([provider], [])

    claude = await router.plan_route("github-copilot/claude-sonnet-4", Operation.NATIVE_ANTHROPIC)
    gpt = await router.plan_route("github-copilot/gpt-5.5", Operation.NATIVE_ANTHROPIC)

    assert claude.primary.capabilities.operation(Operation.NATIVE_ANTHROPIC).is_supported
    assert gpt.primary.capabilities.operation(Operation.NATIVE_ANTHROPIC).is_unsupported


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "request_features"),
    [
        (Feature.TOOLS, RequestFeatures(tools=True)),
        (Feature.VISION, RequestFeatures(vision=True)),
        (Feature.REASONING, RequestFeatures(reasoning=True)),
        (Feature.PARALLEL_TOOLS, RequestFeatures(parallel_tools=True)),
    ],
)
async def test_required_features_participate_in_candidate_support(feature, request_features):
    provider = _CapabilityProvider(
        "p",
        [
            _model(
                "p",
                "no",
                operations={Operation.CHAT: True},
                features={feature: False},
            ),
            _model(
                "p",
                "yes",
                operations={Operation.CHAT: True},
                features={feature: True},
            ),
        ],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    router = _router([provider], ["p/no", "p/yes"])

    plan = await router.plan_route(
        "router-maestro",
        Operation.CHAT,
        request_features,
    )

    assert plan.primary.model.upstream_id == "yes"
    assert plan.primary.support is CapabilitySupport.SUPPORTED


@pytest.mark.asyncio
async def test_auto_responses_filters_unsupported_and_ranks_supported_before_unknown():
    provider = _CapabilityProvider(
        "p",
        [
            _model("p", "unsupported", operations={Operation.RESPONSES: False}),
            _model("p", "unknown"),
            _model("p", "supported", operations={Operation.RESPONSES: True}),
        ],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    router = _router([provider], ["p/unsupported", "p/unknown", "p/supported"])

    plan = await router.plan_route("router-maestro", Operation.RESPONSES)

    assert plan.primary.model.upstream_id == "supported"
    assert [candidate.model.upstream_id for candidate in plan.fallbacks] == ["unknown"]


@pytest.mark.asyncio
async def test_auto_responses_executes_supported_model_not_first_unsupported_priority():
    provider = _CapabilityProvider(
        "p",
        [
            _model("p", "unsupported", operations={Operation.RESPONSES: False}),
            _model("p", "supported", operations={Operation.RESPONSES: True}),
        ],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    router = _router([provider], ["p/unsupported", "p/supported"])

    response, used_provider = await router.responses_completion(
        ResponsesRequest(model="router-maestro", input="hi")
    )

    assert response.model == "supported"
    assert used_provider == "p"
    assert provider.calls == [(Operation.RESPONSES, "supported")]


@pytest.mark.asyncio
async def test_public_facade_normalizes_no_compatible_plan_to_provider_400():
    provider = _CapabilityProvider(
        "p",
        [_model("p", "unsupported", operations={Operation.RESPONSES: False})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    router = _router([provider], ["p/unsupported"])

    with pytest.raises(ProviderError) as exc:
        await router.responses_completion(ResponsesRequest(model="router-maestro", input="hi"))

    assert exc.value.status_code == 400
    assert exc.value.retryable is False
    assert not isinstance(exc.value, RequestOptionError)


@pytest.mark.asyncio
async def test_auto_none_strategy_has_no_fallback_and_does_not_retry():
    first = _CapabilityProvider(
        "first",
        [_model("first", "one", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        fail_retryable=True,
    )
    second = _CapabilityProvider(
        "second",
        [_model("second", "two", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    router = _router([first, second], ["first/one", "second/two"], strategy="none")

    plan = await router.plan_route("router-maestro", Operation.CHAT)

    assert plan.fallbacks == ()
    with pytest.raises(ProviderError):
        await router.chat_completion(
            ChatRequest(
                model="router-maestro",
                messages=[Message(role="user", content="hi")],
            )
        )
    assert second.calls == []


@pytest.mark.asyncio
async def test_auto_same_model_strategy_keeps_only_other_providers_with_same_id():
    first = _CapabilityProvider(
        "first",
        [_model("first", "same", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    second = _CapabilityProvider(
        "second",
        [_model("second", "same", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    third = _CapabilityProvider(
        "third",
        [_model("third", "different", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    router = _router(
        [first, second, third],
        ["first/same", "third/different", "second/same"],
        strategy="same-model",
    )

    plan = await router.plan_route("router-maestro", Operation.CHAT)

    assert plan.primary.model == ModelRef("first", "same")
    assert [candidate.model for candidate in plan.fallbacks] == [ModelRef("second", "same")]


@pytest.mark.asyncio
async def test_auto_same_model_adds_unconfigured_catalog_match():
    first = _CapabilityProvider(
        "first",
        [_model("first", "same", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    second = _CapabilityProvider(
        "second",
        [_model("second", "same", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    third = _CapabilityProvider(
        "third",
        [_model("third", "different", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    router = _router(
        [first, second, third],
        ["first/same"],
        strategy="same-model",
    )

    plan = await router.plan_route("router-maestro", Operation.CHAT)

    assert plan.primary.model == ModelRef("first", "same")
    assert [candidate.model for candidate in plan.fallbacks] == [ModelRef("second", "same")]


@pytest.mark.asyncio
async def test_plan_snapshots_fallback_limit_and_execution_ignores_later_config_mutation():
    primary = _CapabilityProvider(
        "primary",
        [_model("primary", "one", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        fail_retryable=True,
    )
    second = _CapabilityProvider(
        "second",
        [_model("second", "two", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        fail_retryable=True,
    )
    third = _CapabilityProvider(
        "third",
        [_model("third", "three", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    router = _router(
        [primary, second, third],
        ["primary/one", "second/two", "third/three"],
        max_retries=1,
    )
    plan = await router.plan_route("router-maestro", Operation.CHAT)
    request = ChatRequest(
        model="router-maestro",
        messages=[Message(role="user", content="hi")],
    )

    assert [candidate.model.upstream_id for candidate in plan.fallbacks] == ["two"]
    router._priorities_cache.set(
        PrioritiesConfig(
            priorities=["third/three"],
            fallback={"strategy": "priority", "maxRetries": 10},
        )
    )
    with pytest.raises(ProviderError):
        await router._execute_plan_nonstream(
            plan,
            request,
            True,
            router._create_request_with_model,
            lambda provider, candidate_request: provider.chat_completion(candidate_request),
        )
    assert primary.calls == [(Operation.CHAT, "one")]
    assert second.calls == [(Operation.CHAT, "two")]
    assert third.calls == []


@pytest.mark.asyncio
async def test_explicit_static_unsupported_does_not_substitute_or_call_provider():
    primary = _CapabilityProvider(
        "primary",
        [_model("primary", "m", operations={Operation.RESPONSES: False})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    fallback = _CapabilityProvider(
        "fallback",
        [_model("fallback", "other", operations={Operation.RESPONSES: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    router = _router([primary, fallback], ["fallback/other"])

    with pytest.raises(RequestOptionError) as exc:
        await router.responses_completion(ResponsesRequest(model="primary/m", input="hi"))

    assert exc.value.status_code == 400
    assert exc.value.retryable is False
    assert exc.value.parameter == "responses"
    assert primary.calls == []
    assert fallback.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "responses_request"),
    [
        (
            Feature.TOOLS,
            ResponsesRequest(
                model="primary/m",
                input="hi",
                tools=[{"type": "function", "name": "lookup"}],
            ),
        ),
        (
            Feature.VISION,
            ResponsesRequest(
                model="primary/m",
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
                        ],
                    }
                ],
            ),
        ),
        (
            Feature.PARALLEL_TOOLS,
            ResponsesRequest(
                model="primary/m",
                input="hi",
                parallel_tool_calls=True,
            ),
        ),
    ],
)
async def test_explicit_static_feature_mismatch_is_typed_before_validation_or_fallback(
    feature: Feature,
    responses_request: ResponsesRequest,
) -> None:
    primary = _CapabilityProvider(
        "primary",
        [
            _model(
                "primary",
                "m",
                operations={Operation.RESPONSES: True},
                features={feature: False},
            )
        ],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    fallback = _CapabilityProvider(
        "fallback",
        [
            _model(
                "fallback",
                "other",
                operations={Operation.RESPONSES: True},
                features={feature: True},
            )
        ],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    router = _router([primary, fallback], ["fallback/other"])

    with pytest.raises(RequestOptionError) as exc:
        await router.responses_completion(responses_request)

    expected_parameter = (
        "parallel_tool_calls" if feature is Feature.PARALLEL_TOOLS else feature.value
    )
    assert exc.value.parameter == expected_parameter
    assert primary.validations == []
    assert fallback.validations == []
    assert primary.calls == []
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_validation_facade_rejects_static_mismatch_before_provider_validation() -> None:
    provider = _CapabilityProvider(
        "primary",
        [_model("primary", "m", operations={Operation.RESPONSES: False})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    router = _router([provider], [])

    with pytest.raises(RequestOptionError) as exc:
        await router.validate_responses_request(ResponsesRequest(model="primary/m", input="hi"))

    assert exc.value.parameter == "responses"
    assert provider.validations == []


@pytest.mark.asyncio
async def test_explicit_unknown_is_attempted_for_backward_compatibility():
    provider = _CapabilityProvider(
        "p",
        [_model("p", "unknown")],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    router = _router([provider], [])

    response, used_provider = await router.responses_completion(
        ResponsesRequest(model="p/unknown", input="hi")
    )

    assert response.content == "ok"
    assert used_provider == "p"
    assert provider.calls == [(Operation.RESPONSES, "unknown")]


@pytest.mark.asyncio
async def test_explicit_unknown_runtime_unsupported_does_not_switch_models():
    primary = _CapabilityProvider(
        "primary",
        [_model("primary", "unknown")],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    fallback = _CapabilityProvider(
        "fallback",
        [_model("fallback", "supported", operations={Operation.RESPONSES: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )

    async def unsupported(_request: ResponsesRequest) -> ResponsesResponse:
        primary.calls.append((Operation.RESPONSES, "unknown"))
        raise ProviderError("unsupported", status_code=501, retryable=False)

    primary.responses_completion = unsupported  # type: ignore[method-assign]
    router = _router([primary, fallback], ["fallback/supported"])

    with pytest.raises(ProviderError) as exc:
        await router.responses_completion(ResponsesRequest(model="primary/unknown", input="hi"))

    assert exc.value.status_code == 501
    assert primary.calls == [(Operation.RESPONSES, "unknown")]
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_auto_unknown_nonretryable_501_does_not_guess_typed_unsupported():
    first = _CapabilityProvider(
        "first",
        [_model("first", "unknown-one")],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )
    second = _CapabilityProvider(
        "second",
        [_model("second", "unknown-two")],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM, Operation.RESPONSES}),
    )

    async def unsupported(_request: ResponsesRequest) -> ResponsesResponse:
        first.calls.append((Operation.RESPONSES, "unknown-one"))
        raise ProviderError("unsupported", status_code=501, retryable=False)

    first.responses_completion = unsupported  # type: ignore[method-assign]
    router = _router([first, second], ["first/unknown-one", "second/unknown-two"])

    with pytest.raises(ProviderError) as exc:
        await router.responses_completion(ResponsesRequest(model="router-maestro", input="hi"))

    assert exc.value.status_code == 501
    assert first.calls == [(Operation.RESPONSES, "unknown-one")]
    assert second.calls == []


@pytest.mark.asyncio
async def test_provider_transport_unsupported_overrides_model_claim():
    provider = _CapabilityProvider(
        "p",
        [_model("p", "m", operations={Operation.RESPONSES: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    router = _router([provider], [])

    plan = await router.plan_route("p/m", Operation.RESPONSES)

    assert plan.primary.capabilities.operation(Operation.RESPONSES) is CapabilitySupport.UNSUPPORTED
    assert plan.primary.support is CapabilitySupport.UNSUPPORTED


@pytest.mark.asyncio
async def test_existing_model_metadata_contributes_only_known_feature_facts():
    provider = _CapabilityProvider(
        "p",
        [
            ModelInfo(
                id="known",
                name="known",
                provider="p",
                operation_capabilities={Operation.CHAT: True},
                supports_thinking=True,
                supports_vision=True,
                reasoning_effort_values=[],
            )
        ],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    router = _router([provider], [])

    plan = await router.plan_route("p/known", Operation.CHAT)

    capabilities = plan.primary.capabilities
    assert capabilities.feature(Feature.VISION) is CapabilitySupport.SUPPORTED
    assert capabilities.feature(Feature.REASONING) is CapabilitySupport.UNSUPPORTED
    assert capabilities.feature(Feature.TOOLS) is CapabilitySupport.UNKNOWN


@pytest.mark.asyncio
async def test_explicit_primary_absent_from_priorities_uses_full_deduplicated_fallback_list():
    primary = _CapabilityProvider(
        "primary",
        [_model("primary", "requested", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        fail_retryable=True,
    )
    first = _CapabilityProvider(
        "first",
        [_model("first", "one", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    second = _CapabilityProvider(
        "second",
        [_model("second", "two", operations={Operation.CHAT: True})],
        frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )
    router = _router(
        [primary, first, second],
        ["first/one", "second/two", "first/one"],
    )

    plan = await router.plan_route("primary/requested", Operation.CHAT)
    response, used_provider = await router.chat_completion(
        ChatRequest(
            model="primary/requested",
            messages=[Message(role="user", content="hi")],
        )
    )

    assert [candidate.model.qualified_id for candidate in plan.fallbacks] == [
        "first/one",
        "second/two",
    ]
    assert response.content == "ok"
    assert used_provider == "first"
    assert primary.calls == [(Operation.CHAT, "requested")]
    assert first.calls == [(Operation.CHAT, "one")]
    assert second.calls == []


def test_request_feature_detection_covers_chat_and_responses_shapes():
    chat_request = ChatRequest(
        model="m",
        messages=[
            Message(
                role="user",
                content=[{"type": "image_url", "image_url": {"url": "https://image"}}],
            )
        ],
        tools=[{"type": "function", "function": {"name": "tool"}}],
        reasoning_effort="high",
    )
    responses_request = ResponsesRequest(
        model="m",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://image"}],
            }
        ],
        tools=[{"type": "function", "name": "tool"}],
        parallel_tool_calls=True,
        reasoning_effort="medium",
    )

    assert RequestFeatures.for_chat(chat_request) == RequestFeatures(
        tools=True,
        vision=True,
        reasoning=True,
        parallel_tools=False,
    )
    assert RequestFeatures.for_responses(responses_request) == RequestFeatures(
        tools=True,
        vision=True,
        reasoning=True,
        parallel_tools=True,
    )


def test_request_feature_detection_tolerates_non_string_schema_type():
    # A prior turn echoes tool JSON Schemas whose ``type`` is a list, e.g.
    # ``{"type": ["string", "null"]}``. Vision detection walks the whole input
    # and must not choke testing an unhashable list against the image-block set.
    responses_request = ResponsesRequest(
        model="m",
        input=[
            {
                "type": "function_call",
                "parameters": {
                    "properties": {
                        "org": {"type": ["string", "null"]},
                        "count": {"type": ["array", "null"]},
                    }
                },
            }
        ],
        tools=[{"type": "function", "name": "tool"}],
    )

    assert RequestFeatures.for_responses(responses_request) == RequestFeatures(
        tools=True,
        vision=False,
        reasoning=False,
        parallel_tools=False,
    )


def test_plan_route_is_not_used_for_token_counting_operation():
    assert {operation.value for operation in Operation} == {
        "chat",
        "chat_stream",
        "responses",
        "responses_stream",
        "native_anthropic",
    }


class TestResponsesEligibilityFallback:
    """The relocated cold-start eligibility helper (now in copilot_support.catalog).

    The live catalog's ``supported_endpoints`` is authoritative; this hardcoded
    list is only the fallback before the catalog is warm.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5-mini",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "mai-code-1-flash-picker",
            "github-copilot/gpt-5.4",  # provider prefix stripped
        ],
    )
    def test_eligible(self, model):
        from router_maestro.providers.copilot_support.catalog import is_model_responses_eligible

        assert is_model_responses_eligible(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4.7",
            "gemini-3.1-pro-preview",
            "gpt-4o",
            "github-copilot/claude-opus-4.6",
            # Pruned: no longer served by GHC.
            "gpt-5.2",
            "gpt-5.2-codex",
        ],
    )
    def test_ineligible(self, model):
        from router_maestro.providers.copilot_support.catalog import is_model_responses_eligible

        assert is_model_responses_eligible(model) is False
