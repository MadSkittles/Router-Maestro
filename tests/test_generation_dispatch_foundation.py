"""Focused contracts for model-only routing and provider-owned transport selection."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

import pytest

from router_maestro.config import (
    FallbackConfig,
    FallbackStrategy,
    PrioritiesConfig,
)
from router_maestro.protocols import ConversionMode, RequestManifest, WireProtocol
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
)
from router_maestro.providers.bindings import (
    COPILOT_ANTHROPIC_MESSAGES_BINDING,
    COPILOT_OPENAI_CHAT_BINDING,
    COPILOT_OPENAI_RESPONSES_BINDING,
    EndpointBinding,
    legacy_endpoint_binding,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.providers.handler import ProviderHandler
from router_maestro.routing.capabilities import Feature, Operation, ProviderCapabilities
from router_maestro.routing.generation_plan import GenerationCandidate, plan_generation_route
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.utils.cache import TTLCache


def _operations(protocol: WireProtocol) -> frozenset[Operation]:
    return {
        WireProtocol.ANTHROPIC_MESSAGES: frozenset({Operation.NATIVE_ANTHROPIC}),
        WireProtocol.OPENAI_CHAT: frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        WireProtocol.OPENAI_RESPONSES: frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM}),
    }[protocol]


def _binding(protocol: WireProtocol, *, binding_id: str | None = None) -> EndpointBinding:
    return legacy_endpoint_binding(
        binding_id=binding_id or f"test-{protocol.value}",
        protocol=protocol,
        operations=_operations(protocol),
    )


_ALL_BINDINGS = (
    _binding(WireProtocol.ANTHROPIC_MESSAGES),
    _binding(WireProtocol.OPENAI_CHAT),
    _binding(WireProtocol.OPENAI_RESPONSES),
)


class _Provider(BaseProvider):
    def __init__(
        self,
        name: str,
        model_ids: tuple[str, ...],
        *,
        bindings: tuple[EndpointBinding, ...] = _ALL_BINDINGS,
        preferences: tuple[str, ...] | None = None,
        aliases: Mapping[str, str] | None = None,
        authenticated: bool = True,
    ) -> None:
        self.name = name
        self._models = tuple(
            ModelInfo(id=model_id, name=model_id, provider=name) for model_id in model_ids
        )
        self._bindings = bindings
        self._preferences = preferences
        self._aliases = dict(aliases or {})
        self._authenticated = authenticated

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(operations=frozenset(Operation))

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="ok", model=request.model)

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="ok", finish_reason="stop")

    async def list_models(self) -> list[ModelInfo]:
        return list(self._models)

    def is_authenticated(self) -> bool:
        return self._authenticated

    def model_aliases(self) -> Mapping[str, str]:
        return self._aliases

    def bindings(self) -> tuple[EndpointBinding, ...]:
        return self._bindings

    def transport_preferences(
        self,
        ingress_protocol: WireProtocol | None = None,
    ) -> tuple[str, ...]:
        del ingress_protocol
        if self._preferences is not None:
            return self._preferences
        return tuple(binding.id for binding in self._bindings)


def _router(
    providers: tuple[_Provider, ...],
    *,
    priorities: tuple[str, ...],
    strategy: FallbackStrategy = FallbackStrategy.PRIORITY,
    max_retries: int = 2,
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
            priorities=list(priorities),
            fallback=FallbackConfig(strategy=strategy, maxRetries=max_retries),
        )
    )
    router._providers_ttl.set(True)
    return router


def _candidate(provider: BaseProvider, info: ModelInfo | None = None) -> GenerationCandidate:
    info = info or ModelInfo(id="model", name="model", provider=provider.name)
    return GenerationCandidate(
        model=ModelRef(provider.name, info.id),
        provider=provider,
        info=info,
    )


@pytest.mark.asyncio
async def test_generation_route_resolves_explicit_model_before_transport_selection() -> None:
    alpha = _Provider("alpha", ("one",))
    beta = _Provider("beta", ("two",))
    router = _router(
        (alpha, beta),
        priorities=("alpha/one", "beta/two"),
        max_retries=1,
    )

    plan = await plan_generation_route(router, "alpha/one")

    assert plan.explicit is True
    assert plan.primary.model == ModelRef("alpha", "one")
    assert [candidate.model for candidate in plan.fallbacks] == [ModelRef("beta", "two")]
    assert plan.max_model_switches == 1


@pytest.mark.asyncio
async def test_generation_route_auto_route_uses_configured_model_priority() -> None:
    alpha = _Provider("alpha", ("one", "unconfigured"))
    beta = _Provider("beta", ("two",))
    router = _router(
        (alpha, beta),
        priorities=("beta/two", "alpha/one"),
        max_retries=1,
    )

    plan = await plan_generation_route(router, "router-maestro")

    assert plan.explicit is False
    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("beta", "two"),
        ModelRef("alpha", "one"),
    ]


@pytest.mark.asyncio
async def test_auto_route_skips_explicit_catalog_feature_negatives() -> None:
    alpha = _Provider("alpha", ("one",))
    beta = _Provider("beta", ("two",))
    alpha._models = (
        ModelInfo(
            id="one",
            name="one",
            provider="alpha",
            feature_capabilities={Feature.TOOLS.value: False},
        ),
    )
    router = _router(
        (alpha, beta),
        priorities=("alpha/one", "beta/two"),
        max_retries=1,
    )

    plan = await plan_generation_route(
        router,
        "router-maestro",
        RequestManifest(protocol=WireProtocol.OPENAI_RESPONSES, tools=True),
    )

    assert plan.primary.model == ModelRef("beta", "two")


@pytest.mark.asyncio
async def test_auto_route_filters_every_fallback_before_applying_switch_limit() -> None:
    alpha = _Provider("alpha", ("one",))
    beta = _Provider("beta", ("two",))
    gamma = _Provider("gamma", ("three",))
    beta._models = (
        ModelInfo(
            id="two",
            name="two",
            provider="beta",
            feature_capabilities={Feature.REASONING.value: False},
        ),
    )
    router = _router(
        (alpha, beta, gamma),
        priorities=("alpha/one", "beta/two", "gamma/three"),
        max_retries=2,
    )

    plan = await plan_generation_route(
        router,
        "router-maestro",
        RequestManifest(protocol=WireProtocol.ANTHROPIC_MESSAGES, reasoning=True),
    )

    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("alpha", "one"),
        ModelRef("gamma", "three"),
    ]
    assert plan.max_model_switches == 1


@pytest.mark.asyncio
async def test_explicit_incompatible_primary_is_retained_and_fallbacks_filtered() -> None:
    alpha = _Provider("alpha", ("one",))
    beta = _Provider("beta", ("two",))
    gamma = _Provider("gamma", ("three",))
    for provider, model_id in ((alpha, "one"), (beta, "two")):
        provider._models = (
            ModelInfo(
                id=model_id,
                name=model_id,
                provider=provider.name,
                feature_capabilities={Feature.VISION.value: False},
            ),
        )
    router = _router(
        (alpha, beta, gamma),
        priorities=("alpha/one", "beta/two", "gamma/three"),
        max_retries=2,
    )

    plan = await plan_generation_route(
        router,
        "alpha/one",
        RequestManifest(protocol=WireProtocol.OPENAI_CHAT, images=True),
    )

    assert plan.explicit is True
    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("alpha", "one"),
        ModelRef("gamma", "three"),
    ]


@pytest.mark.asyncio
async def test_non_explicit_alias_skips_feature_incompatible_provider() -> None:
    alpha = _Provider("alpha", ("shared",))
    beta = _Provider("beta", ("shared",))
    alpha._models = (
        ModelInfo(
            id="shared",
            name="shared",
            provider="alpha",
            feature_capabilities={Feature.PARALLEL_TOOLS.value: False},
        ),
    )
    router = _router(
        (alpha, beta),
        priorities=("alpha/shared", "beta/shared"),
        max_retries=1,
    )

    plan = await plan_generation_route(
        router,
        "shared",
        RequestManifest(
            protocol=WireProtocol.OPENAI_RESPONSES,
            tools=True,
            parallel_tools=True,
        ),
    )

    assert plan.explicit is False
    assert plan.primary.model == ModelRef("beta", "shared")


@pytest.mark.asyncio
async def test_non_explicit_alias_with_no_compatible_model_is_a_static_error() -> None:
    alpha = _Provider("alpha", ("shared",))
    alpha._models = (
        ModelInfo(
            id="shared",
            name="shared",
            provider="alpha",
            feature_capabilities={Feature.TOOLS.value: False},
        ),
    )
    router = _router((alpha,), priorities=("alpha/shared",))

    with pytest.raises(ProviderError) as raised:
        await plan_generation_route(
            router,
            "shared",
            RequestManifest(protocol=WireProtocol.OPENAI_CHAT, tools=True),
        )

    assert raised.value.status_code == 400
    assert raised.value.retryable is False


@pytest.mark.asyncio
async def test_explicit_unconfigured_model_falls_back_only_to_configured_priorities() -> None:
    alpha = _Provider("alpha", ("explicit", "catalog-only"))
    beta = _Provider("beta", ("configured",))
    router = _router(
        (alpha, beta),
        priorities=("beta/configured",),
        max_retries=2,
    )

    plan = await plan_generation_route(router, "alpha/explicit")

    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("alpha", "explicit"),
        ModelRef("beta", "configured"),
    ]


@pytest.mark.asyncio
async def test_explicit_model_has_no_priority_fallback_when_priorities_are_empty() -> None:
    alpha = _Provider("alpha", ("explicit", "catalog-only"))
    router = _router((alpha,), priorities=(), max_retries=2)

    plan = await plan_generation_route(router, "alpha/explicit")

    assert plan.candidates == (plan.primary,)


@pytest.mark.asyncio
async def test_generation_route_bare_model_alias_spans_providers_in_priority_order() -> None:
    alpha = _Provider("alpha", ("shared",))
    beta = _Provider("beta", ("shared",))
    router = _router(
        (alpha, beta),
        priorities=("beta/shared", "alpha/shared"),
        max_retries=1,
    )

    plan = await plan_generation_route(router, "shared")

    assert plan.explicit is False
    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("beta", "shared"),
        ModelRef("alpha", "shared"),
    ]


@pytest.mark.asyncio
async def test_generation_route_normalizes_provider_declared_alias() -> None:
    provider = _Provider(
        "alpha",
        ("target",),
        aliases={"guardian": "target"},
    )
    router = _router(providers=(provider,), priorities=("alpha/target",))

    plan = await plan_generation_route(router, "GuArDiAn")

    assert plan.explicit is True
    assert plan.primary.model == ModelRef("alpha", "target")


@pytest.mark.asyncio
async def test_generation_route_same_model_fallback_excludes_other_models() -> None:
    alpha = _Provider("alpha", ("shared",))
    beta = _Provider("beta", ("other", "shared"))
    gamma = _Provider("gamma", ("shared",))
    router = _router(
        (alpha, beta, gamma),
        priorities=("alpha/shared", "beta/other", "beta/shared", "gamma/shared"),
        strategy=FallbackStrategy.SAME_MODEL,
        max_retries=2,
    )

    plan = await plan_generation_route(router, "alpha/shared")

    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("alpha", "shared"),
        ModelRef("beta", "shared"),
        ModelRef("gamma", "shared"),
    ]
    assert plan.max_model_switches == 2


@pytest.mark.asyncio
async def test_generation_route_none_disables_model_fallback() -> None:
    alpha = _Provider("alpha", ("one",))
    beta = _Provider("beta", ("two",))
    router = _router(
        (alpha, beta),
        priorities=("alpha/one", "beta/two"),
        strategy=FallbackStrategy.NONE,
        max_retries=2,
    )

    plan = await plan_generation_route(router, "router-maestro")

    # Auto priority-chain owns its complete explicit fallback chain; the
    # legacy global fallback strategy only governs direct-model requests.
    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("alpha", "one"),
        ModelRef("beta", "two"),
    ]
    assert plan.max_model_switches == 1


@pytest.mark.asyncio
async def test_max_retries_limits_model_switches_not_provider_transports() -> None:
    alpha = _Provider("alpha", ("one",))
    beta = _Provider("beta", ("two",))
    gamma = _Provider("gamma", ("three",))
    router = _router(
        (alpha, beta, gamma),
        priorities=("alpha/one", "beta/two", "gamma/three"),
        max_retries=1,
    )

    plan = await plan_generation_route(router, "router-maestro")
    transports = [
        ProviderHandler(candidate.provider).bindings_for(
            candidate,
            WireProtocol.ANTHROPIC_MESSAGES,
        )
        for candidate in plan.candidates
    ]

    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("alpha", "one"),
        ModelRef("beta", "two"),
        ModelRef("gamma", "three"),
    ]
    assert [len(options) for options in transports] == [3, 3, 3]
    assert plan.max_model_switches == 2


@pytest.mark.parametrize(
    ("ingress", "expected_ids", "expected_modes"),
    [
        (
            WireProtocol.ANTHROPIC_MESSAGES,
            (
                COPILOT_ANTHROPIC_MESSAGES_BINDING,
                COPILOT_OPENAI_RESPONSES_BINDING,
                COPILOT_OPENAI_CHAT_BINDING,
            ),
            (
                ConversionMode.IDENTITY,
                ConversionMode.SEMANTIC_IR,
                ConversionMode.SEMANTIC_IR,
            ),
        ),
        (
            WireProtocol.OPENAI_CHAT,
            (
                COPILOT_OPENAI_CHAT_BINDING,
                COPILOT_OPENAI_RESPONSES_BINDING,
                COPILOT_ANTHROPIC_MESSAGES_BINDING,
            ),
            (
                ConversionMode.IDENTITY,
                ConversionMode.SEMANTIC_IR,
                ConversionMode.SEMANTIC_IR,
            ),
        ),
        (
            WireProtocol.OPENAI_RESPONSES,
            (
                COPILOT_OPENAI_RESPONSES_BINDING,
                COPILOT_OPENAI_CHAT_BINDING,
                COPILOT_ANTHROPIC_MESSAGES_BINDING,
            ),
            (
                ConversionMode.IDENTITY,
                ConversionMode.SEMANTIC_IR,
                ConversionMode.SEMANTIC_IR,
            ),
        ),
        (
            WireProtocol.GEMINI,
            (
                COPILOT_OPENAI_RESPONSES_BINDING,
                COPILOT_OPENAI_CHAT_BINDING,
                COPILOT_ANTHROPIC_MESSAGES_BINDING,
            ),
            (
                ConversionMode.SEMANTIC_IR,
                ConversionMode.SEMANTIC_IR,
                ConversionMode.SEMANTIC_IR,
            ),
        ),
    ],
)
def test_copilot_handler_uses_ingress_specific_transport_order(
    ingress: WireProtocol,
    expected_ids: tuple[str, ...],
    expected_modes: tuple[ConversionMode, ...],
) -> None:
    provider = CopilotProvider.__new__(CopilotProvider)
    plans = ProviderHandler(provider).bindings_for(_candidate(provider), ingress)

    assert tuple(plan.binding.id for plan in plans) == expected_ids
    assert tuple(plan.conversion_mode for plan in plans) == expected_modes


def test_handler_filters_explicit_transport_false_but_probes_missing_metadata() -> None:
    provider = _Provider("alpha", ("model",))
    info = ModelInfo(
        id="model",
        name="model",
        provider="alpha",
        transport_capabilities={WireProtocol.ANTHROPIC_MESSAGES.value: False},
    )

    plans = ProviderHandler(provider).bindings_for(
        _candidate(provider, info),
        WireProtocol.GEMINI,
    )

    assert [plan.target_protocol for plan in plans] == [
        WireProtocol.OPENAI_CHAT,
        WireProtocol.OPENAI_RESPONSES,
    ]


@pytest.mark.parametrize(
    ("feature", "manifest"),
    [
        (
            Feature.TOOLS,
            RequestManifest(protocol=WireProtocol.OPENAI_RESPONSES, tools=True),
        ),
        (
            Feature.VISION,
            RequestManifest(protocol=WireProtocol.OPENAI_RESPONSES, images=True),
        ),
        (
            Feature.REASONING,
            RequestManifest(protocol=WireProtocol.OPENAI_RESPONSES, reasoning=True),
        ),
        (
            Feature.PARALLEL_TOOLS,
            RequestManifest(protocol=WireProtocol.OPENAI_RESPONSES, parallel_tools=True),
        ),
    ],
)
def test_handler_filters_explicit_model_feature_false(
    feature: Feature,
    manifest: RequestManifest,
) -> None:
    provider = _Provider("alpha", ("model",))
    info = ModelInfo(
        id="model",
        name="model",
        provider="alpha",
        feature_capabilities={feature.value: False},
    )
    plans = ProviderHandler(provider).bindings_for(
        _candidate(provider, info),
        WireProtocol.OPENAI_RESPONSES,
        manifest,
    )

    assert plans == ()


@pytest.mark.parametrize("declared", [None, True])
def test_handler_keeps_unknown_or_supported_model_features_probeable(
    declared: bool | None,
) -> None:
    provider = _Provider("alpha", ("model",))
    info = ModelInfo(
        id="model",
        name="model",
        provider="alpha",
        feature_capabilities={} if declared is None else {Feature.TOOLS.value: declared},
    )
    manifest = RequestManifest(
        protocol=WireProtocol.OPENAI_RESPONSES,
        tools=True,
    )

    plans = ProviderHandler(provider).bindings_for(
        _candidate(provider, info),
        WireProtocol.OPENAI_RESPONSES,
        manifest,
    )

    assert len(plans) == 3


def test_handler_does_not_guess_a_model_feature_for_files() -> None:
    provider = _Provider("alpha", ("model",))
    info = ModelInfo(
        id="model",
        name="model",
        provider="alpha",
        feature_capabilities={"files": False},
    )
    manifest = RequestManifest(
        protocol=WireProtocol.OPENAI_RESPONSES,
        files=True,
    )

    plans = ProviderHandler(provider).bindings_for(
        _candidate(provider, info),
        WireProtocol.OPENAI_RESPONSES,
        manifest,
    )

    assert len(plans) == 3


def test_handler_ignores_explicit_negative_for_an_unused_feature() -> None:
    provider = _Provider("alpha", ("model",))
    info = ModelInfo(
        id="model",
        name="model",
        provider="alpha",
        feature_capabilities={Feature.TOOLS.value: False},
    )

    plans = ProviderHandler(provider).bindings_for(
        _candidate(provider, info),
        WireProtocol.OPENAI_RESPONSES,
        RequestManifest(protocol=WireProtocol.OPENAI_RESPONSES),
    )

    assert len(plans) == 3


def test_handler_rejects_manifest_for_another_ingress_protocol() -> None:
    provider = _Provider("alpha", ("model",))

    with pytest.raises(ValueError, match="manifest protocol"):
        ProviderHandler(provider).bindings_for(
            _candidate(provider),
            WireProtocol.OPENAI_CHAT,
            RequestManifest(protocol=WireProtocol.GEMINI),
        )


def test_handler_places_identity_binding_before_provider_cross_protocol_preference() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT)
    responses = _binding(WireProtocol.OPENAI_RESPONSES)
    provider = _Provider(
        "alpha",
        ("model",),
        bindings=(chat, responses),
        preferences=(responses.id, chat.id),
    )

    plans = ProviderHandler(provider).bindings_for(
        _candidate(provider),
        WireProtocol.OPENAI_CHAT,
    )

    assert [plan.binding.id for plan in plans] == [chat.id, responses.id]
    assert [plan.conversion_mode for plan in plans] == [
        ConversionMode.IDENTITY,
        ConversionMode.SEMANTIC_IR,
    ]


def test_handler_recognizes_arbitrary_messages_endpoint_path() -> None:
    provider = _Provider("alpha", ("model",))
    info = ModelInfo(
        id="model",
        name="model",
        provider="alpha",
        supported_endpoints=("/api/anthropic/v1/messages",),
    )

    plans = ProviderHandler(provider).bindings_for(
        _candidate(provider, info),
        WireProtocol.GEMINI,
    )

    assert [plan.target_protocol for plan in plans] == [WireProtocol.ANTHROPIC_MESSAGES]


def test_handler_keeps_binding_when_one_operation_capability_is_unknown() -> None:
    chat_binding = _binding(WireProtocol.OPENAI_CHAT)
    provider = _Provider("alpha", ("model",), bindings=(chat_binding,))
    partially_known = ModelInfo(
        id="model",
        name="model",
        provider="alpha",
        operation_capabilities={Operation.CHAT.value: False},
    )
    explicitly_denied = ModelInfo(
        id="model",
        name="model",
        provider="alpha",
        operation_capabilities={
            Operation.CHAT.value: False,
            Operation.CHAT_STREAM.value: False,
        },
    )

    probeable = ProviderHandler(provider).bindings_for(
        _candidate(provider, partially_known),
        WireProtocol.ANTHROPIC_MESSAGES,
    )
    denied = ProviderHandler(provider).bindings_for(
        _candidate(provider, explicitly_denied),
        WireProtocol.ANTHROPIC_MESSAGES,
    )

    assert [plan.binding.id for plan in probeable] == [chat_binding.id]
    assert denied == ()


def test_handler_rejects_duplicate_binding_ids() -> None:
    provider = _Provider(
        "alpha",
        ("model",),
        bindings=(
            _binding(WireProtocol.OPENAI_CHAT, binding_id="duplicate"),
            _binding(WireProtocol.OPENAI_RESPONSES, binding_id="duplicate"),
        ),
    )

    with pytest.raises(ValueError, match="duplicate binding IDs"):
        ProviderHandler(provider).bindings_for(
            _candidate(provider),
            WireProtocol.OPENAI_CHAT,
        )


def test_handler_rejects_duplicate_transport_preferences() -> None:
    chat = _binding(WireProtocol.OPENAI_CHAT)
    provider = _Provider(
        "alpha",
        ("model",),
        bindings=(chat,),
        preferences=(chat.id, chat.id),
    )

    with pytest.raises(ValueError, match="duplicate transport preferences"):
        ProviderHandler(provider).bindings_for(
            _candidate(provider),
            WireProtocol.OPENAI_CHAT,
        )
