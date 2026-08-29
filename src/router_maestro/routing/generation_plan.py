"""Model-only route planning for protocol-independent generation dispatch.

The legacy :mod:`route_plan` module freezes one provider operation (Chat or
Responses) into every candidate.  A generation dispatcher must instead select
the provider/model first and let the provider choose one of its endpoint
bindings afterwards.  This module deliberately contains no protocol, binding,
payload, or execution state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from router_maestro.config import AutoCapabilityPolicy, AutoMode, FallbackStrategy
from router_maestro.protocols.models import RequestManifest
from router_maestro.providers.base import (
    BaseProvider,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
)
from router_maestro.routing.capabilities import model_supports_manifest
from router_maestro.routing.model_ref import ModelRef
from router_maestro.utils.model_match import AmbiguousModelMatchError, fuzzy_match_model

if TYPE_CHECKING:
    from router_maestro.routing.router import Router


AUTO_CONTEXT_SAFETY_NUMERATOR = 7
AUTO_CONTEXT_SAFETY_DENOMINATOR = 10


@dataclass(frozen=True, slots=True)
class GenerationCandidate:
    """One provider/model selected without choosing an upstream protocol."""

    model: ModelRef
    provider: BaseProvider
    info: ModelInfo

    def __post_init__(self) -> None:
        if self.provider.name != self.model.provider:
            raise ValueError("generation candidate provider must match its model")
        if self.info.provider != self.model.provider or self.info.id != self.model.upstream_id:
            raise ValueError("generation candidate catalog metadata must match its model")


@dataclass(frozen=True, slots=True)
class GenerationRoutePlan:
    """An immutable model fallback plan, intentionally independent of transport."""

    primary: GenerationCandidate
    fallbacks: tuple[GenerationCandidate, ...] = ()
    explicit: bool = False
    max_model_switches: int = 0
    fallback_on_context_overflow_only: bool = False

    def __post_init__(self) -> None:
        fallbacks = tuple(self.fallbacks)
        object.__setattr__(self, "fallbacks", fallbacks)
        if self.max_model_switches < 0:
            raise ValueError("max_model_switches cannot be negative")
        if len(fallbacks) > self.max_model_switches:
            raise ValueError("fallbacks exceed max_model_switches")
        models = (self.primary.model, *(candidate.model for candidate in fallbacks))
        if len(models) != len(set(models)):
            raise ValueError("generation route cannot contain duplicate models")

    @property
    def candidates(self) -> tuple[GenerationCandidate, ...]:
        return (self.primary, *self.fallbacks)


def model_prompt_capacity(info: ModelInfo) -> int | None:
    """Return the largest prompt budget this concrete model advertises."""
    options = info.effective_context_window_options()
    if options:
        return max(option.max_prompt_tokens for option in options)
    return info.max_prompt_tokens


def auto_context_is_safe(info: ModelInfo, estimated_input_tokens: int | None) -> bool:
    """Keep Auto below 70% of the advertised prompt budget.

    Tokenizers and protocol overhead differ between Router-Maestro's estimate
    and the provider's final accounting. Reaching the threshold is therefore a
    signal to choose a larger configured task model before provider I/O.
    """
    if estimated_input_tokens is None:
        return True
    capacity = model_prompt_capacity(info)
    if capacity is None:
        return True
    return (
        estimated_input_tokens * AUTO_CONTEXT_SAFETY_DENOMINATOR
        < capacity * AUTO_CONTEXT_SAFETY_NUMERATOR
    )


def _catalog_candidate(
    router: Any,
    provider_name: str,
    model: ModelInfo,
) -> GenerationCandidate | None:
    provider = router.providers.get(provider_name)
    if provider is None or not provider.is_authenticated():
        return None
    return GenerationCandidate(
        model=ModelRef(provider_name, model.id),
        provider=provider,
        info=deepcopy(model),
    )


def _ordered_catalog_candidates(router: Any) -> list[GenerationCandidate]:
    """Return configured models first, then every remaining canonical entry."""
    ordered: list[GenerationCandidate] = []
    seen: set[ModelRef] = set()

    def append_entry(key: str) -> None:
        entry = router._models_cache.get(key)
        if entry is None or not router._is_qualified_cache_entry(key, entry):
            return
        candidate = _catalog_candidate(router, entry[0], entry[1])
        if candidate is None or candidate.model in seen:
            return
        seen.add(candidate.model)
        ordered.append(candidate)

    for key in router._get_priorities_config().priorities:
        append_entry(key)
    for key in router._models_cache:
        append_entry(key)
    return ordered


def auto_candidate_for_id(
    router: Any,
    model_id: str,
    *,
    manifest: RequestManifest | None,
    strict_unknown: bool,
) -> GenerationCandidate | None:
    """Resolve one configured Auto model against hard request requirements."""
    entry = router._models_cache.get(model_id)
    if entry is None or not router._is_qualified_cache_entry(model_id, entry):
        return None
    candidate = _catalog_candidate(router, entry[0], entry[1])
    if candidate is None:
        return None
    if manifest is not None and not model_supports_manifest(
        candidate.info.feature_capabilities,
        manifest,
        strict_unknown=strict_unknown,
    ):
        return None
    if manifest is not None:
        requested_output = manifest.max_output_tokens
        if (
            requested_output is not None
            and candidate.info.max_output_tokens is not None
            and candidate.info.max_output_tokens < requested_output
        ):
            return None
    return candidate


def select_auto_context_candidates(
    candidates: tuple[GenerationCandidate, ...],
    estimated_input_tokens: int | None,
) -> tuple[GenerationCandidate, ...]:
    """Apply Auto's 70% context preference without turning it into rejection.

    Models below the safety threshold are omitted while at least one safer
    configured model exists. If every hard-compatible candidate is at or over
    its threshold, retain every model tied for the largest advertised prompt
    window so Auto can still attempt the request.
    """
    safe = tuple(
        candidate
        for candidate in candidates
        if auto_context_is_safe(candidate.info, estimated_input_tokens)
    )
    if safe or not candidates:
        return safe

    capacities = tuple(model_prompt_capacity(candidate.info) for candidate in candidates)
    if any(capacity is None for capacity in capacities):
        # Unknown is not safely smaller than any advertised limit. Keep every
        # unknown-capacity model rather than inventing a hard ceiling for it.
        return tuple(
            candidate
            for candidate, capacity in zip(candidates, capacities, strict=True)
            if capacity is None
        )
    known_capacities = tuple(capacity for capacity in capacities if capacity is not None)
    largest = max(known_capacities)
    return tuple(
        candidate
        for candidate, capacity in zip(candidates, capacities, strict=True)
        if capacity == largest
    )


def configured_auto_model_ids(config: Any) -> tuple[str, ...]:
    """Return the stable execution-model set represented by the Auto profile."""
    auto = config.auto
    if auto.mode is AutoMode.PRIORITY_CHAIN:
        return tuple(auto.priority_chain)
    return tuple(dict.fromkeys(auto.task_router.task_models.values()))


def eligible_auto_candidates(
    router: Any,
    manifest: RequestManifest | None,
) -> tuple[GenerationCandidate, ...]:
    """Resolve configured Auto targets without consulting provider catalog order."""
    config = router._get_priorities_config()
    strict_unknown = (
        config.auto.mode is AutoMode.TASK_ROUTER
        and config.auto.capability_policy is AutoCapabilityPolicy.STRICT
    )
    hard_candidates = tuple(
        candidate
        for model_id in configured_auto_model_ids(config)
        if (
            candidate := auto_candidate_for_id(
                router,
                model_id,
                manifest=manifest,
                strict_unknown=strict_unknown,
            )
        )
        is not None
    )
    return select_auto_context_candidates(
        hard_candidates,
        manifest.estimated_input_tokens if manifest is not None else None,
    )


async def auto_model_info(router: Any) -> ModelInfo:
    """Aggregate the configured Auto execution set into one virtual catalog model."""
    await router._ensure_models_cache()
    config = router._get_priorities_config()
    candidates: list[GenerationCandidate] = []
    seen_ids: set[str] = set()
    for model_id in configured_auto_model_ids(config):
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        candidate = auto_candidate_for_id(
            router,
            model_id,
            manifest=None,
            strict_unknown=False,
        )
        if candidate is None:
            entry = router._models_cache.get(model_id)
            if entry is not None and router._is_qualified_cache_entry(model_id, entry):
                provider = router.providers.get(entry[0])
                if provider is not None:
                    candidate = GenerationCandidate(
                        model=ModelRef(entry[0], entry[1].id),
                        provider=provider,
                        info=deepcopy(entry[1]),
                    )
        if candidate is not None:
            candidates.append(candidate)
    context_by_tier: dict[str, Any] = {}
    default_tiers: set[str] = set()
    for candidate in candidates:
        for option in candidate.info.effective_context_window_options():
            existing = context_by_tier.get(option.tier)
            if existing is None or option.max_prompt_tokens > existing.max_prompt_tokens:
                context_by_tier[option.tier] = option
            if option.is_default:
                default_tiers.add(option.tier)
    selected_default_tier = (
        max(
            default_tiers,
            key=lambda tier: context_by_tier[tier].max_prompt_tokens,
        )
        if default_tiers
        else None
    )
    from router_maestro.providers.base import ContextWindowOption

    context_options = tuple(
        ContextWindowOption(
            tier=option.tier,
            max_prompt_tokens=option.max_prompt_tokens,
            is_default=tier == selected_default_tier,
        )
        for tier, option in sorted(
            context_by_tier.items(),
            key=lambda item: item[1].max_prompt_tokens,
        )
    )

    def union_capabilities(attribute: str) -> dict[str, bool]:
        mappings = [getattr(candidate.info, attribute) for candidate in candidates]
        keys = set().union(*mappings)
        result: dict[str, bool] = {}
        for key in keys:
            values = [mapping.get(key) for mapping in mappings]
            if True in values:
                result[key] = True
            elif values and all(value is False for value in values):
                result[key] = False
        return result

    def maximum(attribute: str) -> int | None:
        values = [getattr(candidate.info, attribute) for candidate in candidates]
        known = [value for value in values if value is not None]
        return max(known) if known else None

    return ModelInfo(
        id="router-maestro",
        name="Router-Maestro Auto",
        provider="router-maestro",
        max_prompt_tokens=maximum("max_prompt_tokens"),
        max_output_tokens=maximum("max_output_tokens"),
        max_context_window_tokens=maximum("max_context_window_tokens"),
        context_window_options=context_options,
        supports_thinking=any(candidate.info.supports_thinking for candidate in candidates),
        supports_vision=any(candidate.info.supports_vision for candidate in candidates),
        reasoning_effort_values=_ordered_reasoning_efforts(candidates),
        operation_capabilities=union_capabilities("operation_capabilities"),
        feature_capabilities=union_capabilities("feature_capabilities"),
        transport_capabilities=union_capabilities("transport_capabilities"),
        virtual=True,
    )


def _ordered_reasoning_efforts(
    candidates: list[GenerationCandidate],
) -> list[str] | None:
    """Return the configured models' effort union in semantic strength order."""
    values = {
        effort
        for candidate in candidates
        for effort in candidate.info.reasoning_effort_values or ()
    }
    if not values:
        return None
    order = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    rank = {effort: index for index, effort in enumerate(order)}
    return sorted(values, key=lambda effort: (rank.get(effort, len(rank)), effort))


async def list_models_with_auto(router: Any) -> list[ModelInfo]:
    """Return the virtual Auto model followed by the real provider catalog."""
    from router_maestro.routing.router import Router

    if not isinstance(router, Router):
        return await router.list_models()
    return [await auto_model_info(router), *(await router.list_models())]


def _invalid_model(message: str, *, cause: BaseException | None = None) -> ProviderError:
    return ProviderError(
        message,
        status_code=400,
        retryable=False,
        kind=ProviderFailureKind.CLIENT_REQUEST,
        cause=cause,
    )


async def plan_generation_route(
    router: Router,
    model_id: str,
    manifest: RequestManifest | None = None,
) -> GenerationRoutePlan:
    """Resolve a model-only plan while preserving existing priority semantics.

    The fallback limit counts only transitions to a different ``ModelRef``.
    Endpoint retries within a candidate are owned by the dispatcher and never
    consume this budget.
    """
    if not isinstance(model_id, str) or not model_id.strip() or model_id != model_id.strip():
        raise _invalid_model("Model ID must be a non-empty provider/model identity")

    await router._ensure_models_cache()
    model_id = router._normalize_model_alias(model_id)
    ordered = _ordered_catalog_candidates(router)
    if not ordered:
        raise ProviderError(
            "No models available for routing",
            status_code=503,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_STATUS,
        )

    config = router._get_priorities_config()
    configured_ids = set(config.priorities)
    eligible_ordered = [
        candidate
        for candidate in ordered
        if manifest is None
        or model_supports_manifest(candidate.info.feature_capabilities, manifest)
    ]
    eligible_configured = [
        candidate
        for candidate in eligible_ordered
        if candidate.model.qualified_id in configured_ids
    ]

    from router_maestro.routing.router import AUTO_ROUTE_MODEL

    explicit = model_id != AUTO_ROUTE_MODEL and router._is_explicit_model_id(model_id)
    primary: GenerationCandidate | None = None
    remaining: list[GenerationCandidate] = []

    if model_id == AUTO_ROUTE_MODEL:
        auto = config.auto
        if auto.mode is AutoMode.PRIORITY_CHAIN and not auto.priority_chain:
            raise ProviderError(
                "Auto priority chain is empty; configure at least one model",
                status_code=503,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
                parameter="auto.priority_chain",
            )
        candidates = list(eligible_auto_candidates(router, manifest))
        if not candidates:
            raise _invalid_model("No configured Auto model supports the requested capabilities")
        primary, *remaining = candidates
    elif explicit:
        try:
            provider_name, upstream_id, _provider = await router._resolve_provider(model_id)
            ref = ModelRef(provider_name, upstream_id)
        except ValueError as error:
            raise _invalid_model(
                "Model ID must be a non-empty provider/model identity",
                cause=error,
            ) from error
        primary = next((candidate for candidate in ordered if candidate.model == ref), None)
        if primary is None:
            raise ProviderError(
                f"Model '{model_id}' not found",
                status_code=404,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )
        remaining = [
            candidate for candidate in eligible_ordered if candidate.model != primary.model
        ]
    else:
        entry = router._models_cache.get(model_id)
        if entry is not None:
            upstream_id = entry[1].id
        else:
            try:
                matched = fuzzy_match_model(model_id, router._models_cache)
            except AmbiguousModelMatchError as error:
                raise _invalid_model(
                    f"Model alias '{model_id}' is ambiguous; use provider/model",
                    cause=error,
                ) from error
            if matched is None:
                raise ProviderError(
                    f"Model alias '{model_id}' not found in any provider",
                    status_code=404,
                    kind=ProviderFailureKind.CLIENT_REQUEST,
                )
            upstream_id = router._models_cache[matched][1].id
        aliases = [candidate for candidate in ordered if candidate.model.upstream_id == upstream_id]
        if not aliases:
            raise ProviderError(
                f"Model alias '{model_id}' not found in any authenticated provider",
                status_code=404,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )
        aliases = [
            candidate
            for candidate in aliases
            if manifest is None
            or model_supports_manifest(candidate.info.feature_capabilities, manifest)
        ]
        if not aliases:
            raise _invalid_model(f"No models for alias '{model_id}' support the requested features")
        primary, *remaining = aliases

    assert primary is not None
    fallback = config.fallback
    if model_id == AUTO_ROUTE_MODEL and config.auto.mode is AutoMode.PRIORITY_CHAIN:
        selected_fallbacks = tuple(remaining)
        limit = len(selected_fallbacks)
        return GenerationRoutePlan(
            primary=primary,
            fallbacks=selected_fallbacks,
            explicit=False,
            max_model_switches=limit,
        )
    if fallback.strategy is FallbackStrategy.NONE or fallback.maxRetries == 0:
        selected_fallbacks: tuple[GenerationCandidate, ...] = ()
        limit = 0
    else:
        if fallback.strategy is FallbackStrategy.SAME_MODEL:
            remaining = [
                candidate
                for candidate in eligible_ordered
                if candidate.model != primary.model
                and candidate.model.upstream_id == primary.model.upstream_id
            ]
        elif explicit:
            configured_refs = [candidate.model for candidate in eligible_configured]
            if primary.model in configured_refs:
                remaining = eligible_configured[configured_refs.index(primary.model) + 1 :]
            else:
                remaining = eligible_configured
        selected_fallbacks = tuple(remaining[: fallback.maxRetries])
        limit = fallback.maxRetries

    return GenerationRoutePlan(
        primary=primary,
        fallbacks=selected_fallbacks,
        explicit=explicit,
        max_model_switches=limit,
    )


__all__ = [
    "AUTO_CONTEXT_SAFETY_DENOMINATOR",
    "AUTO_CONTEXT_SAFETY_NUMERATOR",
    "GenerationCandidate",
    "GenerationRoutePlan",
    "auto_context_is_safe",
    "auto_candidate_for_id",
    "auto_model_info",
    "configured_auto_model_ids",
    "eligible_auto_candidates",
    "list_models_with_auto",
    "model_prompt_capacity",
    "plan_generation_route",
    "select_auto_context_candidates",
]
