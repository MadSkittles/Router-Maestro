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

from router_maestro.config import FallbackStrategy
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

    configured_ids = set(router._get_priorities_config().priorities)
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
        candidates = eligible_configured or eligible_ordered
        if not candidates:
            raise _invalid_model("No models support the requested features")
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
    fallback = router._get_priorities_config().fallback
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


__all__ = ["GenerationCandidate", "GenerationRoutePlan", "plan_generation_route"]
