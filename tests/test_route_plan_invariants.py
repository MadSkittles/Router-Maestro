"""Construction invariants for immutable route plans."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from router_maestro.routing.capabilities import (
    CapabilitySupport,
    Feature,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan


def _candidate(provider: str, model: str) -> RouteCandidate:
    ref = ModelRef(provider, model)
    operation = Operation.CHAT
    features = RequestFeatures()
    capabilities = ModelCapabilities(
        model=ref,
        operations={operation: CapabilitySupport.SUPPORTED},
    )
    return RouteCandidate(
        model=ref,
        provider=SimpleNamespace(name=provider),
        capabilities=capabilities,
        evaluated_operation=operation,
        evaluated_features=features,
        support=CapabilitySupport.SUPPORTED,
    )


def _plan(
    primary: RouteCandidate,
    fallbacks: tuple[RouteCandidate, ...] = (),
    *,
    fallback_pool: tuple[RouteCandidate, ...] | None = None,
    max_fallback_attempts: int | None = None,
) -> RoutePlan:
    return RoutePlan(
        operation=Operation.CHAT,
        features=RequestFeatures(),
        primary=primary,
        fallbacks=fallbacks,
        explicit=False,
        fallback_pool=fallback_pool,
        max_fallback_attempts=max_fallback_attempts,
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda primary, _one, _two: _plan(
                primary,
                fallback_pool=(),
                max_fallback_attempts=-1,
            ),
            id="negative-fallback-limit",
        ),
        pytest.param(
            lambda primary, one, _two: _plan(
                primary,
                (one,),
                fallback_pool=(one,),
                max_fallback_attempts=0,
            ),
            id="fallback-count-exceeds-limit",
        ),
        pytest.param(
            lambda primary, one, two: _plan(
                primary,
                (two, one),
                fallback_pool=(one, two),
                max_fallback_attempts=2,
            ),
            id="fallbacks-not-ordered-pool-subsequence",
        ),
        pytest.param(
            lambda primary, one, _two: _plan(
                primary,
                (one,),
                fallback_pool=(),
                max_fallback_attempts=1,
            ),
            id="nonempty-fallbacks-with-empty-pool",
        ),
        pytest.param(
            lambda primary, _one, _two: _plan(
                primary,
                fallback_pool=(primary,),
                max_fallback_attempts=0,
            ),
            id="primary-in-fallback-pool",
        ),
        pytest.param(
            lambda primary, _one, _two: _plan(
                primary,
                (primary,),
                fallback_pool=(primary,),
                max_fallback_attempts=1,
            ),
            id="primary-in-fallbacks",
        ),
        pytest.param(
            lambda primary, one, _two: _plan(
                primary,
                (one,),
                fallback_pool=(one, replace(one)),
                max_fallback_attempts=1,
            ),
            id="duplicate-model-ref-in-pool",
        ),
    ],
)
def test_route_plan_rejects_incoherent_fallback_state(build) -> None:
    primary = _candidate("primary", "model")
    one = _candidate("one", "model")
    two = _candidate("two", "model")

    with pytest.raises(ValueError):
        build(primary, one, two)


def test_route_plan_rejects_candidate_capability_identity_mismatch() -> None:
    with pytest.raises(ValueError):
        RouteCandidate(
            model=ModelRef("fallback", "selected-model"),
            provider=SimpleNamespace(name="fallback"),
            capabilities=ModelCapabilities(model=ModelRef("fallback", "different-model")),
            evaluated_operation=Operation.CHAT,
            evaluated_features=RequestFeatures(),
            support=CapabilitySupport.UNKNOWN,
        )


@pytest.mark.parametrize(
    ("capabilities", "features"),
    [
        pytest.param(
            ModelCapabilities(
                model=ModelRef("primary", "model"),
                operations={Operation.CHAT: CapabilitySupport.UNSUPPORTED},
            ),
            RequestFeatures(),
            id="operation",
        ),
        pytest.param(
            ModelCapabilities(
                model=ModelRef("primary", "model"),
                operations={Operation.CHAT: CapabilitySupport.SUPPORTED},
                features={Feature.TOOLS: CapabilitySupport.UNSUPPORTED},
            ),
            RequestFeatures(tools=True),
            id="required-feature",
        ),
    ],
)
def test_route_candidate_rejects_support_inconsistent_with_evaluation_provenance(
    capabilities: ModelCapabilities,
    features: RequestFeatures,
) -> None:
    ref = ModelRef("primary", "model")

    with pytest.raises(ValueError, match="support"):
        RouteCandidate(
            model=ref,
            provider=SimpleNamespace(name="primary"),
            capabilities=capabilities,
            evaluated_operation=Operation.CHAT,
            evaluated_features=features,
            support=CapabilitySupport.SUPPORTED,
        )


def test_route_candidate_rejects_provider_name_inconsistent_with_model_ref() -> None:
    ref = ModelRef("declared", "model")
    capabilities = ModelCapabilities(
        model=ref,
        operations={Operation.CHAT: CapabilitySupport.SUPPORTED},
    )

    with pytest.raises(ValueError, match="provider"):
        RouteCandidate(
            model=ref,
            provider=SimpleNamespace(name="actual"),
            capabilities=capabilities,
            evaluated_operation=Operation.CHAT,
            evaluated_features=RequestFeatures(),
            support=CapabilitySupport.SUPPORTED,
        )


def test_legacy_five_argument_route_plan_normalizes_fallback_contract() -> None:
    primary = _candidate("primary", "model")
    fallback = _candidate("fallback", "model")

    plan = RoutePlan(
        Operation.CHAT,
        RequestFeatures(),
        primary,
        (fallback,),
        False,
    )

    assert plan.fallback_pool == (fallback,)
    assert plan.max_fallback_attempts == 1
    assert plan.prevalidation_fallbacks == (fallback,)
    assert plan.fallback_limit == 1


def test_normalized_route_plan_remains_frozen() -> None:
    primary = _candidate("primary", "model")
    fallback = _candidate("fallback", "model")
    plan = RoutePlan(
        Operation.CHAT,
        RequestFeatures(),
        primary,
        (fallback,),
        False,
    )

    with pytest.raises(FrozenInstanceError):
        plan.max_fallback_attempts = 2  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        plan.fallback_pool = ()  # type: ignore[misc]


def test_route_plan_defensively_freezes_mutable_candidate_sequences() -> None:
    primary = _candidate("primary", "model")
    first = _candidate("first", "model")
    injected = _candidate("injected", "model")
    fallbacks = [first]
    fallback_pool = [first]

    plan = RoutePlan(
        Operation.CHAT,
        RequestFeatures(),
        primary,
        fallbacks,  # type: ignore[arg-type]
        False,
        fallback_pool=fallback_pool,  # type: ignore[arg-type]
        max_fallback_attempts=1,
    )
    fallbacks.append(injected)
    fallback_pool.append(injected)

    assert plan.fallbacks == (first,)
    assert plan.prevalidation_fallbacks == (first,)
    assert plan.candidates == (primary, first)
    assert plan.fallback_limit == 1
