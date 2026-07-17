"""Immutable capability-aware route and prepared-stream state."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from router_maestro.routing.capabilities import (
    CapabilitySupport,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef

if TYPE_CHECKING:
    from router_maestro.providers.base import ChatRequest, ResponsesRequest


class NoCompatibleRouteError(Exception):
    """No authenticated catalog candidate can perform the requested operation."""


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    model: ModelRef
    provider: Any
    capabilities: ModelCapabilities
    evaluated_operation: Operation
    evaluated_features: RequestFeatures
    support: CapabilitySupport

    def __post_init__(self) -> None:
        if self.model != self.capabilities.model:
            raise ValueError("route candidate model must match its capability model")
        if getattr(self.provider, "name", None) != self.model.provider:
            raise ValueError("route candidate provider must match its model provider")
        evaluated_support = self.capabilities.support_for(
            self.evaluated_operation,
            self.evaluated_features,
        )
        if self.support is not evaluated_support:
            raise ValueError("route candidate support must match its capability evaluation")


@dataclass(frozen=True, slots=True)
class RoutePlan:
    operation: Operation
    features: RequestFeatures
    primary: RouteCandidate
    fallbacks: tuple[RouteCandidate, ...]
    explicit: bool
    fallback_pool: tuple[RouteCandidate, ...] | None = None
    max_fallback_attempts: int | None = None

    def __post_init__(self) -> None:
        """Normalize legacy construction and reject incoherent execution state."""
        fallbacks = tuple(self.fallbacks)
        fallback_pool = fallbacks if self.fallback_pool is None else tuple(self.fallback_pool)
        fallback_limit = (
            len(fallbacks) if self.max_fallback_attempts is None else self.max_fallback_attempts
        )
        object.__setattr__(self, "fallbacks", fallbacks)
        object.__setattr__(self, "fallback_pool", fallback_pool)
        object.__setattr__(self, "max_fallback_attempts", fallback_limit)

        if fallback_limit < 0:
            raise ValueError("max_fallback_attempts cannot be negative")
        if len(fallbacks) > fallback_limit:
            raise ValueError("fallbacks exceed max_fallback_attempts")

        candidates = (self.primary, *fallback_pool, *fallbacks)
        for candidate in candidates:
            if candidate.evaluated_features != self.features:
                raise ValueError("route candidate features must match the route plan")
            if not self._operation_matches(candidate):
                raise ValueError("route candidate operation must match the route plan")

        primary_model = self.primary.model
        if any(candidate.model == primary_model for candidate in fallback_pool):
            raise ValueError("primary cannot appear in fallback_pool")
        if any(candidate.model == primary_model for candidate in fallbacks):
            raise ValueError("primary cannot appear in fallbacks")

        for name, candidate_group in (
            ("fallback_pool", fallback_pool),
            ("fallbacks", fallbacks),
        ):
            models = [candidate.model for candidate in candidate_group]
            if len(models) != len(set(models)):
                raise ValueError(f"{name} cannot contain duplicate model references")

        pool_iterator = iter(fallback_pool)
        if not all(any(candidate == pooled for pooled in pool_iterator) for candidate in fallbacks):
            raise ValueError("fallbacks must be an ordered subsequence of fallback_pool")

    def _operation_matches(self, candidate: RouteCandidate) -> bool:
        return candidate.evaluated_operation is self.operation

    @property
    def candidates(self) -> tuple[RouteCandidate, ...]:
        return (self.primary, *self.fallbacks)

    @property
    def prevalidation_fallbacks(self) -> tuple[RouteCandidate, ...]:
        """Return the frozen pool from which executable fallbacks are selected."""
        assert self.fallback_pool is not None
        return self.fallback_pool

    @property
    def fallback_limit(self) -> int:
        """Return the frozen retry limit, preserving manually constructed plans."""
        assert self.max_fallback_attempts is not None
        return self.max_fallback_attempts


@dataclass(frozen=True, slots=True)
class PreparedChatStream:
    """A Chat stream plan bound to the exact request that passed validation."""

    plan: RoutePlan
    _request_snapshot: ChatRequest = field(repr=False)
    _candidate_request_snapshots: tuple[tuple[ModelRef, ChatRequest], ...] = field(
        default=(),
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.plan.operation is not Operation.CHAT_STREAM:
            raise ValueError("PreparedChatStream requires a chat_stream plan")
        object.__setattr__(self, "_request_snapshot", deepcopy(self._request_snapshot))
        object.__setattr__(
            self,
            "_candidate_request_snapshots",
            tuple(
                (model, deepcopy(request)) for model, request in self._candidate_request_snapshots
            ),
        )

    @classmethod
    def capture(
        cls,
        plan: RoutePlan,
        request: ChatRequest,
        candidate_requests: Mapping[ModelRef, ChatRequest] | None = None,
    ) -> PreparedChatStream:
        """Capture an isolated snapshot after the request and plan validate."""
        return cls(
            plan=plan,
            _request_snapshot=request,
            _candidate_request_snapshots=tuple((candidate_requests or {}).items()),
        )

    def matches(self, request: ChatRequest) -> bool:
        """Return whether the caller supplied the exact validated request value."""
        return request == self._request_snapshot

    def request_for_execution(self, model: ModelRef | None = None) -> ChatRequest:
        """Return an isolated copy of the request that actually passed validation."""
        if model is not None:
            for candidate_model, request in self._candidate_request_snapshots:
                if candidate_model == model:
                    return deepcopy(request)
        return deepcopy(self._request_snapshot)


@dataclass(frozen=True, slots=True)
class PreparedChatCompletion:
    """A non-stream Chat plan bound to the exact request that passed validation."""

    plan: RoutePlan
    _request_snapshot: ChatRequest = field(repr=False)
    _candidate_request_snapshots: tuple[tuple[ModelRef, ChatRequest], ...] = field(
        default=(),
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.plan.operation is not Operation.CHAT:
            raise ValueError("PreparedChatCompletion requires a chat plan")
        object.__setattr__(self, "_request_snapshot", deepcopy(self._request_snapshot))
        object.__setattr__(
            self,
            "_candidate_request_snapshots",
            tuple(
                (model, deepcopy(request)) for model, request in self._candidate_request_snapshots
            ),
        )

    @classmethod
    def capture(
        cls,
        plan: RoutePlan,
        request: ChatRequest,
        candidate_requests: Mapping[ModelRef, ChatRequest] | None = None,
    ) -> PreparedChatCompletion:
        return cls(
            plan=plan,
            _request_snapshot=request,
            _candidate_request_snapshots=tuple((candidate_requests or {}).items()),
        )

    def matches(self, request: ChatRequest) -> bool:
        return request == self._request_snapshot

    def request_for_execution(self, model: ModelRef | None = None) -> ChatRequest:
        if model is not None:
            for candidate_model, request in self._candidate_request_snapshots:
                if candidate_model == model:
                    return deepcopy(request)
        return deepcopy(self._request_snapshot)


@dataclass(frozen=True, slots=True)
class PreparedResponsesStream:
    """A Responses stream plan bound to the exact request that passed validation."""

    plan: RoutePlan
    _request_snapshot: ResponsesRequest = field(repr=False)

    def __post_init__(self) -> None:
        if self.plan.operation is not Operation.RESPONSES_STREAM:
            raise ValueError("PreparedResponsesStream requires a responses_stream plan")
        object.__setattr__(self, "_request_snapshot", deepcopy(self._request_snapshot))

    @classmethod
    def capture(
        cls,
        plan: RoutePlan,
        request: ResponsesRequest,
    ) -> PreparedResponsesStream:
        """Capture an isolated snapshot after the request and plan validate."""
        return cls(plan=plan, _request_snapshot=request)

    def matches(self, request: ResponsesRequest) -> bool:
        """Return whether the caller supplied the exact validated request value."""
        return request == self._request_snapshot

    def request_for_execution(self) -> ResponsesRequest:
        """Return an isolated copy of the request that actually passed validation."""
        return deepcopy(self._request_snapshot)
