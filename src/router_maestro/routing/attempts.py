"""Per-request fallback attempt records and decisions."""

from __future__ import annotations

from dataclasses import dataclass

from router_maestro.providers.base import ProviderError, ProviderFailureKind
from router_maestro.routing.capabilities import CapabilitySupport, Operation
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Safe metadata for one failed provider attempt."""

    provider: str
    model: ModelRef
    operation: Operation
    downstream_status_code: int
    upstream_status_code: int | None
    failure_kind: ProviderFailureKind
    retryable: bool

    @classmethod
    def from_failure(
        cls,
        candidate: RouteCandidate,
        operation: Operation,
        error: ProviderError,
    ) -> AttemptRecord:
        """Create a record without retaining the exception or its cause."""
        return cls(
            provider=candidate.model.provider,
            model=candidate.model,
            operation=operation,
            downstream_status_code=error.downstream_status_code,
            upstream_status_code=error.upstream_status_code,
            failure_kind=error.kind,
            retryable=error.retryable,
        )


class AttemptLedger:
    """Request-local ordered collection of failed attempts."""

    def __init__(self) -> None:
        self._records: list[AttemptRecord] = []

    def record(self, attempt: AttemptRecord) -> None:
        self._records.append(attempt)

    @property
    def records(self) -> tuple[AttemptRecord, ...]:
        return tuple(self._records)

    def snapshot(self) -> tuple[AttemptRecord, ...]:
        return self.records


def failure_allows_fallback(
    plan: RoutePlan,
    candidate: RouteCandidate,
    error: ProviderError,
) -> bool:
    """Return whether one pre-commit failure permits the next planned candidate."""
    if error.kind is ProviderFailureKind.UNSUPPORTED_OPERATION:
        return not plan.explicit and candidate.support is CapabilitySupport.UNKNOWN
    return error.retryable
