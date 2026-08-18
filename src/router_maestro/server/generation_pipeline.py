"""Application wiring for the protocol-aware generation dispatcher.

Routes own only their public HTTP shape.  This module assembles the request-
scoped protocol runtime, lazy envelope, provider execution bridge, and response
bridge used by every generation endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from router_maestro.protocols import RequestEnvelope, WireProtocol
from router_maestro.providers.base import ProviderError, ProviderFailureKind
from router_maestro.routing.router import Router
from router_maestro.runtime.reasoning_capsule import ReasoningCapsuleCodec
from router_maestro.server.dispatcher import DispatchAttemptObserver, GenerationDispatcher
from router_maestro.server.legacy_execution import legacy_execution_adapter
from router_maestro.server.protocols.runtime_factory import ProtocolRuntimeFactory
from router_maestro.server.response_bridge import GenerationResponseBridge


@dataclass(frozen=True, slots=True)
class GenerationPipeline:
    """One request's immutable dispatcher wiring and lazy request envelope."""

    dispatcher: GenerationDispatcher
    responses: GenerationResponseBridge
    envelope: RequestEnvelope
    runtimes: ProtocolRuntimeFactory


def build_generation_pipeline(
    router: Router,
    capsule_codec: ReasoningCapsuleCodec,
    protocol: WireProtocol,
    payload: Mapping[str, Any],
    *,
    path: str = "",
    query: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    model: str | None = None,
    stream: bool = False,
    attempt_observer: DispatchAttemptObserver | None = None,
) -> GenerationPipeline:
    """Build the shared pipeline without decoding semantic IR.

    ``model`` and ``stream`` are endpoint context for Gemini, whose wire body
    intentionally omits both values.  Other protocols discover them through
    their shallow request inspector.
    """
    if not isinstance(payload, Mapping):
        raise ProviderError(
            "Request body must be a JSON object",
            status_code=400,
            retryable=False,
            kind=ProviderFailureKind.CLIENT_REQUEST,
        )

    runtimes = ProtocolRuntimeFactory.for_router(router, capsule_codec)
    ingress_runtime = runtimes.ingress(protocol, model=model, stream=stream)
    envelope = RequestEnvelope(
        ingress_runtime,
        payload,
        path=path,
        query=query,
        headers=headers,
        take_ownership=True,
    )
    dispatcher = GenerationDispatcher(
        {},
        execution=legacy_execution_adapter(),
        reasoning_capsule_codec=capsule_codec,
        attempt_observer=attempt_observer,
        runtime_resolver=runtimes.resolve,
    )
    responses = GenerationResponseBridge({}, runtime_resolver=runtimes.resolve)
    return GenerationPipeline(
        dispatcher=dispatcher,
        responses=responses,
        envelope=envelope,
        runtimes=runtimes,
    )


def attempt_observer_for_request(request: object) -> DispatchAttemptObserver | None:
    """Resolve the app-owned low-cardinality attempt observer, when configured."""
    app = getattr(request, "app", None)
    metrics = getattr(getattr(app, "state", None), "http_metrics", None)
    observer = getattr(metrics, "observe_dispatch_attempt", None)
    return cast(DispatchAttemptObserver, observer) if callable(observer) else None


__all__ = [
    "GenerationPipeline",
    "attempt_observer_for_request",
    "build_generation_pipeline",
]
