"""Routing module for router-maestro."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from router_maestro.routing.generation_plan import (
        GenerationCandidate,
        GenerationRoutePlan,
    )
    from router_maestro.routing.router import Router


def get_router() -> Any:
    from router_maestro.routing.router import get_router as _get_router

    return _get_router()


def reset_router() -> None:
    from router_maestro.routing.router import reset_router as _reset_router

    _reset_router()


async def plan_generation_route(router: Any, model_id: str, manifest: Any | None = None) -> Any:
    """Resolve a provider/model plan without freezing an upstream protocol."""
    from router_maestro.routing.generation_plan import plan_generation_route as _plan

    return await _plan(router, model_id, manifest)


def __getattr__(name: str) -> Any:
    if name == "Router":
        from router_maestro.routing.router import Router

        return Router
    if name in {"GenerationCandidate", "GenerationRoutePlan"}:
        from router_maestro.routing.generation_plan import (
            GenerationCandidate,
            GenerationRoutePlan,
        )

        return {
            "GenerationCandidate": GenerationCandidate,
            "GenerationRoutePlan": GenerationRoutePlan,
        }[name]
    raise AttributeError(name)


__all__ = [
    "GenerationCandidate",
    "GenerationRoutePlan",
    "Router",
    "get_router",
    "plan_generation_route",
    "reset_router",
]
