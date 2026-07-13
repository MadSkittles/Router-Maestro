"""Routing module for router-maestro."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from router_maestro.routing.router import Router


def get_router() -> Any:
    from router_maestro.routing.router import get_router as _get_router

    return _get_router()


def reset_router() -> None:
    from router_maestro.routing.router import reset_router as _reset_router

    _reset_router()


def __getattr__(name: str) -> Any:
    if name == "Router":
        from router_maestro.routing.router import Router

        return Router
    raise AttributeError(name)


__all__ = ["Router", "get_router", "reset_router"]
