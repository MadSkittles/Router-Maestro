"""Application-owned dependencies shared by server routes."""

from collections.abc import AsyncIterator

from fastapi import Depends, Request

from router_maestro.config.repository import RuntimeConfigRepository
from router_maestro.routing.router import Router, RouterOwner
from router_maestro.runtime import ReasoningCapsuleCodec


def get_router_owner(request: Request) -> RouterOwner:
    """Return the application-owned Router generation manager."""
    return request.app.state.router_owner


def get_runtime_config_repository(request: Request) -> RuntimeConfigRepository:
    """Return the application-owned runtime configuration repository."""
    return request.app.state.runtime_config_repository


def get_reasoning_capsule_codec(request: Request) -> ReasoningCapsuleCodec:
    """Return the process-wide reasoning capsule keyring loaded at startup."""
    codec = getattr(request.app.state, "reasoning_capsule_codec", None)
    if not isinstance(codec, ReasoningCapsuleCodec):
        raise RuntimeError("reasoning capsule codec is not initialized")
    return codec


def generation_dispatcher_is_configured(request: Request | None) -> bool:
    """Whether this FastAPI app was wired for protocol generation dispatch.

    ``create_app`` installs the state slot before lifespan startup.  Checking
    for that slot, rather than for a non-null codec, keeps bare compatibility
    routers usable in direct tests while making a missing production keyring a
    startup/wiring error instead of silently selecting the legacy route.
    """
    app = getattr(request, "app", None)
    state = getattr(app, "state", None)
    return state is not None and hasattr(state, "reasoning_capsule_codec")


async def get_app_router(
    owner: RouterOwner = Depends(get_router_owner),
) -> AsyncIterator[Router]:
    """Yield the current app-owned Router generation under a request lease."""
    lease = await owner.acquire()
    try:
        yield lease.router
    finally:
        await lease.release()
