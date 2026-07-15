"""Application-owned dependencies shared by server routes."""

from collections.abc import AsyncIterator

from fastapi import Depends, Request

from router_maestro.config.repository import RuntimeConfigRepository
from router_maestro.routing.router import Router, RouterOwner


def get_router_owner(request: Request) -> RouterOwner:
    """Return the application-owned Router generation manager."""
    return request.app.state.router_owner


def get_runtime_config_repository(request: Request) -> RuntimeConfigRepository:
    """Return the application-owned runtime configuration repository."""
    return request.app.state.runtime_config_repository


async def get_app_router(
    owner: RouterOwner = Depends(get_router_owner),
) -> AsyncIterator[Router]:
    """Yield the current app-owned Router generation under a request lease."""
    lease = await owner.acquire()
    try:
        yield lease.router
    finally:
        await lease.release()
