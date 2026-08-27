"""Loopback FastAPI application for the local Router-Maestro portal."""

from __future__ import annotations

from importlib.resources import files

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from router_maestro import __version__
from router_maestro.web.service import (
    PortalConfigRequest,
    PortalConfigResult,
    PortalContext,
    PortalHealth,
    PortalModel,
    PortalModels,
    PortalProject,
    PortalService,
    PortalServiceError,
)


class AddProjectRequest(BaseModel):
    """Explicit project path submitted by the local portal."""

    path: str


class ApiKeyResponse(BaseModel):
    """Sensitive response used only by the explicit clipboard action."""

    api_key: str


class PortalMeta(BaseModel):
    """Runtime metadata displayed by the local portal."""

    version: str


def _portal_html() -> str:
    return files("router_maestro.web").joinpath("static/index.html").read_text(encoding="utf-8")


def _favicon_svg() -> str:
    return files("router_maestro.web").joinpath("static/favicon.svg").read_text(encoding="utf-8")


def create_portal_app(service: PortalService | None = None) -> FastAPI:
    """Create the local-only web portal application."""
    app = FastAPI(
        title="Router-Maestro Local Portal",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.portal_service = service or PortalService()
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )

    @app.exception_handler(PortalServiceError)
    async def portal_error_handler(
        request: Request,
        error: PortalServiceError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail})

    @app.middleware("http")
    async def secure_local_responses(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(
            _portal_html(),
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                    "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
                ),
            },
        )

    @app.get("/favicon.svg", response_class=Response)
    async def favicon() -> Response:
        return Response(
            _favicon_svg(),
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/contexts", response_model=list[PortalContext])
    async def contexts() -> list[PortalContext]:
        return app.state.portal_service.list_contexts()

    @app.get("/api/meta", response_model=PortalMeta)
    async def meta() -> PortalMeta:
        return PortalMeta(version=__version__)

    @app.get("/api/contexts/{context_name}/health", response_model=PortalHealth)
    async def health(context_name: str) -> PortalHealth:
        return await app.state.portal_service.health(context_name)

    @app.get("/api/contexts/{context_name}/models", response_model=PortalModels)
    async def models(
        context_name: str,
        refresh: bool = Query(default=False),
    ) -> PortalModels:
        return await app.state.portal_service.list_models(
            context_name,
            force_refresh=refresh,
        )

    @app.get("/api/contexts/{context_name}/key", response_model=ApiKeyResponse)
    async def api_key(context_name: str) -> ApiKeyResponse:
        return ApiKeyResponse(api_key=app.state.portal_service.get_api_key(context_name))

    @app.get("/api/projects", response_model=list[PortalProject])
    async def projects() -> list[PortalProject]:
        return app.state.portal_service.list_projects()

    @app.post("/api/projects", response_model=PortalProject)
    async def add_project(request: AddProjectRequest) -> PortalProject:
        return app.state.portal_service.add_project(request.path)

    @app.post("/api/config/preview", response_model=PortalConfigResult)
    async def preview_config(request: PortalConfigRequest) -> PortalConfigResult:
        return await app.state.portal_service.preview_config(request)

    @app.post("/api/config/apply", response_model=PortalConfigResult)
    async def apply_config(request: PortalConfigRequest) -> PortalConfigResult:
        return await app.state.portal_service.apply_config(request)

    return app


__all__ = [
    "AddProjectRequest",
    "ApiKeyResponse",
    "PortalMeta",
    "PortalConfigRequest",
    "PortalConfigResult",
    "PortalContext",
    "PortalHealth",
    "PortalModel",
    "PortalModels",
    "PortalProject",
    "create_portal_app",
]
