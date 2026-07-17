"""FastAPI application for router-maestro."""

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exception_handlers import (
    http_exception_handler as default_http_exception_handler,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler as default_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from router_maestro import __version__
from router_maestro.config.repository import RuntimeConfigRepository
from router_maestro.providers import ProviderError
from router_maestro.routing.router import RouterOwner
from router_maestro.runtime import RequestContextMiddleware
from router_maestro.server.middleware import (
    REQUEST_ID_HEADER,
    ObservabilityMiddleware,
    verify_api_key,
)
from router_maestro.server.observability import (
    CONTENT_TYPE_LATEST,
    create_http_metrics,
    render_metrics,
)
from router_maestro.server.protocols.errors import (
    protocol_error_response,
    protocol_surface_for_path,
)
from router_maestro.server.routes import (
    admin_router,
    anthropic_beta_router,
    anthropic_router,
    chat_router,
    gemini_router,
    models_router,
    openai_responses_beta_router,
    responses_router,
)
from router_maestro.utils import get_logger, setup_logging

logger = get_logger("server")
METRICS_TOKEN_ENV = "ROUTER_MAESTRO_METRICS_TOKEN"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup - initialize logging
    log_level = os.environ.get("ROUTER_MAESTRO_LOG_LEVEL", "INFO")
    setup_logging(level=log_level)
    logger.info("Router-Maestro server starting up")

    repository: RuntimeConfigRepository = app.state.runtime_config_repository
    owner: RouterOwner = app.state.router_owner
    try:
        snapshot = repository.read()
        await owner.start(snapshot)
        lease = await owner.acquire()
        try:
            router = lease.router
            authenticated_providers = [
                name for name, provider in router.providers.items() if provider.is_authenticated()
            ]
            if authenticated_providers:
                logger.info(
                    "Pre-warming model cache for authenticated providers: %s",
                    authenticated_providers,
                )
                try:
                    models = await router.list_models()
                    logger.info("Model cache pre-warmed with %d models", len(models))
                except Exception as e:
                    logger.warning("Failed to pre-warm model cache: %s", e)
        finally:
            await lease.release()

        yield
    finally:
        await owner.close()
        logger.info("Router-Maestro server shutting down")


def get_metrics_token() -> str | None:
    """Get the optional metrics endpoint token from the environment."""
    return os.environ.get(METRICS_TOKEN_ENV)


def verify_metrics_access(request: Request) -> None:
    """Verify access to the metrics endpoint when a metrics token is configured."""
    metrics_token = get_metrics_token()
    if not metrics_token:
        return

    auth_header = request.headers.get("Authorization")
    provided_token = None
    if auth_header:
        if auth_header.startswith("Bearer "):
            provided_token = auth_header[7:]
        else:
            provided_token = auth_header

    provided_token_bytes = (provided_token or "").encode("utf-8")
    metrics_token_bytes = metrics_token.encode("utf-8")
    if not hmac.compare_digest(provided_token_bytes, metrics_token_bytes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid metrics token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def unhandled_exception_handler(request: Request, _exc: Exception) -> Response:
    """Encode protocol failures while retaining the existing non-protocol 500."""
    surface = protocol_surface_for_path(request.url.path)
    request_id = getattr(request.state, "request_id", None)
    headers = {REQUEST_ID_HEADER: request_id} if isinstance(request_id, str) else None
    if surface is not None:
        response = protocol_error_response(_exc, surface)
        if headers:
            response.headers.update(headers)
        return response
    return Response(
        content="Internal Server Error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers=headers,
        media_type="text/plain",
    )


async def protocol_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> Response:
    """Dispatch framework HTTP errors by public protocol namespace."""
    surface = protocol_surface_for_path(request.url.path)
    if surface is None:
        return await default_http_exception_handler(request, exc)
    return protocol_error_response(exc, surface)


async def protocol_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    """Dispatch boundary validation failures by public protocol namespace."""
    surface = protocol_surface_for_path(request.url.path)
    if surface is None:
        return await default_validation_exception_handler(request, exc)
    return protocol_error_response(exc, surface)


async def protocol_provider_exception_handler(
    request: Request,
    exc: ProviderError,
) -> Response:
    """Encode an uncaught provider failure for inference endpoints."""
    surface = protocol_surface_for_path(request.url.path)
    if surface is not None:
        return protocol_error_response(exc, surface)
    return Response(
        content="Internal Server Error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        media_type="text/plain",
    )


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Router-Maestro",
        description="Multi-model routing and load balancing with OpenAI-compatible API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.http_metrics = create_http_metrics()
    app.state.runtime_config_repository = RuntimeConfigRepository()
    app.state.router_owner = RouterOwner()
    app.add_exception_handler(RequestValidationError, protocol_validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, protocol_http_exception_handler)
    app.add_exception_handler(ProviderError, protocol_provider_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(ObservabilityMiddleware)

    # Include routers with API key verification
    app.include_router(chat_router, dependencies=[Depends(verify_api_key)])
    app.include_router(models_router, dependencies=[Depends(verify_api_key)])
    app.include_router(responses_router, dependencies=[Depends(verify_api_key)])
    app.include_router(openai_responses_beta_router, dependencies=[Depends(verify_api_key)])
    app.include_router(anthropic_router, dependencies=[Depends(verify_api_key)])
    app.include_router(anthropic_beta_router, dependencies=[Depends(verify_api_key)])
    app.include_router(gemini_router, dependencies=[Depends(verify_api_key)])
    app.include_router(admin_router, dependencies=[Depends(verify_api_key)])

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Router-Maestro",
            "version": __version__,
            "status": "running",
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/metrics")
    async def metrics(request: Request):
        """Prometheus metrics endpoint."""
        verify_metrics_access(request)
        return Response(
            content=render_metrics(app.state.http_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = create_app()
