"""FastAPI application for router-maestro."""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from router_maestro import __version__
from router_maestro.routing import get_router
from router_maestro.server.middleware import verify_api_key
from router_maestro.server.observability import (
    CONTENT_TYPE_LATEST,
    create_http_metrics,
    render_metrics,
)
from router_maestro.server.routes import (
    admin_router,
    anthropic_router,
    chat_router,
    gemini_router,
    models_router,
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

    # Pre-warm model cache if any providers are authenticated
    router = get_router()
    authenticated_providers = [
        name for name, provider in router.providers.items() if provider.is_authenticated()
    ]
    if authenticated_providers:
        logger.info(
            "Pre-warming model cache for authenticated providers: %s", authenticated_providers
        )
        try:
            models = await router.list_models()
            logger.info("Model cache pre-warmed with %d models", len(models))
        except Exception as e:
            logger.warning("Failed to pre-warm model cache: %s", e)

    yield
    # Shutdown
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

    if provided_token != metrics_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid metrics token",
            headers={"WWW-Authenticate": "Bearer"},
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

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers with API key verification
    app.include_router(chat_router, dependencies=[Depends(verify_api_key)])
    app.include_router(models_router, dependencies=[Depends(verify_api_key)])
    app.include_router(responses_router, dependencies=[Depends(verify_api_key)])
    app.include_router(anthropic_router, dependencies=[Depends(verify_api_key)])
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
