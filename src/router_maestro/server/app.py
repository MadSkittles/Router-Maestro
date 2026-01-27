"""FastAPI application for router-maestro."""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router_maestro import __version__
from router_maestro.routing import get_router
from router_maestro.server.middleware import verify_api_key
from router_maestro.server.routes import admin_router, anthropic_router, chat_router, models_router
from router_maestro.utils import get_logger, setup_logging

logger = get_logger("server")


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


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Router-Maestro",
        description="Multi-model routing and load balancing with OpenAI-compatible API",
        version=__version__,
        lifespan=lifespan,
    )

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
    app.include_router(anthropic_router, dependencies=[Depends(verify_api_key)])
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

    return app


app = create_app()
