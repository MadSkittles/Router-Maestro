"""Server routes module."""

from router_maestro.server.routes.admin import router as admin_router
from router_maestro.server.routes.anthropic import router as anthropic_router
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.server.routes.models import router as models_router

__all__ = ["admin_router", "anthropic_router", "chat_router", "models_router"]
