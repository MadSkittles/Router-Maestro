"""Server configuration management.

API keys are stored in contexts.json under context.
This module provides utilities to manage API keys.
"""

import secrets

from router_maestro.config.contexts import ContextConfig
from router_maestro.config.settings import load_contexts_config, save_contexts_config


def generate_api_key() -> str:
    """Generate a random API key."""
    return f"sk-rm-{secrets.token_urlsafe(32)}"


def get_local_api_key() -> str | None:
    """Get API key for local context.

    Returns:
        The API key if configured, None otherwise.
    """
    config = load_contexts_config()
    local_ctx = config.contexts.get("local")
    if local_ctx:
        return local_ctx.api_key
    return None


def get_current_context_api_key() -> str | None:
    """Get API key for current context.

    Returns:
        The API key if configured, None otherwise.
    """
    config = load_contexts_config()
    ctx_name = config.current
    ctx = config.contexts.get(ctx_name)
    if ctx:
        return ctx.api_key
    return None


def set_local_api_key(api_key: str) -> None:
    """Set API key for local context.

    Args:
        api_key: The API key to set.
    """
    config = load_contexts_config()

    # Ensure local context exists
    if "local" not in config.contexts:
        config.contexts["local"] = ContextConfig(endpoint="http://localhost:8080")

    config.contexts["local"].api_key = api_key
    save_contexts_config(config)


def get_or_create_api_key(api_key: str | None = None) -> tuple[str, bool]:
    """Get or create an API key for local server.

    Priority order:
    1. Provided api_key argument (from CLI --api-key)
    2. ROUTER_MAESTRO_API_KEY environment variable
    3. Existing key in contexts.json
    4. Generate new key

    Args:
        api_key: Optional API key to use.

    Returns:
        Tuple of (api_key, was_generated)
    """
    import os

    if api_key:
        # User provided API key via CLI, save it to local context
        set_local_api_key(api_key)
        return api_key, False

    # Check environment variable
    env_key = os.environ.get("ROUTER_MAESTRO_API_KEY")
    if env_key:
        # Save to local context for persistence
        set_local_api_key(env_key)
        return env_key, False

    # Try to load from local context
    existing_key = get_local_api_key()
    if existing_key:
        return existing_key, False

    # Generate new key and save to local context
    new_key = generate_api_key()
    set_local_api_key(new_key)
    return new_key, True


# Legacy compatibility - ServerConfig is no longer used but kept for reference
class ServerConfig:
    """Deprecated: Server configuration is now stored in contexts.json."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key


def load_server_config() -> ServerConfig:
    """Load server configuration (for backward compatibility).

    Now reads from contexts.json local context.
    """
    api_key = get_local_api_key() or ""
    return ServerConfig(api_key=api_key)
