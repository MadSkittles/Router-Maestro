"""Per-client config generation for `router-maestro config`."""

from router_maestro.cli.client_configs.base import (
    ClientConfig,
    GenerateContext,
    IdStyle,
)
from router_maestro.cli.client_configs.registry import (
    CLIENT_CONFIGS,
    get_client,
    list_clients,
)

__all__ = [
    "CLIENT_CONFIGS",
    "ClientConfig",
    "GenerateContext",
    "IdStyle",
    "get_client",
    "list_clients",
]
