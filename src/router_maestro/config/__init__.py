"""Config module for router-maestro."""

from router_maestro.config.contexts import ContextConfig, ContextsConfig
from router_maestro.config.paths import (
    AUTH_FILE,
    CONTEXTS_FILE,
    LOG_FILE,
    PRIORITIES_FILE,
    PROVIDERS_FILE,
    SERVER_CONFIG_FILE,
    get_config_dir,
    get_data_dir,
)
from router_maestro.config.priorities import (
    FallbackConfig,
    FallbackStrategy,
    PrioritiesConfig,
)
from router_maestro.config.providers import (
    CustomProviderConfig,
    ModelConfig,
    ProvidersConfig,
)
from router_maestro.config.server import (
    get_current_context_api_key,
    get_or_create_api_key,
    set_local_api_key,
)
from router_maestro.config.settings import (
    load_contexts_config,
    load_priorities_config,
    load_providers_config,
    save_contexts_config,
    save_priorities_config,
    save_providers_config,
)

__all__ = [
    # Paths
    "get_data_dir",
    "get_config_dir",
    "AUTH_FILE",
    "SERVER_CONFIG_FILE",
    "PROVIDERS_FILE",
    "PRIORITIES_FILE",
    "CONTEXTS_FILE",
    "LOG_FILE",
    # Provider models
    "ModelConfig",
    "CustomProviderConfig",
    "ProvidersConfig",
    # Priority models
    "PrioritiesConfig",
    "FallbackConfig",
    "FallbackStrategy",
    # Context models
    "ContextConfig",
    "ContextsConfig",
    # Settings functions
    "load_providers_config",
    "save_providers_config",
    "load_priorities_config",
    "save_priorities_config",
    "load_contexts_config",
    "save_contexts_config",
    # Server functions
    "get_current_context_api_key",
    "get_or_create_api_key",
    "set_local_api_key",
]
