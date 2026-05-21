"""Global settings and configuration management."""

import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from router_maestro.config.contexts import ContextsConfig
from router_maestro.config.paths import CONTEXTS_FILE, PRIORITIES_FILE, PROVIDERS_FILE
from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.config.providers import ProvidersConfig

T = TypeVar("T", bound=BaseModel)


def write_json_owner_only(path: Path, data: Any) -> None:
    """Write JSON with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        with suppress(OSError):
            os.close(fd)
        raise

    path.chmod(0o600)


def load_config(path: Path, model: type[T], default_factory: callable) -> T:
    """Load configuration from JSON file.

    Args:
        path: Path to configuration file
        model: Pydantic model class to parse into
        default_factory: Function to create default configuration

    Returns:
        Parsed configuration object
    """
    if not path.exists():
        config = default_factory()
        save_config(path, config)
        return config
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return model.model_validate(data)


def save_config(path: Path, config: BaseModel) -> None:
    """Save configuration to JSON file.

    Args:
        path: Path to configuration file
        config: Configuration object to save
    """
    write_json_owner_only(path, config.model_dump(mode="json"))


def load_providers_config() -> ProvidersConfig:
    """Load providers configuration."""
    return load_config(PROVIDERS_FILE, ProvidersConfig, ProvidersConfig.get_default)


def save_providers_config(config: ProvidersConfig) -> None:
    """Save providers configuration."""
    save_config(PROVIDERS_FILE, config)


def load_priorities_config() -> PrioritiesConfig:
    """Load priorities configuration."""
    return load_config(PRIORITIES_FILE, PrioritiesConfig, PrioritiesConfig.get_default)


def save_priorities_config(config: PrioritiesConfig) -> None:
    """Save priorities configuration."""
    save_config(PRIORITIES_FILE, config)


def load_contexts_config() -> ContextsConfig:
    """Load contexts configuration."""
    return load_config(CONTEXTS_FILE, ContextsConfig, ContextsConfig.get_default)


def save_contexts_config(config: ContextsConfig) -> None:
    """Save contexts configuration."""
    save_config(CONTEXTS_FILE, config)
