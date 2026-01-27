"""Global settings and configuration management."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from router_maestro.config.contexts import ContextsConfig
from router_maestro.config.paths import CONTEXTS_FILE, PRIORITIES_FILE, PROVIDERS_FILE
from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.config.providers import ProvidersConfig

T = TypeVar("T", bound=BaseModel)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, ensure_ascii=False)


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
