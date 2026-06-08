"""Global settings and configuration management."""

import json
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from router_maestro.config.contexts import ContextsConfig
from router_maestro.config.paths import CONTEXTS_FILE, PRIORITIES_FILE, PROVIDERS_FILE
from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.config.providers import ProvidersConfig

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("router_maestro.config.settings")


def write_json_owner_only(path: Path, data: Any) -> None:
    """Write JSON with owner-only permissions, atomically.

    Writes to a temporary file in the same directory and renames it into place,
    so a crash mid-write cannot truncate or corrupt the destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    fdopen_took_fd = False
    try:
        # fchmod before fdopen takes ownership of fd; if it raises, fd is still
        # ours to close (the except below handles it).
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fdopen_took_fd = True
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if not fdopen_took_fd:
            with suppress(OSError):
                os.close(fd)
        with suppress(OSError):
            tmp_path.unlink()
        raise


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
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return model.model_validate(data)
    except (json.JSONDecodeError, ValidationError, OSError) as e:
        logger.error("Failed to load config from %s (%s); falling back to defaults", path, e)
        return default_factory()


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
