"""Global settings and configuration management."""

import json
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from router_maestro.config.contexts import ContextsConfig
from router_maestro.config.paths import CONTEXTS_FILE, PRIORITIES_FILE, PROVIDERS_FILE
from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.config.providers import (
    RESERVED_PROVIDER_NAMES,
    CustomProviderConfig,
    ProvidersConfig,
)
from router_maestro.routing.model_ref import validate_provider_id

logger = logging.getLogger("router_maestro.config.settings")


def _safe_provider_validation_location(location: tuple[str | int, ...]) -> str:
    """Format a schema path without exposing dynamic model IDs or control characters."""
    parts: list[str] = []
    redact_model_id = False
    for part in location:
        if redact_model_id:
            parts.append("<model-id>")
            redact_model_id = False
            continue
        rendered = str(part)
        parts.append(rendered if rendered.isidentifier() else repr(rendered))
        redact_model_id = rendered == "models"
    return ".".join(parts) or "provider"


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
        # ours to close (the except below handles it). os.fchmod is POSIX-only
        # and absent on Windows, where mkstemp already restricts the file to the
        # creating user and the POSIX permission bits do not apply.
        if hasattr(os, "fchmod"):
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


def load_config[T: BaseModel](path: Path, model: type[T], default_factory: callable) -> T:
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
    except ValidationError:
        logger.error(
            "Failed to validate config from %s as %s; falling back to defaults",
            path,
            model.__name__,
        )
        return default_factory()
    except (json.JSONDecodeError, OSError) as e:
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
    """Load custom providers independently while preserving compatible entries."""
    if not PROVIDERS_FILE.exists():
        config = ProvidersConfig.get_default()
        save_config(PROVIDERS_FILE, config)
        return config

    try:
        with open(PROVIDERS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        logger.error(
            "Failed to load providers config from %s (invalid_encoding); falling back to defaults",
            PROVIDERS_FILE,
        )
        return ProvidersConfig.get_default()
    except json.JSONDecodeError:
        logger.error(
            "Failed to load providers config from %s (invalid_json); falling back to defaults",
            PROVIDERS_FILE,
        )
        return ProvidersConfig.get_default()
    except OSError:
        logger.error(
            "Failed to load providers config from %s (os_error); falling back to defaults",
            PROVIDERS_FILE,
        )
        return ProvidersConfig.get_default()

    if not isinstance(data, dict):
        logger.error(
            "Failed to validate providers config from %s at root (mapping_type); "
            "falling back to defaults",
            PROVIDERS_FILE,
        )
        return ProvidersConfig.get_default()

    raw_providers = data.get("providers", {})
    if not isinstance(raw_providers, dict):
        logger.error(
            "Failed to validate providers config from %s at providers (mapping_type); "
            "falling back to defaults",
            PROVIDERS_FILE,
        )
        return ProvidersConfig.get_default()

    providers: dict[str, CustomProviderConfig] = {}
    canonical_names: dict[str, str] = {}
    for provider_name, raw_provider in raw_providers.items():
        try:
            validate_provider_id(provider_name)
        except (TypeError, ValueError):
            logger.warning(
                "Skipping custom provider %r: provider_id:invalid_provider_name",
                provider_name,
            )
            continue

        canonical_name = provider_name.casefold()
        if canonical_name in RESERVED_PROVIDER_NAMES:
            logger.warning(
                "Skipping custom provider %r: provider_id:reserved_provider_name",
                provider_name,
            )
            continue
        duplicate = canonical_names.get(canonical_name)
        if duplicate is not None:
            logger.warning(
                "Skipping custom provider %r: provider_id:duplicate_provider_name "
                "(already loaded as %r)",
                provider_name,
                duplicate,
            )
            continue

        try:
            provider = CustomProviderConfig.model_validate(raw_provider)
        except ValidationError as error:
            diagnostics = []
            for detail in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            ):
                location = _safe_provider_validation_location(detail["loc"])
                diagnostics.append(f"{location}:{detail['type']}")
            logger.warning(
                "Skipping custom provider %r: validation failed at %s",
                provider_name,
                ", ".join(diagnostics),
            )
            continue

        providers[provider_name] = provider
        canonical_names[canonical_name] = provider_name
        legacy_options = sorted((provider.options.model_extra or {}).keys())
        if legacy_options:
            logger.warning(
                "Custom provider %r retains ignored legacy option keys: %s",
                provider_name,
                ", ".join(repr(option) for option in legacy_options),
            )

    return ProvidersConfig(providers=providers)


def save_providers_config(config: ProvidersConfig) -> None:
    """Save providers configuration."""
    save_config(PROVIDERS_FILE, config)


def load_priorities_config() -> PrioritiesConfig:
    """Load priorities configuration."""
    from router_maestro.config.repository import RuntimeConfigRepository

    return RuntimeConfigRepository(PRIORITIES_FILE).read().config


def save_priorities_config(config: PrioritiesConfig) -> None:
    """Save priorities configuration."""
    from router_maestro.config.repository import RuntimeConfigRepository

    RuntimeConfigRepository(PRIORITIES_FILE).write_compat(config)


def load_contexts_config() -> ContextsConfig:
    """Load contexts configuration."""
    return load_config(CONTEXTS_FILE, ContextsConfig, ContextsConfig.get_default)


def save_contexts_config(config: ContextsConfig) -> None:
    """Save contexts configuration."""
    save_config(CONTEXTS_FILE, config)
