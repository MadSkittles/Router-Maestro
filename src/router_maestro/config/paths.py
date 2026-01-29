"""File path definitions for router-maestro."""

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Get the data directory for router-maestro.

    Returns ~/.local/share/router-maestro on Unix-like systems.
    Returns %LOCALAPPDATA%/router-maestro on Windows.
    """
    if os.name == "nt":
        # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        # Unix-like (Linux, macOS)
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    data_dir = base / "router-maestro"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_config_dir() -> Path:
    """Get the config directory for router-maestro.

    Returns ~/.config/router-maestro on Unix-like systems.
    Returns %LOCALAPPDATA%/router-maestro on Windows.
    """
    if os.name == "nt":
        # Windows - use same as data dir
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        # Unix-like (Linux, macOS)
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    config_dir = base / "router-maestro"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


# File paths
AUTH_FILE = get_data_dir() / "auth.json"
SERVER_CONFIG_FILE = get_data_dir() / "server.json"
PROVIDERS_FILE = get_config_dir() / "providers.json"
PRIORITIES_FILE = get_config_dir() / "priorities.json"
CONTEXTS_FILE = get_config_dir() / "contexts.json"
LOG_FILE = get_data_dir() / "router-maestro.log"
