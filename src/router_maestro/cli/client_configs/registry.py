"""Ordered registry of supported clients for `router-maestro config`.

Drives both the interactive tool picker and the ``config <key>`` dispatch.
Adding a client is one line here plus its module.
"""

from __future__ import annotations

from router_maestro.cli.client_configs.base import ClientConfig
from router_maestro.cli.client_configs.claude_code import ClaudeCodeConfig
from router_maestro.cli.client_configs.codex import CodexConfig
from router_maestro.cli.client_configs.gemini import GeminiConfig

# Insertion order defines the menu order (claude-code, codex, gemini).
_CLIENT_CLASSES: tuple[type[ClientConfig], ...] = (
    ClaudeCodeConfig,
    CodexConfig,
    GeminiConfig,
)

CLIENT_CONFIGS: dict[str, type[ClientConfig]] = {cls.key: cls for cls in _CLIENT_CLASSES}


def list_clients() -> list[type[ClientConfig]]:
    """Return the client classes in registry (menu) order."""
    return list(_CLIENT_CLASSES)


def get_client(key: str) -> type[ClientConfig]:
    """Return the client class for ``key`` (e.g. ``"codex"``)."""
    return CLIENT_CONFIGS[key]
