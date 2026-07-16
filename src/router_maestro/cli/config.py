"""`router-maestro config` — generate config files for external CLI clients.

The per-client generation logic lives in :mod:`router_maestro.cli.client_configs`.
This module is the Typer surface: the command bindings, the interactive tool
picker, and a small back-compat re-export block.
"""

from pathlib import Path

import typer
from rich.prompt import Confirm, Prompt

from router_maestro.cli.client_configs import get_client, list_clients
from router_maestro.cli.client_configs.base import console

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context) -> None:
    """Generate configuration for CLI tools (interactive selection if not specified)."""
    if ctx.invoked_subcommand is not None:
        return

    # Interactive selection
    console.print("\n[bold]Available CLI tools:[/bold]")
    clients = list_clients()
    for i, client in enumerate(clients, 1):
        console.print(f"  {i}. {client.display_name} - {client.description}")

    console.print()
    choice = Prompt.ask(
        "Select tool to configure",
        choices=[str(i) for i in range(1, len(clients) + 1)],
        default="1",
    )

    idx = int(choice) - 1
    clients[idx]().generate()


@app.command(name="claude-code")
def claude_code_config() -> None:
    """Generate Claude Code CLI settings.json for router-maestro."""
    get_client("claude-code")().generate()


@app.command(name="codex")
def codex_config() -> None:
    """Generate OpenAI Codex CLI config.toml for router-maestro."""
    get_client("codex")().generate()


@app.command(name="gemini")
def gemini_cli_config() -> None:
    """Generate Gemini CLI .env for router-maestro."""
    get_client("gemini")().generate()


# --- back-compat re-exports -------------------------------------------------
# Keep the private helpers that tests/external callers import from this module
# resolvable after the split into ``client_configs``. ``Prompt``/``Confirm``/
# ``Path`` are re-imported above so ``config.Prompt.ask`` etc. still name the
# shared class objects the client modules use (class-attribute monkeypatch
# seams keep working unchanged).
from router_maestro.cli.client_configs.base import (  # noqa: E402
    _bare_upstream_model_id,
    _display_models,
    _fetch_and_display_models,
    _fetch_models,
    _model_key,
    _select_model,
    _select_model_dict,
)
from router_maestro.cli.client_configs.claude_code import (  # noqa: E402
    _OPUS_1M_NATIVE_KEY,
    _OPUS_1M_SOURCE_MODEL,
    _OPUS_47_1M_NATIVE_KEY,
    _OPUS_48_1M_NATIVE_KEY,
    _SONNET_46_1M_NATIVE_KEY,
    _maybe_inject_opus_1m,
    _prompt_auto_compact_window,
    _prompt_endpoint_mode,
)
from router_maestro.cli.client_configs.gemini import _parse_env_file  # noqa: E402

__all__ = [
    "_OPUS_1M_NATIVE_KEY",
    "_OPUS_1M_SOURCE_MODEL",
    "_OPUS_47_1M_NATIVE_KEY",
    "_OPUS_48_1M_NATIVE_KEY",
    "_SONNET_46_1M_NATIVE_KEY",
    "Confirm",
    "Path",
    "Prompt",
    "_bare_upstream_model_id",
    "_display_models",
    "_fetch_and_display_models",
    "_fetch_models",
    "_maybe_inject_opus_1m",
    "_model_key",
    "_parse_env_file",
    "_prompt_auto_compact_window",
    "_prompt_endpoint_mode",
    "_select_model",
    "_select_model_dict",
    "app",
    "claude_code_config",
    "codex_config",
    "config_callback",
    "console",
    "gemini_cli_config",
]
