"""Shared foundation for per-client config generation.

Each external client (Claude Code, Codex, Gemini CLI) subclasses
:class:`ClientConfig` and owns its *entire* generation flow via the template
method :meth:`ClientConfig.generate`. This module holds the pieces every client
shares: model fetch/display/selection, the backup prompt, the level picker, the
base-URL/auth resolvers, and the model-id-style resolution.

Dependency rule: ``cli/config.py`` imports from this package; nothing in this
package imports ``cli/config.py`` (one-way, no circular import).
"""

from __future__ import annotations

import asyncio
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from router_maestro.cli.client import ServerNotRunningError, get_admin_client
from router_maestro.config.server import get_current_context_api_key
from router_maestro.routing.model_ref import qualify_model_id

console = Console()


class IdStyle(StrEnum):
    """How a selected model is spelled in the generated config.

    ``QUALIFIED`` writes the internal ``provider/upstream-id`` (the default,
    unambiguous across providers). ``OFFICIAL`` writes the vendor's native id
    (recognized directly by the client/TUI). Official conversion is wired up in
    Part B; Part A always resolves to ``QUALIFIED``.
    """

    QUALIFIED = "qualified"
    OFFICIAL = "official"


@dataclass
class GenerateContext:
    """Everything a client's ``write``/``render_success`` needs beyond paths.

    ``extras`` carries client-specific prompt results (Claude Code uses
    ``auto_compact_window`` and ``use_beta_endpoint``).
    """

    id_style: IdStyle
    selected_dicts: list[dict | None]
    extras: dict = field(default_factory=dict)


def _backup_if_exists(path: Path) -> None:
    """Prompt to backup an existing config file before overwriting."""
    if not path.exists():
        return
    console.print(f"\n[yellow]{path.name} already exists at {path}[/yellow]")
    if Confirm.ask("Backup existing file?", default=True):
        backup_path = path.with_suffix(
            f"{path.suffix}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy(path, backup_path)
        console.print(f"[green]Backed up to {backup_path}[/green]")


def _fetch_models() -> list[dict]:
    """Fetch models from the server.

    Exits the CLI if the server is unreachable or no models are available.
    """
    try:
        client = get_admin_client()
        models = asyncio.run(client.list_models())
    except ServerNotRunningError as e:
        console.print(f"[red]{e}[/red]")
        console.print("[dim]Tip: Start router-maestro server first.[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not models:
        console.print("[red]No models available. Please authenticate first.[/red]")
        raise typer.Exit(1)

    return models


def _display_models(models: list[dict]) -> None:
    """Display models in a Rich table."""
    console.print("\n[bold]Available models:[/bold]")
    table = Table()
    table.add_column("#", style="dim")
    table.add_column("Model Key", style="green")
    table.add_column("Name", style="white")
    for i, model in enumerate(models, 1):
        key = model.get("display_key", _model_key(model))
        table.add_row(str(i), key, model["name"])
    console.print(table)


def _fetch_and_display_models() -> list[dict]:
    """Fetch models from the server and display them in a table."""
    models = _fetch_models()
    _display_models(models)
    return models


def _select_model(models: list[dict], prompt: str, default: str = "0") -> str:
    """Prompt the user to select a model from the list.

    Returns the ``provider/id`` model key, or ``"router-maestro"`` for
    auto-routing (choice ``0``).
    """
    selected = _select_model_dict(models, prompt, default=default)
    return _model_key(selected) if selected else "router-maestro"


def _select_model_dict(models: list[dict], prompt: str, default: str = "0") -> dict | None:
    """Prompt the user to select a model and return the model dict.

    Returns ``None`` for the auto-routing choice (``0`` or invalid input).
    """
    choice = Prompt.ask(prompt, default=default)
    if choice != "0" and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
        console.print(f"[yellow]Invalid selection '{choice}', using auto-routing.[/yellow]")
    return None


def _model_key(model: dict) -> str:
    """Resolve the wire model key for a model dict from the CLI's model list."""
    if "wire_key" in model:
        return model["wire_key"]
    if "custom_key" in model:
        return model["custom_key"]
    return qualify_model_id(model["provider"], model["id"])


def _bare_upstream_model_id(model: dict) -> str:
    """Return the upstream ID from qualified server or legacy bare model entries."""
    provider = model.get("provider", "")
    model_id = model.get("id", "")
    prefix = f"{provider}/"
    return model_id[len(prefix) :] if provider and model_id.startswith(prefix) else model_id


def _upstream_context_window(model: dict) -> int | None:
    """Compute the displayed upstream context window for a Copilot model.

    Mirrors what VS Code's Copilot model picker shows: prompt + output, which
    matches the catalog's advertised window in most cases. Falls back to the
    server-reported ``max_context_window_tokens`` if either component is
    missing.
    """
    prompt = model.get("max_prompt_tokens")
    output = model.get("max_output_tokens")
    if isinstance(prompt, int) and prompt > 0 and isinstance(output, int) and output > 0:
        return prompt + output
    ctx = model.get("max_context_window_tokens")
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    return None


class ClientConfig(ABC):
    """A supported client whose config Router-Maestro can generate.

    Subclasses declare their identity (``key``/``display_name``/``description``),
    their file paths and level menu, and how they write/announce config. The
    base owns the whole :meth:`generate` orchestration and the shared prompts.
    """

    #: Registry key and ``config <key>`` subcommand name (e.g. ``"codex"``).
    key: str
    #: Human name shown in the interactive tool picker.
    display_name: str
    #: One-line description shown in the interactive tool picker.
    description: str

    # ---- per-client structure (abstract) -------------------------------

    @abstractmethod
    def paths(self) -> dict[str, Path]:
        """Return ``{"user": Path, "project": Path}`` config targets."""

    @abstractmethod
    def level_menu(self) -> tuple[str, str]:
        """Return the ``(user_label, project_label)`` shown in Step 1."""

    @abstractmethod
    def write(self, *, level: str, path: Path, models: list[str], ctx: GenerateContext) -> None:
        """Persist the generated config to ``path``."""

    @abstractmethod
    def render_success(
        self, *, level: str, path: Path, models: list[str], ctx: GenerateContext
    ) -> None:
        """Print the post-generation success panel."""

    # ---- overridable hooks (single-model, no injection, no extras) ------

    def load_models(self) -> list[dict]:
        """Fetch and display the model list. Claude Code overrides to inject 1M."""
        return _fetch_and_display_models()

    def select_models(self, models: list[dict]) -> list[dict | None]:
        """Prompt for the model(s) this client writes (default: one)."""
        console.print("\n[bold]Step 2: Select model[/bold]")
        return [_select_model_dict(models, "Enter number (or 0 for auto-routing)")]

    def prompt_extras(self, selected_dicts: list[dict | None]) -> dict:
        """Prompt for any client-specific options (default: none)."""
        return {}

    # ---- id-style resolution (base-owned; Part B extends) ---------------

    def resolve_id_style(self, id_style: IdStyle | None, selected: list[dict | None]) -> IdStyle:
        """Resolve the effective id style. Part A: always ``QUALIFIED``."""
        return id_style or IdStyle.QUALIFIED

    def resolve_model_string(self, model: dict | None, id_style: IdStyle) -> str:
        """Resolve one selected model dict to the string written into config.

        Part A always produces the provider-qualified wire key (or the
        auto-routing sentinel). Part B adds official-id conversion here.
        """
        if model is None:
            return "router-maestro"
        return _model_key(model)

    # ---- shared resolvers ----------------------------------------------

    def _base_url(self) -> str:
        """Router-Maestro server base URL (from the admin client endpoint)."""
        client = get_admin_client()
        return (
            client.endpoint.rstrip("/") if hasattr(client, "endpoint") else "http://localhost:8080"
        )

    def _auth_token(self) -> str:
        """API key for the active context, or the ``router-maestro`` fallback."""
        return get_current_context_api_key() or "router-maestro"

    def _select_level_and_path(self) -> tuple[str, Path]:
        """Step 1: prompt user vs project level, return ``(level, path)``."""
        user_label, project_label = self.level_menu()
        console.print("\n[bold]Step 1: Select configuration level[/bold]")
        console.print(f"  1. {user_label}")
        console.print(f"  2. {project_label}")
        choice = Prompt.ask("Select", choices=["1", "2"], default="1")
        level = "user" if choice == "1" else "project"
        return level, self.paths()[level]

    # ---- template method (owns the whole flow) -------------------------

    def generate(self, *, id_style: IdStyle | None = None) -> None:
        """Run the full interactive config-generation flow for this client."""
        level, path = self._select_level_and_path()
        _backup_if_exists(path)
        models = self.load_models()
        selected = self.select_models(models)
        id_style = self.resolve_id_style(id_style, selected)
        extras = self.prompt_extras(selected)
        model_strings = [self.resolve_model_string(d, id_style) for d in selected]
        ctx = GenerateContext(id_style=id_style, selected_dicts=selected, extras=extras)
        self.write(level=level, path=path, models=model_strings, ctx=ctx)
        self.render_success(level=level, path=path, models=model_strings, ctx=ctx)
