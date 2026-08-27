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
from router_maestro.cli.client_configs.prompts import select_dropdown, supports_dropdowns
from router_maestro.config.server import get_current_context_api_key
from router_maestro.routing.model_ref import qualify_model_id

console = Console()


class IdStyle(StrEnum):
    """How a selected model is spelled in the generated config.

    ``QUALIFIED`` writes the internal ``provider/upstream-id`` (the default,
    unambiguous across providers). ``BARE`` removes only the provider prefix.
    ``OFFICIAL`` retains the legacy vendor-native spelling behavior used by the
    explicit ``--id-style official`` option.
    """

    QUALIFIED = "qualified"
    BARE = "bare"
    OFFICIAL = "official"


class ContextWindowChoice(StrEnum):
    """Client-side context window hint attached to a selected model."""

    DEFAULT = "default"
    CONTEXT_200K = "200k"
    CONTEXT_1M = "1m"


@dataclass(frozen=True)
class ModelSelection:
    """One named model slot selected by a client config wizard."""

    slot: str
    model: dict | None
    context_window: ContextWindowChoice = ContextWindowChoice.DEFAULT


@dataclass
class GenerateContext:
    """Everything a client's ``write``/``render_success`` needs beyond paths.

    ``selections`` preserves the model slot and context choice used to produce
    each entry in the resolved model mapping.
    """

    id_style: IdStyle
    selections: tuple[ModelSelection, ...]
    extras: dict = field(default_factory=dict)
    endpoint: str | None = None
    api_key: str | None = None

    @property
    def selected_dicts(self) -> list[dict | None]:
        """Compatibility view of the selected model dictionaries."""
        return [selection.model for selection in self.selections]


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
    table.add_column("Contexts", style="cyan")
    for i, model in enumerate(models, 1):
        key = model.get("display_key", _model_key(model))
        table.add_row(str(i), key, model["name"], _context_windows_label(model))
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


def _select_model_dict(
    models: list[dict],
    prompt: str,
    default: str = "0",
    *,
    default_model: dict | None = None,
    allow_auto: bool = True,
) -> dict | None:
    """Prompt the user to select a model and return the model dict.

    Interactive terminals receive a searchable dropdown. The numeric prompt is
    retained for non-interactive/test environments.
    """
    if supports_dropdowns():
        choices: list[tuple[str, dict | None]] = []
        if allow_auto:
            choices.append(("router-maestro — Auto routing", None))
        choices.extend((_model_choice_label(model), model) for model in models)
        selected_default = default_model
        if selected_default is None and not allow_auto and models:
            selected_default = models[0]
        return select_dropdown(
            prompt,
            choices,
            default=selected_default,
            searchable=True,
        )

    if default_model is not None:
        for index, model in enumerate(models, 1):
            if model == default_model:
                default = str(index)
                break
    choice = Prompt.ask(prompt, default=default)
    if choice == "0" and allow_auto:
        return None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    if allow_auto:
        console.print(f"[yellow]Invalid selection '{choice}', using auto-routing.[/yellow]")
        return None
    fallback = default_model or (models[0] if models else None)
    console.print(f"[yellow]Invalid selection '{choice}', keeping the default model.[/yellow]")
    return fallback


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


def _model_operation_support(model: dict, operation: str) -> bool | None:
    """Return the live-catalog verdict for one Operation, or ``None`` if unknown.

    The server's admin ``/models`` response carries ``operation_capabilities``
    (keyed by ``Operation`` value, e.g. ``"responses"`` / ``"native_anthropic"``)
    derived from the provider's live catalog. When present it is authoritative,
    so beta-endpoint prompts track whatever the upstream currently serves. An
    older server that predates this field omits it entirely; ``None`` then lets
    callers fall back to their model-name heuristic.
    """
    caps = model.get("operation_capabilities")
    if not isinstance(caps, dict) or operation not in caps:
        return None
    value = caps.get(operation)
    return value if isinstance(value, bool) else None


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


def _model_choice_label(model: dict) -> str:
    """Build the searchable display label for one catalog model."""
    model_key = model.get("display_key", _model_key(model))
    name = model.get("name") or model_key
    return f"{model_key} — {name} — {_context_windows_label(model)}"


def _context_window_limits(model: dict) -> list[int]:
    """Return server-advertised context choices, with a legacy scalar fallback."""
    limits: list[int] = []
    options = model.get("context_window_options")
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            value = option.get("max_prompt_tokens")
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                and value not in limits
            ):
                limits.append(value)
    if limits:
        return limits

    context = _upstream_context_window(model)
    return [context] if context is not None else []


def _context_windows_label(model: dict) -> str:
    limits = _context_window_limits(model)
    if not limits:
        return "unknown"
    return " / ".join(_format_token_count(limit) for limit in limits)


def _format_token_count(tokens: int) -> str:
    if tokens >= 1_000_000:
        value = int(tokens / 100_000) / 10
        return f"{value:g}M"
    if tokens > 900_000:
        return "1M"
    if tokens >= 1_000:
        return f"{round(tokens / 1_000)}K"
    return str(tokens)


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
    def write(
        self, *, level: str, path: Path, models: dict[str, str], ctx: GenerateContext
    ) -> None:
        """Persist the generated config to ``path``."""

    @abstractmethod
    def render_success(
        self, *, level: str, path: Path, models: dict[str, str], ctx: GenerateContext
    ) -> None:
        """Print the post-generation success panel."""

    @abstractmethod
    def is_native_family(self, bare_id: str) -> bool:
        """Whether ``bare_id`` belongs to this client's native vendor.

        The official-id option is only offered for native models (Codex↔OpenAI,
        Claude Code↔Anthropic, Gemini CLI↔Google).
        """

    @abstractmethod
    def to_official_id(self, bare_id: str) -> str:
        """Convert a native ``bare_id`` to this vendor's official spelling."""

    # ---- overridable hooks (single-model, no injection, no extras) ------

    def load_models(self) -> list[dict]:
        """Fetch and display the live model list."""
        return _fetch_and_display_models()

    def select_models(self, models: list[dict], *, level: str, path: Path) -> list[ModelSelection]:
        """Prompt for the model(s) this client writes (default: one)."""
        del level, path
        console.print("\n[bold]Step 2: Select model[/bold]")
        return [
            ModelSelection(
                slot="main",
                model=_select_model_dict(models, "Select model (or auto-routing)"),
            )
        ]

    def prompt_extras(self, selections: list[ModelSelection]) -> dict:
        """Prompt for any client-specific options (default: none)."""
        del selections
        return {}

    # ---- id-style resolution (base-owned) ------------------------------

    def _has_removable_provider_prefix(self, model: dict | None) -> bool:
        """Whether ``model`` has a provider prefix the wizard can remove.

        The auto-routing sentinel and entries carrying an explicit
        ``wire_key``/``custom_key`` are never rewritten, so they never gate the
        prompt on.
        """
        if model is None or "wire_key" in model or "custom_key" in model:
            return False
        return _model_key(model) != _bare_upstream_model_id(model)

    def resolve_id_style(self, id_style: IdStyle | None, selected: list[dict | None]) -> IdStyle:
        """Resolve the effective id style, prompting only when it matters.

        An explicit ``id_style`` (from ``--id-style``) wins with no prompt.
        Otherwise the final interactive choice asks whether to retain the
        provider prefix. If no selected model has a removable prefix, the
        option is meaningless and the default stays ``QUALIFIED``.
        """
        if id_style is not None:
            return id_style
        if any(self._has_removable_provider_prefix(model) for model in selected):
            choice = Prompt.ask(
                "Keep the provider prefix in model IDs?",
                choices=["yes", "no"],
                default="yes",
            )
            return IdStyle.QUALIFIED if choice == "yes" else IdStyle.BARE
        return IdStyle.QUALIFIED

    def resolve_model_string(self, model: dict | None, id_style: IdStyle) -> str:
        """Resolve one selected model dict to the string written into config.

        ``None`` -> the auto-routing sentinel. An explicit ``wire_key``/
        ``custom_key`` is returned unchanged regardless of style. ``BARE``
        removes only the provider prefix. Under ``OFFICIAL``, native models are
        converted to the vendor's spelling; non-native models retain the
        provider-qualified ID for backward compatibility.
        """
        if model is None:
            return "router-maestro"
        if "wire_key" in model or "custom_key" in model:
            return _model_key(model)
        qualified = _model_key(model)
        if id_style is IdStyle.QUALIFIED:
            return qualified
        bare = _bare_upstream_model_id(model)
        if id_style is IdStyle.BARE:
            return bare
        if not self.is_native_family(bare):
            console.print(
                f"[yellow]{qualified} is not a native {self.display_name} model; "
                f"keeping the provider-qualified id.[/yellow]"
            )
            return qualified
        return self.to_official_id(bare)

    def resolve_model_selection(self, selection: ModelSelection, id_style: IdStyle) -> str:
        """Resolve a typed model selection to the string written to config."""
        return self.resolve_model_string(selection.model, id_style)

    # ---- shared resolvers ----------------------------------------------

    def _base_url(self) -> str:
        """Router-Maestro server base URL from the active CLI context."""
        client = get_admin_client()
        return (
            client.endpoint.rstrip("/") if hasattr(client, "endpoint") else "http://localhost:8080"
        )

    def _auth_token(self) -> str:
        """API key from the active CLI context, or the legacy fallback."""
        return get_current_context_api_key() or "router-maestro"

    def _base_url_for(self, ctx: GenerateContext) -> str:
        """Resolve an explicit request endpoint without changing the active context."""
        return ctx.endpoint.rstrip("/") if ctx.endpoint is not None else self._base_url()

    def _auth_token_for(self, ctx: GenerateContext) -> str:
        """Resolve an explicit request key without changing the active context."""
        if ctx.endpoint is not None:
            return ctx.api_key or "router-maestro"
        return self._auth_token()

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
        selections = self.select_models(models, level=level, path=path)
        extras = self.prompt_extras(selections)
        id_style = self.resolve_id_style(id_style, [selection.model for selection in selections])
        model_strings = {
            selection.slot: self.resolve_model_selection(selection, id_style)
            for selection in selections
        }
        ctx = GenerateContext(id_style=id_style, selections=tuple(selections), extras=extras)
        self.write(level=level, path=path, models=model_strings, ctx=ctx)
        self.render_success(level=level, path=path, models=model_strings, ctx=ctx)
