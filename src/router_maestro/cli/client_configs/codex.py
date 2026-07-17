"""OpenAI Codex (`~/.codex/config.toml`) config generation."""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomlkit
from rich.panel import Panel
from rich.prompt import Prompt

from router_maestro.cli.client_configs.base import (
    ClientConfig,
    GenerateContext,
    _bare_upstream_model_id,
    _model_operation_support,
    console,
)
from router_maestro.cli.client_configs.model_id import (
    ModelFamily,
    detect_family,
    to_openai_official,
)
from router_maestro.providers.copilot_support.catalog import is_model_responses_eligible
from router_maestro.routing.capabilities import Operation


def get_codex_paths() -> dict[str, Path]:
    """Get Codex config paths."""
    return {
        "user": Path.home() / ".codex" / "config.toml",
        "project": Path.cwd() / ".codex" / "config.toml",
    }


def _prompt_endpoint_mode(model: dict | None) -> bool:
    """Prompt whether to use the beta native Responses passthrough endpoint.

    Offered when the selected GitHub Copilot model natively serves the Responses
    API. Eligibility tracks the server's live catalog
    (``operation_capabilities['responses']``) so a newly-added GHC model is
    recognized in real time; the hardcoded ``is_model_responses_eligible`` name
    heuristic is only the fallback for servers that predate that field. Returns
    True to use the beta endpoint, False for the standard translated endpoint.
    """
    if model is None:
        return False
    provider = model.get("provider", "")
    if provider != "github-copilot":
        return False
    supported = _model_operation_support(model, Operation.RESPONSES.value)
    if supported is None:
        supported = is_model_responses_eligible(_bare_upstream_model_id(model))
    if not supported:
        return False

    console.print("\n[bold]Endpoint mode[/bold]")
    console.print("  1. Standard (translation-based, battle-tested)")
    console.print(
        "  2. Beta (native Copilot Responses passthrough — full reasoning/cache fidelity)"
    )
    choice = Prompt.ask("Select", choices=["1", "2"], default="2")
    return choice == "2"


def _build_router_maestro_provider_table(openai_url: str) -> tomlkit.items.Table:
    """Build the `[model_providers.router-maestro]` TOML table for Codex user config."""
    table = tomlkit.table()
    table["name"] = "Router Maestro"
    table["base_url"] = openai_url
    table["env_key"] = "ROUTER_MAESTRO_API_KEY"
    table["wire_api"] = "responses"
    return table


def _user_codex_has_router_maestro_provider(user_config_path: Path) -> bool:
    """Return True iff the user-level Codex config sets `model_provider = "router-maestro"`."""
    if not user_config_path.exists():
        return False
    try:
        with open(user_config_path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return False
    return data.get("model_provider") == "router-maestro"


class CodexConfig(ClientConfig):
    """Generate OpenAI Codex CLI config.toml for router-maestro."""

    key = "codex"
    display_name = "OpenAI Codex"
    description = "Generate config.toml for OpenAI Codex CLI"

    def paths(self) -> dict[str, Path]:
        return get_codex_paths()

    def level_menu(self) -> tuple[str, str]:
        return (
            "User-level (~/.codex/config.toml)",
            "Project-level (./.codex/config.toml)",
        )

    def is_native_family(self, bare_id: str) -> bool:
        return detect_family(bare_id) is ModelFamily.OPENAI

    def to_official_id(self, bare_id: str) -> str:
        return to_openai_official(bare_id)

    def prompt_extras(self, selected_dicts: list[dict | None]) -> dict:
        main_model_dict = selected_dicts[0] if selected_dicts else None
        return {"use_beta_endpoint": _prompt_endpoint_mode(main_model_dict)}

    def _openai_url(self, ctx: GenerateContext) -> str:
        path = "/api/openai/beta/v1" if ctx.extras.get("use_beta_endpoint") else "/api/openai/v1"
        return f"{self._base_url()}{path}"

    def write(self, *, level: str, path: Path, models: list[str], ctx: GenerateContext) -> None:
        selected_model = models[0]
        openai_url = self._openai_url(ctx)

        # Load existing config to preserve other sections
        existing_config: tomlkit.TOMLDocument = tomlkit.document()
        if path.exists():
            try:
                with open(path, "rb") as f:
                    existing_config = tomlkit.load(f)
            except (tomllib.TOMLDecodeError, OSError):
                pass  # If file is corrupted, start fresh

        # Update configuration
        existing_config["model"] = selected_model

        if level == "user":
            existing_config["model_provider"] = "router-maestro"
            if "model_providers" not in existing_config:
                existing_config["model_providers"] = tomlkit.table()
            existing_config["model_providers"]["router-maestro"] = (
                _build_router_maestro_provider_table(openai_url)
            )
        else:
            # Codex CLI 0.130+ rejects model_provider/model_providers at project scope.
            # Strip the keys this command wrote in older releases so the file stops
            # tripping the "Ignored unsupported project-local config keys" warning.
            existing_config.pop("model_provider", None)
            providers = existing_config.get("model_providers")
            if providers is not None:
                providers.pop("router-maestro", None)
                if len(providers) == 0:
                    existing_config.pop("model_providers", None)

        # Write config
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(existing_config))

    def render_success(
        self, *, level: str, path: Path, models: list[str], ctx: GenerateContext
    ) -> None:
        selected_model = models[0]
        openai_url = self._openai_url(ctx)

        if level == "user":
            body = (
                f"[green]Created {path}[/green]\n\n"
                f"Model: {selected_model}\n\n"
                f"Endpoint: {openai_url}\n\n"
                "[dim]Start router-maestro server before using Codex:[/dim]\n"
                "  router-maestro server start\n\n"
                "[dim]Set API key environment variable (optional):[/dim]\n"
                "  export ROUTER_MAESTRO_API_KEY=your-key"
            )
        else:
            if _user_codex_has_router_maestro_provider(self.paths()["user"]):
                inheritance_line = f"[dim]Inheriting provider from {self.paths()['user']}.[/dim]"
            else:
                inheritance_line = (
                    "[yellow]User-level Router-Maestro config not found.[/yellow]\n"
                    "Run [bold]router-maestro config codex[/bold] and pick option 1 first,\n"
                    "otherwise Codex won't know how to reach the server."
                )
            body = f"[green]Created {path}[/green]\n\nModel: {selected_model}\n\n{inheritance_line}"

        console.print(
            Panel(
                body,
                title="Success",
                border_style="green",
            )
        )
