"""OpenAI Codex (`~/.codex/config.toml`) config generation."""

from __future__ import annotations

import copy
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import tomlkit
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from tomlkit.items import AbstractTable, Table

from router_maestro.cli.client_configs.base import (
    ClientConfig,
    GenerateContext,
    ModelSelection,
    _bare_upstream_model_id,
    _model_key,
    _model_operation_support,
    _upstream_context_window,
    console,
)
from router_maestro.cli.client_configs.model_id import (
    ModelFamily,
    detect_family,
    to_openai_official,
)
from router_maestro.config.settings import write_json_owner_only
from router_maestro.providers.copilot_support.catalog import is_model_responses_eligible
from router_maestro.routing.capabilities import Operation

_CODEX_MODEL_CATALOG_FILENAME = "router-maestro-models.json"
_CODEX_CATALOG_BASELINE_SLUG = "gpt-5.6-terra"
_CODEX_MODEL_SPECIFIC_EXTENSION_FIELDS = frozenset(
    {
        "comp_hash",
        "include_apps_usage_instructions",
        "include_plugin_usage_instructions",
        "include_skills_usage_instructions",
        "multi_agent_version",
        "node_repl_auto_review_required",
        "node_repl_disabled",
        "tool_mode",
        "use_responses_lite",
    }
)


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


def _build_router_maestro_provider_table(openai_url: str) -> Table:
    """Build the `[model_providers.router-maestro]` TOML table for Codex user config."""
    table = tomlkit.table()
    table["name"] = "Router Maestro"
    table["base_url"] = openai_url
    table["env_key"] = "ROUTER_MAESTRO_API_KEY"
    table["wire_api"] = "responses"
    return table


def _catalog_path_for(
    *,
    level: str,
    config_path: Path,
    ctx: GenerateContext,
) -> Path:
    """Resolve the user-level catalog refreshed by this configuration run."""
    explicit_path = ctx.extras.get("model_catalog_path")
    if isinstance(explicit_path, str) and explicit_path:
        return Path(explicit_path).expanduser().resolve()
    if level == "user":
        return config_path.with_name(_CODEX_MODEL_CATALOG_FILENAME).resolve()
    return get_codex_paths()["user"].with_name(_CODEX_MODEL_CATALOG_FILENAME).resolve()


def _catalog_result_line(ctx: GenerateContext) -> str:
    """Render the catalog outcome without claiming a skipped or failed update succeeded."""
    if ctx.extras.get("update_model_catalog", True) is not True:
        return "[dim]Model catalog: not updated (skipped)[/dim]"
    path = ctx.extras.get("model_catalog_path")
    if ctx.extras.get("model_catalog_updated") is True:
        return f"Model catalog: {path}"
    error = ctx.extras.get("model_catalog_error")
    if isinstance(error, str) and error:
        return f"[yellow]Model catalog: update failed ({error})[/yellow]"
    return f"[dim]Model catalog: {path}[/dim]"


def _load_bundled_codex_catalog() -> dict[str, Any] | None:
    """Read the installed Codex catalog so generated metadata stays version-aligned."""
    try:
        result = subprocess.run(
            ["codex", "debug", "models", "--bundled"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        catalog = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list) or not models:
        return None
    return catalog


def _catalog_context_windows(model: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return the default and maximum total context windows for Codex metadata."""
    max_output = model.get("max_output_tokens")
    output_tokens = (
        max_output
        if isinstance(max_output, int) and not isinstance(max_output, bool) and max_output > 0
        else 0
    )
    default_prompt = None
    options = model.get("context_window_options")
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict) or option.get("is_default") is not True:
                continue
            value = option.get("max_prompt_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                default_prompt = value
                break

    default_window = default_prompt + output_tokens if default_prompt is not None else None
    maximum = model.get("max_context_window_tokens")
    max_window = (
        maximum
        if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0
        else _upstream_context_window(model)
    )
    if default_window is None:
        default_window = max_window
    if max_window is not None and default_window is not None:
        default_window = min(default_window, max_window)
    return default_window, max_window


def _build_codex_model_catalog(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extend Codex's bundled catalog with exact Router-Maestro model slugs."""
    catalog = _load_bundled_codex_catalog()
    if catalog is None:
        return None
    bundled_models = catalog.get("models")
    if not isinstance(bundled_models, list) or not bundled_models:
        return None

    baseline = next(
        (
            model
            for model in bundled_models
            if isinstance(model, dict) and model.get("slug") == _CODEX_CATALOG_BASELINE_SLUG
        ),
        next((model for model in bundled_models if isinstance(model, dict)), None),
    )
    if baseline is None:
        return None

    extended = copy.deepcopy(catalog)
    extended_models = extended["models"]
    known_slugs = {model.get("slug") for model in extended_models if isinstance(model, dict)}

    for priority, model in enumerate(models, start=1_000):
        slug = _model_key(model)
        if slug in known_slugs:
            continue
        entry = copy.deepcopy(baseline)
        for field in _CODEX_MODEL_SPECIFIC_EXTENSION_FIELDS:
            entry.pop(field, None)
        entry["slug"] = slug
        entry["display_name"] = model.get("name") or slug
        entry["description"] = f"{entry['display_name']} via Router-Maestro"
        entry["priority"] = priority
        entry["visibility"] = "list"
        entry["supported_in_api"] = True
        entry["additional_speed_tiers"] = []
        entry["service_tiers"] = []
        entry["availability_nux"] = None
        entry["upgrade"] = None
        entry["supports_search_tool"] = False
        entry["shell_type"] = "default"
        entry["model_messages"] = None
        entry["supports_reasoning_summaries"] = False
        entry["support_verbosity"] = False
        entry["default_verbosity"] = None
        entry["apply_patch_tool_type"] = None
        entry["web_search_tool_type"] = "text"
        entry["truncation_policy"] = {"mode": "bytes", "limit": 10_000}
        entry["supports_parallel_tool_calls"] = False
        entry["supports_image_detail_original"] = False
        entry["experimental_supported_tools"] = []
        default_window, max_window = _catalog_context_windows(model)
        if default_window is not None:
            entry["context_window"] = default_window
            entry["auto_compact_token_limit"] = None
        if max_window is not None:
            entry["max_context_window"] = max_window
        extended_models.append(entry)
        known_slugs.add(slug)

    if "router-maestro" not in known_slugs:
        auto_entry = copy.deepcopy(baseline)
        for field in _CODEX_MODEL_SPECIFIC_EXTENSION_FIELDS:
            auto_entry.pop(field, None)
        auto_entry["slug"] = "router-maestro"
        auto_entry["display_name"] = "Router-Maestro Auto"
        auto_entry["description"] = "Router-Maestro automatic model routing"
        auto_entry["priority"] = 999
        auto_entry["visibility"] = "list"
        auto_entry["supported_in_api"] = True
        auto_entry["additional_speed_tiers"] = []
        auto_entry["service_tiers"] = []
        auto_entry["availability_nux"] = None
        auto_entry["upgrade"] = None
        auto_entry["supports_search_tool"] = False
        auto_entry["shell_type"] = "default"
        auto_entry["model_messages"] = None
        auto_entry["supports_reasoning_summaries"] = False
        auto_entry["support_verbosity"] = False
        auto_entry["default_verbosity"] = None
        auto_entry["apply_patch_tool_type"] = None
        auto_entry["web_search_tool_type"] = "text"
        auto_entry["truncation_policy"] = {"mode": "bytes", "limit": 10_000}
        auto_entry["supports_parallel_tool_calls"] = False
        auto_entry["supports_image_detail_original"] = False
        auto_entry["experimental_supported_tools"] = []
        extended_models.append(auto_entry)
    return extended


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

    def load_models(self) -> list[dict]:
        models = super().load_models()
        self._available_models = models
        return models

    def level_menu(self) -> tuple[str, str]:
        return (
            "User-level (~/.codex/config.toml)",
            "Project-level (./.codex/config.toml)",
        )

    def is_native_family(self, bare_id: str) -> bool:
        return detect_family(bare_id) is ModelFamily.OPENAI

    def to_official_id(self, bare_id: str) -> str:
        return to_openai_official(bare_id)

    def prompt_extras(self, selections: list[ModelSelection]) -> dict:
        del selections
        console.print("\n[bold]Update Codex model catalog[/bold]")
        update_model_catalog = Confirm.ask(
            "Refresh router-maestro-models.json from the current Router-Maestro context?",
            default=True,
        )
        return {"update_model_catalog": update_model_catalog}

    def _openai_url(self, ctx: GenerateContext) -> str:
        return f"{self._base_url_for(ctx)}/api/openai/v1"

    def write(
        self, *, level: str, path: Path, models: dict[str, str], ctx: GenerateContext
    ) -> None:
        selected_model = models["main"]
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

        update_model_catalog = ctx.extras.get("update_model_catalog", True) is True
        preview_only = ctx.extras.get("preview_only") is True
        catalog_path = _catalog_path_for(level=level, config_path=path, ctx=ctx)
        ctx.extras["model_catalog_path"] = str(catalog_path)
        ctx.extras["model_catalog_updated"] = False
        ctx.extras.pop("model_catalog_error", None)

        if level == "user":
            existing_config["model_provider"] = "router-maestro"
            providers = existing_config.get("model_providers")
            if providers is None:
                providers = tomlkit.table()
                existing_config["model_providers"] = providers
            if not isinstance(providers, AbstractTable):
                raise TypeError("model_providers must be a TOML table")
            providers["router-maestro"] = _build_router_maestro_provider_table(openai_url)
            if update_model_catalog and preview_only:
                existing_config["model_catalog_json"] = str(catalog_path.resolve())
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
            if update_model_catalog and preview_only:
                existing_config["model_catalog_json"] = str(catalog_path.resolve())

        if update_model_catalog and not preview_only:
            available_models = getattr(
                self,
                "_available_models",
                [model for model in ctx.selected_dicts if model is not None],
            )
            catalog = _build_codex_model_catalog(available_models)
            if catalog is None:
                message = (
                    "Could not read the installed Codex model catalog; "
                    "router-maestro-models.json was not updated."
                )
                ctx.extras["model_catalog_error"] = message
                console.print(f"[yellow]{message}[/yellow]")
            else:
                write_json_owner_only(catalog_path, catalog)
                ctx.extras["model_catalog_updated"] = True
                existing_config["model_catalog_json"] = str(catalog_path.resolve())

        # Write config
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(existing_config))

    def render_success(
        self, *, level: str, path: Path, models: dict[str, str], ctx: GenerateContext
    ) -> None:
        selected_model = models["main"]
        openai_url = self._openai_url(ctx)

        if level == "user":
            catalog_line = _catalog_result_line(ctx)
            body = (
                f"[green]Created {path}[/green]\n\n"
                f"Model: {selected_model}\n\n"
                f"Endpoint: {openai_url}\n\n"
                f"{catalog_line}\n\n"
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
            body = (
                f"[green]Created {path}[/green]\n\n"
                f"Model: {selected_model}\n\n"
                f"{_catalog_result_line(ctx)}\n\n"
                f"{inheritance_line}"
            )

        console.print(
            Panel(
                body,
                title="Success",
                border_style="green",
            )
        )
