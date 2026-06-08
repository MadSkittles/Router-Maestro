"""Configuration management commands."""

import asyncio
import json
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

import tomlkit
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from router_maestro.cli.client import ServerNotRunningError, get_admin_client
from router_maestro.config.server import get_current_context_api_key

app = typer.Typer(invoke_without_command=True)
console = Console()

# Available CLI tools for configuration
CLI_TOOLS = {
    "claude-code": {
        "name": "Claude Code",
        "description": "Generate settings.json for Claude Code CLI",
    },
    "codex": {
        "name": "OpenAI Codex",
        "description": "Generate config.toml for OpenAI Codex CLI",
    },
    "gemini": {
        "name": "Gemini CLI",
        "description": "Generate .env for Gemini CLI",
    },
}

# Claude Code native model IDs for 1M context variants.
# When set as ANTHROPIC_MODEL, Claude Code sends the `anthropic-beta: context-1m-*`
# header, which the router resolves to the actual provider model.
_OPUS_1M_NATIVE_KEY = "claude-opus-4-6[1m]"
_OPUS_1M_SOURCE_MODEL = "github-copilot/claude-opus-4.6-1m"
_OPUS_47_1M_NATIVE_KEY = "claude-opus-4-7[1m]"
_OPUS_47_1M_SOURCE_MODEL = "github-copilot/claude-opus-4.7-1m-internal"
# Opus 4.8 and Sonnet 4.6 don't ship a dedicated `-1m` variant — their base
# catalog entry already advertises max_context_window_tokens=1000000, so the
# native key maps straight to the base id. The `[1m]` suffix here only exists
# so Claude Code raises its auto-compact threshold to ~1M instead of clamping
# at the default 200K.
_OPUS_48_1M_NATIVE_KEY = "claude-opus-4-8[1m]"
_OPUS_48_1M_SOURCE_MODEL = "github-copilot/claude-opus-4.8"
_SONNET_46_1M_NATIVE_KEY = "claude-sonnet-4-6[1m]"
_SONNET_46_1M_SOURCE_MODEL = "github-copilot/claude-sonnet-4.6"

_INJECTABLE_1M_VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    # (source_model, native_key, bare_id, display_name)
    (
        _OPUS_1M_SOURCE_MODEL,
        _OPUS_1M_NATIVE_KEY,
        "claude-opus-4.6-1m",
        "Opus 4.6 1M (Auto-activated)",
    ),
    (
        _OPUS_47_1M_SOURCE_MODEL,
        _OPUS_47_1M_NATIVE_KEY,
        "claude-opus-4.7-1m-internal",
        "Opus 4.7 1M Internal (Auto-activated)",
    ),
    (
        _OPUS_48_1M_SOURCE_MODEL,
        _OPUS_48_1M_NATIVE_KEY,
        "claude-opus-4.8",
        "Opus 4.8 1M (Auto-activated)",
    ),
    (
        _SONNET_46_1M_SOURCE_MODEL,
        _SONNET_46_1M_NATIVE_KEY,
        "claude-sonnet-4.6",
        "Sonnet 4.6 1M (Auto-activated)",
    ),
)


def get_claude_code_paths() -> dict[str, Path]:
    """Get Claude Code settings paths."""
    return {
        "user": Path.home() / ".claude" / "settings.json",
        "project": Path.cwd() / ".claude" / "settings.json",
    }


def get_codex_paths() -> dict[str, Path]:
    """Get Codex config paths."""
    return {
        "user": Path.home() / ".codex" / "config.toml",
        "project": Path.cwd() / ".codex" / "config.toml",
    }


def get_gemini_cli_paths() -> dict[str, Path]:
    """Get Gemini CLI config paths."""
    return {
        "user": Path.home() / ".gemini" / ".env",
        "project": Path.cwd() / ".gemini" / ".env",
    }


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
        key = model.get("display_key", f"{model['provider']}/{model['id']}")
        table.add_row(str(i), key, model["name"])
    console.print(table)


def _fetch_and_display_models() -> list[dict]:
    """Fetch models from the server and display them in a table."""
    models = _fetch_models()
    _display_models(models)
    return models


def _maybe_inject_opus_1m(models: list[dict]) -> list[dict]:
    """Prepend Claude Code-native 1M context options for any source models present.

    Returns a new list (never mutates the input).
    """
    available_keys = {f"{m['provider']}/{m['id']}" for m in models}
    injected: list[dict] = []
    for source_model, native_key, bare_id, display_name in _INJECTABLE_1M_VARIANTS:
        if source_model in available_keys:
            injected.append(
                {
                    "provider": "github-copilot",
                    "id": bare_id,
                    "name": display_name,
                    "display_key": native_key,
                    "custom_key": native_key,
                }
            )
    if not injected:
        return models
    return [*injected, *models]


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
    if "custom_key" in model:
        return model["custom_key"]
    return f"{model['provider']}/{model['id']}"


# Claude Code recognizes 1M context windows natively for these model keys (the
# ones we inject via `_maybe_inject_opus_1m`) — the prompt offers a 1M default
# for them instead of the upstream-value option.
_CLAUDE_CODE_NATIVE_1M_KEYS: frozenset[str] = frozenset(
    {
        _OPUS_1M_NATIVE_KEY,
        _OPUS_47_1M_NATIVE_KEY,
        _OPUS_48_1M_NATIVE_KEY,
        _SONNET_46_1M_NATIVE_KEY,
    }
)

# Default CLAUDE_CODE_AUTO_COMPACT_WINDOW for non-Claude models. Matches
# Claude Code's built-in window for Claude Opus / Sonnet (200K).
_CLAUDE_CODE_DEFAULT_AUTO_COMPACT_WINDOW = 200_000


def _prompt_auto_compact_window(model: dict | None) -> int | None:
    """Prompt the user whether to set CLAUDE_CODE_AUTO_COMPACT_WINDOW.

    Returns the chosen token count to write, or ``None`` to skip the env var.

    In Claude Code 2.1.162+, auto-compact's threshold check is short-circuited
    in interactive mode when the window source is "auto" — only ``env`` or
    ``settings`` source actually arms the trigger. So setting this env var is
    what turns the feature on at all; the exact value is secondary.

    For Claude Code-native 1M model keys (e.g. ``claude-opus-4-7[1m]``), the
    default offered is 1M and the upstream-value option is dropped — Copilot's
    real prompt cap on the 1M variant is below 1M but matching Claude Code's
    own view of the window is the more useful default here.

    For everything else, the prompt offers:
      * ``y`` — use the upstream context window (``max_prompt_tokens`` +
        ``max_output_tokens``, matching what Copilot's own model picker shows)
      * ``n`` — skip; do not set the env var
      * ``d`` — set the default 200K (matches Claude Opus/Sonnet's window)
    """
    if model is None:
        return None
    model_key = _model_key(model)
    is_native_1m = model_key in _CLAUDE_CODE_NATIVE_1M_KEYS

    upstream = _upstream_context_window(model)
    default_value = 1_000_000 if is_native_1m else _CLAUDE_CODE_DEFAULT_AUTO_COMPACT_WINDOW

    console.print()
    if is_native_1m:
        console.print(
            "[bold]Set CLAUDE_CODE_AUTO_COMPACT_WINDOW?[/bold]\n"
            f"  Selected: {model_key}\n"
            f"  [dim]Claude Code's interactive auto-compact only arms when this env var\n"
            f"  (or settings.autoCompactWindow) is set. Without it, the trigger is\n"
            f"  short-circuited regardless of model. Default ({default_value}) matches\n"
            f"  Claude Code's native 1M window for this model.[/dim]"
        )
        choices = ["n", "d"]
        prompt_text = f"n = skip / d = default: {default_value}"
    else:
        upstream_line = (
            f"  Upstream context window: {upstream}"
            if upstream is not None
            else "  Upstream context window: (unknown)"
        )
        console.print(
            "[bold]Set CLAUDE_CODE_AUTO_COMPACT_WINDOW?[/bold]\n"
            f"  Selected: {model_key}\n"
            f"{upstream_line}\n"
            f"  [dim]Claude Code's interactive auto-compact only arms when this env var\n"
            f"  (or settings.autoCompactWindow) is set. Without it, the trigger is\n"
            f"  short-circuited regardless of model. Default ({default_value}) matches\n"
            f"  Claude Opus/Sonnet's 200K window.[/dim]"
        )
        can_use_upstream = upstream is not None
        if can_use_upstream:
            choices = ["y", "n", "d"]
            prompt_text = f"y = upstream: {upstream} / n = skip / d = default: {default_value}"
        else:
            choices = ["n", "d"]
            prompt_text = f"n = skip / d = default: {default_value}"

    choice = Prompt.ask(prompt_text, choices=choices, default="d").lower()

    if choice == "n":
        return None
    if choice == "y" and not is_native_1m and upstream is not None:
        return int(upstream)
    return default_value


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


@app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context) -> None:
    """Generate configuration for CLI tools (interactive selection if not specified)."""
    if ctx.invoked_subcommand is not None:
        return

    # Interactive selection
    console.print("\n[bold]Available CLI tools:[/bold]")
    tools = list(CLI_TOOLS.items())
    for i, (key, info) in enumerate(tools, 1):
        console.print(f"  {i}. {info['name']} - {info['description']}")

    console.print()
    choice = Prompt.ask(
        "Select tool to configure",
        choices=[str(i) for i in range(1, len(tools) + 1)],
        default="1",
    )

    idx = int(choice) - 1
    tool_key = tools[idx][0]

    # Dispatch to the appropriate command
    if tool_key == "claude-code":
        claude_code_config()
    elif tool_key == "codex":
        codex_config()
    elif tool_key == "gemini":
        gemini_cli_config()


@app.command(name="claude-code")
def claude_code_config() -> None:
    """Generate Claude Code CLI settings.json for router-maestro."""
    # Step 1: Select level
    console.print("\n[bold]Step 1: Select configuration level[/bold]")
    console.print("  1. User-level (~/.claude/settings.json)")
    console.print("  2. Project-level (./.claude/settings.json)")
    choice = Prompt.ask("Select", choices=["1", "2"], default="1")

    paths = get_claude_code_paths()
    level = "user" if choice == "1" else "project"
    settings_path = paths[level]

    # Step 2: Backup if exists
    _backup_if_exists(settings_path)

    # Step 3 & 4: Select models from server
    models = _fetch_models()

    # If the 1M variant is available, offer the Claude Code-native model key
    # as an extra option. Claude Code sends the extended-context beta header
    # when this key is used, and the router resolves it automatically.
    models = _maybe_inject_opus_1m(models)

    _display_models(models)

    console.print("\n[bold]Step 3: Select main model[/bold]")
    main_model_dict = _select_model_dict(models, "Enter number (or 0 for auto-routing)")
    main_model = _model_key(main_model_dict) if main_model_dict else "router-maestro"

    console.print("\n[bold]Step 4: Select small/fast model[/bold]")
    fast_model = _select_model(models, "Enter number", default="1")

    # Step 4b: Optional auto-compact window override (Claude Code only)
    auto_compact_window = _prompt_auto_compact_window(main_model_dict)

    # Step 5: Generate config
    auth_token = get_current_context_api_key() or "router-maestro"
    client = get_admin_client()
    base_url = (
        client.endpoint.rstrip("/") if hasattr(client, "endpoint") else "http://localhost:8080"
    )
    anthropic_url = f"{base_url}/api/anthropic"

    env_config = {
        "ANTHROPIC_BASE_URL": anthropic_url,
        "ANTHROPIC_AUTH_TOKEN": auth_token,
        "ANTHROPIC_MODEL": main_model,
        "ANTHROPIC_SMALL_FAST_MODEL": fast_model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_ENABLE_LSP": "1",
    }
    if auto_compact_window is not None:
        env_config["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(auto_compact_window)

    # Load existing settings to preserve other sections (e.g., MCP servers)
    existing_config: dict = {}
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # If file is corrupted, start fresh

    # Merge: update env variables while preserving existing ones
    existing_env = existing_config.get("env", {})
    if not isinstance(existing_env, dict):
        existing_env = {}
    existing_config["env"] = {**existing_env, **env_config}

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2)

    auto_compact_line = (
        f"Auto-compact window: {auto_compact_window} tokens\n\n"
        if auto_compact_window is not None
        else ""
    )
    console.print(
        Panel(
            f"[green]Created {settings_path}[/green]\n\n"
            f"Main model: {main_model}\n"
            f"Fast model: {fast_model}\n\n"
            f"{auto_compact_line}"
            f"Endpoint: {anthropic_url}\n\n"
            "[dim]Start router-maestro server before using Claude Code:[/dim]\n"
            "  router-maestro server start",
            title="Success",
            border_style="green",
        )
    )


@app.command(name="codex")
def codex_config() -> None:
    """Generate OpenAI Codex CLI config.toml for router-maestro."""
    # Step 1: Select level
    console.print("\n[bold]Step 1: Select configuration level[/bold]")
    console.print("  1. User-level (~/.codex/config.toml)")
    console.print("  2. Project-level (./.codex/config.toml)")
    choice = Prompt.ask("Select", choices=["1", "2"], default="1")

    paths = get_codex_paths()
    level = "user" if choice == "1" else "project"
    config_path = paths[level]

    # Step 2: Backup if exists
    _backup_if_exists(config_path)

    # Step 3: Get models from server
    models = _fetch_and_display_models()

    # Select model
    console.print("\n[bold]Step 2: Select model[/bold]")
    selected_model = _select_model(models, "Enter number (or 0 for auto-routing)")

    # Step 4: Generate config
    client = get_admin_client()
    base_url = (
        client.endpoint.rstrip("/") if hasattr(client, "endpoint") else "http://localhost:8080"
    )
    openai_url = f"{base_url}/api/openai/v1"

    # Load existing config to preserve other sections
    existing_config: tomlkit.TOMLDocument = tomlkit.document()
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                existing_config = tomlkit.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            pass  # If file is corrupted, start fresh

    # Update configuration
    existing_config["model"] = selected_model

    if level == "user":
        existing_config["model_provider"] = "router-maestro"
        if "model_providers" not in existing_config:
            existing_config["model_providers"] = tomlkit.table()
        existing_config["model_providers"]["router-maestro"] = _build_router_maestro_provider_table(
            openai_url
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
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(existing_config))

    if level == "user":
        body = (
            f"[green]Created {config_path}[/green]\n\n"
            f"Model: {selected_model}\n\n"
            f"Endpoint: {openai_url}\n\n"
            "[dim]Start router-maestro server before using Codex:[/dim]\n"
            "  router-maestro server start\n\n"
            "[dim]Set API key environment variable (optional):[/dim]\n"
            "  export ROUTER_MAESTRO_API_KEY=your-key"
        )
    else:
        if _user_codex_has_router_maestro_provider(paths["user"]):
            inheritance_line = f"[dim]Inheriting provider from {paths['user']}.[/dim]"
        else:
            inheritance_line = (
                "[yellow]User-level Router-Maestro config not found.[/yellow]\n"
                "Run [bold]router-maestro config codex[/bold] and pick option 1 first,\n"
                "otherwise Codex won't know how to reach the server."
            )
        body = (
            f"[green]Created {config_path}[/green]\n\nModel: {selected_model}\n\n{inheritance_line}"
        )

    console.print(
        Panel(
            body,
            title="Success",
            border_style="green",
        )
    )


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, preserving order."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    except OSError:
        pass
    return env


def _write_env_file(path: Path, env: dict[str, str]) -> None:
    """Write a dict as a .env file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in env.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command(name="gemini")
def gemini_cli_config() -> None:
    """Generate Gemini CLI .env for router-maestro."""
    # Step 1: Select level
    console.print("\n[bold]Step 1: Select configuration level[/bold]")
    console.print("  1. User-level (~/.gemini/.env)")
    console.print("  2. Project-level (./.gemini/.env)")
    choice = Prompt.ask("Select", choices=["1", "2"], default="1")

    paths = get_gemini_cli_paths()
    level = "user" if choice == "1" else "project"
    env_path = paths[level]

    # Step 2: Backup if exists
    _backup_if_exists(env_path)

    # Step 3: Select model
    models = _fetch_and_display_models()

    console.print("\n[bold]Step 2: Select model[/bold]")
    selected_model = _select_model(models, "Enter number (or 0 for auto-routing)")

    # Step 4: Generate config
    auth_key = get_current_context_api_key() or "router-maestro"
    client = get_admin_client()
    base_url = (
        client.endpoint.rstrip("/") if hasattr(client, "endpoint") else "http://localhost:8080"
    )
    gemini_url = f"{base_url}/api/gemini"

    # Load existing .env to preserve other variables
    existing_env = _parse_env_file(env_path)

    # Strip provider prefix (e.g. "github-copilot/gemini-2.5-pro" -> "gemini-2.5-pro")
    # Gemini CLI puts model name in URL path, so "/" would break routing
    model_name = selected_model.split("/", 1)[-1] if "/" in selected_model else selected_model

    # Set Gemini CLI variables
    existing_env["GOOGLE_GEMINI_BASE_URL"] = gemini_url
    existing_env["GEMINI_API_KEY"] = auth_key
    existing_env["GEMINI_MODEL"] = model_name
    existing_env["GEMINI_TELEMETRY_ENABLED"] = "false"

    _write_env_file(env_path, existing_env)

    console.print(
        Panel(
            f"[green]Created {env_path}[/green]\n\n"
            f"Model: {model_name}\n"
            f"Backend URL: {gemini_url}\n"
            f"Telemetry: disabled\n\n"
            "[dim]Start router-maestro server before using Gemini CLI:[/dim]\n"
            "  router-maestro server start\n\n"
            "[dim]Then run Gemini CLI normally:[/dim]\n"
            "  gemini",
            title="Success",
            border_style="green",
        )
    )
