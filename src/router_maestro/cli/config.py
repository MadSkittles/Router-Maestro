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
    choice = Prompt.ask(prompt, default=default)
    if choice != "0" and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            m = models[idx]
            if "custom_key" in m:
                return m["custom_key"]
            return f"{m['provider']}/{m['id']}"
        console.print(f"[yellow]Invalid selection '{choice}', using auto-routing.[/yellow]")
    return "router-maestro"


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
    main_model = _select_model(models, "Enter number (or 0 for auto-routing)")

    console.print("\n[bold]Step 4: Select small/fast model[/bold]")
    fast_model = _select_model(models, "Enter number", default="1")

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

    console.print(
        Panel(
            f"[green]Created {settings_path}[/green]\n\n"
            f"Main model: {main_model}\n"
            f"Fast model: {fast_model}\n\n"
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
    existing_config["model_provider"] = "router-maestro"

    # Create or update model_providers section
    if "model_providers" not in existing_config:
        existing_config["model_providers"] = tomlkit.table()

    provider_config = tomlkit.table()
    provider_config["name"] = "Router Maestro"
    provider_config["base_url"] = openai_url
    provider_config["env_key"] = "ROUTER_MAESTRO_API_KEY"
    provider_config["wire_api"] = "responses"
    existing_config["model_providers"]["router-maestro"] = provider_config

    # Write config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(existing_config))

    console.print(
        Panel(
            f"[green]Created {config_path}[/green]\n\n"
            f"Model: {selected_model}\n\n"
            f"Endpoint: {openai_url}\n\n"
            "[dim]Start router-maestro server before using Codex:[/dim]\n"
            "  router-maestro server start\n\n"
            "[dim]Set API key environment variable (optional):[/dim]\n"
            "  export ROUTER_MAESTRO_API_KEY=your-key",
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
