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
}


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
    if settings_path.exists():
        console.print(f"\n[yellow]settings.json already exists at {settings_path}[/yellow]")
        if Confirm.ask("Backup existing file?", default=True):
            backup_path = settings_path.with_suffix(
                f".json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy(settings_path, backup_path)
            console.print(f"[green]Backed up to {backup_path}[/green]")

    # Step 3 & 4: Select models from server
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

    # Display models
    console.print("\n[bold]Available models:[/bold]")
    table = Table()
    table.add_column("#", style="dim")
    table.add_column("Model Key", style="green")
    table.add_column("Name", style="white")
    for i, model in enumerate(models, 1):
        table.add_row(str(i), f"{model['provider']}/{model['id']}", model["name"])
    console.print(table)

    # Select main model
    console.print("\n[bold]Step 3: Select main model[/bold]")
    main_choice = Prompt.ask("Enter number (or 0 for auto-routing)", default="0")
    main_model = "router-maestro"
    if main_choice != "0" and main_choice.isdigit():
        idx = int(main_choice) - 1
        if 0 <= idx < len(models):
            m = models[idx]
            main_model = f"{m['provider']}/{m['id']}"

    # Select fast model
    console.print("\n[bold]Step 4: Select small/fast model[/bold]")
    fast_choice = Prompt.ask("Enter number", default="1")
    fast_model = "router-maestro"
    if fast_choice.isdigit():
        idx = int(fast_choice) - 1
        if 0 <= idx < len(models):
            m = models[idx]
            fast_model = f"{m['provider']}/{m['id']}"

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
    }

    # Load existing settings to preserve other sections (e.g., MCP servers)
    existing_config: dict = {}
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # If file is corrupted, start fresh

    # Merge: update env section while preserving other sections
    existing_config["env"] = env_config

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
    if config_path.exists():
        console.print(f"\n[yellow]config.toml already exists at {config_path}[/yellow]")
        if Confirm.ask("Backup existing file?", default=True):
            backup_path = config_path.with_suffix(
                f".toml.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy(config_path, backup_path)
            console.print(f"[green]Backed up to {backup_path}[/green]")

    # Step 3: Get models from server
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

    # Display models
    console.print("\n[bold]Available models:[/bold]")
    table = Table()
    table.add_column("#", style="dim")
    table.add_column("Model Key", style="green")
    table.add_column("Name", style="white")
    for i, model in enumerate(models, 1):
        table.add_row(str(i), f"{model['provider']}/{model['id']}", model["name"])
    console.print(table)

    # Select model
    console.print("\n[bold]Step 2: Select model[/bold]")
    model_choice = Prompt.ask("Enter number (or 0 for auto-routing)", default="0")
    selected_model = "router-maestro"
    if model_choice != "0" and model_choice.isdigit():
        idx = int(model_choice) - 1
        if 0 <= idx < len(models):
            m = models[idx]
            selected_model = f"{m['provider']}/{m['id']}"

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
