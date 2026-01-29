"""Configuration management commands."""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

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
}


def get_claude_code_paths() -> dict[str, Path]:
    """Get Claude Code settings paths."""
    return {
        "user": Path.home() / ".claude" / "settings.json",
        "project": Path.cwd() / ".claude" / "settings.json",
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
