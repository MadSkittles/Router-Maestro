"""Model management commands."""

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from router_maestro.cli.client import ServerNotRunningError, get_admin_client

app = typer.Typer(no_args_is_help=True)
console = Console()


def _handle_server_error(e: Exception) -> None:
    """Handle server connection errors."""
    if isinstance(e, ServerNotRunningError):
        console.print(f"[red]{e}[/red]")
    else:
        console.print(f"[red]Error: {e}[/red]")
    raise typer.Exit(1)


@app.command(name="list")
def list_models() -> None:
    """List all available models with their priorities."""
    client = get_admin_client()

    # Get models and priorities
    try:
        models = asyncio.run(client.list_models())
        priorities_data = asyncio.run(client.get_priorities())
        priorities_list = priorities_data.get("priorities", [])
    except Exception as e:
        _handle_server_error(e)
        return

    if not models:
        console.print("[dim]No models available.[/dim]")
        console.print("[dim]Make sure you have authenticated with at least one provider.[/dim]")
        return

    table = Table(title="Available Models")
    table.add_column("Priority", style="cyan", justify="right")
    table.add_column("Model Key", style="green")
    table.add_column("Display Name", style="white")
    table.add_column("Provider", style="magenta")

    for model in models:
        model_key = f"{model['provider']}/{model['id']}"
        # Check if this model is in the priority list
        try:
            priority_idx = priorities_list.index(model_key)
            priority_str = str(priority_idx + 1)
        except ValueError:
            priority_str = "-"

        table.add_row(
            priority_str,
            model_key,
            model["name"],
            model["provider"],
        )

    console.print(table)


# Priority subcommand group
priority_app = typer.Typer(no_args_is_help=True, help="Manage model priorities")
app.add_typer(priority_app, name="priority")


@priority_app.command(name="list")
def priority_list() -> None:
    """List current model priorities."""
    client = get_admin_client()

    try:
        data = asyncio.run(client.get_priorities())
        priorities = data.get("priorities", [])
    except Exception as e:
        _handle_server_error(e)
        return

    if not priorities:
        console.print("[dim]No priorities configured.[/dim]")
        console.print(
            "[dim]Use 'router-maestro model priority add <provider/model>' to add priorities.[/dim]"
        )
        return

    table = Table(title="Model Priorities")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Model Key", style="green")

    for idx, model_key in enumerate(priorities):
        table.add_row(str(idx + 1), model_key)

    console.print(table)


@priority_app.command(name="add")
def priority_add(
    model_key: Annotated[str, typer.Argument(help="Model key in format 'provider/model'")],
    position: Annotated[
        int | None,
        typer.Option("--position", "-p", help="Position in priority list (1-based)"),
    ] = None,
) -> None:
    """Add or move a model in the priority list."""
    if "/" not in model_key:
        console.print("[red]Model key must be in format 'provider/model'[/red]")
        raise typer.Exit(1)

    client = get_admin_client()

    try:
        # Get current priorities
        data = asyncio.run(client.get_priorities())
        priorities = data.get("priorities", [])
        fallback = data.get("fallback")

        # Remove if already exists
        if model_key in priorities:
            priorities.remove(model_key)

        # Insert at position
        if position is None:
            priorities.append(model_key)
        else:
            pos = position - 1  # Convert 1-based to 0-based
            priorities.insert(pos, model_key)

        # Save
        asyncio.run(client.set_priorities(priorities, fallback))

        if position:
            console.print(f"[green]Added '{model_key}' at position {position}[/green]")
        else:
            console.print(f"[green]Added '{model_key}' to end of priority list[/green]")

    except Exception as e:
        _handle_server_error(e)


@priority_app.command(name="remove")
def priority_remove(
    model_key: Annotated[str, typer.Argument(help="Model key in format 'provider/model'")],
) -> None:
    """Remove a model from the priority list."""
    if "/" not in model_key:
        console.print("[red]Model key must be in format 'provider/model'[/red]")
        raise typer.Exit(1)

    client = get_admin_client()

    try:
        # Get current priorities
        data = asyncio.run(client.get_priorities())
        priorities = data.get("priorities", [])
        fallback = data.get("fallback")

        if model_key in priorities:
            priorities.remove(model_key)
            asyncio.run(client.set_priorities(priorities, fallback))
            console.print(f"[green]Removed '{model_key}' from priority list[/green]")
        else:
            console.print(f"[yellow]'{model_key}' was not in the priority list[/yellow]")

    except Exception as e:
        _handle_server_error(e)


@priority_app.command(name="clear")
def priority_clear() -> None:
    """Clear all priorities."""
    client = get_admin_client()

    try:
        data = asyncio.run(client.get_priorities())
        fallback = data.get("fallback")
        asyncio.run(client.set_priorities([], fallback))
        console.print("[green]Cleared all priorities[/green]")
    except Exception as e:
        _handle_server_error(e)
