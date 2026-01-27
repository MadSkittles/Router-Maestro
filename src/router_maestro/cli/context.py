"""Context management commands for remote deployments."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from router_maestro.cli.client import AdminClientError, get_admin_client
from router_maestro.config import (
    ContextConfig,
    load_contexts_config,
    save_contexts_config,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command(name="list")
def list_contexts() -> None:
    """List all configured contexts."""
    config = load_contexts_config()

    if not config.contexts:
        console.print("[dim]No contexts configured.[/dim]")
        return

    table = Table(title="Deployment Contexts")
    table.add_column("Name", style="cyan")
    table.add_column("Endpoint", style="green")
    table.add_column("API Key", style="yellow")
    table.add_column("Current", style="magenta")

    for name, ctx in config.contexts.items():
        is_current = "✓" if name == config.current else ""
        api_key_display = "****" if ctx.api_key else "-"
        table.add_row(name, ctx.endpoint, api_key_display, is_current)

    console.print(table)


@app.command(name="set")
def set_context(
    name: str = typer.Argument(..., help="Context name to switch to"),
) -> None:
    """Switch to a different context."""
    config = load_contexts_config()

    if name not in config.contexts:
        console.print(f"[red]Context '{name}' not found[/red]")
        console.print(f"[dim]Available: {', '.join(config.contexts.keys())}[/dim]")
        raise typer.Exit(1)

    config.current = name
    save_contexts_config(config)
    console.print(f"[green]Switched to context: {name}[/green]")
    console.print(f"  Endpoint: {config.contexts[name].endpoint}")


@app.command()
def add(
    name: str = typer.Argument(..., help="Name for the new context"),
    endpoint: str = typer.Option(..., "--endpoint", "-e", help="API endpoint URL"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key (optional)"),
) -> None:
    """Add a new context."""
    config = load_contexts_config()

    if name in config.contexts:
        console.print(f"[yellow]Context '{name}' already exists. Overwriting...[/yellow]")

    config.contexts[name] = ContextConfig(endpoint=endpoint, api_key=api_key)
    save_contexts_config(config)

    console.print(f"[green]Added context: {name}[/green]")
    console.print(f"  Endpoint: {endpoint}")
    if api_key:
        console.print("  API Key: ****")


@app.command()
def remove(
    name: str = typer.Argument(..., help="Context name to remove"),
) -> None:
    """Remove a context."""
    config = load_contexts_config()

    if name not in config.contexts:
        console.print(f"[red]Context '{name}' not found[/red]")
        raise typer.Exit(1)

    if name == "local":
        console.print("[red]Cannot remove the 'local' context[/red]")
        raise typer.Exit(1)

    if name == config.current:
        console.print("[yellow]Switching to 'local' context first...[/yellow]")
        config.current = "local"

    del config.contexts[name]
    save_contexts_config(config)
    console.print(f"[green]Removed context: {name}[/green]")


@app.command()
def current() -> None:
    """Show the current context."""
    config = load_contexts_config()
    ctx = config.contexts.get(config.current)

    if ctx:
        console.print(f"[bold]Current context:[/bold] {config.current}")
        console.print(f"  Endpoint: {ctx.endpoint}")
        if ctx.api_key:
            console.print("  API Key: ****")
    else:
        console.print("[yellow]No context selected[/yellow]")


@app.command()
def test() -> None:
    """Test connection to the current context's server."""
    config = load_contexts_config()
    ctx = config.contexts.get(config.current)

    if not ctx:
        console.print("[red]No context selected[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Testing connection to:[/bold] {config.current}")
    console.print(f"  Endpoint: {ctx.endpoint}")

    client = get_admin_client()

    try:
        result = asyncio.run(client.test_connection())
        console.print("\n[green]✓ Connection successful![/green]")
        console.print(f"  Server: {result.get('name', 'Unknown')}")
        console.print(f"  Version: {result.get('version', 'Unknown')}")
    except AdminClientError as e:
        console.print(f"\n[red]✗ {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Connection failed: {e}[/red]")
        raise typer.Exit(1)
