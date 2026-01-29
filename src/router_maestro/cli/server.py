"""Server management commands."""

import os
import socket

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel

from router_maestro.config.server import get_current_context_api_key, get_or_create_api_key

app = typer.Typer(no_args_is_help=True)
console = Console()


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


@app.command()
def start(
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    api_key: str | None = typer.Option(None, "--api-key", "-k", help="API key for authentication"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (development)"),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
) -> None:
    """Start router-maestro API server."""
    if is_port_in_use(port, host):
        console.print(f"[red]Error: Port {port} is already in use[/red]")
        raise typer.Exit(1)

    # Validate log level
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
    log_level = log_level.upper()
    if log_level not in valid_levels:
        console.print(
            f"[red]Error: Invalid log level '{log_level}'. Valid: {', '.join(valid_levels)}[/red]"
        )
        raise typer.Exit(1)

    # Get or create API key
    key, was_generated = get_or_create_api_key(api_key)

    # Set environment variables for server process to read
    os.environ["ROUTER_MAESTRO_API_KEY"] = key
    os.environ["ROUTER_MAESTRO_LOG_LEVEL"] = log_level

    console.print(f"[green]Starting Router-Maestro server on {host}:{port}...[/green]")
    console.print(f"[dim]API endpoint: http://{host}:{port}/v1[/dim]")
    console.print(f"[dim]Log level: {log_level}[/dim]")

    if was_generated:
        console.print(
            Panel(
                f"[yellow]API Key (auto-generated):[/yellow]\n[bold]{key}[/bold]",
                title="Authentication",
                border_style="yellow",
            )
        )
    else:
        console.print(f"[dim]API Key: {key[:12]}...{key[-4:]}[/dim]")

    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Map log level to uvicorn format
    uvicorn_level = log_level.lower()

    uvicorn.run(
        "router_maestro.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=uvicorn_level,
    )


@app.command()
def status() -> None:
    """Show current server status (uses current context)."""
    import asyncio

    from router_maestro.cli.client import get_admin_client, get_current_endpoint

    client = get_admin_client()
    endpoint = get_current_endpoint()

    console.print(f"[dim]Context endpoint: {endpoint}[/dim]")

    try:
        data = asyncio.run(client.test_connection())
        console.print("[green]Server is running[/green]")
        console.print(f"  Version: {data.get('version', 'unknown')}")
        console.print(f"  Status: {data.get('status', 'unknown')}")
    except Exception as e:
        console.print(f"[yellow]Server not reachable: {e}[/yellow]")


@app.command()
def show_key() -> None:
    """Show current server API key."""
    api_key = get_current_context_api_key()
    if api_key:
        console.print(f"[green]API Key:[/green] {api_key}")
    else:
        console.print(
            "[yellow]No API key configured. One will be generated on server start.[/yellow]"
        )
