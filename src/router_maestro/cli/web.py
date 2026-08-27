"""Launch the loopback-only Router-Maestro web portal."""

from __future__ import annotations

import threading
import webbrowser

import typer
import uvicorn
from rich.console import Console

from router_maestro.web import create_portal_app

console = Console()


def _is_loopback_host(host: str) -> bool:
    return host.casefold() in {"127.0.0.1", "localhost"}


def _browser_url(host: str, port: int) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    return f"http://{rendered_host}:{port}"


def run(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Loopback address for the local portal.",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        min=1,
        max=65535,
        help="Local portal port.",
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Do not open the system browser automatically.",
    ),
) -> None:
    """Open the local Router-Maestro configuration portal."""
    if not _is_loopback_host(host):
        console.print("[red]The web portal can only bind to a loopback address.[/red]")
        raise typer.Exit(2)

    url = _browser_url(host, port)
    console.print(f"[green]Router-Maestro portal:[/green] {url}")
    if not no_open:
        timer = threading.Timer(0.6, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    uvicorn.run(
        create_portal_app(),
        host=host,
        port=port,
        log_level="warning",
    )


__all__ = ["run"]
