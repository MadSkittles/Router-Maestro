"""Router-Maestro CLI application."""

import typer
from rich.console import Console

from router_maestro import __version__

app = typer.Typer(
    name="router-maestro",
    help="Multi-model routing and load balancing system with OpenAI-compatible API",
    no_args_is_help=True,
)

console = Console()

# Import and register sub-commands
from router_maestro.cli import auth, config, context, model, server  # noqa: E402

app.add_typer(server.app, name="server", help="Manage the API server")
app.add_typer(auth.app, name="auth", help="Manage authentication for providers")
app.add_typer(model.app, name="model", help="Manage models and priorities")
app.add_typer(context.app, name="context", help="Manage deployment contexts")
app.add_typer(config.app, name="config", help="Manage configuration")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
) -> None:
    """Router-Maestro: Multi-model routing and load balancing system."""
    if version:
        console.print(f"router-maestro version {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        console.print("[yellow]Use --help for usage information[/yellow]")


if __name__ == "__main__":
    app()
