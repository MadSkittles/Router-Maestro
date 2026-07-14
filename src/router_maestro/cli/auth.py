"""Authentication management commands."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from router_maestro.auth.discovery import (
    ProviderAuthDefinition,
    provider_auth_definitions,
)
from router_maestro.cli.client import AdminClient, ServerNotRunningError, get_admin_client
from router_maestro.config import load_contexts_config
from router_maestro.config.providers import ProvidersConfig
from router_maestro.config.settings import load_providers_config

app = typer.Typer(no_args_is_help=True)
console = Console()


class AuthProviderDiscoveryClient(Protocol):
    """Adapter supplied by the shared AdminClient integration."""

    async def list_auth_providers(
        self,
    ) -> Sequence[ProviderAuthDefinition | Mapping[str, Any]]: ...


async def discover_auth_providers(
    client: AuthProviderDiscoveryClient,
    *,
    context_name: str,
    load_local_config: Callable[[], ProvidersConfig] = load_providers_config,
) -> tuple[ProviderAuthDefinition, ...]:
    """Prefer server definitions; only local connection failure may use local config."""
    try:
        records = await client.list_auth_providers()
    except ServerNotRunningError:
        if context_name != "local":
            raise
        return provider_auth_definitions(load_local_config())
    return tuple(
        record
        if isinstance(record, ProviderAuthDefinition)
        else ProviderAuthDefinition.from_mapping(record)
        for record in records
    )


def _handle_server_error(e: Exception) -> None:
    """Handle server connection errors."""
    if isinstance(e, ServerNotRunningError):
        console.print(f"[red]{e}[/red]")
    else:
        console.print(f"[red]Error: {e}[/red]")
    raise typer.Exit(1)


@app.command()
def login(
    provider: str = typer.Argument(None, help="Provider to authenticate with"),
) -> None:
    """Authenticate with a provider (interactive selection if not specified)."""
    client = get_admin_client()

    try:
        context_name = load_contexts_config().current
        definitions = asyncio.run(discover_auth_providers(client, context_name=context_name))
    except Exception as e:
        _handle_server_error(e)
        return

    providers = {definition.provider: definition for definition in definitions}

    if provider is None:
        # Interactive selection - get current status from server
        try:
            authenticated = asyncio.run(client.list_auth())
            auth_providers = {p["provider"] for p in authenticated}
        except Exception as e:
            _handle_server_error(e)
            return

        console.print("\n[bold]Available providers:[/bold]")
        for i, definition in enumerate(definitions, 1):
            status = "[green]✓[/green]" if definition.provider in auth_providers else "[dim]○[/dim]"
            console.print(f"  {i}. {status} {definition.display_name} ({definition.provider})")

        console.print()
        choice = Prompt.ask(
            "Select provider",
            choices=[str(i) for i in range(1, len(definitions) + 1)],
        )
        provider = definitions[int(choice) - 1].provider

    if provider not in providers:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(f"[dim]Available: {', '.join(providers)}[/dim]")
        raise typer.Exit(1)

    provider_info = providers[provider]
    console.print(f"\n[bold]Authenticating with {provider_info.display_name}...[/bold]\n")

    asyncio.run(_do_login(client, provider, provider_info))


async def _do_login(
    client: AdminClient,
    provider: str,
    provider_info: ProviderAuthDefinition,
) -> None:
    """Handle authentication flow via HTTP API."""
    try:
        if provider_info.auth_type.value == "oauth":
            # OAuth device flow
            result = await client.login_oauth(provider)

            console.print(
                "[bold green]Please visit the following URL and enter the code:[/bold green]"
            )
            uri = result["verification_uri"]
            console.print(f"  URL: [link={uri}]{uri}[/link]")
            console.print(f"  Code: [bold cyan]{result['user_code']}[/bold cyan]")
            console.print()
            console.print("[dim]Waiting for authorization...[/dim]")

            # Poll for completion
            session_id = result["session_id"]
            while True:
                await asyncio.sleep(5)
                status = await client.poll_oauth_status(session_id)

                if status["status"] == "complete":
                    console.print("[bold green]Successfully authenticated![/bold green]")
                    break
                elif status["status"] in ("error", "expired"):
                    error_msg = status.get("error", "Authentication failed")
                    console.print(f"[red]{error_msg}[/red]")
                    raise typer.Exit(1)
                # status == "pending" - continue polling
        else:
            # API key auth
            api_key = Prompt.ask(f"Enter API key for {provider_info.display_name}", password=True)
            if not api_key:
                console.print("[red]API key cannot be empty[/red]")
                raise typer.Exit(1)

            success = await client.login_api_key(provider, api_key)
            if success:
                console.print(f"[green]Successfully saved API key for {provider}[/green]")
            else:
                console.print(f"[red]Failed to save API key for {provider}[/red]")
                raise typer.Exit(1)

    except ServerNotRunningError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Failed to authenticate: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def logout(
    provider: str = typer.Argument(..., help="Provider to log out from"),
) -> None:
    """Log out from a provider."""
    client = get_admin_client()

    try:
        success = asyncio.run(client.logout(provider))
        if success:
            console.print(f"[green]Successfully logged out from {provider}[/green]")
        else:
            console.print(f"[yellow]Not authenticated with {provider}[/yellow]")
    except Exception as e:
        _handle_server_error(e)


@app.command(name="list")
def list_auth() -> None:
    """List all authenticated providers."""
    client = get_admin_client()

    try:
        authenticated = asyncio.run(client.list_auth())
    except Exception as e:
        _handle_server_error(e)
        return

    if not authenticated:
        console.print("[dim]No providers authenticated yet.[/dim]")
        console.print("[dim]Use 'router-maestro auth login' to authenticate.[/dim]")
        return

    table = Table(title="Authenticated Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")

    for provider_info in authenticated:
        auth_type = "OAuth" if provider_info["auth_type"] == "oauth" else "API Key"
        status = "✓ Active" if provider_info["status"] == "active" else "⚠ Expired"
        table.add_row(provider_info["provider"], auth_type, status)

    console.print(table)
