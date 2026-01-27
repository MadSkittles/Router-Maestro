"""Authentication manager for all providers."""

import asyncio

import httpx
from rich.console import Console

from router_maestro.auth.github_oauth import (
    GitHubOAuthError,
    get_copilot_token,
    poll_access_token,
    request_device_code,
)
from router_maestro.auth.storage import (
    ApiKeyCredential,
    AuthStorage,
    OAuthCredential,
)

console = Console()


class AuthManager:
    """Manager for authentication with various providers."""

    def __init__(self) -> None:
        self.storage = AuthStorage.load()

    def save(self) -> None:
        """Save credentials to storage."""
        self.storage.save()

    def list_authenticated(self) -> list[str]:
        """List all authenticated providers."""
        return self.storage.list_providers()

    def is_authenticated(self, provider: str) -> bool:
        """Check if a provider is authenticated."""
        return self.storage.get(provider) is not None

    def get_credential(self, provider: str):
        """Get credential for a provider."""
        return self.storage.get(provider)

    def logout(self, provider: str) -> bool:
        """Log out from a provider."""
        result = self.storage.remove(provider)
        if result:
            self.save()
        return result

    async def login_copilot(self) -> bool:
        """Authenticate with GitHub Copilot using Device Flow.

        Returns:
            True if authentication was successful
        """
        async with httpx.AsyncClient() as client:
            # Step 1: Request device code
            console.print("[yellow]Requesting device code from GitHub...[/yellow]")
            try:
                device_code = await request_device_code(client)
            except httpx.HTTPError as e:
                console.print(f"[red]Failed to get device code: {e}[/red]")
                return False

            # Step 2: Show user code and verification URL
            console.print()
            console.print(
                "[bold green]Please visit the following URL and enter the code:[/bold green]"
            )
            uri = device_code.verification_uri
            console.print(f"  URL: [link={uri}]{uri}[/link]")
            console.print(f"  Code: [bold cyan]{device_code.user_code}[/bold cyan]")
            console.print()
            console.print("[dim]Waiting for authorization...[/dim]")

            # Step 3: Poll for access token
            try:
                access_token = await poll_access_token(
                    client,
                    device_code.device_code,
                    interval=device_code.interval,
                )
            except GitHubOAuthError as e:
                console.print(f"[red]Authorization failed: {e}[/red]")
                return False

            console.print("[green]GitHub authorization successful![/green]")

            # Step 4: Get Copilot token
            console.print("[yellow]Getting Copilot token...[/yellow]")
            try:
                copilot_token = await get_copilot_token(client, access_token.access_token)
            except httpx.HTTPError as e:
                console.print(f"[red]Failed to get Copilot token: {e}[/red]")
                console.print(
                    "[dim]Note: Make sure you have an active GitHub Copilot subscription.[/dim]"
                )
                return False

            # Step 5: Save credentials
            self.storage.set(
                "github-copilot",
                OAuthCredential(
                    refresh=access_token.access_token,  # GitHub token for refresh
                    access=copilot_token.token,  # Copilot token for API calls
                    expires=copilot_token.expires_at,
                ),
            )
            self.save()

            console.print(
                "[bold green]Successfully authenticated with GitHub Copilot![/bold green]"
            )
            return True

    def login_api_key(self, provider: str, api_key: str) -> bool:
        """Authenticate with an API key.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            api_key: API key

        Returns:
            True if authentication was successful
        """
        self.storage.set(provider, ApiKeyCredential(key=api_key))
        self.save()
        console.print(f"[green]Successfully saved API key for {provider}[/green]")
        return True


def run_async(coro):
    """Run an async coroutine in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)
