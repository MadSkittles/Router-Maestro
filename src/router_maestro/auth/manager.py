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
from router_maestro.auth.repository import CredentialRepository
from router_maestro.auth.storage import (
    ApiKeyCredential,
    AuthStorage,
    Credential,
    OAuthCredential,
)

console = Console()


class _RepositoryStorageAdapter:
    """Compatibility surface for callers not yet migrated off ``manager.storage``."""

    def __init__(self, repository: CredentialRepository) -> None:
        self._repository = repository

    def get(self, provider: str) -> Credential | None:
        return self._repository.get_provider(provider)

    def set(self, provider: str, credential: Credential) -> None:
        self._repository.update_provider(provider, credential)

    def remove(self, provider: str) -> bool:
        return self._repository.remove_provider(provider)

    def list_providers(self) -> list[str]:
        return self._repository.list_providers()

    def save(self) -> None:
        """Mutations are already persisted atomically by ``set`` and ``remove``."""


class AuthManager:
    """Manager for authentication with various providers."""

    def __init__(self, repository: CredentialRepository | None = None) -> None:
        self.repository = repository or CredentialRepository()
        self._storage_adapter = _RepositoryStorageAdapter(self.repository)

    @property
    def storage(self) -> _RepositoryStorageAdapter | AuthStorage:
        """Compatibility adapter; new code should use repository-backed methods."""
        legacy = getattr(self, "_legacy_storage", None)
        if legacy is not None:
            return legacy
        return self._storage_adapter

    @storage.setter
    def storage(self, storage: AuthStorage) -> None:
        """Retain the existing in-memory injection seam used by provider tests."""
        self._legacy_storage = storage

    @property
    def uses_legacy_storage(self) -> bool:
        return getattr(self, "_legacy_storage", None) is not None

    def save(self) -> None:
        """Flush explicitly injected legacy storage; repository writes are immediate."""
        if self.uses_legacy_storage:
            self.storage.save()

    def list_authenticated(self) -> list[str]:
        """List all authenticated providers."""
        if self.uses_legacy_storage:
            return self.storage.list_providers()
        return self.repository.list_providers()

    def is_authenticated(self, provider: str) -> bool:
        """Check if a provider is authenticated."""
        return self.get_credential(provider) is not None

    def get_credential(self, provider: str) -> Credential | None:
        """Get credential for a provider."""
        if self.uses_legacy_storage:
            return self.storage.get(provider)
        return self.repository.get_provider(provider)

    def logout(self, provider: str) -> bool:
        """Log out from a provider."""
        if self.uses_legacy_storage:
            result = self.storage.remove(provider)
            if result:
                self.save()
            return result
        return self.repository.remove_provider(provider)

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
            credential = OAuthCredential(
                refresh=access_token.access_token,  # GitHub token for refresh
                access=copilot_token.token,  # Copilot token for API calls
                expires=copilot_token.expires_at,
                api_endpoint=copilot_token.api_endpoint,
            )
            if self.uses_legacy_storage:
                self.storage.set("github-copilot", credential)
                self.save()
            else:
                self.repository.update_provider("github-copilot", credential)

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
        credential = ApiKeyCredential(key=api_key)
        if self.uses_legacy_storage:
            self.storage.set(provider, credential)
            self.save()
        else:
            self.repository.update_provider(provider, credential)
        console.print(f"[green]Successfully saved API key for {provider}[/green]")
        return True


def run_async(coro):
    """Run an async coroutine in sync context."""
    return asyncio.run(coro)
