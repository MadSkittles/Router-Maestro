"""Authentication state and token minting for the GitHub Copilot provider."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, NoReturn

import httpx

from router_maestro.auth import AuthManager, AuthType, CredentialRepository
from router_maestro.auth.github_oauth import (
    GitHubOAuthError,
    _async_sleep,
    get_copilot_token,
)
from router_maestro.auth.storage import OAuthCredential
from router_maestro.providers.base import ProviderError, ProviderFailureKind
from router_maestro.utils import get_logger

logger = get_logger("providers.copilot.auth")

COPILOT_BASE_URL = "https://api.githubcopilot.com"
AUTH_RETRY_STATUSES = frozenset({401, 403})

# Token-mint retry policy. Retry only genuinely transient failures — NOT
# AUTHENTICATION: this codebase raises 401/403 with retryable=True to drive
# the router's cross-provider fallback, which is unrelated to re-minting.
_RETRYABLE_MINT_KINDS = frozenset(
    {
        ProviderFailureKind.RATE_LIMIT,
        ProviderFailureKind.UPSTREAM_STATUS,
        ProviderFailureKind.TRANSPORT,
    }
)
_MINT_MAX_RETRIES = 3
_MINT_BACKOFF_BASE = 0.3


class CopilotAuthSession:
    """Own Copilot credential lookup, short-lived token state, and token minting."""

    provider_name = "github-copilot"

    def __init__(self, credential_repository: CredentialRepository | None = None) -> None:
        self.credential_repository = credential_repository or CredentialRepository()
        self.auth_manager = AuthManager(self.credential_repository)
        self.cached_token: str | None = None
        self.token_expires: int = 0
        self.api_base = COPILOT_BASE_URL
        self.token_refresh_lock = asyncio.Lock()

    def is_authenticated(self) -> bool:
        cred = self.auth_manager.get_credential(self.provider_name)
        return cred is not None and cred.type == AuthType.OAUTH

    async def _mint_token(
        self,
        cred: OAuthCredential,
        mint: Callable[[httpx.AsyncClient, str], Awaitable[Any]],
    ) -> Any:
        """Mint a Copilot token, mapping HTTP errors to ProviderError.

        Raises the mapped ProviderError on failure; the retry policy lives in
        the caller (_mint_with_retry).
        """
        logger.debug("Refreshing Copilot token")
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            ) as client:
                return await mint(client, cred.refresh)
        except httpx.HTTPStatusError as error:
            if error.response.status_code in AUTH_RETRY_STATUSES:
                logger.error(
                    "GitHub Copilot authentication failed (%s)",
                    error.response.status_code,
                )
                raise ProviderError(
                    "GitHub Copilot authentication expired. If Copilot is your only "
                    "provider, re-authenticate: `router-maestro auth login github-copilot`.",
                    status_code=401,
                    retryable=True,
                    kind=ProviderFailureKind.AUTHENTICATION,
                    upstream_status_code=error.response.status_code,
                    provider=self.provider_name,
                    cause=error,
                ) from error
            logger.error(
                "Failed to refresh Copilot token: status=%d",
                error.response.status_code,
            )
            kind = (
                ProviderFailureKind.RATE_LIMIT
                if error.response.status_code in (429, 529)
                else ProviderFailureKind.UPSTREAM_STATUS
            )
            raise ProviderError(
                "Failed to refresh Copilot token",
                status_code=error.response.status_code,
                retryable=(error.response.status_code == 429 or error.response.status_code >= 500),
                kind=kind,
                upstream_status_code=error.response.status_code,
                provider=self.provider_name,
                cause=error,
            ) from error
        except (httpx.HTTPError, GitHubOAuthError) as error:
            logger.error("Failed to refresh Copilot token (%s)", type(error).__name__)
            raise ProviderError(
                "Failed to refresh Copilot token",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.TRANSPORT,
                provider=self.provider_name,
                cause=error,
            ) from error

    async def _mint_with_retry(
        self,
        cred: OAuthCredential,
        mint: Callable[[httpx.AsyncClient, str], Awaitable[Any]],
    ) -> Any:
        """Mint a token, retrying transient failures with bounded backoff.

        Only RATE_LIMIT / UPSTREAM_STATUS / TRANSPORT failures are retried;
        AUTHENTICATION (a genuinely expired credential) surfaces immediately.
        """
        attempt = 0
        while True:
            try:
                return await self._mint_token(cred, mint)
            except ProviderError as error:
                if error.kind not in _RETRYABLE_MINT_KINDS or attempt >= _MINT_MAX_RETRIES:
                    raise
                delay = _MINT_BACKOFF_BASE * (2**attempt) + random.uniform(0, 0.1)
                logger.warning(
                    "copilot_token_mint_retry attempt=%d kind=%s delay=%.2f",
                    attempt + 1,
                    error.kind.value,
                    delay,
                )
                await _async_sleep(delay)
                attempt += 1

    async def ensure_token(
        self,
        force: bool = False,
        *,
        persist: Callable[[OAuthCredential], Awaitable[None]] | None = None,
        mint: Callable[[httpx.AsyncClient, str], Awaitable[Any]] = get_copilot_token,
    ) -> None:
        """Ensure a usable Copilot token, serializing concurrent refreshes."""
        # Fast-path: a valid in-memory token needs no disk read. This must run
        # before get_credential() so a transient auth.json read failure cannot
        # turn a perfectly good cached token into a spurious auth error.
        current_time = int(time.time())
        if not force and self.cached_token and self.token_expires > current_time + 60:
            return

        cred = self.auth_manager.get_credential(self.provider_name)
        if not cred or not isinstance(cred, OAuthCredential):
            logger.error("Not authenticated with GitHub Copilot")
            raise ProviderError(
                "Not authenticated with GitHub Copilot",
                status_code=401,
                kind=ProviderFailureKind.AUTHENTICATION,
                provider=self.provider_name,
            )

        if cred.api_endpoint:
            self.api_base = cred.api_endpoint

        token_before_lock = self.cached_token
        async with self.token_refresh_lock:
            current_time = int(time.time())
            if not force and self.cached_token and self.token_expires > current_time + 60:
                return
            if force and self.cached_token and self.cached_token != token_before_lock:
                return

            cred = self.auth_manager.get_credential(self.provider_name)
            if not cred or not isinstance(cred, OAuthCredential):
                logger.error("Not authenticated with GitHub Copilot")
                raise ProviderError(
                    "Not authenticated with GitHub Copilot",
                    status_code=401,
                    kind=ProviderFailureKind.AUTHENTICATION,
                    provider=self.provider_name,
                )

            copilot_token = await self._mint_with_retry(cred, mint)

            self.cached_token = copilot_token.token
            self.token_expires = copilot_token.expires_at
            self.api_base = copilot_token.api_endpoint or self.api_base or COPILOT_BASE_URL
            credential = OAuthCredential(
                refresh=cred.refresh,
                access=copilot_token.token,
                expires=copilot_token.expires_at,
                api_endpoint=copilot_token.api_endpoint or cred.api_endpoint,
            )
            if persist is None:
                await self.persist_credential(credential)
            else:
                await persist(credential)
            logger.debug("Copilot token refreshed, expires at %d", copilot_token.expires_at)

    async def persist_credential(self, credential: OAuthCredential) -> None:
        """Store and flush a credential without blocking the event loop."""
        uses_legacy_storage = getattr(
            self.auth_manager,
            "uses_legacy_storage",
            not hasattr(self.auth_manager, "repository"),
        )
        if uses_legacy_storage:
            self.auth_manager.storage.set(self.provider_name, credential)
            await asyncio.to_thread(self.auth_manager.save)
            return
        await asyncio.to_thread(
            self.auth_manager.repository.update_provider,
            self.provider_name,
            credential,
        )

    async def refresh_for_auth_status(
        self,
        path: str,
        status_code: int,
        *,
        ensure_token: Callable[..., Awaitable[None]] | None = None,
    ) -> bool:
        if status_code not in AUTH_RETRY_STATUSES:
            return False
        logger.info(
            "Copilot %s returned %d; forcing token refresh and retrying",
            path,
            status_code,
        )
        refresh = ensure_token or self.ensure_token
        await refresh(force=True)
        return True

    def raise_auth_failure(
        self,
        path: str,
        status_code: int,
        *,
        model: str | None = None,
    ) -> NoReturn:
        logger.error("Copilot %s still returned %d after token refresh", path, status_code)
        raise ProviderError(
            f"Copilot authentication rejected ({status_code}) after refresh. If Copilot is "
            "your only provider, re-authenticate: `router-maestro auth login github-copilot`.",
            status_code=status_code,
            retryable=True,
            kind=ProviderFailureKind.AUTHENTICATION,
            upstream_status_code=status_code,
            provider=self.provider_name,
            model=model,
        )
