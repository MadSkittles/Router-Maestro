"""Resilience contracts for CopilotAuthSession.ensure_token."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from router_maestro.auth.storage import OAuthCredential
from router_maestro.providers.base import ProviderError, ProviderFailureKind
from router_maestro.providers.copilot_support.auth_session import CopilotAuthSession


def _oauth(access: str = "copilot-token") -> OAuthCredential:
    return OAuthCredential(
        refresh="github-refresh",
        access=access,
        expires=2**31,
        api_endpoint="https://api.githubcopilot.com",
    )


def _session_with_stub_manager() -> tuple[CopilotAuthSession, MagicMock]:
    """A session whose auth_manager.get_credential is a controllable mock."""
    session = CopilotAuthSession.__new__(CopilotAuthSession)
    session.provider_name = "github-copilot"
    session.cached_token = None
    session.token_expires = 0
    session.api_base = "https://api.githubcopilot.com"
    import asyncio

    session.token_refresh_lock = asyncio.Lock()
    manager = MagicMock()
    session.auth_manager = manager
    return session, manager


@pytest.mark.asyncio
async def test_valid_cached_token_skips_disk_read():
    """A valid in-memory token must return without reading auth.json."""
    session, manager = _session_with_stub_manager()
    session.cached_token = "live-token"
    session.token_expires = int(time.time()) + 3600
    # If the fast-path is correct, get_credential is never consulted; make it
    # explode so any disk read fails the test.
    manager.get_credential.side_effect = AssertionError("disk read must not happen")

    await session.ensure_token()

    manager.get_credential.assert_not_called()
    assert session.cached_token == "live-token"


class _FlakyMint:
    """A mint callable that raises `errors[i]` on call i, else returns a token."""

    def __init__(self, errors: list[ProviderError | None]):
        self._errors = errors
        self.calls = 0

    async def __call__(self, _client, _refresh):
        idx = self.calls
        self.calls += 1
        err = self._errors[idx] if idx < len(self._errors) else None
        if err is not None:
            raise err
        return SimpleNamespace(
            token="minted-token",
            expires_at=2**31,
            api_endpoint="https://api.githubcopilot.com",
        )


def _transient(kind=ProviderFailureKind.UPSTREAM_STATUS) -> ProviderError:
    return ProviderError(
        "Failed to refresh Copilot token",
        status_code=502,
        retryable=True,
        kind=kind,
        provider="github-copilot",
    )


def _auth_error() -> ProviderError:
    return ProviderError(
        "GitHub Copilot authentication expired.",
        status_code=401,
        retryable=True,  # router-fallback flag; must NOT trigger local retry
        kind=ProviderFailureKind.AUTHENTICATION,
        provider="github-copilot",
    )


async def _noop_persist(_cred) -> None:
    return None


@pytest.fixture
def _fast_backoff(monkeypatch):
    """Make retry backoff instant so tests add no wall-clock delay."""
    from unittest.mock import AsyncMock

    from router_maestro.providers.copilot_support import auth_session as mod

    monkeypatch.setattr(mod, "_async_sleep", AsyncMock())


def _session_needs_refresh() -> tuple[CopilotAuthSession, MagicMock]:
    session, manager = _session_with_stub_manager()
    manager.get_credential.return_value = _oauth()
    manager.get_credential.side_effect = None
    return session, manager


@pytest.mark.asyncio
async def test_retry_rides_out_transient_mint_failures(_fast_backoff):
    """Two transient failures then success => ensure_token succeeds, mint x3."""
    session, _ = _session_needs_refresh()
    mint = _FlakyMint([_transient(), _transient(ProviderFailureKind.TRANSPORT), None])

    await session.ensure_token(persist=_noop_persist, mint=mint)

    assert mint.calls == 3
    assert session.cached_token == "minted-token"


@pytest.mark.asyncio
async def test_authentication_failure_is_not_retried(_fast_backoff):
    """A 401/403 mint error surfaces immediately, mint called exactly once."""
    session, _ = _session_needs_refresh()
    mint = _FlakyMint([_auth_error()])

    with pytest.raises(ProviderError) as exc:
        await session.ensure_token(persist=_noop_persist, mint=mint)

    assert exc.value.kind == ProviderFailureKind.AUTHENTICATION
    assert mint.calls == 1


@pytest.mark.asyncio
async def test_retry_exhaustion_reraises_transient_error(_fast_backoff):
    """Always-transient mint => raises after 1 + 3 retries = 4 attempts."""
    session, _ = _session_needs_refresh()
    mint = _FlakyMint([_transient(), _transient(), _transient(), _transient(), _transient()])

    with pytest.raises(ProviderError) as exc:
        await session.ensure_token(persist=_noop_persist, mint=mint)

    assert exc.value.retryable is True
    assert mint.calls == 4
