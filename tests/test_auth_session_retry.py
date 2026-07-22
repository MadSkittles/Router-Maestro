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
