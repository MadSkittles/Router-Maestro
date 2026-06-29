"""Regression tests for API-key verification middleware.

Covers the constant-time comparison fix — in particular that a non-ASCII
bearer token is rejected with a clean 401 rather than crashing
``hmac.compare_digest`` with a TypeError (which would surface as a 500).
"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.datastructures import Headers
from starlette.requests import Request

from router_maestro.server.middleware.auth import verify_api_key


def _make_request(
    path: str = "/api/openai/v1/chat/completions",
    headers: dict[str, str] | None = None,
) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": Headers(headers or {}).raw,
    }
    return Request(scope)


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


@pytest.mark.asyncio
async def test_correct_key_passes(monkeypatch):
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "sk-correct")
    # Should not raise.
    await verify_api_key(_make_request(), _creds("sk-correct"))


@pytest.mark.asyncio
async def test_wrong_key_rejected_401(monkeypatch):
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "sk-correct")
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(_make_request(), _creds("sk-wrong"))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_non_ascii_key_rejected_401_not_500(monkeypatch):
    """A non-ASCII bearer token must yield 401, not a TypeError-driven 500."""
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "sk-correct")
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(_make_request(), _creds("café-key-中文-🔑"))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_health_endpoint_skips_auth(monkeypatch):
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "sk-correct")
    # No credentials, but /health is exempt — must not raise.
    await verify_api_key(_make_request("/health"), None)


@pytest.mark.asyncio
async def test_x_api_key_passes_despite_wrong_authorization(monkeypatch):
    """Claude Code sends an OAuth Authorization bearer plus the correct x-api-key.

    The correct x-api-key must be honored even when another credential header is
    present, otherwise the request 401s and the proxy cannot route.
    """
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "sk-correct")
    req = _make_request(
        path="/api/anthropic/v1/messages",
        headers={"x-api-key": "sk-correct"},
    )
    # Authorization bearer carries an unrelated/wrong OAuth token.
    await verify_api_key(req, _creds("sk-wrong-oauth"))


@pytest.mark.asyncio
async def test_x_goog_api_key_passes_despite_wrong_authorization(monkeypatch):
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "sk-correct")
    req = _make_request(headers={"x-goog-api-key": "sk-correct"})
    await verify_api_key(req, _creds("sk-wrong"))
