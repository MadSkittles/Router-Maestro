"""Tests for auth storage module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from router_maestro.auth.github_oauth import (
    AccessTokenResponse,
    CopilotTokenResponse,
    DeviceCodeResponse,
    get_copilot_token,
)
from router_maestro.auth.manager import AuthManager
from router_maestro.auth.storage import (
    ApiKeyCredential,
    AuthStorage,
    AuthType,
    OAuthCredential,
)


class TestAuthType:
    """Tests for AuthType enum."""

    def test_oauth_type(self):
        """Test OAuth auth type."""
        assert AuthType.OAUTH == "oauth"

    def test_api_key_type(self):
        """Test API key auth type."""
        assert AuthType.API_KEY == "api"


class TestOAuthCredential:
    """Tests for OAuthCredential."""

    def test_basic_credential(self):
        """Test basic OAuth credential."""
        cred = OAuthCredential(
            refresh="refresh-token",
            access="access-token",
            expires=1234567890,
        )
        assert cred.type == AuthType.OAUTH
        assert cred.refresh == "refresh-token"
        assert cred.access == "access-token"
        assert cred.expires == 1234567890

    def test_default_expires(self):
        """Test default expires value."""
        cred = OAuthCredential(
            refresh="refresh-token",
            access="access-token",
        )
        assert cred.expires == 0

    def test_copilot_api_endpoint_metadata(self):
        """OAuth credentials can persist the Copilot API endpoint returned with the token."""
        cred = OAuthCredential(
            refresh="refresh-token",
            access="access-token",
            expires=1234567890,
            api_endpoint="https://api.enterprise.githubcopilot.com",
        )

        assert cred.api_endpoint == "https://api.enterprise.githubcopilot.com"


class TestGitHubOAuth:
    """Tests for GitHub Copilot OAuth token exchange."""

    @pytest.mark.asyncio
    async def test_get_copilot_token_includes_endpoint_metadata(self):
        """Copilot token exchange keeps endpoint metadata for model calls."""
        response = httpx.Response(
            200,
            json={
                "token": "copilot-token",
                "expires_at": 1234567890,
                "refresh_in": 1000,
                "endpoints": {"api": "https://api.enterprise.githubcopilot.com"},
            },
            request=httpx.Request("GET", "https://api.github.com/copilot_internal/v2/token"),
        )
        client = AsyncMock()
        client.get.return_value = response

        token = await get_copilot_token(client, "github-token")

        assert token.token == "copilot-token"
        assert token.api_endpoint == "https://api.enterprise.githubcopilot.com"

    @pytest.mark.asyncio
    async def test_get_copilot_token_retries_transient_errors(self):
        """Transient refresh failures should be retried before surfacing to callers."""
        ok = httpx.Response(
            200,
            json={
                "token": "copilot-token",
                "expires_at": 1234567890,
                "refresh_in": 1000,
            },
            request=httpx.Request("GET", "https://api.github.com/copilot_internal/v2/token"),
        )
        client = AsyncMock()
        client.get.side_effect = [
            httpx.ConnectError("temporary"),
            ok,
        ]

        with patch("router_maestro.auth.github_oauth._async_sleep", new=AsyncMock()):
            token = await get_copilot_token(client, "github-token")

        assert token.token == "copilot-token"
        assert client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_copilot_token_does_not_retry_auth_errors(self):
        """Permanent auth failures should surface immediately instead of being retried."""
        response = httpx.Response(
            401,
            json={"message": "bad credentials"},
            request=httpx.Request("GET", "https://api.github.com/copilot_internal/v2/token"),
        )
        client = AsyncMock()
        client.get.return_value = response
        sleep = AsyncMock()

        with (
            patch("router_maestro.auth.github_oauth._async_sleep", new=sleep),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await get_copilot_token(client, "github-token")

        assert client.get.call_count == 1
        sleep.assert_not_called()


class TestAuthManager:
    """Tests for provider auth flows."""

    @pytest.mark.asyncio
    async def test_login_copilot_persists_endpoint_metadata(self):
        """Initial Copilot login should persist the API endpoint returned with the token."""
        manager = AuthManager.__new__(AuthManager)
        manager.storage = AuthStorage()
        manager.save = Mock()  # type: ignore[method-assign]

        with (
            patch(
                "router_maestro.auth.manager.request_device_code",
                new=AsyncMock(
                    return_value=DeviceCodeResponse(
                        device_code="device-code",
                        user_code="user-code",
                        verification_uri="https://github.com/login/device",
                        expires_in=900,
                        interval=1,
                    )
                ),
            ),
            patch(
                "router_maestro.auth.manager.poll_access_token",
                new=AsyncMock(
                    return_value=AccessTokenResponse(
                        access_token="github-token",
                        token_type="bearer",
                        scope="read:user",
                    )
                ),
            ),
            patch(
                "router_maestro.auth.manager.get_copilot_token",
                new=AsyncMock(
                    return_value=CopilotTokenResponse(
                        token="copilot-token",
                        expires_at=1234567890,
                        refresh_in=1000,
                        api_endpoint="https://api.enterprise.githubcopilot.com",
                    )
                ),
            ),
        ):
            assert await manager.login_copilot()

        cred = manager.storage.get("github-copilot")
        assert isinstance(cred, OAuthCredential)
        assert cred.api_endpoint == "https://api.enterprise.githubcopilot.com"
        manager.save.assert_called_once()


class TestApiKeyCredential:
    """Tests for ApiKeyCredential."""

    def test_basic_credential(self):
        """Test basic API key credential."""
        cred = ApiKeyCredential(key="sk-test-key")
        assert cred.type == AuthType.API_KEY
        assert cred.key == "sk-test-key"


class TestAuthStorage:
    """Tests for AuthStorage."""

    def test_empty_storage(self):
        """Test empty auth storage."""
        storage = AuthStorage()
        assert storage.credentials == {}

    def test_set_and_get(self):
        """Test setting and getting credentials."""
        storage = AuthStorage()
        cred = ApiKeyCredential(key="test-key")
        storage.set("openai", cred)

        retrieved = storage.get("openai")
        assert retrieved is not None
        assert retrieved.key == "test-key"

    def test_get_nonexistent(self):
        """Test getting nonexistent credential."""
        storage = AuthStorage()
        assert storage.get("nonexistent") is None

    def test_remove_existing(self):
        """Test removing existing credential."""
        storage = AuthStorage()
        storage.set("openai", ApiKeyCredential(key="key"))

        result = storage.remove("openai")
        assert result is True
        assert storage.get("openai") is None

    def test_remove_nonexistent(self):
        """Test removing nonexistent credential."""
        storage = AuthStorage()
        result = storage.remove("nonexistent")
        assert result is False

    def test_list_providers(self):
        """Test listing providers."""
        storage = AuthStorage()
        storage.set("openai", ApiKeyCredential(key="key1"))
        storage.set("anthropic", ApiKeyCredential(key="key2"))

        providers = storage.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_save_and_load(self):
        """Test saving and loading storage."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Create and save storage
            storage = AuthStorage()
            storage.set("openai", ApiKeyCredential(key="test-key"))
            storage.set(
                "github-copilot",
                OAuthCredential(
                    refresh="refresh",
                    access="access",
                    expires=12345,
                    api_endpoint="https://api.enterprise.githubcopilot.com",
                ),
            )
            storage.save(path)

            # Load and verify
            loaded = AuthStorage.load(path)

            openai_cred = loaded.get("openai")
            assert openai_cred is not None
            assert openai_cred.type == AuthType.API_KEY

            copilot_cred = loaded.get("github-copilot")
            assert copilot_cred is not None
            assert copilot_cred.type == AuthType.OAUTH
            assert copilot_cred.api_endpoint == "https://api.enterprise.githubcopilot.com"
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.skipif(
        not hasattr(os, "fchmod"), reason="POSIX permission bits not applicable on this platform"
    )
    def test_save_writes_owner_only_permissions(self, tmp_path):
        """auth.json contains tokens and API keys, so it must not be group/world-readable."""
        path = tmp_path / "auth.json"
        storage = AuthStorage()
        storage.set("openai", ApiKeyCredential(key="test-key"))

        with patch("os.umask", return_value=0):
            storage.save(path)

        assert path.stat().st_mode & 0o777 == 0o600

    def test_load_nonexistent_returns_empty(self):
        """Test loading from nonexistent file returns empty storage."""
        storage = AuthStorage.load(Path("/nonexistent/path/auth.json"))
        assert storage.credentials == {}
