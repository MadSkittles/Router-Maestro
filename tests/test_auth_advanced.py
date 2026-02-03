"""Tests for auth storage module."""

import json
import tempfile
from pathlib import Path

import pytest

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
            storage.set("github-copilot", OAuthCredential(
                refresh="refresh",
                access="access",
                expires=12345,
            ))
            storage.save(path)

            # Load and verify
            loaded = AuthStorage.load(path)

            openai_cred = loaded.get("openai")
            assert openai_cred is not None
            assert openai_cred.type == AuthType.API_KEY

            copilot_cred = loaded.get("github-copilot")
            assert copilot_cred is not None
            assert copilot_cred.type == AuthType.OAUTH
        finally:
            path.unlink(missing_ok=True)

    def test_load_nonexistent_returns_empty(self):
        """Test loading from nonexistent file returns empty storage."""
        storage = AuthStorage.load(Path("/nonexistent/path/auth.json"))
        assert storage.credentials == {}
