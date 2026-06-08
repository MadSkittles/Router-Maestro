"""Tests for authentication module."""

import tempfile
from pathlib import Path

from router_maestro.auth.storage import ApiKeyCredential, AuthStorage, AuthType, OAuthCredential


class TestAuthStorage:
    """Tests for AuthStorage."""

    def test_empty_storage(self):
        """Test empty storage."""
        storage = AuthStorage()
        assert storage.list_providers() == []
        assert storage.get("nonexistent") is None

    def test_set_and_get_oauth(self):
        """Test setting and getting OAuth credentials."""
        storage = AuthStorage()

        cred = OAuthCredential(
            refresh="refresh_token",
            access="access_token",
            expires=12345,
        )
        storage.set("github-copilot", cred)

        retrieved = storage.get("github-copilot")
        assert retrieved is not None
        assert retrieved.type == AuthType.OAUTH
        assert retrieved.refresh == "refresh_token"
        assert retrieved.access == "access_token"

    def test_set_and_get_api_key(self):
        """Test setting and getting API key credentials."""
        storage = AuthStorage()

        cred = ApiKeyCredential(key="test-api-key")
        storage.set("openai", cred)

        retrieved = storage.get("openai")
        assert retrieved is not None
        assert retrieved.type == AuthType.API_KEY
        assert retrieved.key == "test-api-key"

    def test_remove(self):
        """Test removing credentials."""
        storage = AuthStorage()
        storage.set("openai", ApiKeyCredential(key="test"))

        assert storage.remove("openai") is True
        assert storage.get("openai") is None
        assert storage.remove("openai") is False

    def test_list_providers(self):
        """Test listing providers."""
        storage = AuthStorage()
        storage.set("openai", ApiKeyCredential(key="key1"))
        storage.set("anthropic", ApiKeyCredential(key="key2"))

        providers = storage.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert len(providers) == 2

    def test_save_and_load(self):
        """Test saving and loading storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "auth.json"

            # Create and save
            storage = AuthStorage()
            storage.set("openai", ApiKeyCredential(key="test-key"))
            storage.set(
                "github-copilot",
                OAuthCredential(
                    refresh="refresh",
                    access="access",
                    expires=12345,
                ),
            )
            storage.save(path)

            # Load and verify
            loaded = AuthStorage.load(path)
            assert loaded.list_providers() == storage.list_providers()

            openai_cred = loaded.get("openai")
            assert openai_cred is not None
            assert openai_cred.key == "test-key"

            copilot_cred = loaded.get("github-copilot")
            assert copilot_cred is not None
            assert copilot_cred.refresh == "refresh"

    def test_load_corrupt_json_returns_empty(self):
        """A syntactically invalid auth file must not crash — load empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "auth.json"
            path.write_text("{ not valid json", encoding="utf-8")
            loaded = AuthStorage.load(path)
            assert loaded.list_providers() == []

    def test_load_non_dict_toplevel_returns_empty(self):
        """A valid JSON whose top level isn't an object must not crash."""
        for payload in ("[]", '"foo"', "null", "42"):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "auth.json"
                path.write_text(payload, encoding="utf-8")
                loaded = AuthStorage.load(path)
                assert loaded.list_providers() == [], f"crashed/loaded on {payload!r}"

    def test_load_skips_invalid_and_unknown_credentials(self):
        """Malformed or unknown-type entries are skipped, valid ones survive."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "auth.json"
            path.write_text(
                json.dumps(
                    {
                        "good": {"type": "api", "key": "k"},
                        "bad": {"type": "oauth"},  # missing required fields
                        "weird": {"type": "mystery"},  # unknown type
                        "notdict": "scalar",
                    }
                ),
                encoding="utf-8",
            )
            loaded = AuthStorage.load(path)
            assert loaded.list_providers() == ["good"]
