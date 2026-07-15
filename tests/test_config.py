"""Tests for configuration module."""

import json
import os
import tempfile
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
import tomlkit
from rich.console import Console

from router_maestro.cli import config as cli_config
from router_maestro.cli.config import (
    _OPUS_1M_NATIVE_KEY,
    _OPUS_1M_SOURCE_MODEL,
    _OPUS_47_1M_NATIVE_KEY,
    _OPUS_48_1M_NATIVE_KEY,
    _SONNET_46_1M_NATIVE_KEY,
    _display_models,
    _maybe_inject_opus_1m,
    _model_key,
    _prompt_auto_compact_window,
    _prompt_endpoint_mode,
    _select_model,
)
from router_maestro.config import settings
from router_maestro.config.contexts import ContextConfig, ContextsConfig
from router_maestro.config.providers import CustomProviderConfig, ModelConfig, ProvidersConfig
from router_maestro.config.settings import load_config, save_config


class TestProvidersConfig:
    """Tests for ProvidersConfig."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = ProvidersConfig.get_default()

        # Default config should be empty (no custom providers)
        assert config.providers == {}

    def test_model_config(self):
        """Test ModelConfig creation."""
        model = ModelConfig(name="Test Model")

        assert model.name == "Test Model"

    def test_custom_provider_config(self):
        """Test CustomProviderConfig creation."""
        provider = CustomProviderConfig(
            type="openai-compatible",
            baseURL="https://api.custom.com/v1",
            models={"custom-model": ModelConfig(name="Custom Model")},
        )

        assert provider.type == "openai-compatible"
        assert provider.baseURL == "https://api.custom.com/v1"
        assert "custom-model" in provider.models


class TestContextsConfig:
    """Tests for ContextsConfig."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = ContextsConfig.get_default()

        assert config.current == "local"
        assert "local" in config.contexts
        assert config.contexts["local"].endpoint == "http://localhost:8080"

    def test_context_config(self):
        """Test ContextConfig creation."""
        ctx = ContextConfig(endpoint="https://example.com", api_key="test-key")

        assert ctx.endpoint == "https://example.com"
        assert ctx.api_key == "test-key"


class TestConfigIO:
    """Tests for configuration I/O."""

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.json"

            # Create and save config
            original = ProvidersConfig.get_default()
            save_config(path, original)

            # Verify file exists
            assert path.exists()

            # Load and verify
            loaded = load_config(path, ProvidersConfig, ProvidersConfig.get_default)
            assert loaded.providers.keys() == original.providers.keys()

    @pytest.mark.skipif(
        not hasattr(os, "fchmod"), reason="POSIX permission bits not applicable on this platform"
    )
    def test_save_config_writes_owner_only_permissions(self, tmp_path):
        """Config files may contain API keys and should be owner-readable only."""
        path = tmp_path / "contexts.json"
        config = ContextsConfig(
            current="local",
            contexts={"local": ContextConfig(endpoint="http://localhost:8080", api_key="sk-test")},
        )

        with patch("os.umask", return_value=0):
            save_config(path, config)

        assert path.stat().st_mode & 0o777 == 0o600

    def test_load_creates_default(self):
        """Test that loading non-existent file creates default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"

            config = load_config(path, ContextsConfig, ContextsConfig.get_default)

            assert config.current == "local"
            assert path.exists()  # Should have created the file

    def test_load_corrupt_json_falls_back_to_default(self, tmp_path):
        """A truncated/corrupt config file must not crash startup."""
        path = tmp_path / "contexts.json"
        path.write_text("{ this is not valid json", encoding="utf-8")

        config = load_config(path, ContextsConfig, ContextsConfig.get_default)
        assert config.current == "local"  # default

    def test_load_empty_file_falls_back_to_default(self, tmp_path):
        """An empty file (e.g. from an interrupted write) loads defaults."""
        path = tmp_path / "contexts.json"
        path.write_text("", encoding="utf-8")

        config = load_config(path, ContextsConfig, ContextsConfig.get_default)
        assert config.current == "local"

    def test_load_validation_error_does_not_log_input_values(self, tmp_path, caplog):
        """Validation errors can include secret values and must not log them."""
        path = tmp_path / "contexts.json"
        secret = "sk-secret-from-invalid-config"
        path.write_text(
            json.dumps(
                {
                    "current": "local",
                    "contexts": {"local": {"endpoint": {"raw": secret}}},
                }
            ),
            encoding="utf-8",
        )

        config = load_config(path, ContextsConfig, ContextsConfig.get_default)

        assert config.current == "local"
        assert secret not in caplog.text
        assert "input_value" not in caplog.text
        assert "contexts.local.endpoint" not in caplog.text

    def test_write_is_atomic_no_partial_file_on_crash(self, tmp_path):
        """If json.dump raises mid-write, the existing file is left intact."""
        import pytest

        from router_maestro.config.settings import write_json_owner_only

        path = tmp_path / "data.json"
        write_json_owner_only(path, {"good": True})

        class _Unserializable:
            pass

        # Serialization must fail, and the original file must survive unchanged.
        with pytest.raises(TypeError):
            write_json_owner_only(path, {"bad": _Unserializable()})

        assert json.loads(path.read_text()) == {"good": True}
        # No leftover temp files in the directory.
        assert list(tmp_path.glob("*.tmp")) == []


class TestProvidersConfigIO:
    """Compatibility and isolation behavior for on-disk custom providers."""

    @staticmethod
    def _provider(*, options=None):
        return {
            "type": "openai-compatible",
            "baseURL": "https://example.invalid/v1",
            "models": {"model": {"name": "Model"}},
            "options": options or {},
        }

    @staticmethod
    def _write(path: Path, providers) -> bytes:
        path.write_text(json.dumps({"providers": providers}), encoding="utf-8")
        return path.read_bytes()

    def test_missing_providers_file_creates_default(self, tmp_path, monkeypatch):
        path = tmp_path / "providers.json"
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        config = settings.load_providers_config()

        assert config.providers == {}
        assert json.loads(path.read_text(encoding="utf-8")) == {"providers": {}}

    def test_unknown_options_round_trip_without_load_rewriting_source(
        self,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        path = tmp_path / "providers.json"
        original = self._write(
            path,
            {
                "healthy": self._provider(options={"allow_unauthenticated": True}),
                "legacy": self._provider(
                    options={
                        "api_key_env": "LEGACY_API_KEY",
                        "request_timeout": 45,
                        "compatibility": {"mode": "old"},
                    }
                ),
            },
        )
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        loaded = settings.load_providers_config()

        assert set(loaded.providers) == {"healthy", "legacy"}
        assert path.read_bytes() == original
        assert loaded.providers["legacy"].options.model_dump(mode="json") == {
            "api_key_env": "LEGACY_API_KEY",
            "allow_unauthenticated": False,
            "request_timeout": 45,
            "compatibility": {"mode": "old"},
        }
        assert "legacy" in caplog.text
        assert "compatibility" in caplog.text
        assert "request_timeout" in caplog.text
        assert "mode" not in caplog.text

        settings.save_providers_config(loaded)
        reloaded = settings.load_providers_config()

        assert reloaded.model_dump(mode="json") == loaded.model_dump(mode="json")

    def test_unknown_option_log_escapes_control_characters_without_values(
        self,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        path = tmp_path / "providers.json"
        legacy_key = "legacy\nFORGED\toption"
        secret = "sk-secret-legacy-option-value"
        self._write(
            path,
            {
                "legacy": self._provider(
                    options={legacy_key: {"nested-secret": secret}},
                )
            },
        )
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        loaded = settings.load_providers_config()

        assert loaded.providers["legacy"].options.model_dump(mode="json")[legacy_key] == {
            "nested-secret": secret
        }
        assert repr(legacy_key) in caplog.text
        assert "legacy\nFORGED" not in caplog.text
        assert secret not in caplog.text
        assert "nested-secret" not in caplog.text

    def test_invalid_known_option_skips_only_that_provider_with_safe_diagnostics(
        self,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        path = tmp_path / "providers.json"
        secret = "sk-secret-invalid-env-name"
        original = self._write(
            path,
            {
                "healthy": self._provider(options={"allow_unauthenticated": True}),
                "broken": self._provider(options={"api_key_env": secret}),
            },
        )
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        loaded = settings.load_providers_config()

        assert set(loaded.providers) == {"healthy"}
        assert path.read_bytes() == original
        assert "broken" in caplog.text
        assert "options.api_key_env" in caplog.text
        assert "string_pattern_mismatch" in caplog.text
        assert secret not in caplog.text
        assert "input_value" not in caplog.text

    def test_validation_log_redacts_dynamic_model_id_from_location(
        self,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        path = tmp_path / "providers.json"
        secret = "sk-secret-model-id"
        malicious_model_id = f"model\nFORGED\t{secret}"
        provider = self._provider()
        provider["models"] = {malicious_model_id: {"name": {"invalid": True}}}
        self._write(path, {"broken": provider})
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        loaded = settings.load_providers_config()

        assert loaded.providers == {}
        assert "models.<model-id>.name:string_type" in caplog.text
        assert malicious_model_id not in caplog.text
        assert "model\nFORGED" not in caplog.text
        assert secret not in caplog.text
        assert "input_value" not in caplog.text

    def test_reserved_and_duplicate_names_are_isolated_deterministically(
        self,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        path = tmp_path / "providers.json"
        self._write(
            path,
            {
                "healthy": self._provider(),
                "OpenAI": self._provider(),
                "First": self._provider(options={"api_key_env": "FIRST_API_KEY"}),
                "first": self._provider(options={"api_key_env": "SECOND_API_KEY"}),
                "later": self._provider(),
            },
        )
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        loaded = settings.load_providers_config()

        assert list(loaded.providers) == ["healthy", "First", "later"]
        assert loaded.providers["First"].options.api_key_env == "FIRST_API_KEY"
        assert "OpenAI" in caplog.text
        assert "reserved_provider_name" in caplog.text
        assert "first" in caplog.text
        assert "duplicate_provider_name" in caplog.text

    @pytest.mark.parametrize(
        "contents",
        [
            "{this is not json",
            json.dumps(["not", "an", "object"]),
            json.dumps({"providers": ["not", "a", "mapping"]}),
        ],
    )
    def test_malformed_provider_documents_fall_back_without_rewriting(
        self,
        contents,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        path = tmp_path / "providers.json"
        path.write_text(contents, encoding="utf-8")
        original = path.read_bytes()
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        loaded = settings.load_providers_config()

        assert loaded.providers == {}
        assert path.read_bytes() == original
        assert contents not in caplog.text

    def test_non_utf8_provider_document_falls_back_without_rewriting(
        self,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        path = tmp_path / "providers.json"
        original = b"\xff\xfeinvalid-provider-config"
        path.write_bytes(original)
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        loaded = settings.load_providers_config()

        assert loaded.providers == {}
        assert path.read_bytes() == original
        assert "invalid_encoding" in caplog.text
        assert "invalid-provider-config" not in caplog.text

    def test_provider_file_os_error_falls_back_without_replacing_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        path = tmp_path / "providers.json"
        path.mkdir()
        monkeypatch.setattr(settings, "PROVIDERS_FILE", path)

        loaded = settings.load_providers_config()

        assert loaded.providers == {}
        assert path.is_dir()


class TestSelectModel:
    """Tests for _select_model helper in CLI config."""

    def _make_models(self):
        return [
            {"provider": "github-copilot", "id": "gpt-4o", "name": "GPT-4o"},
            {"provider": "github-copilot", "id": "claude-opus-4.6", "name": "Claude Opus 4.6"},
        ]

    def test_returns_provider_id(self, monkeypatch):
        """Standard selection returns provider/id format."""
        monkeypatch.setattr("router_maestro.cli.config.Prompt.ask", lambda *a, **kw: "1")
        result = _select_model(self._make_models(), "Pick")
        assert result == "github-copilot/gpt-4o"

    def test_qualified_server_id_is_not_double_prefixed(self, monkeypatch):
        monkeypatch.setattr("router_maestro.cli.config.Prompt.ask", lambda *a, **kw: "1")
        models = [
            {
                "provider": "github-copilot",
                "id": "github-copilot/gpt-5.5",
                "name": "GPT-5.5",
            }
        ]

        assert _select_model(models, "Pick") == "github-copilot/gpt-5.5"
        assert _model_key(models[0]) == "github-copilot/gpt-5.5"

    def test_display_uses_qualified_server_id_once(self, monkeypatch):
        output = Console(record=True, width=120)
        monkeypatch.setattr(cli_config, "console", output)

        _display_models(
            [
                {
                    "provider": "github-copilot",
                    "id": "github-copilot/gpt-5.5",
                    "name": "GPT-5.5",
                }
            ]
        )

        rendered = output.export_text()
        assert "github-copilot/gpt-5.5" in rendered
        assert "github-copilot/github-copilot/gpt-5.5" not in rendered

    def test_returns_custom_key(self, monkeypatch):
        """Selection of model with custom_key returns the custom key."""
        models = [
            *self._make_models(),
            {
                "provider": "github-copilot",
                "id": "claude-opus-4.6-1m",
                "name": "Opus 4.6 1M (Auto-activated)",
                "custom_key": _OPUS_1M_NATIVE_KEY,
            },
        ]
        monkeypatch.setattr("router_maestro.cli.config.Prompt.ask", lambda *a, **kw: "3")
        result = _select_model(models, "Pick")
        assert result == _OPUS_1M_NATIVE_KEY

    def test_injected_1m_model_keeps_native_display_and_provider_qualified_wire_key(
        self,
        monkeypatch,
    ):
        models = _maybe_inject_opus_1m(
            [
                {
                    "provider": "github-copilot",
                    "id": "claude-opus-4.6",
                    "name": "Claude Opus 4.6",
                    "max_context_window_tokens": 1_000_000,
                }
            ]
        )
        synthetic = models[0]
        monkeypatch.setattr("router_maestro.cli.config.Prompt.ask", lambda *a, **kw: "1")

        assert synthetic["display_key"] == _OPUS_1M_NATIVE_KEY
        assert _model_key(synthetic) == f"github-copilot/{_OPUS_1M_NATIVE_KEY}"
        assert _select_model(models, "Pick") == f"github-copilot/{_OPUS_1M_NATIVE_KEY}"

    def test_auto_routing(self, monkeypatch):
        """Choice 0 returns router-maestro for auto-routing."""
        monkeypatch.setattr("router_maestro.cli.config.Prompt.ask", lambda *a, **kw: "0")
        result = _select_model(self._make_models(), "Pick")
        assert result == "router-maestro"

    def test_out_of_bounds_falls_back_to_auto_routing(self, monkeypatch):
        """Out-of-range selection falls back to auto-routing."""
        monkeypatch.setattr("router_maestro.cli.config.Prompt.ask", lambda *a, **kw: "99")
        result = _select_model(self._make_models(), "Pick")
        assert result == "router-maestro"

    def test_non_numeric_input_falls_back_to_auto_routing(self, monkeypatch):
        """Non-numeric input falls back to auto-routing."""
        monkeypatch.setattr("router_maestro.cli.config.Prompt.ask", lambda *a, **kw: "gpt-4o")
        result = _select_model(self._make_models(), "Pick")
        assert result == "router-maestro"


class TestMaybeInjectOpus1M:
    """Tests for _maybe_inject_opus_1m — the production injection function."""

    def test_prepends_synthetic_entry_when_1m_model_present(self):
        """Synthetic entry is prepended when the source model exists."""
        models = [
            {
                "provider": "github-copilot",
                "id": "claude-opus-4.6",
                "name": "Claude Opus 4.6",
                "max_context_window_tokens": 1_000_000,
            },
        ]

        result = _maybe_inject_opus_1m(models)

        assert len(result) == 2
        assert len(models) == 1  # original list not mutated
        synthetic = result[0]
        assert synthetic["wire_key"] == f"github-copilot/{_OPUS_1M_NATIVE_KEY}"
        assert synthetic["display_key"] == _OPUS_1M_NATIVE_KEY
        assert synthetic["name"] == "Opus 4.6 1M (Auto-activated)"
        assert synthetic["provider"] == "github-copilot"
        assert synthetic["id"] == "claude-opus-4.6"

    @pytest.mark.parametrize(
        "context_metadata",
        [
            {"max_context_window_tokens": 1_000_000},
            {"max_prompt_tokens": 936_000, "max_output_tokens": 64_000},
        ],
        ids=["explicit-context-window", "prompt-plus-output"],
    )
    def test_qualified_server_source_with_1m_metadata_injects_synthetic_entry(
        self,
        context_metadata,
    ):
        models = [
            {
                "provider": "github-copilot",
                "id": "github-copilot/claude-opus-4.6",
                "name": "Claude Opus 4.6",
                **context_metadata,
            }
        ]

        result = _maybe_inject_opus_1m(models)

        assert result[0]["wire_key"] == f"github-copilot/{_OPUS_1M_NATIVE_KEY}"
        assert result[0]["id"] == "claude-opus-4.6"
        assert _model_key(result[0]) == f"github-copilot/{_OPUS_1M_NATIVE_KEY}"

    def test_source_without_advertised_1m_context_is_not_injected(self):
        models = [
            {
                "provider": "github-copilot",
                "id": "claude-opus-4.6",
                "name": "Claude Opus 4.6",
                "max_context_window_tokens": 999_999,
            }
        ]

        result = _maybe_inject_opus_1m(models)

        assert result is models

    def test_no_injection_when_1m_model_absent(self):
        """No synthetic entry when the source model is not in the list."""
        models = [
            {"provider": "github-copilot", "id": "gpt-4.1", "name": "GPT-4.1"},
            {"provider": "github-copilot", "id": "gpt-4o", "name": "GPT-4o"},
        ]

        result = _maybe_inject_opus_1m(models)

        assert len(result) == 2
        assert result is models  # same list returned, no copy needed

    def test_does_not_mutate_input_list(self):
        """The input list is never mutated."""
        models = [
            {
                "provider": "github-copilot",
                "id": "claude-opus-4.6",
                "name": "Claude Opus 4.6",
            },
        ]
        original_len = len(models)

        _maybe_inject_opus_1m(models)

        assert len(models) == original_len

    def test_source_model_constant_matches_expected_value(self):
        """Guard against accidental changes to the source model constant.

        Copilot dropped the dedicated ``-1m`` Opus variants, so the 4.6 native
        key now maps to the base catalog id like 4.8 / sonnet-4.6 do.
        """
        assert _OPUS_1M_SOURCE_MODEL == "github-copilot/claude-opus-4.6"
        assert _OPUS_1M_NATIVE_KEY == "claude-opus-4-6[1m]"

    def test_injects_opus_48_and_sonnet_46_from_base_ids(self):
        """4.8 and sonnet-4.6 have no dedicated -1m variant; the synthetic
        entries map their [1m] native key straight to the base catalog id."""
        models = [
            {
                "provider": "github-copilot",
                "id": "claude-opus-4.8",
                "name": "Claude Opus 4.8",
                "max_context_window_tokens": 1_000_000,
            },
            {
                "provider": "github-copilot",
                "id": "claude-sonnet-4.6",
                "name": "Claude Sonnet 4.6",
                "max_context_window_tokens": 1_000_000,
            },
        ]

        result = _maybe_inject_opus_1m(models)

        # Two new synthetic entries appear before the originals.
        assert len(result) == 4
        display_keys = {m.get("display_key") for m in result if "display_key" in m}
        assert _OPUS_48_1M_NATIVE_KEY in display_keys
        assert _SONNET_46_1M_NATIVE_KEY in display_keys
        # The synthetic entries point at the base ids — there is no -1m suffix
        # on the catalog side for these.
        synthetic_by_key = {m["display_key"]: m for m in result if "display_key" in m}
        assert synthetic_by_key[_OPUS_48_1M_NATIVE_KEY]["id"] == "claude-opus-4.8"
        assert synthetic_by_key[_SONNET_46_1M_NATIVE_KEY]["id"] == "claude-sonnet-4.6"


@pytest.mark.parametrize(
    "model_id",
    ["claude-opus-4.6", "github-copilot/claude-opus-4.6"],
    ids=["legacy-bare", "qualified-server"],
)
def test_claude_endpoint_prompt_recognizes_bare_upstream_id(monkeypatch, model_id):
    monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: "2")

    assert (
        _prompt_endpoint_mode(
            {
                "provider": "github-copilot",
                "id": model_id,
                "name": "Claude Opus 4.6",
            }
        )
        is True
    )


class TestPromptAutoCompactWindow:
    """Tests for ``_prompt_auto_compact_window`` — the Claude Code auto-compact
    env var selection. Native 1M model keys must offer 1M as the default; every
    other model must fall back to the 200K default.
    """

    @staticmethod
    def _synthetic(native_key: str) -> dict:
        """A model dict shaped like the synthetic entries _maybe_inject_opus_1m emits."""
        return {
            "provider": "github-copilot",
            "id": "ignored-base-id",
            "display_key": native_key,
            "wire_key": f"github-copilot/{native_key}",
            "name": "test",
        }

    def test_returns_none_when_model_is_none(self):
        assert _prompt_auto_compact_window(None) is None

    def test_user_skip_returns_none(self, monkeypatch):
        monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: "n")
        assert _prompt_auto_compact_window(self._synthetic(_OPUS_1M_NATIVE_KEY)) is None

    def test_opus_48_native_key_defaults_to_1m(self, monkeypatch):
        """4.8 has no -1m catalog variant, but its [1m] native key must still
        unlock the 1M default in the auto-compact prompt."""
        monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: "d")
        assert _prompt_auto_compact_window(self._synthetic(_OPUS_48_1M_NATIVE_KEY)) == 1_000_000

    def test_sonnet_46_native_key_defaults_to_1m(self, monkeypatch):
        """Same as 4.8 — sonnet-4.6 ships only the base id but [1m] gets 1M."""
        monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: "d")
        assert _prompt_auto_compact_window(self._synthetic(_SONNET_46_1M_NATIVE_KEY)) == 1_000_000

    def test_opus_46_and_47_native_keys_defaults_to_1m(self, monkeypatch):
        """Regression guard: the pre-existing 4.6 / 4.7 native keys keep their
        1M default after adding 4.8 / sonnet-4.6 to the set."""
        monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: "d")
        assert _prompt_auto_compact_window(self._synthetic(_OPUS_1M_NATIVE_KEY)) == 1_000_000
        assert _prompt_auto_compact_window(self._synthetic(_OPUS_47_1M_NATIVE_KEY)) == 1_000_000

    def test_non_native_model_defaults_to_200k(self, monkeypatch):
        """A plain catalog model (no [1m] native key) gets the 200K default."""
        monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: "d")
        plain = {"provider": "github-copilot", "id": "claude-opus-4.8", "name": "Claude Opus 4.8"}
        assert _prompt_auto_compact_window(plain) == 200_000

    def test_native_1m_prompt_omits_upstream_choice(self, monkeypatch):
        """For native 1M keys we must not offer ``y = upstream``; the Copilot
        catalog's prompt cap (~936K) is below Claude Code's own 1M view, and
        using it would arm auto-compact earlier than the user expects."""
        captured = {}

        def fake_ask(prompt_text, choices=None, default=None):
            captured["choices"] = choices
            return "d"

        monkeypatch.setattr(cli_config.Prompt, "ask", fake_ask)
        _prompt_auto_compact_window(self._synthetic(_SONNET_46_1M_NATIVE_KEY))
        assert "y" not in captured["choices"]


class _StubAdminClient:
    endpoint = "http://localhost:8080"


def _setup_codex_env(
    monkeypatch,
    tmp_path: Path,
    *,
    level_choice: str,
    model_choice: str = "1",
    backup_yes: bool = False,
):
    """Patch the world for an in-process call to ``cli_config.codex_config()``.

    ``level_choice`` is "1" (user) or "2" (project). ``model_choice`` is the
    1-indexed table choice (or "0" for auto-routing). ``backup_yes`` controls
    the response to the backup prompt that fires when the target file exists.
    """
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(cli_config.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(cli_config.Path, "cwd", classmethod(lambda cls: cwd))

    fake_models = [
        {"provider": "github-copilot", "id": "gpt-5.5", "name": "GPT-5.5"},
        {"provider": "github-copilot", "id": "claude-opus-4.6", "name": "Claude Opus 4.6"},
    ]
    monkeypatch.setattr(cli_config, "_fetch_and_display_models", lambda: list(fake_models))
    monkeypatch.setattr(cli_config, "get_admin_client", lambda: _StubAdminClient())

    answers = iter([level_choice, model_choice])
    monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: next(answers))
    monkeypatch.setattr(cli_config.Confirm, "ask", lambda *a, **kw: backup_yes)

    return home, cwd


def _qualified_server_models() -> list[dict]:
    return [
        {
            "provider": "github-copilot",
            "id": "github-copilot/gpt-5.5",
            "name": "GPT-5.5",
        },
        {
            "provider": "github-copilot",
            "id": "github-copilot/claude-opus-4.6",
            "name": "Claude Opus 4.6",
        },
        {
            "provider": "github-copilot",
            "id": "github-copilot/gemini-2.5-pro",
            "name": "Gemini 2.5 Pro",
        },
    ]


class TestCodexConfig:
    """Tests for ``router-maestro config codex`` (the ``codex_config`` CLI command)."""

    def test_user_level_writes_full_config(self, tmp_path, monkeypatch):
        """User-level scope writes ``model``, ``model_provider``, and the provider table."""
        home, _ = _setup_codex_env(monkeypatch, tmp_path, level_choice="1")

        cli_config.codex_config()

        user_path = home / ".codex" / "config.toml"
        assert user_path.exists()
        with open(user_path, "rb") as f:
            data = tomllib.load(f)
        assert data["model"] == "github-copilot/gpt-5.5"
        assert data["model_provider"] == "router-maestro"
        provider = data["model_providers"]["router-maestro"]
        assert provider == {
            "name": "Router Maestro",
            "base_url": "http://localhost:8080/api/openai/v1",
            "env_key": "ROUTER_MAESTRO_API_KEY",
            "wire_api": "responses",
        }

    def test_qualified_server_model_writes_single_provider_prefix(self, tmp_path, monkeypatch):
        home, _ = _setup_codex_env(monkeypatch, tmp_path, level_choice="1")
        monkeypatch.setattr(
            cli_config,
            "_fetch_and_display_models",
            lambda: _qualified_server_models(),
        )

        cli_config.codex_config()

        with open(home / ".codex" / "config.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["model"] == "github-copilot/gpt-5.5"

    def test_project_level_writes_only_model(self, tmp_path, monkeypatch):
        """Project-level scope must NOT write ``model_provider``/``model_providers``."""
        _, cwd = _setup_codex_env(monkeypatch, tmp_path, level_choice="2")

        cli_config.codex_config()

        project_path = cwd / ".codex" / "config.toml"
        assert project_path.exists()
        with open(project_path, "rb") as f:
            data = tomllib.load(f)
        assert data == {"model": "github-copilot/gpt-5.5"}

    def test_project_level_self_heals_stale_keys(self, tmp_path, monkeypatch):
        """Re-running at project level strips the unsupported keys older versions wrote."""
        _, cwd = _setup_codex_env(
            monkeypatch, tmp_path, level_choice="2", model_choice="2", backup_yes=False
        )

        project_path = cwd / ".codex" / "config.toml"
        project_path.parent.mkdir(parents=True, exist_ok=True)
        stale = tomlkit.document()
        stale["model"] = "github-copilot/old-model"
        stale["model_provider"] = "router-maestro"
        providers = tomlkit.table()
        rm_table = tomlkit.table()
        rm_table["name"] = "Router Maestro"
        rm_table["base_url"] = "http://stale/v1"
        rm_table["env_key"] = "ROUTER_MAESTRO_API_KEY"
        rm_table["wire_api"] = "responses"
        providers["router-maestro"] = rm_table
        stale["model_providers"] = providers
        # Unrelated key the user might have hand-added — must survive untouched.
        stale["model_context_window"] = 400000
        with open(project_path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(stale))

        cli_config.codex_config()

        with open(project_path, "rb") as f:
            data = tomllib.load(f)
        assert data["model"] == "github-copilot/claude-opus-4.6"
        assert "model_provider" not in data
        assert "model_providers" not in data
        assert data["model_context_window"] == 400000

    def test_project_level_preserves_other_model_providers(self, tmp_path, monkeypatch):
        """Project-level cleanup removes only ``router-maestro``, not user-added providers."""
        _, cwd = _setup_codex_env(monkeypatch, tmp_path, level_choice="2", backup_yes=False)

        project_path = cwd / ".codex" / "config.toml"
        project_path.parent.mkdir(parents=True, exist_ok=True)
        seed = tomlkit.document()
        seed["model_provider"] = "router-maestro"  # stale top-level key
        providers = tomlkit.table()
        rm_table = tomlkit.table()
        rm_table["name"] = "Router Maestro"
        rm_table["base_url"] = "http://stale/v1"
        providers["router-maestro"] = rm_table
        other_table = tomlkit.table()
        other_table["name"] = "User Custom"
        other_table["base_url"] = "https://other.example.com/v1"
        providers["other"] = other_table
        seed["model_providers"] = providers
        with open(project_path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(seed))

        cli_config.codex_config()

        with open(project_path, "rb") as f:
            data = tomllib.load(f)
        assert data["model"] == "github-copilot/gpt-5.5"
        assert "model_provider" not in data
        assert "router-maestro" not in data["model_providers"]
        assert data["model_providers"]["other"]["name"] == "User Custom"


def test_claude_config_qualified_models_write_single_prefix_and_offer_beta(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cli_config.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(cli_config.Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli_config, "_fetch_models", _qualified_server_models)
    monkeypatch.setattr(cli_config, "get_admin_client", lambda: _StubAdminClient())
    monkeypatch.setattr(cli_config, "get_current_context_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_config, "_prompt_auto_compact_window", lambda _model: None)
    answers = iter(["1", "2", "1", "2"])
    monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: next(answers))
    monkeypatch.setattr(cli_config.Confirm, "ask", lambda *a, **kw: False)

    cli_config.claude_code_config()

    data = json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert data["env"]["ANTHROPIC_MODEL"] == "github-copilot/claude-opus-4.6"
    assert data["env"]["ANTHROPIC_SMALL_FAST_MODEL"] == "github-copilot/gpt-5.5"
    assert data["env"]["ANTHROPIC_BASE_URL"].endswith("/api/anthropic/beta")


def test_claude_config_writes_provider_qualified_native_1m_key(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cli_config.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(cli_config.Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        cli_config,
        "_fetch_models",
        lambda: [
            {
                "provider": "github-copilot",
                "id": "github-copilot/claude-opus-4.6",
                "name": "Claude Opus 4.6",
                "max_context_window_tokens": 1_000_000,
            }
        ],
    )
    monkeypatch.setattr(cli_config, "get_admin_client", lambda: _StubAdminClient())
    monkeypatch.setattr(cli_config, "get_current_context_api_key", lambda: "test-key")
    monkeypatch.setattr(cli_config, "_prompt_auto_compact_window", lambda _model: None)
    monkeypatch.setattr(cli_config, "_prompt_endpoint_mode", lambda _model: False)
    answers = iter(["1", "1", "2"])
    monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: next(answers))
    monkeypatch.setattr(cli_config.Confirm, "ask", lambda *a, **kw: False)

    cli_config.claude_code_config()

    data = json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert data["env"]["ANTHROPIC_MODEL"] == f"github-copilot/{_OPUS_1M_NATIVE_KEY}"


def test_gemini_config_qualified_model_preserves_public_id(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cli_config.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(cli_config.Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        cli_config,
        "_fetch_and_display_models",
        lambda: [_qualified_server_models()[2]],
    )
    monkeypatch.setattr(cli_config, "get_admin_client", lambda: _StubAdminClient())
    monkeypatch.setattr(cli_config, "get_current_context_api_key", lambda: "test-key")
    answers = iter(["1", "1"])
    monkeypatch.setattr(cli_config.Prompt, "ask", lambda *a, **kw: next(answers))
    monkeypatch.setattr(cli_config.Confirm, "ask", lambda *a, **kw: False)

    cli_config.gemini_cli_config()

    env = cli_config._parse_env_file(home / ".gemini" / ".env")
    assert env["GEMINI_MODEL"] == "github-copilot/gemini-2.5-pro"
