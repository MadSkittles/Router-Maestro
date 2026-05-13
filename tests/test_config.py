"""Tests for configuration module."""

import tempfile
import tomllib
from pathlib import Path

import tomlkit

from router_maestro.cli import config as cli_config
from router_maestro.cli.config import (
    _OPUS_1M_NATIVE_KEY,
    _OPUS_1M_SOURCE_MODEL,
    _maybe_inject_opus_1m,
    _select_model,
)
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

    def test_load_creates_default(self):
        """Test that loading non-existent file creates default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"

            config = load_config(path, ContextsConfig, ContextsConfig.get_default)

            assert config.current == "local"
            assert path.exists()  # Should have created the file


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
            {"provider": "github-copilot", "id": "claude-opus-4.6", "name": "Claude Opus 4.6"},
            {
                "provider": "github-copilot",
                "id": "claude-opus-4.6-1m",
                "name": "Claude Opus 4.6 1M",
            },
        ]

        result = _maybe_inject_opus_1m(models)

        assert len(result) == 3
        assert len(models) == 2  # original list not mutated
        synthetic = result[0]
        assert synthetic["custom_key"] == _OPUS_1M_NATIVE_KEY
        assert synthetic["display_key"] == _OPUS_1M_NATIVE_KEY
        assert synthetic["name"] == "Opus 4.6 1M (Auto-activated)"
        assert synthetic["provider"] == "github-copilot"

    def test_no_injection_when_1m_model_absent(self):
        """No synthetic entry when the source model is not in the list."""
        models = [
            {"provider": "github-copilot", "id": "claude-opus-4.6", "name": "Claude Opus 4.6"},
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
                "id": "claude-opus-4.6-1m",
                "name": "Claude Opus 4.6 1M",
            },
        ]
        original_len = len(models)

        _maybe_inject_opus_1m(models)

        assert len(models) == original_len

    def test_source_model_constant_matches_expected_value(self):
        """Guard against accidental changes to the source model constant."""
        assert _OPUS_1M_SOURCE_MODEL == "github-copilot/claude-opus-4.6-1m"
        assert _OPUS_1M_NATIVE_KEY == "claude-opus-4-6[1m]"


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
