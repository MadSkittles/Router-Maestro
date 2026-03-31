"""Tests for configuration module."""

import tempfile
from pathlib import Path

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
                "name": "Opus 4.6 (1M context)",
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
        assert synthetic["name"] == "Opus 4.6 (1M context)"
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
