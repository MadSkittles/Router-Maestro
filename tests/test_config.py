"""Tests for configuration module."""

import tempfile
from pathlib import Path

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
