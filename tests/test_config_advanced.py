"""Tests for configuration modules."""

from router_maestro.config import (
    FallbackConfig,
    FallbackStrategy,
    PrioritiesConfig,
    ProvidersConfig,
)
from router_maestro.config.contexts import ContextConfig, ContextsConfig


class TestFallbackStrategy:
    """Tests for FallbackStrategy enum."""

    def test_none_strategy(self):
        """Test none fallback strategy."""
        assert FallbackStrategy.NONE == "none"

    def test_priority_strategy(self):
        """Test priority fallback strategy."""
        assert FallbackStrategy.PRIORITY == "priority"

    def test_same_model_strategy(self):
        """Test same-model fallback strategy."""
        assert FallbackStrategy.SAME_MODEL == "same-model"


class TestFallbackConfig:
    """Tests for FallbackConfig."""

    def test_default_values(self):
        """Test default fallback config values."""
        config = FallbackConfig()
        assert config.strategy == FallbackStrategy.PRIORITY
        assert config.maxRetries == 2

    def test_custom_values(self):
        """Test custom fallback config values."""
        config = FallbackConfig(strategy=FallbackStrategy.NONE, maxRetries=5)
        assert config.strategy == FallbackStrategy.NONE
        assert config.maxRetries == 5

    def test_from_dict(self):
        """Test creating from dict."""
        config = FallbackConfig(**{"strategy": "same-model", "maxRetries": 3})
        assert config.strategy == FallbackStrategy.SAME_MODEL
        assert config.maxRetries == 3


class TestPrioritiesConfig:
    """Tests for PrioritiesConfig."""

    def test_default_values(self):
        """Test default priorities config."""
        config = PrioritiesConfig()
        assert config.priorities == []
        assert config.fallback.strategy == FallbackStrategy.PRIORITY

    def test_with_priorities(self):
        """Test config with priorities."""
        config = PrioritiesConfig(
            priorities=["github-copilot/gpt-4o", "openai/gpt-4o"],
        )
        assert len(config.priorities) == 2
        assert config.priorities[0] == "github-copilot/gpt-4o"

    def test_with_fallback_dict(self):
        """Test config with fallback as dict."""
        config = PrioritiesConfig(
            priorities=["model-1"],
            fallback={"strategy": "none", "maxRetries": 0},
        )
        assert config.fallback.strategy == FallbackStrategy.NONE


class TestProvidersConfig:
    """Tests for ProvidersConfig."""

    def test_default_values(self):
        """Test default providers config."""
        config = ProvidersConfig()
        assert config.providers == {}

    def test_with_providers(self):
        """Test config with providers."""
        config = ProvidersConfig(
            providers={
                "custom": {
                    "baseURL": "https://api.example.com/v1",
                    "models": {"model-1": {"name": "Model One"}},
                }
            }
        )
        assert "custom" in config.providers
        assert config.providers["custom"].baseURL == "https://api.example.com/v1"


class TestContextConfig:
    """Tests for ContextConfig."""

    def test_basic_context(self):
        """Test basic context config."""
        config = ContextConfig(endpoint="http://localhost:8080")
        assert config.endpoint == "http://localhost:8080"
        assert config.api_key is None

    def test_context_with_api_key(self):
        """Test context config with API key."""
        config = ContextConfig(endpoint="https://api.example.com", api_key="sk-test")
        assert config.endpoint == "https://api.example.com"
        assert config.api_key == "sk-test"


class TestContextsConfig:
    """Tests for ContextsConfig."""

    def test_default_values(self):
        """Test default contexts config."""
        config = ContextsConfig()
        assert config.contexts == {}
        assert config.current == "local"

    def test_with_contexts(self):
        """Test config with contexts."""
        config = ContextsConfig(
            contexts={"work": ContextConfig(endpoint="https://work.api.com")},
            current="work",
        )
        assert "work" in config.contexts
        assert config.current == "work"

    def test_get_default(self):
        """Test getting default config."""
        config = ContextsConfig.get_default()
        assert config.current == "local"
        assert "local" in config.contexts
        assert config.contexts["local"].endpoint == "http://localhost:8080"


class TestConfigSerialization:
    """Tests for config serialization."""

    def test_priorities_config_json(self):
        """Test PrioritiesConfig JSON serialization."""
        config = PrioritiesConfig(
            priorities=["model-1", "model-2"],
            fallback=FallbackConfig(strategy=FallbackStrategy.NONE),
        )
        data = config.model_dump()
        assert data["priorities"] == ["model-1", "model-2"]
        assert data["fallback"]["strategy"] == "none"

    def test_contexts_config_json(self):
        """Test ContextsConfig JSON serialization."""
        config = ContextsConfig(
            current="local",
            contexts={"local": ContextConfig(endpoint="http://localhost:8080", api_key="test")},
        )
        data = config.model_dump()
        assert data["current"] == "local"
        assert data["contexts"]["local"]["api_key"] == "test"
