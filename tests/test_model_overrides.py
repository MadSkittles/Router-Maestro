"""Tests for per-model token overrides (Feature 2)."""

from router_maestro.config.priorities import ModelOverride, PrioritiesConfig
from router_maestro.providers.base import ModelInfo


class TestModelInfoWithOverrides:
    """Tests for ModelInfo.with_overrides()."""

    def _make_info(self) -> ModelInfo:
        return ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            max_prompt_tokens=128000,
            max_output_tokens=16384,
            max_context_window_tokens=200000,
            supports_thinking=True,
            supports_vision=True,
        )

    def test_override_single_field(self):
        """Override max_prompt_tokens, leave others unchanged."""
        info = self._make_info()
        updated = info.with_overrides(max_prompt_tokens=900000)
        assert updated.max_prompt_tokens == 900000
        assert updated.max_output_tokens == 16384
        assert updated.max_context_window_tokens == 200000
        assert updated.supports_thinking is True

    def test_override_multiple_fields(self):
        """Override all three token fields."""
        info = self._make_info()
        updated = info.with_overrides(
            max_prompt_tokens=500000,
            max_output_tokens=32000,
            max_context_window_tokens=600000,
        )
        assert updated.max_prompt_tokens == 500000
        assert updated.max_output_tokens == 32000
        assert updated.max_context_window_tokens == 600000

    def test_override_no_fields(self):
        """Calling with no overrides returns equivalent copy."""
        info = self._make_info()
        updated = info.with_overrides()
        assert updated.max_prompt_tokens == info.max_prompt_tokens
        assert updated.max_output_tokens == info.max_output_tokens
        assert updated.max_context_window_tokens == info.max_context_window_tokens

    def test_override_is_immutable(self):
        """Original ModelInfo is not mutated."""
        info = self._make_info()
        updated = info.with_overrides(max_prompt_tokens=1)
        assert info.max_prompt_tokens == 128000
        assert updated.max_prompt_tokens == 1
        assert info is not updated

    def test_override_preserves_non_token_fields(self):
        """id, name, provider, supports_* are preserved."""
        info = self._make_info()
        updated = info.with_overrides(max_prompt_tokens=1)
        assert updated.id == "claude-opus-4.6"
        assert updated.name == "Claude Opus 4.6"
        assert updated.provider == "github-copilot"
        assert updated.supports_thinking is True
        assert updated.supports_vision is True


class TestModelOverrideConfig:
    """Tests for ModelOverride and PrioritiesConfig parsing."""

    def test_model_override_defaults(self):
        """All fields default to None."""
        override = ModelOverride()
        assert override.max_prompt_tokens is None
        assert override.max_output_tokens is None
        assert override.max_context_window_tokens is None

    def test_model_override_from_dict(self):
        """Parse from JSON-like dict."""
        override = ModelOverride.model_validate({"max_prompt_tokens": 900000})
        assert override.max_prompt_tokens == 900000
        assert override.max_output_tokens is None

    def test_priorities_config_with_overrides(self):
        """PrioritiesConfig parses model_overrides."""
        config = PrioritiesConfig.model_validate(
            {
                "priorities": ["github-copilot/claude-opus-4.6"],
                "model_overrides": {
                    "github-copilot/claude-opus-4.6": {"max_prompt_tokens": 900000},
                },
            }
        )
        assert "github-copilot/claude-opus-4.6" in config.model_overrides
        assert config.model_overrides["github-copilot/claude-opus-4.6"].max_prompt_tokens == 900000

    def test_priorities_config_empty_overrides(self):
        """Empty model_overrides is the default."""
        config = PrioritiesConfig()
        assert config.model_overrides == {}

    def test_priorities_config_with_thinking(self):
        """PrioritiesConfig parses thinking section."""
        config = PrioritiesConfig.model_validate(
            {
                "thinking": {
                    "default_budget": 24000,
                    "auto_enable": True,
                    "model_budgets": {"claude-opus-4.6": 32000},
                },
            }
        )
        assert config.thinking.default_budget == 24000
        assert config.thinking.auto_enable is True
        assert config.thinking.model_budgets["claude-opus-4.6"] == 32000
