"""Tests for configurable thinking budget (Feature 4)."""

from router_maestro.config.priorities import ThinkingBudgetConfig
from router_maestro.providers.base import ChatRequest, Message
from router_maestro.utils.context_window import resolve_thinking_budget


class TestResolveThinkingBudget:
    """Tests for resolve_thinking_budget priority chain."""

    def _config(self, **kwargs) -> ThinkingBudgetConfig:
        return ThinkingBudgetConfig(**kwargs)

    def test_client_specified_budget(self):
        """Client budget takes priority over everything."""
        budget, ttype = resolve_thinking_budget(
            client_budget=8000,
            client_thinking_type="enabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=16384,
            thinking_config=self._config(default_budget=16000),
            supports_thinking=True,
        )
        assert budget == 8000
        assert ttype == "enabled"

    def test_client_disabled(self):
        """Client disabling thinking returns None."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type="disabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=16384,
            thinking_config=self._config(default_budget=16000, auto_enable=True),
            supports_thinking=True,
        )
        assert budget is None
        assert ttype is None

    def test_client_enabled_no_budget_uses_server_default(self):
        """Client enables thinking without budget → server default applied."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type="enabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(default_budget=16000),
            supports_thinking=True,
        )
        assert budget == 16000
        assert ttype == "enabled"

    def test_client_enabled_no_budget_uses_model_budget(self):
        """Client enables thinking, per-model budget overrides default."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type="enabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(
                default_budget=16000,
                model_budgets={"claude-opus-4.6": 24000},
            ),
            supports_thinking=True,
        )
        assert budget == 24000
        assert ttype == "enabled"

    def test_auto_enable_for_capable_model(self):
        """auto_enable activates thinking for capable models."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type=None,
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(default_budget=16000, auto_enable=True),
            supports_thinking=True,
        )
        assert budget == 16000
        assert ttype == "enabled"

    def test_auto_enable_skipped_for_incapable_model(self):
        """auto_enable does nothing when model doesn't support thinking."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type=None,
            model_id="github-copilot/gpt-4o",
            max_output_tokens=16384,
            thinking_config=self._config(default_budget=16000, auto_enable=True),
            supports_thinking=False,
        )
        assert budget is None
        assert ttype is None

    def test_no_config_no_client_thinking(self):
        """No config and no client thinking → None."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type=None,
            model_id="some-model",
            max_output_tokens=16384,
            thinking_config=None,
            supports_thinking=True,
        )
        assert budget is None
        assert ttype is None

    def test_budget_normalized(self):
        """Budget is clamped to valid range."""
        budget, ttype = resolve_thinking_budget(
            client_budget=500,  # Below 1024 min
            client_thinking_type="enabled",
            model_id="test-model",
            max_output_tokens=16384,
        )
        assert budget == 1024  # Clamped to minimum
        assert ttype == "enabled"

    def test_auto_enable_false_skips(self):
        """When auto_enable=False and client didn't request, no thinking."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type=None,
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(default_budget=16000, auto_enable=False),
            supports_thinking=True,
        )
        assert budget is None
        assert ttype is None

    def test_per_model_budget_with_full_key(self):
        """Per-model budget matched by full provider/model key."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type="enabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(
                default_budget=16000,
                model_budgets={"github-copilot/claude-opus-4.6": 28000},
            ),
            supports_thinking=True,
        )
        assert budget == 28000

    def test_adaptive_thinking_type_preserved(self):
        """Adaptive thinking type is preserved through resolution."""
        budget, ttype = resolve_thinking_budget(
            client_budget=10000,
            client_thinking_type="adaptive",
            model_id="test-model",
            max_output_tokens=16384,
        )
        assert budget == 10000
        assert ttype == "adaptive"


class TestChatRequestWithThinking:
    """Tests for ChatRequest.with_thinking()."""

    def _make_request(self) -> ChatRequest:
        return ChatRequest(
            model="test-model",
            messages=[Message(role="user", content="hello")],
            thinking_budget=None,
            thinking_type=None,
        )

    def test_with_thinking_returns_new_request(self):
        """with_thinking returns a new object."""
        req = self._make_request()
        updated = req.with_thinking(thinking_budget=8000, thinking_type="enabled")
        assert updated is not req
        assert updated.thinking_budget == 8000
        assert updated.thinking_type == "enabled"

    def test_with_thinking_preserves_other_fields(self):
        """Non-thinking fields are preserved."""
        req = self._make_request()
        req.temperature = 0.5
        req.max_tokens = 1000
        updated = req.with_thinking(thinking_budget=8000, thinking_type="enabled")
        assert updated.model == "test-model"
        assert updated.temperature == 0.5
        assert updated.max_tokens == 1000
        assert updated.messages == req.messages

    def test_original_not_mutated(self):
        """Original request is not mutated."""
        req = self._make_request()
        req.with_thinking(thinking_budget=8000, thinking_type="enabled")
        assert req.thinking_budget is None
        assert req.thinking_type is None
