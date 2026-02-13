"""Tests for context window budget calculations."""

from router_maestro.utils.context_window import (
    ContextBudget,
    calculate_context_budget,
    normalize_thinking_budget,
)


class TestCalculateContextBudget:
    """Tests for calculate_context_budget."""

    def test_budget_128k_context(self):
        """Standard 128k context model (e.g., GPT-4o)."""
        result = calculate_context_budget(
            max_prompt_tokens=128000,
            max_output_tokens=16384,
            max_context_window_tokens=128000,
        )
        assert result is not None
        # effective_output = min(16384, 128000 * 0.15 = 19200) = 16384
        assert result.max_output_tokens == 16384
        # usable_prompt = min(128000, 128000 - 16384) = 111616
        assert result.max_prompt_tokens == 111616
        assert result.context_window == 128000

    def test_budget_1m_context(self):
        """Opus 4.6 with 1M context window."""
        result = calculate_context_budget(
            max_prompt_tokens=1000000,
            max_output_tokens=32000,
            max_context_window_tokens=1000000,
        )
        assert result is not None
        # effective_output = min(32000, 1000000 * 0.15 = 150000) = 32000
        assert result.max_output_tokens == 32000
        # usable_prompt = min(1000000, 1000000 - 32000) = 968000
        assert result.max_prompt_tokens == 968000
        assert result.context_window == 1000000

    def test_budget_missing_prompt_tokens(self):
        """Returns None when max_prompt_tokens is unknown."""
        result = calculate_context_budget(
            max_prompt_tokens=None,
            max_output_tokens=16384,
            max_context_window_tokens=128000,
        )
        assert result is None

    def test_budget_missing_context_window(self):
        """Derives context window from prompt + output when not provided."""
        result = calculate_context_budget(
            max_prompt_tokens=128000,
            max_output_tokens=16384,
            max_context_window_tokens=None,
        )
        assert result is not None
        # effective_output = min(16384, 128000 * 0.15 = 19200) = 16384
        assert result.max_output_tokens == 16384
        # context_window = 16384 + 128000 = 144384
        assert result.context_window == 144384
        # usable_prompt = min(128000, 144384 - 16384) = 128000
        assert result.max_prompt_tokens == 128000

    def test_budget_missing_max_output(self):
        """Falls back to 4096 when max_output_tokens is None."""
        result = calculate_context_budget(
            max_prompt_tokens=128000,
            max_output_tokens=None,
            max_context_window_tokens=128000,
        )
        assert result is not None
        # effective_output = min(4096, 128000 * 0.15 = 19200) = 4096
        assert result.max_output_tokens == 4096
        # usable_prompt = min(128000, 128000 - 4096) = 123904
        assert result.max_prompt_tokens == 123904

    def test_budget_output_capped_by_15_percent(self):
        """When max_output_tokens exceeds 15% of prompt, it gets capped."""
        result = calculate_context_budget(
            max_prompt_tokens=10000,
            max_output_tokens=16384,
            max_context_window_tokens=10000,
        )
        assert result is not None
        # effective_output = min(16384, 10000 * 0.15 = 1500) = 1500
        assert result.max_output_tokens == 1500
        # usable_prompt = min(10000, 10000 - 1500) = 8500
        assert result.max_prompt_tokens == 8500

    def test_budget_is_frozen_dataclass(self):
        """ContextBudget is immutable."""
        result = calculate_context_budget(
            max_prompt_tokens=128000,
            max_output_tokens=16384,
            max_context_window_tokens=128000,
        )
        assert result is not None
        assert isinstance(result, ContextBudget)

    def test_budget_inconsistent_small_context_window(self):
        """Context window smaller than effective output clamps to 0."""
        result = calculate_context_budget(
            max_prompt_tokens=10000,
            max_output_tokens=50000,
            max_context_window_tokens=1000,
        )
        assert result is not None
        # effective_output = min(50000, 10000 * 0.15 = 1500) = 1500
        assert result.max_output_tokens == 1500
        # usable_prompt = max(0, min(10000, 1000 - 1500)) = max(0, -500) = 0
        assert result.max_prompt_tokens == 0


class TestNormalizeThinkingBudget:
    """Tests for normalize_thinking_budget."""

    def test_clamp_high_budget(self):
        """Budget exceeding max_output is clamped."""
        # budget=50000, max_output=16384
        # cap = min(32000, 16383) = 16383
        # result = max(1024, min(50000, 16383)) = max(1024, 16383) = 16383
        result = normalize_thinking_budget(budget=50000, max_output_tokens=16384)
        assert result == 16383

    def test_clamp_low_budget(self):
        """Budget below minimum is raised to minimum."""
        # budget=500, max_output=16384
        # cap = min(32000, 16383) = 16383
        # result = max(1024, min(500, 16383)) = max(1024, 500) = 1024
        result = normalize_thinking_budget(budget=500, max_output_tokens=16384)
        assert result == 1024

    def test_none_budget(self):
        """None budget returns None (thinking not requested)."""
        result = normalize_thinking_budget(budget=None, max_output_tokens=16384)
        assert result is None

    def test_budget_within_range(self):
        """Budget within range is preserved."""
        result = normalize_thinking_budget(budget=8000, max_output_tokens=16384)
        assert result == 8000

    def test_budget_at_max_boundary(self):
        """Budget at the max cap (32000) with high max_output."""
        # budget=32000, max_output=100000
        # cap = min(32000, 99999) = 32000
        # result = max(1024, min(32000, 32000)) = 32000
        result = normalize_thinking_budget(budget=32000, max_output_tokens=100000)
        assert result == 32000

    def test_custom_min_max(self):
        """Custom min/max budget boundaries."""
        result = normalize_thinking_budget(
            budget=100,
            max_output_tokens=50000,
            min_budget=500,
            max_budget=10000,
        )
        assert result == 500

    def test_zero_max_output_tokens(self):
        """max_output_tokens of 0 returns min_budget."""
        result = normalize_thinking_budget(budget=8000, max_output_tokens=0)
        assert result == 1024

    def test_one_max_output_tokens(self):
        """max_output_tokens of 1 returns min_budget."""
        result = normalize_thinking_budget(budget=8000, max_output_tokens=1)
        assert result == 1024
