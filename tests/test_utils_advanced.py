"""Tests for utility modules."""

import pytest

from router_maestro.utils.tokens import (
    estimate_tokens,
    estimate_tokens_from_char_count,
    map_openai_stop_reason_to_anthropic,
)


class TestTokenEstimationAdvanced:
    """Advanced tests for token estimation."""

    def test_estimate_tokens_unicode(self):
        """Test token estimation with unicode characters."""
        # Unicode typically uses more bytes but estimation is character-based
        text = "Hello 世界 🌍"
        result = estimate_tokens(text)
        assert result > 0

    def test_estimate_tokens_newlines(self):
        """Test token estimation with newlines."""
        text = "Line 1\nLine 2\nLine 3"
        result = estimate_tokens(text)
        assert result > 0

    def test_estimate_from_char_count_large(self):
        """Test estimation from large character count."""
        # 4000 chars should be about 1000 tokens
        result = estimate_tokens_from_char_count(4000)
        assert result == 1000

    def test_estimate_from_char_count_small(self):
        """Test estimation from small character count."""
        result = estimate_tokens_from_char_count(10)
        assert result >= 1


class TestStopReasonMappingAdvanced:
    """Advanced tests for stop reason mapping."""

    def test_map_function_call(self):
        """Test mapping tool_calls stop reason."""
        result = map_openai_stop_reason_to_anthropic("tool_calls")
        assert result == "tool_use"

    def test_map_content_filter(self):
        """Test mapping content_filter stop reason."""
        result = map_openai_stop_reason_to_anthropic("content_filter")
        assert result == "end_turn"

    def test_map_stop(self):
        """Test mapping stop reason."""
        result = map_openai_stop_reason_to_anthropic("stop")
        assert result == "end_turn"

    def test_map_length(self):
        """Test mapping length reason."""
        result = map_openai_stop_reason_to_anthropic("length")
        assert result == "max_tokens"

    def test_map_unknown_defaults_to_end_turn(self):
        """Test unknown reason defaults to end_turn."""
        result = map_openai_stop_reason_to_anthropic("some_unknown_reason")
        assert result == "end_turn"
