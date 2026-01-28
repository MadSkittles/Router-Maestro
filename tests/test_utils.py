"""Tests for utility functions."""

import pytest

from router_maestro.utils import (
    estimate_tokens,
    estimate_tokens_from_char_count,
    map_openai_stop_reason_to_anthropic,
)


class TestTokenEstimation:
    """Tests for token estimation utilities."""

    def test_estimate_tokens_empty_string(self):
        """Test estimating tokens for empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test estimating tokens for short text."""
        # 12 characters / 4 = 3 tokens
        assert estimate_tokens("Hello world!") == 3

    def test_estimate_tokens_longer_text(self):
        """Test estimating tokens for longer text."""
        # 100 characters / 4 = 25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_estimate_tokens_from_char_count(self):
        """Test estimating tokens from character count."""
        assert estimate_tokens_from_char_count(0) == 0
        assert estimate_tokens_from_char_count(4) == 1
        assert estimate_tokens_from_char_count(100) == 25
        assert estimate_tokens_from_char_count(1000) == 250


class TestStopReasonMapping:
    """Tests for OpenAI to Anthropic stop reason mapping."""

    def test_map_stop(self):
        """Test mapping 'stop' to 'end_turn'."""
        assert map_openai_stop_reason_to_anthropic("stop") == "end_turn"

    def test_map_length(self):
        """Test mapping 'length' to 'max_tokens'."""
        assert map_openai_stop_reason_to_anthropic("length") == "max_tokens"

    def test_map_tool_calls(self):
        """Test mapping 'tool_calls' to 'tool_use'."""
        assert map_openai_stop_reason_to_anthropic("tool_calls") == "tool_use"

    def test_map_content_filter(self):
        """Test mapping 'content_filter' to 'end_turn'."""
        assert map_openai_stop_reason_to_anthropic("content_filter") == "end_turn"

    def test_map_none(self):
        """Test mapping None returns None."""
        assert map_openai_stop_reason_to_anthropic(None) is None

    def test_map_unknown(self):
        """Test mapping unknown reason defaults to 'end_turn'."""
        assert map_openai_stop_reason_to_anthropic("unknown") == "end_turn"
