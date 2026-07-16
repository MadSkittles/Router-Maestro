"""Tests for the structural official-model-id primitives.

The conversion from an internal ``provider/upstream-id`` to a vendor's official
id is derived from naming conventions, NOT a per-model lookup table. These tests
pin the family-detection rules and the per-vendor spelling conventions:
Anthropic uses dashes (``claude-opus-4.6`` -> ``claude-opus-4-6``) while OpenAI
and Google keep dots (``gpt-4.1`` stays, ``gemini-2.5-pro`` stays).
"""

import pytest

from router_maestro.cli.client_configs.model_id import (
    ModelFamily,
    detect_family,
    to_anthropic_official,
    to_gemini_official,
    to_openai_official,
)


class TestDetectFamily:
    @pytest.mark.parametrize(
        "bare_id,expected",
        [
            ("claude-opus-4.6", ModelFamily.ANTHROPIC),
            ("claude-sonnet-4-20250514", ModelFamily.ANTHROPIC),
            ("Claude-Haiku-4.5", ModelFamily.ANTHROPIC),
            ("gpt-4.1", ModelFamily.OPENAI),
            ("gpt-5.5", ModelFamily.OPENAI),
            ("GPT-4o", ModelFamily.OPENAI),
            ("o1", ModelFamily.OPENAI),
            ("o3-mini", ModelFamily.OPENAI),
            ("o4-preview", ModelFamily.OPENAI),
            ("codex-mini-latest", ModelFamily.OPENAI),
            ("gemini-2.5-pro", ModelFamily.GOOGLE),
            ("Gemini-2.0-flash", ModelFamily.GOOGLE),
            ("llama-3.1-70b", ModelFamily.UNKNOWN),
            ("mistral-large", ModelFamily.UNKNOWN),
            ("", ModelFamily.UNKNOWN),
        ],
    )
    def test_detects_family_by_naming_convention(self, bare_id, expected):
        assert detect_family(bare_id) is expected

    def test_o_series_does_not_match_other_o_words(self):
        # "olmo" / "orca" start with 'o' but are not the OpenAI o-series.
        assert detect_family("olmo-7b") is ModelFamily.UNKNOWN
        assert detect_family("orca-2") is ModelFamily.UNKNOWN


class TestAnthropicOfficial:
    def test_dot_becomes_dash(self):
        assert to_anthropic_official("claude-opus-4.6") == "claude-opus-4-6"
        assert to_anthropic_official("claude-sonnet-4.6") == "claude-sonnet-4-6"

    def test_idempotent_on_already_dashed(self):
        assert to_anthropic_official("claude-opus-4-6") == "claude-opus-4-6"

    def test_preserves_dated_official_id(self):
        # A dated Anthropic id is already official spelling.
        assert to_anthropic_official("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"

    def test_lowercases(self):
        assert to_anthropic_official("Claude-Opus-4.6") == "claude-opus-4-6"


class TestOpenAIOfficial:
    def test_keeps_dots(self):
        assert to_openai_official("gpt-4.1") == "gpt-4.1"
        assert to_openai_official("gpt-5.5") == "gpt-5.5"

    def test_identity(self):
        assert to_openai_official("o3-mini") == "o3-mini"
        assert to_openai_official("gpt-4o") == "gpt-4o"


class TestGeminiOfficial:
    def test_keeps_dots_and_structure(self):
        assert to_gemini_official("gemini-2.5-pro") == "gemini-2.5-pro"
        assert to_gemini_official("gemini-2.0-flash") == "gemini-2.0-flash"
