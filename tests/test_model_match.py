"""Tests for model ID fuzzy matching utilities."""

from __future__ import annotations

import pytest

from router_maestro.providers.base import ModelInfo
from router_maestro.utils.model_match import fuzzy_match_model, normalize_model_id
from router_maestro.utils.model_sort import strip_date_suffix


# ── strip_date_suffix ──────────────────────────────────────────────────


class TestStripDateSuffix:
    def test_plain_date(self):
        assert strip_date_suffix("claude-opus-4-6-20250617") == "claude-opus-4-6"

    def test_dashed_date(self):
        assert strip_date_suffix("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"

    def test_no_date(self):
        assert strip_date_suffix("gpt-4o") == "gpt-4o"

    def test_empty_string(self):
        assert strip_date_suffix("") == ""


# ── normalize_model_id ─────────────────────────────────────────────────


class TestNormalizeModelId:
    def test_case_conversion(self):
        assert normalize_model_id("Claude-Opus-4") == "claude-opus-4"

    def test_spaces_to_hyphens(self):
        assert normalize_model_id("Opus 4.6") == "opus-4-6"

    def test_dots_to_hyphens(self):
        assert normalize_model_id("claude-sonnet-4.5") == "claude-sonnet-4-5"

    def test_date_strip_plain(self):
        assert normalize_model_id("claude-opus-4-6-20250617") == "claude-opus-4-6"

    def test_date_strip_dashed(self):
        assert normalize_model_id("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"

    def test_noop(self):
        assert normalize_model_id("gpt-4o") == "gpt-4o"

    def test_combined(self):
        assert normalize_model_id("Claude Sonnet 4.5") == "claude-sonnet-4-5"


# ── fuzzy_match_model ──────────────────────────────────────────────────


def _make_cache(
    entries: list[tuple[str, str, str]],
) -> dict[str, tuple[str, ModelInfo]]:
    """Build a mock models cache from (model_id, provider_name, display_name) triples.

    Populates both bare keys and provider-prefixed keys, matching real router behavior.
    """
    cache: dict[str, tuple[str, ModelInfo]] = {}
    for model_id, provider_name, display_name in entries:
        info = ModelInfo(id=model_id, name=display_name, provider=provider_name)
        # Bare key (first provider wins, matching router convention)
        if model_id not in cache:
            cache[model_id] = (provider_name, info)
        # Provider-prefixed key
        cache[f"{provider_name}/{model_id}"] = (provider_name, info)
    return cache


@pytest.fixture()
def sample_cache() -> dict[str, tuple[str, ModelInfo]]:
    """A realistic model cache with multiple providers and versions."""
    return _make_cache([
        # Anthropic models via github-copilot
        ("claude-opus-4-6-20250617", "github-copilot", "Claude Opus 4.6"),
        ("claude-sonnet-4-5-20250514", "github-copilot", "Claude Sonnet 4.5"),
        ("claude-sonnet-4-5-20250929", "github-copilot", "Claude Sonnet 4.5"),
        # Anthropic models via anthropic provider
        ("claude-opus-4-6-20250617", "anthropic", "Claude Opus 4.6"),
        ("claude-sonnet-4-5-20250514", "anthropic", "Claude Sonnet 4.5"),
        # OpenAI models
        ("gpt-4o", "github-copilot", "GPT-4o"),
        ("gpt-4o-mini", "github-copilot", "GPT-4o Mini"),
        ("gpt-4o-mini-2024-07-18", "github-copilot", "GPT-4o Mini"),
        ("o3-mini", "github-copilot", "o3-mini"),
    ])


class TestFuzzyMatchModel:
    def test_opus_with_spaces(self, sample_cache):
        result = fuzzy_match_model("Opus 4.6", sample_cache)
        assert result == "claude-opus-4-6-20250617"

    def test_opus_hyphenated(self, sample_cache):
        result = fuzzy_match_model("opus-4-6", sample_cache)
        assert result == "claude-opus-4-6-20250617"

    def test_claude_opus_no_date(self, sample_cache):
        result = fuzzy_match_model("claude-opus-4-6", sample_cache)
        assert result == "claude-opus-4-6-20250617"

    def test_sonnet_with_date_picks_available(self, sample_cache):
        """Requesting a date-suffixed model should match the family, picking newest available."""
        result = fuzzy_match_model("claude-sonnet-4-5-20250929", sample_cache)
        # Should match the 20250929 version since it exists
        assert result == "claude-sonnet-4-5-20250929"

    def test_sonnet_short_picks_newest(self, sample_cache):
        """Short query should pick the newest version."""
        result = fuzzy_match_model("sonnet-4-5", sample_cache)
        assert result == "claude-sonnet-4-5-20250929"

    def test_provider_filter_anthropic(self, sample_cache):
        """Provider prefix should filter to only that provider and return prefixed key."""
        result = fuzzy_match_model("anthropic/opus-4-6", sample_cache)
        assert result == "anthropic/claude-opus-4-6-20250617"
        # Verify the returned key resolves to the correct provider
        provider_name, _ = sample_cache[result]
        assert provider_name == "anthropic"

    def test_provider_filter_nonexistent(self, sample_cache):
        """Provider prefix pointing to wrong provider returns None."""
        result = fuzzy_match_model("openai/opus-4-6", sample_cache)
        assert result is None

    def test_nonexistent_model(self, sample_cache):
        result = fuzzy_match_model("nonexistent-xyz", sample_cache)
        assert result is None

    def test_gpt4o_no_false_positive(self, sample_cache):
        """gpt-4o should not false-positive to gpt-4o-mini."""
        result = fuzzy_match_model("gpt-4o", sample_cache)
        assert result == "gpt-4o"

    def test_multiple_versions_picks_newest(self):
        """When multiple date versions match, pick newest."""
        cache = _make_cache([
            ("claude-opus-4-6-20250101", "anthropic", "Claude Opus 4.6"),
            ("claude-opus-4-6-20250617", "anthropic", "Claude Opus 4.6"),
            ("claude-opus-4-6-20250301", "anthropic", "Claude Opus 4.6"),
        ])
        result = fuzzy_match_model("opus-4-6", cache)
        assert result == "claude-opus-4-6-20250617"

    def test_dated_wins_over_undated(self):
        """A dated model should win over an undated one in the same family."""
        cache = _make_cache([
            ("claude-opus-4-6", "anthropic", "Claude Opus 4.6"),
            ("claude-opus-4-6-20250617", "anthropic", "Claude Opus 4.6"),
        ])
        result = fuzzy_match_model("opus-4-6", cache)
        assert result == "claude-opus-4-6-20250617"

    def test_cross_provider_first_registered_wins(self):
        """When same model in two providers, first-registered wins (consistent with router)."""
        cache = _make_cache([
            ("claude-opus-4-6-20250617", "github-copilot", "Claude Opus 4.6"),
            ("claude-opus-4-6-20250617", "anthropic", "Claude Opus 4.6"),
        ])
        result = fuzzy_match_model("opus-4-6", cache)
        # First registered is github-copilot, which gets the bare key
        assert result == "claude-opus-4-6-20250617"

    def test_empty_cache(self):
        result = fuzzy_match_model("gpt-4o", {})
        assert result is None

    def test_case_insensitive(self, sample_cache):
        result = fuzzy_match_model("CLAUDE-OPUS-4-6", sample_cache)
        assert result == "claude-opus-4-6-20250617"

    def test_dots_in_query(self, sample_cache):
        result = fuzzy_match_model("claude-sonnet-4.5", sample_cache)
        assert result == "claude-sonnet-4-5-20250929"
