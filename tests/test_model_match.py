"""Tests for model ID fuzzy matching utilities."""

from __future__ import annotations

import pytest

from router_maestro.providers.base import ModelInfo
from router_maestro.utils.model_match import (
    find_extended_context_variant,
    find_reasoning_variant,
    fuzzy_match_model,
    normalize_model_id,
)
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
    return _make_cache(
        [
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
        ]
    )


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
        cache = _make_cache(
            [
                ("claude-opus-4-6-20250101", "anthropic", "Claude Opus 4.6"),
                ("claude-opus-4-6-20250617", "anthropic", "Claude Opus 4.6"),
                ("claude-opus-4-6-20250301", "anthropic", "Claude Opus 4.6"),
            ]
        )
        result = fuzzy_match_model("opus-4-6", cache)
        assert result == "claude-opus-4-6-20250617"

    def test_dated_wins_over_undated(self):
        """A dated model should win over an undated one in the same family."""
        cache = _make_cache(
            [
                ("claude-opus-4-6", "anthropic", "Claude Opus 4.6"),
                ("claude-opus-4-6-20250617", "anthropic", "Claude Opus 4.6"),
            ]
        )
        result = fuzzy_match_model("opus-4-6", cache)
        assert result == "claude-opus-4-6-20250617"

    def test_cross_provider_first_registered_wins(self):
        """When same model in two providers, first-registered wins (consistent with router)."""
        cache = _make_cache(
            [
                ("claude-opus-4-6-20250617", "github-copilot", "Claude Opus 4.6"),
                ("claude-opus-4-6-20250617", "anthropic", "Claude Opus 4.6"),
            ]
        )
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

    def test_1m_context_window_suffix(self):
        """claude-opus-4-6-1m (hyphens) should match claude-opus-4.6-1m (dot) in cache."""
        cache = _make_cache(
            [
                ("claude-opus-4.6-1m", "anthropic", "Claude Opus 4.6 1M"),
            ]
        )
        result = fuzzy_match_model("claude-opus-4-6-1m", cache)
        assert result == "claude-opus-4.6-1m"


# ── find_extended_context_variant ─────────────────────────────────────


class TestFindExtendedContextVariant:
    def test_finds_1m_variant_with_dot(self):
        """Base model claude-opus-4-6 finds claude-opus-4.6-1m in cache."""
        cache = _make_cache(
            [
                ("claude-opus-4-6-20250617", "github-copilot", "Claude Opus 4.6"),
                ("claude-opus-4.6-1m", "github-copilot", "Claude Opus 4.6 1M"),
            ]
        )
        result = find_extended_context_variant("claude-opus-4-6", cache)
        assert result == "claude-opus-4.6-1m"

    def test_finds_1m_variant_from_dated_model(self):
        """Dated base model claude-opus-4-6-20250617 finds the -1m variant."""
        cache = _make_cache(
            [
                ("claude-opus-4-6-20250617", "github-copilot", "Claude Opus 4.6"),
                ("claude-opus-4.6-1m", "github-copilot", "Claude Opus 4.6 1M"),
            ]
        )
        result = find_extended_context_variant("claude-opus-4-6-20250617", cache)
        assert result == "claude-opus-4.6-1m"

    def test_returns_none_when_no_1m_variant(self):
        """Returns None when no -1m variant exists."""
        cache = _make_cache(
            [
                ("claude-opus-4-6-20250617", "github-copilot", "Claude Opus 4.6"),
            ]
        )
        result = find_extended_context_variant("claude-opus-4-6", cache)
        assert result is None

    def test_provider_prefix_filters(self):
        """Provider prefix narrows search to that provider only."""
        cache = _make_cache(
            [
                ("claude-opus-4.6-1m", "github-copilot", "Claude Opus 4.6 1M"),
                ("claude-opus-4.6-1m", "anthropic", "Claude Opus 4.6 1M"),
            ]
        )
        result = find_extended_context_variant("anthropic/claude-opus-4-6", cache)
        assert result == "anthropic/claude-opus-4.6-1m"

    def test_provider_prefix_no_match(self):
        """Returns None when provider doesn't have the 1m variant."""
        cache = _make_cache(
            [
                ("claude-opus-4.6-1m", "github-copilot", "Claude Opus 4.6 1M"),
            ]
        )
        result = find_extended_context_variant("anthropic/claude-opus-4-6", cache)
        assert result is None

    def test_empty_cache(self):
        result = find_extended_context_variant("claude-opus-4-6", {})
        assert result is None

    def test_sonnet_1m_variant(self):
        """Works for other model families too (sonnet)."""
        cache = _make_cache(
            [
                ("claude-sonnet-4.5-1m", "github-copilot", "Claude Sonnet 4.5 1M"),
            ]
        )
        result = find_extended_context_variant("claude-sonnet-4-5", cache)
        assert result == "claude-sonnet-4.5-1m"

    def test_finds_1m_internal_variant(self):
        """Base model claude-opus-4-7 finds claude-opus-4.7-1m-internal in cache."""
        cache = _make_cache(
            [
                ("claude-opus-4.7", "github-copilot", "Claude Opus 4.7"),
                (
                    "claude-opus-4.7-1m-internal",
                    "github-copilot",
                    "Claude Opus 4.7 1M (Internal)",
                ),
            ]
        )
        result = find_extended_context_variant("claude-opus-4-7", cache)
        assert result == "claude-opus-4.7-1m-internal"

    def test_prefers_plain_1m_over_internal(self):
        """When both -1m and -1m-internal exist, the plain -1m wins."""
        cache = _make_cache(
            [
                ("claude-opus-4.7-1m", "github-copilot", "Claude Opus 4.7 1M"),
                (
                    "claude-opus-4.7-1m-internal",
                    "github-copilot",
                    "Claude Opus 4.7 1M (Internal)",
                ),
            ]
        )
        result = find_extended_context_variant("claude-opus-4-7", cache)
        assert result == "claude-opus-4.7-1m"


# ── find_reasoning_variant ─────────────────────────────────────────────


class TestFindReasoningVariant:
    def test_finds_high_variant(self):
        """Base model claude-opus-4-7 with effort=high finds the -high variant."""
        cache = _make_cache(
            [
                ("claude-opus-4.7", "github-copilot", "Claude Opus 4.7"),
                ("claude-opus-4.7-high", "github-copilot", "Claude Opus 4.7 High"),
                ("claude-opus-4.7-xhigh", "github-copilot", "Claude Opus 4.7 xHigh"),
            ]
        )
        result = find_reasoning_variant("claude-opus-4-7", "high", cache)
        assert result == "claude-opus-4.7-high"

    def test_finds_xhigh_variant(self):
        cache = _make_cache(
            [
                ("claude-opus-4.7", "github-copilot", "Claude Opus 4.7"),
                ("claude-opus-4.7-xhigh", "github-copilot", "Claude Opus 4.7 xHigh"),
            ]
        )
        result = find_reasoning_variant("claude-opus-4-7", "xhigh", cache)
        assert result == "claude-opus-4.7-xhigh"

    def test_returns_none_when_variant_missing(self):
        cache = _make_cache(
            [
                ("claude-opus-4.7", "github-copilot", "Claude Opus 4.7"),
            ]
        )
        assert find_reasoning_variant("claude-opus-4-7", "high", cache) is None

    def test_skips_when_already_effort_encoded(self):
        """A request that already targets -high should not re-route to itself."""
        cache = _make_cache(
            [
                ("claude-opus-4.7-high", "github-copilot", "Claude Opus 4.7 High"),
            ]
        )
        assert find_reasoning_variant("claude-opus-4-7-high", "high", cache) is None

    def test_provider_prefix_filters(self):
        cache = _make_cache(
            [
                ("claude-opus-4.7-high", "github-copilot", "Claude Opus 4.7 High"),
            ]
        )
        result = find_reasoning_variant("github-copilot/claude-opus-4-7", "high", cache)
        assert result == "github-copilot/claude-opus-4.7-high"

    def test_empty_cache(self):
        assert find_reasoning_variant("claude-opus-4-7", "high", {}) is None
