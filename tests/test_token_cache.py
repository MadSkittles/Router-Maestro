"""Tests for token counting LRU cache (Feature 1)."""

from router_maestro.utils.tokens import clear_token_cache, count_tokens


class TestTokenCountingCache:
    """Tests for count_tokens LRU cache."""

    def setup_method(self):
        """Clear cache before each test for isolation."""
        clear_token_cache()

    def test_cache_returns_consistent_results(self):
        """Same input produces same output across cached calls."""
        text = "Hello, world!"
        first = count_tokens(text)
        second = count_tokens(text)
        assert first == second
        assert first > 0

    def test_cache_hit_after_first_call(self):
        """Cache info shows hits after repeated calls."""
        clear_token_cache()
        text = "test string for caching"
        count_tokens(text)
        count_tokens(text)
        info = count_tokens.cache_info()
        assert info.hits >= 1
        assert info.misses >= 1

    def test_different_texts_cached_separately(self):
        """Different texts have different cache entries."""
        a = count_tokens("hello")
        b = count_tokens("goodbye world, this is a longer string")
        assert a != b

    def test_different_encodings_cached_separately(self):
        """Same text with different encodings has separate cache entries."""
        text = "test encoding separation"
        a = count_tokens(text, "cl100k_base")
        b = count_tokens(text, "o200k_base")
        # Both should return valid token counts (may or may not be equal)
        assert a > 0
        assert b > 0

    def test_clear_cache_resets(self):
        """clear_token_cache resets all cached entries."""
        count_tokens("cached text")
        info_before = count_tokens.cache_info()
        assert info_before.currsize > 0

        clear_token_cache()
        info_after = count_tokens.cache_info()
        assert info_after.currsize == 0
        assert info_after.hits == 0
        assert info_after.misses == 0

    def test_empty_string_returns_zero(self):
        """Empty string returns 0 tokens (short-circuit, not cached)."""
        result = count_tokens("")
        assert result == 0

    def test_cache_maxsize(self):
        """Cache has a maxsize of 5000."""
        info = count_tokens.cache_info()
        assert info.maxsize == 5000
