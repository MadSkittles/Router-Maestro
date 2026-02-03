"""Tests for Anthropic model display name generation."""

from router_maestro.server.routes.anthropic import _generate_display_name


class TestGenerateDisplayName:
    """Tests for display name generation."""

    def test_with_provider_prefix(self):
        """Test generating display name with provider prefix."""
        result = _generate_display_name("github-copilot/claude-sonnet-4")
        assert "Claude" in result
        assert "Sonnet" in result
        assert "github-copilot" in result

    def test_without_provider_prefix(self):
        """Test generating display name without provider prefix."""
        result = _generate_display_name("claude-sonnet-4")
        assert "Claude" in result
        assert "Sonnet" in result
        assert "(" not in result  # No provider suffix

    def test_preserves_version_numbers(self):
        """Test that version numbers are preserved."""
        result = _generate_display_name("gpt-4o")
        assert "4" in result

    def test_underscores_to_spaces(self):
        """Test that underscores are converted to spaces."""
        result = _generate_display_name("model_name_here")
        # Should have spaces instead of underscores
        assert "_" not in result

    def test_capitalizes_words(self):
        """Test that words are capitalized."""
        result = _generate_display_name("test-model")
        assert result[0].isupper()

    def test_gpt_model(self):
        """Test GPT model display name."""
        result = _generate_display_name("openai/gpt-4o-mini")
        assert "Gpt" in result
        assert "openai" in result

    def test_version_with_dot(self):
        """Test version number with dot."""
        result = _generate_display_name("claude-3.5-sonnet")
        assert "3.5" in result
