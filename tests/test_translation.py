"""Tests for the translation module."""

import pytest

from router_maestro.server.translation import (
    _extract_text_content,
    _map_stop_reason,
    _translate_model_name,
    _translate_tool_choice,
    _translate_tools,
    translate_anthropic_to_openai,
)
from router_maestro.server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicTextBlock,
    AnthropicUserMessage,
)


class TestModelNameTranslation:
    """Tests for model name translation."""

    def test_translate_claude_with_date_suffix(self):
        """Test removing date suffix from Claude model names."""
        result = _translate_model_name("claude-sonnet-4-20250514")
        assert result == "claude-sonnet-4"

    def test_translate_claude_opus_with_date_suffix(self):
        """Test removing date suffix from Claude Opus."""
        result = _translate_model_name("claude-opus-4-20250101")
        assert result == "claude-opus-4"

    def test_preserve_version_suffix(self):
        """Test preserving version numbers like .5."""
        result = _translate_model_name("claude-opus-4.5")
        assert result == "claude-opus-4.5"

    def test_preserve_version_with_date(self):
        """Test model with version and date suffix."""
        result = _translate_model_name("claude-sonnet-4.5-20250514")
        assert result == "claude-sonnet-4.5"

    def test_non_claude_model_unchanged(self):
        """Test that non-Claude models pass through unchanged."""
        result = _translate_model_name("gpt-4o")
        assert result == "gpt-4o"


class TestToolTranslation:
    """Tests for tool format translation."""

    def test_translate_single_tool(self):
        """Test translating a single Anthropic tool to OpenAI format."""
        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]
        result = _translate_tools(anthropic_tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get current weather"
        assert "properties" in result[0]["function"]["parameters"]


class TestToolChoiceTranslation:
    """Tests for tool_choice translation."""

    def test_translate_auto(self):
        """Test translating auto tool choice."""
        result = _translate_tool_choice({"type": "auto"})
        assert result == "auto"

    def test_translate_any(self):
        """Test translating any tool choice to required."""
        result = _translate_tool_choice({"type": "any"})
        assert result == "required"

    def test_translate_specific_tool(self):
        """Test translating specific tool choice."""
        result = _translate_tool_choice({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}


class TestStopReasonMapping:
    """Tests for stop reason mapping."""

    def test_map_stop_to_end_turn(self):
        """Test mapping 'stop' to 'end_turn'."""
        assert _map_stop_reason("stop") == "end_turn"

    def test_map_length_to_max_tokens(self):
        """Test mapping 'length' to 'max_tokens'."""
        assert _map_stop_reason("length") == "max_tokens"

    def test_map_tool_calls_to_tool_use(self):
        """Test mapping 'tool_calls' to 'tool_use'."""
        assert _map_stop_reason("tool_calls") == "tool_use"

    def test_map_none(self):
        """Test mapping None returns None."""
        assert _map_stop_reason(None) is None

    def test_map_unknown_to_end_turn(self):
        """Test mapping unknown reason defaults to 'end_turn'."""
        assert _map_stop_reason("unknown_reason") == "end_turn"


class TestTextContentExtraction:
    """Tests for text content extraction from blocks."""

    def test_extract_from_dict_text_block(self):
        """Test extracting text from dict text block."""
        blocks = [{"type": "text", "text": "Hello world"}]
        result = _extract_text_content(blocks)
        assert result == "Hello world"

    def test_extract_from_multiple_blocks(self):
        """Test extracting text from multiple blocks."""
        blocks = [
            {"type": "text", "text": "First"},
            {"type": "text", "text": "Second"},
        ]
        result = _extract_text_content(blocks)
        assert result == "First\n\nSecond"

    def test_extract_from_anthropic_text_block(self):
        """Test extracting text from AnthropicTextBlock."""
        blocks = [AnthropicTextBlock(type="text", text="From Anthropic block")]
        result = _extract_text_content(blocks)
        assert result == "From Anthropic block"

    def test_ignore_non_text_blocks(self):
        """Test that non-text blocks are ignored."""
        blocks = [
            {"type": "text", "text": "Text content"},
            {"type": "image", "source": {}},
        ]
        result = _extract_text_content(blocks)
        assert result == "Text content"


class TestAnthropicToOpenAITranslation:
    """Tests for full request translation."""

    def test_translate_simple_request(self):
        """Test translating a simple Anthropic request."""
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4",
            max_tokens=1024,
            messages=[AnthropicUserMessage(role="user", content="Hello")],
        )
        result = translate_anthropic_to_openai(request)

        assert result.model == "claude-sonnet-4"
        assert result.max_tokens == 1024
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"

    def test_translate_with_system_prompt(self):
        """Test translating request with system prompt."""
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[AnthropicUserMessage(role="user", content="Hi")],
        )
        result = translate_anthropic_to_openai(request)

        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are a helpful assistant."
        assert result.messages[1].role == "user"
