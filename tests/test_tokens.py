"""Tests for token estimation utilities."""

import json

from router_maestro.server.schemas.anthropic import (
    AnthropicAssistantMessage,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicTool,
    AnthropicToolResultBlock,
    AnthropicToolResultContentBlock,
    AnthropicToolUseBlock,
    AnthropicUserMessage,
)
from router_maestro.utils.tokens import (
    CHARS_PER_TOKEN,
    MESSAGE_OVERHEAD_TOKENS,
    STRUCTURE_OVERHEAD_MULTIPLIER,
    estimate_anthropic_request_tokens,
    estimate_tokens,
    estimate_tokens_from_char_count,
)


class TestEstimateTokens:
    """Tests for basic token estimation."""

    def test_estimate_tokens_simple(self):
        """Test basic character to token conversion."""
        # 12 characters / 3 chars per token = 4 tokens
        assert estimate_tokens("Hello World!") == 4

    def test_estimate_tokens_empty(self):
        """Test empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test text shorter than chars_per_token."""
        # 2 characters / 3 = 0 tokens (integer division)
        assert estimate_tokens("ab") == 0

    def test_estimate_tokens_long_text(self):
        """Test longer text."""
        text = "a" * 300
        assert estimate_tokens(text) == 100  # 300 / 3 = 100


class TestEstimateTokensFromCharCount:
    """Tests for character count to token conversion."""

    def test_estimate_tokens_from_char_count(self):
        """Test direct character count conversion."""
        assert estimate_tokens_from_char_count(30) == 10  # 30 / 3 = 10

    def test_estimate_tokens_from_char_count_zero(self):
        """Test zero characters."""
        assert estimate_tokens_from_char_count(0) == 0


class TestEstimateAnthropicRequestTokens:
    """Tests for Anthropic request token estimation."""

    def test_simple_text_message(self):
        """Test estimation with simple text messages."""
        messages = [
            AnthropicUserMessage(content="Hello, how are you?"),  # 20 chars
            AnthropicAssistantMessage(content="I'm doing well, thank you!"),  # 27 chars
        ]
        result = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        # Base: (20 + 27) / 3 = 15 tokens (integer division)
        # Message overhead: 2 * 4 = 8 tokens
        # Total before multiplier: 23
        # With 1.25 multiplier: 28.75 -> 28
        expected = int(
            (47 // CHARS_PER_TOKEN + 2 * MESSAGE_OVERHEAD_TOKENS) * STRUCTURE_OVERHEAD_MULTIPLIER
        )
        assert result == expected

    def test_with_system_prompt_string(self):
        """Test estimation with string system prompt."""
        messages = [AnthropicUserMessage(content="Hi")]  # 2 chars
        result = estimate_anthropic_request_tokens(
            system="You are a helpful assistant.",  # 30 chars
            messages=messages,
            tools=None,
        )
        # Base: (30 + 2) / 3 = 10 tokens
        # Message overhead: 1 * 4 = 4 tokens
        # Total before multiplier: 14
        # With 1.25 multiplier: 17.5 -> 17
        expected = int(
            (32 // CHARS_PER_TOKEN + 1 * MESSAGE_OVERHEAD_TOKENS) * STRUCTURE_OVERHEAD_MULTIPLIER
        )
        assert result == expected

    def test_with_system_prompt_blocks(self):
        """Test estimation with system prompt as list of text blocks."""
        messages = [AnthropicUserMessage(content="Hi")]  # 2 chars
        system_blocks = [
            AnthropicTextBlock(text="You are helpful."),  # 16 chars
            AnthropicTextBlock(text="Be concise."),  # 11 chars
        ]
        result = estimate_anthropic_request_tokens(
            system=system_blocks,
            messages=messages,
            tools=None,
        )
        # System: 16 + 11 = 27 chars
        # Message: 2 chars
        # Base: 29 / 3 = 9 tokens
        # Message overhead: 1 * 4 = 4 tokens
        # Total before multiplier: 13
        # With 1.25 multiplier: 16.25 -> 16
        expected = int(
            (29 // CHARS_PER_TOKEN + 1 * MESSAGE_OVERHEAD_TOKENS) * STRUCTURE_OVERHEAD_MULTIPLIER
        )
        assert result == expected

    def test_with_tools(self):
        """Test estimation with tool definitions."""
        messages = [AnthropicUserMessage(content="Use a tool")]  # 10 chars
        tools = [
            AnthropicTool(
                name="get_weather",  # 11 chars
                description="Get weather for a location",  # 27 chars
                input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ]
        result = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=tools,
        )
        # Should include tool name, description, and schema
        assert result > 0
        # Result should be substantially larger than just message tokens
        message_only = int(
            (10 // CHARS_PER_TOKEN + 1 * MESSAGE_OVERHEAD_TOKENS) * STRUCTURE_OVERHEAD_MULTIPLIER
        )
        assert result > message_only

    def test_with_tool_no_description(self):
        """Test estimation with tool that has no description."""
        messages = [AnthropicUserMessage(content="Hi")]  # 2 chars
        tools = [
            AnthropicTool(
                name="test_tool",  # 9 chars
                description=None,
                input_schema={"type": "object"},
            )
        ]
        result = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=tools,
        )
        assert result > 0

    def test_empty_request(self):
        """Test estimation with minimal input."""
        result = estimate_anthropic_request_tokens(
            system=None,
            messages=[],
            tools=None,
        )
        assert result == 0

    def test_with_tool_use_block(self):
        """Test estimation with tool_use content block."""
        tool_input = {"location": "San Francisco"}
        messages = [
            AnthropicUserMessage(content="Get weather"),
            AnthropicAssistantMessage(
                content=[
                    AnthropicToolUseBlock(
                        id="tool_123",
                        name="get_weather",  # 11 chars
                        input=tool_input,
                    )
                ]
            ),
        ]
        result = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        # Should include tool name and input JSON
        assert result > 0
        input_json_len = len(json.dumps(tool_input))
        expected_chars = 11 + 11 + input_json_len  # "Get weather" + tool name + input
        expected = int(
            (expected_chars // CHARS_PER_TOKEN + 2 * MESSAGE_OVERHEAD_TOKENS)
            * STRUCTURE_OVERHEAD_MULTIPLIER
        )
        assert result == expected

    def test_with_tool_result_block_string(self):
        """Test estimation with tool_result block containing string content."""
        messages = [
            AnthropicUserMessage(
                content=[
                    AnthropicToolResultBlock(
                        tool_use_id="tool_123",
                        content="The weather is sunny.",  # 21 chars
                    )
                ]
            ),
        ]
        result = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        expected = int(
            (21 // CHARS_PER_TOKEN + 1 * MESSAGE_OVERHEAD_TOKENS) * STRUCTURE_OVERHEAD_MULTIPLIER
        )
        assert result == expected

    def test_with_tool_result_block_list(self):
        """Test estimation with tool_result block containing list content."""
        messages = [
            AnthropicUserMessage(
                content=[
                    AnthropicToolResultBlock(
                        tool_use_id="tool_123",
                        content=[
                            AnthropicToolResultContentBlock(
                                type="text", text="Result part 1"
                            ),  # 13 chars
                            AnthropicToolResultContentBlock(
                                type="text", text="Result part 2"
                            ),  # 13 chars
                        ],
                    )
                ]
            ),
        ]
        result = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        expected = int(
            (26 // CHARS_PER_TOKEN + 1 * MESSAGE_OVERHEAD_TOKENS) * STRUCTURE_OVERHEAD_MULTIPLIER
        )
        assert result == expected

    def test_with_thinking_block(self):
        """Test estimation with thinking content block."""
        messages = [
            AnthropicAssistantMessage(
                content=[
                    AnthropicThinkingBlock(thinking="Let me think about this..."),  # 27 chars
                    AnthropicTextBlock(text="Here is my answer."),  # 19 chars
                ]
            ),
        ]
        result = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        # Should count both thinking and text content
        # Result should be > 0 and include both blocks
        assert result > 0
        # With 46 chars total / 3 = 15 tokens + 4 message overhead = 19 * 1.25 = 23
        # But text block also has "text" attribute so it may be counted
        # Just verify it's a reasonable value
        assert result >= 20

    def test_complex_conversation(self):
        """Test estimation with a complex multi-turn conversation."""
        messages = [
            AnthropicUserMessage(content="Hello!"),  # 6 chars
            AnthropicAssistantMessage(content="Hi there! How can I help?"),  # 25 chars
            AnthropicUserMessage(content="What's the weather?"),  # 19 chars
            AnthropicAssistantMessage(
                content=[
                    AnthropicToolUseBlock(
                        id="tool_1",
                        name="weather",  # 7 chars
                        input={"city": "NYC"},  # 14 chars as JSON
                    )
                ]
            ),
            AnthropicUserMessage(
                content=[
                    AnthropicToolResultBlock(
                        tool_use_id="tool_1",
                        content="Sunny, 72F",  # 10 chars
                    )
                ]
            ),
            AnthropicAssistantMessage(content="It's sunny and 72F in NYC!"),  # 26 chars
        ]
        result = estimate_anthropic_request_tokens(
            system="You are a weather assistant.",  # 28 chars
            messages=messages,
            tools=None,
        )
        # Should handle all message types correctly
        assert result > 0
        # With 6 messages, overhead should be significant
        assert result >= 6 * MESSAGE_OVERHEAD_TOKENS


class TestConstants:
    """Tests for token estimation constants."""

    def test_chars_per_token_is_conservative(self):
        """CHARS_PER_TOKEN should be 3 for conservative estimation."""
        assert CHARS_PER_TOKEN == 3

    def test_message_overhead_tokens(self):
        """MESSAGE_OVERHEAD_TOKENS should be 4."""
        assert MESSAGE_OVERHEAD_TOKENS == 4

    def test_structure_overhead_multiplier(self):
        """STRUCTURE_OVERHEAD_MULTIPLIER should be 1.25."""
        assert STRUCTURE_OVERHEAD_MULTIPLIER == 1.25
