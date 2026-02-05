"""Tests for token counting utilities using tiktoken."""

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
    calibrate_tokens,
    count_anthropic_request_tokens,
    count_tokens,
    estimate_anthropic_request_tokens,
    estimate_tokens,
    estimate_tokens_from_char_count,
)


class TestCountTokens:
    """Tests for tiktoken-based token counting."""

    def test_count_tokens_simple(self):
        """Test basic token counting."""
        # "Hello World!" is typically 3 tokens with cl100k_base
        result = count_tokens("Hello World!")
        assert result > 0
        assert result < 10  # Should be small

    def test_count_tokens_empty(self):
        """Test empty string returns zero."""
        assert count_tokens("") == 0

    def test_count_tokens_long_text(self):
        """Test longer text."""
        text = "This is a test sentence. " * 100
        result = count_tokens(text)
        # Should be reasonable (each sentence is ~6 tokens)
        assert 500 < result < 800

    def test_count_tokens_code(self):
        """Test code snippet."""
        code = """
def hello_world():
    print("Hello, World!")
"""
        result = count_tokens(code)
        assert result > 0

    def test_count_tokens_json(self):
        """Test JSON structure."""
        json_str = '{"type": "tool_use", "name": "Read", "input": {"file_path": "/path"}}'
        result = count_tokens(json_str)
        assert result > 0


class TestEstimateTokens:
    """Tests for estimate_tokens (now uses tiktoken)."""

    def test_estimate_tokens_uses_tiktoken(self):
        """Verify estimate_tokens now uses tiktoken."""
        text = "Hello World!"
        # Both should return the same value now
        assert estimate_tokens(text) == count_tokens(text)

    def test_estimate_tokens_empty(self):
        """Test empty string."""
        assert estimate_tokens("") == 0


class TestEstimateTokensFromCharCount:
    """Tests for legacy character-based estimation."""

    def test_estimate_tokens_from_char_count(self):
        """Test legacy character count conversion."""
        # 30 chars / 3 = 10 tokens
        assert estimate_tokens_from_char_count(30) == 10

    def test_estimate_tokens_from_char_count_zero(self):
        """Test zero characters."""
        assert estimate_tokens_from_char_count(0) == 0


class TestCalibrateTokens:
    """Tests for calibrate_tokens (now a no-op)."""

    def test_calibrate_is_noop(self):
        """Test that calibrate_tokens returns input unchanged."""
        assert calibrate_tokens(100) == 100
        assert calibrate_tokens(50000) == 50000
        assert calibrate_tokens(0) == 0

    def test_calibrate_ignores_parameters(self):
        """Test that parameters are ignored."""
        assert calibrate_tokens(100, is_input=True, model="opus") == 100
        assert calibrate_tokens(100, is_input=False, model="sonnet") == 100


class TestCountAnthropicRequestTokens:
    """Tests for Anthropic request token counting."""

    def test_simple_text_message(self):
        """Test counting with simple text messages."""
        messages = [
            AnthropicUserMessage(content="Hello, how are you?"),
            AnthropicAssistantMessage(content="I'm doing well, thank you!"),
        ]
        result = count_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        # Should be positive and include message overhead
        assert result > 0
        assert result >= 2 * MESSAGE_OVERHEAD_TOKENS

    def test_with_system_prompt_string(self):
        """Test counting with string system prompt."""
        messages = [AnthropicUserMessage(content="Hi")]
        result = count_anthropic_request_tokens(
            system="You are a helpful assistant.",
            messages=messages,
            tools=None,
        )
        assert result > 0

    def test_with_system_prompt_blocks(self):
        """Test counting with system prompt as list of text blocks."""
        messages = [AnthropicUserMessage(content="Hi")]
        system_blocks = [
            AnthropicTextBlock(text="You are helpful."),
            AnthropicTextBlock(text="Be concise."),
        ]
        result = count_anthropic_request_tokens(
            system=system_blocks,
            messages=messages,
            tools=None,
        )
        assert result > 0

    def test_with_tools(self):
        """Test counting with tool definitions."""
        messages = [AnthropicUserMessage(content="Use a tool")]
        tools = [
            AnthropicTool(
                name="get_weather",
                description="Get weather for a location",
                input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ]
        result = count_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=tools,
        )
        # Should include tool definition tokens
        assert result > 0

    def test_with_tool_no_description(self):
        """Test counting with tool that has no description."""
        messages = [AnthropicUserMessage(content="Hi")]
        tools = [
            AnthropicTool(
                name="test_tool",
                description=None,
                input_schema={"type": "object"},
            )
        ]
        result = count_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=tools,
        )
        assert result > 0

    def test_empty_request(self):
        """Test counting with minimal input."""
        result = count_anthropic_request_tokens(
            system=None,
            messages=[],
            tools=None,
        )
        assert result == 0

    def test_with_tool_use_block(self):
        """Test counting with tool_use content block."""
        messages = [
            AnthropicUserMessage(content="Get weather"),
            AnthropicAssistantMessage(
                content=[
                    AnthropicToolUseBlock(
                        id="tool_123",
                        name="get_weather",
                        input={"location": "San Francisco"},
                    )
                ]
            ),
        ]
        result = count_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        assert result > 0

    def test_with_tool_result_block_string(self):
        """Test counting with tool_result block containing string content."""
        messages = [
            AnthropicUserMessage(
                content=[
                    AnthropicToolResultBlock(
                        tool_use_id="tool_123",
                        content="The weather is sunny.",
                    )
                ]
            ),
        ]
        result = count_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        assert result > 0

    def test_with_tool_result_block_list(self):
        """Test counting with tool_result block containing list content."""
        messages = [
            AnthropicUserMessage(
                content=[
                    AnthropicToolResultBlock(
                        tool_use_id="tool_123",
                        content=[
                            AnthropicToolResultContentBlock(type="text", text="Result part 1"),
                            AnthropicToolResultContentBlock(type="text", text="Result part 2"),
                        ],
                    )
                ]
            ),
        ]
        result = count_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        assert result > 0

    def test_with_thinking_block(self):
        """Test counting with thinking content block."""
        messages = [
            AnthropicAssistantMessage(
                content=[
                    AnthropicThinkingBlock(thinking="Let me think about this..."),
                    AnthropicTextBlock(text="Here is my answer."),
                ]
            ),
        ]
        result = count_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        assert result > 0

    def test_complex_conversation(self):
        """Test counting with a complex multi-turn conversation."""
        messages = [
            AnthropicUserMessage(content="Hello!"),
            AnthropicAssistantMessage(content="Hi there! How can I help?"),
            AnthropicUserMessage(content="What's the weather?"),
            AnthropicAssistantMessage(
                content=[
                    AnthropicToolUseBlock(
                        id="tool_1",
                        name="weather",
                        input={"city": "NYC"},
                    )
                ]
            ),
            AnthropicUserMessage(
                content=[
                    AnthropicToolResultBlock(
                        tool_use_id="tool_1",
                        content="Sunny, 72F",
                    )
                ]
            ),
            AnthropicAssistantMessage(content="It's sunny and 72F in NYC!"),
        ]
        result = count_anthropic_request_tokens(
            system="You are a weather assistant.",
            messages=messages,
            tools=None,
        )
        assert result > 0
        # With 6 messages, overhead should be significant
        assert result >= 6 * MESSAGE_OVERHEAD_TOKENS


class TestEstimateAnthropicRequestTokens:
    """Tests for estimate_anthropic_request_tokens (backward compat alias)."""

    def test_is_alias_for_count(self):
        """Verify estimate is an alias for count."""
        messages = [AnthropicUserMessage(content="Hello")]
        count_result = count_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        estimate_result = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
        )
        assert count_result == estimate_result


class TestConstants:
    """Tests for constants."""

    def test_message_overhead_tokens(self):
        """MESSAGE_OVERHEAD_TOKENS should be 4."""
        assert MESSAGE_OVERHEAD_TOKENS == 4

    def test_chars_per_token_legacy(self):
        """CHARS_PER_TOKEN is kept for legacy compatibility."""
        assert CHARS_PER_TOKEN == 3
