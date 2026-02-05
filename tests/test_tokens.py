"""Tests for token estimation utilities."""

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
    CALIBRATION_CONFIG,
    CHARS_PER_TOKEN,
    MESSAGE_OVERHEAD_TOKENS,
    STRUCTURE_OVERHEAD_MULTIPLIER,
    calibrate_tokens,
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


class TestCalibrateTokens:
    """Tests for token calibration."""

    def test_calibrate_zero_tokens(self):
        """Test calibration with zero tokens returns zero."""
        assert calibrate_tokens(0) == 0
        assert calibrate_tokens(-10) == 0

    def test_calibrate_small_tokens_default(self):
        """Test calibration for small token counts with default config."""
        # Small range: slope=1.065, base_offset=-120
        base_tokens = 1000
        expected = max(1, round(1.065 * base_tokens + (-120)))
        assert calibrate_tokens(base_tokens, is_input=True, model=None) == expected

    def test_calibrate_medium_tokens_default(self):
        """Test calibration for medium token counts with default config."""
        # Medium range (10K-50K): slope=1.082, base_offset=1300
        base_tokens = 20000
        expected = round(1.082 * base_tokens + 1300)
        assert calibrate_tokens(base_tokens, is_input=True, model=None) == expected

    def test_calibrate_large_tokens_default(self):
        """Test calibration for large token counts with default config."""
        # Large range (50K-100K): slope=1.05, base_offset=2000
        base_tokens = 70000
        expected = round(1.05 * base_tokens + 2000)
        assert calibrate_tokens(base_tokens, is_input=True, model=None) == expected

    def test_calibrate_xlarge_tokens_default(self):
        """Test calibration for xlarge token counts with default config."""
        # XLarge range (>=100K): slope=1.05, base_offset=1500
        base_tokens = 120000
        expected = round(1.05 * base_tokens + 1500)
        assert calibrate_tokens(base_tokens, is_input=True, model=None) == expected

    def test_calibrate_opus_model(self):
        """Test calibration uses opus config for opus models."""
        # Opus small range: slope=1.1, base_offset=0
        base_tokens = 5000
        expected = round(1.1 * base_tokens + 0)
        assert calibrate_tokens(base_tokens, is_input=True, model="claude-opus-4") == expected
        assert (
            calibrate_tokens(base_tokens, is_input=True, model="github-copilot/claude-opus-4.5")
            == expected
        )

    def test_calibrate_opus_large_range(self):
        """Test opus calibration for large token ranges."""
        # Opus large range: slope=1.12, base_offset=1500
        base_tokens = 70000
        expected = round(1.12 * base_tokens + 1500)
        assert calibrate_tokens(base_tokens, is_input=True, model="opus") == expected

    def test_calibrate_output_tokens(self):
        """Test calibration for output tokens."""
        # Output tokens default: slope=0.67, base_offset=170
        base_tokens = 5000
        expected = round(0.67 * base_tokens + 170)
        assert calibrate_tokens(base_tokens, is_input=False, model=None) == expected

    def test_calibrate_output_tokens_opus(self):
        """Test calibration for opus output tokens."""
        # Opus output: slope=1.0, base_offset=150
        base_tokens = 5000
        expected = round(1.0 * base_tokens + 150)
        assert calibrate_tokens(base_tokens, is_input=False, model="opus") == expected

    def test_calibrate_non_opus_model(self):
        """Test non-opus models use default config."""
        base_tokens = 5000
        # Default small range: slope=1.065, base_offset=-120
        expected = round(1.065 * base_tokens + (-120))
        assert calibrate_tokens(base_tokens, is_input=True, model="claude-sonnet-4") == expected
        assert calibrate_tokens(base_tokens, is_input=True, model="gpt-4") == expected


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
        # Result should be positive and calibrated
        assert result > 0

    def test_with_system_prompt_string(self):
        """Test estimation with string system prompt."""
        messages = [AnthropicUserMessage(content="Hi")]  # 2 chars
        result = estimate_anthropic_request_tokens(
            system="You are a helpful assistant.",  # 30 chars
            messages=messages,
            tools=None,
        )
        # Result should be positive
        assert result > 0

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
        # Result should be positive
        assert result > 0

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
        assert result > 0

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
        assert result > 0

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
        assert result > 0

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

    def test_model_specific_calibration(self):
        """Test that model parameter affects calibration."""
        messages = [AnthropicUserMessage(content="A" * 3000)]  # ~1000 tokens base
        result_default = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
            model="claude-sonnet-4",
        )
        result_opus = estimate_anthropic_request_tokens(
            system=None,
            messages=messages,
            tools=None,
            model="claude-opus-4",
        )
        # Opus has different calibration coefficients
        assert result_default != result_opus


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

    def test_calibration_config_has_default(self):
        """CALIBRATION_CONFIG should have default and opus configs."""
        assert "default" in CALIBRATION_CONFIG
        assert "opus" in CALIBRATION_CONFIG

    def test_calibration_config_structure(self):
        """CALIBRATION_CONFIG should have input and output ranges."""
        for config_name in ["default", "opus"]:
            config = CALIBRATION_CONFIG[config_name]
            assert hasattr(config, "input")
            assert hasattr(config, "output")
            assert hasattr(config.input, "small")
            assert hasattr(config.input, "medium")
            assert hasattr(config.input, "large")
            assert hasattr(config.input, "xlarge")
