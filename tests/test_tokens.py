"""Tests for token counting utilities using tiktoken."""

import math

import pytest

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
from router_maestro.utils.token_config import (
    ANTHROPIC_CONFIG,
    COPILOT_CONFIG,
    OPENAI_CONFIG,
    TokenCountingConfig,
)
from router_maestro.utils.tokens import (
    BASE_TOOL_TOKENS,
    CHARS_PER_TOKEN,
    TOKENS_PER_COMPLETION,
    TOKENS_PER_MESSAGE,
    TOKENS_PER_NAME,
    TOKENS_PER_TOOL,
    TOOL_CALLS_MULTIPLIER,
    TOOL_DEFINITION_MULTIPLIER,
    _count_message_object_tokens,
    _count_message_tokens,
    _count_object_tokens,
    _select_encoding,
    calculate_image_token_cost,
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
        result = count_tokens("Hello World!")
        assert result > 0
        assert result < 10

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_long_text(self):
        text = "This is a test sentence. " * 100
        result = count_tokens(text)
        assert 500 < result < 800

    def test_count_tokens_code(self):
        code = 'def hello_world():\n    print("Hello, World!")\n'
        result = count_tokens(code)
        assert result > 0

    def test_count_tokens_json(self):
        json_str = '{"type": "tool_use", "name": "Read", "input": {"file_path": "/path"}}'
        result = count_tokens(json_str)
        assert result > 0

    def test_count_tokens_o200k_encoding(self):
        text = "Hello World!"
        cl100k = count_tokens(text, "cl100k_base")
        o200k = count_tokens(text, "o200k_base")
        # Both should return positive values (may differ slightly)
        assert cl100k > 0
        assert o200k > 0


class TestSelectEncoding:
    """Tests for model-specific encoding selection."""

    def test_default_encoding(self):
        assert _select_encoding(None) == "cl100k_base"
        assert _select_encoding("claude-3-opus") == "cl100k_base"
        assert _select_encoding("gpt-4") == "cl100k_base"

    def test_o_series_encoding(self):
        assert _select_encoding("o1") == "o200k_base"
        assert _select_encoding("o1-preview") == "o200k_base"
        assert _select_encoding("o3-mini") == "o200k_base"
        assert _select_encoding("o4-mini") == "o200k_base"

    def test_o_series_with_provider_prefix(self):
        assert _select_encoding("github-copilot/o1") == "o200k_base"
        assert _select_encoding("openai/o3-mini") == "o200k_base"

    def test_non_o_series_not_matched(self):
        assert _select_encoding("gpt-4o") == "cl100k_base"
        assert _select_encoding("claude-opus") == "cl100k_base"


class TestEstimateTokens:
    """Tests for estimate_tokens (legacy alias)."""

    def test_estimate_tokens_uses_tiktoken(self):
        text = "Hello World!"
        assert estimate_tokens(text) == count_tokens(text)

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0


class TestEstimateTokensFromCharCount:
    """Tests for legacy character-based estimation."""

    def test_estimate_tokens_from_char_count(self):
        assert estimate_tokens_from_char_count(30) == 10

    def test_estimate_tokens_from_char_count_zero(self):
        assert estimate_tokens_from_char_count(0) == 0


class TestCalibrateTokens:
    """Tests for calibrate_tokens (now a no-op)."""

    def test_calibrate_is_noop(self):
        assert calibrate_tokens(100) == 100
        assert calibrate_tokens(50000) == 50000
        assert calibrate_tokens(0) == 0

    def test_calibrate_ignores_parameters(self):
        assert calibrate_tokens(100, is_input=True, model="opus") == 100
        assert calibrate_tokens(100, is_input=False, model="sonnet") == 100


class TestConstants:
    """Tests for token counting constants."""

    def test_tokens_per_message(self):
        assert TOKENS_PER_MESSAGE == 3

    def test_tokens_per_name(self):
        assert TOKENS_PER_NAME == 1

    def test_tokens_per_completion(self):
        assert TOKENS_PER_COMPLETION == 3

    def test_base_tool_tokens(self):
        assert BASE_TOOL_TOKENS == 16

    def test_tokens_per_tool(self):
        assert TOKENS_PER_TOOL == 8

    def test_tool_definition_multiplier(self):
        assert TOOL_DEFINITION_MULTIPLIER == 1.1

    def test_tool_calls_multiplier(self):
        assert TOOL_CALLS_MULTIPLIER == 1.5

    def test_chars_per_token_legacy(self):
        assert CHARS_PER_TOKEN == 3


class TestCalculateImageTokenCost:
    """Tests for image token cost calculation."""

    def test_low_detail(self):
        assert calculate_image_token_cost(1024, 768, detail="low") == 85

    def test_high_detail_small_image(self):
        # 512x512 -> scale to 768x768 -> 2x2 tiles = 4 tiles
        result = calculate_image_token_cost(512, 512, detail="high")
        assert result == 4 * 170 + 85  # 765

    def test_high_detail_large_image(self):
        # 4096x4096 -> scale to 2048x2048 -> scale to 768x768 -> 2x2 tiles
        result = calculate_image_token_cost(4096, 4096, detail="high")
        expected_tiles = math.ceil(768 / 512) * math.ceil(768 / 512)  # 2*2=4
        assert result == expected_tiles * 170 + 85

    def test_default_detail_is_high(self):
        # No detail specified should default to high-detail calculation
        result = calculate_image_token_cost(512, 512)
        assert result > 85  # More than low-detail cost

    def test_rectangular_image(self):
        # 1920x1080: shortest=1080, scale factor=768/1080
        # -> 1365x768, tiles = ceil(1365/512)*ceil(768/512) = 3*2 = 6
        result = calculate_image_token_cost(1920, 1080, detail="high")
        assert result > 85


class TestCountObjectTokens:
    """Tests for _count_object_tokens (tool definition counting)."""

    def test_counts_keys_and_values(self):
        obj = {"name": "test"}
        result = _count_object_tokens(obj, "cl100k_base")
        # Should count both "name" key and "test" value
        name_tokens = count_tokens("name")
        value_tokens = count_tokens("test")
        assert result == name_tokens + value_tokens

    def test_nested_dict(self):
        obj = {"outer": {"inner": "value"}}
        result = _count_object_tokens(obj, "cl100k_base")
        assert result > 0
        # Should count: "outer" + "inner" + "value"
        expected = count_tokens("outer") + count_tokens("inner") + count_tokens("value")
        assert result == expected

    def test_skips_none_values(self):
        obj = {"name": "test", "description": None}
        result = _count_object_tokens(obj, "cl100k_base")
        # description=None is skipped entirely
        expected = count_tokens("name") + count_tokens("test")
        assert result == expected

    def test_boolean_values(self):
        obj = {"required": True}
        result = _count_object_tokens(obj, "cl100k_base")
        # key "required" + True (1 token)
        assert result == count_tokens("required") + 1

    def test_numeric_values(self):
        obj = {"count": 42}
        result = _count_object_tokens(obj, "cl100k_base")
        assert result > 0

    def test_list_values(self):
        obj = {"items": ["a", "b", "c"]}
        result = _count_object_tokens(obj, "cl100k_base")
        assert result > 0


class TestCountMessageObjectTokens:
    """Tests for _count_message_object_tokens (message counting)."""

    def test_string_value(self):
        result = _count_message_object_tokens("hello", "cl100k_base")
        assert result == count_tokens("hello")

    def test_dict_values_only(self):
        obj = {"role": "user", "content": "hello"}
        result = _count_message_object_tokens(obj, "cl100k_base")
        # Keys are NOT counted, only values
        expected = count_tokens("user") + count_tokens("hello")
        assert result == expected

    def test_name_field_adds_extra_token(self):
        obj = {"name": "get_weather"}
        result = _count_message_object_tokens(obj, "cl100k_base")
        expected = count_tokens("get_weather") + TOKENS_PER_NAME
        assert result == expected

    def test_tool_calls_multiplier(self):
        tool_call = {"id": "call_1", "function": {"name": "test", "arguments": "{}"}}
        base_tokens = _count_message_object_tokens(tool_call, "cl100k_base")

        obj_with_tool_calls = {"tool_calls": [tool_call]}
        result = _count_message_object_tokens(obj_with_tool_calls, "cl100k_base")
        expected = int(base_tokens * TOOL_CALLS_MULTIPLIER)
        assert result == expected

    def test_image_url_low_detail(self):
        obj = {"type": "image_url", "image_url": {"url": "data:...", "detail": "low"}}
        result = _count_message_object_tokens(obj, "cl100k_base")
        # Should include 85 for low-detail image + type value tokens
        assert result >= 85

    def test_image_url_high_detail(self):
        obj = {"type": "image_url", "image_url": {"url": "data:...", "detail": "high"}}
        result = _count_message_object_tokens(obj, "cl100k_base")
        # Should include 765 for high-detail image + type value tokens
        assert result >= 765


class TestCountMessageTokens:
    """Tests for _count_message_tokens with tool_use multiplier."""

    def test_simple_message(self):
        msg = {"role": "user", "content": "hello"}
        result = _count_message_tokens(msg, "cl100k_base")
        expected = count_tokens("user") + count_tokens("hello")
        assert result == expected

    def test_tool_use_block_gets_multiplier(self):
        tool_block = {
            "type": "tool_use",
            "id": "tool_1",
            "name": "weather",
            "input": {"city": "NYC"},
        }
        base_tokens = _count_message_object_tokens(tool_block, "cl100k_base")

        msg = {"role": "assistant", "content": [tool_block]}
        result = _count_message_tokens(msg, "cl100k_base")
        # The tool_use block should get 1.5x, role "assistant" is counted normally
        expected = count_tokens("assistant") + int(base_tokens * TOOL_CALLS_MULTIPLIER)
        assert result == expected

    def test_mixed_content_blocks(self):
        text_block = {"type": "text", "text": "Let me check the weather."}
        tool_block = {
            "type": "tool_use",
            "id": "tool_1",
            "name": "weather",
            "input": {"city": "NYC"},
        }
        msg = {"role": "assistant", "content": [text_block, tool_block]}
        result = _count_message_tokens(msg, "cl100k_base")
        assert result > 0

    def test_openai_format_tool_calls(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}],
        }
        result = _count_message_tokens(msg, "cl100k_base")
        assert result > 0

    def test_message_with_name_field(self):
        msg = {"role": "user", "name": "John", "content": "hello"}
        result = _count_message_tokens(msg, "cl100k_base")
        # Should include TOKENS_PER_NAME for the name field
        base = count_tokens("user") + count_tokens("hello")
        name_tokens = count_tokens("John") + TOKENS_PER_NAME
        assert result == base + name_tokens


class TestCountAnthropicRequestTokens:
    """Tests for Anthropic request token counting."""

    def test_simple_text_message(self):
        messages = [
            AnthropicUserMessage(content="Hello, how are you?"),
            AnthropicAssistantMessage(content="I'm doing well, thank you!"),
        ]
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        assert result > 0
        # Should include: outer base (3) + 2*per_message (6) + completion (3) + content
        min_overhead = TOKENS_PER_MESSAGE + 2 * TOKENS_PER_MESSAGE + TOKENS_PER_COMPLETION
        assert result >= min_overhead

    def test_with_system_prompt_string(self):
        messages = [AnthropicUserMessage(content="Hi")]
        result = count_anthropic_request_tokens(
            system="You are a helpful assistant.",
            messages=messages,
            tools=None,
        )
        assert result > 0

    def test_with_system_prompt_blocks(self):
        messages = [AnthropicUserMessage(content="Hi")]
        system_blocks = [
            AnthropicTextBlock(text="You are helpful."),
            AnthropicTextBlock(text="Be concise."),
        ]
        result = count_anthropic_request_tokens(system=system_blocks, messages=messages, tools=None)
        assert result > 0

    def test_with_tools(self):
        messages = [AnthropicUserMessage(content="Use a tool")]
        tools = [
            AnthropicTool(
                name="get_weather",
                description="Get weather for a location",
                input_schema={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ]
        result_with_tools = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools
        )
        result_without_tools = count_anthropic_request_tokens(
            system=None, messages=messages, tools=None
        )
        # With tools should be significantly more due to BASE_TOOL_TOKENS + schema
        assert result_with_tools > result_without_tools
        assert result_with_tools >= result_without_tools + int(
            (BASE_TOOL_TOKENS + TOKENS_PER_TOOL) * TOOL_DEFINITION_MULTIPLIER
        )

    def test_with_tools_dict(self):
        messages = [AnthropicUserMessage(content="Use a tool")]
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=tools)
        assert result > 0

    def test_with_tool_no_description(self):
        messages = [AnthropicUserMessage(content="Hi")]
        tools = [
            AnthropicTool(
                name="test_tool",
                description=None,
                input_schema={"type": "object"},
            )
        ]
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=tools)
        assert result > 0

    def test_empty_request(self):
        result = count_anthropic_request_tokens(system=None, messages=[], tools=None)
        assert result == 0

    def test_with_tool_use_block(self):
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
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        assert result > 0

    def test_with_tool_use_block_dict(self):
        messages = [
            {"role": "user", "content": "Get weather"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    }
                ],
            },
        ]
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        assert result > 0

    def test_with_tool_result_block_string(self):
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
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        assert result > 0

    def test_with_tool_result_block_string_dict(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "The weather is sunny.",
                    }
                ],
            },
        ]
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        assert result > 0

    def test_with_tool_result_block_list(self):
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
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        assert result > 0

    def test_with_tool_result_block_list_dict(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": [
                            {"type": "text", "text": "Result part 1"},
                            {"type": "text", "text": "Result part 2"},
                        ],
                    }
                ],
            },
        ]
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        assert result > 0

    def test_with_thinking_block(self):
        messages = [
            AnthropicAssistantMessage(
                content=[
                    AnthropicThinkingBlock(thinking="Let me think about this..."),
                    AnthropicTextBlock(text="Here is my answer."),
                ]
            ),
        ]
        result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        assert result > 0

    def test_complex_conversation(self):
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
        # Should include: outer base (3) + 6*per_message (18) + completion (3) + content
        min_overhead = TOKENS_PER_MESSAGE + 6 * TOKENS_PER_MESSAGE + TOKENS_PER_COMPLETION
        assert result >= min_overhead

    def test_tool_use_gets_multiplier(self):
        """Verify that tool_use blocks are counted with the 1.5x multiplier."""
        messages_without_tool = [
            AnthropicUserMessage(content="Hello"),
        ]
        messages_with_tool = [
            AnthropicAssistantMessage(
                content=[
                    AnthropicToolUseBlock(
                        id="tool_1",
                        name="get_data",
                        input={"query": "select *"},
                    )
                ]
            ),
        ]
        result_no_tool = count_anthropic_request_tokens(system=None, messages=messages_without_tool)
        result_with_tool = count_anthropic_request_tokens(system=None, messages=messages_with_tool)
        # The tool version should be notably larger due to tool_use content + multiplier
        assert result_with_tool > result_no_tool

    def test_model_specific_encoding(self):
        """Verify that model parameter affects encoding selection."""
        messages = [AnthropicUserMessage(content="Hello world, this is a test.")]
        result_claude = count_anthropic_request_tokens(
            system=None, messages=messages, model="claude-3-opus"
        )
        result_o1 = count_anthropic_request_tokens(system=None, messages=messages, model="o1")
        # Both should be positive; values may differ due to different encodings
        assert result_claude > 0
        assert result_o1 > 0

    def test_multiple_tools_overhead(self):
        """Verify that multiple tools accumulate proper overhead."""
        messages = [AnthropicUserMessage(content="Hi")]
        one_tool = [
            AnthropicTool(
                name="tool_a",
                description="First tool",
                input_schema={"type": "object"},
            )
        ]
        two_tools = [
            AnthropicTool(
                name="tool_a",
                description="First tool",
                input_schema={"type": "object"},
            ),
            AnthropicTool(
                name="tool_b",
                description="Second tool",
                input_schema={"type": "object"},
            ),
        ]
        result_one = count_anthropic_request_tokens(system=None, messages=messages, tools=one_tool)
        result_two = count_anthropic_request_tokens(system=None, messages=messages, tools=two_tools)
        # Two tools should have more tokens than one
        assert result_two > result_one


class TestEstimateAnthropicRequestTokens:
    """Tests for estimate_anthropic_request_tokens (backward compat alias)."""

    def test_is_alias_for_count(self):
        messages = [AnthropicUserMessage(content="Hello")]
        count_result = count_anthropic_request_tokens(system=None, messages=messages, tools=None)
        estimate_result = estimate_anthropic_request_tokens(
            system=None, messages=messages, tools=None
        )
        assert count_result == estimate_result


class TestCountAnthropicRequestTokensWithConfig:
    """Parameterized tests for count_anthropic_request_tokens with different configs."""

    ALL_CONFIGS = [
        pytest.param(COPILOT_CONFIG, id="copilot"),
        pytest.param(ANTHROPIC_CONFIG, id="anthropic"),
        pytest.param(OPENAI_CONFIG, id="openai"),
    ]

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_simple_message_positive(self, config: TokenCountingConfig):
        messages = [AnthropicUserMessage(content="Hello, how are you?")]
        result = count_anthropic_request_tokens(system=None, messages=messages, config=config)
        assert result > 0

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_empty_messages_zero(self, config: TokenCountingConfig):
        result = count_anthropic_request_tokens(system=None, messages=[], config=config)
        assert result == 0

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_with_system_prompt(self, config: TokenCountingConfig):
        messages = [AnthropicUserMessage(content="Hi")]
        result = count_anthropic_request_tokens(
            system="You are a helpful assistant.",
            messages=messages,
            config=config,
        )
        assert result > 0

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_with_tools(self, config: TokenCountingConfig):
        messages = [AnthropicUserMessage(content="Use a tool")]
        tools = [
            AnthropicTool(
                name="get_weather",
                description="Get weather for a location",
                input_schema={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ]
        with_tools = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools, config=config
        )
        without_tools = count_anthropic_request_tokens(
            system=None, messages=messages, config=config
        )
        assert with_tools > without_tools

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_with_tool_use_block(self, config: TokenCountingConfig):
        messages = [
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
        result = count_anthropic_request_tokens(system=None, messages=messages, config=config)
        assert result > 0

    def test_copilot_vs_anthropic_tool_definitions(self):
        """Copilot config inflates tool definitions (1.1x, 16 base), Anthropic does not."""
        messages = [AnthropicUserMessage(content="Use tools")]
        tools = [
            AnthropicTool(
                name="tool_a",
                description="A longer tool description for meaningful difference",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                },
            ),
            AnthropicTool(
                name="tool_b",
                description="Another tool with some description text",
                input_schema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            ),
        ]
        copilot = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools, config=COPILOT_CONFIG
        )
        anthropic = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools, config=ANTHROPIC_CONFIG
        )
        openai = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools, config=OPENAI_CONFIG
        )
        # Copilot has highest inflation (1.1x multiplier + 16 base)
        assert copilot > anthropic
        assert copilot > openai
        # OpenAI has 8 base tool tokens, Anthropic has 0 -> OpenAI slightly higher
        assert openai > anthropic

    def test_copilot_vs_anthropic_tool_calls(self):
        """Copilot config inflates tool calls (1.5x), Anthropic does not (1.0x)."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "get_weather",
                        "input": {"location": "San Francisco", "units": "celsius"},
                    }
                ],
            },
        ]
        copilot = count_anthropic_request_tokens(
            system=None, messages=messages, config=COPILOT_CONFIG
        )
        anthropic = count_anthropic_request_tokens(
            system=None, messages=messages, config=ANTHROPIC_CONFIG
        )
        assert copilot > anthropic

    def test_estimate_alias_forwards_config(self):
        """estimate_anthropic_request_tokens forwards config parameter."""
        messages = [AnthropicUserMessage(content="Hello")]
        tools = [
            AnthropicTool(
                name="test",
                description="A test tool",
                input_schema={"type": "object"},
            )
        ]
        for config in [COPILOT_CONFIG, ANTHROPIC_CONFIG, OPENAI_CONFIG]:
            count_result = count_anthropic_request_tokens(
                system=None, messages=messages, tools=tools, config=config
            )
            estimate_result = estimate_anthropic_request_tokens(
                system=None, messages=messages, tools=tools, config=config
            )
            assert count_result == estimate_result
