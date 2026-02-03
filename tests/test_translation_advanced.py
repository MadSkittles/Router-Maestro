"""Advanced tests for translation module."""

from router_maestro.server.schemas.anthropic import (
    AnthropicAssistantMessage,
    AnthropicImageBlock,
    AnthropicImageSource,
    AnthropicStreamState,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolUseBlock,
    AnthropicUserMessage,
)
from router_maestro.server.translation import (
    _extract_multimodal_content,
    _extract_text_content,
    _extract_tool_calls,
    _handle_assistant_message,
    _handle_user_message,
    _is_tool_block_open,
    _sanitize_system_prompt,
    _translate_messages,
    _translate_model_name,
    _translate_tool_choice,
    _translate_tools,
    translate_openai_chunk_to_anthropic_events,
    translate_openai_to_anthropic,
)


class TestModelNameTranslationAdvanced:
    """Advanced tests for model name translation."""

    def test_translate_haiku_with_hyphenated_version(self):
        """Test translating haiku with hyphenated version number."""
        result = _translate_model_name("claude-haiku-4-5-20251001")
        assert result == "claude-haiku-4.5"

    def test_translate_sonnet_with_hyphenated_version(self):
        """Test translating sonnet with hyphenated version number."""
        result = _translate_model_name("claude-sonnet-4-5-20250514")
        assert result == "claude-sonnet-4.5"

    def test_preserve_plain_model_name(self):
        """Test that plain model names are unchanged."""
        result = _translate_model_name("claude-3-opus")
        assert result == "claude-3-opus"

    def test_preserve_gpt_model(self):
        """Test that GPT models are unchanged."""
        result = _translate_model_name("gpt-4o-2024-08-06")
        assert result == "gpt-4o-2024-08-06"


class TestSanitizeSystemPrompt:
    """Tests for system prompt sanitization."""

    def test_remove_billing_header(self):
        """Test removing x-anthropic-billing-header."""
        prompt = "You are helpful.\nx-anthropic-billing-header: some-value\nBe nice."
        result = _sanitize_system_prompt(prompt)
        assert "x-anthropic-billing-header" not in result
        assert "You are helpful." in result
        assert "Be nice." in result

    def test_preserve_normal_prompt(self):
        """Test that normal prompts are preserved."""
        prompt = "You are a helpful assistant."
        result = _sanitize_system_prompt(prompt)
        assert result == prompt

    def test_strip_whitespace(self):
        """Test that whitespace is stripped."""
        prompt = "  You are helpful.  "
        result = _sanitize_system_prompt(prompt)
        assert result == "You are helpful."


class TestToolTranslationAdvanced:
    """Advanced tests for tool translation."""

    def test_translate_multiple_tools(self):
        """Test translating multiple tools."""
        tools = [
            {"name": "tool1", "description": "First tool", "input_schema": {}},
            {"name": "tool2", "description": "Second tool", "input_schema": {"type": "object"}},
        ]
        result = _translate_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"

    def test_translate_tool_with_object_attributes(self):
        """Test translating tool with object-like attributes."""

        class ToolLike:
            name = "mock_tool"
            description = "A mock tool"
            input_schema = {"type": "object"}

        result = _translate_tools([ToolLike()])
        assert result[0]["function"]["name"] == "mock_tool"


class TestToolChoiceTranslationAdvanced:
    """Advanced tests for tool choice translation."""

    def test_translate_none_returns_none(self):
        """Test that None returns None."""
        result = _translate_tool_choice(None)
        assert result is None

    def test_translate_unknown_type_returns_none(self):
        """Test that unknown type returns None."""
        result = _translate_tool_choice({"type": "unknown"})
        assert result is None


class TestExtractToolCalls:
    """Tests for tool call extraction."""

    def test_extract_from_dict_tool_use(self):
        """Test extracting tool calls from dict blocks."""
        blocks = [
            {"type": "tool_use", "id": "tc-1", "name": "get_weather", "input": {"loc": "NYC"}}
        ]
        result = _extract_tool_calls(blocks)
        assert len(result) == 1
        assert result[0]["id"] == "tc-1"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        # Input should be JSON string
        assert '"loc"' in result[0]["function"]["arguments"]

    def test_extract_from_anthropic_tool_use_block(self):
        """Test extracting tool calls from AnthropicToolUseBlock."""
        blocks = [
            AnthropicToolUseBlock(
                type="tool_use", id="tc-2", name="search", input={"query": "test"}
            )
        ]
        result = _extract_tool_calls(blocks)
        assert len(result) == 1
        assert result[0]["id"] == "tc-2"
        assert result[0]["function"]["name"] == "search"

    def test_extract_empty_returns_none(self):
        """Test that empty blocks returns None."""
        result = _extract_tool_calls([])
        assert result is None

    def test_extract_no_tool_use_returns_none(self):
        """Test that blocks without tool_use returns None."""
        blocks = [{"type": "text", "text": "Hello"}]
        result = _extract_tool_calls(blocks)
        assert result is None


class TestExtractTextContent:
    """Tests for text content extraction."""

    def test_extract_thinking_block(self):
        """Test extracting text from thinking blocks."""
        blocks = [{"type": "thinking", "thinking": "Let me think..."}]
        result = _extract_text_content(blocks)
        assert result == "Let me think..."

    def test_extract_anthropic_thinking_block(self):
        """Test extracting from AnthropicThinkingBlock."""
        blocks = [AnthropicThinkingBlock(type="thinking", thinking="Deep thought")]
        result = _extract_text_content(blocks)
        assert result == "Deep thought"

    def test_extract_mixed_blocks(self):
        """Test extracting from mixed block types."""
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": "Hmm"},
            AnthropicTextBlock(type="text", text="World"),
        ]
        result = _extract_text_content(blocks)
        assert "Hello" in result
        assert "Hmm" in result
        assert "World" in result


class TestExtractMultimodalContent:
    """Tests for multimodal content extraction."""

    def test_text_only_returns_string(self):
        """Test that text-only content returns a string."""
        blocks = [{"type": "text", "text": "Hello"}]
        result = _extract_multimodal_content(blocks)
        assert result == "Hello"

    def test_image_returns_list(self):
        """Test that content with images returns a list."""
        blocks = [
            {"type": "text", "text": "Look at this:"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
            },
        ]
        result = _extract_multimodal_content(blocks)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in result[1]["image_url"]["url"]

    def test_anthropic_image_block(self):
        """Test extracting AnthropicImageBlock."""
        blocks = [
            AnthropicImageBlock(
                type="image",
                source=AnthropicImageSource(type="base64", media_type="image/jpeg", data="xyz789"),
            )
        ]
        result = _extract_multimodal_content(blocks)
        assert isinstance(result, list)
        assert result[0]["type"] == "image_url"
        assert "image/jpeg" in result[0]["image_url"]["url"]


class TestHandleUserMessage:
    """Tests for user message handling."""

    def test_simple_string_content(self):
        """Test handling simple string content."""
        msg = AnthropicUserMessage(role="user", content="Hello")
        result = _handle_user_message(msg)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "Hello"

    def test_dict_message(self):
        """Test handling dict message."""
        msg = {"role": "user", "content": "Hi there"}
        result = _handle_user_message(msg)
        assert len(result) == 1
        assert result[0].content == "Hi there"

    def test_tool_result_content(self):
        """Test handling tool result content."""
        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tc-1", "content": "Result data"}],
        }
        result = _handle_user_message(msg)
        assert len(result) == 1
        assert result[0].role == "tool"
        assert result[0].tool_call_id == "tc-1"
        assert result[0].content == "Result data"

    def test_tool_result_with_array_content(self):
        """Test handling tool result with array content."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tc-2",
                    "content": [
                        {"type": "text", "text": "Line 1"},
                        {"type": "text", "text": "Line 2"},
                    ],
                }
            ],
        }
        result = _handle_user_message(msg)
        assert result[0].content == "Line 1\nLine 2"


class TestHandleAssistantMessage:
    """Tests for assistant message handling."""

    def test_simple_string_content(self):
        """Test handling simple string content."""
        msg = AnthropicAssistantMessage(role="assistant", content="Hello")
        result = _handle_assistant_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "Hello"

    def test_content_with_tool_use(self):
        """Test handling content with tool use blocks."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "tc-1", "name": "search", "input": {"q": "test"}},
            ],
        }
        result = _handle_assistant_message(msg)
        assert len(result) == 1
        assert result[0].content == "Let me check"
        assert result[0].tool_calls is not None
        assert len(result[0].tool_calls) == 1


class TestTranslateMessages:
    """Tests for full message translation."""

    def test_with_string_system(self):
        """Test translation with string system prompt."""
        messages = [AnthropicUserMessage(role="user", content="Hi")]
        result = _translate_messages(messages, "You are helpful")
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[0].content == "You are helpful"

    def test_with_text_block_system(self):
        """Test translation with text block system prompt."""
        messages = [AnthropicUserMessage(role="user", content="Hi")]
        system = [
            AnthropicTextBlock(type="text", text="First part"),
            AnthropicTextBlock(type="text", text="Second part"),
        ]
        result = _translate_messages(messages, system)
        assert result[0].role == "system"
        assert "First part" in result[0].content
        assert "Second part" in result[0].content


class TestTranslateOpenAIToAnthropic:
    """Tests for OpenAI to Anthropic response translation."""

    def test_translate_simple_response(self):
        """Test translating a simple response."""
        openai_response = {
            "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = translate_openai_to_anthropic(openai_response, "claude-3", "req-123")

        assert result.id == "req-123"
        assert result.model == "claude-3"
        assert result.role == "assistant"
        assert len(result.content) == 1
        assert result.content[0].text == "Hello world"
        assert result.stop_reason == "end_turn"

    def test_translate_response_with_no_content(self):
        """Test translating response with no content."""
        openai_response = {
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }
        result = translate_openai_to_anthropic(openai_response, "claude-3", "req-123")

        assert result.id == "req-123"
        assert len(result.content) == 0


class TestTranslateOpenAIChunkToAnthropicEvents:
    """Tests for streaming chunk translation."""

    def test_message_start_event(self):
        """Test that first chunk generates message_start."""
        state = AnthropicStreamState()
        chunk = {"id": "chunk-1", "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}

        events = translate_openai_chunk_to_anthropic_events(chunk, state, "claude-3")

        # Should have message_start
        assert any(e["type"] == "message_start" for e in events)
        assert state.message_start_sent is True

    def test_content_delta_event(self):
        """Test content delta event generation."""
        state = AnthropicStreamState()
        state.message_start_sent = True

        chunk = {
            "id": "chunk-1",
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        }

        events = translate_openai_chunk_to_anthropic_events(chunk, state, "claude-3")

        # Should have content_block_start and content_block_delta
        delta_events = [e for e in events if e["type"] == "content_block_delta"]
        assert len(delta_events) == 1
        assert delta_events[0]["delta"]["text"] == "Hello"

    def test_finish_event(self):
        """Test finish event generation."""
        state = AnthropicStreamState()
        state.message_start_sent = True
        state.content_block_open = True

        chunk = {"id": "chunk-1", "choices": [{"delta": {}, "finish_reason": "stop"}]}

        events = translate_openai_chunk_to_anthropic_events(chunk, state, "claude-3")

        # Should have message_delta and message_stop
        assert any(e["type"] == "message_delta" for e in events)
        assert any(e["type"] == "message_stop" for e in events)
        assert state.message_complete is True

    def test_tool_call_events(self):
        """Test tool call event generation."""
        state = AnthropicStreamState()
        state.message_start_sent = True

        chunk = {
            "id": "chunk-1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc-1",
                                "function": {"name": "test", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        events = translate_openai_chunk_to_anthropic_events(chunk, state, "claude-3")

        # Should have content_block_start for tool_use
        start_events = [e for e in events if e["type"] == "content_block_start"]
        assert len(start_events) == 1
        assert start_events[0]["content_block"]["type"] == "tool_use"

    def test_no_events_after_complete(self):
        """Test that no events are generated after message is complete."""
        state = AnthropicStreamState()
        state.message_complete = True

        chunk = {
            "id": "chunk-1",
            "choices": [{"delta": {"content": "More"}, "finish_reason": None}],
        }

        events = translate_openai_chunk_to_anthropic_events(chunk, state, "claude-3")
        assert len(events) == 0


class TestIsToolBlockOpen:
    """Tests for tool block open detection."""

    def test_not_open_when_no_block(self):
        """Test returns False when no block is open."""
        state = AnthropicStreamState()
        state.content_block_open = False
        assert _is_tool_block_open(state) is False

    def test_not_open_when_text_block(self):
        """Test returns False when text block is open."""
        state = AnthropicStreamState()
        state.content_block_open = True
        state.content_block_index = 0
        state.tool_calls = {}  # No tool calls
        assert _is_tool_block_open(state) is False

    def test_open_when_tool_block(self):
        """Test returns True when tool block is open."""
        state = AnthropicStreamState()
        state.content_block_open = True
        state.content_block_index = 1
        state.tool_calls = {0: {"id": "tc-1", "name": "test", "anthropic_block_index": 1}}
        assert _is_tool_block_open(state) is True


class TestAnthropicStreamState:
    """Tests for AnthropicStreamState."""

    def test_default_values(self):
        """Test default state values."""
        state = AnthropicStreamState()
        assert state.message_start_sent is False
        assert state.content_block_index == 0
        assert state.content_block_open is False
        assert state.tool_calls == {}
        assert state.message_complete is False

    def test_estimated_input_tokens(self):
        """Test estimated input tokens."""
        state = AnthropicStreamState(estimated_input_tokens=1000)
        assert state.estimated_input_tokens == 1000
