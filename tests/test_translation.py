"""Tests for the translation module."""

from router_maestro.server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicTextBlock,
    AnthropicToolChoice,
    AnthropicUserMessage,
)
from router_maestro.server.translation import (
    _extract_text_content,
    _translate_model_name,
    _translate_tool_choice,
    _translate_tools,
    translate_anthropic_to_openai,
)
from router_maestro.utils.tokens import map_openai_stop_reason_to_anthropic


class TestModelNameTranslation:
    """Tests for model name translation."""

    def test_translate_claude_with_date_suffix(self):
        """A concrete dated catalog identity must reach Router unchanged."""
        result = _translate_model_name("claude-sonnet-4-20250514")
        assert result == "claude-sonnet-4-20250514"

    def test_translate_claude_opus_with_date_suffix(self):
        """Dated Opus identities are exact model selections, not family aliases."""
        result = _translate_model_name("claude-opus-4-20250101")
        assert result == "claude-opus-4-20250101"

    def test_preserve_version_suffix(self):
        """Test preserving version numbers like .5."""
        result = _translate_model_name("claude-opus-4.5")
        assert result == "claude-opus-4.5"

    def test_preserve_version_with_date(self):
        """Preserve both dotted version and date for exact routing."""
        result = _translate_model_name("claude-sonnet-4.5-20250514")
        assert result == "claude-sonnet-4.5-20250514"

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

    def test_translate_pydantic_object(self):
        """Regression: tool_choice as a Pydantic object must translate, not None.

        After schema parsing, request.tool_choice is an AnthropicToolChoice
        instance (never a dict), so the translator must handle Pydantic objects.
        """
        assert _translate_tool_choice(AnthropicToolChoice(type="auto")) == "auto"
        assert _translate_tool_choice(AnthropicToolChoice(type="any")) == "required"
        assert _translate_tool_choice(AnthropicToolChoice(type="none")) == "none"
        assert _translate_tool_choice(AnthropicToolChoice(type="tool", name="get_weather")) == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_translate_via_full_request(self):
        """End-to-end: tool_choice survives translate_anthropic_to_openai."""
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[AnthropicUserMessage(role="user", content="hi")],
            tool_choice=AnthropicToolChoice(type="any"),
        )
        result = translate_anthropic_to_openai(request)
        assert result.tool_choice == "required"


class TestStopReasonMapping:
    """Tests for stop reason mapping."""

    def test_map_stop_to_end_turn(self):
        """Test mapping 'stop' to 'end_turn'."""
        assert map_openai_stop_reason_to_anthropic("stop") == "end_turn"

    def test_map_length_to_max_tokens(self):
        """Test mapping 'length' to 'max_tokens'."""
        assert map_openai_stop_reason_to_anthropic("length") == "max_tokens"

    def test_map_tool_calls_to_tool_use(self):
        """Test mapping 'tool_calls' to 'tool_use'."""
        assert map_openai_stop_reason_to_anthropic("tool_calls") == "tool_use"

    def test_map_none(self):
        """Test mapping None returns None."""
        assert map_openai_stop_reason_to_anthropic(None) is None

    def test_map_unknown_to_end_turn(self):
        """Test mapping unknown reason defaults to 'end_turn'."""
        assert map_openai_stop_reason_to_anthropic("unknown_reason") == "end_turn"


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


class TestToolResultWithToolReference:
    """Tests for tool_result handling with tool_reference blocks."""

    def test_tool_reference_blocks_are_filtered_out(self):
        """Test that tool_reference blocks are silently skipped."""
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4",
            max_tokens=1000,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_123", "name": "test_tool", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": [
                                {"type": "tool_reference", "tool_name": "mcp__github__create_pr"},
                                {
                                    "type": "tool_reference",
                                    "tool_name": "mcp__github__list_issues",
                                },
                                {"type": "text", "text": "Tool result content"},
                            ],
                        }
                    ],
                },
            ],
        )

        result = translate_anthropic_to_openai(request)

        # Find the tool message
        tool_messages = [m for m in result.messages if m.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "Tool result content"
        assert "tool_reference" not in tool_messages[0].content
        assert "mcp__github" not in tool_messages[0].content

    def test_tool_result_with_only_tool_references(self):
        """Test tool_result containing only tool_reference blocks."""
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4",
            max_tokens=1000,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_456", "name": "list_tools", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_456",
                            "content": [
                                {"type": "tool_reference", "tool_name": "mcp__slack__send"},
                                {"type": "tool_reference", "tool_name": "mcp__slack__read"},
                            ],
                        }
                    ],
                },
            ],
        )

        result = translate_anthropic_to_openai(request)

        # Find the tool message - should have empty content
        tool_messages = [m for m in result.messages if m.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == ""
        assert tool_messages[0].tool_call_id == "toolu_456"


class TestPermissiveRequestParsing:
    """Inbound permissiveness: clients send options RM doesn't model (F2, F3)."""

    def test_builtin_tool_without_input_schema_is_accepted(self):
        # Anthropic built-in/server tools carry no input_schema (F2).
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-sonnet-4.5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            }
        )
        assert req.tools[0].name == "web_search"

    def test_redacted_thinking_block_is_accepted(self):
        # Extended thinking is default-on for opus-4.8; redacted blocks replay (F3).
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4.8",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "xyz"}],
                    },
                    {"role": "user", "content": "continue"},
                ],
            }
        )
        assert len(req.messages) == 3

    def test_unknown_user_content_block_is_accepted(self):
        # A future/unmodeled user content block must not 422 (F3).
        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-sonnet-4.5",
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "future_block_2026", "payload": {"k": 1}}],
                    },
                ],
            }
        )
        assert len(req.messages) == 1
