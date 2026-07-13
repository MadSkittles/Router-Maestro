"""Advanced tests for translation module."""

import json

import pytest

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
    AnthropicStreamProtocolError,
    _extract_multimodal_content,
    _extract_text_content,
    _extract_tool_calls,
    _handle_assistant_message,
    _handle_user_message,
    _sanitize_system_prompt,
    _translate_messages,
    _translate_model_name,
    _translate_tool_choice,
    _translate_tools,
    translate_openai_chunk_to_anthropic_events,
    translate_openai_to_anthropic,
)


def _stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "id": "chunk-test",
        "choices": [{"delta": delta, "finish_reason": finish_reason}],
    }


def _tool_delta(
    *,
    index: int | None = None,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> dict:
    tool_call: dict = {"function": {}}
    if index is not None:
        tool_call["index"] = index
    if tool_id is not None:
        tool_call["id"] = tool_id
    if name is not None:
        tool_call["function"]["name"] = name
    if arguments is not None:
        tool_call["function"]["arguments"] = arguments
    return tool_call


def _tool_blocks(events: list[dict]) -> list[dict]:
    blocks: list[dict] = []
    for position, event in enumerate(events):
        block = event.get("content_block", {})
        if event.get("type") != "content_block_start" or block.get("type") != "tool_use":
            continue
        index = event["index"]
        delta_position, delta = next(
            (candidate_position, candidate)
            for candidate_position, candidate in enumerate(events)
            if candidate.get("type") == "content_block_delta"
            and candidate.get("index") == index
            and candidate.get("delta", {}).get("type") == "input_json_delta"
        )
        stop_position = next(
            candidate_position
            for candidate_position, candidate in enumerate(events)
            if candidate.get("type") == "content_block_stop" and candidate.get("index") == index
        )
        blocks.append(
            {
                "id": block["id"],
                "json": delta["delta"]["partial_json"],
                "start_index": position,
                "delta_index": delta_position,
                "stop_index": stop_position,
            }
        )
    return blocks


class TestModelNameTranslationAdvanced:
    """Advanced tests for model name translation."""

    def test_translate_haiku_with_hyphenated_version(self):
        """Hyphenated dated aliases remain concrete identities for Router."""
        result = _translate_model_name("claude-haiku-4-5-20251001")
        assert result == "claude-haiku-4-5-20251001"

    def test_translate_sonnet_with_hyphenated_version(self):
        """Do not collapse a dated Sonnet selection into an undated alias."""
        result = _translate_model_name("claude-sonnet-4-5-20250514")
        assert result == "claude-sonnet-4-5-20250514"

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

    def test_extract_thinking_block_dropped(self):
        """Thinking blocks are dropped — they don't belong in OpenAI content."""
        blocks = [{"type": "thinking", "thinking": "Let me think..."}]
        result = _extract_text_content(blocks)
        assert result == ""

    def test_extract_anthropic_thinking_block_dropped(self):
        """AnthropicThinkingBlock is dropped from extracted text content."""
        blocks = [AnthropicThinkingBlock(type="thinking", thinking="Deep thought")]
        result = _extract_text_content(blocks)
        assert result == ""

    def test_extract_mixed_blocks(self):
        """Test extracting from mixed block types — thinking is excluded."""
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": "Hmm"},
            AnthropicTextBlock(type="text", text="World"),
        ]
        result = _extract_text_content(blocks)
        assert "Hello" in result
        assert "Hmm" not in result
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

    def test_document_block_passes_through(self):
        """Document blocks should be preserved in Anthropic-native shape."""
        blocks = [
            {"type": "text", "text": "Summarize this:"},
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjQK",
                },
                "title": "spec.pdf",
            },
        ]
        result = _extract_multimodal_content(blocks)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Summarize this:"}
        assert result[1]["type"] == "document"
        assert result[1]["source"] == {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBERi0xLjQK",
        }
        assert result[1]["title"] == "spec.pdf"


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

    def test_tool_result_with_image_injects_user_message(self):
        """Test that images in tool_result are injected as a follow-up user message."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tc-img",
                    "content": [
                        {"type": "text", "text": "Image file contents"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        },
                    ],
                }
            ],
        }
        result = _handle_user_message(msg)
        # Should produce: tool message + user message with image
        assert len(result) == 2
        assert result[0].role == "tool"
        assert result[0].tool_call_id == "tc-img"
        assert result[0].content == "Image file contents"
        assert result[1].role == "user"
        assert isinstance(result[1].content, list)
        assert result[1].content[0]["type"] == "image_url"
        assert "data:image/png;base64,iVBORw0KGgo=" in result[1].content[0]["image_url"]["url"]

    def test_tool_result_with_only_image_no_text(self):
        """Test tool_result containing only an image block."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tc-img2",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "/9j/4AAQ",
                            },
                        },
                    ],
                }
            ],
        }
        result = _handle_user_message(msg)
        assert len(result) == 2
        assert result[0].role == "tool"
        assert result[0].content == ""
        assert result[1].role == "user"
        assert isinstance(result[1].content, list)
        assert result[1].content[0]["type"] == "image_url"

    def test_multiple_tool_results_with_images_no_interleaving(self):
        """Test that multiple tool_results with images produce consecutive tool messages
        followed by a single user message with all images (no interleaving)."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tc-1",
                    "content": [
                        {"type": "text", "text": "First result"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "img1data",
                            },
                        },
                    ],
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tc-2",
                    "content": [
                        {"type": "text", "text": "Second result"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "img2data",
                            },
                        },
                    ],
                },
            ],
        }
        result = _handle_user_message(msg)
        # Should produce: tool, tool, user (with both images) — no interleaving
        assert len(result) == 3
        assert result[0].role == "tool"
        assert result[0].tool_call_id == "tc-1"
        assert result[0].content == "First result"
        assert result[1].role == "tool"
        assert result[1].tool_call_id == "tc-2"
        assert result[1].content == "Second result"
        assert result[2].role == "user"
        assert isinstance(result[2].content, list)
        # Both images should be in the single user message
        assert len(result[2].content) == 2
        assert "img1data" in result[2].content[0]["image_url"]["url"]
        assert "img2data" in result[2].content[1]["image_url"]["url"]

    def test_tool_result_without_image_no_extra_message(self):
        """Test that tool_result without images does NOT inject extra user message."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tc-text",
                    "content": [
                        {"type": "text", "text": "Just text"},
                    ],
                }
            ],
        }
        result = _handle_user_message(msg)
        assert len(result) == 1
        assert result[0].role == "tool"

    def test_tool_result_with_document_injects_user_message(self):
        """Documents in tool_result should be carried in a follow-up user message."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tc-doc",
                    "content": [
                        {"type": "text", "text": "PDF file read: spec.pdf (2KB)"},
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQK",
                            },
                        },
                    ],
                }
            ],
        }
        result = _handle_user_message(msg)
        assert len(result) == 2
        assert result[0].role == "tool"
        assert result[0].tool_call_id == "tc-doc"
        assert result[0].content == "PDF file read: spec.pdf (2KB)"
        assert result[1].role == "user"
        assert isinstance(result[1].content, list)
        assert result[1].content[0]["type"] == "document"
        assert result[1].content[0]["source"]["data"] == "JVBERi0xLjQK"

    def test_user_message_with_document_block_parses(self):
        """AnthropicMessagesRequest should accept user messages containing document blocks."""
        from router_maestro.server.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4-5",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What does this PDF say?"},
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": "JVBERi0xLjQK",
                                },
                            },
                        ],
                    }
                ],
            }
        )
        assert len(req.messages) == 1
        blocks = req.messages[0].content
        assert isinstance(blocks, list)
        assert blocks[1].type == "document"
        assert blocks[1].source.media_type == "application/pdf"


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

    def test_translate_response_with_tool_calls(self):
        """Test translating response with tool_calls produces AnthropicToolUseBlock."""
        openai_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "exec",
                                    "arguments": '{"command": "hostname"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
        result = translate_openai_to_anthropic(openai_response, "claude-3", "req-456")

        assert result.stop_reason == "tool_use"
        tool_blocks = [b for b in result.content if b.type == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].id == "call_abc"
        assert tool_blocks[0].name == "exec"
        assert tool_blocks[0].input == {"command": "hostname"}


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

    def test_tool_call_is_buffered_until_explicit_terminal(self):
        """Tool blocks are transactional and cannot be exposed before validation."""
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
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

        events = translate_openai_chunk_to_anthropic_events(chunk, state, "claude-3")

        assert events == []

        terminal = {
            "id": "chunk-2",
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
        }
        events = translate_openai_chunk_to_anthropic_events(terminal, state, "claude-3")

        assert [event["type"] for event in events] == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[0]["content_block"] == {
            "type": "tool_use",
            "id": "tc-1",
            "name": "test",
            "input": {},
        }
        assert events[1]["delta"] == {
            "type": "input_json_delta",
            "partial_json": "{}",
        }

    def test_interleaved_parallel_tools_flush_in_stable_explicit_index_order(self):
        state = AnthropicStreamState(message_start_sent=True)

        chunks = [
            _stream_chunk(
                tool_calls=[_tool_delta(index=7, tool_id="tool-a", name="alpha", arguments='{"a":')]
            ),
            _stream_chunk(
                tool_calls=[
                    _tool_delta(index=2, tool_id="tool-b", name="beta", arguments='{"b":2}')
                ]
            ),
            _stream_chunk(tool_calls=[_tool_delta(index=7, arguments="1}")]),
        ]

        before_terminal = [
            event
            for chunk in chunks
            for event in translate_openai_chunk_to_anthropic_events(chunk, state, "claude-3")
        ]
        terminal = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(finish_reason="tool_calls"), state, "claude-3"
        )

        assert before_terminal == []
        blocks = _tool_blocks(terminal)
        assert [(block["id"], block["json"]) for block in blocks] == [
            ("tool-b", '{"b":2}'),
            ("tool-a", '{"a":1}'),
        ]
        assert all(block["delta_index"] < block["stop_index"] for block in blocks)

    def test_indexless_calls_use_id_identity_and_arrival_order(self):
        state = AnthropicStreamState(message_start_sent=True)

        chunks = [
            _stream_chunk(
                tool_calls=[_tool_delta(tool_id="tool-a", name="alpha", arguments='{"a":')]
            ),
            _stream_chunk(
                tool_calls=[_tool_delta(tool_id="tool-b", name="beta", arguments='{"b":2}')]
            ),
            _stream_chunk(tool_calls=[_tool_delta(tool_id="tool-a", arguments="1}")]),
        ]
        for chunk in chunks:
            assert translate_openai_chunk_to_anthropic_events(chunk, state, "claude-3") == []

        events = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(finish_reason="tool_calls"), state, "claude-3"
        )

        assert [(block["id"], block["json"]) for block in _tool_blocks(events)] == [
            ("tool-a", '{"a":1}'),
            ("tool-b", '{"b":2}'),
        ]

    def test_tool_text_tool_keeps_text_live_and_flushes_tools_after_text(self):
        state = AnthropicStreamState(message_start_sent=True)

        assert (
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(
                    tool_calls=[
                        _tool_delta(index=1, tool_id="tool-a", name="alpha", arguments="{}")
                    ]
                ),
                state,
                "claude-3",
            )
            == []
        )
        text_events = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(content="still working"), state, "claude-3"
        )
        assert any(event.get("delta", {}).get("text") == "still working" for event in text_events)
        assert (
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(
                    tool_calls=[_tool_delta(index=0, tool_id="tool-b", name="beta", arguments="{}")]
                ),
                state,
                "claude-3",
            )
            == []
        )

        terminal = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(finish_reason="tool_calls"), state, "claude-3"
        )

        assert terminal[0] == {"type": "content_block_stop", "index": 0}
        blocks = _tool_blocks(terminal)
        assert [block["id"] for block in blocks] == ["tool-b", "tool-a"]
        assert [event["index"] for event in terminal if event["type"] == "content_block_start"] == [
            1,
            2,
        ]

    def test_tool_fragment_does_not_close_live_thinking_block(self):
        state = AnthropicStreamState(message_start_sent=True)
        first_thinking = translate_openai_chunk_to_anthropic_events(
            {
                "id": "chunk-thinking-1",
                "choices": [{"delta": {"reasoning_text": "first"}, "finish_reason": None}],
            },
            state,
            "claude-3",
        )
        assert [event["type"] for event in first_thinking] == [
            "content_block_start",
            "content_block_delta",
        ]

        tool_events = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(
                tool_calls=[_tool_delta(index=0, tool_id="tool-a", name="alpha", arguments="{}")]
            ),
            state,
            "claude-3",
        )
        second_thinking = translate_openai_chunk_to_anthropic_events(
            {
                "id": "chunk-thinking-2",
                "choices": [{"delta": {"reasoning_text": "second"}, "finish_reason": None}],
            },
            state,
            "claude-3",
        )

        assert tool_events == []
        assert second_thinking == [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "second"},
            }
        ]

        terminal = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(finish_reason="tool_calls"), state, "claude-3"
        )
        assert terminal[0] == {"type": "content_block_stop", "index": 0}
        tool_block = _tool_blocks(terminal)[0]
        assert tool_block["id"] == "tool-a"
        assert terminal[tool_block["start_index"]]["index"] == 1

    @pytest.mark.parametrize(
        ("tool_calls", "match"),
        [
            (
                [_tool_delta(index=0, tool_id="tool-a", name="alpha", arguments="{")],
                "valid JSON object",
            ),
            (
                [_tool_delta(index=0, tool_id="tool-a", name="alpha", arguments="[]")],
                "JSON object",
            ),
            (
                [_tool_delta(index=0, name="alpha", arguments="{}")],
                "missing id",
            ),
            (
                [_tool_delta(index=0, tool_id="tool-a", arguments="{}")],
                "missing name",
            ),
        ],
        ids=["truncated-json", "non-object-json", "missing-id", "missing-name"],
    )
    def test_terminal_validation_is_transactional(self, tool_calls, match):
        state = AnthropicStreamState(message_start_sent=True)
        assert (
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(tool_calls=tool_calls), state, "claude-3"
            )
            == []
        )

        with pytest.raises(AnthropicStreamProtocolError, match=match):
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(finish_reason="tool_calls"), state, "claude-3"
            )

        assert state.message_complete is False
        assert state.content_block_open is False

    def test_explicit_incomplete_terminal_still_validates_tool_arguments(self):
        state = AnthropicStreamState(message_start_sent=True)
        translate_openai_chunk_to_anthropic_events(
            _stream_chunk(
                tool_calls=[_tool_delta(index=0, tool_id="tool-a", name="alpha", arguments="{")]
            ),
            state,
            "claude-3",
        )

        with pytest.raises(AnthropicStreamProtocolError, match="valid JSON object"):
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(finish_reason="length"), state, "claude-3"
            )

    def test_conflicting_tool_identity_is_protocol_error(self):
        state = AnthropicStreamState(message_start_sent=True)
        translate_openai_chunk_to_anthropic_events(
            _stream_chunk(
                tool_calls=[_tool_delta(index=0, tool_id="tool-a", name="alpha", arguments="{")]
            ),
            state,
            "claude-3",
        )

        with pytest.raises(AnthropicStreamProtocolError, match="conflicting id"):
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(
                    tool_calls=[_tool_delta(index=0, tool_id="tool-b", name="alpha", arguments="}")]
                ),
                state,
                "claude-3",
            )

    def test_indexless_tool_is_upgraded_when_later_delta_adds_index(self):
        state = AnthropicStreamState(message_start_sent=True)
        translate_openai_chunk_to_anthropic_events(
            _stream_chunk(tool_calls=[_tool_delta(tool_id="tool-a", name="alpha", arguments="{")]),
            state,
            "claude-3",
        )
        translate_openai_chunk_to_anthropic_events(
            _stream_chunk(tool_calls=[_tool_delta(index=4, tool_id="tool-a", arguments='"a":1}')]),
            state,
            "claude-3",
        )

        events = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(finish_reason="tool_calls"), state, "claude-3"
        )

        assert [(block["id"], json.loads(block["json"])) for block in _tool_blocks(events)] == [
            ("tool-a", {"a": 1})
        ]

    def test_indexed_tool_is_completed_by_later_id_only_delta(self):
        state = AnthropicStreamState(message_start_sent=True)
        translate_openai_chunk_to_anthropic_events(
            _stream_chunk(
                tool_calls=[_tool_delta(index=4, tool_id="tool-a", name="alpha", arguments="{")]
            ),
            state,
            "claude-3",
        )
        translate_openai_chunk_to_anthropic_events(
            _stream_chunk(tool_calls=[_tool_delta(tool_id="tool-a", arguments='"a":1}')]),
            state,
            "claude-3",
        )

        events = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(finish_reason="tool_calls"), state, "claude-3"
        )

        assert [(block["id"], json.loads(block["json"])) for block in _tool_blocks(events)] == [
            ("tool-a", {"a": 1})
        ]

    def test_empty_object_split_across_multiple_argument_fragments(self):
        state = AnthropicStreamState(message_start_sent=True)
        for fragment in ("{", "", "}"):
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(
                    tool_calls=[
                        _tool_delta(
                            index=0,
                            tool_id="tool-a",
                            name="alpha",
                            arguments=fragment,
                        )
                    ]
                ),
                state,
                "claude-3",
            )

        events = translate_openai_chunk_to_anthropic_events(
            _stream_chunk(finish_reason="tool_calls"), state, "claude-3"
        )

        assert _tool_blocks(events)[0]["json"] == "{}"

    @pytest.mark.parametrize(
        "second_delta",
        [
            _tool_delta(index=1, tool_id="tool-a", name="alpha", arguments="}"),
            _tool_delta(index=0, tool_id="tool-b", name="alpha", arguments="}"),
            _tool_delta(index=0, tool_id="tool-a", name="beta", arguments="}"),
        ],
        ids=["id-matches-other-index", "index-matches-other-id", "name-conflict"],
    )
    def test_conflicting_index_id_or_name_is_protocol_error(self, second_delta):
        state = AnthropicStreamState(message_start_sent=True)
        translate_openai_chunk_to_anthropic_events(
            _stream_chunk(
                tool_calls=[_tool_delta(index=0, tool_id="tool-a", name="alpha", arguments="{")]
            ),
            state,
            "claude-3",
        )
        if second_delta["index"] == 1:
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(
                    tool_calls=[
                        _tool_delta(
                            index=1,
                            tool_id="tool-b",
                            name="beta",
                            arguments="{}",
                        )
                    ]
                ),
                state,
                "claude-3",
            )

        with pytest.raises(AnthropicStreamProtocolError, match="conflicting"):
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(tool_calls=[second_delta]), state, "claude-3"
            )

    def test_fragment_without_index_or_id_is_protocol_error(self):
        state = AnthropicStreamState(message_start_sent=True)

        with pytest.raises(AnthropicStreamProtocolError, match="missing both index and id"):
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(tool_calls=[_tool_delta(arguments='{"a":1}')]),
                state,
                "claude-3",
            )

    @pytest.mark.parametrize(
        "tool_call",
        [
            _tool_delta(index=True, tool_id="tool-a", name="alpha", arguments="{}"),
            {"index": 0, "id": "tool-a", "function": []},
            _tool_delta(index=0, tool_id="tool-a", name="alpha", arguments=None)
            | {"function": {"name": "alpha", "arguments": {}}},
        ],
        ids=["boolean-index", "non-object-function", "non-string-arguments"],
    )
    def test_malformed_tool_delta_shape_is_protocol_error(self, tool_call):
        state = AnthropicStreamState(message_start_sent=True)

        with pytest.raises(AnthropicStreamProtocolError):
            translate_openai_chunk_to_anthropic_events(
                _stream_chunk(tool_calls=[tool_call]), state, "claude-3"
            )

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


class TestAnthropicStreamState:
    """Tests for AnthropicStreamState."""

    def test_default_values(self):
        """Test default state values."""
        state = AnthropicStreamState()
        assert state.message_start_sent is False
        assert state.content_block_index == 0
        assert state.content_block_open is False
        assert state.tool_calls == []
        assert state.message_complete is False

    def test_estimated_input_tokens(self):
        """Test estimated input tokens."""
        state = AnthropicStreamState(estimated_input_tokens=1000)
        assert state.estimated_input_tokens == 1000
