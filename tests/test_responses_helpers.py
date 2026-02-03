"""Tests for server routes - responses API helpers."""

from router_maestro.server.routes.responses import (
    convert_content_to_serializable,
    convert_input_to_internal,
    convert_tool_choice_to_internal,
    convert_tools_to_internal,
    extract_text_from_content,
    generate_id,
    make_function_call_item,
    make_message_item,
    make_text_content,
    make_usage,
    sse_event,
)


class TestGenerateId:
    """Tests for ID generation."""

    def test_generate_id_with_prefix(self):
        """Test ID generation with prefix."""
        result = generate_id("req")
        assert result.startswith("req-")
        assert len(result) == 20  # "req-" + 16 hex chars

    def test_generate_id_unique(self):
        """Test that generated IDs are unique."""
        ids = {generate_id("test") for _ in range(100)}
        assert len(ids) == 100


class TestSseEvent:
    """Tests for SSE event formatting."""

    def test_sse_event_format(self):
        """Test SSE event format."""
        data = {"type": "test_event", "content": "Hello"}
        result = sse_event(data)
        assert result == 'event: test_event\ndata: {"type": "test_event", "content": "Hello"}\n\n'

    def test_sse_event_empty_type(self):
        """Test SSE event with no type."""
        data = {"content": "Hello"}
        result = sse_event(data)
        assert result.startswith("event: \ndata:")


class TestExtractTextFromContent:
    """Tests for text extraction from content."""

    def test_extract_from_string(self):
        """Test extracting text from string content."""
        result = extract_text_from_content("Hello world")
        assert result == "Hello world"

    def test_extract_from_list_with_input_text(self):
        """Test extracting text from list with input_text blocks."""
        content = [
            {"type": "input_text", "text": "First"},
            {"type": "input_text", "text": "Second"},
        ]
        result = extract_text_from_content(content)
        assert result == "FirstSecond"

    def test_extract_from_list_with_output_text(self):
        """Test extracting text from list with output_text blocks."""
        content = [{"type": "output_text", "text": "Output"}]
        result = extract_text_from_content(content)
        assert result == "Output"

    def test_extract_from_list_with_text_key(self):
        """Test extracting text from blocks with text key."""
        content = [{"text": "Plain text"}]
        result = extract_text_from_content(content)
        assert result == "Plain text"

    def test_extract_from_empty_list(self):
        """Test extracting text from empty list."""
        result = extract_text_from_content([])
        assert result == ""


class TestConvertContentToSerializable:
    """Tests for content serialization."""

    def test_string_passthrough(self):
        """Test that strings pass through unchanged."""
        result = convert_content_to_serializable("Hello")
        assert result == "Hello"

    def test_dict_passthrough(self):
        """Test that dicts pass through."""
        data = {"key": "value"}
        result = convert_content_to_serializable(data)
        assert result == {"key": "value"}

    def test_nested_list(self):
        """Test nested list conversion."""
        data = [{"a": 1}, {"b": 2}]
        result = convert_content_to_serializable(data)
        assert result == [{"a": 1}, {"b": 2}]


class TestConvertInputToInternal:
    """Tests for input conversion."""

    def test_string_input(self):
        """Test string input passes through."""
        result = convert_input_to_internal("Hello")
        assert result == "Hello"

    def test_message_input(self):
        """Test message input conversion."""
        input_data = [{"type": "message", "role": "user", "content": "Hello"}]
        result = convert_input_to_internal(input_data)
        assert result == [{"type": "message", "role": "user", "content": "Hello"}]

    def test_message_without_type(self):
        """Test message input without explicit type."""
        input_data = [{"role": "user", "content": "Hello"}]
        result = convert_input_to_internal(input_data)
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "user"

    def test_function_call_input(self):
        """Test function_call input conversion."""
        input_data = [
            {
                "type": "function_call",
                "id": "fc-123",
                "call_id": "call-456",
                "name": "get_weather",
                "arguments": '{"location": "NYC"}',
            }
        ]
        result = convert_input_to_internal(input_data)
        assert result[0]["type"] == "function_call"
        assert result[0]["name"] == "get_weather"

    def test_function_call_output_input(self):
        """Test function_call_output input conversion."""
        input_data = [{"type": "function_call_output", "call_id": "call-456", "output": "Sunny"}]
        result = convert_input_to_internal(input_data)
        assert result[0]["type"] == "function_call_output"
        assert result[0]["output"] == "Sunny"

    def test_function_call_output_dict_output(self):
        """Test function_call_output with dict output converts to JSON."""
        input_data = [
            {"type": "function_call_output", "call_id": "call-456", "output": {"temp": 72}}
        ]
        result = convert_input_to_internal(input_data)
        assert result[0]["output"] == '{"temp": 72}'


class TestConvertToolsToInternal:
    """Tests for tools conversion."""

    def test_none_tools(self):
        """Test None tools returns None."""
        result = convert_tools_to_internal(None)
        assert result is None

    def test_empty_tools(self):
        """Test empty tools list returns None."""
        result = convert_tools_to_internal([])
        assert result is None

    def test_dict_tools(self):
        """Test dict tools pass through."""
        tools = [{"type": "function", "function": {"name": "test"}}]
        result = convert_tools_to_internal(tools)
        assert result == tools


class TestConvertToolChoiceToInternal:
    """Tests for tool_choice conversion."""

    def test_none_tool_choice(self):
        """Test None tool_choice returns None."""
        result = convert_tool_choice_to_internal(None)
        assert result is None

    def test_string_tool_choice(self):
        """Test string tool_choice passes through."""
        result = convert_tool_choice_to_internal("auto")
        assert result == "auto"

    def test_dict_tool_choice(self):
        """Test dict tool_choice passes through."""
        choice = {"type": "function", "function": {"name": "test"}}
        result = convert_tool_choice_to_internal(choice)
        assert result == choice


class TestMakeTextContent:
    """Tests for text content creation."""

    def test_make_text_content(self):
        """Test creating text content block."""
        result = make_text_content("Hello world")
        assert result == {"type": "output_text", "text": "Hello world", "annotations": []}

    def test_make_text_content_empty(self):
        """Test creating empty text content block."""
        result = make_text_content("")
        assert result["text"] == ""


class TestMakeUsage:
    """Tests for usage creation."""

    def test_make_usage_none(self):
        """Test None usage returns None."""
        result = make_usage(None)
        assert result is None

    def test_make_usage_basic(self):
        """Test basic usage creation."""
        raw = {"input_tokens": 10, "output_tokens": 20}
        result = make_usage(raw)
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30
        assert result["input_tokens_details"]["cached_tokens"] == 0
        assert result["output_tokens_details"]["reasoning_tokens"] == 0


class TestMakeMessageItem:
    """Tests for message item creation."""

    def test_make_message_item(self):
        """Test creating message item."""
        result = make_message_item("msg-123", "Hello world")
        assert result["type"] == "message"
        assert result["id"] == "msg-123"
        assert result["role"] == "assistant"
        assert result["status"] == "completed"
        assert result["content"][0]["type"] == "output_text"
        assert result["content"][0]["text"] == "Hello world"

    def test_make_message_item_in_progress(self):
        """Test creating in-progress message item."""
        result = make_message_item("msg-123", "Hello", status="in_progress")
        assert result["status"] == "in_progress"


class TestMakeFunctionCallItem:
    """Tests for function call item creation."""

    def test_make_function_call_item(self):
        """Test creating function call item."""
        result = make_function_call_item(
            fc_id="fc-123",
            call_id="call-456",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )
        assert result["type"] == "function_call"
        assert result["id"] == "fc-123"
        assert result["call_id"] == "call-456"
        assert result["name"] == "get_weather"
        assert result["arguments"] == '{"location": "NYC"}'
        assert result["status"] == "completed"

    def test_make_function_call_item_in_progress(self):
        """Test creating in-progress function call item."""
        result = make_function_call_item(
            fc_id="fc-123",
            call_id="call-456",
            name="test",
            arguments="{}",
            status="in_progress",
        )
        assert result["status"] == "in_progress"
