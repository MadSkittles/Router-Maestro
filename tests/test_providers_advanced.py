"""Tests for provider base classes and error handling."""

import pytest

from router_maestro.providers import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ModelInfo,
    ProviderError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
)
from router_maestro.providers.base import (
    BaseProvider,
    ResponsesToolCall,
)


class TestProviderError:
    """Tests for ProviderError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ProviderError("Test error")
        assert str(error) == "Test error"
        assert error.status_code == 500
        assert error.retryable is False

    def test_error_with_status_code(self):
        """Test error with status code."""
        error = ProviderError("Not found", status_code=404)
        assert error.status_code == 404

    def test_retryable_error(self):
        """Test retryable error."""
        error = ProviderError("Rate limited", status_code=429, retryable=True)
        assert error.retryable is True
        assert error.status_code == 429


class TestModelInfo:
    """Tests for ModelInfo."""

    def test_basic_model_info(self):
        """Test basic model info creation."""
        info = ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")
        assert info.id == "gpt-4o"
        assert info.name == "GPT-4o"
        assert info.provider == "openai"

    def test_model_info_equality(self):
        """Test model info equality."""
        info1 = ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")
        info2 = ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")
        assert info1 == info2


class TestMessage:
    """Tests for Message."""

    def test_basic_message(self):
        """Test basic message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_call_id is None
        assert msg.tool_calls is None

    def test_message_with_tool_call_id(self):
        """Test message with tool call ID."""
        msg = Message(role="tool", content="Result", tool_call_id="tc-123")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc-123"

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_calls = [{"id": "tc-1", "type": "function", "function": {"name": "test"}}]
        msg = Message(role="assistant", content="", tool_calls=tool_calls)
        assert msg.tool_calls == tool_calls

    def test_message_with_multimodal_content(self):
        """Test message with multimodal content."""
        content = [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        msg = Message(role="user", content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


class TestChatRequest:
    """Tests for ChatRequest."""

    def test_default_values(self):
        """Test default values."""
        request = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hi")],
        )
        assert request.temperature == 1.0
        assert request.max_tokens is None
        assert request.stream is False
        assert request.tools is None
        assert request.tool_choice is None
        assert request.extra == {}

    def test_with_all_options(self):
        """Test with all options."""
        request = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hi")],
            temperature=0.7,
            max_tokens=1000,
            stream=True,
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
            extra={"top_p": 0.9},
        )
        assert request.temperature == 0.7
        assert request.max_tokens == 1000
        assert request.stream is True
        assert request.tools is not None
        assert request.tool_choice == "auto"
        assert request.extra["top_p"] == 0.9


class TestChatResponse:
    """Tests for ChatResponse."""

    def test_basic_response(self):
        """Test basic response."""
        response = ChatResponse(
            content="Hello!",
            model="gpt-4o",
            finish_reason="stop",
        )
        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.finish_reason == "stop"
        assert response.usage is None

    def test_response_with_usage(self):
        """Test response with usage."""
        response = ChatResponse(
            content="Hello!",
            model="gpt-4o",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert response.usage["total_tokens"] == 15


class TestChatStreamChunk:
    """Tests for ChatStreamChunk."""

    def test_content_chunk(self):
        """Test content chunk."""
        chunk = ChatStreamChunk(content="Hello", finish_reason=None)
        assert chunk.content == "Hello"
        assert chunk.finish_reason is None

    def test_finish_chunk(self):
        """Test finish chunk."""
        chunk = ChatStreamChunk(content="", finish_reason="stop")
        assert chunk.finish_reason == "stop"

    def test_chunk_with_tool_calls(self):
        """Test chunk with tool calls."""
        chunk = ChatStreamChunk(
            content="",
            finish_reason=None,
            tool_calls=[{"index": 0, "id": "tc-1", "function": {"name": "test"}}],
        )
        assert chunk.tool_calls is not None


class TestResponsesRequest:
    """Tests for ResponsesRequest."""

    def test_string_input(self):
        """Test with string input."""
        request = ResponsesRequest(model="gpt-4o", input="Hello")
        assert request.input == "Hello"
        assert request.stream is False
        assert request.temperature == 1.0

    def test_list_input(self):
        """Test with list input."""
        request = ResponsesRequest(
            model="gpt-4o",
            input=[{"type": "message", "role": "user", "content": "Hi"}],
        )
        assert isinstance(request.input, list)

    def test_with_tools(self):
        """Test with tools."""
        request = ResponsesRequest(
            model="gpt-4o",
            input="Hello",
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        assert request.tools is not None
        assert request.tool_choice == "auto"
        assert request.parallel_tool_calls is True


class TestResponsesResponse:
    """Tests for ResponsesResponse."""

    def test_basic_response(self):
        """Test basic response."""
        response = ResponsesResponse(content="Hello!", model="gpt-4o")
        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.usage is None
        assert response.tool_calls is None

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        response = ResponsesResponse(
            content="",
            model="gpt-4o",
            tool_calls=[ResponsesToolCall(call_id="tc-1", name="test", arguments="{}")],
        )
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1


class TestResponsesStreamChunk:
    """Tests for ResponsesStreamChunk."""

    def test_content_chunk(self):
        """Test content chunk."""
        chunk = ResponsesStreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.finish_reason is None

    def test_tool_call_chunk(self):
        """Test chunk with tool call."""
        tool_call = ResponsesToolCall(call_id="tc-1", name="test", arguments="{}")
        chunk = ResponsesStreamChunk(content="", tool_call=tool_call)
        assert chunk.tool_call is not None
        assert chunk.tool_call.call_id == "tc-1"

    def test_tool_call_delta_chunk(self):
        """Test chunk with tool call delta."""
        chunk = ResponsesStreamChunk(
            content="",
            tool_call_delta={
                "type": "function_call_arguments_delta",
                "item_id": "item-1",
                "call_id": "tc-1",
                "delta": '{"loc":',
            },
        )
        assert chunk.tool_call_delta is not None


class TestResponsesToolCall:
    """Tests for ResponsesToolCall."""

    def test_basic_tool_call(self):
        """Test basic tool call."""
        tc = ResponsesToolCall(call_id="tc-1", name="get_weather", arguments='{"loc": "NYC"}')
        assert tc.call_id == "tc-1"
        assert tc.name == "get_weather"
        assert tc.arguments == '{"loc": "NYC"}'


class TestBaseProviderAbstract:
    """Tests for BaseProvider abstract methods."""

    def test_cannot_instantiate_base_provider(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider()

    @pytest.mark.asyncio
    async def test_base_provider_default_implementations(self):
        """Test default implementations in BaseProvider."""

        class MinimalProvider(BaseProvider):
            name = "minimal"

            def is_authenticated(self):
                return True

            async def chat_completion(self, request):
                pass

            async def chat_completion_stream(self, request):
                yield None

            async def list_models(self):
                return []

        provider = MinimalProvider()

        # Test default ensure_token
        await provider.ensure_token()

        # Test default responses methods raise NotImplementedError
        with pytest.raises(NotImplementedError):
            await provider.responses_completion(ResponsesRequest(model="test", input="hi"))
