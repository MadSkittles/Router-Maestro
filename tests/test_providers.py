"""Tests for providers module."""

import pytest

from router_maestro.providers import (
    AnthropicProvider,
    ChatRequest,
    CopilotProvider,
    Message,
    OpenAICompatibleProvider,
    OpenAIProvider,
)


class TestProviderBase:
    """Tests for provider base functionality."""

    def test_copilot_provider_init(self):
        """Test CopilotProvider initialization."""
        provider = CopilotProvider()
        assert provider.name == "github-copilot"
        # Note: is_authenticated() depends on whether GitHub Copilot credentials
        # are stored in the system. We only test the provider initializes correctly.
        assert isinstance(provider.is_authenticated(), bool)

    def test_openai_provider_init(self):
        """Test OpenAIProvider initialization."""
        provider = OpenAIProvider()
        assert provider.name == "openai"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_openai_provider_custom_url(self):
        """Test OpenAIProvider with custom URL."""
        provider = OpenAIProvider(base_url="https://custom.api.com/v1/")
        assert provider.base_url == "https://custom.api.com/v1"  # Trailing slash removed

    def test_anthropic_provider_init(self):
        """Test AnthropicProvider initialization."""
        provider = AnthropicProvider()
        assert provider.name == "anthropic"
        assert provider.base_url == "https://api.anthropic.com/v1"

    def test_openai_compatible_provider_init(self):
        """Test OpenAICompatibleProvider initialization."""
        provider = OpenAICompatibleProvider(
            name="custom",
            base_url="https://example.com/v1",
            api_key="test-key",
            models={"model-1": "Model One"},
        )
        assert provider.name == "custom"
        assert provider.is_authenticated() is True


class TestChatRequest:
    """Tests for ChatRequest."""

    def test_basic_request(self):
        """Test basic chat request creation."""
        request = ChatRequest(
            model="gpt-4o",
            messages=[
                Message(role="user", content="Hello"),
            ],
        )
        assert request.model == "gpt-4o"
        assert len(request.messages) == 1
        assert request.temperature == 1.0
        assert request.stream is False

    def test_request_with_options(self):
        """Test chat request with options."""
        request = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Test")],
            temperature=0.5,
            max_tokens=100,
            stream=True,
        )
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert request.stream is True


class TestAnthropicMessageConversion:
    """Tests for Anthropic message format conversion."""

    def test_system_message_extraction(self):
        """Test that system messages are extracted correctly."""
        provider = AnthropicProvider()

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]

        system, converted = provider._convert_messages(messages)

        assert system == "You are helpful"
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_no_system_message(self):
        """Test conversion without system message."""
        provider = AnthropicProvider()

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]

        system, converted = provider._convert_messages(messages)

        assert system is None
        assert len(converted) == 2
