"""Tests for providers module."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from router_maestro.auth.github_oauth import CopilotTokenResponse
from router_maestro.auth.storage import AuthStorage, OAuthCredential
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

    def test_copilot_api_base_defaults_to_public_endpoint(self):
        """Copilot calls default to the public API endpoint."""
        provider = CopilotProvider()

        assert provider._api_base == "https://api.githubcopilot.com"
        assert provider._url("/chat/completions") == (
            "https://api.githubcopilot.com/chat/completions"
        )

    def test_copilot_api_base_uses_token_endpoint_metadata(self):
        """Copilot calls use the API base returned by the token endpoint."""
        provider = CopilotProvider()
        provider._api_base = "https://api.enterprise.githubcopilot.com/"

        assert provider._url("/models") == "https://api.enterprise.githubcopilot.com/models"

    @pytest.mark.asyncio
    async def test_copilot_api_base_uses_persisted_endpoint_metadata(self):
        """Token refresh should keep using a previously persisted API endpoint."""
        provider = CopilotProvider()
        provider.auth_manager.storage = AuthStorage()
        provider.auth_manager.storage.set(
            "github-copilot",
            OAuthCredential(
                refresh="github-token",
                access="old-copilot-token",
                expires=0,
                api_endpoint="https://api.enterprise.githubcopilot.com",
            ),
        )
        provider.auth_manager.save = Mock()  # type: ignore[method-assign]

        with patch(
            "router_maestro.providers.copilot.get_copilot_token",
            new=AsyncMock(
                return_value=CopilotTokenResponse(
                    token="new-copilot-token",
                    expires_at=1234567890,
                    refresh_in=1000,
                    api_endpoint=None,
                )
            ),
        ):
            await provider.ensure_token()

        assert provider._api_base == "https://api.enterprise.githubcopilot.com"

    def test_copilot_headers_include_standard_metadata(self):
        """Copilot requests carry the same compatibility headers as reference clients."""
        provider = CopilotProvider()
        provider._cached_token = "token"

        headers = provider._get_headers()

        assert headers["Authorization"] == "Bearer token"
        assert headers["Copilot-Integration-Id"] == "vscode-chat"
        assert headers["User-Agent"] == "GitHubCopilotChat/0.26.7"
        assert headers["OpenAI-Intent"] == "conversation-panel"
        assert headers["X-GitHub-Api-Version"] == "2025-04-01"
        assert headers["X-Vscode-User-Agent-Library-Version"] == "electron-fetch"
        assert "X-Request-Id" in headers

    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            ([Message(role="user", content="hi")], "user"),
            ([Message(role="assistant", content="previous")], "agent"),
            ([Message(role="tool", content="result", tool_call_id="call_1")], "agent"),
        ],
    )
    def test_copilot_headers_include_initiator_for_chat(self, messages, expected):
        """Chat calls mark whether the request continues an agent/tool turn."""
        provider = CopilotProvider()
        provider._cached_token = "token"

        headers = provider._get_headers(messages=messages)

        assert headers["X-Initiator"] == expected

    def test_copilot_response_headers_include_initiator_for_input_items(self):
        """Responses calls mark role-less items such as function_call as agent turns."""
        provider = CopilotProvider()
        provider._cached_token = "token"

        headers = provider._get_headers(
            response_input=[
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "function_call", "call_id": "call_1", "name": "lookup"},
            ]
        )

        assert headers["X-Initiator"] == "agent"

    def test_copilot_response_vision_detection_is_recursive(self):
        """Responses input_image blocks require the Copilot vision header."""
        provider = CopilotProvider()

        assert provider._responses_input_has_vision(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": "https://example/image.png"},
                    ],
                }
            ]
        )

    @pytest.mark.asyncio
    async def test_copilot_models_skip_completion_only_catalog_entries(self):
        """Completion-only Copilot catalog models should not be exposed as chat models."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "gpt-41-copilot",
                            "name": "GPT-4.1 Copilot",
                            "model_picker_enabled": True,
                            "capabilities": {"type": "completion", "supports": {}},
                        },
                        {
                            "id": "gpt-4o",
                            "name": "GPT-4o",
                            "model_picker_enabled": True,
                            "capabilities": {"type": "chat", "supports": {}},
                        },
                    ]
                },
                request=httpx.Request("GET", "https://api.githubcopilot.com/models"),
            )

        provider = CopilotProvider()
        provider._cached_token = "token"
        provider.ensure_token = AsyncMock()  # type: ignore[method-assign]

        with patch(
            "httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ):
            models = await provider.list_models(force_refresh=True)

        assert [model.id for model in models] == ["gpt-4o"]

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

    def test_assistant_tool_calls_convert_to_tool_use_blocks(self):
        """Assistant tool calls must be sent in Anthropic content block format."""
        provider = AnthropicProvider()

        messages = [
            Message(role="user", content="Check the weather"),
            Message(
                role="assistant",
                content="I'll check.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Shanghai"}',
                        },
                    }
                ],
            ),
        ]

        system, converted = provider._convert_messages(messages)

        assert system is None
        assert converted[1] == {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check."},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"location": "Shanghai"},
                },
            ],
        }

    def test_tool_messages_convert_to_user_tool_result_blocks(self):
        """Internal tool-role messages must become Anthropic user tool_result blocks."""
        provider = AnthropicProvider()

        messages = [
            Message(role="tool", content='{"temperature":22}', tool_call_id="call_1"),
        ]

        _system, converted = provider._convert_messages(messages)

        assert converted == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": '{"temperature":22}',
                    }
                ],
            }
        ]
