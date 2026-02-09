"""Tests for provider-aware token counting configuration."""

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from router_maestro.server.schemas.anthropic import (
    AnthropicTool,
    AnthropicUserMessage,
)
from router_maestro.utils.token_config import (
    ANTHROPIC_CONFIG,
    COPILOT_CONFIG,
    DEFAULT_CONFIG,
    OPENAI_CONFIG,
    TokenCountingConfig,
    count_tokens_via_anthropic_api,
    get_config_for_provider,
)
from router_maestro.utils.tokens import count_anthropic_request_tokens


class TestTokenCountingConfig:
    """Tests for TokenCountingConfig dataclass."""

    def test_default_values(self):
        config = TokenCountingConfig()
        assert config.tokens_per_message == 3
        assert config.tokens_per_name == 1
        assert config.tokens_per_completion == 3
        assert config.base_tool_tokens == 16
        assert config.tokens_per_tool == 8
        assert config.tool_definition_multiplier == 1.1
        assert config.tool_calls_multiplier == 1.5

    def test_immutability(self):
        config = TokenCountingConfig()
        with pytest.raises(FrozenInstanceError):
            config.tokens_per_message = 99  # type: ignore[misc]

    def test_custom_values(self):
        config = TokenCountingConfig(
            tokens_per_message=5,
            base_tool_tokens=0,
            tool_definition_multiplier=1.0,
            tool_calls_multiplier=1.0,
        )
        assert config.tokens_per_message == 5
        assert config.base_tool_tokens == 0
        assert config.tool_definition_multiplier == 1.0
        assert config.tool_calls_multiplier == 1.0

    def test_equality(self):
        a = TokenCountingConfig()
        b = TokenCountingConfig()
        assert a == b

    def test_inequality(self):
        a = COPILOT_CONFIG
        b = ANTHROPIC_CONFIG
        assert a != b


class TestPrebuiltConfigs:
    """Tests for pre-built provider configs."""

    def test_copilot_config_has_inflation(self):
        assert COPILOT_CONFIG.tool_definition_multiplier == 1.1
        assert COPILOT_CONFIG.tool_calls_multiplier == 1.5
        assert COPILOT_CONFIG.base_tool_tokens == 16

    def test_anthropic_config_no_inflation(self):
        assert ANTHROPIC_CONFIG.tool_definition_multiplier == 1.0
        assert ANTHROPIC_CONFIG.tool_calls_multiplier == 1.0
        assert ANTHROPIC_CONFIG.base_tool_tokens == 0

    def test_openai_config_no_inflation(self):
        assert OPENAI_CONFIG.tool_definition_multiplier == 1.0
        assert OPENAI_CONFIG.tool_calls_multiplier == 1.0
        assert OPENAI_CONFIG.base_tool_tokens == 8

    def test_default_is_copilot(self):
        assert DEFAULT_CONFIG is COPILOT_CONFIG


class TestGetConfigForProvider:
    """Tests for get_config_for_provider() mapping."""

    def test_github_copilot(self):
        assert get_config_for_provider("github-copilot") is COPILOT_CONFIG

    def test_anthropic(self):
        assert get_config_for_provider("anthropic") is ANTHROPIC_CONFIG

    def test_openai(self):
        assert get_config_for_provider("openai") is OPENAI_CONFIG

    def test_none_returns_default(self):
        assert get_config_for_provider(None) is DEFAULT_CONFIG

    def test_unknown_provider_returns_default(self):
        assert get_config_for_provider("some-custom-provider") is DEFAULT_CONFIG

    def test_empty_string_returns_default(self):
        assert get_config_for_provider("") is DEFAULT_CONFIG


class TestConfigNoneEquivalence:
    """Verify that config=None produces the same results as config=COPILOT_CONFIG."""

    def test_simple_message(self):
        messages = [AnthropicUserMessage(content="Hello, how are you?")]
        result_none = count_anthropic_request_tokens(system=None, messages=messages, config=None)
        result_copilot = count_anthropic_request_tokens(
            system=None, messages=messages, config=COPILOT_CONFIG
        )
        assert result_none == result_copilot

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
        result_none = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools, config=None
        )
        result_copilot = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools, config=COPILOT_CONFIG
        )
        assert result_none == result_copilot

    def test_with_system(self):
        messages = [AnthropicUserMessage(content="Hi")]
        result_none = count_anthropic_request_tokens(
            system="You are helpful.", messages=messages, config=None
        )
        result_copilot = count_anthropic_request_tokens(
            system="You are helpful.", messages=messages, config=COPILOT_CONFIG
        )
        assert result_none == result_copilot


class TestAnthropicConfigLowerCounts:
    """Verify that ANTHROPIC_CONFIG produces lower counts due to no inflation."""

    def test_with_tools_anthropic_lower(self):
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
        copilot = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools, config=COPILOT_CONFIG
        )
        anthropic = count_anthropic_request_tokens(
            system=None, messages=messages, tools=tools, config=ANTHROPIC_CONFIG
        )
        # Anthropic config has no inflation multipliers -> lower count
        assert anthropic < copilot

    def test_with_tool_use_block_anthropic_lower(self):
        messages = [
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
        copilot = count_anthropic_request_tokens(
            system=None, messages=messages, config=COPILOT_CONFIG
        )
        anthropic = count_anthropic_request_tokens(
            system=None, messages=messages, config=ANTHROPIC_CONFIG
        )
        # tool_calls_multiplier=1.5 vs 1.0 -> copilot should be higher
        assert anthropic < copilot

    def test_simple_text_no_difference(self):
        """Without tools or tool_calls, configs should produce the same count."""
        messages = [AnthropicUserMessage(content="Hello, how are you?")]
        copilot = count_anthropic_request_tokens(
            system=None, messages=messages, config=COPILOT_CONFIG
        )
        anthropic = count_anthropic_request_tokens(
            system=None, messages=messages, config=ANTHROPIC_CONFIG
        )
        # tokens_per_message/name/completion are the same -> equal
        assert copilot == anthropic


class TestCountTokensViaAnthropicApi:
    """Tests for count_tokens_via_anthropic_api() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_basic_call(self):
        mock_response = httpx.Response(
            200,
            json={"input_tokens": 42},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens"),
        )

        with patch("router_maestro.utils.token_config.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await count_tokens_via_anthropic_api(
                base_url="https://api.anthropic.com/v1",
                api_key="test-key",
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result == 42
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "x-api-key" in call_kwargs.kwargs["headers"]
        assert call_kwargs.kwargs["headers"]["x-api-key"] == "test-key"

    @pytest.mark.asyncio
    async def test_with_system_and_tools(self):
        mock_response = httpx.Response(
            200,
            json={"input_tokens": 100},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens"),
        )

        with patch("router_maestro.utils.token_config.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await count_tokens_via_anthropic_api(
                base_url="https://api.anthropic.com/v1",
                api_key="test-key",
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
                system="You are helpful.",
                tools=[{"name": "test", "description": "A test tool", "input_schema": {}}],
            )

        assert result == 100
        payload = mock_client.post.call_args.kwargs["json"]
        assert "system" in payload
        assert "tools" in payload

    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        mock_response = httpx.Response(
            400,
            json={"error": "bad request"},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens"),
        )

        with patch("router_maestro.utils.token_config.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await count_tokens_via_anthropic_api(
                    base_url="https://api.anthropic.com/v1",
                    api_key="test-key",
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "Hello"}],
                )

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped(self):
        mock_response = httpx.Response(
            200,
            json={"input_tokens": 10},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens"),
        )

        with patch("router_maestro.utils.token_config.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await count_tokens_via_anthropic_api(
                base_url="https://api.anthropic.com/v1/",
                api_key="test-key",
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hello"}],
            )

        url_arg = mock_client.post.call_args.args[0]
        assert url_arg == "https://api.anthropic.com/v1/messages/count_tokens"
