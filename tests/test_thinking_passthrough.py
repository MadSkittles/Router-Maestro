"""Tests for thinking configuration passthrough."""

from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from router_maestro.providers.base import ChatRequest, Message
from router_maestro.routing.router import Router
from router_maestro.server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicThinkingConfig,
    AnthropicUserMessage,
)
from router_maestro.server.translation import translate_anthropic_to_openai


class TestTranslateThinkingConfig:
    """Tests for thinking config extraction in translation."""

    def test_translate_extracts_thinking_config(self):
        """Input with thinking={type="enabled", budget_tokens=16000} extracts correctly."""
        request = AnthropicMessagesRequest(
            model="claude-opus-4.6",
            max_tokens=4096,
            messages=[AnthropicUserMessage(role="user", content="Hello")],
            thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=16000),
        )
        result = translate_anthropic_to_openai(request)

        assert result.thinking_type == "enabled"
        assert result.thinking_budget == 16000

    def test_translate_no_thinking(self):
        """Input with thinking=None results in None fields."""
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4",
            max_tokens=1024,
            messages=[AnthropicUserMessage(role="user", content="Hello")],
        )
        result = translate_anthropic_to_openai(request)

        assert result.thinking_type is None
        assert result.thinking_budget is None

    def test_translate_adaptive_thinking(self):
        """Input with adaptive thinking type is preserved."""
        request = AnthropicMessagesRequest(
            model="claude-opus-4.6",
            max_tokens=4096,
            messages=[AnthropicUserMessage(role="user", content="Hello")],
            thinking=AnthropicThinkingConfig(type="adaptive", budget_tokens=8000),
        )
        result = translate_anthropic_to_openai(request)

        assert result.thinking_type == "adaptive"
        assert result.thinking_budget == 8000

    def test_translate_disabled_thinking(self):
        """Input with disabled thinking preserves the type."""
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4",
            max_tokens=1024,
            messages=[AnthropicUserMessage(role="user", content="Hello")],
            thinking=AnthropicThinkingConfig(type="disabled"),
        )
        result = translate_anthropic_to_openai(request)

        assert result.thinking_type == "disabled"
        assert result.thinking_budget is None

    def test_translate_thinking_without_budget(self):
        """Thinking enabled without explicit budget_tokens."""
        request = AnthropicMessagesRequest(
            model="claude-opus-4.6",
            max_tokens=4096,
            messages=[AnthropicUserMessage(role="user", content="Hello")],
            thinking=AnthropicThinkingConfig(type="enabled"),
        )
        result = translate_anthropic_to_openai(request)

        assert result.thinking_type == "enabled"
        assert result.thinking_budget is None

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
    def test_translate_output_config_effort(self, effort):
        """Top-level Anthropic effort is preserved as internal reasoning effort."""
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "claude-opus-4.8",
                "max_tokens": 64000,
                "messages": [{"role": "user", "content": "Hello"}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }
        )

        result = translate_anthropic_to_openai(request)

        assert request.output_config is not None
        assert request.output_config.effort == effort
        assert result.thinking_type == "adaptive"
        assert result.thinking_budget is None
        assert result.reasoning_effort == effort

    @pytest.mark.parametrize("effort", ["ultra", "none"])
    def test_output_config_rejects_invalid_effort(self, effort):
        with pytest.raises(ValidationError):
            AnthropicMessagesRequest.model_validate(
                {
                    "model": "claude-opus-4.8",
                    "max_tokens": 64000,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "output_config": {"effort": effort},
                }
            )

    def test_anthropic_options_translate_to_typed_fields(self):
        request = AnthropicMessagesRequest(
            model="claude-opus-4.8",
            max_tokens=4096,
            messages=[AnthropicUserMessage(role="user", content="Hello")],
            temperature=0.4,
            top_p=0.8,
            top_k=32,
            stop_sequences=["END"],
            metadata={"user_id": "user-123"},
            service_tier="standard_only",
        )

        result = translate_anthropic_to_openai(request)

        assert result.temperature == 0.4
        assert result.top_p == 0.8
        assert result.top_k == 32
        assert result.stop_sequences == ["END"]
        assert result.metadata == {"user_id": "user-123"}
        assert result.service_tier == "standard_only"

    def test_anthropic_responses_opt_in_clone_preserves_typed_options(self):
        request = AnthropicMessagesRequest(
            model="claude-opus-4.8",
            max_tokens=4096,
            messages=[AnthropicUserMessage(role="user", content="Hello")],
            top_p=0.8,
            top_k=32,
            stop_sequences=["END"],
            metadata={"user_id": "user-123"},
            service_tier="standard_only",
        )
        translated = translate_anthropic_to_openai(request)

        opted_in = replace(translated, use_responses_api=True, extra={})

        assert opted_in.use_responses_api is True
        assert opted_in.top_p == 0.8
        assert opted_in.top_k == 32
        assert opted_in.stop_sequences == ["END"]
        assert opted_in.metadata == {"user_id": "user-123"}
        assert opted_in.service_tier == "standard_only"


class TestCopilotPayloadThinking:
    """Tests for thinking_budget in Copilot payload construction."""

    @pytest.mark.asyncio
    async def test_copilot_payload_includes_thinking_budget(self):
        """Verify Claude payload contains reasoning_effort when client requests thinking."""
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        provider._cached_token = "test-token"
        provider._token_expires = 9999999999

        request = ChatRequest(
            model="claude-opus-4.7",
            messages=[Message(role="user", content="Hello")],
            thinking_budget=16000,
        )

        captured_payload = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured_payload.update(json)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = lambda: None
            mock_response.json = lambda: {
                "choices": [
                    {
                        "message": {"content": "test response"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "claude-opus-4.7",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.is_closed = False
        provider._client = mock_client

        with patch.object(provider, "ensure_token", new_callable=AsyncMock):
            await provider.chat_completion(request)

        assert "thinking_budget" not in captured_payload
        # opus-4.7 now accepts high (Copilot opened the upper tiers via the
        # model catalog). budget=16000 → desired "high" → passed through.
        assert captured_payload.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_copilot_payload_omits_thinking_when_none(self):
        """Verify payload has no thinking_budget key when None."""
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        provider._cached_token = "test-token"
        provider._token_expires = 9999999999

        request = ChatRequest(
            model="claude-sonnet-4",
            messages=[Message(role="user", content="Hello")],
        )

        captured_payload = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured_payload.update(json)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = lambda: None
            mock_response.json = lambda: {
                "choices": [
                    {
                        "message": {"content": "test response"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "claude-sonnet-4",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.is_closed = False
        provider._client = mock_client

        with patch.object(provider, "ensure_token", new_callable=AsyncMock):
            await provider.chat_completion(request)

        assert "thinking_budget" not in captured_payload


class TestCopilotNonstreamingTools:
    """Tests for non-streaming Copilot path including tools."""

    @pytest.mark.asyncio
    async def test_copilot_nonstreaming_includes_tools(self):
        """Verify non-streaming payload contains tools and tool_choice."""
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        provider._cached_token = "test-token"
        provider._token_expires = 9999999999

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object"},
                },
            }
        ]
        request = ChatRequest(
            model="claude-sonnet-4",
            messages=[Message(role="user", content="Use the test tool")],
            tools=tools,
            tool_choice="auto",
        )

        captured_payload = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured_payload.update(json)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = lambda: None
            mock_response.json = lambda: {
                "choices": [
                    {
                        "message": {"content": "calling tool"},
                        "finish_reason": "tool_calls",
                    }
                ],
                "model": "claude-sonnet-4",
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.is_closed = False
        provider._client = mock_client

        with patch.object(provider, "ensure_token", new_callable=AsyncMock):
            await provider.chat_completion(request)

        assert "tools" in captured_payload
        assert captured_payload["tools"] == tools
        assert "tool_choice" in captured_payload
        assert captured_payload["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_copilot_tool_calls_force_tool_calls_finish_reason(self):
        """Copilot can return tool_calls with finish_reason=stop; normalize it."""
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        provider._cached_token = "test-token"
        provider._token_expires = 9999999999

        request = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Use the test tool")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "test_tool"}},
        )

        async def mock_post(url, json=None, headers=None, timeout=None):
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = lambda: None
            mock_response.json = lambda: {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "test_tool",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "gpt-4o",
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.is_closed = False
        provider._client = mock_client

        with patch.object(provider, "ensure_token", new_callable=AsyncMock):
            response = await provider.chat_completion(request)

        assert response.tool_calls is not None
        assert response.finish_reason == "tool_calls"


class TestRouterPreservesThinkingFields:
    """Tests for thinking fields preserved through router model-swap."""

    def test_router_preserves_thinking_fields(self):
        """_create_request_with_model preserves thinking_budget and thinking_type."""
        router = Router.__new__(Router)

        original = ChatRequest(
            model="router-maestro",
            messages=[Message(role="user", content="Hello")],
            thinking_budget=16000,
            thinking_type="enabled",
            reasoning_effort="high",
        )

        result = router._create_request_with_model(original, "claude-opus-4.6")

        assert result.model == "claude-opus-4.6"
        assert result.thinking_budget == 16000
        assert result.thinking_type == "enabled"
        assert result.reasoning_effort == "high"

    def test_router_preserves_none_thinking_fields(self):
        """_create_request_with_model handles None thinking fields."""
        router = Router.__new__(Router)

        original = ChatRequest(
            model="claude-sonnet-4",
            messages=[Message(role="user", content="Hello")],
        )

        result = router._create_request_with_model(original, "claude-sonnet-4")

        assert result.thinking_budget is None
        assert result.thinking_type is None


class TestAnthropicProviderThinking:
    """Tests for thinking config forwarding in AnthropicProvider."""

    @pytest.mark.asyncio
    async def test_anthropic_provider_forwards_thinking(self):
        """Verify Anthropic provider includes thinking dict in payload."""
        from router_maestro.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()

        request = ChatRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message(role="user", content="Hello")],
            max_tokens=64000,
            thinking_type="enabled",
            thinking_budget=16000,
        )

        captured_payload = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured_payload.update(json)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = lambda: None
            mock_response.json = lambda: {
                "content": [{"type": "text", "text": "response"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return mock_response

        with patch("router_maestro.providers.anthropic.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with patch.object(provider, "_get_api_key", return_value="test-key"):
                await provider.chat_completion(request)

        assert "thinking" in captured_payload
        assert captured_payload["thinking"]["type"] == "enabled"
        assert captured_payload["thinking"]["budget_tokens"] == 16000
        assert "output_config" not in captured_payload

    def test_anthropic_provider_prefers_effort_over_budget(self):
        """Explicit effort uses output_config and suppresses a conflicting budget."""
        from router_maestro.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        request = ChatRequest(
            model="claude-opus-4.8",
            messages=[Message(role="user", content="Hello")],
            max_tokens=64000,
            thinking_type="adaptive",
            thinking_budget=16000,
            reasoning_effort="xhigh",
        )

        payload = provider._build_payload(request)

        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {"effort": "xhigh"}

    def test_anthropic_provider_omits_adaptive_budget_without_effort(self):
        """Adaptive thinking is a type-only wire union even without effort."""
        from router_maestro.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        request = ChatRequest(
            model="claude-opus-4.8",
            messages=[Message(role="user", content="Hello")],
            max_tokens=64000,
            thinking_type="adaptive",
            thinking_budget=16000,
        )

        payload = provider._build_payload(request)

        assert payload["thinking"] == {"type": "adaptive"}

    def test_anthropic_provider_preserves_enabled_budget_with_effort(self):
        """Manual thinking remains a valid enabled union when effort is present."""
        from router_maestro.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        request = ChatRequest(
            model="claude-opus-4.6",
            messages=[Message(role="user", content="Hello")],
            max_tokens=64000,
            thinking_type="enabled",
            thinking_budget=16000,
            reasoning_effort="xhigh",
        )

        payload = provider._build_payload(request)

        assert payload["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert payload["output_config"] == {"effort": "xhigh"}

    def test_anthropic_provider_omits_enabled_thinking_without_budget(self):
        """Never emit the enabled union without its required budget."""
        from router_maestro.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        request = ChatRequest(
            model="claude-opus-4.6",
            messages=[Message(role="user", content="Hello")],
            max_tokens=1024,
            thinking_type="enabled",
            reasoning_effort="xhigh",
        )

        payload = provider._build_payload(request)

        assert "thinking" not in payload
        assert payload["output_config"] == {"effort": "xhigh"}

    def test_anthropic_provider_caps_enabled_budget_to_payload_max_tokens(self):
        """Provider normalization protects non-Anthropic entry routes too."""
        from router_maestro.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        request = ChatRequest(
            model="claude-opus-4.6",
            messages=[Message(role="user", content="Hello")],
            thinking_type="enabled",
            thinking_budget=8192,
            reasoning_effort="high",
        )

        payload = provider._build_payload(request)

        assert payload["max_tokens"] == 4096
        assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4095}
        assert payload["output_config"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_anthropic_provider_omits_disabled_thinking(self):
        """Verify disabled thinking is not forwarded in payload."""
        from router_maestro.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()

        request = ChatRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message(role="user", content="Hello")],
            max_tokens=4096,
            thinking_type="disabled",
        )

        captured_payload = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured_payload.update(json)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = lambda: None
            mock_response.json = lambda: {
                "content": [{"type": "text", "text": "response"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return mock_response

        with patch("router_maestro.providers.anthropic.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with patch.object(provider, "_get_api_key", return_value="test-key"):
                await provider.chat_completion(request)

        assert "thinking" not in captured_payload

    @pytest.mark.asyncio
    async def test_anthropic_streaming_forwards_tools_and_tool_choice(self):
        """Streaming Anthropic requests should preserve tool declarations."""
        from router_maestro.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]

        request = ChatRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message(role="user", content="Hello")],
            tools=tools,
            tool_choice={"type": "tool", "name": "get_weather"},
        )

        payload = provider._build_payload(request, stream=True)

        assert payload["stream"] is True
        assert payload["tools"] == tools
        assert payload["tool_choice"] == {"type": "tool", "name": "get_weather"}
