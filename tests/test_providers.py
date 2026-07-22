"""Tests for providers module."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from router_maestro.auth.github_oauth import CopilotTokenResponse
from router_maestro.auth.storage import AuthStorage, OAuthCredential
from router_maestro.providers import (
    AnthropicProvider,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    CopilotProvider,
    Message,
    ModelInfo,
    OpenAICompatibleProvider,
    OpenAIProvider,
    ResponsesRequest,
)
from router_maestro.providers.base import ProviderError, ProviderFailureKind


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
    async def test_copilot_counts_native_anthropic_tokens_via_public_method(self):
        """Native token counting is a provider operation, not a route transport seam."""
        provider = CopilotProvider()
        provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
        provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
            return_value=httpx.Response(
                200,
                json={"input_tokens": 42},
                request=httpx.Request(
                    "POST", "https://api.githubcopilot.com/v1/messages/count_tokens"
                ),
            )
        )
        payload = {
            "model": "claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await provider.count_native_anthropic_tokens(
            payload,
            model="claude-sonnet-4.5",
        )

        assert result == 42
        provider.ensure_token.assert_awaited_once_with()
        provider._send_with_auth_retry.assert_awaited_once_with(
            "POST",
            "/v1/messages/count_tokens",
            json=payload,
            model="claude-sonnet-4.5",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "upstream_payload",
        [
            {},
            [],
            {"input_tokens": True},
            {"input_tokens": -1},
            {"input_tokens": 1.5},
            {"input_tokens": "1"},
        ],
        ids=["missing", "non-object", "boolean", "negative", "float", "string"],
    )
    async def test_copilot_native_token_count_rejects_malformed_success(
        self,
        upstream_payload,
    ):
        provider = CopilotProvider()
        provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
        provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
            return_value=httpx.Response(
                200,
                json=upstream_payload,
                request=httpx.Request(
                    "POST", "https://api.githubcopilot.com/v1/messages/count_tokens"
                ),
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await provider.count_native_anthropic_tokens(
                {"model": "claude-sonnet-4.5", "messages": []},
                model="claude-sonnet-4.5",
            )

        error = exc_info.value
        assert error.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert error.status_code == 502
        assert error.upstream_status_code == 200
        assert error.provider == "github-copilot"
        assert error.model == "claude-sonnet-4.5"

    @pytest.mark.asyncio
    async def test_copilot_native_token_count_maps_upstream_status_to_provider_failure(self):
        provider = CopilotProvider()
        provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
        provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
            return_value=httpx.Response(
                429,
                json={"type": "error", "error": {"type": "rate_limit_error"}},
                request=httpx.Request(
                    "POST", "https://api.githubcopilot.com/v1/messages/count_tokens"
                ),
            )
        )

        with pytest.raises(ProviderError) as exc_info:
            await provider.count_native_anthropic_tokens(
                {"model": "claude-sonnet-4.5", "messages": []},
                model="claude-sonnet-4.5",
            )

        error = exc_info.value
        assert error.kind is ProviderFailureKind.RATE_LIMIT
        assert error.status_code == 429
        assert error.upstream_status_code == 429
        assert error.provider == "github-copilot"
        assert error.model == "claude-sonnet-4.5"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("upstream_error", "expected_status"),
        [
            (httpx.ReadTimeout("slow upstream"), 504),
            (httpx.ReadError("broken transport"), 502),
        ],
        ids=["timeout", "http-error"],
    )
    async def test_copilot_native_token_count_maps_transport_failures(
        self,
        upstream_error,
        expected_status,
    ):
        provider = CopilotProvider()
        provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
        provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
            side_effect=upstream_error
        )

        with pytest.raises(ProviderError) as exc_info:
            await provider.count_native_anthropic_tokens(
                {"model": "claude-sonnet-4.5", "messages": []},
                model="claude-sonnet-4.5",
            )

        error = exc_info.value
        assert error.kind is ProviderFailureKind.TRANSPORT
        assert error.status_code == expected_status
        assert error.provider == "github-copilot"
        assert error.model == "claude-sonnet-4.5"

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

    @pytest.mark.asyncio
    async def test_copilot_models_parse_structured_reasoning_effort_values(self):
        """Copilot advertises effort tiers as objects under a values list."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "claude-opus-4.8",
                            "name": "Claude Opus 4.8",
                            "model_picker_enabled": True,
                            "capabilities": {
                                "type": "chat",
                                "supports": {
                                    "reasoning_effort": {
                                        "values": [
                                            {"value": "low"},
                                            {"value": "xhigh"},
                                            {"value": "max"},
                                        ]
                                    }
                                },
                            },
                        }
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

        assert models[0].reasoning_effort_values == ["low", "xhigh", "max"]

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

    @pytest.mark.asyncio
    async def test_responses_completion_default_raises_provider_error_501(self):
        """Providers without Responses API support should surface a protocol error."""

        class MinimalProvider(BaseProvider):
            async def chat_completion(self, request: ChatRequest) -> ChatResponse:
                return ChatResponse(content="ok", model=request.model)

            async def chat_completion_stream(self, request: ChatRequest):
                yield ChatStreamChunk(content="ok")

            async def list_models(self) -> list[ModelInfo]:
                return []

            def is_authenticated(self) -> bool:
                return True

        provider = MinimalProvider()

        with pytest.raises(ProviderError) as exc:
            await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

        assert exc.value.status_code == 501
        assert exc.value.retryable is False
        assert "Responses API" in str(exc.value)

    @pytest.mark.asyncio
    async def test_responses_completion_stream_default_raises_provider_error_501(self):
        """Streaming defaults should raise ProviderError instead of NotImplementedError."""

        class MinimalProvider(BaseProvider):
            async def chat_completion(self, request: ChatRequest) -> ChatResponse:
                return ChatResponse(content="ok", model=request.model)

            async def chat_completion_stream(self, request: ChatRequest):
                yield ChatStreamChunk(content="ok")

            async def list_models(self) -> list[ModelInfo]:
                return []

            def is_authenticated(self) -> bool:
                return True

        provider = MinimalProvider()

        with pytest.raises(ProviderError) as exc:
            async for _chunk in provider.responses_completion_stream(
                ResponsesRequest(model="gpt-5", input="hi", stream=True)
            ):
                pass

        assert exc.value.status_code == 501
        assert exc.value.retryable is False
        assert "Responses API" in str(exc.value)


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
        assert request.temperature is None
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

    def test_build_payload_converts_openai_tools_to_anthropic_tools(self):
        """Anthropic payloads require native tool schema, not OpenAI wrappers."""
        provider = AnthropicProvider()

        payload = provider._build_payload(
            ChatRequest(
                model="claude-sonnet-4-5",
                messages=[Message(role="user", content="weather?")],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
            )
        )

        assert payload["tools"] == [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

    @pytest.mark.parametrize(
        ("tool_choice", "expected"),
        [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ("required", {"type": "any"}),
            (
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "tool", "name": "get_weather"},
            ),
        ],
    )
    def test_build_payload_converts_openai_tool_choice_to_anthropic(self, tool_choice, expected):
        """OpenAI tool_choice values must be mapped to Anthropic equivalents."""
        provider = AnthropicProvider()

        payload = provider._build_payload(
            ChatRequest(
                model="claude-sonnet-4-5",
                messages=[Message(role="user", content="weather?")],
                tool_choice=tool_choice,
            )
        )

        assert payload["tool_choice"] == expected


def _copilot_with_cred(**cred_kwargs):
    """Build a CopilotProvider with an in-memory stored credential and saved-no-op."""
    provider = CopilotProvider()
    provider.auth_manager.storage = AuthStorage()
    defaults = {"refresh": "ghu_user", "access": "copilot-token", "expires": 0}
    defaults.update(cred_kwargs)
    provider.auth_manager.storage.set("github-copilot", OAuthCredential(**defaults))
    provider.auth_manager.save = Mock()  # type: ignore[method-assign]
    return provider


class TestCopilotResponsesNonStreaming:
    """Tests for Copilot non-streaming /responses parsing."""

    def test_extract_tool_calls_preserves_special_tool_call_kinds(self):
        """custom_tool_call and tool_search_call must keep their Responses item kinds."""
        provider = CopilotProvider()

        tool_calls = provider._extract_tool_calls(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_fn",
                        "name": "lookup",
                        "arguments": '{"query":"x"}',
                        "namespace": "mcp__search__",
                    },
                    {
                        "type": "custom_tool_call",
                        "call_id": "call_custom",
                        "name": "apply_patch",
                        "input": "*** Begin Patch\n*** End Patch",
                    },
                    {
                        "type": "tool_search_call",
                        "call_id": "call_search",
                        "execution": "client",
                        "arguments": {"query": "tools", "limit": 5},
                    },
                ]
            }
        )

        assert [(tc.name, tc.kind) for tc in tool_calls] == [
            ("lookup", "function"),
            ("apply_patch", "custom"),
            ("tool_search", "tool_search"),
        ]
        assert tool_calls[0].namespace == "mcp__search__"
        assert tool_calls[1].arguments == "*** Begin Patch\n*** End Patch"
        assert tool_calls[1].is_custom is True
        assert json.loads(tool_calls[2].arguments) == {"query": "tools", "limit": 5}


class TestCopilotTokenRefresh:
    """Tests for automatic Copilot token refresh + 401/403 retry."""

    @pytest.mark.asyncio
    async def test_ensure_token_reuses_unexpired_persisted_token(self):
        """A restarted server should not re-mint a still-valid persisted token."""
        provider = _copilot_with_cred(access="persisted-token", expires=2**31)

        with patch("router_maestro.providers.copilot.get_copilot_token", new=AsyncMock()) as mint:
            await provider.ensure_token()

        mint.assert_not_awaited()
        assert provider._cached_token == "persisted-token"
        assert provider._token_expires == 2**31

    @pytest.mark.asyncio
    async def test_transient_refresh_failure_uses_token_until_actual_expiration(self):
        """A refresh-window 502 should not discard a token that remains usable."""
        import time

        provider = _copilot_with_cred(
            access="nearly-expired-token",
            expires=int(time.time()) + 30,
        )
        response = httpx.Response(
            502,
            request=httpx.Request("GET", "https://api.github.com/copilot_internal/v2/token"),
        )
        error = httpx.HTTPStatusError(
            "bad gateway",
            request=response.request,
            response=response,
        )

        with patch(
            "router_maestro.providers.copilot.get_copilot_token",
            new=AsyncMock(side_effect=error),
        ):
            await provider.ensure_token()

        assert provider._cached_token == "nearly-expired-token"

    @pytest.mark.asyncio
    async def test_forced_refresh_does_not_reuse_rejected_token(self):
        """The 401/403 recovery path must not fall back to a rejected old token."""
        provider = _copilot_with_cred(access="rejected-token", expires=2**31)
        response = httpx.Response(
            502,
            request=httpx.Request("GET", "https://api.github.com/copilot_internal/v2/token"),
        )
        error = httpx.HTTPStatusError(
            "bad gateway",
            request=response.request,
            response=response,
        )

        with (
            patch(
                "router_maestro.providers.copilot.get_copilot_token",
                new=AsyncMock(side_effect=error),
            ),
            pytest.raises(ProviderError) as exc_info,
        ):
            await provider.ensure_token(force=True)

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_ensure_token_dead_token_raises_reauth_error(self):
        """No refresh token + a 403 mint surfaces a clear re-auth error.

        It stays retryable=True so the router can still fall back to other
        configured providers; the message tells the user how to recover when
        Copilot is their only provider.
        """
        provider = _copilot_with_cred()  # mint will be rejected

        reject = httpx.HTTPStatusError(
            "forbidden",
            request=httpx.Request("GET", "https://api.github.com"),
            response=httpx.Response(403),
        )
        with patch(
            "router_maestro.providers.copilot.get_copilot_token",
            new=AsyncMock(side_effect=reject),
        ):
            with pytest.raises(ProviderError) as exc:
                await provider.ensure_token()

        assert exc.value.status_code == 401
        assert exc.value.retryable is True
        assert "auth login github-copilot" in str(exc.value)

    @pytest.mark.asyncio
    async def test_chat_completion_retries_once_on_403(self):
        """A 403 from the chat API forces a token refresh and retries the call once."""
        provider = _copilot_with_cred()
        provider._cached_token = "stale-token"
        provider._token_expires = 2**31  # clock thinks it's valid

        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(403, json={"detail": "forbidden"})
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o",
                    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                },
            )

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(provider, "ensure_token", new=AsyncMock()) as mock_ensure:
            response = await provider.chat_completion(
                ChatRequest(
                    model="github-copilot/gpt-4o", messages=[Message(role="user", content="hey")]
                )
            )

        assert calls["n"] == 2  # original + retry
        # ensure_token called once at entry, once forced after the 403.
        assert mock_ensure.await_count == 2
        assert mock_ensure.await_args.kwargs.get("force") is True
        assert response.content == "hi"

    @pytest.mark.asyncio
    async def test_chat_stream_retries_once_on_403(self):
        """A 403 opening the chat stream forces a refresh and retries before yielding."""
        provider = _copilot_with_cred()
        provider._cached_token = "stale-token"
        provider._token_expires = 2**31

        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(403, json={"detail": "forbidden"})
            body = (
                'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body)

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(provider, "ensure_token", new=AsyncMock()) as mock_ensure:
            chunks = [
                c
                async for c in provider.chat_completion_stream(
                    ChatRequest(
                        model="github-copilot/gpt-4o",
                        messages=[Message(role="user", content="hey")],
                    )
                )
            ]

        assert calls["n"] == 2
        assert mock_ensure.await_count == 2
        assert "".join(c.content for c in chunks if c.content) == "hi"

    @pytest.mark.asyncio
    async def test_force_refresh_skips_when_token_changed_under_lock(self):
        """force=True early-returns if another coro re-minted while we waited (#1/#3)."""
        provider = _copilot_with_cred()
        provider._cached_token = "token-A"
        provider._token_expires = 0

        # Acquire the refresh lock so our ensure_token(force=True) blocks on it,
        # then swap the cached token (as a winning coroutine would) before release.
        await provider._token_refresh_lock.acquire()

        async def swap_then_release():
            provider._cached_token = "token-B"  # another coro minted a fresh token
            provider._token_refresh_lock.release()

        mint = AsyncMock()
        with patch("router_maestro.providers.copilot.get_copilot_token", new=mint):
            import asyncio

            waiter = asyncio.create_task(provider.ensure_token(force=True))
            await asyncio.sleep(0)  # let the waiter block on the lock
            await swap_then_release()
            await waiter

        # The waiter saw token-B (!= token-A snapshot) and skipped the mint.
        mint.assert_not_awaited()
        assert provider._cached_token == "token-B"

    @pytest.mark.asyncio
    async def test_concurrent_force_refresh_mints_once(self):
        """N concurrent 401-driven force refreshes collapse to a single mint (#3)."""
        import asyncio

        provider = _copilot_with_cred()
        provider._cached_token = "stale"
        provider._token_expires = 0

        mint_calls = {"n": 0}

        async def slow_mint(_client, _gh_token):
            mint_calls["n"] += 1
            await asyncio.sleep(0.05)  # hold the lock so others queue behind us
            return CopilotTokenResponse(
                token=f"fresh-{mint_calls['n']}", expires_at=2**31, refresh_in=1000
            )

        with patch("router_maestro.providers.copilot.get_copilot_token", new=slow_mint):
            await asyncio.gather(*(provider.ensure_token(force=True) for _ in range(5)))

        # The first waiter mints; the rest see the freshly-swapped token and skip.
        assert mint_calls["n"] == 1

    @pytest.mark.asyncio
    async def test_mint_persists_auth_json_once(self):
        """A successful mint persists the new Copilot token exactly once."""
        provider = _copilot_with_cred()

        with patch(
            "router_maestro.providers.copilot.get_copilot_token",
            new=AsyncMock(
                return_value=CopilotTokenResponse(
                    token="new-copilot", expires_at=2**31, refresh_in=1000
                )
            ),
        ):
            await provider.ensure_token()

        assert provider.auth_manager.save.call_count == 1
        stored = provider.auth_manager.storage.get("github-copilot")
        assert stored.access == "new-copilot"
        assert provider._cached_token == "new-copilot"

    @pytest.mark.asyncio
    async def test_chat_403_retries_once_then_raises_retryable(self):
        """A persistent 403 on chat refreshes once, retries once, then raises retryable.

        retryable=True is load-bearing: it lets the router fall back to other
        providers. This must match the /responses behavior (no stream-vs-chat
        divergence).
        """
        provider = _copilot_with_cred()
        provider._cached_token = "tok"
        provider._token_expires = 2**31

        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(403, json={"detail": "policy"})

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(provider, "ensure_token", new=AsyncMock()) as mock_ensure:
            with pytest.raises(ProviderError) as exc:
                await provider.chat_completion(
                    ChatRequest(
                        model="github-copilot/gpt-4o",
                        messages=[Message(role="user", content="hey")],
                    )
                )

        assert calls["n"] == 2  # exactly one retry, never an unbounded storm
        assert mock_ensure.await_count == 2  # entry + one forced refresh
        assert exc.value.status_code == 403
        assert exc.value.retryable is True

    @pytest.mark.asyncio
    async def test_list_models_serves_stale_cache_on_dead_token(self):
        """An unrecoverable token serves the stale model cache instead of hard-failing (#4)."""
        provider = _copilot_with_cred()
        provider._cached_token = "stale"
        provider._token_expires = 2**31
        # Seed a stale cache.
        cached = [
            ModelInfo(id="gpt-4o", name="GPT-4o", provider="github-copilot"),
        ]
        provider._models_ttl_cache.set(cached)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"detail": "forbidden"})

        # Entry ensure_token() succeeds; the in-loop force=True refresh raises
        # (token unrecoverable) — that ProviderError must degrade to stale cache.
        async def maybe_dead(force: bool = False):
            if force:
                raise ProviderError("auth login github-copilot", status_code=401, retryable=True)

        with patch(
            "httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ):
            with patch.object(provider, "ensure_token", new=AsyncMock(side_effect=maybe_dead)):
                models = await provider.list_models(force_refresh=True)

        assert [m.id for m in models] == ["gpt-4o"]

    @pytest.mark.asyncio
    async def test_responses_stream_403_is_retryable_for_fallback(self):
        """A surviving 403 on /responses stream is retryable so the router can fall back (#3)."""
        provider = _copilot_with_cred()
        provider._cached_token = "tok"
        provider._token_expires = 2**31

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"detail": "forbidden"})

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(provider, "ensure_token", new=AsyncMock()):
            with pytest.raises(ProviderError) as exc:
                async for _ in provider.responses_completion_stream(
                    ResponsesRequest(model="github-copilot/gpt-5", input="hi", stream=True)
                ):
                    pass

        assert exc.value.status_code == 403
        assert exc.value.retryable is True
