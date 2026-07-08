"""Tests for the experimental ChatRequest -> /responses bridge."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from router_maestro.providers.base import (
    ChatRequest,
    Message,
    ProviderError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponsesToolCall,
)
from router_maestro.utils.responses_bridge import (
    ENV_FLAG,
    _content_to_responses_blocks,
    chat_request_to_responses_request,
    is_experimental_responses_enabled,
    is_model_responses_eligible,
    map_responses_status_to_chat,
    responses_chunk_to_chat_chunk,
    responses_response_to_chat_response,
    should_use_responses_for_chat,
)


class TestEnvFlag:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "ON"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(ENV_FLAG, value)
        assert is_experimental_responses_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "anything"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv(ENV_FLAG, value)
        assert is_experimental_responses_enabled() is False

    def test_default_off(self, monkeypatch):
        monkeypatch.delenv(ENV_FLAG, raising=False)
        assert is_experimental_responses_enabled() is False


class TestModelEligibility:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.2",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5-mini",
            "gpt-5.4-mini",
            "gpt-5.2-codex",
            "gpt-5.3-codex",
            "github-copilot/gpt-5.4",  # provider prefix
        ],
    )
    def test_eligible(self, model):
        assert is_model_responses_eligible(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4.7",
            "claude-sonnet-4.6",
            "gemini-3.1-pro-preview",
            "gpt-4.1",
            "gpt-4o",
            "github-copilot/claude-opus-4.6",
        ],
    )
    def test_ineligible(self, model):
        assert is_model_responses_eligible(model) is False


class TestShouldUseResponses:
    @pytest.fixture(autouse=True)
    def _enable_flag(self, monkeypatch):
        monkeypatch.setenv(ENV_FLAG, "1")

    def _req(self, model: str, *, opt_in: bool) -> ChatRequest:
        return ChatRequest(
            model=model,
            messages=[Message(role="user", content="hi")],
            use_responses_api=opt_in,
        )

    def test_requires_opt_in(self):
        req = self._req("gpt-5.4", opt_in=False)
        assert should_use_responses_for_chat(req, "github-copilot") is False

    def test_requires_copilot_provider(self):
        req = self._req("gpt-5.4", opt_in=True)
        assert should_use_responses_for_chat(req, "openai") is False

    def test_requires_eligible_model(self):
        req = self._req("claude-opus-4.7", opt_in=True)
        assert should_use_responses_for_chat(req, "github-copilot") is False

    def test_happy_path(self):
        req = self._req("gpt-5.4", opt_in=True)
        assert should_use_responses_for_chat(req, "github-copilot") is True

    def test_env_flag_disabled_blocks_even_with_opt_in(self, monkeypatch):
        """Kill-switch: env off must block even if a caller set use_responses_api."""
        monkeypatch.delenv(ENV_FLAG, raising=False)
        req = self._req("gpt-5.4", opt_in=True)
        assert should_use_responses_for_chat(req, "github-copilot") is False

    def test_env_flag_off_value_blocks(self, monkeypatch):
        monkeypatch.setenv(ENV_FLAG, "off")
        req = self._req("gpt-5.4", opt_in=True)
        assert should_use_responses_for_chat(req, "github-copilot") is False

    def test_multimodal_content_still_eligible(self):
        """Multimodal content is now supported — should not trigger fallback."""
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                    ],
                )
            ],
            use_responses_api=True,
        )
        assert should_use_responses_for_chat(req, "github-copilot") is True

    def test_text_only_list_content_still_eligible(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(role="user", content=[{"type": "text", "text": "hi"}]),
            ],
            use_responses_api=True,
        )
        assert should_use_responses_for_chat(req, "github-copilot") is True


class TestContentToResponsesBlocks:
    def test_plain_string(self):
        assert _content_to_responses_blocks("hello") == [{"type": "input_text", "text": "hello"}]

    def test_empty_string(self):
        assert _content_to_responses_blocks("") == []

    def test_text_block(self):
        content = [{"type": "text", "text": "hi"}]
        assert _content_to_responses_blocks(content) == [{"type": "input_text", "text": "hi"}]

    def test_image_url_nested(self):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        assert _content_to_responses_blocks(content) == [
            {"type": "input_image", "image_url": "data:image/png;base64,abc"}
        ]

    def test_image_url_flat(self):
        content = [{"type": "image_url", "image_url": "https://example.com/img.png"}]
        assert _content_to_responses_blocks(content) == [
            {"type": "input_image", "image_url": "https://example.com/img.png"}
        ]

    def test_mixed_text_and_image(self):
        content = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        ]
        assert _content_to_responses_blocks(content) == [
            {"type": "input_text", "text": "describe this"},
            {"type": "input_image", "image_url": "data:image/png;base64,xyz"},
        ]

    def test_multiple_images(self):
        content = [
            {"type": "image_url", "image_url": {"url": "https://a.png"}},
            {"type": "image_url", "image_url": {"url": "https://b.png"}},
        ]
        assert _content_to_responses_blocks(content) == [
            {"type": "input_image", "image_url": "https://a.png"},
            {"type": "input_image", "image_url": "https://b.png"},
        ]

    def test_unknown_block_passed_through(self):
        content = [{"type": "document", "source": {"type": "base64", "data": "xxx"}}]
        assert _content_to_responses_blocks(content) == [
            {"type": "document", "source": {"type": "base64", "data": "xxx"}}
        ]

    def test_bare_string_in_list(self):
        content = ["hello", "world"]
        assert _content_to_responses_blocks(content) == [
            {"type": "input_text", "text": "hello"},
            {"type": "input_text", "text": "world"},
        ]

    def test_untyped_dict_with_text(self):
        content = [{"text": "legacy"}]
        assert _content_to_responses_blocks(content) == [{"type": "input_text", "text": "legacy"}]

    def test_empty_text_blocks_skipped(self):
        content = [{"type": "text", "text": ""}, {"type": "text", "text": "real"}]
        assert _content_to_responses_blocks(content) == [{"type": "input_text", "text": "real"}]


class TestStatusMapping:
    @pytest.mark.parametrize(
        "status,reason,expected",
        [
            ("completed", None, "stop"),
            ("incomplete", "max_output_tokens", "length"),
            ("incomplete", "content_filter", "content_filter"),
            ("incomplete", "other", "stop"),
            ("failed", None, None),
            ("cancelled", None, None),
            ("unknown_status", None, None),
            (None, None, None),
        ],
    )
    def test_map(self, status, reason, expected):
        assert map_responses_status_to_chat(status, reason) == expected


class TestChatToResponses:
    def test_system_message_becomes_instructions(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(role="system", content="You are a poet."),
                Message(role="user", content="Write a haiku."),
            ],
        )
        out = chat_request_to_responses_request(req)
        assert out.instructions == "You are a poet."
        assert isinstance(out.input, list)
        assert out.input[0]["role"] == "user"
        assert out.input[0]["content"][0]["type"] == "input_text"

    def test_assistant_tool_call_round_trip(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(role="user", content="weather?"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                        }
                    ],
                ),
                Message(
                    role="tool",
                    content="sunny",
                    tool_call_id="call_1",
                ),
            ],
        )
        out = chat_request_to_responses_request(req)
        types = [item["type"] for item in out.input]
        assert types == ["message", "function_call", "function_call_output"]
        assert out.input[1] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"SF"}',
        }
        assert out.input[2] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "sunny",
        }

    def test_tools_translated_to_responses_shape(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "fetch",
                        "description": "fetch a URL",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        out = chat_request_to_responses_request(req)
        assert out.tools == [
            {
                "type": "function",
                "name": "fetch",
                "description": "fetch a URL",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    def test_thinking_budget_derives_effort(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            thinking_budget=8192,
        )
        out = chat_request_to_responses_request(req)
        assert out.reasoning_effort == "high"

    def test_explicit_effort_wins(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            thinking_budget=8192,
            reasoning_effort="low",
        )
        out = chat_request_to_responses_request(req)
        assert out.reasoning_effort == "low"

    def test_max_tokens_maps_to_max_output_tokens(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            max_tokens=1234,
        )
        out = chat_request_to_responses_request(req)
        assert out.max_output_tokens == 1234

    def test_multi_block_text_produces_separate_blocks(self):
        """Multi-block user text becomes separate input_text blocks."""
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "foo"},
                        {"type": "text", "text": "bar"},
                    ],
                )
            ],
        )
        out = chat_request_to_responses_request(req)
        assert out.input[0]["content"] == [
            {"type": "input_text", "text": "foo"},
            {"type": "input_text", "text": "bar"},
        ]

    def test_assistant_history_uses_output_text(self):
        """Replayed assistant turns must use output_text, not input_text.

        Copilot's /responses endpoint rejects input_text in assistant messages;
        assistant content blocks must be typed as output_text.
        """
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(role="user", content="ping"),
                Message(role="assistant", content="pong"),
                Message(role="user", content="again?"),
            ],
        )
        out = chat_request_to_responses_request(req)
        assert out.input[1] == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "pong"}],
        }
        # User messages still use input_text.
        for item in out.input:
            if item.get("type") == "message" and item.get("role") == "user":
                for block in item["content"]:
                    if "text" in block:
                        assert block["type"] == "input_text"

    def test_image_content_converted_to_input_image(self):
        """Image blocks in user messages must become input_image in Responses API."""
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                    ],
                )
            ],
        )
        out = chat_request_to_responses_request(req)
        assert out.input[0] == {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
            ],
        }


class TestResponsesToChat:
    def test_text_only(self):
        resp = ResponsesResponse(
            content="hello",
            model="gpt-5.4",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.content == "hello"
        assert chat.finish_reason == "stop"
        assert chat.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_tool_calls_reshaped(self):
        resp = ResponsesResponse(
            content="",
            model="gpt-5.4",
            tool_calls=[ResponsesToolCall(call_id="c1", name="fn", arguments="{}")],
        )
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.finish_reason == "tool_calls"
        assert chat.tool_calls == [
            {"id": "c1", "type": "function", "function": {"name": "fn", "arguments": "{}"}}
        ]

    def test_thinking_passthrough(self):
        resp = ResponsesResponse(
            content="answer",
            model="gpt-5.4",
            thinking="step 1, step 2",
            thinking_signature="opaque-id",
        )
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.thinking == "step 1, step 2"
        assert chat.thinking_signature == "opaque-id"

    def test_finish_reason_length_preserved(self):
        resp = ResponsesResponse(
            content="truncated",
            model="gpt-5.4",
            finish_reason="length",
        )
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.finish_reason == "length"

    def test_finish_reason_content_filter_preserved(self):
        resp = ResponsesResponse(
            content="",
            model="gpt-5.4",
            finish_reason="content_filter",
        )
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.finish_reason == "content_filter"

    def test_finish_reason_default_when_unset(self):
        resp = ResponsesResponse(content="hi", model="gpt-5.4")
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.finish_reason == "stop"

    def test_tool_calls_finish_default_when_status_missing(self):
        resp = ResponsesResponse(
            content="",
            model="gpt-5.4",
            tool_calls=[ResponsesToolCall(call_id="c1", name="fn", arguments="{}")],
        )
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.finish_reason == "tool_calls"

    def test_completed_status_with_tool_calls_upgrades_to_tool_calls(self):
        """A 'completed' /responses payload that emitted tool calls is a tool-use turn.

        Regression: previously map_responses_status_to_chat("completed") -> "stop"
        beat the tool-call default, leaking 'end_turn' to Anthropic translators.
        """
        resp = ResponsesResponse(
            content="",
            model="gpt-5.4",
            tool_calls=[ResponsesToolCall(call_id="c1", name="fn", arguments="{}")],
            finish_reason="stop",
        )
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.finish_reason == "tool_calls"

    def test_explicit_length_with_tool_calls_keeps_length(self):
        """A non-stop upstream finish_reason should NOT be downgraded by tool calls."""
        resp = ResponsesResponse(
            content="",
            model="gpt-5.4",
            tool_calls=[ResponsesToolCall(call_id="c1", name="fn", arguments="{}")],
            finish_reason="length",
        )
        chat = responses_response_to_chat_response(resp, "gpt-5.4")
        assert chat.finish_reason == "length"


class TestStreamChunkConversion:
    def test_text_delta(self):
        chunk = ResponsesStreamChunk(content="hi")
        out = responses_chunk_to_chat_chunk(chunk)
        assert out.content == "hi"
        assert out.tool_calls is None

    def test_thinking_delta(self):
        chunk = ResponsesStreamChunk(content="", thinking="thought", thinking_signature="sig")
        out = responses_chunk_to_chat_chunk(chunk)
        assert out.thinking == "thought"
        assert out.thinking_signature == "sig"

    def test_tool_call_chunk(self):
        chunk = ResponsesStreamChunk(
            content="",
            tool_call=ResponsesToolCall(call_id="c2", name="g", arguments='{"a":1}'),
        )
        out = responses_chunk_to_chat_chunk(chunk)
        assert out.tool_calls == [
            {"id": "c2", "type": "function", "function": {"name": "g", "arguments": '{"a":1}'}}
        ]

    def test_finish_with_usage(self):
        chunk = ResponsesStreamChunk(
            content="",
            finish_reason="stop",
            usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )
        out = responses_chunk_to_chat_chunk(chunk)
        assert out.finish_reason == "stop"
        assert out.usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


class TestCopilotResponsesFailureRaises:
    """Terminal upstream statuses must surface as ProviderError, not stop."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["failed", "cancelled"])
    async def test_non_streaming_failed_raises(self, status):
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        mock_resp = httpx.Response(
            200,
            json={
                "status": status,
                "model": "gpt-5.4",
                "output": [],
                "error": {"message": f"upstream {status}"},
            },
            request=httpx.Request("POST", "https://api.githubcopilot.com/responses"),
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with (
            patch.object(provider, "ensure_token", AsyncMock(return_value=None)),
            patch.object(provider, "_get_headers", return_value={}),
            patch.object(provider, "_get_client", return_value=mock_client),
        ):
            with pytest.raises(ProviderError) as exc:
                await provider.responses_completion(ResponsesRequest(model="gpt-5.4", input="hi"))
            assert status in str(exc.value)
            assert "upstream" in str(exc.value)

    @pytest.mark.asyncio
    async def test_non_streaming_completed_does_not_raise(self):
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        mock_resp = httpx.Response(
            200,
            json={
                "status": "completed",
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
            request=httpx.Request("POST", "https://api.githubcopilot.com/responses"),
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with (
            patch.object(provider, "ensure_token", AsyncMock(return_value=None)),
            patch.object(provider, "_get_headers", return_value={}),
            patch.object(provider, "_get_client", return_value=mock_client),
        ):
            out = await provider.responses_completion(ResponsesRequest(model="gpt-5.4", input="hi"))
        assert out.content == "ok"
        assert out.finish_reason == "stop"


class TestGeminiRouteFieldPreservation:
    """Regression: Gemini stream route must preserve reasoning + experimental fields."""

    def test_stream_route_helper_preserves_all_fields(self, monkeypatch):
        """_maybe_enable_responses_api must keep reasoning_effort, thinking_*, extra."""
        monkeypatch.setenv(ENV_FLAG, "1")
        from router_maestro.server.routes.gemini import _maybe_enable_responses_api

        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            temperature=0.5,
            max_tokens=512,
            stream=True,
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="auto",
            thinking_budget=4096,
            thinking_type="enabled",
            reasoning_effort="high",
            extra={"custom": "value"},
        )
        out = _maybe_enable_responses_api(req, "gpt-5.4")
        assert out.use_responses_api is True
        assert out.reasoning_effort == "high"
        assert out.thinking_budget == 4096
        assert out.thinking_type == "enabled"
        assert out.tool_choice == "auto"
        assert out.tools == [{"type": "function", "function": {"name": "x"}}]
        assert out.extra == {"custom": "value"}
        assert out.stream is True
        assert out.temperature == 0.5
        assert out.max_tokens == 512

    def test_stream_route_helper_noop_when_flag_off(self, monkeypatch):
        monkeypatch.delenv(ENV_FLAG, raising=False)
        from router_maestro.server.routes.gemini import _maybe_enable_responses_api

        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            reasoning_effort="medium",
        )
        out = _maybe_enable_responses_api(req, "gpt-5.4")
        # Returns the original request unchanged.
        assert out is req
        assert out.use_responses_api is False
        assert out.reasoning_effort == "medium"
