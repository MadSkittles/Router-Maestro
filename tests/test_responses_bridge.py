"""Tests for the experimental ChatRequest -> /responses bridge."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from router_maestro.providers.base import (
    ChatRequest,
    Message,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
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
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "mai-code-1-flash-picker",
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

    def test_catalog_support_overrides_static_ineligibility(self):
        req = self._req("future-responses-only", opt_in=True)
        assert (
            should_use_responses_for_chat(
                req,
                "github-copilot",
                responses_supported=True,
            )
            is True
        )

    def test_catalog_rejection_overrides_static_eligibility(self):
        req = self._req("gpt-5.4", opt_in=True)
        assert (
            should_use_responses_for_chat(
                req,
                "github-copilot",
                responses_supported=False,
            )
            is False
        )

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
    def test_conversion_deeply_isolates_mutable_payload_from_source(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(
                    role="user",
                    content=[
                        {
                            "type": "document",
                            "source": {
                                "pages": [{"text": "original-document"}],
                            },
                        }
                    ],
                )
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"properties": {"query": {"enum": ["original-query"]}}},
                    },
                }
            ],
            tool_choice={
                "type": "vendor_choice",
                "vendor": {"flags": ["original-choice"]},
            },
            metadata={"trace": {"tags": ["original-tag"]}},
            provider_extensions={"vendor": {"flags": ["original-extension"]}},
        )

        out = chat_request_to_responses_request(req)
        assert isinstance(out.input, list)
        assert out.tools is not None
        assert isinstance(out.tool_choice, dict)
        assert out.metadata is not None

        out.input[0]["content"][0]["source"]["pages"][0]["text"] = "polluted-document"
        out.tools[0]["parameters"]["properties"]["query"]["enum"][0] = "polluted-query"
        out.tool_choice["vendor"]["flags"][0] = "polluted-choice"
        out.metadata["trace"]["tags"][0] = "polluted-tag"
        out.provider_extensions["vendor"]["flags"][0] = "polluted-extension"

        source_content = req.messages[0].content
        assert isinstance(source_content, list)
        assert source_content[0]["source"]["pages"][0]["text"] == "original-document"
        assert req.tools is not None
        assert (
            req.tools[0]["function"]["parameters"]["properties"]["query"]["enum"][0]
            == "original-query"
        )
        assert isinstance(req.tool_choice, dict)
        assert req.tool_choice["vendor"]["flags"] == ["original-choice"]
        assert req.metadata == {"trace": {"tags": ["original-tag"]}}
        assert req.provider_extensions == {"vendor": {"flags": ["original-extension"]}}

        assert out.input[0]["content"][0] is not source_content[0]
        assert out.tools[0]["parameters"] is not req.tools[0]["function"]["parameters"]
        assert out.tool_choice is not req.tool_choice
        assert out.metadata is not req.metadata
        assert out.provider_extensions is not req.provider_extensions

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

    @pytest.mark.parametrize("thinking_budget", [1, 1023])
    def test_positive_budget_below_lowest_tier_is_rejected(self, thinking_budget):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            thinking_type="enabled",
            thinking_budget=thinking_budget,
        )

        with pytest.raises(RequestOptionError) as exc_info:
            chat_request_to_responses_request(req)

        assert exc_info.value.status_code == 400
        assert exc_info.value.retryable is False
        assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert exc_info.value.parameter == "thinking_budget"

    def test_lowest_reasoning_budget_maps_to_low(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            thinking_type="enabled",
            thinking_budget=1024,
        )

        out = chat_request_to_responses_request(req)

        assert out.reasoning_effort == "low"

    @pytest.mark.parametrize("thinking_type", [None, "disabled"])
    def test_unset_or_disabled_thinking_does_not_invent_reasoning(self, thinking_type):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            thinking_type=thinking_type,
        )

        out = chat_request_to_responses_request(req)

        assert out.reasoning_effort is None

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

    def test_assistant_refusal_history_uses_refusal_block(self):
        req = ChatRequest(
            model="gpt-5.4",
            messages=[
                Message(role="assistant", content=None, refusal="I cannot help"),
                Message(role="user", content="Why?"),
            ],
        )

        out = chat_request_to_responses_request(req)

        assert out.input[0] == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "I cannot help"}],
        }

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

    def test_refusal_remains_distinct_from_text(self):
        resp = ResponsesResponse(
            content="",
            model="gpt-5.4",
            refusal="I cannot help",
        )

        chat = responses_response_to_chat_response(resp, "gpt-5.4")

        assert chat.content is None
        assert chat.refusal == "I cannot help"

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

    def test_canonical_completed_with_tool_calls_maps_to_tool_calls(self):
        resp = ResponsesResponse(
            content="",
            model="gpt-5.4",
            tool_calls=[ResponsesToolCall(call_id="c1", name="fn", arguments="{}")],
            terminal_outcome=TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.COMPLETED,
            ),
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

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("max_output_tokens", "length"),
            ("content_filter", "content_filter"),
            ("other", "stop"),
        ],
    )
    def test_canonical_incomplete_maps_only_at_chat_adapter(self, reason, expected):
        resp = ResponsesResponse(
            content="partial",
            model="gpt-5.4",
            terminal_outcome=TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.INCOMPLETE,
                incomplete_details={"reason": reason, "vendor": "preserved"},
            ),
        )

        chat = responses_response_to_chat_response(resp, "gpt-5.4")

        assert chat.finish_reason == expected
        assert chat.terminal_outcome == resp.terminal_outcome

    @pytest.mark.parametrize("status", [ResponseStatus.FAILED, ResponseStatus.CANCELLED])
    def test_canonical_failure_cannot_become_non_stream_chat_success(self, status):
        resp = ResponsesResponse(
            content="partial",
            model="gpt-5.4",
            terminal_outcome=TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=status,
                error=TerminalError(code="upstream_terminal", message=status.value),
            ),
        )

        with pytest.raises(ProviderError, match=status.value):
            responses_response_to_chat_response(resp, "gpt-5.4")


class TestStreamChunkConversion:
    def test_text_delta(self):
        chunk = ResponsesStreamChunk(content="hi")
        out = responses_chunk_to_chat_chunk(chunk)
        assert out.content == "hi"
        assert out.tool_calls is None

    def test_refusal_delta_remains_distinct_from_text(self):
        out = responses_chunk_to_chat_chunk(
            ResponsesStreamChunk(content="", refusal="I cannot help")
        )

        assert out.content == ""
        assert out.refusal == "I cannot help"

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

    def test_canonical_terminal_outcome_is_preserved(self):
        from router_maestro.providers.base import (
            ResponseStatus,
            TerminalOutcome,
            TransportTermination,
        )

        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.INCOMPLETE,
            incomplete_details={"reason": "max_output_tokens"},
        )

        out = responses_chunk_to_chat_chunk(
            ResponsesStreamChunk(content="", terminal_outcome=outcome)
        )

        assert out.terminal_outcome is outcome

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("max_output_tokens", "length"),
            ("content_filter", "content_filter"),
            ("other", "stop"),
        ],
    )
    def test_canonical_incomplete_maps_to_lossy_chat_finish(self, reason, expected):
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.INCOMPLETE,
            incomplete_details={"reason": reason},
        )

        out = responses_chunk_to_chat_chunk(
            ResponsesStreamChunk(content="", terminal_outcome=outcome)
        )

        assert out.finish_reason == expected
        assert out.terminal_outcome is outcome

    @pytest.mark.parametrize("status", [ResponseStatus.FAILED, ResponseStatus.CANCELLED])
    def test_canonical_failure_stream_has_no_success_finish(self, status):
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=status,
            error=TerminalError(code="upstream_terminal", message=status.value),
        )

        out = responses_chunk_to_chat_chunk(
            ResponsesStreamChunk(content="", terminal_outcome=outcome)
        )

        assert out.finish_reason is None
        assert out.terminal_outcome is outcome

    @pytest.mark.asyncio
    async def test_copilot_chat_stream_completed_after_tool_call_maps_to_tool_calls(
        self, monkeypatch
    ):
        from router_maestro.providers.copilot import CopilotProvider

        monkeypatch.setenv(ENV_FLAG, "1")
        provider = CopilotProvider()
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.COMPLETED,
        )

        async def responses_stream(_request):
            yield ResponsesStreamChunk(
                content="",
                tool_call=ResponsesToolCall(call_id="c1", name="fn", arguments="{}"),
            )
            yield ResponsesStreamChunk(content="", terminal_outcome=outcome)

        monkeypatch.setattr(provider, "responses_completion_stream", responses_stream)
        request = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            stream=True,
            use_responses_api=True,
        )

        chunks = [chunk async for chunk in provider.chat_completion_stream(request)]

        assert chunks[0].tool_calls is not None
        assert chunks[-1].finish_reason == "tool_calls"
        assert chunks[-1].terminal_outcome is outcome

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reason", "expected_finish"),
        [
            (None, "stop"),
            ("vendor_limit", "stop"),
            ("max_output_tokens", "length"),
            ("content_filter", "content_filter"),
        ],
    )
    async def test_copilot_chat_stream_tool_call_keeps_incomplete_semantics(
        self, monkeypatch, reason, expected_finish
    ):
        from router_maestro.providers.copilot import CopilotProvider

        monkeypatch.setenv(ENV_FLAG, "1")
        provider = CopilotProvider()
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.INCOMPLETE,
            incomplete_details={"reason": reason} if reason is not None else None,
        )

        async def responses_stream(_request):
            yield ResponsesStreamChunk(
                content="",
                tool_call=ResponsesToolCall(call_id="c1", name="fn", arguments="{}"),
            )
            yield ResponsesStreamChunk(content="", terminal_outcome=outcome)

        monkeypatch.setattr(provider, "responses_completion_stream", responses_stream)
        request = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            stream=True,
            use_responses_api=True,
        )

        chunks = [chunk async for chunk in provider.chat_completion_stream(request)]

        assert chunks[-1].finish_reason == expected_finish
        assert chunks[-1].terminal_outcome is outcome

        class RouteRouter:
            async def chat_completion_stream(self, _request, fallback=True):
                async def generate():
                    for chunk in chunks:
                        yield chunk

                return generate(), "github-copilot"

        from router_maestro.server.routes.chat import stream_response

        route_events = [event async for event in stream_response(RouteRouter(), request)]
        assert not any("upstream_protocol_error" in event for event in route_events)
        assert route_events[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_copilot_chat_stream_legacy_stop_after_tool_call_still_upgrades(
        self, monkeypatch
    ):
        from router_maestro.providers.copilot import CopilotProvider

        monkeypatch.setenv(ENV_FLAG, "1")
        provider = CopilotProvider()

        async def responses_stream(_request):
            yield ResponsesStreamChunk(
                content="",
                tool_call=ResponsesToolCall(call_id="c1", name="fn", arguments="{}"),
            )
            yield ResponsesStreamChunk(content="", finish_reason="stop")

        monkeypatch.setattr(provider, "responses_completion_stream", responses_stream)
        request = ChatRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="hi")],
            stream=True,
            use_responses_api=True,
        )

        chunks = [chunk async for chunk in provider.chat_completion_stream(request)]

        assert chunks[-1].finish_reason == "tool_calls"
        assert chunks[-1].terminal_outcome is None


class TestCopilotResponsesCanonicalStatus:
    """Provider results preserve Responses semantics for the protocol adapters."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "details", "error", "expected_status"),
        [
            ("completed", None, None, ResponseStatus.COMPLETED),
            (
                "incomplete",
                {"reason": "max_output_tokens", "vendor": {"limit": 10}},
                None,
                ResponseStatus.INCOMPLETE,
            ),
            (
                "incomplete",
                {"reason": "content_filter", "vendor": "safety"},
                None,
                ResponseStatus.INCOMPLETE,
            ),
            (
                "failed",
                None,
                {"code": "upstream_failed", "message": "upstream failed"},
                ResponseStatus.FAILED,
            ),
            (
                "cancelled",
                None,
                {"code": "upstream_cancelled", "message": "upstream cancelled"},
                ResponseStatus.CANCELLED,
            ),
        ],
    )
    async def test_non_streaming_preserves_canonical_terminal(
        self, status, details, error, expected_status
    ):
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        mock_resp = httpx.Response(
            200,
            json={
                "status": status,
                "model": "gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}],
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                "incomplete_details": details,
                "error": error,
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

        assert out.content == "partial"
        assert out.usage == {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        assert out.finish_reason is None
        assert out.terminal_outcome is not None
        assert out.terminal_outcome.transport is TransportTermination.EXPLICIT_TERMINAL
        assert out.terminal_outcome.response_status is expected_status
        assert out.terminal_outcome.incomplete_details == details
        if error is None:
            assert out.terminal_outcome.error is None
        else:
            assert out.terminal_outcome.error == TerminalError(
                code=error["code"], message=error["message"]
            )

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
        assert out.finish_reason is None
        assert out.terminal_outcome == TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.COMPLETED,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "terminal_fields",
        [
            {"status": "unknown"},
            {"status": "in_progress"},
            {"status": None},
            {"status": []},
            {
                "status": "completed",
                "error": {"code": "conflict", "message": "must not coexist"},
            },
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "error": {"code": "conflict", "message": "must not coexist"},
            },
            {
                "status": "completed",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            {"status": "failed", "error": "not-an-object"},
            {"status": "incomplete", "incomplete_details": "not-an-object"},
        ],
        ids=[
            "unknown",
            "in-progress",
            "null",
            "non-string",
            "completed-with-error",
            "incomplete-with-error",
            "completed-with-details",
            "malformed-error",
            "malformed-details",
        ],
    )
    async def test_non_streaming_malformed_terminal_is_protocol_failure(self, terminal_fields):
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        mock_resp = httpx.Response(
            200,
            json={"model": "gpt-5.4", "output": [], **terminal_fields},
            request=httpx.Request("POST", "https://api.githubcopilot.com/responses"),
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with (
            patch.object(provider, "ensure_token", AsyncMock(return_value=None)),
            patch.object(provider, "_get_headers", return_value={}),
            patch.object(provider, "_get_client", return_value=mock_client),
        ):
            with pytest.raises(ProviderError) as exc_info:
                await provider.responses_completion(ResponsesRequest(model="gpt-5.4", input="hi"))

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502
        assert exc_info.value.upstream_status_code == 200
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [None, [], ["DO_NOT_LEAK"]],
        ids=["null", "list", "list-secret"],
    )
    async def test_non_streaming_terminal_payload_must_be_object(self, payload):
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        mock_resp = httpx.Response(
            200,
            content=json.dumps(payload).encode(),
            request=httpx.Request("POST", "https://api.githubcopilot.com/responses"),
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with (
            patch.object(provider, "ensure_token", AsyncMock(return_value=None)),
            patch.object(provider, "_get_headers", return_value={}),
            patch.object(provider, "_get_client", return_value=mock_client),
        ):
            with pytest.raises(ProviderError) as exc_info:
                await provider.responses_completion(ResponsesRequest(model="gpt-5.4", input="hi"))

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert "DO_NOT_LEAK" not in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raw_error", "expected"),
        [
            ({}, TerminalError(code="upstream_error", message="Upstream response failed")),
            (
                {"code": "quota_exhausted"},
                TerminalError(code="quota_exhausted", message="Upstream response failed"),
            ),
            (
                {"message": "safe upstream message"},
                TerminalError(code="upstream_error", message="safe upstream message"),
            ),
            (
                {"code": None, "message": None, "extra": "ignored"},
                TerminalError(code="upstream_error", message="Upstream response failed"),
            ),
        ],
        ids=["empty", "code-only", "message-only", "null-fields"],
    )
    async def test_non_streaming_error_object_uses_safe_defaults(self, raw_error, expected):
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        mock_resp = httpx.Response(
            200,
            json={"status": "failed", "model": "gpt-5.4", "output": [], "error": raw_error},
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

        assert out.terminal_outcome is not None
        assert out.terminal_outcome.error == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "terminal_fields",
        [
            {"status": "failed", "error": {"code": ["DO_NOT_LEAK"]}},
            {"status": "failed", "error": {"code": {"secret": "DO_NOT_LEAK"}}},
            {"status": "failed", "error": {"code": 7}},
            {"status": "failed", "error": {"code": True}},
            {"status": "failed", "error": {"message": ["DO_NOT_LEAK"]}},
            {"status": "failed", "error": {"message": {"secret": "DO_NOT_LEAK"}}},
            {"status": "failed", "error": {"message": 7}},
            {"status": "failed", "error": {"message": False}},
            {"status": "incomplete", "incomplete_details": {"reason": ["DO_NOT_LEAK"]}},
            {
                "status": "incomplete",
                "incomplete_details": {"reason": {"secret": "DO_NOT_LEAK"}},
            },
            {"status": "incomplete", "incomplete_details": {"reason": 7}},
            {"status": "incomplete", "incomplete_details": {"reason": True}},
        ],
        ids=[
            "code-list",
            "code-dict",
            "code-int",
            "code-bool",
            "message-list",
            "message-dict",
            "message-int",
            "message-bool",
            "reason-list",
            "reason-dict",
            "reason-int",
            "reason-bool",
        ],
    )
    async def test_non_streaming_terminal_subfields_require_strings_without_leaking(
        self, terminal_fields
    ):
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        mock_resp = httpx.Response(
            200,
            json={"model": "gpt-5.4", "output": [], **terminal_fields},
            request=httpx.Request("POST", "https://api.githubcopilot.com/responses"),
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with (
            patch.object(provider, "ensure_token", AsyncMock(return_value=None)),
            patch.object(provider, "_get_headers", return_value={}),
            patch.object(provider, "_get_client", return_value=mock_client),
        ):
            with pytest.raises(ProviderError) as exc_info:
                await provider.responses_completion(ResponsesRequest(model="gpt-5.4", input="hi"))

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert "DO_NOT_LEAK" not in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "details",
        [{}, {"reason": None, "vendor": 1}, {"reason": "vendor_limit", "vendor": [1]}],
        ids=["reason-absent", "reason-null", "reason-unknown-string"],
    )
    async def test_non_streaming_preserves_valid_incomplete_details(self, details):
        from router_maestro.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        mock_resp = httpx.Response(
            200,
            json={
                "status": "incomplete",
                "model": "gpt-5.4",
                "output": [],
                "incomplete_details": details,
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

        assert out.terminal_outcome is not None
        assert out.terminal_outcome.response_status is ResponseStatus.INCOMPLETE
        assert out.terminal_outcome.incomplete_details == details


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
