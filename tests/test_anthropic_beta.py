"""Tests for the Anthropic beta passthrough route."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from router_maestro.config import PrioritiesConfig, ThinkingBudgetConfig
from router_maestro.providers.base import (
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    ResponseStatus,
    TransportTermination,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    Feature,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan
from router_maestro.server.routes.anthropic import ANTHROPIC_PING_FRAME
from router_maestro.server.routes.anthropic_beta import (
    _apply_thinking_budget_native,
    _clean_stream_frame,
    _encode_native_stream_errors,
    _is_native_eligible,
    _is_signature_error,
    _iter_sse_frames,
    _NativeModelResolution,
    _parse_native_message_response,
    _ResolvedModel,
    _sanitize_output_config,
    _send_native_nonstream,
    _stream_native_candidate,
    _strip_history_thinking_blocks,
    _strip_response,
    router,
)

_VALID_MESSAGE_START = {
    "type": "message_start",
    "message": {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": "claude-sonnet-4.5",
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 0},
    },
}
_MISSING = object()


def _message_start_with_model(value=_MISSING):
    event = deepcopy(_VALID_MESSAGE_START)
    if value is _MISSING:
        event["message"].pop("model")
    else:
        event["message"]["model"] = value
    return event


def _valid_native_message_response() -> dict:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-sonnet-4.5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }


@pytest.mark.asyncio
async def test_native_stream_guard_abort_uses_one_anthropic_overload_terminal() -> None:
    class _Response:
        async def aiter_lines(self):
            yield "event: message_start"
            yield f"data: {json.dumps(_VALID_MESSAGE_START)}"
            yield ""
            yield "event: message_stop"
            yield 'data: {"type":"message_stop"}'
            yield ""

    class _Guard:
        def feed_frame(self, _event_type, _data):
            return "guard overloaded"

    frames = [frame async for frame in _iter_sse_frames(_Response(), leak_guard=_Guard())]
    event_types = [
        line.removeprefix("event: ")
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("event: ")
    ]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("data: ")
    ]

    assert event_types == ["error"]
    assert "message_stop" not in event_types
    assert payloads == [
        {
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Overloaded: please retry this request",
            },
        }
    ]


def _assert_native_protocol_failure(error: ProviderError) -> None:
    assert error.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert error.status_code == 502
    assert error.upstream_status_code == 200
    assert error.retryable is True
    assert error.provider == "github-copilot"
    assert error.model == "claude-malformed"
    assert error.safe_message == "Native Anthropic upstream returned a malformed response"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda data: data["content"].__setitem__(0, 7), id="content-element"),
        pytest.param(lambda data: data["content"][0].pop("type"), id="missing-block-type"),
        pytest.param(
            lambda data: data["content"][0].__setitem__("type", 7),
            id="invalid-block-type",
        ),
        pytest.param(lambda data: data["content"][0].__setitem__("text", 7), id="text-text"),
        pytest.param(
            lambda data: data["content"].__setitem__(
                0, {"type": "thinking", "thinking": 7, "signature": "sig"}
            ),
            id="thinking-text",
        ),
        pytest.param(
            lambda data: data["content"].__setitem__(
                0, {"type": "thinking", "thinking": "ok", "signature": 7}
            ),
            id="thinking-signature",
        ),
        pytest.param(
            lambda data: data["content"].__setitem__(0, {"type": "redacted_thinking", "data": 7}),
            id="redacted-thinking-data",
        ),
        pytest.param(
            lambda data: data["content"].__setitem__(
                0, {"type": "tool_use", "id": 7, "name": "tool", "input": {}}
            ),
            id="tool-use-id",
        ),
        pytest.param(
            lambda data: data["content"].__setitem__(
                0, {"type": "tool_use", "id": "id", "name": 7, "input": {}}
            ),
            id="tool-use-name",
        ),
        pytest.param(
            lambda data: data["content"].__setitem__(
                0, {"type": "tool_use", "id": "id", "name": "tool", "input": []}
            ),
            id="tool-use-input",
        ),
        pytest.param(
            lambda data: data["usage"].__setitem__("input_tokens", True),
            id="usage-bool",
        ),
        pytest.param(
            lambda data: data["usage"].__setitem__("output_tokens", -1),
            id="usage-negative",
        ),
        pytest.param(lambda data: data.__setitem__("stop_reason", 7), id="stop-reason"),
        pytest.param(lambda data: data.__setitem__("stop_sequence", []), id="stop-sequence"),
    ],
)
def test_native_nonstream_rejects_malformed_known_response_fields(mutate) -> None:
    payload = _valid_native_message_response()
    mutate(payload)
    response = MagicMock()
    response.json.return_value = payload

    with pytest.raises(ProviderError) as exc_info:
        _parse_native_message_response(
            response,
            provider="github-copilot",
            model="claude-malformed",
        )

    _assert_native_protocol_failure(exc_info.value)


def test_native_nonstream_preserves_unknown_string_block_type() -> None:
    payload = _valid_native_message_response()
    payload["content"] = [{"type": "future_block", "future": {"x": 1}}]
    response = MagicMock()
    response.json.return_value = payload

    parsed = _parse_native_message_response(
        response,
        provider="github-copilot",
        model="claude-malformed",
    )

    assert parsed["content"] == payload["content"]


# --- Unit tests for helper functions ---


class TestIsNativeEligible:
    def test_copilot_claude_model(self):
        assert _is_native_eligible("github-copilot", "claude-sonnet-4.5") is True

    def test_copilot_claude_with_prefix(self):
        assert _is_native_eligible("github-copilot", "github-copilot/claude-opus-4.6") is True

    def test_copilot_non_claude(self):
        assert _is_native_eligible("github-copilot", "gpt-5.4") is False

    def test_non_copilot_claude(self):
        assert _is_native_eligible("anthropic", "claude-sonnet-4.5") is False

    def test_non_copilot_non_claude(self):
        assert _is_native_eligible("openai", "gpt-4o") is False


class TestStripResponse:
    def test_strips_copilot_usage(self):
        data = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "hi"}],
            "copilot_usage": {"token_details": []},
            "stop_details": {"reason": "end"},
            "model": "claude-sonnet-4.5",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = _strip_response(data)
        assert "copilot_usage" not in result
        assert "stop_details" not in result
        assert result["id"] == "msg_123"
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}

    def test_preserves_standard_fields(self):
        data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-sonnet-4.5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = _strip_response(data)
        assert result == data

    def test_handles_no_copilot_fields(self):
        data = {"id": "msg_123", "content": []}
        result = _strip_response(data)
        assert result == {"id": "msg_123", "content": []}


class TestCleanStreamFrame:
    def test_filters_copilot_usage_event(self):
        assert _clean_stream_frame("copilot_usage", '{"some": "data"}') is None

    def test_strips_message_start_copilot_fields(self):
        event = deepcopy(_VALID_MESSAGE_START)
        event["message"]["copilot_usage"] = {"x": 1}
        event["message"]["stop_details"] = {"y": 2}
        event["message"]["usage"] = {"input_tokens": 10}
        data = json.dumps(event)
        result = _clean_stream_frame("message_start", data)
        parsed = json.loads(result)
        assert "copilot_usage" not in parsed["message"]
        assert "stop_details" not in parsed["message"]
        assert parsed["message"]["usage"] == {"input_tokens": 10}

    def test_strips_message_stop_bedrock_metrics(self):
        data = json.dumps(
            {
                "type": "message_stop",
                "amazon-bedrock-invocationMetrics": {"latency": 1000},
                "copilot_usage": {"x": 1},
            }
        )
        result = _clean_stream_frame("message_stop", data)
        parsed = json.loads(result)
        assert "amazon-bedrock-invocationMetrics" not in parsed
        assert "copilot_usage" not in parsed
        assert parsed["type"] == "message_stop"

    def test_passes_through_content_block_delta(self):
        data = (
            '{"type": "content_block_delta", "index": 0, '
            '"delta": {"type": "text_delta", "text": "hi"}}'
        )
        result = _clean_stream_frame("content_block_delta", data)
        assert result == data

    def test_passes_through_thinking_delta(self):
        data = (
            '{"type": "content_block_delta", "delta": '
            '{"type": "thinking_delta", "thinking": "hmm"}, "index": 0}'
        )
        result = _clean_stream_frame("content_block_delta", data)
        assert json.loads(result) == json.loads(data)

    def test_rejects_event_missing_required_anthropic_shape(self):
        with pytest.raises(ProviderError) as exc_info:
            _clean_stream_frame("message_start", '{"type":"message_start"}')

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.retryable is True

    @pytest.mark.parametrize(
        "model_value",
        [
            pytest.param(_MISSING, id="missing"),
            pytest.param(None, id="none"),
            pytest.param(42, id="non-string"),
            pytest.param("", id="empty"),
        ],
    )
    def test_rejects_message_start_without_nonempty_string_model(self, model_value):
        event = _message_start_with_model(model_value)

        with pytest.raises(ProviderError) as exc_info:
            _clean_stream_frame("message_start", json.dumps(event))

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.retryable is True

    @pytest.mark.parametrize(
        ("event_type", "payload"),
        [
            ("message_start", {"type": "message_start", "message": {}}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": "0", "content_block": {}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": ""}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": True}),
            ("message_delta", {"type": "message_delta", "delta": {}, "usage": []}),
            ("error", {"type": "error", "error": "bad"}),
        ],
        ids=[
            "message-start",
            "content-block-start",
            "content-block-delta",
            "content-block-stop",
            "message-delta",
            "error",
        ],
    )
    def test_rejects_known_event_with_malformed_minimum_shape(self, event_type, payload):
        with pytest.raises(ProviderError) as exc_info:
            _clean_stream_frame(event_type, json.dumps(payload))

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.retryable is True

    @pytest.mark.parametrize(
        ("event_type", "payload"),
        [
            (
                "message_start",
                {
                    **_VALID_MESSAGE_START,
                    "message": {
                        **_VALID_MESSAGE_START["message"],
                        "content": [7],
                    },
                },
            ),
            (
                "message_start",
                {
                    **_VALID_MESSAGE_START,
                    "message": {
                        **_VALID_MESSAGE_START["message"],
                        "usage": {"input_tokens": True, "output_tokens": 0},
                    },
                },
            ),
            (
                "message_start",
                {
                    **_VALID_MESSAGE_START,
                    "message": {
                        **_VALID_MESSAGE_START["message"],
                        "stop_reason": 7,
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": 7},
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": 7},
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "redacted_thinking", "data": {}},
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "id",
                        "name": "tool",
                        "input": [],
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": 7},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": {}},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": 7},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": []},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": 7, "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": None, "stop_sequence": []},
                    "usage": {"output_tokens": 1},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": None, "stop_sequence": None},
                    "usage": {"output_tokens": "1"},
                },
            ),
            (
                "error",
                {
                    "type": "error",
                    "error": {"type": 7, "message": "safe"},
                },
            ),
            (
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": {}},
                },
            ),
        ],
        ids=[
            "message-content",
            "message-usage",
            "message-stop",
            "start-text",
            "start-thinking",
            "start-redacted",
            "start-tool",
            "delta-text",
            "delta-thinking",
            "delta-signature",
            "delta-json",
            "message-delta-reason",
            "message-delta-sequence",
            "message-delta-usage",
            "error-type",
            "error-message",
        ],
    )
    def test_rejects_malformed_deep_fields_for_known_events(self, event_type, payload):
        with pytest.raises(ProviderError) as exc_info:
            _clean_stream_frame(event_type, json.dumps(payload))

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.status_code == 502
        assert exc_info.value.upstream_status_code == 200
        assert exc_info.value.retryable is True

    @pytest.mark.parametrize(
        ("event_type", "payload"),
        [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "future_block", "future": {"x": 1}},
                    "future_event_field": True,
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "future_delta", "future": [1, 2]},
                    "future_event_field": True,
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"future": "value"},
                    "usage": {"output_tokens": 1, "future": 2},
                    "future_event_field": True,
                },
            ),
            ("message_stop", {"type": "message_stop", "future": True}),
            ("ping", {"type": "ping", "future": True}),
        ],
    )
    def test_known_event_allows_extra_fields_and_future_subtypes(self, event_type, payload):
        assert json.loads(_clean_stream_frame(event_type, json.dumps(payload))) == payload


class TestIsSignatureError:
    def test_detects_signature_in_thinking(self):
        text = '{"message":"messages.3.content.0: Invalid `signature` in `thinking` block"}'
        assert _is_signature_error(text) is True

    def test_ignores_unrelated_400(self):
        text = '{"message":"max_tokens: Field required"}'
        assert _is_signature_error(text) is False

    def test_ignores_signature_without_thinking(self):
        text = '{"message":"Invalid signature format"}'
        assert _is_signature_error(text) is False


class TestStripHistoryThinkingBlocks:
    def test_strips_thinking_from_assistant(self):
        body = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "...", "signature": "sig"},
                        {"type": "text", "text": "Hello"},
                    ],
                },
                {"role": "user", "content": "Bye"},
            ]
        }
        _strip_history_thinking_blocks(body)
        assistant_content = body["messages"][1]["content"]
        assert len(assistant_content) == 1
        assert assistant_content[0]["type"] == "text"

    def test_preserves_user_messages(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            ]
        }
        _strip_history_thinking_blocks(body)
        assert len(body["messages"][0]["content"]) == 1

    def test_preserves_tool_use_blocks(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "x", "signature": "s"},
                        {"type": "tool_use", "id": "t1", "name": "fn", "input": {}},
                        {"type": "text", "text": "done"},
                    ],
                },
            ]
        }
        _strip_history_thinking_blocks(body)
        content = body["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "tool_use"
        assert content[1]["type"] == "text"

    def test_no_op_when_no_thinking(self):
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            ]
        }
        _strip_history_thinking_blocks(body)
        assert len(body["messages"][0]["content"]) == 1

    def test_handles_string_content(self):
        body = {"messages": [{"role": "assistant", "content": "plain text"}]}
        _strip_history_thinking_blocks(body)
        assert body["messages"][0]["content"] == "plain text"


class TestApplyThinkingBudgetNative:
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_effort_removes_conflicting_budget(self, mock_resolve_tb):
        body = {
            "thinking": {"type": "adaptive", "budget_tokens": 16000},
            "output_config": {"effort": "xhigh"},
        }

        result = _apply_thinking_budget_native(body, "claude-opus-4.8")

        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "xhigh"}
        mock_resolve_tb.assert_not_called()

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_effort_preserves_required_enabled_budget(
        self, mock_resolve_tb, mock_config, mock_router
    ):
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        mock_router.return_value = MagicMock(_models_cache={})
        mock_resolve_tb.return_value = (16000, "enabled")
        body = {
            "thinking": {"type": "enabled", "budget_tokens": 16000},
            "output_config": {"effort": "xhigh"},
        }

        result = _apply_thinking_budget_native(body, "claude-opus-4.6")

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert result["output_config"] == {"effort": "xhigh"}
        mock_resolve_tb.assert_called_once()

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_effort_maps_to_catalog_supported_tier(self, mock_resolve_tb, mock_config, mock_router):
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        model_info = ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            max_output_tokens=64000,
            supports_thinking=True,
            reasoning_effort_values=["low", "medium", "high", "max"],
        )
        mock_router.return_value = MagicMock(
            _models_cache={"claude-opus-4.6": ("github-copilot", model_info)}
        )
        mock_resolve_tb.return_value = (16000, "enabled")

        result = _apply_thinking_budget_native(
            {
                "max_tokens": 64000,
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "output_config": {"effort": "xhigh"},
            },
            "claude-opus-4.6",
            ("low", "medium", "high", "max"),
        )

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        # Decision C: xhigh must substitute down to high, never up to max.
        assert result["output_config"] == {"effort": "high"}

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_enabled_budget_normalization_preserves_display(
        self, mock_resolve_tb, mock_config, mock_router
    ):
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        mock_router.return_value = MagicMock(_models_cache={})
        mock_resolve_tb.return_value = (16000, "enabled")

        result = _apply_thinking_budget_native(
            {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 16000,
                    "display": "summarized",
                },
                "output_config": {"effort": "xhigh"},
            },
            "claude-opus-4.6",
        )

        assert result["thinking"] == {
            "type": "enabled",
            "budget_tokens": 16000,
            "display": "summarized",
        }

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_effort_fills_missing_enabled_budget(self, mock_resolve_tb, mock_config, mock_router):
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        mock_router.return_value = MagicMock(_models_cache={})
        mock_resolve_tb.return_value = (16000, "enabled")

        result = _apply_thinking_budget_native(
            {
                "thinking": {"type": "enabled"},
                "output_config": {"effort": "xhigh"},
            },
            "claude-opus-4.6",
        )

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert result["output_config"] == {"effort": "xhigh"}
        mock_resolve_tb.assert_called_once()

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_effort_normalizes_enabled_budget(self, mock_resolve_tb, mock_config, mock_router):
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        mock_router.return_value = MagicMock(_models_cache={})
        mock_resolve_tb.return_value = (4095, "enabled")

        result = _apply_thinking_budget_native(
            {
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "output_config": {"effort": "xhigh"},
            },
            "claude-opus-4.6",
        )

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 4095}
        assert result["output_config"] == {"effort": "xhigh"}
        mock_resolve_tb.assert_called_once()
        assert mock_resolve_tb.call_args.kwargs["max_output_tokens"] == 4096

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_effort_removes_disabled_thinking(self, mock_resolve_tb, mock_config, mock_router):
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        mock_router.return_value = MagicMock(_models_cache={})
        mock_resolve_tb.return_value = (None, None)

        result = _apply_thinking_budget_native(
            {
                "thinking": {"type": "disabled", "budget_tokens": 5000},
                "output_config": {"effort": "xhigh"},
            },
            "claude-opus-4.6",
        )

        assert "thinking" not in result
        assert result["output_config"] == {"effort": "xhigh"}
        mock_resolve_tb.assert_called_once()

    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_adaptive_without_effort_omits_budget(self, mock_resolve_tb):
        result = _apply_thinking_budget_native(
            {"thinking": {"type": "adaptive", "budget_tokens": 16000}},
            "claude-opus-4.8",
        )

        assert result["thinking"] == {"type": "adaptive"}
        mock_resolve_tb.assert_not_called()

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_enabled_without_budget_headroom_is_removed(
        self, mock_resolve_tb, mock_config, mock_router
    ):
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        mock_router.return_value = MagicMock(_models_cache={})
        mock_resolve_tb.return_value = (None, None)

        result = _apply_thinking_budget_native(
            {
                "max_tokens": 1024,
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "output_config": {"effort": "xhigh"},
            },
            "claude-opus-4.6",
        )

        assert "thinking" not in result
        mock_resolve_tb.assert_called_once()

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_no_change_when_client_sets_budget(self, mock_resolve_tb, mock_config, mock_router):
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        mock_router.return_value = MagicMock(_models_cache={})
        mock_resolve_tb.return_value = (5000, "enabled")

        body = {"thinking": {"type": "enabled", "budget_tokens": 5000}}
        result = _apply_thinking_budget_native(body, "claude-sonnet-4.5")
        assert result["thinking"]["budget_tokens"] == 5000

    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.resolve_thinking_budget")
    def test_removes_thinking_when_server_disables(self, mock_resolve_tb, mock_config, mock_router):
        """Server config forces thinking off when client requested it."""
        mock_config.return_value = MagicMock(
            thinking=MagicMock(default_budget=16000, auto_enable=False, model_budgets={})
        )
        mock_router.return_value = MagicMock(_models_cache={})
        # Client asked for thinking, but server resolves to disabled
        mock_resolve_tb.return_value = (None, "disabled")

        body = {"thinking": {"type": "enabled", "budget_tokens": 5000}}
        result = _apply_thinking_budget_native(body, "claude-sonnet-4.5")
        assert "thinking" not in result


class TestSanitizeOutputConfig:
    def test_preserves_only_valid_effort(self):
        body = {"output_config": {"effort": "xhigh", "format": "json"}}

        _sanitize_output_config(body)

        assert body["output_config"] == {"effort": "xhigh"}

    @pytest.mark.parametrize(
        "value",
        [None, {}, {"format": "json"}, {"effort": "invalid"}, "xhigh"],
    )
    def test_removes_output_config_without_valid_effort(self, value):
        body = {"output_config": value}

        _sanitize_output_config(body)

        assert "output_config" not in body


# --- Integration tests with TestClient ---


@pytest.fixture
def app():
    """Create a test FastAPI app with only the beta router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def non_raising_client(app):
    return TestClient(app, raise_server_exceptions=False)


def _native_provider() -> MagicMock:
    provider = MagicMock(spec=CopilotProvider)
    provider.name = "github-copilot"
    provider.ensure_token = AsyncMock()

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-sonnet-4.5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    provider._send_with_auth_retry = AsyncMock(return_value=response)
    provider.count_native_anthropic_tokens = AsyncMock(return_value=42)
    return provider


def _guard_config(
    *,
    leak_enabled: bool,
    runaway_enabled: bool,
    max_bytes: int = 100_000,
) -> PrioritiesConfig:
    return PrioritiesConfig.model_validate(
        {
            "guards": {
                "leak_guard": {"enabled": leak_enabled},
                "runaway_guard": {
                    "enabled": runaway_enabled,
                    "max_bytes": max_bytes,
                    "max_deltas": 1_000,
                },
            }
        }
    )


class _CapturedGuardContext:
    """Small RequestContext surface used to prove pipeline snapshot binding."""

    def __init__(self, configs: list[PrioritiesConfig], router=None) -> None:
        self.request_id = "req-beta-guard"
        self._configs = configs
        self.router = router or MagicMock(_models_cache={})
        self.pipeline = None
        self.audit = None

    @property
    def config(self) -> PrioritiesConfig:
        return self._configs[0]


class _GuardSnapshot:
    def __init__(self, config: PrioritiesConfig) -> None:
        self.revision = "guard-revision"
        self._config = config

    @property
    def config(self) -> PrioritiesConfig:
        return self._config.model_copy(deep=True)


class _GuardLease:
    def __init__(self, config: PrioritiesConfig) -> None:
        self.generation_id = 1
        self.router = MagicMock(_models_cache={})
        self.config_snapshot = _GuardSnapshot(config)
        self.release_count = 0

    async def release(self) -> None:
        self.release_count += 1


def _real_native_router(provider, models: list[ModelInfo], priorities: list[str]):
    from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
    from router_maestro.utils.cache import TTLCache

    provider.is_authenticated.return_value = True
    provider.capabilities = CopilotProvider().capabilities
    router_instance = Router.__new__(Router)
    router_instance.providers = {"github-copilot": provider}
    router_instance._models_cache = {}
    for model in models:
        router_instance._models_cache.setdefault(
            model.id,
            ("github-copilot", model),
        )
        router_instance._models_cache[f"github-copilot/{model.id}"] = (
            "github-copilot",
            model,
        )
    router_instance._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router_instance._models_cache_ttl.set(True)
    router_instance._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router_instance._priorities_cache.set(
        PrioritiesConfig(
            priorities=priorities,
            fallback={"strategy": "priority", "maxRetries": 10},
        )
    )
    router_instance._fuzzy_cache = {}
    router_instance._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router_instance._providers_ttl.set(True)
    return router_instance


def _native_feature_body(case: str, *, model: str, stream: bool) -> dict:
    body: dict = {
        "model": model,
        "max_tokens": 100,
        "stream": stream,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    if case == "tools":
        body["tools"] = [
            {
                "name": "lookup",
                "description": "lookup",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
    elif case == "thinking":
        body["thinking"] = {"type": "adaptive"}
    elif case == "output-effort":
        body["output_config"] = {"effort": "low"}
    elif case == "vision":
        body["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                    }
                ],
            }
        ]
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unknown feature case {case}")
    return body


class TestBetaMessagesEndpoint:
    def _native_plan(self, provider, support, *, model="model"):
        ref = ModelRef("github-copilot", model)
        operation = Operation.NATIVE_ANTHROPIC
        features = RequestFeatures()
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: support},
        )
        return RoutePlan(
            operation=operation,
            features=features,
            primary=RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=support,
            ),
            fallbacks=(),
            explicit=True,
        )

    @pytest.mark.parametrize("path", ["legacy", "planned"])
    @pytest.mark.parametrize("guard", ["leak", "runaway"])
    @pytest.mark.parametrize("initially_enabled", [False, True])
    def test_native_stream_guards_use_frozen_enable_flags_on_both_paths(
        self,
        client,
        path,
        guard,
        initially_enabled,
    ):
        leak_enabled = guard == "leak" and initially_enabled
        runaway_enabled = guard == "runaway" and initially_enabled
        configs = [
            _guard_config(
                leak_enabled=leak_enabled,
                runaway_enabled=runaway_enabled,
            )
        ]
        provider = _native_provider()
        plan = self._native_plan(
            provider,
            CapabilitySupport.SUPPORTED,
            model="guarded-model",
        )
        resolution = _NativeModelResolution(
            _ResolvedModel("github-copilot", "guarded-model", provider),
            CapabilitySupport.SUPPORTED,
            plan if path == "planned" else None,
        )
        context = _CapturedGuardContext(configs)
        delta = (
            {"type": "text_delta", "text": "<tick>control</tick>"}
            if guard == "leak"
            else {"type": "text_delta", "text": "x" * 100_001}
        )

        async def selected_stream():
            yield f"event: message_start\ndata: {json.dumps(_VALID_MESSAGE_START)}\n\n"
            # Simulate a live config replacement after the request pipeline has
            # captured its initial snapshot. The active stream must not change.
            configs[0] = _guard_config(
                leak_enabled=guard == "leak" and not initially_enabled,
                runaway_enabled=guard == "runaway" and not initially_enabled,
            )
            payload = {"type": "content_block_delta", "index": 0, "delta": delta}
            yield f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n"
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

        patches = [
            patch(
                "router_maestro.server.routes.anthropic_beta._resolve_native_model",
                new_callable=AsyncMock,
                return_value=resolution,
            ),
            patch(
                "router_maestro.runtime.request_context.get_current_request_context",
                return_value=context,
            ),
        ]
        if path == "planned":
            patches.append(
                patch(
                    "router_maestro.server.routes.anthropic_beta.Router.execute_plan_stream",
                    new_callable=AsyncMock,
                    return_value=(selected_stream(), "github-copilot"),
                )
            )
        else:
            patches.append(
                patch(
                    "router_maestro.server.routes.anthropic_beta._stream_passthrough",
                    return_value=selected_stream(),
                )
            )

        with patches[0], patches[1], patches[2]:
            downstream = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "guarded-model",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        event_types = [
            line.removeprefix("event: ")
            for line in downstream.text.splitlines()
            if line.startswith("event: ")
        ]
        assert context.pipeline is not None
        assert context.pipeline.outcome is not None
        assert (context.pipeline.leak_guard is not None) is leak_enabled
        if initially_enabled:
            assert event_types == ["message_start", "error"]
            assert context.pipeline.outcome.transport is TransportTermination.EXCEPTION
            assert context.pipeline.outcome.response_status is ResponseStatus.FAILED
            assert context.pipeline.outcome.error.code == "overloaded"
        else:
            assert event_types == ["message_start", "content_block_delta", "message_stop"]
            assert context.pipeline.outcome.transport is TransportTermination.EXPLICIT_TERMINAL
            assert context.pipeline.outcome.response_status is ResponseStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_native_stream_with_both_guards_disabled_never_invokes_raw_leak_guard(self):
        from router_maestro.pipeline import RequestPipeline

        config = _guard_config(leak_enabled=False, runaway_enabled=False)
        context = _CapturedGuardContext([config])
        with patch(
            "router_maestro.runtime.request_context.get_current_request_context",
            return_value=context,
        ):
            pipeline = RequestPipeline.create(
                request_id="req-no-native-guards",
                model="guarded-model",
            )

        async def stream():
            payload = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "<tick>control</tick>" + "x" * 100_001,
                },
            }
            yield f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n"
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

        with patch(
            "router_maestro.pipeline.leak_guard.RawFrameLeakGuard",
            side_effect=AssertionError("disabled leak guard must not be constructed"),
        ):
            frames = [
                frame
                async for frame in _encode_native_stream_errors(
                    stream(),
                    pipeline=pipeline,
                )
            ]

        assert [
            line.removeprefix("event: ")
            for frame in frames
            for line in frame.splitlines()
            if line.startswith("event: ")
        ] == ["content_block_delta", "message_stop"]
        assert pipeline.outcome is not None
        assert pipeline.outcome.response_status is ResponseStatus.COMPLETED

    @pytest.mark.parametrize(
        "delta",
        [
            pytest.param(
                {"type": "thinking_delta", "thinking": "x" * 100_001},
                id="thinking-delta",
            ),
            pytest.param(
                {"type": "input_json_delta", "partial_json": "x" * 100_001},
                id="tool-input-delta",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_native_runaway_guard_counts_non_text_raw_deltas(self, delta):
        from router_maestro.pipeline import RequestPipeline

        config = _guard_config(leak_enabled=False, runaway_enabled=True)
        context = _CapturedGuardContext([config])
        with patch(
            "router_maestro.runtime.request_context.get_current_request_context",
            return_value=context,
        ):
            pipeline = RequestPipeline.create(
                request_id="req-raw-delta",
                model="guarded-model",
            )

        async def stream():
            payload = {"type": "content_block_delta", "index": 0, "delta": delta}
            yield f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n"
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

        frames = [
            frame
            async for frame in _encode_native_stream_errors(
                stream(),
                pipeline=pipeline,
            )
        ]

        assert [
            line.removeprefix("event: ")
            for frame in frames
            for line in frame.splitlines()
            if line.startswith("event: ")
        ] == ["error"]
        assert pipeline.outcome is not None
        assert pipeline.outcome.error.code == "overloaded"

    @pytest.mark.asyncio
    async def test_native_leak_guard_scans_raw_thinking_delta(self):
        from router_maestro.pipeline import RequestPipeline

        config = _guard_config(leak_enabled=True, runaway_enabled=False)
        context = _CapturedGuardContext([config])
        with patch(
            "router_maestro.runtime.request_context.get_current_request_context",
            return_value=context,
        ):
            pipeline = RequestPipeline.create(
                request_id="req-raw-thinking-leak",
                model="guarded-model",
            )

        async def stream():
            payload = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "<tick>private control</tick>",
                },
            }
            yield f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n"
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

        frames = [
            frame
            async for frame in _encode_native_stream_errors(
                stream(),
                pipeline=pipeline,
            )
        ]

        assert [
            line.removeprefix("event: ")
            for frame in frames
            for line in frame.splitlines()
            if line.startswith("event: ")
        ] == ["error"]
        assert pipeline.outcome is not None
        assert pipeline.outcome.error.code == "overloaded"

    @pytest.mark.parametrize(
        ("terminal", "expected_transport", "expected_status", "expected_code"),
        [
            pytest.param(
                "error",
                TransportTermination.EXPLICIT_TERMINAL,
                ResponseStatus.FAILED,
                "overloaded_error",
                id="upstream-error",
            ),
            pytest.param(
                "incomplete",
                TransportTermination.EXPLICIT_TERMINAL,
                ResponseStatus.INCOMPLETE,
                None,
                id="incomplete",
            ),
            pytest.param(
                "unexpected-eof",
                TransportTermination.UNEXPECTED_EOF,
                ResponseStatus.UNKNOWN,
                "unexpected_eof",
                id="unexpected-eof",
            ),
            pytest.param(
                "provider-error",
                TransportTermination.EXCEPTION,
                ResponseStatus.FAILED,
                "provider_error",
                id="postcommit-provider-error",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_native_stream_records_non_success_semantic_outcome(
        self,
        terminal,
        expected_transport,
        expected_status,
        expected_code,
    ):
        from router_maestro.pipeline import RequestPipeline

        pipeline = RequestPipeline(
            request_id="req-native-terminal",
            guards=[],
            leak_guard=None,
            audit=None,
            config=PrioritiesConfig(),
        )

        async def stream():
            yield f"event: message_start\ndata: {json.dumps(_VALID_MESSAGE_START)}\n\n"
            if terminal == "error":
                error = {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "retry"},
                }
                yield f"event: error\ndata: {json.dumps(error)}\n\n"
            elif terminal == "incomplete":
                delta = {
                    "type": "message_delta",
                    "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                    "usage": {"output_tokens": 3},
                }
                yield f"event: message_delta\ndata: {json.dumps(delta)}\n\n"
                yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            elif terminal == "provider-error":
                raise ProviderError(
                    "safe postcommit failure",
                    status_code=502,
                    kind=ProviderFailureKind.TRANSPORT,
                )

        frames = [
            frame
            async for frame in _encode_native_stream_errors(
                stream(),
                pipeline=pipeline,
            )
        ]

        assert frames
        if terminal == "error":
            assert sum(frame.count("event: error") for frame in frames) == 1
        assert pipeline.outcome is not None
        assert pipeline.outcome.transport is expected_transport
        assert pipeline.outcome.response_status is expected_status
        assert (pipeline.outcome.error.code if pipeline.outcome.error else None) == expected_code

    @pytest.mark.asyncio
    async def test_native_stream_records_cancelled_semantic_outcome(self):
        from router_maestro.pipeline import RequestPipeline

        pipeline = RequestPipeline(
            request_id="req-native-cancel",
            guards=[],
            leak_guard=None,
            audit=None,
            config=PrioritiesConfig(),
        )

        async def stream():
            yield f"event: message_start\ndata: {json.dumps(_VALID_MESSAGE_START)}\n\n"
            raise asyncio.CancelledError

        encoded = _encode_native_stream_errors(stream(), pipeline=pipeline)
        await anext(encoded)
        with pytest.raises(asyncio.CancelledError):
            await anext(encoded)

        assert pipeline.outcome is not None
        assert pipeline.outcome.transport is TransportTermination.CLIENT_CANCELLED
        assert pipeline.outcome.response_status is ResponseStatus.CANCELLED

    @pytest.mark.parametrize(
        ("terminal", "expected_transport", "expected_status"),
        [
            pytest.param(
                "error",
                TransportTermination.EXPLICIT_TERMINAL,
                ResponseStatus.FAILED,
                id="upstream-error",
            ),
            pytest.param(
                "incomplete",
                TransportTermination.EXPLICIT_TERMINAL,
                ResponseStatus.INCOMPLETE,
                id="incomplete",
            ),
            pytest.param(
                "unexpected-eof",
                TransportTermination.UNEXPECTED_EOF,
                ResponseStatus.UNKNOWN,
                id="unexpected-eof",
            ),
            pytest.param(
                "provider-error",
                TransportTermination.EXCEPTION,
                ResponseStatus.FAILED,
                id="postcommit-provider-error",
            ),
        ],
    )
    def test_native_stream_terminal_reaches_request_context(
        self,
        app,
        terminal,
        expected_transport,
        expected_status,
    ):
        from router_maestro.runtime import RequestContextMiddleware

        config = _guard_config(leak_enabled=False, runaway_enabled=False)
        snapshot = _GuardSnapshot(config)
        lease = _GuardLease(config)
        captured_contexts = []

        class Repository:
            def read(self):
                return snapshot

        class Owner:
            async def start(self, _snapshot):
                return None

            async def acquire(self):
                return lease

        app.state.runtime_config_repository = Repository()
        app.state.router_owner = Owner()
        app.add_middleware(RequestContextMiddleware)

        async def selected_stream():
            from router_maestro.runtime import current_request_context

            captured_contexts.append(current_request_context())
            yield f"event: message_start\ndata: {json.dumps(_VALID_MESSAGE_START)}\n\n"
            if terminal == "error":
                error = {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "retry"},
                }
                yield f"event: error\ndata: {json.dumps(error)}\n\n"
            elif terminal == "incomplete":
                delta = {
                    "type": "message_delta",
                    "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                    "usage": {"output_tokens": 3},
                }
                yield f"event: message_delta\ndata: {json.dumps(delta)}\n\n"
                yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            elif terminal == "provider-error":
                raise ProviderError(
                    "safe postcommit failure",
                    status_code=502,
                    kind=ProviderFailureKind.TRANSPORT,
                )

        provider = _native_provider()
        resolution = _NativeModelResolution(
            _ResolvedModel("github-copilot", "context-model", provider),
            CapabilitySupport.SUPPORTED,
            self._native_plan(
                provider,
                CapabilitySupport.SUPPORTED,
                model="context-model",
            ),
        )
        with (
            patch(
                "router_maestro.server.routes.anthropic_beta._resolve_native_model",
                new_callable=AsyncMock,
                return_value=resolution,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.Router.execute_plan_stream",
                new_callable=AsyncMock,
                return_value=(selected_stream(), "github-copilot"),
            ),
            TestClient(app) as context_client,
        ):
            downstream = context_client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "context-model",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert downstream.status_code == 200
        assert len(captured_contexts) == 1
        assert captured_contexts[0].outcome is not None
        assert captured_contexts[0].outcome.transport is expected_transport
        assert captured_contexts[0].outcome.response_status is expected_status
        assert captured_contexts[0].pipeline is not None
        assert captured_contexts[0].pipeline.outcome == captured_contexts[0].outcome
        assert lease.release_count == 1

    @pytest.mark.parametrize("path", ["planned", "legacy"])
    @pytest.mark.parametrize(
        ("terminal", "expected_transport", "expected_status", "expected_code"),
        [
            pytest.param(
                "clean-eof",
                TransportTermination.UNEXPECTED_EOF,
                ResponseStatus.UNKNOWN,
                "unexpected_eof",
                id="clean-eof",
            ),
            pytest.param(
                "provider-error",
                TransportTermination.EXCEPTION,
                ResponseStatus.FAILED,
                "provider_error",
                id="provider-error",
            ),
            pytest.param(
                "native-error",
                TransportTermination.EXPLICIT_TERMINAL,
                ResponseStatus.FAILED,
                "overloaded_error",
                id="native-error",
            ),
        ],
    )
    def test_native_transport_terminal_reaches_canonical_context(
        self,
        app,
        path,
        terminal,
        expected_transport,
        expected_status,
        expected_code,
    ):
        from router_maestro.runtime import RequestContextMiddleware, current_request_context

        config = _guard_config(leak_enabled=False, runaway_enabled=False)
        snapshot = _GuardSnapshot(config)
        lease = _GuardLease(config)
        captured_contexts = []

        class Repository:
            def read(self):
                return snapshot

        class Owner:
            async def start(self, _snapshot):
                return None

            async def acquire(self):
                return lease

        class StreamResponse:
            status_code = 200

            async def aiter_lines(self):
                captured_contexts.append(current_request_context())
                lines = [
                    "event: message_start",
                    f"data: {json.dumps(_VALID_MESSAGE_START)}",
                    "",
                ]
                if terminal == "native-error":
                    error = {
                        "type": "error",
                        "error": {"type": "overloaded_error", "message": "retry"},
                    }
                    lines.extend(("event: error", f"data: {json.dumps(error)}", ""))
                else:
                    lines.extend(
                        (
                            "event: content_block_start",
                            'data: {"type":"content_block_start","index":0,'
                            '"content_block":{"type":"text","text":""}}',
                            "",
                            "event: content_block_delta",
                            'data: {"type":"content_block_delta","index":0,'
                            '"delta":{"type":"text_delta","text":"partial"}}',
                            "",
                        )
                    )
                for line in lines:
                    yield line
                if terminal == "provider-error":
                    raise ProviderError(
                        "safe upstream transport failure",
                        status_code=502,
                        kind=ProviderFailureKind.TRANSPORT,
                    )

        @asynccontextmanager
        async def stream_context():
            yield StreamResponse()

        provider = _native_provider()
        provider._stream_with_auth_retry = MagicMock(return_value=stream_context())
        plan = self._native_plan(
            provider,
            CapabilitySupport.SUPPORTED,
            model="clean-eof-context",
        )
        resolution = _NativeModelResolution(
            _ResolvedModel("github-copilot", "clean-eof-context", provider),
            CapabilitySupport.SUPPORTED,
            plan if path == "planned" else None,
        )
        app.state.runtime_config_repository = Repository()
        app.state.router_owner = Owner()
        app.add_middleware(RequestContextMiddleware)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta._resolve_native_model",
                new_callable=AsyncMock,
                return_value=resolution,
            ),
            TestClient(app) as context_client,
        ):
            downstream = context_client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "clean-eof-context",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert downstream.status_code == 200
        assert downstream.text.count("event: error") == 1
        assert len(captured_contexts) == 1
        context = captured_contexts[0]
        assert context.outcome is not None
        assert context.outcome.transport is expected_transport
        assert context.outcome.response_status is expected_status
        assert context.outcome.error.code == expected_code
        assert context.pipeline is not None
        assert context.pipeline.outcome == context.outcome
        assert context.pipeline.wire_status == 200
        assert lease.release_count == 1

    @pytest.mark.asyncio
    async def test_native_stream_disconnect_reaches_request_context_as_cancelled(self, app):
        from router_maestro.runtime import RequestContextMiddleware

        config = _guard_config(leak_enabled=False, runaway_enabled=False)
        snapshot = _GuardSnapshot(config)
        lease = _GuardLease(config)
        captured_contexts = []

        class Repository:
            def read(self):
                return snapshot

        class Owner:
            async def start(self, _snapshot):
                return None

            async def acquire(self):
                return lease

        app.state.runtime_config_repository = Repository()
        app.state.router_owner = Owner()
        app.add_middleware(RequestContextMiddleware)

        async def selected_stream():
            from router_maestro.runtime import current_request_context

            captured_contexts.append(current_request_context())
            yield f"event: message_start\ndata: {json.dumps(_VALID_MESSAGE_START)}\n\n"
            await asyncio.Event().wait()

        provider = _native_provider()
        resolution = _NativeModelResolution(
            _ResolvedModel("github-copilot", "context-model", provider),
            CapabilitySupport.SUPPORTED,
            self._native_plan(
                provider,
                CapabilitySupport.SUPPORTED,
                model="context-model",
            ),
        )
        with (
            patch(
                "router_maestro.server.routes.anthropic_beta._resolve_native_model",
                new_callable=AsyncMock,
                return_value=resolution,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.Router.execute_plan_stream",
                new_callable=AsyncMock,
                return_value=(selected_stream(), "github-copilot"),
            ),
        ):
            scope = {
                "type": "http",
                "asgi": {"spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/anthropic/beta/v1/messages",
                "raw_path": b"/api/anthropic/beta/v1/messages",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("test", 1),
                "server": ("test", 80),
                "state": {"request_id": "req-beta-cancel"},
                "app": app,
            }
            request_body = json.dumps(
                {
                    "model": "context-model",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                }
            ).encode()
            received_request = False

            async def receive():
                nonlocal received_request
                if not received_request:
                    received_request = True
                    return {
                        "type": "http.request",
                        "body": request_body,
                        "more_body": False,
                    }
                return {"type": "http.disconnect"}

            async def send(_message):
                return None

            await app(scope, receive, send)

        assert len(captured_contexts) == 1
        assert captured_contexts[0].outcome is not None
        assert captured_contexts[0].outcome.transport is TransportTermination.CLIENT_CANCELLED
        assert captured_contexts[0].outcome.response_status is ResponseStatus.CANCELLED
        assert captured_contexts[0].pipeline is not None
        assert captured_contexts[0].pipeline.outcome == captured_contexts[0].outcome
        assert lease.release_count == 1

    @pytest.mark.parametrize("transport", ["native", "translated"])
    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    @pytest.mark.parametrize(
        ("option_payload", "parameter"),
        [
            ({"context_management": {"enabled": True}}, "context_management"),
            (
                {"output_config": {"effort": "low", "format": "text"}},
                "output_config.format",
            ),
            ({"service_tier": "standard_only"}, "service_tier"),
        ],
    )
    def test_beta_rejects_unknown_semantic_options_before_transport_selection(
        self,
        client,
        transport,
        stream,
        option_payload,
        parameter,
    ):
        provider = _native_provider()
        support = (
            CapabilitySupport.SUPPORTED if transport == "native" else CapabilitySupport.UNSUPPORTED
        )
        resolution = _NativeModelResolution(
            _ResolvedModel(
                "github-copilot",
                "claude-options" if transport == "native" else "gpt-5.5",
                provider,
            ),
            support,
        )
        body = {
            "model": "github-copilot/model",
            "max_tokens": 100,
            "stream": stream,
            "messages": [{"role": "user", "content": "Hi"}],
            **option_payload,
        }

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta._resolve_native_model",
                new_callable=AsyncMock,
                return_value=resolution,
            ) as resolve_native,
            patch(
                "router_maestro.server.routes.anthropic_beta._parse_as_anthropic_request",
                new_callable=AsyncMock,
            ) as parse_request,
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
                return_value=JSONResponse(content={"path": "standard"}),
            ) as standard,
            patch(
                "router_maestro.server.routes.anthropic_beta.sse_streaming_response",
                return_value=JSONResponse(content={"path": "stream"}),
            ),
        ):
            response = client.post("/api/anthropic/beta/v1/messages", json=body)

        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert parameter in response.json()["error"]["message"]
        assert "event:" not in response.text
        resolve_native.assert_not_awaited()
        parse_request.assert_not_awaited()
        standard.assert_not_awaited()
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()
        provider._stream_with_auth_retry.assert_not_called()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    @pytest.mark.parametrize(
        ("option_payload", "parameter"),
        [
            ({"output_config": "low"}, "output_config"),
            ({"output_config": {}}, "output_config.effort"),
            ({"output_config": {"effort": "invalid"}}, "output_config.effort"),
            (
                {"output_config": {"effort": "low", "format": "json"}},
                "output_config.format",
            ),
            ({"temperature": 0.2, "top_p": 0.8}, "top_p"),
            ({"service_tier": "standard_only"}, "service_tier"),
        ],
    )
    def test_native_rejects_unpreservable_options_before_transport(
        self,
        client,
        stream,
        option_payload,
        parameter,
    ):
        provider = _native_provider()
        model = ModelInfo(
            id="claude-options",
            name="Claude options",
            provider="github-copilot",
            operation_capabilities={Operation.NATIVE_ANTHROPIC: True},
        )
        model_router = _real_native_router(
            provider,
            [model],
            ["github-copilot/claude-options"],
        )
        body = {
            "model": "github-copilot/claude-options",
            "max_tokens": 100,
            "stream": stream,
            "messages": [{"role": "user", "content": "Hi"}],
            **option_payload,
        }

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            response = client.post("/api/anthropic/beta/v1/messages", json=body)

        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        assert parameter in response.json()["error"]["message"]
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()
        provider._stream_with_auth_retry.assert_not_called()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    def test_native_rejects_reasoning_tier_when_only_higher_tiers_exist(
        self,
        client,
        stream,
    ):
        provider = _native_provider()
        model = ModelInfo(
            id="claude-higher-only",
            name="Claude higher only",
            provider="github-copilot",
            operation_capabilities={Operation.NATIVE_ANTHROPIC: True},
            reasoning_effort_values=["medium", "high"],
        )
        model_router = _real_native_router(
            provider,
            [model],
            ["github-copilot/claude-higher-only"],
        )

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "github-copilot/claude-higher-only",
                    "max_tokens": 100,
                    "stream": stream,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "output_config": {"effort": "low"},
                },
            )

        assert response.status_code == 400
        assert "output_config.effort" in response.json()["error"]["message"]
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()
        provider._stream_with_auth_retry.assert_not_called()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    def test_native_effort_prevalidation_does_not_spend_retry_slot_on_incompatible_fallback(
        self,
        client,
        stream,
    ):
        from router_maestro.routing.router import Router

        def candidate(provider, model: str, efforts: tuple[str, ...]) -> RouteCandidate:
            ref = ModelRef("github-copilot", model)
            operation = Operation.NATIVE_ANTHROPIC
            features = RequestFeatures(reasoning=True)
            capabilities = ModelCapabilities(
                model=ref,
                operations={
                    operation: CapabilitySupport.SUPPORTED,
                },
                features={Feature.REASONING: CapabilitySupport.SUPPORTED},
                reasoning_effort_values=efforts,
            )
            return RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            )

        primary = _native_provider()
        rejected = _native_provider()
        compatible = _native_provider()
        primary_candidate = candidate(primary, "claude-primary", ("low", "medium"))
        rejected_candidate = candidate(rejected, "claude-higher-only", ("medium", "high"))
        compatible_candidate = candidate(compatible, "claude-compatible", ("low", "medium"))
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(reasoning=True),
            primary=primary_candidate,
            fallbacks=(rejected_candidate,),
            explicit=False,
            fallback_pool=(rejected_candidate, compatible_candidate),
            max_fallback_attempts=1,
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        if stream:

            class _StreamResponse:
                def __init__(self, status_code: int, lines: list[str]) -> None:
                    self.status_code = status_code
                    self._lines = lines

                async def aread(self) -> bytes:
                    return b'{"error":{"message":"retryable"}}'

                async def aiter_lines(self):
                    for line in self._lines:
                        yield line

            def context(response):
                @asynccontextmanager
                async def manager():
                    yield response

                return manager()

            primary._stream_with_auth_retry = MagicMock(
                return_value=context(_StreamResponse(503, []))
            )
            rejected._stream_with_auth_retry = MagicMock(
                side_effect=AssertionError("incompatible fallback must not open")
            )
            compatible._stream_with_auth_retry = MagicMock(
                return_value=context(
                    _StreamResponse(
                        200,
                        [
                            "event: message_start",
                            f"data: {json.dumps(_VALID_MESSAGE_START)}",
                            "",
                        ],
                    )
                )
            )
        else:
            primary._send_with_auth_retry.side_effect = ProviderError(
                "primary retryable",
                status_code=503,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_STATUS,
            )

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "stream": stream,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "output_config": {"effort": "low"},
                },
            )

        assert response.status_code == 200
        rejected.ensure_token.assert_not_awaited()
        rejected._send_with_auth_retry.assert_not_awaited()
        if stream:
            rejected._stream_with_auth_retry.assert_not_called()
            compatible._stream_with_auth_retry.assert_called_once()
        else:
            compatible._send_with_auth_retry.assert_awaited_once()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    def test_native_primary_effort_error_does_not_switch_to_compatible_fallback(
        self,
        client,
        stream,
    ):
        from router_maestro.routing.router import Router

        primary = _native_provider()
        fallback = _native_provider()

        def candidate(provider, model: str, efforts: tuple[str, ...]) -> RouteCandidate:
            ref = ModelRef("github-copilot", model)
            operation = Operation.NATIVE_ANTHROPIC
            features = RequestFeatures(reasoning=True)
            capabilities = ModelCapabilities(
                model=ref,
                operations={
                    operation: CapabilitySupport.SUPPORTED,
                },
                features={Feature.REASONING: CapabilitySupport.SUPPORTED},
                reasoning_effort_values=efforts,
            )
            return RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            )

        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(reasoning=True),
            primary=candidate(primary, "claude-higher-only", ("medium", "high")),
            fallbacks=(candidate(fallback, "claude-compatible", ("low", "medium")),),
            explicit=False,
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "stream": stream,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "output_config": {"effort": "low"},
                },
            )

        assert response.status_code == 400
        assert "output_config.effort" in response.json()["error"]["message"]
        primary.ensure_token.assert_not_awaited()
        fallback.ensure_token.assert_not_awaited()
        primary._send_with_auth_retry.assert_not_awaited()
        fallback._send_with_auth_retry.assert_not_awaited()
        primary._stream_with_auth_retry.assert_not_called()
        fallback._stream_with_auth_retry.assert_not_called()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    @pytest.mark.parametrize(
        ("case", "feature"),
        [
            ("tools", Feature.TOOLS),
            ("thinking", Feature.REASONING),
            ("output-effort", Feature.REASONING),
            ("vision", Feature.VISION),
        ],
    )
    def test_real_native_auto_planning_selects_feature_compatible_candidate(
        self,
        client,
        stream,
        case,
        feature,
    ):
        provider = _native_provider()
        first = ModelInfo(
            id="claude-incompatible",
            name="Claude incompatible",
            provider="github-copilot",
            operation_capabilities={Operation.NATIVE_ANTHROPIC: True},
            feature_capabilities={feature: False},
        )
        second = ModelInfo(
            id="claude-compatible",
            name="Claude compatible",
            provider="github-copilot",
            operation_capabilities={Operation.NATIVE_ANTHROPIC: True},
            feature_capabilities={feature: True},
        )
        model_router = _real_native_router(
            provider,
            [first, second],
            ["github-copilot/claude-incompatible", "github-copilot/claude-compatible"],
        )

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.sse_streaming_response",
                return_value=JSONResponse(content={"path": "stream"}),
            )
            if stream
            else nullcontext(),
            patch(
                "router_maestro.server.routes.anthropic_beta.Router.execute_plan_stream",
                new_callable=AsyncMock,
                return_value=(object(), "github-copilot"),
            ) as execute_stream,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json=_native_feature_body(case, model="router-maestro", stream=stream),
            )

        assert response.status_code == 200
        if stream:
            selected_plan = execute_stream.await_args.args[0]
            assert selected_plan.primary.model.upstream_id == "claude-compatible"
            provider._send_with_auth_retry.assert_not_awaited()
        else:
            forwarded = provider._send_with_auth_retry.await_args.kwargs
            assert forwarded["model"] == "claude-compatible"
            assert forwarded["json"]["model"] == "claude-compatible"

    @pytest.mark.parametrize(
        ("status_code", "kind", "expected_type"),
        [
            (400, ProviderFailureKind.CLIENT_REQUEST, "invalid_request_error"),
            (401, ProviderFailureKind.AUTHENTICATION, "authentication_error"),
            (429, ProviderFailureKind.RATE_LIMIT, "rate_limit_error"),
            (529, ProviderFailureKind.RATE_LIMIT, "overloaded_error"),
        ],
    )
    def test_planned_native_stream_open_failure_uses_typed_anthropic_json(
        self,
        client,
        status_code,
        kind,
        expected_type,
    ):
        from router_maestro.routing.router import Router

        provider = _native_provider()
        plan = self._native_plan(
            provider,
            CapabilitySupport.SUPPORTED,
            model="typed-open-failure",
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)
        error = ProviderError(
            "Safe planned stream failure",
            status_code=status_code,
            retryable=status_code in {429, 529},
            kind=kind,
        )

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.Router.execute_plan_stream",
                new_callable=AsyncMock,
                side_effect=error,
            ),
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == status_code
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {
            "type": "error",
            "error": {
                "type": expected_type,
                "message": "Safe planned stream failure",
            },
        }

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    @pytest.mark.parametrize(
        ("case", "feature"),
        [
            ("tools", Feature.TOOLS),
            ("thinking", Feature.REASONING),
            ("output-effort", Feature.REASONING),
            ("vision", Feature.VISION),
        ],
    )
    def test_real_native_explicit_feature_unsupported_returns_400_before_transport(
        self,
        client,
        stream,
        case,
        feature,
    ):
        provider = _native_provider()
        model = ModelInfo(
            id="claude-incompatible",
            name="Claude incompatible",
            provider="github-copilot",
            operation_capabilities={Operation.NATIVE_ANTHROPIC: True},
            feature_capabilities={feature: False},
        )
        model_router = _real_native_router(
            provider,
            [model],
            ["github-copilot/claude-incompatible"],
        )

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
            ) as standard,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json=_native_feature_body(
                    case,
                    model="github-copilot/claude-incompatible",
                    stream=stream,
                ),
            )

        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["type"] == "invalid_request_error"
        standard.assert_not_awaited()
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()

    def test_native_nonstream_consumes_frozen_plan_fallback_after_retryable_primary(
        self,
        client,
    ):
        primary = _native_provider()
        primary._send_with_auth_retry.side_effect = ProviderError(
            "primary transport failed",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.TRANSPORT,
        )
        secondary = _native_provider()
        primary_plan = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="claude-primary",
        )
        secondary_plan = self._native_plan(
            secondary,
            CapabilitySupport.SUPPORTED,
            model="claude-secondary",
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=primary_plan.primary,
            fallbacks=(secondary_plan.primary,),
            explicit=False,
        )
        from router_maestro.routing.router import Router

        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        primary._send_with_auth_retry.assert_awaited_once()
        secondary._send_with_auth_retry.assert_awaited_once()
        assert secondary._send_with_auth_retry.await_args.kwargs["model"] == "claude-secondary"
        assert secondary._send_with_auth_retry.await_args.kwargs["json"]["model"] == (
            "claude-secondary"
        )

    @pytest.mark.parametrize(
        ("candidate_upstream_model", "response_upstream_model", "expected_public_model"),
        [
            ("shared-model", "shared-model", "second/shared-model"),
            ("shared-model", "second/shared-model", "second/shared-model"),
            (
                "second/team/model",
                "second/team/model",
                "second/second/team/model",
            ),
        ],
    )
    def test_native_nonstream_response_uses_actual_fallback_provider_once(
        self,
        client,
        candidate_upstream_model,
        response_upstream_model,
        expected_public_model,
    ):
        def candidate(provider, provider_name: str, upstream_model: str) -> RouteCandidate:
            ref = ModelRef(provider_name, upstream_model)
            operation = Operation.NATIVE_ANTHROPIC
            features = RequestFeatures()
            capabilities = ModelCapabilities(
                model=ref,
                operations={operation: CapabilitySupport.SUPPORTED},
            )
            return RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            )

        primary = _native_provider()
        primary.name = "first"
        primary._send_with_auth_retry.side_effect = ProviderError(
            "primary transport failed",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.TRANSPORT,
        )
        secondary = _native_provider()
        secondary.name = "second"
        secondary_payload = _valid_native_message_response()
        secondary_payload["model"] = response_upstream_model
        secondary._send_with_auth_retry.return_value.json.return_value = secondary_payload
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=candidate(primary, "first", "shared-model"),
            fallbacks=(candidate(secondary, "second", candidate_upstream_model),),
            explicit=False,
        )
        model_router = MagicMock(_models_cache={})
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        assert response.json()["model"] == expected_public_model
        primary._send_with_auth_retry.assert_awaited_once()
        secondary._send_with_auth_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_native_nonstream_signature_retry_does_not_mutate_fallback_payload(self):
        from router_maestro.routing.router import Router

        body = {
            "model": "original",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private", "signature": "bad"},
                        {"type": "text", "text": "answer"},
                    ],
                }
            ],
        }
        signature_response = MagicMock(
            status_code=400,
            text='{"message":"Invalid signature in thinking block"}',
        )
        primary = _native_provider()
        primary_payloads: list[dict] = []
        primary_nested_ids: list[tuple[int, int, int]] = []

        async def primary_send(*_args, **kwargs):
            payload = kwargs["json"]
            primary_payloads.append(deepcopy(payload))
            primary_nested_ids.append(
                (
                    id(payload["messages"]),
                    id(payload["messages"][0]),
                    id(payload["messages"][0]["content"]),
                )
            )
            if len(primary_payloads) == 1:
                return signature_response
            raise ProviderError(
                "retryable after signature retry",
                status_code=503,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_STATUS,
            )

        primary._send_with_auth_retry.side_effect = primary_send
        secondary = _native_provider()
        secondary_payloads: list[dict] = []
        secondary_nested_ids: list[tuple[int, int, int]] = []
        secondary_response = secondary._send_with_auth_retry.return_value

        async def secondary_send(*_args, **kwargs):
            payload = kwargs["json"]
            secondary_payloads.append(deepcopy(payload))
            secondary_nested_ids.append(
                (
                    id(payload["messages"]),
                    id(payload["messages"][0]),
                    id(payload["messages"][0]["content"]),
                )
            )
            return secondary_response

        secondary._send_with_auth_retry.side_effect = secondary_send
        primary_candidate = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="primary",
        ).primary
        secondary_candidate = self._native_plan(
            secondary,
            CapabilitySupport.SUPPORTED,
            model="secondary",
        ).primary
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=primary_candidate,
            fallbacks=(secondary_candidate,),
            explicit=False,
        )

        await Router.execute_plan_nonstream(
            plan,
            lambda candidate: _send_native_nonstream(candidate, body),
        )

        assert [block["type"] for block in primary_payloads[0]["messages"][0]["content"]] == [
            "thinking",
            "text",
        ]
        assert [block["type"] for block in primary_payloads[1]["messages"][0]["content"]] == [
            "text"
        ]
        assert [block["type"] for block in secondary_payloads[0]["messages"][0]["content"]] == [
            "thinking",
            "text",
        ]
        assert primary_nested_ids[0] != secondary_nested_ids[0]
        assert body["messages"][0]["content"][0]["type"] == "thinking"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_response",
        [
            pytest.param("malformed-json", id="malformed-json"),
            pytest.param("invalid-shape", id="invalid-shape"),
        ],
    )
    async def test_native_nonstream_malformed_2xx_falls_back_inside_attempt(
        self,
        invalid_response,
    ):
        from router_maestro.routing.router import Router

        primary = _native_provider()
        primary_response = MagicMock(status_code=200)
        if invalid_response == "malformed-json":
            primary_response.json.side_effect = json.JSONDecodeError(
                "invalid upstream json",
                "private-upstream-body",
                0,
            )
        else:
            primary_response.json.return_value = {
                "type": "message",
                "role": "assistant",
                "content": "not-a-list",
            }
        primary._send_with_auth_retry.return_value = primary_response
        secondary = _native_provider()
        secondary_response = secondary._send_with_auth_retry.return_value
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="malformed-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="valid-secondary",
                ).primary,
            ),
            explicit=False,
        )

        result, provider_name = await Router.execute_plan_nonstream(
            plan,
            lambda candidate: _send_native_nonstream(candidate, {"model": "original"}),
        )

        response, data = result
        assert response is secondary_response
        assert data["type"] == "message"
        assert provider_name == "github-copilot"
        primary._send_with_auth_retry.assert_awaited_once()
        secondary._send_with_auth_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_native_nonstream_deep_malformed_2xx_falls_back_inside_attempt(self):
        from router_maestro.routing.router import Router

        primary = _native_provider()
        primary_response = MagicMock(status_code=200)
        primary_payload = _valid_native_message_response()
        primary_payload["content"] = [{"type": "text", "text": 7}]
        primary_response.json.return_value = primary_payload
        primary._send_with_auth_retry.return_value = primary_response
        secondary = _native_provider()
        secondary_response = secondary._send_with_auth_retry.return_value
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="malformed-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="valid-secondary",
                ).primary,
            ),
            explicit=False,
        )

        result, provider_name = await Router.execute_plan_nonstream(
            plan,
            lambda candidate: _send_native_nonstream(candidate, {"model": "original"}),
        )

        response, data = result
        assert response is secondary_response
        assert data["content"] == [{"type": "text", "text": "ok"}]
        assert provider_name == "github-copilot"
        primary._send_with_auth_retry.assert_awaited_once()
        secondary._send_with_auth_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_native_stream_signature_retry_does_not_mutate_fallback_payload(self):
        from router_maestro.routing.router import Router

        body = {
            "model": "original",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private", "signature": "bad"},
                        {"type": "text", "text": "answer"},
                    ],
                }
            ],
        }
        seen_payloads: dict[str, list[dict]] = {"primary": [], "secondary": []}

        class _StreamResponse:
            def __init__(self, status_code: int, *, signature: bool = False) -> None:
                self.status_code = status_code
                self.signature = signature

            async def aread(self) -> bytes:
                if self.signature:
                    return b'{"message":"Invalid signature in thinking block"}'
                return b'{"error":{"message":"retryable"}}'

            async def aiter_lines(self):
                yield "event: message_start"
                yield f"data: {json.dumps(_VALID_MESSAGE_START)}"
                yield ""

        def context(response):
            @asynccontextmanager
            async def manager():
                yield response

            return manager()

        primary = _native_provider()
        primary.name = "github-copilot"

        def primary_open(*_args, **kwargs):
            seen_payloads["primary"].append(deepcopy(kwargs["json"]))
            response = (
                _StreamResponse(400, signature=True)
                if len(seen_payloads["primary"]) == 1
                else _StreamResponse(503)
            )
            return context(response)

        primary._stream_with_auth_retry = MagicMock(side_effect=primary_open)
        secondary = _native_provider()
        secondary.name = "github-copilot"

        def secondary_open(*_args, **kwargs):
            seen_payloads["secondary"].append(deepcopy(kwargs["json"]))
            return context(_StreamResponse(200))

        secondary._stream_with_auth_retry = MagicMock(side_effect=secondary_open)
        primary_candidate = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="primary",
        ).primary
        secondary_candidate = self._native_plan(
            secondary,
            CapabilitySupport.SUPPORTED,
            model="secondary",
        ).primary
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=primary_candidate,
            fallbacks=(secondary_candidate,),
            explicit=False,
        )

        stream, _provider_name = await Router.execute_plan_stream(
            plan,
            lambda candidate: _stream_native_candidate(candidate, body),
        )
        await anext(stream)

        assert [
            block["type"] for block in seen_payloads["primary"][1]["messages"][0]["content"]
        ] == ["text"]
        assert [
            block["type"] for block in seen_payloads["secondary"][0]["messages"][0]["content"]
        ] == ["thinking", "text"]
        assert body["messages"][0]["content"][0]["type"] == "thinking"

    @pytest.mark.asyncio
    async def test_native_nonstream_normalizes_each_candidate_model(self):
        from router_maestro.routing.router import Router

        primary = _native_provider()
        primary._send_with_auth_retry.side_effect = ProviderError(
            "primary retryable",
            status_code=503,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_STATUS,
        )
        secondary = _native_provider()
        primary_candidate = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="primary-model",
        ).primary
        secondary_candidate = self._native_plan(
            secondary,
            CapabilitySupport.SUPPORTED,
            model="secondary-model",
        ).primary
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=primary_candidate,
            fallbacks=(secondary_candidate,),
            explicit=False,
        )
        normalized_models: list[str] = []

        def normalize(payload: dict, model: str, _efforts=None) -> dict:
            normalized_models.append(model)
            payload["normalization_marker"] = model
            return payload

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=normalize,
        ):
            await Router.execute_plan_nonstream(
                plan,
                lambda candidate: _send_native_nonstream(candidate, {"model": "original"}),
            )

        assert normalized_models == ["primary-model", "secondary-model"]
        assert (
            primary._send_with_auth_retry.await_args.kwargs["json"]["normalization_marker"]
            == "primary-model"
        )
        assert (
            secondary._send_with_auth_retry.await_args.kwargs["json"]["normalization_marker"]
            == "secondary-model"
        )

    @pytest.mark.asyncio
    async def test_native_stream_normalizes_each_candidate_model(self):
        from router_maestro.routing.router import Router

        class _StreamResponse:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code

            async def aread(self) -> bytes:
                return b"{}"

            async def aiter_lines(self):
                yield "event: message_start"
                yield f"data: {json.dumps(_VALID_MESSAGE_START)}"
                yield ""

        def context(response):
            @asynccontextmanager
            async def manager():
                yield response

            return manager()

        primary = _native_provider()
        primary.name = "github-copilot"
        primary._stream_with_auth_retry = MagicMock(return_value=context(_StreamResponse(503)))
        secondary = _native_provider()
        secondary.name = "github-copilot"
        secondary._stream_with_auth_retry = MagicMock(return_value=context(_StreamResponse(200)))
        primary_candidate = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="primary-model",
        ).primary
        secondary_candidate = self._native_plan(
            secondary,
            CapabilitySupport.SUPPORTED,
            model="secondary-model",
        ).primary
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=primary_candidate,
            fallbacks=(secondary_candidate,),
            explicit=False,
        )
        normalized_models: list[str] = []

        def normalize(payload: dict, model: str, _efforts=None) -> dict:
            normalized_models.append(model)
            payload["normalization_marker"] = model
            return payload

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=normalize,
        ):
            stream, _provider_name = await Router.execute_plan_stream(
                plan,
                lambda candidate: _stream_native_candidate(candidate, {"model": "original"}),
            )
            await anext(stream)

        assert normalized_models == ["primary-model", "secondary-model"]
        assert (
            primary._stream_with_auth_retry.call_args.kwargs["json"]["normalization_marker"]
            == "primary-model"
        )
        assert (
            secondary._stream_with_auth_retry.call_args.kwargs["json"]["normalization_marker"]
            == "secondary-model"
        )

    @pytest.mark.asyncio
    async def test_native_nonstream_secondary_fatal_status_stops_before_tertiary(self):
        from router_maestro.routing.router import Router

        primary = _native_provider()
        primary._send_with_auth_retry.side_effect = ProviderError(
            "primary transport failed",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.TRANSPORT,
        )
        secondary = _native_provider()
        secondary_response = MagicMock()
        secondary_response.status_code = 400
        secondary_response.text = '{"error":{"message":"private upstream body"}}'
        secondary_response.json.return_value = {
            "error": {"type": "invalid_request_error", "message": "bad request"}
        }
        secondary._send_with_auth_retry.return_value = secondary_response
        tertiary = _native_provider()
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="secondary",
                ).primary,
                self._native_plan(
                    tertiary,
                    CapabilitySupport.SUPPORTED,
                    model="tertiary",
                ).primary,
            ),
            explicit=False,
        )

        with pytest.raises(ProviderError) as exc_info:
            await Router.execute_plan_nonstream(
                plan,
                lambda candidate: _send_native_nonstream(candidate, {"model": "original"}),
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_STATUS
        assert exc_info.value.retryable is False
        assert [attempt.model.upstream_id for attempt in exc_info.value.attempts] == [
            "primary",
            "secondary",
        ]
        tertiary._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_native_nonstream_all_retryable_statuses_raise_last_with_attempts(self):
        from router_maestro.routing.router import Router

        providers = [_native_provider(), _native_provider()]
        for provider, status in zip(providers, (503, 502), strict=True):
            response = MagicMock()
            response.status_code = status
            response.text = '{"error":{"message":"private upstream body"}}'
            provider._send_with_auth_retry.return_value = response
        candidates = [
            self._native_plan(
                provider,
                CapabilitySupport.SUPPORTED,
                model=model,
            ).primary
            for provider, model in zip(providers, ("primary", "secondary"), strict=True)
        ]
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=candidates[0],
            fallbacks=(candidates[1],),
            explicit=False,
        )

        with pytest.raises(ProviderError) as exc_info:
            await Router.execute_plan_nonstream(
                plan,
                lambda candidate: _send_native_nonstream(candidate, {"model": "original"}),
            )

        assert exc_info.value.status_code == 502
        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_STATUS
        assert exc_info.value.retryable is True
        assert [attempt.model.upstream_id for attempt in exc_info.value.attempts] == [
            "primary",
            "secondary",
        ]
        assert [attempt.downstream_status_code for attempt in exc_info.value.attempts] == [503, 502]

    @pytest.mark.asyncio
    async def test_native_auto_unknown_non_copilot_runtime_unsupported_advances(self):
        from router_maestro.routing.router import Router

        unsupported_provider = MagicMock()
        unsupported_provider.name = "github-copilot"
        unsupported_plan = self._native_plan(
            unsupported_provider,
            CapabilitySupport.UNKNOWN,
            model="compatibility-unknown",
        )
        fallback = _native_provider()
        fallback_plan = self._native_plan(
            fallback,
            CapabilitySupport.SUPPORTED,
            model="native-supported",
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=unsupported_plan.primary,
            fallbacks=(fallback_plan.primary,),
            explicit=False,
        )

        result, provider_name = await Router.execute_plan_nonstream(
            plan,
            lambda candidate: _send_native_nonstream(candidate, {"model": "original"}),
        )

        response, _data = result
        assert response.status_code == 200
        assert provider_name == "github-copilot"
        fallback._send_with_auth_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_native_explicit_unknown_runtime_unsupported_does_not_switch(self):
        from router_maestro.routing.router import Router

        unsupported_provider = MagicMock()
        unsupported_provider.name = "github-copilot"
        unsupported_plan = self._native_plan(
            unsupported_provider,
            CapabilitySupport.UNKNOWN,
            model="explicit-unknown",
        )
        fallback = _native_provider()
        fallback_plan = self._native_plan(
            fallback,
            CapabilitySupport.SUPPORTED,
            model="must-not-run",
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=unsupported_plan.primary,
            fallbacks=(fallback_plan.primary,),
            explicit=True,
        )

        with pytest.raises(ProviderError) as exc_info:
            await Router.execute_plan_nonstream(
                plan,
                lambda candidate: _send_native_nonstream(candidate, {"model": "original"}),
            )

        assert exc_info.value.kind is ProviderFailureKind.UNSUPPORTED_OPERATION
        assert [attempt.model.upstream_id for attempt in exc_info.value.attempts] == [
            "explicit-unknown"
        ]
        fallback._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_native_stream_pre_first_retryable_failure_selects_secondary_frame(self):
        from router_maestro.routing.router import Router

        primary = _native_provider()
        secondary = _native_provider()
        primary_plan = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="primary-stream",
        )
        secondary_plan = self._native_plan(
            secondary,
            CapabilitySupport.SUPPORTED,
            model="secondary-stream",
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=primary_plan.primary,
            fallbacks=(secondary_plan.primary,),
            explicit=False,
        )
        calls: list[str] = []

        async def open_candidate(candidate: RouteCandidate) -> AsyncIterator[str]:
            calls.append(candidate.model.upstream_id)
            if candidate is plan.primary:
                if False:
                    yield ""
                raise ProviderError(
                    "pre-first transport failure",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.TRANSPORT,
                )
            yield 'event: message_start\ndata: {"type":"message_start"}\n\n'

        stream, provider_name = await Router.execute_plan_stream(plan, open_candidate)

        assert provider_name == "github-copilot"
        assert calls == ["primary-stream", "secondary-stream"]
        assert await anext(stream) == ('event: message_start\ndata: {"type":"message_start"}\n\n')

    @pytest.mark.asyncio
    async def test_native_stream_post_first_failure_never_switches_and_closes_once(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        from router_maestro.routing.router import Router

        primary = _native_provider()
        secondary = _native_provider()
        primary_plan = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="primary-stream",
        )
        secondary_plan = self._native_plan(
            secondary,
            CapabilitySupport.SUPPORTED,
            model="secondary-stream",
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=primary_plan.primary,
            fallbacks=(secondary_plan.primary,),
            explicit=False,
        )
        calls: list[str] = []
        close_calls = 0
        post_first_error = ProviderError(
            "private-post-first-message",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            cause=ValueError("private-post-first-cause"),
        )

        async def open_candidate(candidate: RouteCandidate) -> AsyncIterator[str]:
            nonlocal close_calls
            calls.append(candidate.model.upstream_id)
            try:
                yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
                raise post_first_error
            finally:
                close_calls += 1

        with caplog.at_level(logging.INFO, logger="router_maestro.routing"):
            stream, provider_name = await Router.execute_plan_stream(plan, open_candidate)
            assert await anext(stream) == (
                'event: message_start\ndata: {"type":"message_start"}\n\n'
            )
            with pytest.raises(ProviderError) as exc_info:
                await anext(stream)

        assert provider_name == "github-copilot"
        assert exc_info.value is post_first_error
        assert calls == ["primary-stream"]
        assert close_calls == 1
        assert "private-post-first-message" not in caplog.text
        assert "private-post-first-cause" not in caplog.text

    @pytest.mark.asyncio
    async def test_native_stream_clean_eof_before_first_frame_is_typed_and_closes(self):
        close_calls = 0

        class _StreamResponse:
            status_code = 200

            async def aiter_lines(self):
                yield "event: copilot_usage"
                yield 'data: {"type":"copilot_usage"}'
                yield ""

        @asynccontextmanager
        async def context():
            nonlocal close_calls
            try:
                yield _StreamResponse()
            finally:
                close_calls += 1

        provider = _native_provider()
        provider._stream_with_auth_retry = MagicMock(return_value=context())
        candidate = self._native_plan(
            provider,
            CapabilitySupport.SUPPORTED,
            model="clean-eof",
        ).primary
        stream = _stream_native_candidate(candidate, {"model": "original"})

        with pytest.raises(ProviderError) as exc_info:
            await anext(stream)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 502
        assert exc_info.value.upstream_status_code == 200
        assert exc_info.value.provider == "github-copilot"
        assert exc_info.value.model == "clean-eof"
        assert close_calls == 1

    @pytest.mark.asyncio
    async def test_native_stream_clean_eof_before_commit_falls_back_with_frozen_plan(self):
        from router_maestro.routing.router import Router

        close_counts = {"primary": 0, "secondary": 0}

        class _StreamResponse:
            status_code = 200

            def __init__(self, lines: list[str]) -> None:
                self._lines = lines

            async def aiter_lines(self):
                for line in self._lines:
                    yield line

        def context(name: str, lines: list[str]):
            @asynccontextmanager
            async def manager():
                try:
                    yield _StreamResponse(lines)
                finally:
                    close_counts[name] += 1

            return manager()

        primary = _native_provider()
        primary._stream_with_auth_retry = MagicMock(
            return_value=context(
                "primary",
                [
                    "event: copilot_usage",
                    'data: {"type":"copilot_usage"}',
                    "",
                ],
            )
        )
        secondary = _native_provider()
        secondary._stream_with_auth_retry = MagicMock(
            return_value=context(
                "secondary",
                [
                    "event: message_start",
                    f"data: {json.dumps(_VALID_MESSAGE_START)}",
                    "",
                    "event: message_stop",
                    'data: {"type":"message_stop"}',
                    "",
                ],
            )
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="clean-eof-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="terminal-secondary",
                ).primary,
            ),
            explicit=False,
        )

        stream, provider_name = await Router.execute_plan_stream(
            plan,
            lambda candidate: _stream_native_candidate(candidate, {"model": "original"}),
        )
        frames = [frame async for frame in stream]

        assert provider_name == "github-copilot"
        assert [
            line.removeprefix("event: ")
            for frame in frames
            for line in frame.splitlines()
            if line.startswith("event: ")
        ] == ["message_start", "message_stop"]
        assert close_counts == {"primary": 1, "secondary": 1}
        primary._stream_with_auth_retry.assert_called_once()
        secondary._stream_with_auth_retry.assert_called_once()

    @pytest.mark.parametrize(
        ("status_code", "expected_type"),
        [(429, "rate_limit_error"), (529, "overloaded_error")],
    )
    def test_native_stream_endpoint_postcommit_provider_error_is_typed_and_safe(
        self,
        client,
        status_code,
        expected_type,
    ):
        from router_maestro.routing.router import Router

        close_calls = 0
        private_cause = "private-native-stream-cause"

        async def selected_stream():
            nonlocal close_calls
            try:
                yield f"event: message_start\ndata: {json.dumps(_VALID_MESSAGE_START)}\n\n"
                raise ProviderError(
                    "Safe native stream failure",
                    status_code=status_code,
                    kind=ProviderFailureKind.RATE_LIMIT,
                    cause=RuntimeError(private_cause),
                )
            finally:
                close_calls += 1

        primary = _native_provider()
        plan = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="committed-primary",
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.Router.execute_plan_stream",
                new_callable=AsyncMock,
                return_value=(selected_stream(), "github-copilot"),
            ),
        ):
            downstream = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        event_types = [
            line.removeprefix("event: ")
            for line in downstream.text.splitlines()
            if line.startswith("event: ")
        ]
        errors = [
            json.loads(line.removeprefix("data: "))
            for line in downstream.text.splitlines()
            if line.startswith("data: ") and '"type": "error"' in line
        ]
        assert downstream.status_code == 200
        assert event_types.count("error") == 1
        assert "message_stop" not in event_types
        assert len(errors) == 1
        assert errors[0]["error"] == {
            "type": expected_type,
            "message": "Safe native stream failure",
        }
        assert private_cause not in downstream.text
        assert close_calls == 1

    def test_native_stream_endpoint_clean_eof_after_commit_emits_one_error_and_closes(
        self,
        client,
        caplog: pytest.LogCaptureFixture,
    ):
        from router_maestro.routing.router import Router

        close_calls = 0

        class _StreamResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in (
                    "event: message_start",
                    f"data: {json.dumps(_VALID_MESSAGE_START)}",
                    "",
                    "event: content_block_start",
                    'data: {"type":"content_block_start","index":0,'
                    '"content_block":{"type":"text","text":""}}',
                    "",
                    "event: content_block_delta",
                    'data: {"type":"content_block_delta","index":0,'
                    '"delta":{"type":"text_delta","text":"partial"}}',
                    "",
                ):
                    yield line

        @asynccontextmanager
        async def context():
            nonlocal close_calls
            try:
                yield _StreamResponse()
            finally:
                close_calls += 1

        primary = _native_provider()
        primary._stream_with_auth_retry = MagicMock(return_value=context())
        secondary = _native_provider()
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="committed-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="must-not-run",
                ).primary,
            ),
            explicit=False,
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
            caplog.at_level(logging.INFO),
        ):
            downstream = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        event_types = [
            line.removeprefix("event: ")
            for line in downstream.text.splitlines()
            if line.startswith("event: ")
        ]
        errors = [
            json.loads(line.removeprefix("data: "))
            for line in downstream.text.splitlines()
            if line.startswith("data: ") and '"type": "error"' in line
        ]
        assert downstream.status_code == 200
        assert event_types.count("error") == 1
        assert "message_stop" not in event_types
        assert len(errors) == 1
        assert errors[0]["error"]["type"] == "api_error"
        assert errors[0]["error"]["message"] == (
            "Native Anthropic upstream returned a malformed stream"
        )
        assert close_calls == 1
        primary._stream_with_auth_retry.assert_called_once()
        secondary._stream_with_auth_retry.assert_not_called()
        assert "route_attempt_succeeded" not in caplog.text

    def test_native_stream_endpoint_upstream_error_is_terminal_without_duplicate(
        self,
        client,
    ):
        from router_maestro.routing.router import Router

        upstream_error = {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "try again"},
        }

        class _StreamResponse:
            status_code = 200

            async def aiter_lines(self):
                yield "event: error"
                yield f"data: {json.dumps(upstream_error)}"
                yield ""
                yield "event: message_start"
                yield 'data: {"type":"message_start"'
                yield ""

        @asynccontextmanager
        async def context():
            yield _StreamResponse()

        provider = _native_provider()
        provider._stream_with_auth_retry = MagicMock(return_value=context())
        plan = self._native_plan(
            provider,
            CapabilitySupport.SUPPORTED,
            model="upstream-error",
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
        ):
            downstream = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "github-copilot/upstream-error",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert downstream.status_code == 200
        assert downstream.text.count("event: error") == 1
        assert "message_stop" not in downstream.text
        assert upstream_error["error"]["message"] in downstream.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "primary_lines",
        [
            pytest.param(
                [
                    "event: message_start",
                    'data: {"type":"message_start","private":"secret"',
                    "",
                ],
                id="malformed-json",
            ),
            pytest.param(
                ["event: message_start", "data: ", ""],
                id="empty-event-data",
            ),
            pytest.param([], id="empty-stream"),
        ],
    )
    async def test_native_stream_bad_before_first_frame_falls_back_and_closes(
        self,
        primary_lines,
    ):
        from router_maestro.routing.router import Router

        close_counts = {"primary": 0, "secondary": 0}

        class _StreamResponse:
            status_code = 200

            def __init__(self, lines):
                self.lines = lines

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

        def context(name, lines):
            @asynccontextmanager
            async def manager():
                try:
                    yield _StreamResponse(lines)
                finally:
                    close_counts[name] += 1

            return manager()

        primary = _native_provider()
        primary.name = "github-copilot"
        primary._stream_with_auth_retry = MagicMock(return_value=context("primary", primary_lines))
        secondary = _native_provider()
        secondary.name = "github-copilot"
        secondary._stream_with_auth_retry = MagicMock(
            return_value=context(
                "secondary",
                [
                    "event: message_start",
                    f"data: {json.dumps(_VALID_MESSAGE_START)}",
                    "",
                ],
            )
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="bad-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="valid-secondary",
                ).primary,
            ),
            explicit=False,
        )

        stream, _provider_name = await Router.execute_plan_stream(
            plan,
            lambda candidate: _stream_native_candidate(candidate, {"model": "original"}),
        )
        first = await anext(stream)
        await stream.aclose()

        expected = deepcopy(_VALID_MESSAGE_START)
        expected["message"]["model"] = "github-copilot/valid-secondary"
        assert json.loads(first.split("data: ", 1)[1]) == expected
        assert close_counts == {"primary": 1, "secondary": 1}
        primary._stream_with_auth_retry.assert_called_once()
        secondary._stream_with_auth_retry.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("event_type", "payload"),
        [
            ("message_start", {"type": "message_start", "message": {}}),
            ("message_start", _message_start_with_model()),
            ("message_start", _message_start_with_model(None)),
            ("message_start", _message_start_with_model(42)),
            ("message_start", _message_start_with_model("")),
            (
                "content_block_start",
                {"type": "content_block_start", "index": "0", "content_block": {}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": ""}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": True}),
            ("message_delta", {"type": "message_delta", "delta": {}, "usage": []}),
            ("error", {"type": "error", "error": "bad"}),
        ],
    )
    async def test_native_stream_malformed_known_first_frame_falls_back(
        self,
        event_type,
        payload,
    ):
        from router_maestro.routing.router import Router

        class _StreamResponse:
            status_code = 200

            def __init__(self, lines):
                self.lines = lines

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

        def context(lines):
            @asynccontextmanager
            async def manager():
                yield _StreamResponse(lines)

            return manager()

        primary = _native_provider()
        primary.name = "github-copilot"
        primary._stream_with_auth_retry = MagicMock(
            return_value=context([f"event: {event_type}", f"data: {json.dumps(payload)}", ""])
        )
        secondary = _native_provider()
        secondary.name = "github-copilot"
        secondary._stream_with_auth_retry = MagicMock(
            return_value=context(
                ["event: message_start", f"data: {json.dumps(_VALID_MESSAGE_START)}", ""]
            )
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="malformed-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="valid-secondary",
                ).primary,
            ),
            explicit=False,
        )

        stream, provider_name = await Router.execute_plan_stream(
            plan,
            lambda candidate: _stream_native_candidate(candidate, {"model": "original"}),
        )
        first = json.loads((await anext(stream)).split("data: ", 1)[1])
        await stream.aclose()

        assert provider_name == "github-copilot"
        expected = deepcopy(_VALID_MESSAGE_START)
        expected["message"]["model"] = "github-copilot/valid-secondary"
        assert first == expected
        primary._stream_with_auth_retry.assert_called_once()
        secondary._stream_with_auth_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_native_stream_deep_malformed_first_frame_falls_back(self):
        from router_maestro.routing.router import Router

        class _StreamResponse:
            status_code = 200

            def __init__(self, lines):
                self.lines = lines

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

        def context(lines):
            @asynccontextmanager
            async def manager():
                yield _StreamResponse(lines)

            return manager()

        malformed = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": 7},
        }
        primary = _native_provider()
        primary.name = "github-copilot"
        primary._stream_with_auth_retry = MagicMock(
            return_value=context(
                ["event: content_block_delta", f"data: {json.dumps(malformed)}", ""]
            )
        )
        secondary = _native_provider()
        secondary.name = "github-copilot"
        secondary._stream_with_auth_retry = MagicMock(
            return_value=context(
                ["event: message_start", f"data: {json.dumps(_VALID_MESSAGE_START)}", ""]
            )
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="malformed-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="valid-secondary",
                ).primary,
            ),
            explicit=False,
        )

        stream, provider_name = await Router.execute_plan_stream(
            plan,
            lambda candidate: _stream_native_candidate(candidate, {"model": "original"}),
        )
        first = json.loads((await anext(stream)).split("data: ", 1)[1])
        await stream.aclose()

        assert provider_name == "github-copilot"
        expected = deepcopy(_VALID_MESSAGE_START)
        expected["message"]["model"] = "github-copilot/valid-secondary"
        assert first == expected
        primary._stream_with_auth_retry.assert_called_once()
        secondary._stream_with_auth_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_native_stream_malformed_after_commit_does_not_switch_and_closes(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        from router_maestro.routing.router import Router

        marker = "private-malformed-stream-body"
        close_calls = 0

        class _StreamResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in (
                    "event: message_start",
                    f"data: {json.dumps(_VALID_MESSAGE_START)}",
                    "",
                    "event: content_block_delta",
                    'data: {"type":"content_block_delta","index":0,"delta":[]}',
                    "",
                ):
                    yield line

        @asynccontextmanager
        async def context():
            nonlocal close_calls
            try:
                yield _StreamResponse()
            finally:
                close_calls += 1

        primary = _native_provider()
        primary.name = "github-copilot"
        primary._stream_with_auth_retry = MagicMock(return_value=context())
        secondary = _native_provider()
        secondary.name = "github-copilot"
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="committed-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="must-not-run",
                ).primary,
            ),
            explicit=False,
        )

        with caplog.at_level(logging.INFO, logger="router_maestro.routing"):
            stream, _provider_name = await Router.execute_plan_stream(
                plan,
                lambda candidate: _stream_native_candidate(candidate, {"model": "original"}),
            )
            expected = deepcopy(_VALID_MESSAGE_START)
            expected["message"]["model"] = "github-copilot/committed-primary"
            assert json.loads((await anext(stream)).split("data: ", 1)[1]) == expected
            with pytest.raises(ProviderError) as exc_info:
                await anext(stream)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.retryable is True
        assert exc_info.value.provider == "github-copilot"
        assert exc_info.value.model == "committed-primary"
        assert exc_info.value.attempts == ()
        assert close_calls == 1
        secondary._stream_with_auth_retry.assert_not_called()
        assert marker not in caplog.text

    @pytest.mark.asyncio
    async def test_native_stream_deep_malformed_after_commit_does_not_switch_and_closes(self):
        from router_maestro.routing.router import Router

        close_calls = 0

        class _StreamResponse:
            status_code = 200

            async def aiter_lines(self):
                for line in (
                    "event: message_start",
                    f"data: {json.dumps(_VALID_MESSAGE_START)}",
                    "",
                    "event: message_delta",
                    'data: {"type":"message_delta","delta":{"stop_reason":7},'
                    '"usage":{"output_tokens":"bad"}}',
                    "",
                ):
                    yield line

        @asynccontextmanager
        async def context():
            nonlocal close_calls
            try:
                yield _StreamResponse()
            finally:
                close_calls += 1

        primary = _native_provider()
        primary.name = "github-copilot"
        primary._stream_with_auth_retry = MagicMock(return_value=context())
        secondary = _native_provider()
        secondary.name = "github-copilot"
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=self._native_plan(
                primary,
                CapabilitySupport.SUPPORTED,
                model="committed-primary",
            ).primary,
            fallbacks=(
                self._native_plan(
                    secondary,
                    CapabilitySupport.SUPPORTED,
                    model="must-not-run",
                ).primary,
            ),
            explicit=False,
        )

        stream, _provider_name = await Router.execute_plan_stream(
            plan,
            lambda candidate: _stream_native_candidate(candidate, {"model": "original"}),
        )
        expected = deepcopy(_VALID_MESSAGE_START)
        expected["message"]["model"] = "github-copilot/committed-primary"
        assert json.loads((await anext(stream)).split("data: ", 1)[1]) == expected
        with pytest.raises(ProviderError) as exc_info:
            await anext(stream)

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.provider == "github-copilot"
        assert exc_info.value.model == "committed-primary"
        assert close_calls == 1
        secondary._stream_with_auth_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_native_stream_all_malformed_records_safe_attempt_ledger(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        from router_maestro.routing.router import Router

        marker = "private-precommit-body"

        class _StreamResponse:
            status_code = 200

            async def aiter_lines(self):
                yield "event: message_start"
                yield f'data: {{"type":"message_start","marker":"{marker}"'
                yield ""

        def context():
            @asynccontextmanager
            async def manager():
                yield _StreamResponse()

            return manager()

        providers = [_native_provider(), _native_provider()]
        for provider in providers:
            provider.name = "github-copilot"
            provider._stream_with_auth_retry = MagicMock(return_value=context())
        candidates = tuple(
            self._native_plan(
                provider,
                CapabilitySupport.SUPPORTED,
                model=model,
            ).primary
            for provider, model in zip(providers, ("first", "second"), strict=True)
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=candidates[0],
            fallbacks=(candidates[1],),
            explicit=False,
        )

        with (
            caplog.at_level(logging.INFO, logger="router_maestro.routing"),
            pytest.raises(ProviderError) as exc_info,
        ):
            await Router.execute_plan_stream(
                plan,
                lambda candidate: _stream_native_candidate(candidate, {"model": "original"}),
            )

        assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
        assert exc_info.value.provider == "github-copilot"
        assert exc_info.value.model == "second"
        assert [attempt.model.upstream_id for attempt in exc_info.value.attempts] == [
            "first",
            "second",
        ]
        assert marker not in caplog.text

    def test_native_stream_endpoint_falls_back_after_pre_first_http_503(self, client):
        from router_maestro.routing.router import Router

        class _UpstreamStreamResponse:
            def __init__(self, status_code: int, lines: list[str]) -> None:
                self.status_code = status_code
                self._lines = lines

            async def aread(self) -> bytes:
                return b'{"error":{"message":"private upstream body"}}'

            async def aiter_lines(self):
                for line in self._lines:
                    yield line

        def stream_context(response):
            @asynccontextmanager
            async def context():
                yield response

            return context()

        primary = _native_provider()
        primary.name = "github-copilot"
        primary._stream_with_auth_retry = MagicMock(
            return_value=stream_context(_UpstreamStreamResponse(503, []))
        )
        secondary = _native_provider()
        secondary.name = "github-copilot"
        secondary._stream_with_auth_retry = MagicMock(
            return_value=stream_context(
                _UpstreamStreamResponse(
                    200,
                    [
                        "event: message_start",
                        f"data: {json.dumps(_VALID_MESSAGE_START)}",
                        "",
                    ],
                )
            )
        )
        primary_plan = self._native_plan(
            primary,
            CapabilitySupport.SUPPORTED,
            model="primary-stream",
        )
        secondary_plan = self._native_plan(
            secondary,
            CapabilitySupport.SUPPORTED,
            model="secondary-stream",
        )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=primary_plan.primary,
            fallbacks=(secondary_plan.primary,),
            explicit=False,
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
        ):
            downstream = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert downstream.status_code == 200
        assert "event: message_start" in downstream.text
        assert '"id": "msg_123"' in downstream.text
        message_start = next(
            json.loads(line.removeprefix("data: "))
            for line in downstream.text.splitlines()
            if line.startswith("data: ") and '"type": "message_start"' in line
        )
        assert message_start["message"]["model"] == "github-copilot/secondary-stream"
        primary._stream_with_auth_retry.assert_called_once()
        secondary._stream_with_auth_retry.assert_called_once()

    @pytest.mark.parametrize(
        ("failure", "expected_status"),
        [
            pytest.param("http-503", 503, id="retryable-status-exhaustion"),
            pytest.param("malformed", 502, id="malformed-protocol-exhaustion"),
        ],
    )
    def test_native_stream_precommit_exhaustion_returns_json_error(
        self,
        client,
        caplog: pytest.LogCaptureFixture,
        failure: str,
        expected_status: int,
    ):
        from router_maestro.routing.router import Router

        marker = f"private-{failure}-upstream-body"

        class _UpstreamStreamResponse:
            status_code = 503 if failure == "http-503" else 200

            async def aread(self) -> bytes:
                return f'{{"error":{{"message":"{marker}"}}}}'.encode()

            async def aiter_lines(self):
                if failure == "malformed":
                    yield "event: message_start"
                    yield f'data: {{"type":"message_start","marker":"{marker}"'
                    yield ""

        def stream_context():
            @asynccontextmanager
            async def context():
                yield _UpstreamStreamResponse()

            return context()

        providers = [_native_provider(), _native_provider()]
        candidates = []
        for index, provider in enumerate(providers, start=1):
            provider.name = "github-copilot"
            provider._stream_with_auth_retry = MagicMock(return_value=stream_context())
            candidates.append(
                self._native_plan(
                    provider,
                    CapabilitySupport.SUPPORTED,
                    model=f"candidate-{index}",
                ).primary
            )
        plan = RoutePlan(
            operation=Operation.NATIVE_ANTHROPIC,
            features=RequestFeatures(),
            primary=candidates[0],
            fallbacks=(candidates[1],),
            explicit=False,
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
            caplog.at_level(logging.INFO),
        ):
            downstream = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert downstream.status_code == expected_status
        assert downstream.headers["content-type"].startswith("application/json")
        assert downstream.json()["type"] == "error"
        assert downstream.json()["error"]["type"] == "api_error"
        assert marker not in downstream.text
        assert marker not in caplog.text
        assert "data:" not in downstream.text
        for provider in providers:
            provider._stream_with_auth_retry.assert_called_once()

    def test_native_plan_final_client_status_preserves_anthropic_error_envelope(self, client):
        from router_maestro.routing.router import Router

        provider = _native_provider()
        response = MagicMock()
        response.status_code = 400
        response.text = '{"error":{"message":"bad request"}}'
        response.json.return_value = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "bad request"},
        }
        provider._send_with_auth_retry.return_value = response
        plan = self._native_plan(
            provider,
            CapabilitySupport.SUPPORTED,
            model="claude-client-error",
        )
        model_router = Router.__new__(Router)
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
        ):
            downstream = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "claude-client-error",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert downstream.status_code == 400
        assert downstream.json() == response.json.return_value

    def test_legacy_native_request_and_upstream_body_do_not_leak_to_logs(
        self,
        client,
        caplog: pytest.LogCaptureFixture,
    ):
        thinking_marker = "private-thinking-marker"
        output_marker = "private-output-config-marker"
        upstream_marker = "private-upstream-body-marker"
        provider = _native_provider()
        response = MagicMock()
        response.status_code = 400
        response.text = f'{{"error":{{"message":"{upstream_marker}"}}}}'
        response.json.return_value = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": upstream_marker},
        }
        provider._send_with_auth_retry.return_value = response

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta._resolve_native_model",
                new_callable=AsyncMock,
                return_value=_NativeModelResolution(
                    _ResolvedModel("github-copilot", "claude-model", provider),
                    CapabilitySupport.SUPPORTED,
                ),
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
                side_effect=lambda body, _model, _efforts=None: body,
            ),
            caplog.at_level(logging.DEBUG, logger="router_maestro.server.routes.anthropic_beta"),
        ):
            downstream = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "claude-model",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1000,
                        "marker": thinking_marker,
                    },
                    "output_config": {"effort": "high"},
                    "metadata": {"user_id": output_marker},
                },
            )

        assert downstream.status_code == 400
        assert downstream.json()["error"]["message"] == upstream_marker
        assert thinking_marker not in caplog.text
        assert output_marker not in caplog.text
        assert upstream_marker not in caplog.text

    @pytest.mark.parametrize(
        ("provider_error", "expected_status", "expected_type"),
        [
            (ProviderError("Unknown model", status_code=404), 404, "not_found_error"),
            (
                ProviderError("Provider authentication required", status_code=401),
                401,
                "authentication_error",
            ),
        ],
        ids=["unknown-model", "authentication"],
    )
    def test_native_planning_provider_error_preserves_http_status(
        self,
        non_raising_client,
        provider_error,
        expected_status,
        expected_type,
    ):
        model_router = MagicMock()
        model_router.plan_route = AsyncMock(side_effect=provider_error)

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            response = non_raising_client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "unknown-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == expected_status
        assert response.json() == {
            "type": "error",
            "error": {
                "type": expected_type,
                "message": str(provider_error),
            },
        }

    def test_native_planning_unexpected_error_is_not_swallowed(self, client):
        model_router = MagicMock()
        model_router.plan_route = AsyncMock(side_effect=RuntimeError("planning failed"))

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            pytest.raises(RuntimeError, match="planning failed"),
        ):
            client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "unknown-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

    def test_explicit_native_unsupported_uses_standard_without_native_call(self, client):
        provider = _native_provider()
        model_router = MagicMock(_models_cache={})
        model_router.plan_route = AsyncMock(
            return_value=self._native_plan(
                provider,
                CapabilitySupport.UNSUPPORTED,
                model="unsupported-model",
            )
        )

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
                return_value=JSONResponse(content={"path": "standard"}),
            ) as standard,
            patch(
                "router_maestro.server.routes.anthropic_beta._parse_as_anthropic_request",
                new_callable=AsyncMock,
            ) as parse_request,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "unsupported-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        assert response.json() == {"path": "standard"}
        parse_request.assert_awaited_once()
        standard.assert_awaited_once()
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    @pytest.mark.parametrize(
        ("invalid_budget", "message"),
        [
            (-1, "thinking.budget_tokens must be a positive integer"),
            (100, "thinking.budget_tokens must be less than max_tokens"),
            (True, "thinking.budget_tokens must be a positive integer"),
            ("10", "thinking.budget_tokens must be a positive integer"),
            ({}, "thinking.budget_tokens must be a positive integer"),
            (1.5, "thinking.budget_tokens must be a positive integer"),
        ],
        ids=["negative", "at-max", "bool", "string", "object", "fraction"],
    )
    def test_native_unsupported_fallback_rejects_invalid_thinking_before_any_transport(
        self,
        non_raising_client,
        stream,
        invalid_budget,
        message,
    ):
        provider = _native_provider()
        model_router = MagicMock(_models_cache={})
        model_router.plan_route = AsyncMock(
            return_value=self._native_plan(
                provider,
                CapabilitySupport.UNSUPPORTED,
                model="gpt-5.5",
            )
        )

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
            ) as standard,
        ):
            response = non_raising_client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "github-copilot/gpt-5.5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": stream,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": invalid_budget,
                    },
                },
            )

        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
        standard.assert_not_awaited()
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    def test_native_unsupported_fallback_encodes_manual_schema_validation_as_anthropic_400(
        self,
        non_raising_client,
        stream,
    ):
        provider = _native_provider()
        model_router = MagicMock(_models_cache={})
        model_router.plan_route = AsyncMock(
            return_value=self._native_plan(
                provider,
                CapabilitySupport.UNSUPPORTED,
                model="gpt-5.5",
            )
        )

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
            ) as standard,
        ):
            response = non_raising_client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "github-copilot/gpt-5.5",
                    "max_tokens": 100,
                    "messages": {"role": "user", "content": "Hi"},
                    "stream": stream,
                },
            )

        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        error = response.json()
        assert error["type"] == "error"
        assert error["error"]["type"] == "invalid_request_error"
        assert error["error"]["message"].startswith("messages:")
        standard.assert_not_awaited()
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    @pytest.mark.parametrize(
        ("budget", "expected_status"),
        [(99, 200), (100, 400), (101, 400)],
        ids=["below", "equal", "above"],
    )
    def test_standard_fallback_validates_budget_against_coerced_max_tokens(
        self,
        non_raising_client,
        stream,
        budget,
        expected_status,
    ):
        provider = _native_provider()
        model_router = MagicMock(_models_cache={})
        model_router.plan_route = AsyncMock(
            return_value=self._native_plan(
                provider,
                CapabilitySupport.UNSUPPORTED,
                model="gpt-5.5",
            )
        )

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
                return_value=JSONResponse(content={"path": "standard"}),
            ) as standard,
        ):
            response = non_raising_client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "github-copilot/gpt-5.5",
                    "max_tokens": "100",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": stream,
                    "thinking": {"type": "enabled", "budget_tokens": budget},
                },
            )

        assert response.status_code == expected_status
        if expected_status == 400:
            assert response.json() == {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "thinking.budget_tokens must be less than max_tokens",
                },
            }
            standard.assert_not_awaited()
        else:
            assert response.json() == {"path": "standard"}
            standard.assert_awaited_once()
            parsed = standard.await_args.kwargs["request"]
            assert parsed.max_tokens == 100
            assert parsed.thinking is not None
            assert parsed.thinking.budget_tokens == 99
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    @pytest.mark.parametrize(
        "tool_support",
        [None, True],
        ids=["unknown-tools", "supported-tools"],
    )
    def test_explicit_qualified_copilot_model_adapts_to_standard_transport(
        self,
        client,
        stream,
        tool_support,
    ):
        provider = MagicMock(spec=CopilotProvider)
        provider.name = "github-copilot"
        provider.is_authenticated.return_value = True
        provider.capabilities = CopilotProvider().capabilities
        provider.ensure_token = AsyncMock()
        provider._send_with_auth_retry = AsyncMock()
        model = ModelInfo(
            id="gpt-5.5",
            name="GPT 5.5",
            provider="github-copilot",
            operation_capabilities={Operation.NATIVE_ANTHROPIC: False},
            feature_capabilities=(
                {Feature.TOOLS: tool_support} if tool_support is not None else {}
            ),
        )
        from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
        from router_maestro.utils.cache import TTLCache

        model_router = Router.__new__(Router)
        model_router.providers = {"github-copilot": provider}
        model_router._models_cache = {
            "gpt-5.5": ("github-copilot", model),
            "github-copilot/gpt-5.5": ("github-copilot", model),
        }
        model_router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        model_router._models_cache_ttl.set(True)
        model_router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        model_router._priorities_cache.set(PrioritiesConfig())
        model_router._fuzzy_cache = {}
        model_router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        model_router._providers_ttl.set(True)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
                return_value=JSONResponse(content={"path": "standard"}),
            ) as standard,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json=_native_feature_body(
                    "tools",
                    model="github-copilot/gpt-5.5",
                    stream=stream,
                ),
            )

        assert response.status_code == 200
        assert response.json() == {"path": "standard"}
        standard.assert_awaited_once()
        assert standard.await_args.kwargs["request"].model == "github-copilot/gpt-5.5"
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()
        provider._stream_with_auth_retry.assert_not_called()

    def test_explicit_native_transport_and_feature_unsupported_returns_400_before_adaptation(
        self,
        client,
    ):
        provider = _native_provider()
        model = ModelInfo(
            id="gpt-no-native-tools",
            name="GPT without native tools",
            provider="github-copilot",
            operation_capabilities={Operation.NATIVE_ANTHROPIC: False},
            feature_capabilities={Feature.TOOLS: False},
        )
        model_router = _real_native_router(
            provider,
            [model],
            ["github-copilot/gpt-no-native-tools"],
        )

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
                return_value=JSONResponse(content={"path": "standard"}),
            ) as standard,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json=_native_feature_body(
                    "tools",
                    model="github-copilot/gpt-no-native-tools",
                    stream=False,
                ),
            )

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
        standard.assert_not_awaited()
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()
        provider._stream_with_auth_retry.assert_not_called()

    def test_explicit_non_copilot_without_native_transport_uses_standard_path(self, client):
        provider = MagicMock()
        provider.name = "anthropic"
        ref = ModelRef("anthropic", "claude-sonnet-4")
        operation = Operation.NATIVE_ANTHROPIC
        features = RequestFeatures()
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: CapabilitySupport.UNSUPPORTED},
        )
        plan = RoutePlan(
            operation=operation,
            features=features,
            primary=RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.UNSUPPORTED,
            ),
            fallbacks=(),
            explicit=True,
        )
        model_router = MagicMock(_models_cache={})
        model_router.plan_route = AsyncMock(return_value=plan)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._parse_as_anthropic_request",
                new_callable=AsyncMock,
            ) as parse_request,
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
                return_value=JSONResponse(content={"path": "standard"}),
            ) as standard,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "anthropic/claude-sonnet-4",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        assert response.json() == {"path": "standard"}
        parse_request.assert_awaited_once()
        standard.assert_awaited_once()

    def test_auto_with_only_non_copilot_native_mismatch_uses_standard_path(self, client):
        from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
        from router_maestro.utils.cache import TTLCache

        provider = MagicMock()
        provider.name = "anthropic"
        provider.is_authenticated.return_value = True
        provider.capabilities.supports.return_value = False
        model = ModelInfo(
            id="claude-sonnet-4",
            name="Claude Sonnet 4",
            provider="anthropic",
        )
        model_router = Router.__new__(Router)
        model_router.providers = {"anthropic": provider}
        model_router._models_cache = {
            "claude-sonnet-4": ("anthropic", model),
            "anthropic/claude-sonnet-4": ("anthropic", model),
        }
        model_router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        model_router._models_cache_ttl.set(True)
        model_router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        model_router._priorities_cache.set(
            PrioritiesConfig(priorities=["anthropic/claude-sonnet-4"])
        )
        model_router._fuzzy_cache = {}
        model_router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        model_router._providers_ttl.set(True)
        original_plan_route = model_router.plan_route
        model_router.plan_route = AsyncMock(wraps=original_plan_route)

        with (
            patch(
                "router_maestro.server.routes.anthropic_beta.get_router",
                return_value=model_router,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta._parse_as_anthropic_request",
                new_callable=AsyncMock,
            ),
            patch(
                "router_maestro.server.routes.anthropic_beta.standard_messages",
                new_callable=AsyncMock,
                return_value=JSONResponse(content={"path": "standard"}),
            ) as standard,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "router-maestro",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        assert response.json() == {"path": "standard"}
        model_router.plan_route.assert_awaited_once_with(
            "router-maestro",
            Operation.NATIVE_ANTHROPIC,
            RequestFeatures(),
        )
        standard.assert_awaited_once()

    @pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "automatic"])
    def test_native_unknown_is_attempted_for_compatibility(self, client, explicit):
        provider = _native_provider()
        plan = self._native_plan(
            provider,
            CapabilitySupport.UNKNOWN,
            model="unknown-model",
        )
        plan = RoutePlan(
            operation=plan.operation,
            features=plan.features,
            primary=plan.primary,
            fallbacks=plan.fallbacks,
            explicit=explicit,
        )
        model_router = MagicMock(_models_cache={})
        model_router.plan_route = AsyncMock(return_value=plan)

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "unknown-model" if explicit else "router-maestro",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        provider._send_with_auth_retry.assert_awaited_once()

    def test_native_messages_requests_native_anthropic_route_plan(self, client):
        provider = _native_provider()
        ref = ModelRef("github-copilot", "claude-sonnet-4.5")
        operation = Operation.NATIVE_ANTHROPIC
        features = RequestFeatures()
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: CapabilitySupport.SUPPORTED},
        )
        plan = RoutePlan(
            operation=operation,
            features=features,
            primary=RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            ),
            fallbacks=(),
            explicit=True,
        )
        model_router = MagicMock()
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "claude-sonnet-4.5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        model_router.plan_route.assert_awaited_once_with(
            "claude-sonnet-4.5",
            Operation.NATIVE_ANTHROPIC,
            RequestFeatures(),
        )

    def test_native_messages_uses_plan_capability_not_name_heuristic(self, client):
        provider = _native_provider()
        ref = ModelRef("github-copilot", "custom-native-model")
        operation = Operation.NATIVE_ANTHROPIC
        features = RequestFeatures()
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: CapabilitySupport.SUPPORTED},
        )
        plan = RoutePlan(
            operation=operation,
            features=features,
            primary=RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            ),
            fallbacks=(),
            explicit=True,
        )
        model_router = MagicMock()
        model_router._models_cache = {}
        model_router.plan_route = AsyncMock(return_value=plan)

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            response = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "custom-native-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert response.status_code == 200
        provider._send_with_auth_retry.assert_awaited_once()

    @pytest.mark.parametrize(
        "invalid_budget",
        ["1024", [], {}, True, 1024.5, 0, -1],
        ids=["string", "list", "dict", "bool", "fractional-float", "zero", "negative"],
    )
    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_invalid_budget_returns_anthropic_client_error(
        self,
        mock_resolve,
        mock_router,
        non_raising_client,
        invalid_budget,
    ):
        provider = _native_provider()
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", provider),
            CapabilitySupport.SUPPORTED,
        )
        mock_router.return_value = MagicMock(_models_cache={})

        response = non_raising_client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "enabled", "budget_tokens": invalid_budget},
            },
        )

        assert response.status_code == 400
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "thinking.budget_tokens must be a positive integer",
            },
        }
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.parametrize(
        "invalid_max_tokens",
        ["2048", [], {}, True, 2048.5, 0, -1],
        ids=["string", "list", "dict", "bool", "fractional-float", "zero", "negative"],
    )
    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_invalid_max_tokens_returns_anthropic_client_error(
        self,
        mock_resolve,
        mock_router,
        non_raising_client,
        invalid_max_tokens,
    ):
        provider = _native_provider()
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", provider),
            CapabilitySupport.SUPPORTED,
        )
        mock_router.return_value = MagicMock(_models_cache={})

        response = non_raising_client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": invalid_max_tokens,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 400
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "max_tokens must be a positive integer",
            },
        }
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.parametrize(
        ("thinking", "message"),
        [
            ("enabled", "thinking must be an object"),
            ({"type": "unknown"}, "thinking.type must be enabled, adaptive, or disabled"),
            ({"type": []}, "thinking.type must be enabled, adaptive, or disabled"),
            ({"type": {}}, "thinking.type must be enabled, adaptive, or disabled"),
        ],
        ids=["non-object", "unknown-type", "list-type", "dict-type"],
    )
    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_invalid_thinking_returns_anthropic_client_error(
        self,
        mock_resolve,
        mock_router,
        non_raising_client,
        thinking,
        message,
    ):
        provider = _native_provider()
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", provider),
            CapabilitySupport.SUPPORTED,
        )
        mock_router.return_value = MagicMock(_models_cache={})

        response = non_raising_client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": thinking,
            },
        )

        assert response.status_code == 400
        assert response.json() == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
        provider.ensure_token.assert_not_awaited()
        provider._send_with_auth_retry.assert_not_awaited()

    @pytest.mark.parametrize(
        ("max_tokens", "expected_status"),
        [(1024, 400), (1025, 200)],
        ids=["no-headroom", "one-token-headroom"],
    )
    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_invalid_budget_must_be_less_than_max_tokens(
        self,
        mock_resolve,
        mock_router,
        non_raising_client,
        max_tokens,
        expected_status,
    ):
        provider = _native_provider()
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", provider),
            CapabilitySupport.SUPPORTED,
        )
        mock_router.return_value = MagicMock(_models_cache={})

        response = non_raising_client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
        )

        assert response.status_code == expected_status
        if expected_status == 400:
            assert response.json() == {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "thinking.budget_tokens must be less than max_tokens",
                },
            }
            provider.ensure_token.assert_not_awaited()
            provider._send_with_auth_retry.assert_not_awaited()
        else:
            forwarded = provider._send_with_auth_retry.call_args.kwargs["json"]
            assert forwarded["thinking"] == {"type": "enabled", "budget_tokens": 1024}

    @patch("router_maestro.config.load_priorities_config")
    @patch("router_maestro.server.routes.anthropic_beta.get_router")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_enabled_thinking_without_budget_uses_configured_default(
        self,
        mock_resolve,
        mock_router,
        mock_load_config,
        non_raising_client,
    ):
        provider = _native_provider()
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", provider),
            CapabilitySupport.SUPPORTED,
        )
        mock_router.return_value = MagicMock(_models_cache={})
        mock_load_config.return_value = PrioritiesConfig(
            thinking=ThinkingBudgetConfig(default_budget=4096)
        )

        response = non_raising_client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "enabled"},
            },
        )

        assert response.status_code == 200
        forwarded = provider._send_with_auth_retry.call_args.kwargs["json"]
        assert forwarded["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_native_passthrough_non_streaming(self, mock_resolve, client):
        """Claude model on Copilot uses native passthrough."""
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_bdrk_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "copilot_usage": {"token_details": []},
            "stop_details": {"type": "end_turn"},
        }
        mock_provider._send_with_auth_retry = AsyncMock(return_value=mock_response)
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", mock_provider),
            CapabilitySupport.SUPPORTED,
        )

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=lambda body, _, _efforts=None: body,
        ):
            resp = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "claude-sonnet-4",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == [{"type": "text", "text": "hello"}]
        assert "copilot_usage" not in data
        assert "stop_details" not in data

    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_planless_native_deep_malformed_2xx_returns_safe_502(
        self,
        mock_resolve,
        non_raising_client,
    ):
        marker = "private-native-malformed-marker"
        provider = _native_provider()
        malformed = _valid_native_message_response()
        malformed["usage"]["input_tokens"] = marker
        provider._send_with_auth_retry.return_value.json.return_value = malformed
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", provider),
            CapabilitySupport.SUPPORTED,
        )

        response = non_raising_client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 502
        assert marker not in response.text
        assert response.json() == {
            "detail": "Native Anthropic upstream returned a malformed response"
        }

    @patch("router_maestro.server.routes.anthropic_beta._parse_as_anthropic_request")
    @patch("router_maestro.server.routes.anthropic_beta.standard_messages")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_fallback_for_non_claude_model(self, mock_resolve, mock_standard, mock_parse, client):
        """Non-Claude model falls back to the standard translation path."""
        mock_provider = _native_provider()
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "gpt-5.4", mock_provider),
            CapabilitySupport.UNSUPPORTED,
        )
        mock_parse.return_value = MagicMock()
        mock_standard.return_value = JSONResponse(
            content={"id": "msg_fake", "content": [{"type": "text", "text": "from standard"}]},
        )

        client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "gpt-5.4",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        mock_standard.assert_called_once()
        mock_provider.ensure_token.assert_not_awaited()
        mock_provider._send_with_auth_retry.assert_not_awaited()

    @patch("router_maestro.server.routes.anthropic_beta._parse_as_anthropic_request")
    @patch("router_maestro.server.routes.anthropic_beta.standard_messages")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_fallback_for_non_copilot_provider(
        self, mock_resolve, mock_standard, mock_parse, client
    ):
        """Claude model on native Anthropic provider falls back."""
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("anthropic", "claude-sonnet-4.5", None), CapabilitySupport.SUPPORTED
        )
        mock_parse.return_value = MagicMock()
        mock_standard.return_value = JSONResponse(content={})

        client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        mock_standard.assert_called_once()

    @patch("router_maestro.server.routes.anthropic_beta.standard_messages")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_non_native_fallback_preserves_pydantic_coercion(
        self, mock_resolve, mock_standard, client
    ):
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("anthropic", "claude-sonnet-4.5", None), CapabilitySupport.SUPPORTED
        )
        mock_standard.return_value = JSONResponse(content={})

        response = client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-sonnet-4.5",
                "max_tokens": "2048",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        mock_standard.assert_awaited_once()
        parsed_request = mock_standard.await_args.kwargs["request"]
        assert parsed_request.max_tokens == 2048

    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_upstream_error_forwarded(self, mock_resolve, client):
        """Upstream 4xx errors are forwarded verbatim."""
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"type": "invalid_request_error", "message": "bad model"}
        }
        mock_provider._send_with_auth_retry = AsyncMock(return_value=mock_response)
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", mock_provider),
            CapabilitySupport.SUPPORTED,
        )

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=lambda body, _, _efforts=None: body,
        ):
            resp = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "claude-sonnet-4.5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "bad model"

    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_signature_error_triggers_retry(self, mock_resolve, client):
        """400 with signature error strips thinking and retries."""
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()

        # First call returns signature error, second succeeds
        error_response = MagicMock()
        error_response.status_code = 400
        error_response.text = (
            '{"message":"messages.3.content.0: Invalid `signature` in `thinking` block"}'
        )

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "id": "msg_bdrk_retry",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "retried ok"}],
            "model": "claude-sonnet-4.5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        mock_provider._send_with_auth_retry = AsyncMock(
            side_effect=[error_response, success_response]
        )
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", mock_provider),
            CapabilitySupport.SUPPORTED,
        )

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=lambda body, _, _efforts=None: body,
        ):
            resp = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "claude-sonnet-4.5",
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": "x", "signature": "bad"},
                                {"type": "text", "text": "Hello"},
                            ],
                        },
                        {"role": "user", "content": "Bye"},
                    ],
                },
            )

        assert resp.status_code == 200
        assert resp.json()["content"][0]["text"] == "retried ok"
        # Should have been called twice (original + retry)
        assert mock_provider._send_with_auth_retry.call_count == 2

    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_non_signature_400_not_retried(self, mock_resolve, client):
        """Non-signature 400 errors are NOT retried."""
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()

        error_response = MagicMock()
        error_response.status_code = 400
        error_response.text = '{"message":"max_tokens: Field required"}'
        error_response.json.return_value = {
            "error": {"type": "invalid_request_error", "message": "max_tokens: Field required"}
        }

        mock_provider._send_with_auth_retry = AsyncMock(return_value=error_response)
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", mock_provider),
            CapabilitySupport.SUPPORTED,
        )

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=lambda body, _, _efforts=None: body,
        ):
            resp = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "claude-sonnet-4.5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 400
        # Should only be called once — no retry
        assert mock_provider._send_with_auth_retry.call_count == 1

    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_nested_vendor_payload_and_represented_fields_are_preserved(
        self,
        mock_resolve,
        client,
    ):
        """The shallow option gate must not recurse into message content."""
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-4.5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        mock_provider._send_with_auth_retry = AsyncMock(return_value=mock_response)
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-sonnet-4.5", mock_provider),
            CapabilitySupport.SUPPORTED,
        )

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=lambda body, _, _efforts=None: body,
        ):
            resp = client.post(
                "/api/anthropic/beta/v1/messages",
                json={
                    "model": "claude-sonnet-4.5",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Hi",
                                    "cache_control": {"type": "ephemeral"},
                                    "vendor_payload": {"future": True},
                                }
                            ],
                        }
                    ],
                    "top_p": 0.8,
                    "metadata": {"user_id": "test-user"},
                },
            )

        assert resp.status_code == 200
        forwarded_body = mock_provider._send_with_auth_retry.call_args.kwargs["json"]
        content = forwarded_body["messages"][0]["content"][0]
        assert content["cache_control"] == {"type": "ephemeral"}
        assert content["vendor_payload"] == {"future": True}
        assert forwarded_body["model"] == "claude-sonnet-4.5"
        assert forwarded_body["max_tokens"] == 100
        assert forwarded_body["top_p"] == 0.8
        assert forwarded_body["metadata"] == {"user_id": "test-user"}

    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_effort_forwarded_and_conflicting_budget_removed(self, mock_resolve, client):
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-opus-4.8",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        mock_provider._send_with_auth_retry = AsyncMock(return_value=mock_response)
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-opus-4.8", mock_provider),
            CapabilitySupport.SUPPORTED,
        )

        resp = client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-opus-4-8",
                "max_tokens": 64000,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "adaptive", "budget_tokens": 16000},
                "output_config": {"effort": "xhigh"},
            },
        )

        assert resp.status_code == 200
        forwarded = mock_provider._send_with_auth_retry.call_args.kwargs["json"]
        assert forwarded["thinking"] == {"type": "adaptive"}
        assert forwarded["output_config"] == {"effort": "xhigh"}

    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_enabled_effort_preserves_required_budget(self, mock_resolve, client):
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-opus-4.6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        mock_provider._send_with_auth_retry = AsyncMock(return_value=mock_response)
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-opus-4.6", mock_provider),
            CapabilitySupport.SUPPORTED,
        )

        resp = client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 64000,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "output_config": {"effort": "xhigh"},
            },
        )

        assert resp.status_code == 200
        forwarded = mock_provider._send_with_auth_retry.call_args.kwargs["json"]
        assert forwarded["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert forwarded["output_config"] == {"effort": "xhigh"}

    @patch("router_maestro.server.routes.anthropic_beta.sse_streaming_response")
    @patch("router_maestro.server.routes.anthropic_beta._stream_passthrough")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_stream_effort_forwarded_and_conflicting_budget_removed(
        self, mock_resolve, mock_stream, mock_sse_response, client
    ):
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-opus-4.8", mock_provider),
            CapabilitySupport.SUPPORTED,
        )
        stream_marker = object()
        mock_stream.return_value = stream_marker
        mock_sse_response.return_value = JSONResponse(content={"stream": "captured"})

        resp = client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-opus-4-8",
                "max_tokens": 64000,
                "stream": True,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "adaptive", "budget_tokens": 16000},
                "output_config": {"effort": "xhigh"},
            },
        )

        assert resp.status_code == 200
        forwarded = mock_stream.call_args.args[1]
        assert forwarded["thinking"] == {"type": "adaptive"}
        assert forwarded["output_config"] == {"effort": "xhigh"}
        mock_sse_response.assert_called_once()
        guarded_stream = mock_sse_response.call_args.args[0]
        assert guarded_stream.__name__ == "_encode_native_stream_errors"
        assert mock_sse_response.call_args.kwargs == {
            "keepalive_frame": ANTHROPIC_PING_FRAME,
        }

    @patch("router_maestro.server.routes.anthropic_beta.sse_streaming_response")
    @patch("router_maestro.server.routes.anthropic_beta._stream_passthrough")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_native_model")
    def test_stream_enabled_effort_preserves_required_budget(
        self, mock_resolve, mock_stream, mock_sse_response, client
    ):
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()
        mock_resolve.return_value = _NativeModelResolution(
            _ResolvedModel("github-copilot", "claude-opus-4.6", mock_provider),
            CapabilitySupport.SUPPORTED,
        )
        stream_marker = object()
        mock_stream.return_value = stream_marker
        mock_sse_response.return_value = JSONResponse(content={"stream": "captured"})

        resp = client.post(
            "/api/anthropic/beta/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 64000,
                "stream": True,
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "output_config": {"effort": "xhigh"},
            },
        )

        assert resp.status_code == 200
        forwarded = mock_stream.call_args.args[1]
        assert forwarded["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert forwarded["output_config"] == {"effort": "xhigh"}

    def test_missing_model_returns_400(self, client):
        """Request without model field returns 400."""
        resp = client.post(
            "/api/anthropic/beta/v1/messages",
            json={"max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 400
        assert "model" in resp.json()["detail"]

    def test_invalid_json_returns_400(self, client):
        """Malformed JSON body returns 400."""
        resp = client.post(
            "/api/anthropic/beta/v1/messages",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400


class TestBetaCountTokensEndpoint:
    @patch("router_maestro.server.routes.anthropic_beta._resolve_model")
    def test_count_tokens_resolution_provider_error_is_anthropic_native(
        self,
        mock_resolve,
        client,
    ):
        mock_resolve.side_effect = ProviderError(
            "Unknown model",
            status_code=404,
            kind=ProviderFailureKind.CLIENT_REQUEST,
        )

        response = client.post(
            "/api/anthropic/beta/v1/messages/count_tokens",
            json={
                "model": "unknown-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 404
        assert response.json() == {
            "type": "error",
            "error": {"type": "not_found_error", "message": "Unknown model"},
        }

    @patch("router_maestro.server.routes.anthropic_beta._resolve_model")
    def test_count_tokens_provider_error_message_does_not_leak_to_logs(
        self,
        mock_resolve,
        client,
        caplog: pytest.LogCaptureFixture,
    ):
        marker = "private-count-tokens-cause-marker"
        provider = _native_provider()
        provider.count_native_anthropic_tokens.side_effect = ProviderError(
            "Safe count tokens failure",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.TRANSPORT,
            cause=RuntimeError(marker),
        )
        mock_resolve.return_value = _ResolvedModel(
            "github-copilot",
            "claude-sonnet-4.5",
            provider,
        )

        with caplog.at_level(
            logging.INFO,
            logger="router_maestro.server.routes.anthropic_beta",
        ):
            downstream = client.post(
                "/api/anthropic/beta/v1/messages/count_tokens",
                json={
                    "model": "claude-sonnet-4.5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert downstream.status_code == 502
        assert downstream.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Safe count tokens failure",
            },
        }
        assert marker not in downstream.text
        assert marker not in caplog.text

    def test_count_tokens_does_not_request_completion_route_plan(self, client):
        provider = _native_provider()
        provider.count_native_anthropic_tokens = AsyncMock(return_value=42)
        model_router = MagicMock()
        model_router._resolve_provider = AsyncMock(
            return_value=("github-copilot", "claude-sonnet-4.5", provider)
        )
        model_router.plan_route = AsyncMock()

        with patch(
            "router_maestro.server.routes.anthropic_beta.get_router",
            return_value=model_router,
        ):
            result = client.post(
                "/api/anthropic/beta/v1/messages/count_tokens",
                json={
                    "model": "claude-sonnet-4.5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert result.status_code == 200
        assert result.json() == {"input_tokens": 42}
        model_router._resolve_provider.assert_awaited_once_with("claude-sonnet-4.5")
        model_router.plan_route.assert_not_awaited()

    @patch("router_maestro.server.routes.anthropic_beta._resolve_model")
    def test_passthrough_count_tokens(self, mock_resolve, client):
        """Claude model count_tokens goes through native endpoint."""
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()

        mock_provider.count_native_anthropic_tokens = AsyncMock(return_value=42)
        mock_resolve.return_value = _ResolvedModel(
            "github-copilot", "claude-sonnet-4.5", mock_provider
        )

        resp = client.post(
            "/api/anthropic/beta/v1/messages/count_tokens",
            json={
                "model": "claude-sonnet-4.5",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"input_tokens": 42}
        mock_provider.count_native_anthropic_tokens.assert_awaited_once_with(
            {
                "model": "claude-sonnet-4.5",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            model="claude-sonnet-4.5",
        )
