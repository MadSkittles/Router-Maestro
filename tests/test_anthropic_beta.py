"""Tests for the Anthropic beta passthrough route."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from router_maestro.providers.copilot import CopilotProvider
from router_maestro.server.routes.anthropic_beta import (
    _apply_thinking_budget_native,
    _clean_stream_frame,
    _is_native_eligible,
    _strip_response,
    router,
)

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
        data = json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "model": "claude-sonnet-4.5",
                    "copilot_usage": {"x": 1},
                    "stop_details": {"y": 2},
                    "usage": {"input_tokens": 10},
                },
            }
        )
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
        data = '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}'
        result = _clean_stream_frame("content_block_delta", data)
        assert result == data

    def test_passes_through_thinking_delta(self):
        data = '{"delta": {"type": "thinking_delta", "thinking": "hmm"}, "index": 0}'
        result = _clean_stream_frame("content_block_delta", data)
        assert result == data


class TestApplyThinkingBudgetNative:
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


class TestBetaMessagesEndpoint:
    @patch("router_maestro.server.routes.anthropic_beta._resolve_model")
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
        mock_resolve.return_value = ("github-copilot", "claude-sonnet-4.5", mock_provider)

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=lambda body, _: body,
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

    @patch("router_maestro.server.routes.anthropic_beta._parse_as_anthropic_request")
    @patch("router_maestro.server.routes.anthropic_beta.standard_messages")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_model")
    def test_fallback_for_non_claude_model(self, mock_resolve, mock_standard, mock_parse, client):
        """Non-Claude model falls back to the standard translation path."""
        mock_resolve.return_value = ("github-copilot", "gpt-5.4", None)
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

    @patch("router_maestro.server.routes.anthropic_beta._parse_as_anthropic_request")
    @patch("router_maestro.server.routes.anthropic_beta.standard_messages")
    @patch("router_maestro.server.routes.anthropic_beta._resolve_model")
    def test_fallback_for_non_copilot_provider(
        self, mock_resolve, mock_standard, mock_parse, client
    ):
        """Claude model on native Anthropic provider falls back."""
        mock_resolve.return_value = ("anthropic", "claude-sonnet-4.5", None)
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

    @patch("router_maestro.server.routes.anthropic_beta._resolve_model")
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
        mock_resolve.return_value = ("github-copilot", "claude-sonnet-4.5", mock_provider)

        with patch(
            "router_maestro.server.routes.anthropic_beta._apply_thinking_budget_native",
            side_effect=lambda body, _: body,
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
    def test_passthrough_count_tokens(self, mock_resolve, client):
        """Claude model count_tokens goes through native endpoint."""
        mock_provider = MagicMock(spec=CopilotProvider)
        mock_provider.ensure_token = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"input_tokens": 42}
        mock_provider._send_with_auth_retry = AsyncMock(return_value=mock_response)
        mock_resolve.return_value = ("github-copilot", "claude-sonnet-4.5", mock_provider)

        resp = client.post(
            "/api/anthropic/beta/v1/messages/count_tokens",
            json={
                "model": "claude-sonnet-4.5",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"input_tokens": 42}
