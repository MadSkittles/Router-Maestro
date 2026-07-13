"""Tests for the mid-conversation-system beta rejection path.

When Claude Code 2.1.x sends the ``mid-conversation-system-2026-04-07``
beta header for ``claude-opus-4-8`` (and other models that default it on),
inline ``role:"system"`` messages MUST be rejected with a 400 carrying the
beta header string. Claude Code's ``Mk8`` error classifier reads that 400
and falls back to ``<system-reminder>`` blocks while sticky-rejecting the
beta for the rest of the session, which preserves prompt-cache locality.

Generic OpenAI-shaped clients (Cline, Aider, etc.) that send inline
``role:"system"`` WITHOUT the beta header still get the silent-hoist
behavior — they have no fallback to negotiate with.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.providers.base import ChatResponse
from router_maestro.routing.capabilities import CapabilitySupport
from router_maestro.server.routes.anthropic import (
    _MID_CONV_SYSTEM_BETA,
    router,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _add_planning_contract(
    model_router: MagicMock,
    *,
    provider: str,
    model: str,
) -> object:
    """Give a route-focused mock the Router planning seam used by Messages."""
    route_plan = MagicMock()
    route_plan.primary.capabilities.feature.return_value = CapabilitySupport.UNKNOWN
    route_plan.primary.capabilities.max_output_tokens = None
    route_plan.primary.model.qualified_id = f"{provider}/{model}"
    prepared_plan = object()
    model_router.plan_chat_completion = AsyncMock(return_value=route_plan)
    model_router.prepare_planned_chat_completion = MagicMock(return_value=prepared_plan)
    return prepared_plan


@pytest.fixture
def mock_router_success():
    """A router that returns a trivial assistant response so non-rejected
    requests can complete end-to-end through the route."""
    response = ChatResponse(
        content="ok",
        model="claude-opus-4-8",
        finish_reason="stop",
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )
    mock = MagicMock()
    mock.chat_completion = AsyncMock(return_value=(response, "anthropic"))
    mock._resolve_provider = AsyncMock(return_value=("anthropic", "claude-opus-4-8", MagicMock()))
    mock.get_model_info = AsyncMock(return_value=None)
    _add_planning_contract(
        mock,
        provider="anthropic",
        model="claude-opus-4-8",
    )
    return mock


def _claude_code_payload(with_inline_system: bool) -> dict:
    """Build a minimal AnthropicMessagesRequest payload."""
    messages: list[dict] = [
        {"role": "user", "content": "Hi"},
    ]
    if with_inline_system:
        messages.append({"role": "system", "content": "Mid-conv reminder."})
    messages.append({"role": "assistant", "content": "Hello."})
    messages.append({"role": "user", "content": "Tell me more."})
    return {
        "model": "claude-opus-4-8",
        "max_tokens": 16,
        "messages": messages,
    }


class TestMidConvSystemRejection:
    def test_inline_system_with_beta_header_returns_400(self, client, mock_router_success):
        """With both inline system + beta header, return 400 in the shape
        Claude Code's Mk8 detector recognises."""
        with patch(
            "router_maestro.server.routes.anthropic.get_router",
            return_value=mock_router_success,
        ):
            response = client.post(
                "/api/anthropic/v1/messages",
                json=_claude_code_payload(with_inline_system=True),
                headers={"anthropic-beta": _MID_CONV_SYSTEM_BETA},
            )
        assert response.status_code == 400
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        # Mk8 trigger: message must contain BOTH the beta header value
        # and the literal string "anthropic-beta" so the client detects
        # this as a beta-rejection and not a generic 400.
        assert _MID_CONV_SYSTEM_BETA in body["error"]["message"]
        assert "anthropic-beta" in body["error"]["message"]

    def test_inline_system_with_multiple_betas_still_rejected(self, client, mock_router_success):
        """Beta header can be comma-separated; mid-conv-system must still match."""
        with patch(
            "router_maestro.server.routes.anthropic.get_router",
            return_value=mock_router_success,
        ):
            response = client.post(
                "/api/anthropic/v1/messages",
                json=_claude_code_payload(with_inline_system=True),
                headers={
                    "anthropic-beta": (
                        f"context-1m-2024-08-07,{_MID_CONV_SYSTEM_BETA},prompt-caching"
                    )
                },
            )
        assert response.status_code == 400

    def test_inline_system_without_beta_header_is_hoisted(self, client, mock_router_success):
        """Non-Claude-Code clients (no beta header) get the silent hoist
        path — should succeed, not 400, not 422."""
        with patch(
            "router_maestro.server.routes.anthropic.get_router",
            return_value=mock_router_success,
        ):
            response = client.post(
                "/api/anthropic/v1/messages",
                json=_claude_code_payload(with_inline_system=True),
            )
        # Should not be a beta-rejection 400 nor a 422 schema error.
        assert response.status_code == 200, response.text

    def test_beta_header_without_inline_system_passes_through(self, client, mock_router_success):
        """Claude Code may send the beta header speculatively even when
        the conversation has no system message yet. Don't reject those."""
        with patch(
            "router_maestro.server.routes.anthropic.get_router",
            return_value=mock_router_success,
        ):
            response = client.post(
                "/api/anthropic/v1/messages",
                json=_claude_code_payload(with_inline_system=False),
                headers={"anthropic-beta": _MID_CONV_SYSTEM_BETA},
            )
        assert response.status_code == 200, response.text

    def test_rejection_response_is_json_serialisable(self, client, mock_router_success):
        """Body must parse cleanly so SDK error wrappers don't choke."""
        with patch(
            "router_maestro.server.routes.anthropic.get_router",
            return_value=mock_router_success,
        ):
            response = client.post(
                "/api/anthropic/v1/messages",
                json=_claude_code_payload(with_inline_system=True),
                headers={"anthropic-beta": _MID_CONV_SYSTEM_BETA},
            )
        # raises if not valid JSON
        json.loads(response.content)
        assert response.headers["content-type"].startswith("application/json")

    def test_rejection_path_for_v1_alias(self, client, mock_router_success):
        """The /v1/messages alias must reject identically to the namespaced path."""
        with patch(
            "router_maestro.server.routes.anthropic.get_router",
            return_value=mock_router_success,
        ):
            response = client.post(
                "/v1/messages",
                json=_claude_code_payload(with_inline_system=True),
                headers={"anthropic-beta": _MID_CONV_SYSTEM_BETA},
            )
        assert response.status_code == 400


def test_standard_messages_routes_exact_older_dated_model_without_family_rewrite(client):
    older = "claude-sonnet-4-20250101"
    newer = "claude-sonnet-4-20260101"
    received_models: list[str] = []

    async def chat_completion(request, *, prepared_plan=None):
        received_models.append(request.model)
        selected = newer if request.model == "claude-sonnet-4" else request.model
        return (
            ChatResponse(
                content="ok",
                model=selected,
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            ),
            "anthropic",
        )

    model_router = MagicMock()
    model_router.chat_completion = AsyncMock(side_effect=chat_completion)
    model_router.get_model_info = AsyncMock(return_value=None)
    _add_planning_contract(
        model_router,
        provider="anthropic",
        model=older,
    )

    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        response = client.post(
            "/v1/messages",
            json={
                "model": older,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 200
    assert received_models == [older]
    assert response.json()["model"] == f"anthropic/{older}"


@pytest.mark.parametrize("upstream_model", ["shared-model", "first/shared-model"])
def test_standard_messages_qualifies_actual_provider_exactly_once(client, upstream_model):
    model_router = MagicMock()
    model_router.chat_completion = AsyncMock(
        return_value=(
            ChatResponse(
                content="ok",
                model=upstream_model,
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            ),
            "first",
        )
    )
    model_router.get_model_info = AsyncMock(return_value=None)
    _add_planning_contract(
        model_router,
        provider="first",
        model="shared-model",
    )

    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        response = client.post(
            "/v1/messages",
            json={
                "model": "shared-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["model"] == "first/shared-model"
