"""Tests for the OpenAI-compatible models route."""

import pytest

from router_maestro.providers import ModelInfo
from router_maestro.server.routes.models import list_models


class _FakeRouter:
    async def list_models(self):
        return [ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")]


@pytest.mark.anyio
async def test_openai_models_route_uses_routing_singleton(monkeypatch):
    """The models route should reuse the routing singleton instead of constructing Router."""
    fake_router = _FakeRouter()

    monkeypatch.setattr("router_maestro.routing.router._router_instance", fake_router)

    response = await list_models()

    assert len(response.data) == 1
    assert response.data[0].id == "gpt-4o"
