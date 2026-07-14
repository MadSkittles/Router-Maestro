"""Tests for the OpenAI-compatible models route."""

from unittest.mock import patch

import pytest
from rich.console import Console

from router_maestro.cli import model as model_cli
from router_maestro.config import PrioritiesConfig
from router_maestro.providers import ChatRequest, ChatResponse, Message, ModelInfo
from router_maestro.providers.base import BaseProvider
from router_maestro.routing.capabilities import Operation
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.server.routes.admin import list_models as list_admin_models
from router_maestro.server.routes.anthropic import list_models as list_anthropic_models
from router_maestro.server.routes.models import list_models
from router_maestro.utils.cache import TTLCache


class _FakeRouter:
    async def list_models(self):
        return [
            ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai"),
            ModelInfo(id="gpt-4o", name="GPT-4o", provider="github-copilot"),
        ]


class _RouterLease:
    def __init__(self, router):
        self.router = router

    async def release(self):
        return None


class _RouterOwner:
    def __init__(self, router):
        self.router = router

    async def acquire(self):
        return _RouterLease(self.router)


@pytest.mark.anyio
async def test_openai_models_route_uses_routing_singleton(monkeypatch):
    """The models route should reuse the routing singleton instead of constructing Router."""
    fake_router = _FakeRouter()

    monkeypatch.setattr("router_maestro.routing.router._router_instance", fake_router)

    response = await list_models()

    assert [model.id for model in response.data] == [
        "openai/gpt-4o",
        "github-copilot/gpt-4o",
    ]


@pytest.mark.anyio
async def test_public_model_lists_encode_provider_namespaced_upstream_id(monkeypatch):
    class _NamespacedRouter:
        async def list_models(self):
            return [ModelInfo(id="openrouter/auto", name="Auto", provider="openrouter")]

    model_router = _NamespacedRouter()
    monkeypatch.setattr("router_maestro.routing.router._router_instance", model_router)

    openai_response = await list_models()
    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        anthropic_response = await list_anthropic_models()
    admin_response = await list_admin_models(_RouterOwner(model_router))

    expected = ["openrouter/openrouter/auto"]
    assert [model.id for model in openai_response.data] == expected
    assert [model.id for model in anthropic_response.data] == expected
    assert [model.id for model in admin_response.models] == expected


@pytest.mark.anyio
async def test_public_model_lists_do_not_double_prefix_qualified_catalog_ids(monkeypatch):
    class _QualifiedRouter:
        async def list_models(self):
            return [
                ModelInfo(
                    id="github-copilot/claude-sonnet-4.6",
                    name="Claude Sonnet 4.6",
                    provider="github-copilot",
                    id_is_qualified=True,
                )
            ]

    model_router = _QualifiedRouter()
    monkeypatch.setattr("router_maestro.routing.router._router_instance", model_router)

    openai_response = await list_models()
    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        anthropic_response = await list_anthropic_models()
    admin_response = await list_admin_models(_RouterOwner(model_router))

    assert [model.id for model in openai_response.data] == ["github-copilot/claude-sonnet-4.6"]
    assert [model.id for model in anthropic_response.data] == ["github-copilot/claude-sonnet-4.6"]
    assert [model.id for model in admin_response.models] == ["github-copilot/claude-sonnet-4.6"]


class _RoundTripProvider(BaseProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = ModelInfo(id="shared-model", name="Shared", provider=name)
        self.requested_models: list[str] = []

    def is_authenticated(self) -> bool:
        return True

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        self.requested_models.append(request.model)
        return ChatResponse(content=self.name, model=request.model, finish_reason="stop")

    async def chat_completion_stream(self, request: ChatRequest):
        raise AssertionError("not used")
        yield

    async def list_models(self) -> list[ModelInfo]:
        return [self.model]


def _round_trip_router() -> Router:
    router = Router.__new__(Router)
    first = _RoundTripProvider("first")
    second = _RoundTripProvider("second")
    router.providers = {"first": first, "second": second}
    router._models_cache = {
        "shared-model": ("first", first.model),
        "first/shared-model": ("first", first.model),
        "second/shared-model": ("second", second.model),
    }
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._models_cache_ttl.set(True)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache.set(PrioritiesConfig(priorities=[]))
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._providers_ttl.set(True)
    return router


@pytest.mark.anyio
async def test_public_model_lists_round_trip_to_same_provider_with_bare_upstream_id(monkeypatch):
    model_router = _round_trip_router()
    monkeypatch.setattr("router_maestro.routing.router._router_instance", model_router)

    openai_response = await list_models()
    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        anthropic_response = await list_anthropic_models()
    admin_response = await list_admin_models(_RouterOwner(model_router))

    for public_ids in (
        [model.id for model in openai_response.data],
        [model.id for model in anthropic_response.data],
        [model.id for model in admin_response.models],
    ):
        assert set(public_ids) == {"first/shared-model", "second/shared-model"}
        for public_id in public_ids:
            plan = await model_router.plan_route(public_id, Operation.CHAT)
            assert plan.primary.model.qualified_id == public_id

            response, provider_name = await model_router.chat_completion(
                ChatRequest(
                    model=public_id,
                    messages=[Message(role="user", content="hello")],
                ),
                fallback=False,
            )
            assert provider_name == public_id.split("/", 1)[0]
            assert response.model == "shared-model"

    for provider in model_router.providers.values():
        assert provider.requested_models == ["shared-model", "shared-model", "shared-model"]


def test_cli_model_list_does_not_double_qualify_public_id(monkeypatch):
    class _Client:
        async def list_models(self):
            return [{"provider": "openai", "id": "openai/gpt-4o", "name": "GPT-4o"}]

        async def get_priorities(self):
            return {"priorities": ["openai/gpt-4o"]}

    output = Console(record=True, width=120)
    monkeypatch.setattr(model_cli, "get_admin_client", _Client)
    monkeypatch.setattr(model_cli, "console", output)

    model_cli.list_models()

    rendered = output.export_text()
    assert "openai/gpt-4o" in rendered
    assert "openai/openai/gpt-4o" not in rendered
