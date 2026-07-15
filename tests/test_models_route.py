"""Tests for server model-listing routes."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from rich.console import Console

from router_maestro.cli import model as model_cli
from router_maestro.config import PrioritiesConfig
from router_maestro.providers import ChatRequest, ChatResponse, Message, ModelInfo
from router_maestro.providers.base import BaseProvider
from router_maestro.routing.capabilities import Operation
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router, RouterOwner
from router_maestro.server.routes.admin import (
    list_models as list_admin_models,
)
from router_maestro.server.routes.admin import (
    router as admin_router,
)
from router_maestro.server.routes.anthropic import (
    list_models as list_anthropic_models,
)
from router_maestro.server.routes.anthropic import (
    router as anthropic_router,
)
from router_maestro.server.routes.models import list_models
from router_maestro.server.routes.models import router as openai_models_router
from router_maestro.utils.cache import TTLCache


class _GenerationListingRouter:
    def __init__(
        self,
        generation: str,
        *,
        entered: asyncio.Event | None = None,
        resume: asyncio.Event | None = None,
        fail: bool = False,
    ) -> None:
        self.generation = generation
        self.entered = entered
        self.resume = resume
        self.fail = fail
        self.close_count = 0
        self.closed = asyncio.Event()

    async def list_models(self) -> list[ModelInfo]:
        if self.entered is not None:
            self.entered.set()
        if self.resume is not None:
            await self.resume.wait()
        if self.fail:
            raise RuntimeError(f"catalog-{self.generation}-failed")
        return [
            ModelInfo(
                id=f"{self.generation.lower()}-one",
                name=f"{self.generation} One",
                provider="controlled",
            ),
            ModelInfo(
                id=f"{self.generation.lower()}-two",
                name=f"{self.generation} Two",
                provider="controlled",
            ),
        ]

    async def close(self) -> None:
        self.close_count += 1
        self.closed.set()


class _RecordingRouterOwner(RouterOwner[_GenerationListingRouter]):
    def __init__(self, factory) -> None:
        super().__init__(factory)
        self.acquire_count = 0

    async def acquire(self):
        self.acquire_count += 1
        return await super().acquire()


class _PoisonedGlobalRouter:
    async def list_models(self):
        raise AssertionError("server model listings must not use the global Router singleton")


def _model_listing_app(owner: RouterOwner) -> FastAPI:
    app = FastAPI()
    app.state.router_owner = owner
    app.include_router(openai_models_router)
    app.include_router(anthropic_router)
    app.include_router(admin_router)
    return app


@pytest.mark.asyncio
async def test_all_http_model_lists_switch_owner_generation_and_ignore_global_singleton(
    monkeypatch,
):
    routers: dict[str, _GenerationListingRouter] = {}

    def factory(generation):
        router = _GenerationListingRouter(generation)
        routers[generation] = router
        return router

    owner = RouterOwner(factory)
    await owner.start("A")
    monkeypatch.setattr("router_maestro.routing.router._router_instance", _PoisonedGlobalRouter())
    transport = httpx.ASGITransport(app=_model_listing_app(owner), raise_app_exceptions=False)

    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            openai_a = await client.get("/api/openai/v1/models")
            anthropic_a = await client.get("/api/anthropic/v1/models", params={"limit": 1})
            admin_a = await client.get("/api/admin/models")

            assert openai_a.status_code == anthropic_a.status_code == admin_a.status_code == 200
            assert [model["id"] for model in openai_a.json()["data"]] == [
                "controlled/a-one",
                "controlled/a-two",
            ]
            assert anthropic_a.json()["first_id"] == "controlled/a-one"
            assert anthropic_a.json()["last_id"] == "controlled/a-one"
            assert anthropic_a.json()["has_more"] is True
            assert [model["id"] for model in admin_a.json()["models"]] == [
                "controlled/a-one",
                "controlled/a-two",
            ]

            await owner.rebuild("B")

            openai_b = await client.get("/api/openai/v1/models")
            anthropic_b = await client.get("/api/anthropic/v1/models", params={"limit": 1})
            admin_b = await client.get("/api/admin/models")

        assert [model["id"] for model in openai_b.json()["data"]] == [
            "controlled/b-one",
            "controlled/b-two",
        ]
        assert anthropic_b.json()["first_id"] == "controlled/b-one"
        assert anthropic_b.json()["last_id"] == "controlled/b-one"
        assert anthropic_b.json()["has_more"] is True
        assert [model["id"] for model in admin_b.json()["models"]] == [
            "controlled/b-one",
            "controlled/b-two",
        ]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_http_model_listing_releases_owner_lease_when_catalog_raises(monkeypatch):
    routers: dict[str, _GenerationListingRouter] = {}

    def factory(generation):
        router = _GenerationListingRouter(generation, fail=generation == "A")
        routers[generation] = router
        return router

    owner = _RecordingRouterOwner(factory)
    await owner.start("A")
    monkeypatch.setattr("router_maestro.routing.router._router_instance", _PoisonedGlobalRouter())
    transport = httpx.ASGITransport(app=_model_listing_app(owner), raise_app_exceptions=False)

    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/openai/v1/models")
        assert response.status_code == 500
        assert owner.acquire_count == 1

        await owner.rebuild("B")
        await asyncio.wait_for(routers["A"].closed.wait(), timeout=1)

        assert routers["A"].close_count == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_inflight_http_model_listing_keeps_old_generation_until_dependency_releases(
    monkeypatch,
):
    entered = asyncio.Event()
    resume = asyncio.Event()
    routers: dict[str, _GenerationListingRouter] = {}

    def factory(generation):
        router = _GenerationListingRouter(
            generation,
            entered=entered if generation == "A" else None,
            resume=resume if generation == "A" else None,
        )
        routers[generation] = router
        return router

    owner = RouterOwner(factory)
    await owner.start("A")
    monkeypatch.setattr("router_maestro.routing.router._router_instance", _PoisonedGlobalRouter())
    transport = httpx.ASGITransport(app=_model_listing_app(owner), raise_app_exceptions=False)

    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            old_request = asyncio.create_task(client.get("/api/openai/v1/models"))
            await asyncio.wait_for(entered.wait(), timeout=1)

            await owner.rebuild("B")
            assert routers["A"].close_count == 0

            new_response = await client.get("/api/openai/v1/models")
            assert [model["id"] for model in new_response.json()["data"]] == [
                "controlled/b-one",
                "controlled/b-two",
            ]
            assert routers["A"].close_count == 0

            resume.set()
            old_response = await old_request

        assert [model["id"] for model in old_response.json()["data"]] == [
            "controlled/a-one",
            "controlled/a-two",
        ]
        assert routers["A"].close_count == 1
    finally:
        resume.set()
        await owner.close()


class _FakeRouter:
    async def list_models(self):
        return [
            ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai"),
            ModelInfo(id="gpt-4o", name="GPT-4o", provider="github-copilot"),
        ]


@pytest.mark.anyio
async def test_openai_models_route_uses_explicitly_injected_router():
    """A direct handler call reads the injected Router argument."""
    fake_router = _FakeRouter()

    response = await list_models(fake_router)

    assert [model.id for model in response.data] == [
        "openai/gpt-4o",
        "github-copilot/gpt-4o",
    ]


@pytest.mark.anyio
async def test_public_model_lists_encode_provider_namespaced_upstream_id():
    class _NamespacedRouter:
        async def list_models(self):
            return [ModelInfo(id="openrouter/auto", name="Auto", provider="openrouter")]

    model_router = _NamespacedRouter()
    openai_response = await list_models(model_router)
    anthropic_response = await list_anthropic_models(model_router=model_router)
    admin_response = await list_admin_models(model_router)

    expected = ["openrouter/openrouter/auto"]
    assert [model.id for model in openai_response.data] == expected
    assert [model.id for model in anthropic_response.data] == expected
    assert [model.id for model in admin_response.models] == expected


@pytest.mark.anyio
async def test_public_model_lists_do_not_double_prefix_qualified_catalog_ids():
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
    openai_response = await list_models(model_router)
    anthropic_response = await list_anthropic_models(model_router=model_router)
    admin_response = await list_admin_models(model_router)

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
async def test_public_model_lists_round_trip_to_same_provider_with_bare_upstream_id():
    model_router = _round_trip_router()
    openai_response = await list_models(model_router)
    anthropic_response = await list_anthropic_models(model_router=model_router)
    admin_response = await list_admin_models(model_router)

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
