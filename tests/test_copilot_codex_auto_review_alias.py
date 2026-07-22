"""Internal alias: Codex's ``codex-auto-review`` guardian model → GHC.

OpenAI Codex's Auto-review mode issues Responses requests with model
``codex-auto-review``, a synthetic id that only exists on the ChatGPT/Codex
subscription backend — it is absent from GitHub Copilot's catalog and from the
public OpenAI API, so it otherwise 404s at routing. The Copilot provider
declares an internal alias (``COPILOT_MODEL_ALIASES``) that the router
normalizes to a real GHC model before route resolution. These tests pin that
contract at both the provider declaration and the two routing chokepoints
(``plan_route`` and ``_resolve_provider``), and prove the alias never leaks into
the public catalog.
"""

import pytest

from router_maestro.config import PrioritiesConfig
from router_maestro.providers import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ProviderError,
)
from router_maestro.providers.base import BaseProvider
from router_maestro.providers.copilot import COPILOT_MODEL_ALIASES, CopilotProvider
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.utils.cache import TTLCache

ALIAS = "codex-auto-review"
TARGET = "gpt-5.4-mini"


class AliasCopilotMock(BaseProvider):
    """Stand-in for the Copilot provider that declares the guardian alias.

    Mirrors the real ``CopilotProvider.model_aliases()`` contract without any
    network/auth dependency, and exposes ``gpt-5.4-mini`` with ``/responses``
    support so a normalized alias has a real catalog target to resolve against.
    """

    name = "github-copilot"

    def __init__(self, *, authenticated: bool = True) -> None:
        self._authenticated = authenticated

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(operations=frozenset(Operation))

    def is_authenticated(self) -> bool:
        return self._authenticated

    async def ensure_token(self) -> None:
        pass

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="x", model=request.model, finish_reason="stop")

    async def chat_completion_stream(self, request: ChatRequest):
        yield ChatResponse(content="x", model=request.model, finish_reason="stop")

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=TARGET,
                name="GPT-5.4 mini",
                provider="github-copilot",
                operation_capabilities={"responses": True, "chat": True},
            )
        ]

    def model_aliases(self):
        return {ALIAS: TARGET}


def _make_router(provider: BaseProvider) -> Router:
    """Build a Router (via ``__new__``) wired to a single provider, no I/O."""
    router = Router.__new__(Router)
    router.providers = {provider.name: provider}
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._model_aliases = None
    router._priorities_cache.set(PrioritiesConfig(priorities=[]))
    router._providers_ttl.set(True)
    return router


def test_copilot_provider_declares_guardian_alias() -> None:
    # The declaration lives with the provider that owns the target model.
    assert CopilotProvider().model_aliases()[ALIAS] == TARGET
    assert COPILOT_MODEL_ALIASES[ALIAS] == TARGET


@pytest.mark.asyncio
async def test_resolve_provider_normalizes_alias_to_target() -> None:
    router = _make_router(AliasCopilotMock())
    provider_name, upstream_id, provider = await router._resolve_provider(ALIAS)
    assert provider_name == "github-copilot"
    assert upstream_id == TARGET
    assert provider is router.providers["github-copilot"]


@pytest.mark.asyncio
async def test_resolve_provider_alias_is_case_insensitive() -> None:
    router = _make_router(AliasCopilotMock())
    provider_name, upstream_id, _ = await router._resolve_provider("Codex-Auto-Review")
    assert (provider_name, upstream_id) == ("github-copilot", TARGET)


@pytest.mark.asyncio
async def test_plan_route_resolves_alias_for_responses() -> None:
    router = _make_router(AliasCopilotMock())
    plan = await router.plan_route(ALIAS, Operation.RESPONSES)
    # Normalization flows the request through the explicit qualified-id path.
    assert plan.explicit is True
    assert plan.primary.model == ModelRef("github-copilot", TARGET)


@pytest.mark.asyncio
async def test_plan_route_alias_is_case_insensitive() -> None:
    router = _make_router(AliasCopilotMock())
    plan = await router.plan_route("CODEX-AUTO-REVIEW", Operation.RESPONSES)
    assert plan.primary.model == ModelRef("github-copilot", TARGET)


@pytest.mark.asyncio
async def test_alias_absent_from_public_catalog() -> None:
    # The alias is purely internal — it must never surface in the model list.
    router = _make_router(AliasCopilotMock())
    listed = await router.list_models()
    ids = {model.id for model in listed}
    assert TARGET in ids
    assert ALIAS not in ids
    assert not any(ALIAS in (model.id or "") for model in listed)


@pytest.mark.asyncio
async def test_unauthenticated_alias_still_404s_no_silent_pass() -> None:
    # Normalization must not fabricate success: with Copilot unauthenticated the
    # target is absent from cache, so the normalized id 404s naturally.
    router = _make_router(AliasCopilotMock(authenticated=False))
    with pytest.raises(ProviderError) as excinfo:
        await router._resolve_provider(ALIAS)
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_beta_responses_resolver_rewrites_model_to_target(monkeypatch) -> None:
    """Real-layer: the beta Responses route resolves the alias to the target.

    Drives ``_resolve_responses_model`` — the exact resolver the beta passthrough
    route feeds into ``body["model"] = resolution.actual_model`` — with a real
    Router. A request whose ``model`` is ``codex-auto-review`` must yield
    ``actual_model == "gpt-5.4-mini"``, i.e. the id sent upstream to GHC. This is
    the layer Codex's guardian request actually hits, not a helper below it.
    """
    from router_maestro.routing.capabilities import RequestFeatures
    from router_maestro.server.routes import openai_responses_beta as beta

    router = _make_router(AliasCopilotMock())
    monkeypatch.setattr(beta, "get_router", lambda: router)

    resolution = await beta._resolve_responses_model(ALIAS, Operation.RESPONSES, RequestFeatures())

    assert resolution.provider_name == "github-copilot"
    assert resolution.actual_model == TARGET
