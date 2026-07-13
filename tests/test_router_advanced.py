"""Advanced tests for the Router module with async operations."""

import pytest

from router_maestro.config import PrioritiesConfig
from router_maestro.providers import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    ResponsesRequest,
    ResponsesResponse,
)
from router_maestro.providers.base import BaseProvider
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef, qualify_model_id
from router_maestro.routing.router import (
    AUTO_ROUTE_MODEL,
    CACHE_TTL_SECONDS,
    Router,
    get_router,
    reset_router,
)
from router_maestro.utils.cache import TTLCache


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        authenticated: bool = True,
        models: list[ModelInfo] | None = None,
        fail_on_request: bool = False,
        fail_retryable: bool = True,
    ):
        self._name = name
        self._authenticated = authenticated
        self._models = models or [ModelInfo(id="test-model", name="Test Model", provider=name)]
        self._fail_on_request = fail_on_request
        self._fail_retryable = fail_retryable
        self._request_count = 0

    @property
    def name(self) -> str:
        return self._name

    def is_authenticated(self) -> bool:
        return self._authenticated

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            operations=frozenset(
                {
                    Operation.CHAT,
                    Operation.CHAT_STREAM,
                    Operation.RESPONSES,
                    Operation.RESPONSES_STREAM,
                }
            )
        )

    async def ensure_token(self) -> None:
        pass

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        self._request_count += 1
        if self._fail_on_request:
            raise ProviderError(
                "Mock provider failure", retryable=self._fail_retryable, status_code=500
            )
        return ChatResponse(
            content=f"Response from {self._name}",
            model=request.model,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    async def chat_completion_stream(self, request: ChatRequest):
        self._request_count += 1
        if self._fail_on_request:
            raise ProviderError(
                "Mock provider failure", retryable=self._fail_retryable, status_code=500
            )
        from router_maestro.providers import ChatStreamChunk

        yield ChatStreamChunk(content="Hello ", finish_reason=None)
        yield ChatStreamChunk(content="world", finish_reason="stop")

    async def list_models(self) -> list[ModelInfo]:
        return self._models

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        self._request_count += 1
        if self._fail_on_request:
            raise ProviderError(
                "Mock provider failure", retryable=self._fail_retryable, status_code=500
            )
        return ResponsesResponse(
            content=f"Response from {self._name}",
            model=request.model,
            usage={"input_tokens": 10, "output_tokens": 20},
        )

    async def responses_completion_stream(self, request: ResponsesRequest):
        self._request_count += 1
        if self._fail_on_request:
            raise ProviderError(
                "Mock provider failure", retryable=self._fail_retryable, status_code=500
            )
        from router_maestro.providers import ResponsesStreamChunk

        yield ResponsesStreamChunk(content="Hello ", finish_reason=None)
        yield ResponsesStreamChunk(content="world", finish_reason="stop")


def _init_router_empty(router: Router) -> None:
    """Initialize a Router (created via __new__) with empty caches."""
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)


def _mark_providers_fresh(router: Router) -> None:
    """Mark providers as freshly loaded so _ensure_providers_fresh is a no-op."""
    router._providers_ttl.set(True)


def _set_priorities(router: Router, config: PrioritiesConfig) -> None:
    """Populate the priorities cache so _get_priorities_config returns the config."""
    router._priorities_cache.set(config)


def _mark_models_cached(router: Router) -> None:
    """Mark the models cache as valid so _ensure_models_cache is a no-op."""
    router._models_cache_ttl.set(True)


class TestRouterSingleton:
    """Tests for Router singleton pattern."""

    def test_get_router_creates_instance(self):
        """Test that get_router creates a singleton instance."""
        reset_router()
        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

    def test_reset_router_clears_instance(self):
        """Test that reset_router clears the singleton."""
        reset_router()
        router1 = get_router()
        reset_router()
        router2 = get_router()
        assert router1 is not router2


class TestRouterCacheManagement:
    """Tests for Router cache management."""

    @pytest.fixture
    def router_with_mock(self):
        """Create a router with mock providers."""
        router = Router.__new__(Router)
        router.providers = {}
        _init_router_empty(router)
        return router

    def test_cache_ttl_constant(self):
        """Test that cache TTL is 5 minutes."""
        assert CACHE_TTL_SECONDS == 300

    def test_auto_route_model_constant(self):
        """Test the auto-route model name."""
        assert AUTO_ROUTE_MODEL == "router-maestro"

    @pytest.mark.asyncio
    async def test_ensure_models_cache_populates(self, router_with_mock):
        """Test that _ensure_models_cache populates the cache."""
        mock_provider = MockProvider(
            name="test-provider",
            models=[
                ModelInfo(id="model-1", name="Model 1", provider="test-provider"),
                ModelInfo(id="model-2", name="Model 2", provider="test-provider"),
            ],
        )
        router_with_mock.providers = {"test-provider": mock_provider}
        _mark_providers_fresh(router_with_mock)

        await router_with_mock._ensure_models_cache()

        assert router_with_mock._models_cache_ttl.is_valid
        assert "model-1" in router_with_mock._models_cache
        assert "model-2" in router_with_mock._models_cache
        assert "test-provider/model-1" in router_with_mock._models_cache
        assert "test-provider/model-2" in router_with_mock._models_cache

    @pytest.mark.asyncio
    async def test_qualified_catalog_id_is_normalized_once_and_round_trips(self, router_with_mock):
        qualified = ModelInfo(
            id="p/m",
            name="Qualified",
            provider="p",
            id_is_qualified=True,
        )
        duplicate_bare = ModelInfo(id="m", name="Bare duplicate", provider="p")
        provider = MockProvider(name="p", models=[qualified, duplicate_bare])
        router_with_mock.providers = {"p": provider}
        _mark_providers_fresh(router_with_mock)
        _set_priorities(router_with_mock, PrioritiesConfig(priorities=[]))

        await router_with_mock._ensure_models_cache()

        assert set(router_with_mock._models_cache) == {"m", "p/m"}
        assert "p/p/m" not in router_with_mock._models_cache
        assert router_with_mock._models_cache["m"] is router_with_mock._models_cache["p/m"]
        cached_model = router_with_mock._models_cache["p/m"][1]
        assert cached_model.id == "m"
        assert cached_model is not qualified
        assert qualified.id == "p/m"

        public_models = await router_with_mock.list_models()
        assert [(model.provider, model.id) for model in public_models] == [("p", "m")]
        public_id = qualify_model_id(public_models[0].provider, public_models[0].id)
        assert public_id == "p/m"

        response, provider_name = await router_with_mock.chat_completion(
            ChatRequest(model=public_id, messages=[Message(role="user", content="hello")]),
            fallback=False,
        )
        assert provider_name == "p"
        assert response.model == "m"

    @pytest.mark.asyncio
    async def test_namespaced_upstream_catalog_id_round_trips_without_hiding_valid_sibling(
        self, router_with_mock
    ):
        namespaced = ModelInfo(id="q/foreign", name="Foreign", provider="p")
        valid = ModelInfo(id="local", name="Local", provider="p")
        router_with_mock.providers = {
            "p": MockProvider(name="p", models=[namespaced, valid]),
        }
        _mark_providers_fresh(router_with_mock)
        _set_priorities(router_with_mock, PrioritiesConfig(priorities=[]))

        await router_with_mock._ensure_models_cache()

        assert set(router_with_mock._models_cache) == {
            "q/foreign",
            "p/q/foreign",
            "local",
            "p/local",
        }
        response, provider_name = await router_with_mock.chat_completion(
            ChatRequest(
                model="p/q/foreign",
                messages=[Message(role="user", content="hello")],
            ),
            fallback=False,
        )

        assert provider_name == "p"
        assert response.model == "q/foreign"
        listed = await router_with_mock.list_models()
        assert len(listed) == 2
        assert {(model.provider, model.id) for model in listed} == {
            ("p", "local"),
            ("p", "q/foreign"),
        }

    @pytest.mark.asyncio
    async def test_upstream_id_starting_with_provider_name_has_unambiguous_public_id(
        self, router_with_mock
    ):
        upstream = ModelInfo(id="p/auto", name="Provider-namespaced", provider="p")
        router_with_mock.providers = {"p": MockProvider(name="p", models=[upstream])}
        _mark_providers_fresh(router_with_mock)
        _set_priorities(router_with_mock, PrioritiesConfig(priorities=[]))

        await router_with_mock._ensure_models_cache()

        listed = await router_with_mock.list_models()
        assert [(model.provider, model.id) for model in listed] == [("p", "p/auto")]
        plan = await router_with_mock.plan_route("p/p/auto", Operation.CHAT)
        assert plan.primary.model == ModelRef("p", "p/auto")

    @pytest.mark.asyncio
    async def test_exact_slash_alias_stays_bare_when_namespace_is_a_provider(
        self, router_with_mock
    ):
        upstream_id = "openrouter/auto"
        openrouter_model = ModelInfo(id=upstream_id, name="Auto", provider="openrouter")
        preferred_model = ModelInfo(id=upstream_id, name="Auto", provider="preferred")
        router_with_mock.providers = {
            "openrouter": MockProvider(name="openrouter", models=[openrouter_model]),
            "preferred": MockProvider(name="preferred", models=[preferred_model]),
        }
        _mark_providers_fresh(router_with_mock)
        _set_priorities(
            router_with_mock,
            PrioritiesConfig(
                priorities=[
                    "preferred/openrouter/auto",
                    "openrouter/openrouter/auto",
                ]
            ),
        )

        plan = await router_with_mock.plan_route(upstream_id, Operation.CHAT)

        assert plan.explicit is False
        assert plan.primary.model == ModelRef("preferred", upstream_id)

    @pytest.mark.asyncio
    async def test_slashless_fuzzy_alias_preserves_raw_upstream_namespace(
        self,
        router_with_mock,
    ):
        upstream_id = "meta-llama/llama-3.1-8b-instruct"
        model = ModelInfo(id=upstream_id, name="Llama 3.1 8B", provider="openrouter")
        router_with_mock.providers = {
            "openrouter": MockProvider(name="openrouter", models=[model]),
        }
        _mark_providers_fresh(router_with_mock)
        _set_priorities(
            router_with_mock,
            PrioritiesConfig(priorities=[f"openrouter/{upstream_id}"]),
        )

        plan = await router_with_mock.plan_route("llama 3.1 8b instruct", Operation.CHAT)

        assert plan.explicit is False
        assert plan.primary.model == ModelRef("openrouter", upstream_id)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "malformed_id",
        ["", "   ", "p/", "p/   "],
        ids=["empty", "whitespace", "empty-qualified", "whitespace-qualified"],
    )
    async def test_malformed_catalog_identity_is_skipped_without_hiding_healthy_sibling(
        self,
        router_with_mock,
        malformed_id: str,
    ):
        malformed = ModelInfo(id=malformed_id, name="Malformed", provider="p")
        healthy = ModelInfo(id="healthy", name="Healthy", provider="p")
        router_with_mock.providers = {
            "p": MockProvider(name="p", models=[malformed, healthy]),
        }
        _mark_providers_fresh(router_with_mock)
        _set_priorities(router_with_mock, PrioritiesConfig(priorities=[]))

        await router_with_mock._ensure_models_cache()

        assert set(router_with_mock._models_cache) == {"healthy", "p/healthy"}
        assert [(model.provider, model.id) for model in await router_with_mock.list_models()] == [
            ("p", "healthy")
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_id",
        ["", "   ", "p/", "p/   "],
        ids=["empty", "whitespace", "empty-qualified", "whitespace-qualified"],
    )
    async def test_empty_explicit_model_is_typed_client_error(
        self,
        router_with_mock,
        model_id: str,
    ):
        router_with_mock.providers = {
            "p": MockProvider(
                name="p",
                models=[ModelInfo(id="healthy", name="Healthy", provider="p")],
            )
        }
        _mark_providers_fresh(router_with_mock)
        _set_priorities(router_with_mock, PrioritiesConfig(priorities=[]))

        with pytest.raises(ProviderError) as exc_info:
            await router_with_mock.plan_route(model_id, Operation.CHAT)

        assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_provider_identity_is_isolated_from_healthy_provider(
        self,
        router_with_mock,
    ):
        class InvalidProvider(MockProvider):
            listed = False

            async def list_models(self) -> list[ModelInfo]:
                self.listed = True
                return await super().list_models()

        invalid = InvalidProvider(
            name="team/p",
            models=[ModelInfo(id="model", name="Invalid", provider="team/p")],
        )
        healthy = MockProvider(
            name="healthy",
            models=[ModelInfo(id="model", name="Healthy", provider="healthy")],
        )
        router_with_mock.providers = {invalid.name: invalid, healthy.name: healthy}
        _mark_providers_fresh(router_with_mock)
        _set_priorities(router_with_mock, PrioritiesConfig(priorities=[]))

        await router_with_mock._ensure_models_cache()

        assert set(router_with_mock._models_cache) == {"model", "healthy/model"}
        assert [(model.provider, model.id) for model in await router_with_mock.list_models()] == [
            ("healthy", "model")
        ]
        # Runtime registries can still be populated programmatically, bypassing
        # ProvidersConfig validation. Router must isolate their catalog entries
        # without hiding healthy providers, even after the catalog was queried.
        assert invalid.listed is True

    @pytest.mark.asyncio
    async def test_public_model_results_are_deep_defensive_copies(self, router_with_mock):
        catalog_model = ModelInfo(
            id="m",
            name="Model",
            provider="p",
            reasoning_effort_values=["low"],
            operation_capabilities={Operation.CHAT.value: True},
            feature_capabilities={"tools": True},
        )
        router_with_mock.providers = {"p": MockProvider(name="p", models=[catalog_model])}
        _mark_providers_fresh(router_with_mock)
        _set_priorities(router_with_mock, PrioritiesConfig(priorities=[]))

        listed = (await router_with_mock.list_models())[0]
        looked_up = await router_with_mock.get_model_info("p/m")
        assert looked_up is not None
        listed.id = "changed"
        listed.reasoning_effort_values.append("high")
        listed.operation_capabilities[Operation.CHAT.value] = False
        looked_up.feature_capabilities["tools"] = False

        plan = await router_with_mock.plan_route("p/m", Operation.CHAT)
        cached = router_with_mock._models_cache["p/m"][1]
        assert plan.primary.model.qualified_id == "p/m"
        assert cached.reasoning_effort_values == ["low"]
        assert cached.operation_capabilities == {Operation.CHAT.value: True}
        assert cached.feature_capabilities == {"tools": True}

    @pytest.mark.asyncio
    async def test_cache_skips_unauthenticated_providers(self, router_with_mock):
        """Test that cache skips unauthenticated providers."""
        unauth_provider = MockProvider(name="unauth", authenticated=False)
        router_with_mock.providers = {"unauth": unauth_provider}
        _mark_providers_fresh(router_with_mock)

        await router_with_mock._ensure_models_cache()

        assert router_with_mock._models_cache_ttl.is_valid
        assert len(router_with_mock._models_cache) == 0


class TestRouterModelResolutionAsync:
    """Tests for async model resolution."""

    @pytest.fixture
    def router_with_providers(self):
        """Create a router with multiple providers."""
        router = Router.__new__(Router)
        router.providers = {
            "primary": MockProvider(
                name="primary",
                models=[ModelInfo(id="gpt-4o", name="GPT-4o", provider="primary")],
            ),
            "secondary": MockProvider(
                name="secondary",
                models=[ModelInfo(id="claude-3", name="Claude 3", provider="secondary")],
            ),
        }
        _init_router_empty(router)
        _mark_providers_fresh(router)
        return router

    @pytest.mark.asyncio
    async def test_find_model_with_provider_prefix(self, router_with_providers):
        """Test finding model with provider prefix."""
        result = await router_with_providers._find_model_in_cache("primary/gpt-4o")
        assert result is not None
        provider_name, model_id, provider = result
        assert provider_name == "primary"
        assert model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_find_model_without_prefix(self, router_with_providers):
        """Test finding model without provider prefix."""
        result = await router_with_providers._find_model_in_cache("gpt-4o")
        assert result is not None
        provider_name, model_id, provider = result
        assert provider_name == "primary"
        assert model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_model_not_found_returns_none(self, router_with_providers):
        """Test that non-existent model returns None."""
        result = await router_with_providers._find_model_in_cache("nonexistent-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_provider_scoped_fuzzy_match_uses_bare_upstream_model(self):
        """Provider-scoped fuzzy matches must not pass provider/model upstream."""
        router = Router.__new__(Router)
        anthropic = MockProvider(
            name="anthropic",
            models=[
                ModelInfo(
                    id="claude-sonnet-4-5-20250929",
                    name="Claude Sonnet 4.5",
                    provider="anthropic",
                )
            ],
        )
        router.providers = {"anthropic": anthropic}
        _init_router_empty(router)
        router._models_cache = {
            "claude-sonnet-4-5-20250929": ("anthropic", anthropic._models[0]),
            "anthropic/claude-sonnet-4-5-20250929": ("anthropic", anthropic._models[0]),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)

        result = await router._find_model_in_cache("anthropic/sonnet-4-5")

        assert result is not None
        provider_name, model_id, _provider = result
        assert provider_name == "anthropic"
        assert model_id == "claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_get_model_info_uses_fuzzy_match_for_dash_dot_alias(self):
        """Metadata lookup must match the same dash/dot aliases routing accepts."""
        router = Router.__new__(Router)
        opus = ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            max_output_tokens=64000,
            supports_thinking=True,
        )
        router.providers = {"github-copilot": MockProvider(name="github-copilot", models=[opus])}
        _init_router_empty(router)
        router._models_cache = {
            "claude-opus-4.6": ("github-copilot", opus),
            "github-copilot/claude-opus-4.6": ("github-copilot", opus),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)

        result = await router.get_model_info("claude-opus-4-6")

        assert result == opus
        assert result is not opus
        assert result.max_output_tokens == 64000

    @pytest.mark.asyncio
    async def test_resolve_auto_route_model(self, router_with_providers):
        """Test resolving auto-route model."""
        _set_priorities(
            router_with_providers,
            PrioritiesConfig(priorities=["primary/gpt-4o", "secondary/claude-3"]),
        )

        result = await router_with_providers._get_auto_route_model()
        assert result is not None
        provider_name, model_id, provider = result
        assert provider_name == "primary"
        assert model_id == "gpt-4o"


class TestRouterChatCompletion:
    """Tests for Router chat completion."""

    @pytest.fixture
    def router_with_provider(self):
        """Create a router with a single mock provider."""
        router = Router.__new__(Router)
        mock = MockProvider(
            name="test",
            models=[ModelInfo(id="test-model", name="Test", provider="test")],
        )
        router.providers = {"test": mock}
        _init_router_empty(router)
        router._models_cache = {
            "test-model": ("test", ModelInfo(id="test-model", name="Test", provider="test")),
            "test/test-model": ("test", ModelInfo(id="test-model", name="Test", provider="test")),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))
        return router, mock

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, router_with_provider):
        """Test successful chat completion."""
        router, mock = router_with_provider
        request = ChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response, provider_name = await router.chat_completion(request)

        assert provider_name == "test"
        assert response.content == "Response from test"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_completion_stream_success(self, router_with_provider):
        """Test successful streaming chat completion."""
        router, mock = router_with_provider
        request = ChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=True,
        )
        stream, provider_name = await router.chat_completion_stream(request)

        assert provider_name == "test"
        chunks = [chunk async for chunk in stream]
        assert len(chunks) == 2
        assert chunks[0].content == "Hello "
        assert chunks[1].finish_reason == "stop"


class TestRouterResponsesAPI:
    """Tests for Router Responses API."""

    @pytest.fixture
    def router_with_provider(self):
        """Create a router with a mock provider supporting Responses API."""
        router = Router.__new__(Router)
        mock = MockProvider(
            name="test",
            models=[ModelInfo(id="test-model", name="Test", provider="test")],
        )
        router.providers = {"test": mock}
        _init_router_empty(router)
        router._models_cache = {
            "test-model": ("test", ModelInfo(id="test-model", name="Test", provider="test")),
            "test/test-model": ("test", ModelInfo(id="test-model", name="Test", provider="test")),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))
        return router, mock

    @pytest.mark.asyncio
    async def test_responses_completion_success(self, router_with_provider):
        """Test successful responses completion."""
        router, mock = router_with_provider
        request = ResponsesRequest(
            model="test-model",
            input="Hello",
        )
        response, provider_name = await router.responses_completion(request)

        assert provider_name == "test"
        assert response.content == "Response from test"

    @pytest.mark.asyncio
    async def test_responses_completion_stream_success(self, router_with_provider):
        """Test successful streaming responses completion."""
        router, mock = router_with_provider
        request = ResponsesRequest(
            model="test-model",
            input="Hello",
            stream=True,
        )
        stream, provider_name = await router.responses_completion_stream(request)

        assert provider_name == "test"
        chunks = [chunk async for chunk in stream]
        assert len(chunks) == 2


class TestRouterResponseOwnership:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("protocol", ["chat", "responses"])
    async def test_router_does_not_mutate_provider_owned_response(self, protocol):
        provider_response = (
            ChatResponse(content="cached", model="shared-model")
            if protocol == "chat"
            else ResponsesResponse(content="cached", model="shared-model")
        )

        class SingletonProvider(MockProvider):
            async def chat_completion(self, request: ChatRequest) -> ChatResponse:
                return provider_response

            async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
                return provider_response

        model = ModelInfo(id="shared-model", name="Shared", provider="singleton")
        provider = SingletonProvider(name="singleton", models=[model])
        router = Router.__new__(Router)
        router.providers = {"singleton": provider}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("singleton", model),
            "singleton/shared-model": ("singleton", model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))

        if protocol == "chat":
            routed, _provider_name = await router.chat_completion(
                ChatRequest(
                    model="singleton/shared-model",
                    messages=[Message(role="user", content="Hello")],
                )
            )
        else:
            routed, _provider_name = await router.responses_completion(
                ResponsesRequest(model="singleton/shared-model", input="Hello")
            )

        assert routed is not provider_response
        assert routed.selected_model == ModelRef("singleton", "shared-model")
        assert provider_response.selected_model is None


class TestRouterFallback:
    """Tests for Router fallback behavior."""

    @pytest.fixture
    def router_with_fallback(self):
        """Create a router with primary (failing) and secondary providers."""
        router = Router.__new__(Router)
        primary = MockProvider(
            name="primary",
            models=[ModelInfo(id="model-1", name="Model 1", provider="primary")],
            fail_on_request=True,
            fail_retryable=True,
        )
        secondary = MockProvider(
            name="secondary",
            models=[ModelInfo(id="model-1", name="Model 1", provider="secondary")],
            fail_on_request=False,
        )
        router.providers = {"primary": primary, "secondary": secondary}
        _init_router_empty(router)
        router._models_cache = {
            "model-1": ("primary", ModelInfo(id="model-1", name="Model 1", provider="primary")),
            "primary/model-1": (
                "primary",
                ModelInfo(id="model-1", name="Model 1", provider="primary"),
            ),
            "secondary/model-1": (
                "secondary",
                ModelInfo(id="model-1", name="Model 1", provider="secondary"),
            ),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(
                priorities=["primary/model-1", "secondary/model-1"],
                fallback={"strategy": "priority", "maxRetries": 3},
            ),
        )
        return router, primary, secondary

    @pytest.mark.asyncio
    async def test_chat_stream_falls_back_when_primary_fails_before_first_chunk(
        self, router_with_fallback
    ):
        """Streaming fallback should handle retryable errors raised when iteration starts."""
        router, primary, secondary = router_with_fallback
        request = ChatRequest(
            model="model-1",
            messages=[Message(role="user", content="Hello")],
            stream=True,
        )

        stream, provider_name = await router.chat_completion_stream(request)
        chunks = [chunk async for chunk in stream]

        assert provider_name == "secondary"
        assert stream.selected_model == ModelRef("secondary", "model-1")
        assert primary._request_count == 1
        assert secondary._request_count == 1
        assert chunks[0].content == "Hello "

    @pytest.mark.asyncio
    async def test_responses_stream_falls_back_when_primary_fails_before_first_chunk(
        self, router_with_fallback
    ):
        """Responses streams should use the same initial-error fallback behavior."""
        router, primary, secondary = router_with_fallback
        request = ResponsesRequest(model="model-1", input="Hello", stream=True)

        stream, provider_name = await router.responses_completion_stream(request)
        chunks = [chunk async for chunk in stream]

        assert provider_name == "secondary"
        assert stream.selected_model == ModelRef("secondary", "model-1")
        assert primary._request_count == 1
        assert secondary._request_count == 1
        assert chunks[0].content == "Hello "


class TestRouterListModels:
    """Tests for Router list_models."""

    @pytest.fixture
    def router_with_models(self):
        """Create a router with multiple models."""
        router = Router.__new__(Router)
        router.providers = {
            "provider1": MockProvider(
                name="provider1",
                models=[
                    ModelInfo(id="model-a", name="Model A", provider="provider1"),
                    ModelInfo(id="model-b", name="Model B", provider="provider1"),
                ],
            ),
        }
        _init_router_empty(router)
        router._models_cache = {
            "model-a": (
                "provider1",
                ModelInfo(id="model-a", name="Model A", provider="provider1"),
            ),
            "model-b": (
                "provider1",
                ModelInfo(id="model-b", name="Model B", provider="provider1"),
            ),
            "provider1/model-a": (
                "provider1",
                ModelInfo(id="model-a", name="Model A", provider="provider1"),
            ),
            "provider1/model-b": (
                "provider1",
                ModelInfo(id="model-b", name="Model B", provider="provider1"),
            ),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(priorities=["provider1/model-b", "provider1/model-a"]),
        )
        return router

    @pytest.mark.asyncio
    async def test_list_models_returns_all(self, router_with_models):
        """Test that list_models returns all models."""
        models = await router_with_models.list_models()
        assert len(models) == 2

    @pytest.mark.asyncio
    async def test_list_models_respects_priority(self, router_with_models):
        """Test that list_models respects priority order."""
        models = await router_with_models.list_models()
        # Priority is model-b first
        assert models[0].id == "model-b"
        assert models[1].id == "model-a"

    @pytest.mark.asyncio
    async def test_duplicate_upstream_ids_list_and_route_with_qualified_identity(self):
        router = Router.__new__(Router)
        first = MockProvider(
            name="first",
            models=[ModelInfo(id="shared-model", name="Shared", provider="first")],
        )
        second = MockProvider(
            name="second",
            models=[ModelInfo(id="shared-model", name="Shared", provider="second")],
        )
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", first._models[0]),
            "first/shared-model": ("first", first._models[0]),
            "second/shared-model": ("second", second._models[0]),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))

        models = await router.list_models()
        assert {(model.provider, model.id) for model in models} == {
            ("first", "shared-model"),
            ("second", "shared-model"),
        }

        for public_id in ("first/shared-model", "second/shared-model"):
            plan = await router.plan_route(public_id, Operation.CHAT)
            assert plan.primary.model.qualified_id == public_id

    @pytest.mark.asyncio
    async def test_duplicate_bare_upstream_id_is_resolved_by_route_priority(self):
        router = Router.__new__(Router)
        first = MockProvider(
            name="first",
            models=[ModelInfo(id="shared-model", name="Shared", provider="first")],
        )
        second = MockProvider(
            name="second",
            models=[ModelInfo(id="shared-model", name="Shared", provider="second")],
        )
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", first._models[0]),
            "first/shared-model": ("first", first._models[0]),
            "second/shared-model": ("second", second._models[0]),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(priorities=["second/shared-model", "first/shared-model"]),
        )

        plan = await router.plan_route("shared-model", Operation.CHAT)

        assert plan.primary.model.qualified_id == "second/shared-model"
        assert [candidate.model.qualified_id for candidate in plan.candidates] == [
            "second/shared-model",
            "first/shared-model",
        ]
        assert plan.explicit is False

    @pytest.mark.asyncio
    async def test_bare_alias_unknown_runtime_unsupported_advances_to_next_provider(self):
        class UnsupportedProvider(MockProvider):
            async def chat_completion(self, request: ChatRequest) -> ChatResponse:
                self._request_count += 1
                raise ProviderError(
                    "chat is unsupported",
                    status_code=400,
                    retryable=False,
                    kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
                )

        first_model = ModelInfo(id="shared-model", name="Shared", provider="first")
        second_model = ModelInfo(id="shared-model", name="Shared", provider="second")
        first = UnsupportedProvider(name="first", models=[first_model])
        second = MockProvider(name="second", models=[second_model])
        router = Router.__new__(Router)
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", first_model),
            "first/shared-model": ("first", first_model),
            "second/shared-model": ("second", second_model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(priorities=["first/shared-model", "second/shared-model"]),
        )

        response, provider_name = await router.chat_completion(
            ChatRequest(
                model="shared-model",
                messages=[Message(role="user", content="Hello")],
            )
        )

        assert provider_name == "second"
        assert response.model == "shared-model"
        assert first._request_count == 1
        assert second._request_count == 1

    @pytest.mark.asyncio
    async def test_unknown_bare_alias_returns_404_without_auto_routing_unrelated_model(self):
        unrelated_model = ModelInfo(
            id="unrelated-model",
            name="Unrelated",
            provider="unrelated",
            operation_capabilities={Operation.CHAT.value: True},
        )
        unrelated = MockProvider(name="unrelated", models=[unrelated_model])
        router = Router.__new__(Router)
        router.providers = {"unrelated": unrelated}
        _init_router_empty(router)
        router._models_cache = {
            "unrelated-model": ("unrelated", unrelated_model),
            "unrelated/unrelated-model": ("unrelated", unrelated_model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(priorities=["unrelated/unrelated-model"]),
        )

        with pytest.raises(ProviderError) as exc_info:
            await router.chat_completion(
                ChatRequest(
                    model="missing-alias-xyz",
                    messages=[Message(role="user", content="Hello")],
                )
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert unrelated._request_count == 0

    @pytest.mark.asyncio
    async def test_static_unsupported_bare_alias_returns_400_without_unrelated_substitution(self):
        first_model = ModelInfo(
            id="shared-model",
            name="Shared",
            provider="first",
            operation_capabilities={Operation.CHAT.value: False},
        )
        second_model = ModelInfo(
            id="shared-model",
            name="Shared",
            provider="second",
            operation_capabilities={Operation.CHAT.value: False},
        )
        unrelated_model = ModelInfo(
            id="unrelated-model",
            name="Unrelated",
            provider="unrelated",
            operation_capabilities={Operation.CHAT.value: True},
        )
        first = MockProvider(name="first", models=[first_model])
        second = MockProvider(name="second", models=[second_model])
        unrelated = MockProvider(name="unrelated", models=[unrelated_model])
        router = Router.__new__(Router)
        router.providers = {"first": first, "second": second, "unrelated": unrelated}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", first_model),
            "first/shared-model": ("first", first_model),
            "second/shared-model": ("second", second_model),
            "unrelated-model": ("unrelated", unrelated_model),
            "unrelated/unrelated-model": ("unrelated", unrelated_model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(
                priorities=[
                    "first/shared-model",
                    "second/shared-model",
                    "unrelated/unrelated-model",
                ]
            ),
        )

        with pytest.raises(ProviderError) as exc_info:
            await router.chat_completion(
                ChatRequest(
                    model="shared-model",
                    messages=[Message(role="user", content="Hello")],
                )
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert first._request_count == 0
        assert second._request_count == 0
        assert unrelated._request_count == 0

    @pytest.mark.asyncio
    async def test_provider_qualified_unknown_runtime_unsupported_does_not_switch(self):
        class UnsupportedProvider(MockProvider):
            async def chat_completion(self, request: ChatRequest) -> ChatResponse:
                self._request_count += 1
                raise ProviderError(
                    "chat is unsupported",
                    status_code=400,
                    retryable=False,
                    kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
                )

        first_model = ModelInfo(id="shared-model", name="Shared", provider="first")
        second_model = ModelInfo(id="shared-model", name="Shared", provider="second")
        first = UnsupportedProvider(name="first", models=[first_model])
        second = MockProvider(name="second", models=[second_model])
        router = Router.__new__(Router)
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", first_model),
            "first/shared-model": ("first", first_model),
            "second/shared-model": ("second", second_model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(priorities=["first/shared-model", "second/shared-model"]),
        )

        with pytest.raises(ProviderError) as exc_info:
            await router.chat_completion(
                ChatRequest(
                    model="first/shared-model",
                    messages=[Message(role="user", content="Hello")],
                )
            )

        assert exc_info.value.kind is ProviderFailureKind.UNSUPPORTED_OPERATION
        assert first._request_count == 1
        assert second._request_count == 0

    @pytest.mark.asyncio
    async def test_provider_prefix_is_case_insensitive_and_uses_registry_name(self):
        router = Router.__new__(Router)
        first_model = ModelInfo(id="shared-model", name="Shared", provider="first")
        second_model = ModelInfo(id="shared-model", name="Shared", provider="second")
        first = MockProvider(name="first", models=[first_model])
        second = MockProvider(name="second", models=[second_model])
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", first_model),
            "first/shared-model": ("first", first_model),
            "second/shared-model": ("second", second_model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))

        plan = await router.plan_route("FIRST/shared-model", Operation.CHAT)

        assert plan.explicit is True
        assert plan.primary.model == ModelRef("first", "shared-model")

    @pytest.mark.asyncio
    async def test_unknown_provider_prefix_returns_404_without_cross_provider_fuzzy(self):
        router = Router.__new__(Router)
        first_model = ModelInfo(id="shared-model", name="Shared", provider="first")
        second_model = ModelInfo(id="shared-model", name="Shared", provider="second")
        first = MockProvider(name="first", models=[first_model])
        second = MockProvider(name="second", models=[second_model])
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", first_model),
            "first/shared-model": ("first", first_model),
            "second/shared-model": ("second", second_model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))

        with pytest.raises(ProviderError) as exc_info:
            await router.chat_completion(
                ChatRequest(
                    model="unknown/shared-model",
                    messages=[Message(role="user", content="Hello")],
                )
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert first._request_count == 0
        assert second._request_count == 0

    @pytest.mark.asyncio
    async def test_duplicate_fuzzy_alias_is_resolved_by_route_priority(self):
        router = Router.__new__(Router)
        first = MockProvider(
            name="first",
            models=[ModelInfo(id="shared.model", name="Shared", provider="first")],
        )
        second = MockProvider(
            name="second",
            models=[ModelInfo(id="shared.model", name="Shared", provider="second")],
        )
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared.model": ("first", first._models[0]),
            "first/shared.model": ("first", first._models[0]),
            "second/shared.model": ("second", second._models[0]),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(priorities=["second/shared.model", "first/shared.model"]),
        )

        plan = await router.plan_route("shared-model", Operation.CHAT)

        assert plan.primary.model.qualified_id == "second/shared.model"

    @pytest.mark.asyncio
    async def test_bare_alias_prefers_supported_provider_over_unsupported_priority(self):
        router = Router.__new__(Router)
        unsupported_model = ModelInfo(
            id="shared-model",
            name="Shared",
            provider="first",
            operation_capabilities={Operation.CHAT.value: False},
        )
        supported_model = ModelInfo(
            id="shared-model",
            name="Shared",
            provider="second",
            operation_capabilities={Operation.CHAT.value: True},
        )
        first = MockProvider(name="first", models=[unsupported_model])
        second = MockProvider(name="second", models=[supported_model])
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", unsupported_model),
            "first/shared-model": ("first", unsupported_model),
            "second/shared-model": ("second", supported_model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(
            router,
            PrioritiesConfig(priorities=["first/shared-model", "second/shared-model"]),
        )

        plan = await router.plan_route("shared-model", Operation.CHAT)

        assert plan.primary.model.qualified_id == "second/shared-model"

    @pytest.mark.asyncio
    async def test_unconfigured_bare_alias_uses_full_catalog_capabilities(self):
        router = Router.__new__(Router)
        unsupported_model = ModelInfo(
            id="shared-model",
            name="Shared",
            provider="first",
            operation_capabilities={Operation.CHAT.value: False},
        )
        supported_model = ModelInfo(
            id="shared-model",
            name="Shared",
            provider="second",
            operation_capabilities={Operation.CHAT.value: True},
        )
        first = MockProvider(name="first", models=[unsupported_model])
        second = MockProvider(name="second", models=[supported_model])
        router.providers = {"first": first, "second": second}
        _init_router_empty(router)
        router._models_cache = {
            "shared-model": ("first", unsupported_model),
            "first/shared-model": ("first", unsupported_model),
            "second/shared-model": ("second", supported_model),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))

        plan = await router.plan_route("shared-model", Operation.CHAT)

        assert plan.primary.model.qualified_id == "second/shared-model"

    @pytest.mark.asyncio
    async def test_exact_dated_bare_id_is_not_replaced_by_newer_family_version(self):
        router = Router.__new__(Router)
        older = ModelInfo(
            id="claude-sonnet-4-20250101",
            name="Claude Sonnet 4",
            provider="anthropic",
        )
        newer = ModelInfo(
            id="claude-sonnet-4-20260101",
            name="Claude Sonnet 4",
            provider="anthropic",
        )
        provider = MockProvider(name="anthropic", models=[older, newer])
        router.providers = {"anthropic": provider}
        _init_router_empty(router)
        router._models_cache = {
            older.id: ("anthropic", older),
            newer.id: ("anthropic", newer),
            f"anthropic/{older.id}": ("anthropic", older),
            f"anthropic/{newer.id}": ("anthropic", newer),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))

        plan = await router.plan_route(older.id, Operation.CHAT)

        assert plan.primary.model.qualified_id == f"anthropic/{older.id}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query",
        [
            "CLAUDE-SONNET-4-20250101",
            "claude.sonnet.4.20250101",
        ],
    )
    async def test_dated_identity_alias_plan_keeps_specific_version(self, query):
        router = Router.__new__(Router)
        older = ModelInfo(
            id="claude-sonnet-4-20250101",
            name="Claude Sonnet 4",
            provider="anthropic",
        )
        newer = ModelInfo(
            id="claude-sonnet-4-20260101",
            name="Claude Sonnet 4",
            provider="anthropic",
        )
        provider = MockProvider(name="anthropic", models=[older, newer])
        router.providers = {"anthropic": provider}
        _init_router_empty(router)
        router._models_cache = {
            older.id: ("anthropic", older),
            newer.id: ("anthropic", newer),
            f"anthropic/{older.id}": ("anthropic", older),
            f"anthropic/{newer.id}": ("anthropic", newer),
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))

        plan = await router.plan_route(query, Operation.CHAT)

        assert plan.primary.model.qualified_id == f"anthropic/{older.id}"

    @pytest.mark.asyncio
    async def test_fuzzy_cross_family_ambiguity_is_a_client_error(self):
        router = Router.__new__(Router)
        provider = MockProvider(
            name="anthropic",
            models=[
                ModelInfo(id="claude-opus-4", name="Claude Opus 4", provider="anthropic"),
                ModelInfo(
                    id="claude-sonnet-4",
                    name="Claude Sonnet 4",
                    provider="anthropic",
                ),
            ],
        )
        router.providers = {"anthropic": provider}
        _init_router_empty(router)
        router._models_cache = {model.id: ("anthropic", model) for model in provider._models} | {
            f"anthropic/{model.id}": ("anthropic", model) for model in provider._models
        }
        _mark_models_cached(router)
        _mark_providers_fresh(router)
        _set_priorities(router, PrioritiesConfig(priorities=[]))

        with pytest.raises(ProviderError) as exc_info:
            await router.plan_route("claude-4", Operation.CHAT)

        assert exc_info.value.status_code == 400
        assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert "ambiguous" in str(exc_info.value).lower()


class TestRouterCreateRequestWithModel:
    """Tests for creating requests with different models."""

    @pytest.fixture
    def router(self):
        """Create a minimal router."""
        return Router.__new__(Router)

    def test_create_responses_request_with_model(self, router):
        """Test creating a ResponsesRequest with different model."""
        original = ResponsesRequest(
            model="original-model",
            input="Hello",
            temperature=0.5,
            max_output_tokens=1000,
            instructions="Be helpful",
        )
        new_request = router._create_responses_request_with_model(original, "new-model")

        assert new_request.model == "new-model"
        assert new_request.input == "Hello"
        assert new_request.temperature == 0.5
        assert new_request.max_output_tokens == 1000
        assert new_request.instructions == "Be helpful"

    def test_create_responses_request_preserves_reasoning_effort(self, router):
        """reasoning_effort must survive provider/fallback rebuild.

        Regression: codex sent ``reasoning: {effort: xhigh}`` on /responses,
        the route set it on the internal request, but the rebuild dropped it
        before the Copilot provider saw it — so ``_build_responses_payload``
        omitted the ``reasoning`` block entirely and Copilot ran at default
        (medium) instead of xhigh.
        """
        original = ResponsesRequest(
            model="github-copilot/gpt-5.5",
            input="Hello",
            reasoning_effort="xhigh",
        )
        new_request = router._create_responses_request_with_model(original, "gpt-5.5")

        assert new_request.reasoning_effort == "xhigh"

    def test_create_chat_request_with_tools(self, router):
        """Test creating a ChatRequest preserving tools."""
        original = ChatRequest(
            model="original-model",
            messages=[Message(role="user", content="Hello")],
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
        )
        new_request = router._create_request_with_model(original, "new-model")

        assert new_request.model == "new-model"
        assert new_request.tools == original.tools
        assert new_request.tool_choice == "auto"
