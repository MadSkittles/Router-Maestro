"""Advanced tests for the Router module with async operations."""

import pytest

from router_maestro.config import FallbackStrategy, PrioritiesConfig
from router_maestro.providers import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    ProviderError,
    ResponsesRequest,
    ResponsesResponse,
)
from router_maestro.providers.base import BaseProvider
from router_maestro.routing.router import (
    AUTO_ROUTE_MODEL,
    CACHE_TTL_SECONDS,
    Router,
    get_router,
    reset_router,
)


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
        router._models_cache = {}
        router._cache_initialized = False
        router._cache_timestamp = 0.0
        router._priorities_config = None
        router._priorities_config_timestamp = 0.0
        router._providers_config_timestamp = 0.0
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
        router_with_mock._providers_config_timestamp = float("inf")

        await router_with_mock._ensure_models_cache()

        assert router_with_mock._cache_initialized is True
        assert "model-1" in router_with_mock._models_cache
        assert "model-2" in router_with_mock._models_cache
        assert "test-provider/model-1" in router_with_mock._models_cache
        assert "test-provider/model-2" in router_with_mock._models_cache

    @pytest.mark.asyncio
    async def test_cache_skips_unauthenticated_providers(self, router_with_mock):
        """Test that cache skips unauthenticated providers."""
        unauth_provider = MockProvider(name="unauth", authenticated=False)
        router_with_mock.providers = {"unauth": unauth_provider}
        router_with_mock._providers_config_timestamp = float("inf")

        await router_with_mock._ensure_models_cache()

        assert router_with_mock._cache_initialized is True
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
        router._models_cache = {}
        router._cache_initialized = False
        router._cache_timestamp = 0.0
        router._priorities_config = None
        router._priorities_config_timestamp = 0.0
        router._providers_config_timestamp = float("inf")
        router._fuzzy_cache = {}
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
    async def test_resolve_auto_route_model(self, router_with_providers):
        """Test resolving auto-route model."""
        # Set up priorities config
        router_with_providers._priorities_config = PrioritiesConfig(
            priorities=["primary/gpt-4o", "secondary/claude-3"],
        )
        router_with_providers._priorities_config_timestamp = float("inf")

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
        router._models_cache = {
            "test-model": ("test", ModelInfo(id="test-model", name="Test", provider="test")),
            "test/test-model": ("test", ModelInfo(id="test-model", name="Test", provider="test")),
        }
        router._cache_initialized = True
        router._cache_timestamp = float("inf")
        router._priorities_config = PrioritiesConfig(priorities=[])
        router._priorities_config_timestamp = float("inf")
        router._providers_config_timestamp = float("inf")
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
        router._models_cache = {
            "test-model": ("test", ModelInfo(id="test-model", name="Test", provider="test")),
            "test/test-model": ("test", ModelInfo(id="test-model", name="Test", provider="test")),
        }
        router._cache_initialized = True
        router._cache_timestamp = float("inf")
        router._priorities_config = PrioritiesConfig(priorities=[])
        router._priorities_config_timestamp = float("inf")
        router._providers_config_timestamp = float("inf")
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
        router._cache_initialized = True
        router._cache_timestamp = float("inf")
        router._priorities_config = PrioritiesConfig(
            priorities=["primary/model-1", "secondary/model-1"],
            fallback={"strategy": "priority", "maxRetries": 3},
        )
        router._priorities_config_timestamp = float("inf")
        router._providers_config_timestamp = float("inf")
        return router, primary, secondary

    def test_get_fallback_candidates_priority(self, router_with_fallback):
        """Test getting fallback candidates with priority strategy."""
        router, _, _ = router_with_fallback
        candidates = router._get_fallback_candidates(
            "primary", "model-1", FallbackStrategy.PRIORITY
        )

        assert len(candidates) == 1
        assert candidates[0][0] == "secondary"
        assert candidates[0][1] == "model-1"

    def test_get_fallback_candidates_same_model(self, router_with_fallback):
        """Test getting fallback candidates with same-model strategy."""
        router, _, _ = router_with_fallback
        candidates = router._get_fallback_candidates(
            "primary", "model-1", FallbackStrategy.SAME_MODEL
        )

        assert len(candidates) == 1
        assert candidates[0][0] == "secondary"

    def test_get_fallback_candidates_none(self, router_with_fallback):
        """Test getting fallback candidates with none strategy."""
        router, _, _ = router_with_fallback
        candidates = router._get_fallback_candidates("primary", "model-1", FallbackStrategy.NONE)

        assert len(candidates) == 0


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
        router._cache_initialized = True
        router._cache_timestamp = float("inf")
        router._priorities_config = PrioritiesConfig(
            priorities=["provider1/model-b", "provider1/model-a"]
        )
        router._priorities_config_timestamp = float("inf")
        router._providers_config_timestamp = float("inf")
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
