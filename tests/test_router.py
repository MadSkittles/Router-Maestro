"""Tests for the Router module."""

import pytest

from router_maestro.providers import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
)
from router_maestro.providers.base import BaseProvider
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.utils.cache import TTLCache


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        authenticated: bool = True,
        models: list[ModelInfo] | None = None,
        fail_on_request: bool = False,
    ):
        self._name = name
        self._authenticated = authenticated
        self._models = models or [ModelInfo(id="test-model", name="Test Model", provider=name)]
        self._fail_on_request = fail_on_request

    @property
    def name(self) -> str:
        return self._name

    def is_authenticated(self) -> bool:
        return self._authenticated

    async def ensure_token(self) -> None:
        pass

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        if self._fail_on_request:
            raise ProviderError("Mock provider failure", retryable=True)
        return ChatResponse(
            content=f"Response from {self._name}",
            model=request.model,
            finish_reason="stop",
        )

    async def chat_completion_stream(self, request: ChatRequest):
        if self._fail_on_request:
            raise ProviderError("Mock provider failure", retryable=True)
        yield ChatResponse(
            content=f"Streaming from {self._name}",
            model=request.model,
            finish_reason="stop",
        )

    async def list_models(self) -> list[ModelInfo]:
        return self._models


def _init_router_caches(router: Router) -> None:
    """Initialize TTLCache attributes on a Router created via __new__."""
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)


class TestRouterModelResolution:
    """Tests for Router model resolution logic."""

    @pytest.fixture
    def router_with_mock(self):
        """Create a router with mock providers."""
        router = Router.__new__(Router)
        router.providers = {}
        _init_router_caches(router)
        return router

    def test_parse_model_key_with_provider(self, router_with_mock):
        """Test parsing model key with provider prefix."""
        provider, model = router_with_mock._parse_model_key("github-copilot/gpt-4o")
        assert provider == "github-copilot"
        assert model == "gpt-4o"

    def test_parse_model_key_without_provider(self, router_with_mock):
        """Test parsing model key without provider prefix."""
        provider, model = router_with_mock._parse_model_key("gpt-4o")
        assert provider == ""
        assert model == "gpt-4o"

    def test_parse_model_key_with_multiple_slashes(self, router_with_mock):
        """Test parsing model key with multiple slashes."""
        provider, model = router_with_mock._parse_model_key("custom/org/model-name")
        assert provider == "custom"
        assert model == "org/model-name"


class TestRouterChatRequest:
    """Tests for Router._create_request_with_model."""

    @pytest.fixture
    def router(self):
        """Create a minimal router instance."""
        router = Router.__new__(Router)
        return router

    def test_create_request_with_model(self, router):
        """Test creating a request with a different model."""
        original = ChatRequest(
            model="original-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        new_request = router._create_request_with_model(original, "new-model")

        assert new_request.model == "new-model"
        assert new_request.messages == original.messages
        assert new_request.temperature == 0.7
        assert new_request.max_tokens == 100
        assert new_request.stream is False


class TestRouterCacheInvalidation:
    """Tests for Router cache invalidation."""

    @pytest.fixture
    def router(self):
        """Create a minimal router for testing cache."""
        router = Router.__new__(Router)
        _init_router_caches(router)
        router._models_cache = {"test": ("provider", None)}
        router._models_cache_ttl.set(True)
        router._priorities_cache.set(object())
        return router

    def test_invalidate_cache_clears_models(self, router):
        """Test that invalidate_cache clears models cache."""
        router.invalidate_cache()

        assert router._models_cache == {}
        assert not router._models_cache_ttl.is_valid

    def test_invalidate_cache_clears_priorities(self, router):
        """Test that invalidate_cache clears priorities config cache."""
        router.invalidate_cache()

        assert router._priorities_cache.get() is None
        assert not router._priorities_cache.is_valid


class TestRouterCatalogRefreshResilience:
    """Cache refresh must not drop a provider on a transient auth failure."""

    def _make_router(self, provider: BaseProvider) -> Router:
        router = Router.__new__(Router)
        router.providers = {provider.name: provider}
        _init_router_caches(router)
        router._managed_generation = True
        return router

    @pytest.mark.asyncio
    async def test_transient_failure_preserves_stale_entries(self):
        """A refresh where ensure_token fails keeps previously cached models."""

        class FlakyProvider(MockProvider):
            def __init__(self):
                super().__init__(
                    name="github-copilot",
                    models=[ModelInfo(id="gpt-5.6-sol", name="GPT", provider="github-copilot")],
                )
                self.fail_next_token = False

            async def ensure_token(self) -> None:
                if self.fail_next_token:
                    raise ProviderError(
                        "Not authenticated with GitHub Copilot",
                        status_code=401,
                    )

        provider = FlakyProvider()
        router = self._make_router(provider)

        # First refresh succeeds and populates the catalog.
        await router._ensure_models_cache()
        assert "gpt-5.6-sol" in router._models_cache
        assert router._models_cache_ttl.is_valid

        # Force the next refresh and make the token check fail transiently.
        router._models_cache_ttl.clear()
        provider.fail_next_token = True
        await router._ensure_models_cache()

        # Stale entries survive so bare-name resolution keeps working, and the
        # TTL stays expired so the next request retries instead of waiting.
        assert "gpt-5.6-sol" in router._models_cache
        assert not router._models_cache_ttl.is_valid

    @pytest.mark.asyncio
    async def test_cold_start_failure_propagates_and_remains_retryable(self):
        """An empty catalog after an upstream failure must not become a model 404."""

        class FailingProvider(MockProvider):
            def __init__(self):
                super().__init__(name="github-copilot")
                self.calls = 0

            async def ensure_token(self) -> None:
                self.calls += 1
                raise ProviderError(
                    "Failed to refresh Copilot token",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_STATUS,
                )

        provider = FailingProvider()
        router = self._make_router(provider)

        with pytest.raises(ProviderError, match="Failed to refresh Copilot token"):
            await router._ensure_models_cache()
        with pytest.raises(ProviderError, match="Failed to refresh Copilot token"):
            await router._ensure_models_cache()

        assert provider.calls == 2
        assert router._models_cache == {}
        assert not router._models_cache_ttl.is_valid
