"""Tests for the Router module."""

import pytest

from router_maestro.providers import ChatRequest, ChatResponse, Message, ModelInfo, ProviderError
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


class TestFindReasoningVariantModel:
    """Provider-scoping behavior of Router.find_reasoning_variant_model.

    Regression coverage for the HIGH bug where a bare-key variant lookup could
    cross providers — e.g. ``anthropic/claude-opus-4.7`` requests being silently
    rewritten to a ``github-copilot/claude-opus-4.7-high`` cache entry.
    """

    @pytest.fixture
    def router(self):
        """Router with two providers serving overlapping models."""
        copilot = MockProvider(
            name="github-copilot",
            models=[
                ModelInfo(id="claude-opus-4.7", name="Opus 4.7", provider="github-copilot"),
                ModelInfo(
                    id="claude-opus-4.7-high", name="Opus 4.7 High", provider="github-copilot"
                ),
                ModelInfo(
                    id="claude-opus-4.7-xhigh", name="Opus 4.7 xHigh", provider="github-copilot"
                ),
            ],
        )
        anthropic = MockProvider(
            name="anthropic",
            models=[
                ModelInfo(id="claude-opus-4.7", name="Opus 4.7", provider="anthropic"),
            ],
        )
        r = Router.__new__(Router)
        r.providers = {"github-copilot": copilot, "anthropic": anthropic}
        _init_router_caches(r)
        # Match the real cache shape: bare key (first registered wins) +
        # provider-prefixed keys for both providers.
        r._models_cache = {
            "claude-opus-4.7": ("github-copilot", copilot._models[0]),
            "github-copilot/claude-opus-4.7": ("github-copilot", copilot._models[0]),
            "github-copilot/claude-opus-4.7-high": ("github-copilot", copilot._models[1]),
            "github-copilot/claude-opus-4.7-xhigh": ("github-copilot", copilot._models[2]),
            "anthropic/claude-opus-4.7": ("anthropic", anthropic._models[0]),
        }
        r._models_cache_ttl.set(True)
        # Without this, _ensure_providers_fresh() reloads providers from disk
        # and clears the mock models cache (CI Python 3.11/3.12 hit this).
        r._providers_ttl.set(True)
        return r

    @pytest.mark.asyncio
    async def test_bare_request_finds_high_variant(self, router):
        """Bare request with high effort routes to the -high variant."""
        result = await router.find_reasoning_variant_model("claude-opus-4.7", "high")
        # Provider gets resolved internally, so the result is provider-qualified.
        assert result == "github-copilot/claude-opus-4.7-high"

    @pytest.mark.asyncio
    async def test_anthropic_prefix_does_not_cross_to_copilot(self, router):
        """An anthropic-scoped request must not borrow Copilot's variant."""
        result = await router.find_reasoning_variant_model("anthropic/claude-opus-4.7", "high")
        # Anthropic doesn't have a -high variant; must return None rather than
        # silently jumping providers.
        assert result is None

    @pytest.mark.asyncio
    async def test_copilot_prefix_finds_xhigh(self, router):
        result = await router.find_reasoning_variant_model(
            "github-copilot/claude-opus-4.7", "xhigh"
        )
        assert result == "github-copilot/claude-opus-4.7-xhigh"

    @pytest.mark.asyncio
    async def test_non_high_effort_returns_none(self, router):
        """low/medium are handled by reasoning_effort passthrough, not variants."""
        assert await router.find_reasoning_variant_model("claude-opus-4.7", "medium") is None
        assert await router.find_reasoning_variant_model("claude-opus-4.7", "low") is None
        assert await router.find_reasoning_variant_model("claude-opus-4.7", None) is None

    @pytest.mark.asyncio
    async def test_already_effort_encoded_returns_none(self, router):
        """Don't re-route a request that already targets a variant."""
        result = await router.find_reasoning_variant_model(
            "github-copilot/claude-opus-4.7-high", "high"
        )
        assert result is None
