"""Tests for the Router module."""

import pytest

from router_maestro.providers import ChatRequest, ChatResponse, Message, ModelInfo, ProviderError
from router_maestro.providers.base import BaseProvider
from router_maestro.routing.router import Router


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
        self._models = models or [
            ModelInfo(id="test-model", name="Test Model", provider=name)
        ]
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


class TestRouterModelResolution:
    """Tests for Router model resolution logic."""

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
        router._models_cache = {"test": ("provider", None)}
        router._cache_initialized = True
        router._cache_timestamp = 100.0
        router._priorities_config = object()  # Mock config
        router._priorities_config_timestamp = 100.0
        return router

    def test_invalidate_cache_clears_models(self, router):
        """Test that invalidate_cache clears models cache."""
        router.invalidate_cache()

        assert router._models_cache == {}
        assert router._cache_initialized is False
        assert router._cache_timestamp == 0.0

    def test_invalidate_cache_clears_priorities(self, router):
        """Test that invalidate_cache clears priorities config cache."""
        router.invalidate_cache()

        assert router._priorities_config is None
        assert router._priorities_config_timestamp == 0.0
