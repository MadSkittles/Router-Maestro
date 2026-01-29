"""Model router with priority-based selection and fallback."""

import time
from collections.abc import AsyncIterator

from router_maestro.auth import ApiKeyCredential, AuthManager
from router_maestro.config import (
    FallbackStrategy,
    PrioritiesConfig,
    load_priorities_config,
    load_providers_config,
)
from router_maestro.providers import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    CopilotProvider,
    ModelInfo,
    OpenAICompatibleProvider,
    ProviderError,
)
from router_maestro.utils import get_logger

logger = get_logger("routing")

# Special model name that triggers auto-routing
AUTO_ROUTE_MODEL = "router-maestro"

# Cache TTL in seconds (5 minutes)
CACHE_TTL_SECONDS = 300

# Global singleton instance
_router_instance: "Router | None" = None


def get_router() -> "Router":
    """Get the singleton Router instance.

    Returns:
        The global Router instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = Router()
        logger.info("Created singleton Router instance")
    return _router_instance


def reset_router() -> None:
    """Reset the singleton Router instance.

    Call this when authentication changes or to force reload.
    """
    global _router_instance
    if _router_instance is not None:
        _router_instance.invalidate_cache()
        _router_instance = None
        logger.info("Reset singleton Router instance")


class Router:
    """Router for model requests with priority and fallback support."""

    def __init__(self) -> None:
        self.providers: dict[str, BaseProvider] = {}
        # Model cache: maps model_id -> (provider_name, ModelInfo)
        self._models_cache: dict[str, tuple[str, ModelInfo]] = {}
        self._cache_initialized: bool = False
        self._cache_timestamp: float = 0.0
        # Priorities config cache
        self._priorities_config: PrioritiesConfig | None = None
        self._priorities_config_timestamp: float = 0.0
        # Providers config cache
        self._providers_config_timestamp: float = 0.0
        self._load_providers()

    def _load_providers(self) -> None:
        """Load providers from configuration."""
        custom_providers_config = load_providers_config()
        auth_manager = AuthManager()

        # Clear existing providers except keep copilot if already exists
        old_copilot = self.providers.get("github-copilot")
        self.providers.clear()

        # Always add built-in GitHub Copilot provider (reuse existing instance if available)
        if old_copilot is not None:
            self.providers["github-copilot"] = old_copilot
        else:
            copilot = CopilotProvider()
            self.providers["github-copilot"] = copilot
        logger.debug("Loaded built-in provider: github-copilot")

        # Load custom providers from providers.json
        for provider_name, provider_config in custom_providers_config.providers.items():
            # Get API key from auth storage
            cred = auth_manager.get_credential(provider_name)
            if isinstance(cred, ApiKeyCredential):
                provider = OpenAICompatibleProvider(
                    name=provider_name,
                    base_url=provider_config.baseURL,
                    api_key=cred.key,
                    models={
                        model_id: model_config.name
                        for model_id, model_config in provider_config.models.items()
                    },
                )
                self.providers[provider_name] = provider
                logger.debug("Loaded custom provider: %s", provider_name)

        self._providers_config_timestamp = time.time()
        logger.info("Loaded %d providers", len(self.providers))

    def _get_priorities_config(self) -> PrioritiesConfig:
        """Get priorities config with caching."""
        # Simple time-based cache (same TTL as models cache)
        current_time = time.time()
        if (
            self._priorities_config is not None
            and current_time - self._priorities_config_timestamp < CACHE_TTL_SECONDS
        ):
            return self._priorities_config

        self._priorities_config = load_priorities_config()
        self._priorities_config_timestamp = current_time
        return self._priorities_config

    def _ensure_providers_fresh(self) -> None:
        """Ensure providers config is fresh, reload if expired."""
        current_time = time.time()
        if current_time - self._providers_config_timestamp >= CACHE_TTL_SECONDS:
            logger.debug("Providers config expired, reloading")
            self._load_providers()
            # Also invalidate models cache since providers may have changed
            self._models_cache.clear()
            self._cache_initialized = False

    def _parse_model_key(self, model_key: str) -> tuple[str, str]:
        """Parse a model key into provider and model.

        Args:
            model_key: Model key in format 'provider/model'

        Returns:
            Tuple of (provider_name, model_id)
        """
        if "/" in model_key:
            parts = model_key.split("/", 1)
            return parts[0], parts[1]
        return "", model_key

    async def _ensure_models_cache(self) -> None:
        """Ensure the models cache is populated and not expired."""
        # First ensure providers config is fresh
        self._ensure_providers_fresh()

        # Check if cache is still valid (initialized and not expired)
        if self._cache_initialized:
            age = time.time() - self._cache_timestamp
            if age < CACHE_TTL_SECONDS:
                return
            logger.debug("Cache expired (age=%.1fs), refreshing", age)
            self._models_cache.clear()

        logger.debug("Initializing models cache")
        for provider_name, provider in self.providers.items():
            if provider.is_authenticated():
                try:
                    await provider.ensure_token()
                    models = await provider.list_models()
                    for model in models:
                        # Store by model_id only (without provider prefix)
                        # If same model_id exists in multiple providers, first one wins
                        if model.id not in self._models_cache:
                            self._models_cache[model.id] = (provider_name, model)
                        # Also store with provider prefix for explicit lookups
                        full_key = f"{provider_name}/{model.id}"
                        self._models_cache[full_key] = (provider_name, model)
                    logger.debug("Cached %d models from %s", len(models), provider_name)
                except ProviderError as e:
                    logger.warning("Failed to load models from %s: %s", provider_name, e)
                    continue

        self._cache_initialized = True
        self._cache_timestamp = time.time()
        logger.info("Models cache initialized with %d entries", len(self._models_cache))

    async def _resolve_provider(self, model_id: str) -> tuple[str, str, BaseProvider]:
        """Resolve model_id to provider.

        Args:
            model_id: Model ID (can be 'router-maestro', 'provider/model', or just 'model')

        Returns:
            Tuple of (provider_name, actual_model_id, provider)

        Raises:
            ProviderError: If model not found or no models available
        """
        # Check for auto-routing
        if model_id == AUTO_ROUTE_MODEL:
            result = await self._get_auto_route_model()
            if not result:
                logger.error("No models available for auto-routing")
                raise ProviderError("No models available for auto-routing", status_code=503)
            return result

        # Explicit model specified - find in cache
        result = await self._find_model_in_cache(model_id)
        if not result:
            logger.warning("Model not found: %s", model_id)
            raise ProviderError(
                f"Model '{model_id}' not found in any provider",
                status_code=404,
            )
        return result

    def _create_request_with_model(
        self, original_request: ChatRequest, model_id: str
    ) -> ChatRequest:
        """Create a new ChatRequest with a different model ID.

        Args:
            original_request: The original request
            model_id: The new model ID to use

        Returns:
            New ChatRequest with updated model
        """
        return ChatRequest(
            model=model_id,
            messages=original_request.messages,
            temperature=original_request.temperature,
            max_tokens=original_request.max_tokens,
            stream=original_request.stream,
            tools=original_request.tools,
            tool_choice=original_request.tool_choice,
        )

    async def _get_auto_route_model(self) -> tuple[str, str, BaseProvider] | None:
        """Get the highest priority available model for auto-routing.

        Returns:
            Tuple of (provider_name, model_id, provider) or None if no model available
        """
        await self._ensure_models_cache()
        priorities_config = self._get_priorities_config()

        # Try each priority in order
        for priority_key in priorities_config.priorities:
            provider_name, model_id = self._parse_model_key(priority_key)
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if provider.is_authenticated():
                    # Verify model exists in cache
                    if priority_key in self._models_cache:
                        logger.debug("Auto-route selected: %s", priority_key)
                        return provider_name, model_id, provider

        # Fallback: return first available model from any provider
        for model_id, (provider_name, _) in self._models_cache.items():
            if "/" not in model_id:  # Skip full keys, only use simple model_ids
                provider = self.providers.get(provider_name)
                if provider and provider.is_authenticated():
                    logger.debug("Auto-route fallback: %s/%s", provider_name, model_id)
                    return provider_name, model_id, provider

        return None

    async def _find_model_in_cache(self, model_id: str) -> tuple[str, str, BaseProvider] | None:
        """Find a model in the cache.

        Args:
            model_id: Model ID (can be 'provider/model' or just 'model')

        Returns:
            Tuple of (provider_name, actual_model_id, provider) or None
        """
        await self._ensure_models_cache()

        # If model_id includes provider prefix (e.g., "github-copilot/gpt-4o")
        if "/" in model_id:
            provider_name, actual_model_id = self._parse_model_key(model_id)
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if provider.is_authenticated():
                    # Check if the model exists for this provider
                    if model_id in self._models_cache:
                        return provider_name, actual_model_id, provider
            return None

        # Simple model_id (e.g., "gpt-4o") - look up in cache
        if model_id in self._models_cache:
            provider_name, _ = self._models_cache[model_id]
            provider = self.providers.get(provider_name)
            if provider and provider.is_authenticated():
                return provider_name, model_id, provider

        return None

    def _get_fallback_candidates(
        self,
        current_provider: str,
        current_model: str,
        strategy: FallbackStrategy,
    ) -> list[tuple[str, str, BaseProvider]]:
        """Get ordered list of fallback candidates based on strategy.

        Args:
            current_provider: The provider that just failed
            current_model: The model that was requested
            strategy: The fallback strategy to use

        Returns:
            List of (provider_name, model_id, provider) tuples to try
        """
        if strategy == FallbackStrategy.NONE:
            return []

        candidates: list[tuple[str, str, BaseProvider]] = []
        current_key = f"{current_provider}/{current_model}"

        if strategy == FallbackStrategy.PRIORITY:
            # Follow the priorities list order, starting after current
            priorities_config = self._get_priorities_config()
            found_current = False

            for priority_key in priorities_config.priorities:
                if priority_key == current_key:
                    found_current = True
                    continue

                if found_current:
                    provider_name, model_id = self._parse_model_key(priority_key)
                    if provider_name in self.providers:
                        provider = self.providers[provider_name]
                        if provider.is_authenticated():
                            if priority_key in self._models_cache:
                                candidates.append((provider_name, model_id, provider))

        elif strategy == FallbackStrategy.SAME_MODEL:
            # Only try other providers that have the same model
            for other_name, other_provider in self.providers.items():
                if other_name == current_provider:
                    continue
                if not other_provider.is_authenticated():
                    continue
                other_key = f"{other_name}/{current_model}"
                if other_key in self._models_cache:
                    candidates.append((other_name, current_model, other_provider))

        return candidates

    async def _execute_with_fallback(
        self,
        request: ChatRequest,
        provider_name: str,
        actual_model_id: str,
        provider: BaseProvider,
        fallback: bool,
        is_stream: bool,
    ) -> tuple[ChatResponse | AsyncIterator[ChatStreamChunk], str]:
        """Execute request with fallback support.

        Args:
            request: Original chat request
            provider_name: Name of the primary provider
            actual_model_id: The actual model ID to use
            provider: The primary provider instance
            fallback: Whether to try fallback providers on error
            is_stream: Whether this is a streaming request

        Returns:
            Tuple of (response or stream, provider_name)

        Raises:
            ProviderError: If all providers fail
        """
        actual_request = self._create_request_with_model(request, actual_model_id)

        try:
            await provider.ensure_token()
            if is_stream:
                stream = await provider.chat_completion_stream(actual_request)
                logger.info("Stream request routed to %s", provider_name)
                return stream, provider_name
            else:
                response = await provider.chat_completion(actual_request)
                logger.info("Request completed via %s", provider_name)
                return response, provider_name
        except ProviderError as e:
            logger.warning("Provider %s failed: %s", provider_name, e)
            if not fallback or not e.retryable:
                raise

            # Load fallback config
            priorities_config = self._get_priorities_config()
            fallback_config = priorities_config.fallback

            if fallback_config.strategy == FallbackStrategy.NONE:
                raise

            # Get fallback candidates
            candidates = self._get_fallback_candidates(
                provider_name, actual_model_id, fallback_config.strategy
            )

            # Try fallback candidates up to maxRetries
            for i, (other_name, other_model_id, other_provider) in enumerate(candidates):
                if i >= fallback_config.maxRetries:
                    break

                logger.info("Trying fallback: %s/%s", other_name, other_model_id)
                fallback_request = self._create_request_with_model(request, other_model_id)

                try:
                    await other_provider.ensure_token()
                    if is_stream:
                        stream = await other_provider.chat_completion_stream(fallback_request)
                        logger.info("Stream fallback succeeded via %s", other_name)
                        return stream, other_name
                    else:
                        response = await other_provider.chat_completion(fallback_request)
                        logger.info("Fallback succeeded via %s", other_name)
                        return response, other_name
                except ProviderError as fallback_error:
                    logger.warning("Fallback %s failed: %s", other_name, fallback_error)
                    continue
            raise

    async def chat_completion(
        self,
        request: ChatRequest,
        fallback: bool = True,
    ) -> tuple[ChatResponse, str]:
        """Route a chat completion request.

        Args:
            request: Chat completion request
            fallback: Whether to try fallback providers on error

        Returns:
            Tuple of (response, provider_name)

        Raises:
            ProviderError: If model not found or all providers fail
        """
        provider_name, actual_model_id, provider = await self._resolve_provider(request.model)
        logger.info("Routing request to %s/%s", provider_name, actual_model_id)

        result, used_provider = await self._execute_with_fallback(
            request, provider_name, actual_model_id, provider, fallback, is_stream=False
        )
        return result, used_provider  # type: ignore

    async def chat_completion_stream(
        self,
        request: ChatRequest,
        fallback: bool = True,
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        """Route a streaming chat completion request.

        Args:
            request: Chat completion request
            fallback: Whether to try fallback providers on error

        Returns:
            Tuple of (stream iterator, provider_name)

        Raises:
            ProviderError: If model not found or all providers fail
        """
        provider_name, actual_model_id, provider = await self._resolve_provider(request.model)
        logger.info("Routing stream request to %s/%s", provider_name, actual_model_id)

        result, used_provider = await self._execute_with_fallback(
            request, provider_name, actual_model_id, provider, fallback, is_stream=True
        )
        return result, used_provider  # type: ignore

    async def list_models(self) -> list[ModelInfo]:
        """List all available models from all authenticated providers.

        Models are sorted by priority configuration.

        Returns:
            List of available models
        """
        await self._ensure_models_cache()
        priorities_config = self._get_priorities_config()

        models: list[ModelInfo] = []
        seen: set[str] = set()

        # Collect all models with their full keys
        all_models: dict[str, ModelInfo] = {}
        for key, (_, model_info) in self._models_cache.items():
            # Only include full keys (provider/model)
            if "/" in key:
                all_models[key] = model_info

        # Add prioritized models first
        for priority_key in priorities_config.priorities:
            if priority_key in all_models and priority_key not in seen:
                models.append(all_models[priority_key])
                seen.add(priority_key)

        # Add remaining models
        for key, model in all_models.items():
            if key not in seen:
                models.append(model)
                seen.add(key)

        logger.debug("Listed %d models", len(models))
        return models

    def invalidate_cache(self) -> None:
        """Invalidate all caches to force refresh."""
        self._models_cache.clear()
        self._cache_initialized = False
        self._cache_timestamp = 0.0
        self._priorities_config = None
        self._priorities_config_timestamp = 0.0
        self._providers_config_timestamp = 0.0
        logger.debug("All caches invalidated")
