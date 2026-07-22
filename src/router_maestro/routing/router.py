"""Model router with priority-based selection and fallback."""

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TypeVar, cast

from router_maestro.config import (
    FallbackStrategy,
    PrioritiesConfig,
    load_priorities_config,
    load_providers_config,
)
from router_maestro.providers import (
    AnthropicProvider,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    CopilotProvider,
    ModelInfo,
    OpenAIProvider,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
)
from router_maestro.routing.attempts import (
    AttemptLedger,
    AttemptRecord,
    failure_allows_fallback,
)
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    Feature,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import (
    NoCompatibleRouteError,
    PreparedChatCompletion,
    PreparedChatStream,
    PreparedResponsesStream,
    RouteCandidate,
    RoutePlan,
)
from router_maestro.utils import get_logger
from router_maestro.utils.async_iterators import close_async_iterator
from router_maestro.utils.cache import TTLCache
from router_maestro.utils.model_match import (
    AmbiguousModelMatchError,
    fuzzy_match_model,
)
from router_maestro.utils.model_sort import sort_models

logger = get_logger("routing")

# Special model name that triggers auto-routing
AUTO_ROUTE_MODEL = "router-maestro"

# Cache TTL in seconds (5 minutes)
CACHE_TTL_SECONDS = 300

# Global singleton instance
_router_instance: Router | None = None

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
ChunkT = TypeVar("ChunkT")


@dataclass(slots=True)
class _RouterGeneration[RouterT]:
    generation_id: int
    router: RouterT
    config_snapshot: object | None = None
    references: int = 0
    retired: bool = False
    closing: bool = False
    closed: bool = False
    closed_event: asyncio.Event | None = None

    def event(self) -> asyncio.Event:
        if self.closed_event is None:
            self.closed_event = asyncio.Event()
        return self.closed_event


async def _close_router_resources(router: object) -> None:
    close = getattr(router, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result
        return

    providers = getattr(router, "providers", {})
    seen: set[int] = set()
    for provider in providers.values():
        if id(provider) in seen:
            continue
        seen.add(id(provider))
        provider_close = getattr(provider, "close", None)
        if not callable(provider_close):
            continue
        result = provider_close()
        if inspect.isawaitable(result):
            await result


async def _await_task_ignoring_cancellation[RouterT](
    task: asyncio.Task[RouterT],
) -> tuple[RouterT, bool]:
    """Wait through repeated caller cancellation and report whether it occurred."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


async def _close_uninstalled_router(router: object) -> None:
    close_task = asyncio.create_task(_close_router_resources(router))
    await _await_task_ignoring_cancellation(close_task)


class RouterLease[RouterT]:
    """Idempotent reference to one immutable Router generation."""

    def __init__(
        self,
        owner: RouterOwner[RouterT],
        generation: _RouterGeneration[RouterT],
    ) -> None:
        self._owner = owner
        self._generation = generation
        self._release_lock = asyncio.Lock()
        self._released = False

    @property
    def generation_id(self) -> int:
        return self._generation.generation_id

    @property
    def router(self) -> RouterT:
        return self._generation.router

    @property
    def config_snapshot(self) -> object | None:
        return self._generation.config_snapshot

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        """Release this generation reference exactly once."""
        async with self._release_lock:
            if self._released:
                return
            release_task = asyncio.create_task(self._owner._release(self._generation))
            _, cancelled = await _await_task_ignoring_cancellation(release_task)
            self._released = True
            if cancelled:
                raise asyncio.CancelledError

    async def __aenter__(self) -> RouterLease[RouterT]:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class RouterOwner[RouterT]:
    """Own atomically swappable Router generations and their resources."""

    def __init__(
        self,
        factory: Callable[[object | None], RouterT | Awaitable[RouterT]] | None = None,
    ) -> None:
        self._factory = factory or cast(
            Callable[[object | None], RouterT],
            lambda snapshot: Router(
                config_snapshot=snapshot,
                managed_generation=True,
            ),
        )
        self._operation_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._active: _RouterGeneration[RouterT] | None = None
        self._generations: dict[int, _RouterGeneration[RouterT]] = {}
        self._next_generation_id = 1
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False

    async def _build(self, config_snapshot: object | None) -> RouterT:
        built = self._factory(config_snapshot)
        if inspect.isawaitable(built):
            return await built
        return built

    async def start(self, config_snapshot: object | None = None) -> int:
        """Build the initial generation and make it available for leases."""
        async with self._operation_lock:
            async with self._state_lock:
                if self._closed or self._closing:
                    raise RuntimeError("RouterOwner is closed")
                if self._active is not None:
                    return self._active.generation_id
            build_task = asyncio.create_task(self._build(config_snapshot))
            router: RouterT | None = None
            installed = False
            try:
                router, cancelled = await _await_task_ignoring_cancellation(build_task)
                if cancelled:
                    raise asyncio.CancelledError
                async with self._state_lock:
                    generation = self._new_generation(router, config_snapshot)
                    self._active = generation
                    installed = True
                    return generation.generation_id
            finally:
                if router is not None and not installed:
                    await _close_uninstalled_router(router)

    def _new_generation(
        self,
        router: RouterT,
        config_snapshot: object | None,
    ) -> _RouterGeneration[RouterT]:
        generation = _RouterGeneration(
            self._next_generation_id,
            router,
            config_snapshot,
        )
        self._next_generation_id += 1
        self._generations[generation.generation_id] = generation
        return generation

    async def _acquire_active(self) -> RouterLease[RouterT]:
        async with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError("RouterOwner is closed")
            generation = self._active
            if generation is None:
                raise RuntimeError("RouterOwner has not been started")
            if generation.retired:
                raise RuntimeError("Active Router generation is retired")
            generation.references += 1
            return RouterLease(self, generation)

    async def acquire(self) -> RouterLease[RouterT]:
        """Acquire the active generation, rebuilding stale provider configuration once."""
        lease = await self._acquire_active()
        needs_reload = getattr(lease.router, "needs_provider_config_reload", None)
        if not callable(needs_reload) or not needs_reload():
            return lease
        stale_generation_id = lease.generation_id
        await lease.release()

        async with self._operation_lock:
            current = await self._acquire_active()
            try:
                current_needs_reload = getattr(
                    current.router,
                    "needs_provider_config_reload",
                    None,
                )
                if (
                    current.generation_id == stale_generation_id
                    and callable(current_needs_reload)
                    and current_needs_reload()
                ):
                    await self._rebuild_locked(current.config_snapshot)
            finally:
                await current.release()
        return await self._acquire_active()

    async def rebuild(
        self,
        config_snapshot: object | None = None,
        *,
        before_swap: Callable[[], object | None] | None = None,
    ) -> int:
        """Build a replacement before atomically retiring the active generation."""
        async with self._operation_lock:
            return await self._rebuild_locked(config_snapshot, before_swap=before_swap)

    async def _rebuild_locked(
        self,
        config_snapshot: object | None,
        *,
        before_swap: Callable[[], object | None] | None = None,
    ) -> int:
        """Rebuild while the caller owns the operation lock."""
        async with self._state_lock:
            if self._closed or self._closing:
                raise RuntimeError("RouterOwner is closed")
            build_snapshot = (
                self._active.config_snapshot
                if config_snapshot is None and self._active is not None
                else config_snapshot
            )
        build_task = asyncio.create_task(self._build(build_snapshot))
        router: RouterT | None = None
        installed = False
        try:
            router, cancelled = await _await_task_ignoring_cancellation(build_task)
            if cancelled:
                raise asyncio.CancelledError

            to_close: _RouterGeneration[RouterT] | None = None
            await self._state_lock.acquire()
            try:
                generation_snapshot = build_snapshot
                if before_swap is not None:
                    committed_snapshot = before_swap()
                    if inspect.isawaitable(committed_snapshot):
                        raise TypeError("before_swap callback must be synchronous")
                    if committed_snapshot is not None:
                        generation_snapshot = committed_snapshot
                previous = self._active
                generation = self._new_generation(router, generation_snapshot)
                self._active = generation
                installed = True
                if previous is not None:
                    previous.retired = True
                    if previous.references == 0 and not previous.closing:
                        previous.closing = True
                        to_close = previous
            finally:
                self._state_lock.release()
            if to_close is not None:
                self._schedule_generation_close(to_close)
            return generation.generation_id
        finally:
            if router is not None and not installed:
                await _close_uninstalled_router(router)

    async def _release(self, generation: _RouterGeneration[RouterT]) -> None:
        to_close = False
        async with self._state_lock:
            if generation.references <= 0:
                return
            generation.references -= 1
            if generation.retired and generation.references == 0 and not generation.closing:
                generation.closing = True
                to_close = True
        if to_close:
            await self._close_generation(generation)

    async def _close_generation(self, generation: _RouterGeneration[RouterT]) -> None:
        try:
            await _close_router_resources(generation.router)
        except Exception:
            logger.warning(
                "Failed to close Router generation %s",
                generation.generation_id,
                exc_info=True,
            )
        finally:
            async with self._state_lock:
                generation.closed = True
                self._generations.pop(generation.generation_id, None)
                generation.event().set()

    def _schedule_generation_close(self, generation: _RouterGeneration[RouterT]) -> None:
        """Own a post-swap close task without extending the rebuild commit path."""
        task = asyncio.create_task(self._close_generation(generation))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _shutdown(self) -> None:
        """Retire every generation and wait for its resources to close."""
        async with self._operation_lock:
            async with self._state_lock:
                if self._closed:
                    return
                self._closing = True
                active = self._active
                self._active = None
                if active is not None:
                    active.retired = True
                generations = list(self._generations.values())
                close_now: list[_RouterGeneration[RouterT]] = []
                for generation in generations:
                    if generation.references == 0 and not generation.closing:
                        generation.closing = True
                        close_now.append(generation)

            for generation in close_now:
                await self._close_generation(generation)
            await asyncio.gather(*(generation.event().wait() for generation in generations))
            async with self._state_lock:
                self._closed = True
                self._closing = False

    async def close(self) -> None:
        """Finish one owner-owned shutdown before propagating caller cancellation."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._shutdown())
        _, cancelled = await _await_task_ignoring_cancellation(self._close_task)
        if cancelled:
            raise asyncio.CancelledError


class _PrimedStream(AsyncIterator[ChunkT]):
    """Own a primed stream and close its inner iterator even before iteration."""

    def __init__(
        self,
        first_chunk: ChunkT | None,
        stream: AsyncIterator[ChunkT],
        selected_model: ModelRef | None = None,
    ) -> None:
        self._first_chunk = first_chunk
        self._has_first_chunk = first_chunk is not None
        self._stream = stream
        self._closed = False
        self.selected_model = selected_model

    def __aiter__(self) -> _PrimedStream[ChunkT]:
        return self

    async def __anext__(self) -> ChunkT:
        if self._closed:
            raise StopAsyncIteration
        if self._has_first_chunk:
            self._has_first_chunk = False
            first_chunk = self._first_chunk
            self._first_chunk = None
            return first_chunk  # type: ignore[return-value]
        try:
            return await anext(self._stream)
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._has_first_chunk = False
        self._first_chunk = None
        await close_async_iterator(self._stream)


def get_router() -> Router:
    """Get the singleton Router instance.

    Returns:
        The global Router instance
    """
    try:
        from router_maestro.runtime.request_context import get_current_request_context

        context = get_current_request_context()
    except ImportError:
        context = None
    if context is not None:
        return cast(Router, context.router)

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

    def __init__(
        self,
        config_snapshot: object | None = None,
        *,
        managed_generation: bool = False,
    ) -> None:
        self._config_snapshot = config_snapshot
        self._managed_generation = managed_generation
        self._close_lock = asyncio.Lock()
        self._closed_provider_ids: set[int] = set()
        self._closed = False
        self.providers: dict[str, BaseProvider] = {}
        # Model cache: maps model_id -> (provider_name, ModelInfo)
        self._models_cache: dict[str, tuple[str, ModelInfo]] = {}
        self._models_cache_ttl: TTLCache[bool] = TTLCache(CACHE_TTL_SECONDS)
        # Priorities config cache
        self._priorities_cache: TTLCache[PrioritiesConfig] = TTLCache(CACHE_TTL_SECONDS)
        # Fuzzy match result cache: raw_query -> resolved_cache_key (or None)
        self._fuzzy_cache: dict[str, str | None] = {}
        # Providers config cache
        self._providers_ttl: TTLCache[bool] = TTLCache(CACHE_TTL_SECONDS)
        snapshot_config = getattr(config_snapshot, "config", config_snapshot)
        if isinstance(snapshot_config, PrioritiesConfig):
            self._priorities_cache.set(deepcopy(snapshot_config))
        self._load_providers()

    def _load_providers(self) -> None:
        """Load providers from configuration."""
        custom_providers_config = load_providers_config()
        old_providers = self.providers
        self.providers = {}

        self._add_builtin_provider("github-copilot", CopilotProvider, old_providers)
        self._add_builtin_provider("openai", OpenAIProvider, old_providers)
        self._add_builtin_provider("anthropic", AnthropicProvider, old_providers)

        # Load custom providers from providers.json
        for provider_name, provider_config in custom_providers_config.providers.items():
            provider = self._create_custom_provider(provider_name, provider_config)
            if provider is not None:
                self.providers[provider_name] = provider

        self._providers_ttl.set(True)
        logger.info("Loaded %d providers", len(self.providers))

    def _add_builtin_provider(
        self,
        name: str,
        provider_cls: type[BaseProvider],
        old_providers: dict[str, BaseProvider],
    ) -> None:
        existing = old_providers.get(name)
        if isinstance(existing, provider_cls):
            self.providers[name] = existing
        else:
            self.providers[name] = provider_cls()
        logger.debug("Loaded built-in provider: %s", name)

    def _create_custom_provider(
        self,
        provider_name: str,
        provider_config,
    ) -> BaseProvider | None:
        from router_maestro.auth.repository import CredentialRepository
        from router_maestro.providers.custom_factory import create_custom_provider

        provider = create_custom_provider(
            provider_name,
            provider_config,
            credential_repository=CredentialRepository(),
        )
        if provider is None and provider_config.type != "openai-compatible":
            logger.warning(
                "Unknown provider type '%s' for %s; skipping",
                provider_config.type,
                provider_name,
            )
        elif provider is None:
            logger.debug("Skipping custom provider %s (no API key)", provider_name)
        else:
            logger.debug("Loaded custom provider: %s", provider_name)
        return provider

    def _get_priorities_config(self) -> PrioritiesConfig:
        """Get priorities config with caching."""
        cached = self._priorities_cache.get()
        if cached is not None:
            return cached

        if getattr(self, "_managed_generation", False):
            frozen = self._priorities_cache.peek()
            if frozen is not None:
                return frozen

        config = load_priorities_config()
        self._priorities_cache.set(config)
        return config

    def _ensure_providers_fresh(self) -> None:
        """Ensure providers config is fresh, reload if expired."""
        if getattr(self, "_managed_generation", False):
            return
        if not self._providers_ttl.is_valid:
            logger.debug("Providers config expired, reloading")
            self._load_providers()
            # Also invalidate models cache since providers may have changed
            self._models_cache.clear()
            self._fuzzy_cache.clear()
            self._models_cache_ttl.clear()

    def needs_provider_config_reload(self) -> bool:
        """Return whether this immutable managed generation needs replacement."""
        return self._managed_generation and not self._providers_ttl.is_valid

    async def close(self) -> None:
        """Close every provider resource owned by this Router exactly once."""
        async with self._close_lock:
            if self._closed:
                return
            seen = set(self._closed_provider_ids)
            for provider in self.providers.values():
                if id(provider) in seen:
                    continue
                seen.add(id(provider))
                close = getattr(provider, "close", None)
                if not callable(close):
                    continue
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.warning(
                        "Failed to close provider %s",
                        getattr(provider, "name", type(provider).__name__),
                        exc_info=True,
                    )
                self._closed_provider_ids.add(id(provider))
            self._closed = True

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

    @staticmethod
    def _is_qualified_cache_entry(
        key: str,
        entry: tuple[str, ModelInfo],
    ) -> bool:
        """Return whether ``key`` is this entry's canonical public identity."""
        provider_name, model = entry
        return key == ModelRef(provider_name, model.id).qualified_id

    def _is_explicit_model_id(self, model_id: str) -> bool:
        """Distinguish public provider prefixes from slashes inside upstream IDs."""
        if "/" not in model_id:
            return False
        entry = self._models_cache.get(model_id)
        return entry is None or self._is_qualified_cache_entry(model_id, entry)

    async def _ensure_models_cache(self) -> None:
        """Ensure the models cache is populated and not expired."""
        # First ensure providers config is fresh
        self._ensure_providers_fresh()

        # Check if cache is still valid
        if self._models_cache_ttl.is_valid:
            return

        logger.debug("Initializing models cache")
        previous_cache = self._models_cache
        new_cache: dict[str, tuple[str, ModelInfo]] = {}
        any_provider_failed = False
        for provider_name, provider in self.providers.items():
            if provider.is_authenticated():
                try:
                    await provider.ensure_token()
                    models = await provider.list_models()
                    for model in models:
                        try:
                            ref = (
                                ModelRef.from_qualified_catalog_id(provider_name, model.id)
                                if model.id_is_qualified
                                else ModelRef.from_catalog_id(provider_name, model.id)
                            )
                        except ValueError:
                            logger.warning(
                                "model_catalog_entry_skipped provider=%s reason=provider_mismatch",
                                provider_name,
                            )
                            continue
                        normalized_model = replace(
                            model,
                            id=ref.upstream_id,
                            provider=provider_name,
                            id_is_qualified=False,
                        )
                        entry = (provider_name, normalized_model)
                        # Canonical public identities always win over convenience
                        # aliases when their strings collide. This can happen when
                        # an upstream ID itself contains '/', for example
                        # ``meta-llama/llama-3``.
                        existing = new_cache.get(ref.qualified_id)
                        if existing is None or not self._is_qualified_cache_entry(
                            ref.qualified_id,
                            existing,
                        ):
                            new_cache[ref.qualified_id] = entry

                        # Keep a bare alias only when it does not shadow a
                        # canonical provider/model identity. Duplicate aliases
                        # retain the first catalog entry; RoutePlan later chooses
                        # the provider by configured priority and capability.
                        alias_entry = new_cache.get(ref.upstream_id)
                        if alias_entry is None:
                            new_cache[ref.upstream_id] = entry
                    logger.debug("Cached %d models from %s", len(models), provider_name)
                except ProviderError as e:
                    any_provider_failed = True
                    logger.warning(
                        "model_catalog_failed provider=%s kind=%s retryable=%s",
                        provider_name,
                        e.kind.value,
                        str(e.retryable).lower(),
                    )
                    # A transient failure (e.g. a token-refresh race that briefly
                    # reports the provider as unauthenticated, or an upstream 502
                    # while minting a token) must not drop a provider's models from
                    # the catalog, so retain its previously cached entries and let
                    # in-flight requests keep resolving. Trade-off: a *persistent*
                    # provider failure keeps serving the last-known catalog until a
                    # refresh finally succeeds; that is surfaced by the repeated
                    # model_catalog_failed warning above (the TTL is not refreshed
                    # on failure, so each request retries) rather than by silently
                    # emptying the cache.
                    for key, cached_entry in previous_cache.items():
                        if cached_entry[0] == provider_name and key not in new_cache:
                            new_cache[key] = cached_entry
                    continue

        self._models_cache = new_cache
        self._fuzzy_cache.clear()
        # Only mark the cache fresh when every authenticated provider refreshed
        # successfully. If any provider failed, leave the TTL expired so the next
        # request retries promptly instead of serving a degraded catalog for a
        # full TTL window.
        if not any_provider_failed:
            self._models_cache_ttl.set(True)
        logger.info("Models cache initialized with %d entries", len(self._models_cache))

        self._apply_model_overrides()

    def _apply_model_overrides(self) -> None:
        """Apply per-model token limit overrides from priorities config.

        Handles both bare model keys (e.g. "claude-opus-4.6") and
        provider-qualified keys (e.g. "github-copilot/claude-opus-4.6").
        A bare key override is applied to all matching cache entries
        (both bare and provider-qualified) for consistency.
        """
        priorities = self._get_priorities_config()
        for key, override in priorities.model_overrides.items():
            # Collect all cache keys that should receive this override
            matching_keys: list[str] = []
            if key in self._models_cache:
                matching_keys.append(key)
            entry_for_key = self._models_cache.get(key)
            key_is_qualified = entry_for_key is not None and self._is_qualified_cache_entry(
                key,
                entry_for_key,
            )
            # For bare model names (including namespaced upstream IDs), also
            # apply to every provider-qualified entry for that upstream ID.
            if not key_is_qualified:
                for cache_key in self._models_cache:
                    cache_entry = self._models_cache[cache_key]
                    if (
                        self._is_qualified_cache_entry(cache_key, cache_entry)
                        and cache_entry[1].id == key
                    ):
                        matching_keys.append(cache_key)
            for cache_key in matching_keys:
                pname, info = self._models_cache[cache_key]
                self._models_cache[cache_key] = (
                    pname,
                    info.with_overrides(
                        max_prompt_tokens=override.max_prompt_tokens,
                        max_output_tokens=override.max_output_tokens,
                        max_context_window_tokens=override.max_context_window_tokens,
                    ),
                )

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get ModelInfo for a model, or None if not found."""
        await self._ensure_models_cache()
        entry = self._models_cache.get(model_id)
        if entry and (
            not self._is_explicit_model_id(model_id)
            or self._is_qualified_cache_entry(model_id, entry)
        ):
            return deepcopy(entry[1])

        if model_id in self._fuzzy_cache:
            matched_key = self._fuzzy_cache[model_id]
            if matched_key is None:
                return None
            entry = self._models_cache.get(matched_key)
            return deepcopy(entry[1]) if entry else None

        try:
            matched_key = fuzzy_match_model(model_id, self._models_cache)
        except AmbiguousModelMatchError:
            return None
        self._fuzzy_cache[model_id] = matched_key
        if matched_key is None:
            return None

        entry = self._models_cache.get(matched_key)
        if entry:
            logger.debug("Fuzzy matched model info '%s' -> '%s'", model_id, matched_key)
            return deepcopy(entry[1])
        return None

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
                raise ProviderError(
                    "No models available for auto-routing",
                    status_code=503,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_STATUS,
                )
            return result

        # Explicit model specified - find in cache
        result = await self._find_model_in_cache(model_id)
        if not result:
            logger.warning("Model not found: %s", model_id)
            raise ProviderError(
                f"Model '{model_id}' not found in any provider",
                status_code=404,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )
        return result

    def _model_capabilities(
        self,
        model: ModelInfo,
        provider: BaseProvider,
    ) -> ModelCapabilities:
        ref = ModelRef(provider=model.provider, upstream_id=model.id)

        def support(value: bool | None) -> CapabilitySupport:
            if value is True:
                return CapabilitySupport.SUPPORTED
            if value is False:
                return CapabilitySupport.UNSUPPORTED
            return CapabilitySupport.UNKNOWN

        operations = {
            operation: (
                support(model.operation_capabilities.get(operation.value))
                if provider.capabilities.supports(operation)
                else CapabilitySupport.UNSUPPORTED
            )
            for operation in Operation
        }
        feature_values: dict[str, bool] = dict(model.feature_capabilities)
        if Feature.VISION.value not in feature_values and model.supports_vision:
            feature_values[Feature.VISION.value] = True
        if Feature.REASONING.value not in feature_values:
            if model.reasoning_effort_values is not None:
                feature_values[Feature.REASONING.value] = bool(model.reasoning_effort_values)
            elif model.supports_thinking:
                feature_values[Feature.REASONING.value] = True
        features = {feature: support(feature_values.get(feature.value)) for feature in Feature}
        return ModelCapabilities(
            model=ref,
            operations=operations,
            features=features,
            reasoning_effort_values=(
                tuple(model.reasoning_effort_values)
                if model.reasoning_effort_values is not None
                else None
            ),
            max_output_tokens=model.max_output_tokens,
        )

    def _route_candidate(
        self,
        provider_name: str,
        model_id: str,
        provider: BaseProvider,
        operation: Operation,
        features: RequestFeatures,
    ) -> RouteCandidate | None:
        entry = self._models_cache.get(f"{provider_name}/{model_id}")
        if entry is None:
            return None
        model = entry[1]
        capabilities = self._model_capabilities(model, provider)
        return RouteCandidate(
            model=capabilities.model,
            provider=provider,
            capabilities=capabilities,
            evaluated_operation=operation,
            evaluated_features=features,
            support=capabilities.support_for(operation, features),
        )

    def _configured_candidates(
        self,
        operation: Operation,
        features: RequestFeatures,
    ) -> list[RouteCandidate]:
        candidates: list[RouteCandidate] = []
        seen: set[ModelRef] = set()
        for key in self._get_priorities_config().priorities:
            provider_name, model_id = self._parse_model_key(key)
            provider = self.providers.get(provider_name)
            if provider is None or not provider.is_authenticated():
                continue
            candidate = self._route_candidate(
                provider_name,
                model_id,
                provider,
                operation,
                features,
            )
            if candidate is None or candidate.model in seen:
                continue
            seen.add(candidate.model)
            candidates.append(candidate)
        return candidates

    def _all_available_candidates(
        self,
        operation: Operation,
        features: RequestFeatures,
    ) -> list[RouteCandidate]:
        candidates: list[RouteCandidate] = []
        seen: set[ModelRef] = set()
        for key, entry in self._models_cache.items():
            provider_name, model = entry
            if not self._is_qualified_cache_entry(key, entry):
                continue
            provider = self.providers.get(provider_name)
            if provider is None or not provider.is_authenticated():
                continue
            candidate = self._route_candidate(
                provider_name,
                model.id,
                provider,
                operation,
                features,
            )
            if candidate is None or candidate.model in seen:
                continue
            seen.add(candidate.model)
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _rank_compatible(candidates: list[RouteCandidate]) -> list[RouteCandidate]:
        supported = [
            candidate
            for candidate in candidates
            if candidate.support is CapabilitySupport.SUPPORTED
        ]
        unknown = [
            candidate for candidate in candidates if candidate.support is CapabilitySupport.UNKNOWN
        ]
        return [*supported, *unknown]

    def _explicit_fallback_candidates(
        self,
        primary: RouteCandidate,
        operation: Operation,
        features: RequestFeatures,
    ) -> list[RouteCandidate]:
        fallback_config = self._get_priorities_config().fallback
        if fallback_config.strategy is FallbackStrategy.NONE:
            return []
        if fallback_config.strategy is FallbackStrategy.SAME_MODEL:
            candidates: list[RouteCandidate] = []
            for provider_name, provider in self.providers.items():
                if provider_name == primary.model.provider or not provider.is_authenticated():
                    continue
                candidate = self._route_candidate(
                    provider_name,
                    primary.model.upstream_id,
                    provider,
                    operation,
                    features,
                )
                if candidate is not None:
                    candidates.append(candidate)
            return self._rank_compatible(candidates)

        configured = self._configured_candidates(operation, features)
        configured_refs = [candidate.model for candidate in configured]
        if primary.model in configured_refs:
            configured = configured[configured_refs.index(primary.model) + 1 :]
        return self._rank_compatible(
            [candidate for candidate in configured if candidate.model != primary.model]
        )

    def _build_route_plan(
        self,
        operation: Operation,
        features: RequestFeatures,
        primary: RouteCandidate,
        candidates: list[RouteCandidate],
        *,
        explicit: bool,
    ) -> RoutePlan:
        """Freeze the ordered fallback pool and retry limit in one route plan."""
        fallback = self._get_priorities_config().fallback
        if fallback.strategy is FallbackStrategy.NONE:
            pool: tuple[RouteCandidate, ...] = ()
            return RoutePlan(
                operation,
                features,
                primary,
                (),
                explicit,
                fallback_pool=pool,
                max_fallback_attempts=0,
            )
        compatible = self._rank_compatible(
            [candidate for candidate in candidates if candidate.model != primary.model]
        )
        if fallback.strategy is FallbackStrategy.SAME_MODEL:
            compatible = [
                candidate
                for candidate in compatible
                if candidate.model.upstream_id == primary.model.upstream_id
            ]
        pool = tuple(compatible)
        return RoutePlan(
            operation,
            features,
            primary,
            pool[: fallback.maxRetries],
            explicit,
            fallback_pool=pool,
            max_fallback_attempts=fallback.maxRetries,
        )

    def _same_model_catalog_candidates(
        self,
        primary: RouteCandidate,
        operation: Operation,
        features: RequestFeatures,
        configured: list[RouteCandidate],
    ) -> list[RouteCandidate]:
        ordered: list[RouteCandidate] = []
        seen: set[ModelRef] = {primary.model}
        for candidate in [
            *configured,
            *self._all_available_candidates(operation, features),
        ]:
            if candidate.model in seen or candidate.model.upstream_id != primary.model.upstream_id:
                continue
            seen.add(candidate.model)
            ordered.append(candidate)
        return ordered

    async def plan_route(
        self,
        model_id: str,
        operation: Operation,
        features: RequestFeatures | None = None,
    ) -> RoutePlan:
        """Resolve one immutable, capability-aware execution plan."""
        try:
            if model_id != AUTO_ROUTE_MODEL:
                ModelRef("model-alias", model_id)
        except (TypeError, ValueError) as error:
            raise ProviderError(
                "Model ID must be a non-empty provider/model identity",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
                cause=error,
            ) from error
        await self._ensure_models_cache()
        features = features or RequestFeatures()
        explicit = self._is_explicit_model_id(model_id)
        if explicit:
            try:
                provider_name, upstream_id = self._parse_model_key(model_id)
                ModelRef(provider_name, upstream_id)
            except (TypeError, ValueError) as error:
                raise ProviderError(
                    "Model ID must be a non-empty provider/model identity",
                    status_code=400,
                    retryable=False,
                    kind=ProviderFailureKind.CLIENT_REQUEST,
                    cause=error,
                ) from error

        if model_id != AUTO_ROUTE_MODEL and not explicit:
            if model_id in self._models_cache:
                upstream_id = model_id
            else:
                try:
                    alias_key = fuzzy_match_model(model_id, self._models_cache)
                except AmbiguousModelMatchError as error:
                    raise ProviderError(
                        f"Model alias '{model_id}' is ambiguous; use provider/model",
                        status_code=400,
                        kind=ProviderFailureKind.CLIENT_REQUEST,
                        cause=error,
                    ) from error
                if alias_key is not None:
                    # A fuzzy result may be a raw namespaced upstream alias
                    # (for example ``meta-llama/llama-3``). The cache entry owns
                    # its provenance; parsing the key would mistake the raw
                    # namespace for a provider prefix and lose part of the ID.
                    upstream_id = self._models_cache[alias_key][1].id
                else:
                    raise ProviderError(
                        f"Model alias '{model_id}' not found in any provider",
                        status_code=404,
                        kind=ProviderFailureKind.CLIENT_REQUEST,
                    )
            ordered_alias_candidates = [
                *self._configured_candidates(operation, features),
                *self._all_available_candidates(operation, features),
            ]
            alias_candidates: list[RouteCandidate] = []
            seen_aliases: set[ModelRef] = set()
            for candidate in ordered_alias_candidates:
                if candidate.model.upstream_id != upstream_id or candidate.model in seen_aliases:
                    continue
                seen_aliases.add(candidate.model)
                alias_candidates.append(candidate)
            ranked_aliases = self._rank_compatible(alias_candidates)
            if ranked_aliases:
                primary = ranked_aliases[0]
                return self._build_route_plan(
                    operation,
                    features,
                    primary,
                    ranked_aliases[1:],
                    explicit=False,
                )
            raise NoCompatibleRouteError(
                f"No model matching alias '{model_id}' supports {operation.value}"
            )

        if explicit:
            provider_name, upstream_id, provider = await self._resolve_provider(model_id)
            primary = self._route_candidate(
                provider_name,
                upstream_id,
                provider,
                operation,
                features,
            )
            if primary is None:
                raise ProviderError(
                    f"Model '{model_id}' not found",
                    status_code=404,
                    kind=ProviderFailureKind.CLIENT_REQUEST,
                )
            return self._build_route_plan(
                operation,
                features,
                primary,
                self._explicit_fallback_candidates(primary, operation, features),
                explicit=True,
            )

        configured = self._configured_candidates(operation, features)
        if not configured:
            configured = self._all_available_candidates(operation, features)
        ranked = self._rank_compatible(configured)
        if not ranked:
            raise NoCompatibleRouteError(f"No configured model supports {operation.value}")
        primary = ranked[0]
        fallback_candidates = ranked[1:]
        if self._get_priorities_config().fallback.strategy is FallbackStrategy.SAME_MODEL:
            fallback_candidates = self._same_model_catalog_candidates(
                primary,
                operation,
                features,
                fallback_candidates,
            )
        return self._build_route_plan(
            operation,
            features,
            primary,
            fallback_candidates,
            explicit=False,
        )

    @staticmethod
    def _validate_plan_primary(plan: RoutePlan) -> None:
        if plan.primary.support is CapabilitySupport.UNSUPPORTED:
            unsupported_feature = next(
                (
                    feature
                    for feature in plan.features.required()
                    if plan.primary.capabilities.feature(feature) is CapabilitySupport.UNSUPPORTED
                ),
                None,
            )
            if unsupported_feature is not None:
                parameter = unsupported_feature.value
                if unsupported_feature is Feature.REASONING:
                    parameter = plan.features.reasoning_parameter or parameter
                elif unsupported_feature is Feature.PARALLEL_TOOLS:
                    parameter = "parallel_tool_calls"
                raise RequestOptionError(
                    f"Model '{plan.primary.model.qualified_id}' does not support "
                    f"the requested {unsupported_feature.value} feature",
                    provider=plan.primary.model.provider,
                    model=plan.primary.model.upstream_id,
                    parameter=parameter,
                )
            raise RequestOptionError(
                f"Model '{plan.primary.model.qualified_id}' does not support "
                f"{plan.operation.value}",
                provider=plan.primary.model.provider,
                model=plan.primary.model.upstream_id,
                parameter=plan.operation.value,
            )

    def _plan_execution_candidates(
        self,
        plan: RoutePlan,
        fallback: bool,
    ) -> tuple[RouteCandidate, ...]:
        self._validate_plan_primary(plan)
        if not fallback:
            return (plan.primary,)
        return plan.candidates

    def _chat_request_for_candidate(
        self,
        request: ChatRequest,
        candidate: RouteCandidate,
    ) -> ChatRequest:
        """Build the candidate chat request with the resolved upstream model."""
        return self._create_request_with_model(request, candidate.model.upstream_id)

    def _validate_chat_candidate(
        self,
        request: ChatRequest,
        candidate: RouteCandidate,
        *,
        stream: bool,
    ) -> None:
        candidate_request = self._chat_request_for_candidate(request, candidate)
        candidate.provider.validate_chat_request(candidate_request, stream=stream)

    async def _execute_chat_plan_nonstream(
        self,
        plan: RoutePlan,
        request: ChatRequest,
        fallback: bool,
        request_for_candidate: Callable[[ModelRef], ChatRequest] | None = None,
    ) -> tuple[ChatResponse, str]:
        """Execute a Chat plan using each candidate's frozen transport operation."""
        candidates = self._plan_execution_candidates(plan, fallback)

        async def attempt(candidate: RouteCandidate) -> ChatResponse:
            source_request = (
                request_for_candidate(candidate.model) if request_for_candidate else request
            )
            candidate_request = self._chat_request_for_candidate(source_request, candidate)
            await candidate.provider.ensure_token()
            return await candidate.provider.chat_completion(candidate_request)

        return await self._execute_attempts(plan, candidates, attempt)

    async def _execute_chat_plan_stream(
        self,
        plan: RoutePlan,
        request: ChatRequest,
        fallback: bool,
        request_for_candidate: Callable[[ModelRef], ChatRequest] | None = None,
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        """Open a Chat stream for each candidate."""
        candidates = self._plan_execution_candidates(plan, fallback)

        async def attempt(candidate: RouteCandidate) -> AsyncIterator[ChatStreamChunk]:
            source_request = (
                request_for_candidate(candidate.model) if request_for_candidate else request
            )
            candidate_request = self._chat_request_for_candidate(source_request, candidate)
            await candidate.provider.ensure_token()
            stream = candidate.provider.chat_completion_stream(candidate_request)
            wrapped = self._wrap_stream_errors(
                stream,
                candidate.model.provider,
                lambda message: message,
            )
            first_chunk = await anext(wrapped, None)
            if first_chunk is None:
                cause = ValueError("upstream stream ended before a canonical chunk")
                raise ProviderError(
                    f"{candidate.model.provider} returned an empty upstream stream",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                    upstream_status_code=200,
                    provider=candidate.model.provider,
                    model=candidate.model.upstream_id,
                    cause=cause,
                ) from cause
            return self._chain_first_chunk(
                first_chunk,
                wrapped,
                selected_model=candidate.model,
            )

        return await self._execute_attempts(plan, candidates, attempt)

    async def _execute_plan_nonstream(
        self,
        plan: RoutePlan,
        request: RequestT,
        fallback: bool,
        build_request: Callable[[RequestT, str], RequestT],
        call: Callable[[BaseProvider, RequestT], Awaitable[ResponseT]],
    ) -> tuple[ResponseT, str]:
        """Execute a non-stream request with the shared pre-commit attempt policy."""
        candidates = self._plan_execution_candidates(plan, fallback)

        async def attempt(candidate: RouteCandidate) -> ResponseT:
            candidate_request = build_request(request, candidate.model.upstream_id)
            await candidate.provider.ensure_token()
            return await call(candidate.provider, candidate_request)

        return await self._execute_attempts(
            plan,
            candidates,
            attempt,
        )

    @staticmethod
    async def execute_plan_nonstream(
        plan: RoutePlan,
        call: Callable[[RouteCandidate], Awaitable[ResponseT]],
        *,
        fallback: bool = True,
    ) -> tuple[ResponseT, str]:
        """Execute an existing plan while keeping protocol transport at the caller."""
        Router._validate_plan_primary(plan)
        candidates = plan.candidates if fallback else (plan.primary,)
        return await Router._execute_attempts(plan, candidates, call)

    @staticmethod
    async def execute_plan_stream(
        plan: RoutePlan,
        call: Callable[[RouteCandidate], AsyncIterator[ChunkT]],
        *,
        fallback: bool = True,
        log_prefix: str = "",
    ) -> tuple[AsyncIterator[ChunkT], str]:
        """Prime a route-owned stream before selecting one candidate."""
        Router._validate_plan_primary(plan)
        candidates = plan.candidates if fallback else (plan.primary,)

        async def attempt(candidate: RouteCandidate) -> AsyncIterator[ChunkT]:
            stream = call(candidate)
            wrapped = Router._wrap_stream_errors(
                stream,
                candidate.model.provider,
                lambda message: f"{log_prefix} {message}".strip(),
            )
            first_chunk = await anext(wrapped, None)
            if first_chunk is None:
                cause = ValueError("upstream stream ended before a canonical frame")
                raise ProviderError(
                    f"{candidate.model.provider} returned an empty upstream stream",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                    upstream_status_code=200,
                    provider=candidate.model.provider,
                    model=candidate.model.upstream_id,
                    cause=cause,
                ) from cause
            return _PrimedStream(first_chunk, wrapped, selected_model=candidate.model)

        return await Router._execute_attempts(
            plan,
            candidates,
            attempt,
            log_prefix,
            success_event="route_attempt_selected",
        )

    async def _execute_plan_stream(
        self,
        plan: RoutePlan,
        request: RequestT,
        fallback: bool,
        build_request: Callable[[RequestT, str], RequestT],
        call: Callable[[BaseProvider, RequestT], AsyncIterator[ChunkT]],
        log_prefix: str,
    ) -> tuple[AsyncIterator[ChunkT], str]:
        """Open and prime a stream with the shared pre-commit attempt policy."""
        candidates = self._plan_execution_candidates(plan, fallback)
        return await self._execute_stream_attempts(
            plan=plan,
            candidates=candidates,
            request=request,
            build_request=build_request,
            call_stream=call,
            log_prefix=log_prefix,
        )

    @staticmethod
    async def _execute_attempts(
        plan: RoutePlan,
        candidates: tuple[RouteCandidate, ...],
        attempt: Callable[[RouteCandidate], Awaitable[ResponseT]],
        log_prefix: str = "",
        success_event: str | None = None,
    ) -> tuple[ResponseT, str]:
        """Execute planned candidates using one failure and stop policy."""
        if not candidates:
            raise ProviderError(
                "No route candidates available",
                status_code=503,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_STATUS,
            )

        ledger = AttemptLedger()
        prefix = f"{log_prefix} " if log_prefix else ""
        for index, candidate in enumerate(candidates, start=1):
            try:
                result = await attempt(candidate)
            except ProviderError as error:
                record = AttemptRecord.from_failure(candidate, plan.operation, error)
                ledger.record(record)
                allows_fallback = failure_allows_fallback(plan, candidate, error)
                has_next = index < len(candidates)
                should_fallback = allows_fallback and has_next
                if should_fallback:
                    decision = "fallback"
                elif allows_fallback:
                    decision = "exhausted"
                else:
                    decision = "stop"
                logger.warning(
                    "%sroute_attempt_failed attempt=%d provider=%s model=%s "
                    "operation=%s kind=%s downstream_status=%d upstream_status=%s "
                    "retryable=%s decision=%s",
                    prefix,
                    index,
                    record.provider,
                    record.model.qualified_id,
                    record.operation.value,
                    record.failure_kind.value,
                    record.downstream_status_code,
                    record.upstream_status_code,
                    str(record.retryable).lower(),
                    decision,
                )
                if should_fallback:
                    continue
                routed_error = error.with_attempts(ledger.snapshot())
                raise routed_error from error

            event = success_event or (
                "route_attempt_selected"
                if plan.operation in {Operation.CHAT_STREAM, Operation.RESPONSES_STREAM}
                else "route_attempt_succeeded"
            )
            logger.info(
                "%s%s attempt=%d provider=%s model=%s operation=%s",
                prefix,
                event,
                index,
                candidate.model.provider,
                candidate.model.qualified_id,
                plan.operation.value,
            )
            if isinstance(result, (ChatResponse, ResponsesResponse)):
                result = replace(result, selected_model=candidate.model)
            return result, candidate.model.provider

        raise AssertionError("route attempt loop ended without a result")

    async def _execute_stream_attempts(
        self,
        plan: RoutePlan,
        candidates: tuple[RouteCandidate, ...],
        request: RequestT,
        build_request: Callable[[RequestT, str], RequestT],
        call_stream: Callable[[BaseProvider, RequestT], AsyncIterator[ChunkT]],
        log_prefix: str,
    ) -> tuple[AsyncIterator[ChunkT], str]:
        """Prime each candidate before committing the selected stream."""

        async def attempt(candidate: RouteCandidate) -> AsyncIterator[ChunkT]:
            candidate_request = build_request(request, candidate.model.upstream_id)
            await candidate.provider.ensure_token()
            stream = call_stream(candidate.provider, candidate_request)
            wrapped = self._wrap_stream_errors(
                stream,
                candidate.model.provider,
                lambda message: f"{log_prefix} {message}".strip(),
            )
            first_chunk = await anext(wrapped, None)
            if first_chunk is None:
                cause = ValueError("upstream stream ended before a canonical chunk")
                raise ProviderError(
                    f"{candidate.model.provider} returned an empty upstream stream",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                    upstream_status_code=200,
                    provider=candidate.model.provider,
                    model=candidate.model.upstream_id,
                    cause=cause,
                ) from cause
            return self._chain_first_chunk(
                first_chunk,
                wrapped,
                selected_model=candidate.model,
            )

        return await self._execute_attempts(
            plan,
            candidates,
            attempt,
            log_prefix,
        )

    async def _plan_completion_route(
        self,
        model_id: str,
        operation: Operation,
        features: RequestFeatures,
    ) -> RoutePlan:
        try:
            return await self.plan_route(model_id, operation, features)
        except NoCompatibleRouteError as error:
            raise ProviderError(
                str(error),
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
                cause=error,
            ) from error

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
        return replace(
            original_request,
            model=model_id,
            messages=deepcopy(original_request.messages),
            tools=deepcopy(original_request.tools),
            tool_choice=deepcopy(original_request.tool_choice),
            stop=deepcopy(original_request.stop),
            stop_sequences=deepcopy(original_request.stop_sequences),
            metadata=deepcopy(original_request.metadata),
            provider_extensions=deepcopy(original_request.provider_extensions),
            extra={},
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
        for key, entry in self._models_cache.items():
            provider_name, model = entry
            if not self._is_qualified_cache_entry(key, entry):
                continue
            provider = self.providers.get(provider_name)
            if provider and provider.is_authenticated():
                logger.debug("Auto-route fallback: %s", key)
                return provider_name, model.id, provider

        return None

    async def _find_model_in_cache(self, model_id: str) -> tuple[str, str, BaseProvider] | None:
        """Find a model in the cache.

        Tries exact match first, then falls back to fuzzy matching.

        Args:
            model_id: Model ID (can be 'provider/model' or just 'model')

        Returns:
            Tuple of (provider_name, actual_model_id, provider) or None
        """
        await self._ensure_models_cache()

        # If model_id includes provider prefix (e.g., "github-copilot/gpt-4o")
        if self._is_explicit_model_id(model_id):
            provider_name, actual_model_id = self._parse_model_key(model_id)
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if provider.is_authenticated():
                    # Check if the model exists for this provider
                    if model_id in self._models_cache:
                        return provider_name, actual_model_id, provider
            # Fall through to fuzzy (don't return None yet)

        # Simple model_id (e.g., "gpt-4o") - look up in cache
        if model_id in self._models_cache:
            provider_name, model = self._models_cache[model_id]
            provider = self.providers.get(provider_name)
            if provider and provider.is_authenticated():
                return provider_name, model.id, provider

        # Check fuzzy cache first
        if model_id in self._fuzzy_cache:
            matched_key = self._fuzzy_cache[model_id]
            if matched_key is None:
                return None  # Negative cache hit
            if matched_key in self._models_cache:
                provider_name, _ = self._models_cache[matched_key]
                provider = self.providers.get(provider_name)
                if provider and provider.is_authenticated():
                    logger.debug("Fuzzy cache hit: '%s' -> '%s'", model_id, matched_key)
                    return provider_name, self._models_cache[matched_key][1].id, provider

        # Fuzzy matching fallback
        try:
            matched_key = fuzzy_match_model(model_id, self._models_cache)
        except AmbiguousModelMatchError as error:
            raise ProviderError(
                f"Model alias '{model_id}' is ambiguous; use provider/model",
                status_code=400,
                kind=ProviderFailureKind.CLIENT_REQUEST,
                cause=error,
            ) from error
        self._fuzzy_cache[model_id] = matched_key  # Cache result (including None)
        if matched_key is not None:
            provider_name, _ = self._models_cache[matched_key]
            provider = self.providers.get(provider_name)
            if provider and provider.is_authenticated():
                logger.info(
                    "Fuzzy matched '%s' -> '%s' (provider: %s)",
                    model_id,
                    matched_key,
                    provider_name,
                )
                return provider_name, self._models_cache[matched_key][1].id, provider

        return None

    async def chat_completion(
        self,
        request: ChatRequest,
        fallback: bool = True,
        *,
        prepared_plan: PreparedChatCompletion | None = None,
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
        request = deepcopy(request)
        prepared = prepared_plan or await self.prepare_chat_completion(
            request,
            fallback=fallback,
        )
        if not isinstance(prepared, PreparedChatCompletion) or not prepared.matches(request):
            raise ProviderError(
                "Prepared Chat completion does not match the current request",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )
        plan = prepared.plan
        execution_request = prepared.request_for_execution()
        logger.info("Routing request to %s", plan.primary.model.qualified_id)
        result, used_provider = await self._execute_chat_plan_nonstream(
            plan,
            execution_request,
            fallback,
            prepared.request_for_execution,
        )
        return result, used_provider

    async def plan_chat_completion(
        self,
        request: ChatRequest,
        *,
        stream: bool,
    ) -> RoutePlan:
        """Plan Chat once so a boundary can preprocess against the same candidate."""
        request = deepcopy(request)
        operation = Operation.CHAT_STREAM if stream else Operation.CHAT
        return await self._plan_completion_route(
            request.model,
            operation,
            RequestFeatures.for_chat(request),
        )

    def prepare_planned_chat_completion(
        self,
        plan: RoutePlan,
        request: ChatRequest,
        *,
        fallback: bool = True,
        candidate_requests: Mapping[ModelRef, ChatRequest] | None = None,
    ) -> PreparedChatCompletion | PreparedChatStream:
        """Prevalidate and bind a transformed request to its existing Chat plan."""
        request = deepcopy(request)
        candidate_requests = {
            model: deepcopy(candidate_request)
            for model, candidate_request in (candidate_requests or {}).items()
        }
        if candidate_requests and candidate_requests.get(plan.primary.model) != request:
            raise ProviderError(
                "Primary candidate request does not match the planned Chat request",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )

        def request_for(candidate: RouteCandidate) -> ChatRequest:
            return candidate_requests.get(candidate.model, request)

        expected_operation = Operation.CHAT_STREAM if request.stream else Operation.CHAT
        if plan.operation is not expected_operation:
            raise ProviderError(
                "Planned Chat operation does not match the current request",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )

        def validate_client_features(
            candidate: RouteCandidate,
            candidate_request: ChatRequest,
        ) -> None:
            transformed_features = RequestFeatures.for_chat(candidate_request)
            client_feature_drift = (
                transformed_features.tools != plan.features.tools
                or transformed_features.vision != plan.features.vision
                or transformed_features.parallel_tools != plan.features.parallel_tools
            )
            if client_feature_drift:
                raise ProviderError(
                    "Planned Chat features do not match the current request",
                    status_code=400,
                    retryable=False,
                    kind=ProviderFailureKind.CLIENT_REQUEST,
                )

            # The primary is the only executable route; when the caller scopes
            # its per-candidate request (via ``candidate_requests``) it owns that
            # request's reasoning shape. The Anthropic route, for example, runs
            # ``_apply_thinking_budget`` per candidate and legitimately disables
            # thinking when the requested ``max_tokens`` leaves no output
            # headroom for a thinking budget. That server-initiated downgrade is
            # safe — a non-reasoning request always runs on a reasoning-capable
            # route — so the primary proceeds instead of 400'ing. Fallbacks that
            # lose reasoning are still dropped (see ``prevalidate_plan``), and an
            # unscoped primary still enforces the client's explicit reasoning.
            primary_scoped = (
                candidate.model == plan.primary.model and candidate.model in candidate_requests
            )
            reasoning_drift = (
                plan.features.reasoning
                and not transformed_features.reasoning
                and not primary_scoped
            )
            unscoped_reasoning_enhancement = (
                transformed_features.reasoning
                and not plan.features.reasoning
                and candidate.model not in candidate_requests
            )
            if reasoning_drift or unscoped_reasoning_enhancement:
                raise RequestOptionError(
                    "Planned Chat features do not match the current request",
                    provider=candidate.model.provider,
                    model=candidate.model.upstream_id,
                    parameter=plan.features.reasoning_parameter or "reasoning",
                )

        def validate_candidate(candidate: RouteCandidate) -> None:
            candidate_request = request_for(candidate)
            validate_client_features(candidate, candidate_request)
            self._validate_chat_candidate(
                candidate_request,
                candidate,
                stream=candidate_request.stream,
            )

        validated = self.prevalidate_plan(
            plan,
            validate_candidate,
            fallback=fallback,
        )
        if request.stream:
            return PreparedChatStream.capture(validated, request, candidate_requests)
        return PreparedChatCompletion.capture(validated, request, candidate_requests)

    async def prepare_chat_completion(
        self,
        request: ChatRequest,
        *,
        fallback: bool = True,
    ) -> PreparedChatCompletion:
        request = deepcopy(request)
        plan = await self.plan_chat_completion(request, stream=False)
        prepared = self.prepare_planned_chat_completion(plan, request, fallback=fallback)
        assert isinstance(prepared, PreparedChatCompletion)
        return prepared

    async def chat_completion_stream(
        self,
        request: ChatRequest,
        fallback: bool = True,
        *,
        prepared_plan: PreparedChatStream | None = None,
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
        request = deepcopy(request)
        prepared = prepared_plan or await self.prepare_chat_completion_stream(
            request, fallback=fallback
        )
        if not isinstance(prepared, PreparedChatStream) or not prepared.matches(request):
            raise ProviderError(
                "Prepared Chat stream does not match the current request",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )
        plan = prepared.plan
        execution_request = prepared.request_for_execution()
        logger.info("Routing stream request to %s", plan.primary.model.qualified_id)
        return await self._execute_chat_plan_stream(
            plan,
            execution_request,
            fallback,
            prepared.request_for_execution,
        )

    async def prepare_chat_completion_stream(
        self,
        request: ChatRequest,
        *,
        fallback: bool = True,
    ) -> PreparedChatStream:
        """Plan and validate a chat stream once, before the response commits to SSE."""
        request = deepcopy(request)
        plan = await self.plan_chat_completion(request, stream=True)
        prepared = self.prepare_planned_chat_completion(plan, request, fallback=fallback)
        assert isinstance(prepared, PreparedChatStream)
        return prepared

    @staticmethod
    def prevalidate_plan(
        plan: RoutePlan,
        validate: Callable[[RouteCandidate], None],
        *,
        fallback: bool = True,
    ) -> RoutePlan:
        """Validate the primary and remove only option-incompatible fallbacks."""
        Router._validate_plan_primary(plan)
        validate(plan.primary)
        if not fallback:
            return replace(
                plan,
                fallbacks=(),
                fallback_pool=(),
                max_fallback_attempts=0,
            )
        fallbacks: list[RouteCandidate] = []
        for candidate in plan.prevalidation_fallbacks:
            if len(fallbacks) >= plan.fallback_limit:
                break
            try:
                validate(candidate)
            except RequestOptionError:
                continue
            fallbacks.append(candidate)
        if tuple(fallbacks) == plan.fallbacks:
            return plan
        return replace(plan, fallbacks=tuple(fallbacks))

    _prevalidate_plan = prevalidate_plan

    async def validate_chat_request(self, request: ChatRequest, *, stream: bool) -> None:
        """Validate the primary planned adapter without starting upstream I/O."""
        request = deepcopy(request)
        operation = Operation.CHAT_STREAM if stream else Operation.CHAT
        plan = await self._plan_completion_route(
            request.model,
            operation,
            RequestFeatures.for_chat(request),
        )
        self._validate_plan_primary(plan)
        candidate = plan.primary
        self._validate_chat_candidate(request, candidate, stream=stream)

    def _create_responses_request_with_model(
        self, original_request: ResponsesRequest, model_id: str
    ) -> ResponsesRequest:
        """Create a new ResponsesRequest with a different model ID.

        Args:
            original_request: The original request
            model_id: The new model ID to use

        Returns:
            New ResponsesRequest with updated model
        """
        return ResponsesRequest(
            model=model_id,
            input=deepcopy(original_request.input),
            stream=original_request.stream,
            instructions=original_request.instructions,
            temperature=original_request.temperature,
            max_output_tokens=original_request.max_output_tokens,
            tools=deepcopy(original_request.tools),
            tool_choice=deepcopy(original_request.tool_choice),
            parallel_tool_calls=original_request.parallel_tool_calls,
            reasoning_effort=original_request.reasoning_effort,
            top_p=original_request.top_p,
            metadata=deepcopy(original_request.metadata),
            service_tier=original_request.service_tier,
            provider_extensions=deepcopy(original_request.provider_extensions),
        )

    def _chain_first_chunk(
        self,
        first_chunk: ChunkT | None,
        stream: AsyncIterator[ChunkT],
        *,
        selected_model: ModelRef | None = None,
    ) -> AsyncIterator[ChunkT]:
        """Yield a primed first chunk followed by the rest of the stream."""
        return _PrimedStream(first_chunk, stream, selected_model=selected_model)

    @staticmethod
    async def _wrap_stream_errors(
        stream: AsyncIterator[ChunkT],
        provider_name: str,
        format_message: Callable[[str], str],
    ) -> AsyncIterator[ChunkT]:
        """Log provider errors that occur after stream delivery has started."""
        try:
            async for chunk in stream:
                yield chunk
        except ProviderError as error:
            logger.warning(
                format_message("stream_provider_failed provider=%s kind=%s retryable=%s"),
                provider_name,
                error.kind.value,
                str(error.retryable).lower(),
            )
            raise
        finally:
            await close_async_iterator(stream)

    async def responses_completion(
        self,
        request: ResponsesRequest,
        fallback: bool = True,
    ) -> tuple[ResponsesResponse, str]:
        """Route a Responses API completion request.

        Args:
            request: Responses completion request
            fallback: Whether to try fallback providers on error

        Returns:
            Tuple of (response, provider_name)

        Raises:
            ProviderError: If model not found or all providers fail
        """
        request = deepcopy(request)
        plan = await self._plan_completion_route(
            request.model,
            Operation.RESPONSES,
            RequestFeatures.for_responses(request),
        )
        plan = self.prevalidate_plan(
            plan,
            lambda candidate: candidate.provider.validate_responses_request(
                self._create_responses_request_with_model(
                    request,
                    candidate.model.upstream_id,
                )
            ),
            fallback=fallback,
        )
        logger.info("Routing responses request to %s", plan.primary.model.qualified_id)
        result, used_provider = await self._execute_plan_nonstream(
            plan,
            request,
            fallback,
            self._create_responses_request_with_model,
            lambda provider, candidate_request: provider.responses_completion(candidate_request),
        )
        return result, used_provider

    async def responses_completion_stream(
        self,
        request: ResponsesRequest,
        fallback: bool = True,
        *,
        prepared_plan: PreparedResponsesStream | None = None,
    ) -> tuple[AsyncIterator[ResponsesStreamChunk], str]:
        """Route a streaming Responses API completion request.

        Args:
            request: Responses completion request
            fallback: Whether to try fallback providers on error

        Returns:
            Tuple of (stream iterator, provider_name)

        Raises:
            ProviderError: If model not found or all providers fail
        """
        request = deepcopy(request)
        prepared = prepared_plan or await self.prepare_responses_completion_stream(
            request, fallback=fallback
        )
        if not isinstance(prepared, PreparedResponsesStream) or not prepared.matches(request):
            raise ProviderError(
                "Prepared Responses stream does not match the current request",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )
        plan = prepared.plan
        execution_request = prepared.request_for_execution()
        logger.info("Routing responses stream request to %s", plan.primary.model.qualified_id)
        return await self._execute_plan_stream(
            plan,
            execution_request,
            fallback,
            self._create_responses_request_with_model,
            lambda provider, candidate_request: provider.responses_completion_stream(
                candidate_request
            ),
            "Responses",
        )

    async def prepare_responses_completion_stream(
        self,
        request: ResponsesRequest,
        *,
        fallback: bool = True,
    ) -> PreparedResponsesStream:
        """Plan and validate a Responses stream once, before committing to SSE."""
        request = deepcopy(request)
        plan = await self._plan_completion_route(
            request.model,
            Operation.RESPONSES_STREAM,
            RequestFeatures.for_responses(request),
        )
        plan = self.prevalidate_plan(
            plan,
            lambda candidate: candidate.provider.validate_responses_request(
                self._create_responses_request_with_model(
                    request,
                    candidate.model.upstream_id,
                )
            ),
            fallback=fallback,
        )
        return PreparedResponsesStream.capture(plan, request)

    async def validate_responses_request(self, request: ResponsesRequest) -> None:
        """Validate the primary planned Responses adapter without starting I/O."""
        request = deepcopy(request)
        operation = Operation.RESPONSES_STREAM if request.stream else Operation.RESPONSES
        plan = await self._plan_completion_route(
            request.model,
            operation,
            RequestFeatures.for_responses(request),
        )
        self._validate_plan_primary(plan)
        candidate = plan.primary
        candidate_request = self._create_responses_request_with_model(
            request,
            candidate.model.upstream_id,
        )
        candidate.provider.validate_responses_request(candidate_request)

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
        for key, entry in self._models_cache.items():
            _provider_name, model_info = entry
            # Only include canonical public keys, not convenience aliases.
            if self._is_qualified_cache_entry(key, entry):
                all_models[key] = model_info

        # Add prioritized models first
        for priority_key in priorities_config.priorities:
            if priority_key in all_models and priority_key not in seen:
                models.append(all_models[priority_key])
                seen.add(priority_key)

        # Add remaining models (sorted by provider, family, version, variant)
        remaining = [model for key, model in all_models.items() if key not in seen]
        models.extend(sort_models(remaining))

        logger.debug("Listed %d models", len(models))
        return deepcopy(models)

    def invalidate_cache(self) -> None:
        """Invalidate all caches to force refresh."""
        self._models_cache.clear()
        self._fuzzy_cache.clear()
        self._models_cache_ttl.clear()
        self._priorities_cache.clear()
        self._providers_ttl.clear()
        logger.debug("All caches invalidated")
