"""Copilot model-catalog parsing, caching, and freshness policy."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from contextvars import Context
from copy import deepcopy
from typing import Any, NoReturn

import httpx

from router_maestro.protocols import WireProtocol
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    ContextWindowOption,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
)
from router_maestro.routing.capabilities import Feature, Operation
from router_maestro.utils import get_logger
from router_maestro.utils.cache import TTLCache
from router_maestro.utils.context_window import calculate_context_budget
from router_maestro.utils.reasoning import VALID_EFFORTS

logger = get_logger("providers.copilot.catalog")

COPILOT_MODELS_PATH = "/models"
MODELS_CACHE_TTL = 300
_COPILOT_REASONING_EFFORT_SENTINELS = frozenset({"none"})
_COPILOT_CATALOG_REASONING_EFFORT_VALUES = frozenset(VALID_EFFORTS).union(
    _COPILOT_REASONING_EFFORT_SENTINELS
)
_CLIENT_UPGRADE_BILLING_WARNING = (
    "update your client to the latest version to see the new billing information"
)

# Cold-start fallback for Responses eligibility. The live catalog's
# ``supported_endpoints`` (``/responses``) is authoritative and used whenever it
# is available (see ``operation_capabilities`` below); this hardcoded set only
# applies before the catalog has been fetched. Confirmed by direct probing of
# api.githubcopilot.com/responses — anything else returns 400
# unsupported_api_for_model. Match by suffix after stripping optional
# ``provider/`` prefix.
RESPONSES_ELIGIBLE_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5-mini",
        "mai-code-1-flash-picker",
    }
)


def _bare_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def is_model_responses_eligible(model: str) -> bool:
    """Whether the upstream serves this model via /responses (cold-start fallback)."""
    return _bare_model(model) in RESPONSES_ELIGIBLE_MODELS


def normalize_supported_endpoints(model: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Preserve missing versus explicit Copilot endpoint catalog state."""
    if "supported_endpoints" not in model:
        return None
    supported_endpoints = model.get("supported_endpoints")
    if not isinstance(supported_endpoints, (list, tuple)):
        raise TypeError("Copilot model supported_endpoints must be a list or tuple")
    if not all(isinstance(endpoint, str) for endpoint in supported_endpoints):
        raise TypeError("Copilot model supported_endpoints entries must be strings")
    return tuple(supported_endpoints)


def transport_capabilities(
    supported_endpoints: tuple[str, ...] | None,
) -> dict[str, bool]:
    """Map an explicit Copilot endpoint contract to wire-protocol support."""
    if supported_endpoints is None:
        return {}
    endpoints = set(supported_endpoints)
    return {
        WireProtocol.ANTHROPIC_MESSAGES: any(
            endpoint.endswith("/messages") for endpoint in endpoints
        ),
        WireProtocol.OPENAI_CHAT: "/chat/completions" in endpoints,
        WireProtocol.OPENAI_RESPONSES: "/responses" in endpoints,
    }


def normalize_catalog_boolean(supports: Mapping[str, Any], key: str) -> bool | None:
    if key not in supports:
        return None
    value = supports[key]
    if not isinstance(value, bool):
        raise TypeError(f"Copilot model capability {key} must be a boolean")
    return value


def normalize_reasoning_effort_values(supports: Mapping[str, Any]) -> list[str] | None:
    if "reasoning_effort" not in supports:
        return None
    raw = supports["reasoning_effort"]
    if isinstance(raw, dict):
        if "values" not in raw:
            raise TypeError("Copilot reasoning_effort object must contain values")
        raw = raw["values"]
    if not isinstance(raw, (list, tuple)):
        raise TypeError("Copilot reasoning_effort must be a list, tuple, or values object")

    values: list[str] = []
    for item in raw:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict) and set(item) == {"value"} and isinstance(item["value"], str):
            value = item["value"]
        else:
            raise TypeError("Copilot reasoning_effort entries must be strings or value objects")
        if not value.strip() or value not in _COPILOT_CATALOG_REASONING_EFFORT_VALUES:
            raise ValueError("Copilot reasoning_effort entry must be a supported non-empty tier")
        if value not in values:
            values.append(value)
    return values


def normalize_catalog_limit(limits: Mapping[str, Any], key: str) -> int | None:
    if key not in limits:
        return None
    value = limits[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"Copilot model capability limit {key} must be a positive integer")
    return value


def _normalize_context_tier_limit(
    tier: Mapping[str, Any],
    *,
    tier_name: str,
) -> int | None:
    values: dict[str, int] = {}
    for key in ("max_prompt_tokens", "context_max"):
        if key not in tier:
            continue
        value = tier[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TypeError(f"Copilot model billing {tier_name}.{key} must be a positive integer")
        values[key] = value
    return values.get("max_prompt_tokens", values.get("context_max"))


def _normalize_context_tier_price(
    tier: Mapping[str, Any],
    *keys: str,
) -> int | float | None:
    selected: int | float | None = None
    for key in keys:
        if key not in tier:
            continue
        value = tier[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise TypeError(f"Copilot model billing {key} must be a non-negative number")
        if selected is None:
            selected = value
    return selected


def _context_tier_price_signature(
    tier: Mapping[str, Any],
) -> tuple[int | float | None, ...]:
    return (
        _normalize_context_tier_price(tier, "input_price"),
        _normalize_context_tier_price(tier, "output_price"),
        _normalize_context_tier_price(tier, "cache_read_price", "cache_price"),
        _normalize_context_tier_price(tier, "cache_write_price"),
    )


def _single_context_window_option(limit: int | None) -> tuple[ContextWindowOption, ...]:
    if limit is None:
        return ()
    return (
        ContextWindowOption(
            tier="default",
            max_prompt_tokens=limit,
            is_default=True,
        ),
    )


def normalize_context_window_options(
    model: Mapping[str, Any],
    limits: Mapping[str, Any],
) -> tuple[ContextWindowOption, ...]:
    """Normalize Copilot default and long-context pricing tiers for clients."""
    max_prompt_tokens = normalize_catalog_limit(limits, "max_prompt_tokens")
    max_output_tokens = normalize_catalog_limit(limits, "max_output_tokens")
    max_context_window_tokens = normalize_catalog_limit(
        limits,
        "max_context_window_tokens",
    )
    fallback_limit = max_prompt_tokens
    if fallback_limit is None and max_context_window_tokens is not None:
        budget = calculate_context_budget(
            max_context_window_tokens,
            max_output_tokens,
            max_context_window_tokens,
        )
        fallback_limit = budget.max_prompt_tokens if budget is not None else None

    billing = model.get("billing")
    if billing is None:
        return _single_context_window_option(fallback_limit)
    if not isinstance(billing, Mapping):
        raise TypeError("Copilot model billing must be an object")
    token_prices = billing.get("token_prices")
    if token_prices is None:
        return _single_context_window_option(fallback_limit)
    if not isinstance(token_prices, Mapping):
        raise TypeError("Copilot model billing token_prices must be an object")

    default_tier = token_prices.get("default")
    if default_tier is None:
        return _single_context_window_option(fallback_limit)
    if not isinstance(default_tier, Mapping):
        raise TypeError("Copilot model billing default tier must be an object")
    default_limit = _normalize_context_tier_limit(
        default_tier,
        tier_name="default",
    )
    if default_limit is None:
        return _single_context_window_option(fallback_limit)

    long_tier = token_prices.get("long_context")
    if long_tier is not None and not isinstance(long_tier, Mapping):
        raise TypeError("Copilot model billing long_context tier must be an object")
    long_limit = (
        _normalize_context_tier_limit(long_tier, tier_name="long_context")
        if isinstance(long_tier, Mapping)
        else None
    )

    full_limit_candidates = [default_limit]
    if max_context_window_tokens is not None:
        full_budget = calculate_context_budget(
            max_context_window_tokens,
            max_output_tokens,
            max_context_window_tokens,
        )
        if full_budget is not None:
            full_limit_candidates.append(full_budget.max_prompt_tokens)
    else:
        if max_prompt_tokens is not None:
            full_limit_candidates.append(max_prompt_tokens)
        if long_limit is not None:
            full_limit_candidates.append(long_limit)
    full_limit = max(full_limit_candidates)
    if default_limit >= full_limit:
        return _single_context_window_option(full_limit)

    has_surcharge = False
    if isinstance(long_tier, Mapping):
        default_prices = _context_tier_price_signature(default_tier)
        long_prices = _context_tier_price_signature(long_tier)
        required_prices_known = all(
            value is not None
            for value in (
                default_prices[0],
                default_prices[1],
                long_prices[0],
                long_prices[1],
            )
        )
        has_surcharge = not required_prices_known or default_prices != long_prices

    return (
        ContextWindowOption(
            tier="default",
            max_prompt_tokens=default_limit,
            is_default=has_surcharge,
        ),
        ContextWindowOption(
            tier="long_context",
            max_prompt_tokens=full_limit,
            is_default=not has_surcharge,
        ),
    )


def _catalog_requires_current_client_token(data: object) -> bool:
    if not isinstance(data, Mapping):
        return False
    models = data.get("data")
    if not isinstance(models, list):
        return False
    for model in models:
        if not isinstance(model, Mapping):
            continue
        warning = model.get("warning_message")
        if isinstance(warning, str) and _CLIENT_UPGRADE_BILLING_WARNING in warning.lower():
            return True
    return False


def operation_capabilities(
    model: Mapping[str, Any],
    *,
    normalize_endpoints: Callable[[Mapping[str, Any]], tuple[str, ...] | None] = (
        normalize_supported_endpoints
    ),
) -> dict[str, bool]:
    """Derive operation support, preferring the live catalog endpoint contract."""
    model_id = str(model.get("id", ""))
    bare_model_id = model_id.split("/", 1)[1] if "/" in model_id else model_id
    supported_endpoints = normalize_endpoints(model)
    if supported_endpoints is not None:
        transports = transport_capabilities(supported_endpoints)
        chat = transports[WireProtocol.OPENAI_CHAT]
        responses = transports[WireProtocol.OPENAI_RESPONSES]
        return {
            Operation.CHAT: chat,
            Operation.CHAT_STREAM: chat,
            Operation.RESPONSES: responses,
            Operation.RESPONSES_STREAM: responses,
            Operation.NATIVE_ANTHROPIC: transports[WireProtocol.ANTHROPIC_MESSAGES],
        }

    operations: dict[str, bool] = {
        Operation.CHAT: True,
        Operation.CHAT_STREAM: True,
        Operation.NATIVE_ANTHROPIC: bare_model_id.lower().startswith("claude-"),
    }
    if is_model_responses_eligible(bare_model_id):
        operations[Operation.RESPONSES] = True
        operations[Operation.RESPONSES_STREAM] = True
    return operations


class CopilotCatalog:
    """Own Copilot catalog freshness, stale fallback, parsing, and lookups."""

    def __init__(self) -> None:
        self.models_ttl_cache: TTLCache[list[ModelInfo]] = TTLCache(MODELS_CACHE_TTL)
        self._refresh_lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[list[ModelInfo]] | None = None
        self._client_upgrade_retry_done = False
        self._closed = False

    def effort_values(self, model: str) -> list[str] | None:
        cached = self.models_ttl_cache.get()
        if not cached:
            return None
        bare = model.split("/", 1)[1] if "/" in model else model
        for info in cached:
            if info.id == bare or info.id == model:
                return deepcopy(info.reasoning_effort_values)
        return None

    @staticmethod
    def parse_models(
        data: object,
        *,
        provider_name: str,
        normalize_endpoints: Callable[[Mapping[str, Any]], tuple[str, ...] | None],
        derive_operations: Callable[[Mapping[str, Any]], dict[str, bool]],
    ) -> list[ModelInfo]:
        if not isinstance(data, dict):
            raise TypeError("Copilot model catalog must be an object")
        if "data" not in data or not isinstance(data["data"], list):
            raise TypeError("Copilot model catalog data must be a list")

        models: list[ModelInfo] = []
        for model in data["data"]:
            if not isinstance(model, dict):
                raise TypeError("Copilot model catalog entry must be an object")
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
                raise TypeError("Copilot model catalog id must be a non-empty, unpadded string")

            if "name" in model:
                name = model["name"]
                if not isinstance(name, str) or not name.strip():
                    raise TypeError("Copilot model catalog name must be a non-empty string")
                name = name.strip()
            else:
                name = model_id

            model_picker_enabled = model.get("model_picker_enabled", True)
            if not isinstance(model_picker_enabled, bool):
                raise TypeError("Copilot model model_picker_enabled must be a boolean")
            caps = model.get("capabilities", {})
            if not isinstance(caps, dict):
                raise TypeError("Copilot model capabilities must be an object")
            capability_type = caps.get("type")
            if "type" in caps and (
                not isinstance(capability_type, str) or not capability_type.strip()
            ):
                raise TypeError("Copilot model capability type must be a non-empty string")
            limits = caps.get("limits", {})
            if not isinstance(limits, dict):
                raise TypeError("Copilot model capability limits must be an object")
            supports = caps.get("supports", {})
            if not isinstance(supports, dict):
                raise TypeError("Copilot model capability supports must be an object")

            if not model_picker_enabled:
                continue
            if capability_type == "completion":
                logger.debug(
                    "Skipping Copilot completion-only model without RM route: %s",
                    model_id,
                )
                continue
            supported_endpoints = normalize_endpoints(model)
            reasoning_values = normalize_reasoning_effort_values(supports)
            tools_support = normalize_catalog_boolean(supports, "tool_calls")
            vision_support = normalize_catalog_boolean(supports, "vision")
            thinking_support = normalize_catalog_boolean(supports, "thinking")
            parallel_tools_support = normalize_catalog_boolean(supports, "parallel_tool_calls")
            models.append(
                ModelInfo(
                    id=model_id,
                    name=name,
                    provider=provider_name,
                    max_prompt_tokens=normalize_catalog_limit(limits, "max_prompt_tokens"),
                    max_output_tokens=normalize_catalog_limit(limits, "max_output_tokens"),
                    max_context_window_tokens=normalize_catalog_limit(
                        limits, "max_context_window_tokens"
                    ),
                    context_window_options=normalize_context_window_options(model, limits),
                    supports_thinking=(thinking_support is True or bool(reasoning_values)),
                    supports_vision=vision_support is True,
                    reasoning_effort_values=reasoning_values,
                    supported_endpoints=supported_endpoints,
                    operation_capabilities=derive_operations(model),
                    feature_capabilities={
                        **({Feature.TOOLS: tools_support} if tools_support is not None else {}),
                        **({Feature.VISION: vision_support} if vision_support is not None else {}),
                        **(
                            {
                                Feature.REASONING: (
                                    thinking_support is True or bool(reasoning_values)
                                )
                            }
                            if thinking_support is not None or reasoning_values is not None
                            else {}
                        ),
                        **(
                            {Feature.PARALLEL_TOOLS: parallel_tools_support}
                            if parallel_tools_support is not None
                            else {}
                        ),
                    },
                    transport_capabilities=transport_capabilities(supported_endpoints),
                )
            )
        return models

    async def list_models(
        self,
        force_refresh: bool = False,
        *,
        provider_name: str,
        ensure_token: Callable[..., Awaitable[None]],
        send: Callable[..., Awaitable[httpx.Response]],
        normalize_endpoints: Callable[[Mapping[str, Any]], tuple[str, ...] | None],
        derive_operations: Callable[[Mapping[str, Any]], dict[str, bool]],
        raise_protocol_error: Callable[..., NoReturn],
    ) -> list[ModelInfo]:
        cached = self.models_ttl_cache.get()
        if cached is not None and not force_refresh:
            logger.debug("Using cached Copilot models (%d models)", len(cached))
            return deepcopy(cached)

        stale = self.models_ttl_cache.peek()
        task = await self._get_or_start_refresh(
            detached=stale is not None and not force_refresh,
            provider_name=provider_name,
            ensure_token=ensure_token,
            send=send,
            normalize_endpoints=normalize_endpoints,
            derive_operations=derive_operations,
            raise_protocol_error=raise_protocol_error,
        )
        if stale is not None and not force_refresh:
            logger.debug("Serving stale Copilot models while refreshing")
            return deepcopy(stale)
        return deepcopy(await asyncio.shield(task))

    async def _get_or_start_refresh(
        self,
        *,
        detached: bool,
        provider_name: str,
        ensure_token: Callable[..., Awaitable[None]],
        send: Callable[..., Awaitable[httpx.Response]],
        normalize_endpoints: Callable[[Mapping[str, Any]], tuple[str, ...] | None],
        derive_operations: Callable[[Mapping[str, Any]], dict[str, bool]],
        raise_protocol_error: Callable[..., NoReturn],
    ) -> asyncio.Task[list[ModelInfo]]:
        async with self._refresh_lock:
            if self._closed:
                raise RuntimeError("Copilot catalog is closed")
            if self._refresh_task is not None and not self._refresh_task.done():
                return self._refresh_task
            refresh = self._refresh(
                provider_name=provider_name,
                ensure_token=ensure_token,
                send=send,
                normalize_endpoints=normalize_endpoints,
                derive_operations=derive_operations,
                raise_protocol_error=raise_protocol_error,
            )
            task = (
                asyncio.create_task(refresh, context=Context())
                if detached
                else asyncio.create_task(refresh)
            )
            task.add_done_callback(self._observe_refresh_result)
            self._refresh_task = task
            return task

    @staticmethod
    def _observe_refresh_result(task: asyncio.Task[list[ModelInfo]]) -> None:
        """Retrieve detached refresh failures even when stale callers do not await them."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning("Background Copilot catalog refresh failed", exc_info=error)

    async def aclose(self) -> None:
        """Cancel and join the detached refresh before its transport is closed."""
        async with self._refresh_lock:
            self._closed = True
            task = self._refresh_task
            self._refresh_task = None
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _refresh(
        self,
        *,
        provider_name: str,
        ensure_token: Callable[..., Awaitable[None]],
        send: Callable[..., Awaitable[httpx.Response]],
        normalize_endpoints: Callable[[Mapping[str, Any]], tuple[str, ...] | None],
        derive_operations: Callable[[Mapping[str, Any]], dict[str, bool]],
        raise_protocol_error: Callable[..., NoReturn],
    ) -> list[ModelInfo]:
        logger.debug("Fetching Copilot models from API")
        try:
            await ensure_token()
            async with httpx.AsyncClient(timeout=TIMEOUT_NON_STREAMING) as client:

                async def fetch_catalog() -> object:
                    response = await send(
                        "GET",
                        COPILOT_MODELS_PATH,
                        client=client,
                        headers_kwargs={"intent": "model-access"},
                        model=None,
                    )
                    response.raise_for_status()
                    return response.json()

                try:
                    data = await fetch_catalog()
                    if (
                        not self._client_upgrade_retry_done
                        and _catalog_requires_current_client_token(data)
                    ):
                        self._client_upgrade_retry_done = True
                        logger.info("Refreshing Copilot token minted by an older client identity")
                        await ensure_token(force=True)
                        data = await fetch_catalog()
                    models = self.parse_models(
                        data,
                        provider_name=provider_name,
                        normalize_endpoints=normalize_endpoints,
                        derive_operations=derive_operations,
                    )
                except (TypeError, ValueError) as error:
                    raise_protocol_error(provider_name, None, error)
            self.models_ttl_cache.set(deepcopy(models))
            logger.info("Fetched %d Copilot models", len(models))
            return deepcopy(models)
        except (httpx.HTTPError, ProviderError) as error:
            stale = self.models_ttl_cache.peek()
            if stale is not None:
                logger.warning(
                    "Failed to refresh Copilot models, using stale cache (%s)",
                    type(error).__name__,
                )
                self.models_ttl_cache.set(deepcopy(stale))
                return deepcopy(stale)
            logger.error("Failed to list Copilot models (%s)", type(error).__name__)
            if isinstance(error, ProviderError):
                raise
            raise ProviderError(
                "Failed to list Copilot models",
                status_code=502,
                retryable=True,
                kind=ProviderFailureKind.TRANSPORT,
                provider=provider_name,
                cause=error,
            ) from error
