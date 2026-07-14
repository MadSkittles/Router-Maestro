"""Copilot model-catalog parsing, caching, and freshness policy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from typing import Any, NoReturn

import httpx

from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
)
from router_maestro.routing.capabilities import Feature, Operation
from router_maestro.utils import get_logger
from router_maestro.utils.cache import TTLCache
from router_maestro.utils.reasoning import VALID_EFFORTS
from router_maestro.utils.responses_bridge import is_model_responses_eligible

logger = get_logger("providers.copilot.catalog")

COPILOT_MODELS_PATH = "/models"
MODELS_CACHE_TTL = 300
_COPILOT_REASONING_EFFORT_SENTINELS = frozenset({"none"})
_COPILOT_CATALOG_REASONING_EFFORT_VALUES = frozenset(VALID_EFFORTS).union(
    _COPILOT_REASONING_EFFORT_SENTINELS
)


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
        endpoints = set(supported_endpoints)
        chat = "/chat/completions" in endpoints
        responses = "/responses" in endpoints
        return {
            Operation.CHAT: chat,
            Operation.CHAT_STREAM: chat,
            Operation.RESPONSES: responses,
            Operation.RESPONSES_STREAM: responses,
            Operation.NATIVE_ANTHROPIC: any(
                endpoint.endswith("/messages") for endpoint in endpoints
            ),
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

    def effort_values(self, model: str) -> list[str] | None:
        cached = self.models_ttl_cache.get()
        if not cached:
            return None
        bare = model.split("/", 1)[1] if "/" in model else model
        for info in cached:
            if info.id == bare or info.id == model:
                return info.reasoning_effort_values
        return None

    def operation_support(self, model: str, operation: Operation) -> bool | None:
        cached = self.models_ttl_cache.get()
        if not cached:
            return None
        bare = model.split("/", 1)[1] if "/" in model else model
        for info in cached:
            if info.id == bare or info.id == model:
                return info.operation_capabilities.get(operation.value)
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
                    supports_thinking=thinking_support is True,
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
                )
            )
        return models

    async def list_models(
        self,
        force_refresh: bool = False,
        *,
        provider_name: str,
        ensure_token: Callable[[], Awaitable[None]],
        send: Callable[..., Awaitable[httpx.Response]],
        normalize_endpoints: Callable[[Mapping[str, Any]], tuple[str, ...] | None],
        derive_operations: Callable[[Mapping[str, Any]], dict[str, bool]],
        raise_protocol_error: Callable[..., NoReturn],
    ) -> list[ModelInfo]:
        if not force_refresh:
            cached = self.models_ttl_cache.get()
            if cached is not None:
                logger.debug("Using cached Copilot models (%d models)", len(cached))
                return deepcopy(cached)

        logger.debug("Fetching Copilot models from API")
        try:
            await ensure_token()
            async with httpx.AsyncClient(timeout=TIMEOUT_NON_STREAMING) as client:
                response = await send("GET", COPILOT_MODELS_PATH, client=client, model=None)
                response.raise_for_status()
                try:
                    models = self.parse_models(
                        response.json(),
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
            stale = self.models_ttl_cache._value
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
