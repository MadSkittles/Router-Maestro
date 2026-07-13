"""Typed provider failure and malformed upstream protocol regressions."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.auth.storage import OAuthCredential
from router_maestro.providers import (
    AnthropicProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    CopilotProvider,
    Message,
    ModelInfo,
    OpenAICompatibleProvider,
    OpenAIProvider,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponseStatus,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
)
from router_maestro.providers.base import BaseProvider, ProviderError, ProviderFailureKind
from router_maestro.providers.tool_parsing import recover_tool_calls_from_content
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    Feature,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import NoCompatibleRouteError, RouteCandidate, RoutePlan
from router_maestro.routing.router import Router
from router_maestro.server.routes.anthropic import router as anthropic_router
from router_maestro.utils import get_logger
from router_maestro.utils.responses_bridge import responses_response_to_chat_response

RAW_MARKER = "private-upstream-marker"


def _chat_request(model: str = "model-a", **kwargs) -> ChatRequest:
    return ChatRequest(model=model, messages=[Message(role="user", content="hi")], **kwargs)


def _response_for_payload(payload: object) -> httpx.Response:
    request = httpx.Request("POST", "https://upstream.invalid/completions")
    if isinstance(payload, bytes):
        return httpx.Response(200, content=payload, request=request)
    return httpx.Response(200, json=payload, request=request)


def _assert_protocol_failure(error: ProviderError, *, provider: str, model: str | None) -> None:
    assert error.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert error.status_code == 502
    assert error.upstream_status_code == 200
    assert error.retryable is True
    assert error.provider == provider
    assert error.model == model
    assert RAW_MARKER not in str(error)
    assert RAW_MARKER not in error.safe_message
    assert error.cause is not None


def _openai_catalog_provider(
    provider_kind: str,
) -> tuple[OpenAIProvider | OpenAICompatibleProvider, str]:
    if provider_kind == "openai":
        provider = OpenAIProvider(base_url="https://upstream.invalid")
        provider._get_headers = lambda: {}  # type: ignore[method-assign]
        return provider, "router_maestro.providers.openai.httpx.AsyncClient"
    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    return provider, "router_maestro.providers.openai_compat.httpx.AsyncClient"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"<html>" + RAW_MARKER.encode(), id="invalid-json"),
        pytest.param([], id="top-level-list"),
        pytest.param({}, id="missing-data"),
        pytest.param({"data": {}}, id="non-list-data"),
        pytest.param({"data": [RAW_MARKER]}, id="non-object-model"),
        pytest.param({"data": [{}]}, id="missing-model-id"),
        pytest.param(
            {"data": [{"id": "model", "capabilities": []}]},
            id="non-object-capabilities",
        ),
        pytest.param(
            {"data": [{"id": "model", "supported_endpoints": None}]},
            id="null-supported-endpoints",
        ),
        pytest.param(
            {"data": [{"id": "model", "supported_endpoints": "/responses"}]},
            id="scalar-supported-endpoints",
        ),
        pytest.param(
            {"data": [{"id": "model", "supported_endpoints": {}}]},
            id="object-supported-endpoints",
        ),
        pytest.param(
            {"data": [{"id": "model", "supported_endpoints": ["/responses", 7]}]},
            id="non-string-supported-endpoint",
        ),
        pytest.param(
            {
                "data": [
                    {
                        "id": "model",
                        "capabilities": {"supports": {"tool_calls": "false"}},
                    }
                ]
            },
            id="non-boolean-feature-capability",
        ),
        pytest.param(
            {
                "data": [
                    {
                        "id": "model",
                        "capabilities": {
                            "supports": {"reasoning_effort": ["low", 7]},
                        },
                    }
                ]
            },
            id="malformed-reasoning-effort-list",
        ),
    ],
)
async def test_copilot_model_catalog_malformed_200_is_typed_protocol_failure(payload) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    response = _response_for_payload(payload)
    provider._send_with_auth_retry = AsyncMock(return_value=response)  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        await provider.list_models(force_refresh=True)

    error = exc_info.value
    assert error.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert error.status_code == 502
    assert error.upstream_status_code == 200
    assert error.retryable is True
    assert error.provider == "github-copilot"
    assert error.model is None
    assert RAW_MARKER not in error.safe_message
    assert error.cause is not None


@pytest.mark.asyncio
async def test_copilot_model_catalog_rejects_padded_id_as_typed_protocol_failure() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload({"data": [{"id": " padded-model "}]})
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.list_models(force_refresh=True)

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(None, id="null"),
        pytest.param(7, id="number"),
        pytest.param([], id="list"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
    ],
)
async def test_copilot_model_catalog_rejects_malformed_present_name(name: object) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload({"data": [{"id": "model", "name": name}]})
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.list_models(force_refresh=True)

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_picker_enabled",
    [
        pytest.param(None, id="null"),
        pytest.param("false", id="string-false"),
        pytest.param(0, id="integer-zero"),
        pytest.param(1, id="integer-one"),
        pytest.param([], id="list"),
    ],
)
async def test_copilot_model_catalog_rejects_non_boolean_picker_flag(
    model_picker_enabled: object,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"data": [{"id": "model", "model_picker_enabled": model_picker_enabled}]}
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.list_models(force_refresh=True)

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capability_type",
    [
        pytest.param(None, id="null"),
        pytest.param(7, id="number"),
        pytest.param([], id="list"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
    ],
)
async def test_copilot_model_catalog_rejects_malformed_present_capability_type(
    capability_type: object,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"data": [{"id": "model", "capabilities": {"type": capability_type}}]}
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.list_models(force_refresh=True)

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits",
    [
        pytest.param(None, id="null"),
        pytest.param([], id="list"),
        pytest.param("limits", id="string"),
    ],
)
async def test_copilot_model_catalog_rejects_non_object_limits(limits: object) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"data": [{"id": "model", "capabilities": {"limits": limits}}]}
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.list_models(force_refresh=True)

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    ["max_prompt_tokens", "max_output_tokens", "max_context_window_tokens"],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="null"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
        pytest.param("1", id="string"),
    ],
)
async def test_copilot_model_catalog_rejects_invalid_present_limit(
    field: str,
    value: object,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "data": [
                    {
                        "id": "model",
                        "capabilities": {"limits": {field: value}},
                    }
                ]
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.list_models(force_refresh=True)

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("", id="empty-string"),
        pytest.param("   ", id="whitespace-string"),
        pytest.param(" low ", id="padded-valid-string"),
        pytest.param("ultra", id="unknown-string"),
        pytest.param({"value": ""}, id="object-empty-string"),
        pytest.param({"value": "   "}, id="object-whitespace-string"),
        pytest.param({"value": " low "}, id="object-padded-valid-string"),
        pytest.param({"value": "ultra"}, id="object-unknown-string"),
        pytest.param({"value": "low", "label": "Low"}, id="object-extra-field"),
    ],
)
async def test_copilot_model_catalog_rejects_invalid_reasoning_effort_entry(
    entry: object,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "data": [
                    {
                        "id": "model",
                        "capabilities": {"supports": {"reasoning_effort": [entry]}},
                    }
                ]
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.list_models(force_refresh=True)

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model=None)


@pytest.mark.asyncio
async def test_copilot_model_catalog_defaults_missing_optional_fields() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload({"data": [{"id": "model"}]})
    )

    models = await provider.list_models(force_refresh=True)

    assert models[0].name == "model"
    assert models[0].max_prompt_tokens is None
    assert models[0].max_output_tokens is None
    assert models[0].max_context_window_tokens is None


@pytest.mark.asyncio
@pytest.mark.parametrize("capability_type", ["chat", "future-model-kind", " completion "])
async def test_copilot_model_catalog_accepts_forward_compatible_capability_type(
    capability_type: str,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"data": [{"id": "model", "capabilities": {"type": capability_type}}]}
        )
    )

    models = await provider.list_models(force_refresh=True)

    assert [model.id for model in models] == ["model"]


@pytest.mark.asyncio
async def test_copilot_model_catalog_materializes_valid_full_entry() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "data": [
                    {
                        "id": "model",
                        "name": "  Model Name  ",
                        "model_picker_enabled": True,
                        "capabilities": {
                            "type": "chat",
                            "limits": {
                                "max_prompt_tokens": 100,
                                "max_output_tokens": 200,
                                "max_context_window_tokens": 300,
                            },
                            "supports": {
                                "thinking": True,
                                "vision": False,
                                "reasoning_effort": [
                                    "none",
                                    {"value": "none"},
                                    "minimal",
                                    {"value": "minimal"},
                                    "low",
                                    {"value": "high"},
                                    "low",
                                    {"value": "max"},
                                ],
                            },
                        },
                    }
                ]
            }
        )
    )

    models = await provider.list_models(force_refresh=True)

    assert len(models) == 1
    model = models[0]
    assert model.id == "model"
    assert model.name == "Model Name"
    assert model.max_prompt_tokens == 100
    assert model.max_output_tokens == 200
    assert model.max_context_window_tokens == 300
    assert model.supports_thinking is True
    assert model.supports_vision is False
    assert model.reasoning_effort_values == ["none", "minimal", "low", "high", "max"]
    assert model.feature_capabilities[Feature.REASONING] is True


@pytest.mark.asyncio
async def test_copilot_model_catalog_none_only_is_known_reasoning_capability() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "data": [
                    {
                        "id": "model",
                        "capabilities": {
                            "supports": {
                                "reasoning_effort": {
                                    "values": ["none", {"value": "none"}],
                                }
                            }
                        },
                    }
                ]
            }
        )
    )

    models = await provider.list_models(force_refresh=True)

    assert models[0].reasoning_effort_values == ["none"]
    assert models[0].feature_capabilities[Feature.REASONING] is True


@pytest.mark.asyncio
async def test_copilot_model_catalog_malformed_200_serves_and_renews_stale_snapshot() -> None:
    provider = CopilotProvider()
    stale = [ModelInfo(id="known", name="Known", provider="github-copilot")]
    provider._models_ttl_cache.set(stale)
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(b"<html>" + RAW_MARKER.encode())
    )

    models = await provider.list_models(force_refresh=True)

    cached = provider._models_ttl_cache.get()
    assert models == stale
    assert cached == stale
    assert models is not stale
    assert cached is not stale
    assert models is not cached


@pytest.mark.asyncio
async def test_copilot_padded_model_id_serves_and_renews_defensive_stale_snapshot() -> None:
    provider = CopilotProvider()
    stale = [ModelInfo(id="known", name="Known", provider="github-copilot")]
    provider._models_ttl_cache.set(stale)
    provider._models_ttl_cache._timestamp = 0
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload({"data": [{"id": " padded-model "}]})
    )

    models = await provider.list_models(force_refresh=True)

    cached = provider._models_ttl_cache.get()
    assert models == stale
    assert cached == stale
    assert models is not stale
    assert cached is not stale
    assert models is not cached
    assert models[0] is not stale[0]
    assert cached[0] is not stale[0]
    assert models[0] is not cached[0]
    assert provider._models_ttl_cache._timestamp > 0


@pytest.mark.asyncio
async def test_copilot_malformed_supported_endpoints_serves_and_renews_stale_snapshot() -> None:
    provider = CopilotProvider()
    stale = [ModelInfo(id="known", name="Known", provider="github-copilot")]
    provider._models_ttl_cache.set(stale)
    provider._models_ttl_cache._timestamp = 0
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"data": [{"id": "model", "supported_endpoints": ["/responses", 7]}]}
        )
    )

    models = await provider.list_models(force_refresh=True)

    cached = provider._models_ttl_cache.get()
    assert models == stale
    assert cached == stale
    assert models is not stale
    assert cached is not stale
    assert models is not cached
    assert provider._models_ttl_cache._timestamp > 0


@pytest.mark.asyncio
async def test_copilot_malformed_strict_catalog_field_serves_and_renews_stale_snapshot() -> None:
    provider = CopilotProvider()
    stale = [ModelInfo(id="known", name="Known", provider="github-copilot")]
    provider._models_ttl_cache.set(stale)
    provider._models_ttl_cache._timestamp = 0
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "data": [
                    {
                        "id": "model",
                        "capabilities": {"limits": {"max_output_tokens": 0}},
                    }
                ]
            }
        )
    )

    models = await provider.list_models(force_refresh=True)

    cached = provider._models_ttl_cache.get()
    assert models == stale
    assert cached == stale
    assert models is not stale
    assert cached is not stale
    assert models is not cached
    assert provider._models_ttl_cache._timestamp > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_kind", ["openai", "openai-compatible"])
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"<html>" + RAW_MARKER.encode(), id="invalid-json"),
        pytest.param([], id="top-level-list"),
        pytest.param({}, id="missing-data"),
        pytest.param({"data": {}}, id="non-list-data"),
        pytest.param({"data": [RAW_MARKER]}, id="non-object-model"),
        pytest.param({"data": [{}]}, id="missing-model-id"),
        pytest.param({"data": [{"id": 7}]}, id="non-string-model-id"),
        pytest.param({"data": [{"id": ""}]}, id="empty-model-id"),
        pytest.param({"data": [{"id": " padded-model "}]}, id="padded-model-id"),
    ],
)
async def test_openai_model_catalog_malformed_200_is_typed_protocol_failure(
    provider_kind: str,
    payload: object,
) -> None:
    provider, client_patch = _openai_catalog_provider(provider_kind)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response_for_payload(payload))
    )

    with patch(client_patch, return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.list_models()

    _assert_protocol_failure(exc_info.value, provider=provider.name, model=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_kind", ["openai", "openai-compatible"])
async def test_openai_model_catalog_valid_empty_data_remains_valid(provider_kind: str) -> None:
    provider, client_patch = _openai_catalog_provider(provider_kind)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response_for_payload({"data": []}))
    )

    with patch(client_patch, return_value=client):
        models = await provider.list_models()

    assert models == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_kind", "expected_ids"),
    [
        ("openai", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]),
        ("openai-compatible", []),
    ],
)
@pytest.mark.parametrize("failure", ["transport", "non-2xx"])
async def test_openai_model_catalog_transport_and_status_defaults_remain_unchanged(
    provider_kind: str,
    expected_ids: list[str],
    failure: str,
) -> None:
    provider, client_patch = _openai_catalog_provider(provider_kind)

    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "transport":
            raise httpx.ConnectError(RAW_MARKER, request=request)
        return httpx.Response(503, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch(client_patch, return_value=client):
        models = await provider.list_models()

    assert [model.id for model in models] == expected_ids


@pytest.mark.asyncio
async def test_router_isolates_malformed_model_catalog_and_lists_healthy_provider() -> None:
    malformed = OpenAICompatibleProvider(
        name="malformed-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    healthy = OpenAICompatibleProvider(
        name="healthy-provider",
        base_url="https://healthy.invalid",
        api_key="secret-token",
        models={"healthy-model": "Healthy Model"},
    )
    router = Router.__new__(Router)
    router.providers = {malformed.name: malformed, healthy.name: healthy}
    router._models_cache = {}
    router._fuzzy_cache = {}

    class _ModelsTTL:
        is_valid = False

        def set(self, _value: object) -> None:
            self.is_valid = True

    class _Priorities:
        priorities: tuple[str, ...] = ()

    router._models_cache_ttl = _ModelsTTL()
    router._ensure_providers_fresh = lambda: None  # type: ignore[method-assign]
    router._apply_model_overrides = lambda: None  # type: ignore[method-assign]
    router._get_priorities_config = lambda: _Priorities()  # type: ignore[method-assign]
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: _response_for_payload(b"<html>" + RAW_MARKER.encode())
        )
    )

    with patch("router_maestro.providers.openai_compat.httpx.AsyncClient", return_value=client):
        models = await router.list_models()

    assert [(model.provider, model.id) for model in models] == [
        ("healthy-provider", "healthy-model")
    ]


def test_provider_error_keeps_legacy_constructor_compatible() -> None:
    error = ProviderError("legacy failure", status_code=503, retryable=True)

    assert str(error) == "legacy failure"
    assert error.safe_message == "legacy failure"
    assert error.status_code == 503
    assert error.upstream_status_code is None
    assert error.retryable is True
    assert error.kind is ProviderFailureKind.UNKNOWN
    assert error.provider is None
    assert error.model is None
    assert error.cause is None


def test_provider_error_exposes_typed_safe_contract_without_leaking_cause() -> None:
    raw_marker = "secret-upstream-payload"
    cause = ValueError(raw_marker)
    error = ProviderError(
        "Upstream returned a malformed response",
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        status_code=502,
        upstream_status_code=200,
        retryable=True,
        provider="custom-provider",
        model="model-a",
        cause=cause,
    )

    assert str(error) == "Upstream returned a malformed response"
    assert error.safe_message == "Upstream returned a malformed response"
    assert raw_marker not in str(error)
    assert error.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert error.status_code == 502
    assert error.upstream_status_code == 200
    assert error.retryable is True
    assert error.provider == "custom-provider"
    assert error.model == "model-a"
    assert error.cause is cause


def test_provider_failure_kind_has_complete_first_party_vocabulary() -> None:
    assert set(ProviderFailureKind) == {
        ProviderFailureKind.TRANSPORT,
        ProviderFailureKind.AUTHENTICATION,
        ProviderFailureKind.RATE_LIMIT,
        ProviderFailureKind.UPSTREAM_STATUS,
        ProviderFailureKind.UPSTREAM_PROTOCOL,
        ProviderFailureKind.UNSUPPORTED_OPERATION,
        ProviderFailureKind.CLIENT_REQUEST,
        ProviderFailureKind.UNKNOWN,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"not-json-" + RAW_MARKER.encode(),
        ("<html>" + RAW_MARKER + "</html>").encode(),
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": RAW_MARKER}]},
    ],
)
async def test_openai_compatible_malformed_2xx_is_typed_protocol_failure(payload) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request())

    _assert_protocol_failure(exc_info.value, provider="custom-provider", model="model-a")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": None, "tool_calls": []}}]},
    ],
)
async def test_openai_compatible_empty_success_is_protocol_failure(payload) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request())

    _assert_protocol_failure(exc_info.value, provider="custom-provider", model="model-a")


@pytest.mark.asyncio
async def test_openai_compatible_tool_only_success_is_valid() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        response = await provider.chat_completion(_chat_request())

    assert response.tool_calls == payload["choices"][0]["message"]["tool_calls"]


@pytest.mark.asyncio
async def test_openai_compatible_malformed_tool_is_not_a_deliverable() -> None:
    payload = {"choices": [{"message": {"content": None, "tool_calls": [{}]}}]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request())

    _assert_protocol_failure(exc_info.value, provider="custom-provider", model="model-a")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("model", 7, id="model-non-string"),
        pytest.param("finish_reason", 7, id="finish-reason-non-string"),
        pytest.param("usage", RAW_MARKER, id="usage-scalar"),
        pytest.param("usage", [], id="usage-list"),
        pytest.param("usage", {"prompt_tokens": True}, id="prompt-tokens-bool"),
        pytest.param("usage", {"prompt_tokens": "1"}, id="prompt-tokens-non-integer"),
        pytest.param("usage", {"completion_tokens": True}, id="completion-tokens-bool"),
        pytest.param(
            "usage",
            {"completion_tokens": "1"},
            id="completion-tokens-non-integer",
        ),
        pytest.param("usage", {"total_tokens": True}, id="total-tokens-bool"),
        pytest.param("usage", {"total_tokens": "1"}, id="total-tokens-non-integer"),
    ],
)
async def test_openai_compatible_nonstream_consumed_metadata_is_typed(
    field: str,
    value: object,
) -> None:
    payload: dict[str, object] = {"choices": [{"message": {"content": "ok"}}]}
    if field == "finish_reason":
        payload["choices"][0][field] = value  # type: ignore[index]
    else:
        payload[field] = value

    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response_for_payload(payload))
    )
    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request())

    _assert_protocol_failure(exc_info.value, provider="custom-provider", model="model-a")


@pytest.mark.asyncio
async def test_openai_compatible_nonstream_ignores_unknown_optional_metadata() -> None:
    payload = {
        "choices": [
            {
                "message": {"content": "ok", "future_message_field": RAW_MARKER},
                "future_choice_field": RAW_MARKER,
            }
        ],
        "usage": {"future_token_field": RAW_MARKER},
        "future_response_field": RAW_MARKER,
    }

    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response_for_payload(payload))
    )
    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        response = await provider.chat_completion(_chat_request())

    assert response.content == "ok"


@pytest.mark.asyncio
async def test_openai_compatible_nonstream_preserves_refusal() -> None:
    payload = {
        "choices": [
            {
                "message": {"content": None, "refusal": "I cannot help"},
                "finish_reason": "stop",
            }
        ]
    }
    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response_for_payload(payload))
    )

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        response = await provider.chat_completion(_chat_request())

    assert response.content is None
    assert response.refusal == "I cannot help"


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["", 7, [], {}])
async def test_openai_compatible_nonstream_rejects_malformed_refusal(refusal) -> None:
    payload = {
        "choices": [
            {
                "message": {"content": "ordinary text", "refusal": refusal},
                "finish_reason": "stop",
            }
        ]
    }
    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response_for_payload(payload))
    )

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request())

    _assert_protocol_failure(exc_info.value, provider="custom-provider", model="model-a")


@pytest.mark.asyncio
async def test_router_does_not_fallback_after_valid_chat_refusal() -> None:
    primary = OpenAICompatibleProvider(
        name="primary",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )

    class _Secondary:
        name = "secondary"

        def __init__(self) -> None:
            self.calls = 0

        async def ensure_token(self) -> None:
            return None

        async def chat_completion(self, request: ChatRequest) -> ChatResponse:
            self.calls += 1
            return ChatResponse(content="unsafe fallback", model=request.model)

    secondary = _Secondary()
    primary_ref = ModelRef("primary", "model-a")
    secondary_ref = ModelRef("secondary", "model-b")
    features = RequestFeatures()
    candidates = (
        RouteCandidate(
            model=primary_ref,
            provider=primary,
            capabilities=ModelCapabilities(
                model=primary_ref,
                operations={Operation.CHAT: CapabilitySupport.SUPPORTED},
            ),
            evaluated_operation=Operation.CHAT,
            evaluated_features=features,
            support=CapabilitySupport.SUPPORTED,
        ),
        RouteCandidate(
            model=secondary_ref,
            provider=secondary,  # type: ignore[arg-type]
            capabilities=ModelCapabilities(
                model=secondary_ref,
                operations={Operation.CHAT: CapabilitySupport.SUPPORTED},
            ),
            evaluated_operation=Operation.CHAT,
            evaluated_features=features,
            support=CapabilitySupport.SUPPORTED,
        ),
    )
    plan = RoutePlan(Operation.CHAT, features, candidates[0], (candidates[1],), False)
    router = Router.__new__(Router)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: _response_for_payload(
                {
                    "choices": [
                        {
                            "message": {"content": None, "refusal": "I cannot help"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
    )

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        response, provider_name = await router._execute_plan_nonstream(
            plan,
            _chat_request(),
            True,
            lambda request, model: _chat_request(model=model),
            lambda provider, request: provider.chat_completion(request),
        )

    assert provider_name == "primary"
    assert response.refusal == "I cannot help"
    assert secondary.calls == 0


@pytest.mark.asyncio
async def test_router_falls_back_before_malformed_nonstream_metadata_is_serialized() -> None:
    primary = OpenAICompatibleProvider(
        name="primary",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )

    class _Secondary:
        name = "secondary"

        async def ensure_token(self) -> None:
            return None

        async def chat_completion(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                content="secondary",
                model=request.model,
                finish_reason="stop",
            )

    secondary = _Secondary()
    primary_ref = ModelRef("primary", "model-a")
    secondary_ref = ModelRef("secondary", "model-b")
    features = RequestFeatures()
    candidates = (
        RouteCandidate(
            model=primary_ref,
            provider=primary,
            capabilities=ModelCapabilities(
                model=primary_ref,
                operations={Operation.CHAT: CapabilitySupport.SUPPORTED},
            ),
            evaluated_operation=Operation.CHAT,
            evaluated_features=features,
            support=CapabilitySupport.SUPPORTED,
        ),
        RouteCandidate(
            model=secondary_ref,
            provider=secondary,  # type: ignore[arg-type]
            capabilities=ModelCapabilities(
                model=secondary_ref,
                operations={Operation.CHAT: CapabilitySupport.SUPPORTED},
            ),
            evaluated_operation=Operation.CHAT,
            evaluated_features=features,
            support=CapabilitySupport.SUPPORTED,
        ),
    )
    plan = RoutePlan(
        Operation.CHAT,
        features,
        candidates[0],
        (candidates[1],),
        False,
    )
    router = Router.__new__(Router)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: _response_for_payload(
                {
                    "choices": [{"message": {"content": "primary"}}],
                    "model": 7,
                }
            )
        )
    )

    with patch("router_maestro.providers.openai_base.httpx.AsyncClient", return_value=client):
        response, provider_name = await router._execute_plan_nonstream(
            plan,
            _chat_request(),
            True,
            lambda request, model: _chat_request(model=model),
            lambda provider, request: provider.chat_completion(request),
        )

    assert provider_name == "secondary"
    assert response.content == "secondary"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"not-json-" + RAW_MARKER.encode(),
        {},
        {"content": RAW_MARKER},
        {"content": [{}]},
        {"content": [{"type": "text"}]},
    ],
)
async def test_anthropic_malformed_2xx_is_typed_protocol_failure(payload) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request(model="claude-test"))

    _assert_protocol_failure(exc_info.value, provider="anthropic", model="claude-test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": ""}],
        [{"type": "future_block", "payload": RAW_MARKER}],
        [{"type": "thinking", "thinking": "valid but not returned by this adapter"}],
    ],
)
async def test_anthropic_empty_or_unhandled_deliverable_is_protocol_failure(content) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload({"content": content})

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request(model="claude-test"))

    _assert_protocol_failure(exc_info.value, provider="anthropic", model="claude-test")


@pytest.mark.asyncio
async def test_anthropic_unknown_block_is_ignored_when_text_deliverable_exists() -> None:
    payload = {
        "content": [
            {"type": "future_block", "payload": RAW_MARKER},
            {"type": "text", "text": "hello"},
        ]
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        response = await provider.chat_completion(_chat_request(model="claude-test"))

    assert response.content == "hello"


@pytest.mark.asyncio
async def test_anthropic_malformed_tool_is_not_a_deliverable() -> None:
    payload = {"content": [{"type": "tool_use", "id": "call-1", "input": {}}]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request(model="claude-test"))

    _assert_protocol_failure(exc_info.value, provider="anthropic", model="claude-test")


@pytest.mark.asyncio
async def test_anthropic_requested_thinking_only_is_preserved() -> None:
    payload = {
        "content": [
            {
                "type": "thinking",
                "thinking": "reasoning",
                "signature": "opaque-signature",
            }
        ]
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        response = await provider.chat_completion(
            _chat_request(
                model="claude-test",
                thinking_type="enabled",
                thinking_budget=1024,
            )
        )

    assert response.thinking == "reasoning"
    assert response.thinking_signature == "opaque-signature"


@pytest.mark.asyncio
@pytest.mark.parametrize("block_type", [None, 7, True])
async def test_anthropic_nonstream_block_type_must_be_string(block_type) -> None:
    payload = {"content": [{"type": block_type, "payload": RAW_MARKER}]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request(model="claude-test"))

    _assert_protocol_failure(exc_info.value, provider="anthropic", model="claude-test")


@pytest.mark.asyncio
async def test_anthropic_requested_redacted_thinking_only_is_preserved() -> None:
    payload = {"content": [{"type": "redacted_thinking", "data": "opaque-redacted"}]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        response = await provider.chat_completion(
            _chat_request(
                model="claude-test",
                thinking_type="enabled",
                thinking_budget=1024,
            )
        )

    assert response.content is None
    assert response.thinking is None
    assert response.thinking_signature == "opaque-redacted"


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 2, "output_tokens": 3}])
async def test_anthropic_nonstream_accepts_nullable_or_valid_usage(usage) -> None:
    payload = {"content": [{"type": "text", "text": "hello"}], "usage": usage}

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        response = await provider.chat_completion(_chat_request(model="claude-test"))

    expected_input = usage.get("input_tokens", 0) if usage else 0
    expected_output = usage.get("output_tokens", 0) if usage else 0
    assert response.usage == {
        "prompt_tokens": expected_input,
        "completion_tokens": expected_output,
        "total_tokens": expected_input + expected_output,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [RAW_MARKER, [], {"input_tokens": True}, {"output_tokens": "1"}],
)
async def test_anthropic_nonstream_usage_must_have_integer_token_counts(usage) -> None:
    payload = {"content": [{"type": "text", "text": "hello"}], "usage": usage}

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request(model="claude-test"))

    _assert_protocol_failure(exc_info.value, provider="anthropic", model="claude-test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("model", 7, id="model-non-string"),
        pytest.param("stop_reason", 7, id="stop-reason-non-string"),
    ],
)
async def test_anthropic_nonstream_consumed_metadata_is_typed(
    field: str,
    value: object,
) -> None:
    payload = {"content": [{"type": "text", "text": "hello"}], field: value}

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response_for_payload(payload))
    )
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(_chat_request(model="claude-test"))

    _assert_protocol_failure(exc_info.value, provider="anthropic", model="claude-test")


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [None, 7, [], {}])
async def test_anthropic_redacted_thinking_data_must_be_string(data) -> None:
    payload = {"content": [{"type": "redacted_thinking", "data": data}]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return _response_for_payload(payload)

    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as exc_info:
            await provider.chat_completion(
                _chat_request(
                    model="claude-test",
                    thinking_type="enabled",
                    thinking_budget=1024,
                )
            )

    _assert_protocol_failure(exc_info.value, provider="anthropic", model="claude-test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"not-json-" + RAW_MARKER.encode(),
        {},
        {"choices": []},
        {"choices": RAW_MARKER},
        {"choices": [{}]},
        {"choices": [{"message": RAW_MARKER}]},
    ],
)
async def test_copilot_chat_malformed_2xx_is_typed_protocol_failure(payload) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(payload)
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(_chat_request(model="gpt-4o"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-4o")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [{}, {"content": ""}, {"content": None, "tool_calls": []}],
)
async def test_copilot_chat_empty_success_is_protocol_failure(message) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload({"choices": [{"message": message}]})
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(_chat_request(model="gpt-4o"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-4o")


@pytest.mark.asyncio
async def test_copilot_chat_malformed_tool_is_not_a_deliverable() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"choices": [{"message": {"content": None, "tool_calls": [{}]}}]}
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(_chat_request(model="gpt-4o"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-4o")


@pytest.mark.asyncio
async def test_copilot_empty_choices_does_not_log_raw_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"choices": [], "private": RAW_MARKER, RAW_MARKER: "private"}
        )
    )

    with pytest.raises(ProviderError):
        await provider.chat_completion(_chat_request(model="gpt-4o"))

    assert RAW_MARKER not in caplog.text


@pytest.mark.asyncio
async def test_copilot_chat_unknown_claude_empty_at_cap_is_protocol_failure() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"completion_tokens": 4096},
                "model": "claude-sonnet-5",
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(
            _chat_request(
                model="claude-sonnet-5",
                max_tokens=4096,
                thinking_type="enabled",
                reasoning_effort="high",
            )
        )

    _assert_protocol_failure(
        exc_info.value,
        provider="github-copilot",
        model="claude-sonnet-5",
    )


@pytest.mark.asyncio
async def test_copilot_chat_empty_reasoning_at_output_cap_is_length_limited() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"completion_tokens": 5120},
                "model": "claude-sonnet-4.6",
            }
        )
    )

    response = await provider.chat_completion(
        _chat_request(
            model="claude-sonnet-4.6",
            max_tokens=5120,
            thinking_type="enabled",
            thinking_budget=4096,
        )
    )

    assert response.content == ""
    assert response.finish_reason == "length"


@pytest.mark.asyncio
async def test_empty_reasoning_at_explicit_budget_is_length_limited() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"completion_tokens": 16000},
                "model": "claude-sonnet-4.6",
            }
        )
    )

    response = await provider.chat_completion(
        _chat_request(
            model="claude-sonnet-4.6",
            max_tokens=16384,
            thinking_type="enabled",
            thinking_budget=16000,
        )
    )

    assert response.content == ""
    assert response.finish_reason == "length"


@pytest.mark.asyncio
async def test_empty_reasoning_at_ignored_budget_with_explicit_effort_is_protocol_failure() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"completion_tokens": 16000},
                "model": "claude-sonnet-4.6",
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(
            _chat_request(
                model="claude-sonnet-4.6",
                max_tokens=16384,
                thinking_type="enabled",
                thinking_budget=16000,
                reasoning_effort="high",
            )
        )

    _assert_protocol_failure(
        exc_info.value,
        provider="github-copilot",
        model="claude-sonnet-4.6",
    )


@pytest.mark.asyncio
async def test_copilot_chat_explicit_effort_uses_only_sent_output_cap_for_empty_reasoning() -> None:
    request = _chat_request(
        model="claude-opus-4.8",
        max_tokens=4096,
        thinking_type="enabled",
        thinking_budget=1024,
        reasoning_effort="xhigh",
    )
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"completion_tokens": 4096},
                "model": request.model,
            }
        )
    )

    response = await provider.chat_completion(request)

    sent_payload = provider._send_with_auth_retry.await_args.kwargs["json"]
    assert request.thinking_budget == 1024
    assert sent_payload["reasoning_effort"] == "high"
    assert "thinking_budget" not in sent_payload
    assert response.finish_reason == "length"


@pytest.mark.asyncio
async def test_copilot_chat_empty_reasoning_below_sent_output_cap_is_protocol_failure() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"completion_tokens": 4095},
                "model": "claude-opus-4.8",
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(
            _chat_request(
                model="claude-opus-4.8",
                max_tokens=4096,
                thinking_type="enabled",
                thinking_budget=1024,
            )
        )

    _assert_protocol_failure(
        exc_info.value,
        provider="github-copilot",
        model="claude-opus-4.8",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [None, 0, -1])
async def test_copilot_chat_empty_reasoning_without_positive_output_cap_is_protocol_failure(
    max_tokens: int | None,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"completion_tokens": 4096},
                "model": "claude-opus-4.8",
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(
            _chat_request(
                model="claude-opus-4.8",
                max_tokens=max_tokens,
                thinking_type="enabled",
                thinking_budget=1024,
            )
        )

    _assert_protocol_failure(
        exc_info.value,
        provider="github-copilot",
        model="claude-opus-4.8",
    )


@pytest.mark.asyncio
async def test_copilot_chat_empty_nonthinking_response_at_output_cap_is_protocol_failure() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"completion_tokens": 4096},
                "model": "claude-opus-4.8",
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(_chat_request(model="claude-opus-4.8", max_tokens=4096))

    _assert_protocol_failure(
        exc_info.value,
        provider="github-copilot",
        model="claude-opus-4.8",
    )


def test_anthropic_route_maps_saturated_empty_reasoning_to_max_tokens() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5120},
                "model": "claude-sonnet-4.6",
            }
        )
    )

    class CopilotBackedRouter:
        def __init__(self):
            self.plan = None

        async def plan_chat_completion(self, request: ChatRequest, *, stream: bool):
            operation = Operation.CHAT_STREAM if stream else Operation.CHAT
            features = RequestFeatures.for_chat(request)
            ref = ModelRef(provider.name, request.model)
            capabilities = ModelCapabilities(
                model=ref,
                operations={operation: CapabilitySupport.SUPPORTED},
                features={Feature.REASONING: CapabilitySupport.SUPPORTED},
                max_output_tokens=5120,
            )
            candidate = RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            )
            self.plan = RoutePlan(operation, features, candidate, (), True)
            return self.plan

        def prepare_planned_chat_completion(self, plan, request, *, candidate_requests=None):
            assert plan is self.plan
            assert candidate_requests == {plan.primary.model: request}
            return request

        async def chat_completion(self, request: ChatRequest, *, prepared_plan=None):
            assert prepared_plan is request
            return await provider.chat_completion(request), provider.name

    app = FastAPI()
    app.include_router(anthropic_router)
    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=CopilotBackedRouter(),
    ):
        response = TestClient(app).post(
            "/api/anthropic/v1/messages",
            json={
                "model": "claude-sonnet-4.6",
                "max_tokens": 5120,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "enabled", "budget_tokens": 4096},
            },
        )

    assert response.status_code == 200
    assert response.json()["content"] == []
    assert response.json()["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [RAW_MARKER, [], {"completion_tokens": True}, {"completion_tokens": "12"}],
)
async def test_copilot_chat_usage_must_have_integer_completion_tokens(usage) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"choices": [], "usage": usage, "model": "claude-sonnet-4.6"}
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(
            _chat_request(
                model="claude-sonnet-4.6",
                thinking_type="enabled",
                thinking_budget=1024,
            )
        )

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="claude-sonnet-4.6")


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, {}, {"completion_tokens": 12}])
async def test_copilot_chat_accepts_nullable_or_valid_usage(usage) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [{"message": {"content": "hello"}}],
                "usage": usage,
                "model": "gpt-4o",
            }
        )
    )

    response = await provider.chat_completion(_chat_request(model="gpt-4o"))

    assert response.content == "hello"
    assert response.usage == usage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("model", 7, id="model-non-string"),
        pytest.param("finish_reason", 7, id="finish-reason-non-string"),
    ],
)
async def test_copilot_chat_nonstream_consumed_metadata_is_typed(
    field: str,
    value: object,
) -> None:
    payload: dict[str, object] = {"choices": [{"message": {"content": "ok"}}]}
    if field == "finish_reason":
        payload["choices"][0][field] = value  # type: ignore[index]
    else:
        payload[field] = value

    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(payload)
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(_chat_request(model="gpt-4o"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-4o")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(RAW_MARKER, id="scalar"),
        pytest.param([], id="list"),
        pytest.param({"prompt_tokens": True}, id="prompt-tokens-bool"),
        pytest.param({"completion_tokens": "1"}, id="completion-tokens-non-integer"),
        pytest.param({"total_tokens": True}, id="total-tokens-bool"),
    ],
)
async def test_copilot_chat_valid_content_usage_must_have_integer_token_counts(usage) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"choices": [{"message": {"content": "ok"}}], "usage": usage}
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(_chat_request(model="gpt-4o"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-4o")


@pytest.mark.asyncio
async def test_copilot_chat_nonstream_preserves_refusal() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [
                    {
                        "message": {"content": None, "refusal": "I cannot help"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
    )

    response = await provider.chat_completion(_chat_request(model="gpt-4o"))

    assert response.content is None
    assert response.refusal == "I cannot help"


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["", 7, [], {}])
async def test_copilot_chat_nonstream_rejects_malformed_refusal(refusal) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "choices": [
                    {
                        "message": {"content": "ordinary text", "refusal": refusal},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat_completion(_chat_request(model="gpt-4o"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-4o")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"not-json-" + RAW_MARKER.encode(),
        {},
        {"status": "completed", "output": RAW_MARKER},
        {"status": RAW_MARKER, "output": []},
        {"status": "completed", "output": [{}]},
    ],
)
async def test_copilot_responses_malformed_2xx_is_typed_protocol_failure(payload) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(payload)
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-5")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        [],
        [{"type": "message", "content": []}],
        [{"type": "message", "content": [{"type": "output_text", "text": ""}]}],
        [{"type": "future_output", "payload": RAW_MARKER}],
    ],
)
async def test_copilot_completed_responses_requires_deliverable(output) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload({"status": "completed", "output": output})
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-5")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
async def test_copilot_business_terminal_may_have_empty_output(status: str) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    payload = {"status": status, "output": []}
    if status == "failed":
        payload["error"] = {"code": "failed", "message": "safe"}
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(payload)
    )

    response = await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    assert response.terminal_outcome is not None
    assert response.terminal_outcome.response_status.value == status


@pytest.mark.asyncio
async def test_copilot_completed_responses_rejects_malformed_tool_deliverable() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {"status": "completed", "output": [{"type": "function_call"}]}
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-5")


@pytest.mark.asyncio
async def test_copilot_completed_responses_accepts_reasoning_deliverable() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs-1",
                        "summary": [{"type": "summary_text", "text": "reasoning"}],
                        "encrypted_content": "opaque-signature",
                    }
                ],
            }
        )
    )

    response = await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    assert response.thinking == "reasoning"
    assert response.thinking_id == "rs-1"
    assert response.thinking_signature == "opaque-signature"


@pytest.mark.asyncio
async def test_copilot_nonstream_rejects_reasoning_identity_split_across_items() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs-first",
                        "summary": [{"type": "summary_text", "text": "first"}],
                    },
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "second"}],
                        "encrypted_content": "opaque-second-signature",
                    },
                ],
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-5")


@pytest.mark.asyncio
async def test_copilot_nonstream_rejects_encrypted_reasoning_without_upstream_id() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "reasoning"}],
                        "encrypted_content": "opaque-signature-without-id",
                    }
                ],
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-5")


@pytest.mark.asyncio
async def test_copilot_completed_responses_preserves_refusal_as_distinct_field() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "safe refusal"}],
                    }
                ],
            }
        )
    )

    response = await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    assert response.content == ""
    assert response.refusal == "safe refusal"


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", [None, "", 7, []])
async def test_copilot_nonstream_refusal_must_be_nonempty_string(refusal) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": refusal}],
                    }
                ],
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-5")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reasoning",
    [
        {"type": "reasoning", "summary": RAW_MARKER},
        {"type": "reasoning", "summary": [RAW_MARKER]},
        {"type": "reasoning", "summary": [{"type": 7, "text": "reasoning"}]},
        {"type": "reasoning", "summary": [{"text": 7}]},
        {"type": "reasoning", "id": 7, "summary": [{"text": "reasoning"}]},
        {
            "type": "reasoning",
            "encrypted_content": 7,
            "summary": [{"text": "reasoning"}],
        },
    ],
)
async def test_copilot_nonstream_reasoning_consumed_fields_are_typed(reasoning) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload({"status": "completed", "output": [reasoning]})
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-5")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(RAW_MARKER, id="scalar"),
        pytest.param([], id="list"),
        pytest.param({"input_tokens": True}, id="input-tokens-bool"),
        pytest.param({"output_tokens": "1"}, id="output-tokens-non-integer"),
        pytest.param({"total_tokens": True}, id="total-tokens-bool"),
        pytest.param(
            {"input_tokens_details": "bad"},
            id="input-token-details-non-object",
        ),
        pytest.param(
            {"output_tokens_details": []},
            id="output-token-details-non-object",
        ),
        pytest.param(
            {"input_tokens_details": {"cached_tokens": True}},
            id="cached-tokens-bool",
        ),
        pytest.param(
            {"input_tokens_details": {"cached_tokens": "1"}},
            id="cached-tokens-non-integer",
        ),
        pytest.param(
            {"output_tokens_details": {"reasoning_tokens": False}},
            id="reasoning-tokens-bool",
        ),
        pytest.param(
            {"output_tokens_details": {"reasoning_tokens": "1"}},
            id="reasoning-tokens-non-integer",
        ),
    ],
)
async def test_copilot_nonstream_responses_usage_must_have_integer_token_counts(
    usage,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=_response_for_payload(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": usage,
            }
        )
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    _assert_protocol_failure(exc_info.value, provider="github-copilot", model="gpt-5")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {
            "input_tokens": 2,
            "output_tokens": 3,
            "input_tokens_details": {"cached_tokens": 1, "future_count": "opaque"},
            "output_tokens_details": {"reasoning_tokens": 2, "future_count": []},
            "future_usage": "opaque",
        },
    ],
)
async def test_responses_stream_accepts_nullable_or_valid_terminal_usage(
    usage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    event = json.dumps(
        {"type": "response.completed", "response": {"status": "completed", "usage": usage}}
    )
    stream, _provider, _model = _stream_for_codec(
        "copilot-responses", _sse_body(event), monkeypatch
    )

    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 1
    assert chunks[0].usage == (usage or None)


def _sse_body(*events: str) -> bytes:
    return "".join(f"data: {event}\n\n" for event in events).encode()


async def _noop() -> None:
    return None


def _stream_for_codec(
    codec: str,
    body: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AsyncIterator[ChatStreamChunk | ResponsesStreamChunk], str, str]:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if codec == "openai-compatible":
        provider = OpenAICompatibleProvider(
            name="custom-provider",
            base_url="https://upstream.invalid",
            api_key="secret-token",
        )
        monkeypatch.setattr(
            "router_maestro.providers.openai_base.httpx.AsyncClient", lambda: client
        )
        return provider.chat_completion_stream(_chat_request(stream=True)), provider.name, "model-a"
    if codec == "anthropic":
        provider = AnthropicProvider()
        provider._get_headers = lambda: {}  # type: ignore[method-assign]
        monkeypatch.setattr("router_maestro.providers.anthropic.httpx.AsyncClient", lambda: client)
        model = "claude-test"
        stream = provider.chat_completion_stream(_chat_request(model=model, stream=True))
        return stream, provider.name, model
    if codec == "copilot-chat":
        provider = CopilotProvider()
        provider._client = client
        provider.ensure_token = _noop  # type: ignore[method-assign]
        provider._get_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        model = "gpt-4o"
        stream = provider.chat_completion_stream(_chat_request(model=model, stream=True))
        return stream, provider.name, model
    if codec == "copilot-responses":
        provider = CopilotProvider()
        provider._client = client
        provider.ensure_token = _noop  # type: ignore[method-assign]
        provider._get_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        model = "gpt-5"
        return (
            provider.responses_completion_stream(
                ResponsesRequest(model=model, input="hi", stream=True)
            ),
            provider.name,
            model,
        )
    raise AssertionError(f"unknown codec: {codec}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("codec", "ignored_event", "first_event"),
    [
        (
            "openai-compatible",
            '{"type":"future.event"}',
            '{"choices":[{"delta":{"content":"first"},"finish_reason":null}]}',
        ),
        (
            "anthropic",
            '{"type":"future_event"}',
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first"}}',
        ),
        (
            "copilot-chat",
            '{"type":"future.event"}',
            '{"choices":[{"delta":{"content":"first"},"finish_reason":null}]}',
        ),
        (
            "copilot-responses",
            '{"type":"response.future"}',
            '{"type":"response.output_text.delta","delta":"first"}',
        ),
    ],
)
@pytest.mark.parametrize("after_first", [False, True], ids=["pre-first", "post-first"])
async def test_sse_malformed_json_is_typed_before_or_after_first_canonical_chunk(
    codec: str,
    ignored_event: str,
    first_event: str,
    after_first: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [ignored_event]
    if after_first:
        events.append(first_event)
    events.append("not-json-" + RAW_MARKER)
    stream, provider, model = _stream_for_codec(codec, _sse_body(*events), monkeypatch)
    chunks: list[ChatStreamChunk | ResponsesStreamChunk] = []

    with pytest.raises(ProviderError) as exc_info:
        async for chunk in stream:
            chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == (["first"] if after_first else [])
    _assert_protocol_failure(exc_info.value, provider=provider, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("codec", "malformed_event"),
    [
        ("openai-compatible", '{"choices":["' + RAW_MARKER + '"]}'),
        (
            "openai-compatible",
            '{"choices":[{"delta":{"content":7}}]}',
        ),
        (
            "openai-compatible",
            '{"choices":[{"delta":{"tool_calls":"bad"}}]}',
        ),
        (
            "openai-compatible",
            '{"choices":[{"delta":{},"finish_reason":7}]}',
        ),
        ("openai-compatible", '{"choices":[],"usage":7}'),
        ("openai-compatible", '{"choices":[],"usage":{"prompt_tokens":true}}'),
        ("openai-compatible", '{"choices":[],"usage":{"completion_tokens":"1"}}'),
        ("openai-compatible", '{"choices":[],"usage":{"total_tokens":false}}'),
        (
            "openai-compatible",
            '{"choices":[{"delta":{"refusal":7}}]}',
        ),
        (
            "openai-compatible",
            '{"choices":[{"delta":{"tool_calls":[{"index":"0"}]}}]}',
        ),
        (
            "openai-compatible",
            '{"choices":[{"delta":{"tool_calls":[{"function":{"name":7}}]}}]}',
        ),
        (
            "anthropic",
            '{"type":"content_block_delta","delta":"' + RAW_MARKER + '"}',
        ),
        (
            "anthropic",
            '{"type":"message_start","message":"bad"}',
        ),
        (
            "anthropic",
            '{"type":"content_block_start","index":"0","content_block":{"type":"text"}}',
        ),
        (
            "anthropic",
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":7}}',
        ),
        (
            "anthropic",
            '{"type":"message_delta","delta":{"stop_reason":7},"usage":[]}',
        ),
        (
            "anthropic",
            '{"type":"message_start","message":{"usage":{"input_tokens":true}}}',
        ),
        (
            "anthropic",
            '{"type":"content_block_start","index":0,"content_block":{"type":7}}',
        ),
        (
            "anthropic",
            '{"type":"content_block_start","index":0,'
            '"content_block":{"type":"tool_use","id":7,"name":"f","input":{}}}',
        ),
        (
            "anthropic",
            '{"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":true}}',
        ),
        ("copilot-chat", '{"choices":["' + RAW_MARKER + '"]}'),
        (
            "copilot-chat",
            '{"choices":[{"delta":{"content":7}}]}',
        ),
        (
            "copilot-chat",
            '{"choices":[{"delta":{"tool_calls":"bad"}}]}',
        ),
        (
            "copilot-chat",
            '{"choices":[{"delta":{},"finish_reason":7}]}',
        ),
        ("copilot-chat", '{"choices":[],"usage":7}'),
        ("copilot-chat", '{"choices":[],"usage":{"prompt_tokens":true}}'),
        ("copilot-chat", '{"choices":[],"usage":{"completion_tokens":"1"}}'),
        ("copilot-chat", '{"choices":[],"usage":{"total_tokens":false}}'),
        (
            "copilot-chat",
            '{"choices":[{"delta":{"refusal":7}}]}',
        ),
        (
            "copilot-chat",
            '{"choices":[{"delta":{"tool_calls":[{"id":7}]}}]}',
        ),
        (
            "copilot-chat",
            '{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":7}}]}}]}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","item":"' + RAW_MARKER + '"}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_text.delta","delta":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.refusal.delta","delta":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.refusal.done","refusal":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","item":'
            '{"type":"function_call","call_id":7,"name":"f","arguments":"{}"}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","item":'
            '{"type":"function_call","call_id":"c","name":7,"arguments":"{}"}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","item":'
            '{"type":"function_call","call_id":"c","name":"f","arguments":7}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_text.delta","delta":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_text.delta","delta":"x","item_id":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_text.done","text":"x","item_id":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.function_call_arguments.delta","output_index":0,"delta":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.function_call_arguments.done","output_index":0,"arguments":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.custom_tool_call_input.done","output_index":0,"input":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_text.done","text":7}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_part.added","output_index":[],'
            '"summary_index":0,"part":{"type":"summary_text","text":""}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_part.added","output_index":true,'
            '"summary_index":0,"part":{"type":"summary_text","text":""}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_part.done","output_index":-1,'
            '"summary_index":0,"part":{"type":"summary_text","text":"x"}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_part.done","output_index":0,'
            '"summary_index":"0","part":{"type":"summary_text","text":"x"}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_part.added","output_index":0,'
            '"summary_index":0,"item_id":7,'
            '"part":{"type":"summary_text","text":""}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_part.added","output_index":0,'
            '"summary_index":0,"part":[]}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_part.done","output_index":0,'
            '"summary_index":0,"part":{"type":"other","text":"x"}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.reasoning_summary_part.done","output_index":0,'
            '"summary_index":0,"part":{"type":"summary_text","text":7}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.added","output_index":0,'
            '"item":{"type":"function_call","call_id":"c","name":"f","namespace":7}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","output_index":0,'
            '"item":{"type":"function_call","call_id":"c","name":"f",'
            '"arguments":"{}","namespace":7}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","item":{"type":"reasoning","summary":7}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","item":{"type":"reasoning","id":7,"summary":[]}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","item":'
            '{"type":"reasoning","summary":[{"text":7}]}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.added","output_index":"0",'
            '"item":{"type":"function_call","call_id":"c","name":"f"}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.added","output_index":0,'
            '"item":{"type":"custom_tool_call","call_id":7,"name":"f"}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","output_index":0,'
            '"item":{"type":"message","content":7}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.output_item.done","output_index":0,'
            '"item":{"type":"custom_tool_call","call_id":"c","name":"f","input":7}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.completed","response":{"status":"completed","usage":7}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.completed","response":{"status":"completed",'
            '"usage":{"input_tokens":true}}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.completed","response":{"status":"completed",'
            '"usage":{"input_tokens_details":"bad"}}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.completed","response":{"status":"completed",'
            '"usage":{"output_tokens_details":[]}}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.completed","response":{"status":"completed",'
            '"usage":{"input_tokens_details":{"cached_tokens":true}}}}',
        ),
        (
            "copilot-responses",
            '{"type":"response.completed","response":{"status":"completed",'
            '"usage":{"output_tokens_details":{"reasoning_tokens":"1"}}}}',
        ),
    ],
)
@pytest.mark.parametrize("after_first", [False, True], ids=["pre-first", "post-first"])
async def test_sse_known_event_with_malformed_shape_is_typed_protocol_failure(
    codec: str,
    malformed_event: str,
    after_first: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_events = {
        "openai-compatible": '{"choices":[{"delta":{"content":"first"}}]}',
        "anthropic": (
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first"}}'
        ),
        "copilot-chat": '{"choices":[{"delta":{"content":"first"}}]}',
        "copilot-responses": '{"type":"response.output_text.delta","delta":"first"}',
    }
    events = [first_events[codec], malformed_event] if after_first else [malformed_event]
    stream, provider, model = _stream_for_codec(codec, _sse_body(*events), monkeypatch)
    chunks: list[ChatStreamChunk | ResponsesStreamChunk] = []

    with pytest.raises(ProviderError) as exc_info:
        async for chunk in stream:
            chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == (["first"] if after_first else [])
    _assert_protocol_failure(exc_info.value, provider=provider, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("codec", "events", "expected_usage"),
    [
        (
            "openai-compatible",
            [
                '{"type":"future.event"}',
                '{"choices":[],"usage":{"total_tokens":3}}',
                "[DONE]",
            ],
            {"total_tokens": 3},
        ),
        ("anthropic", ['{"type":"future_event"}'], None),
        (
            "copilot-chat",
            [
                '{"type":"future.event"}',
                '{"choices":[],"usage":{"total_tokens":3}}',
                "[DONE]",
            ],
            {"total_tokens": 3},
        ),
        ("copilot-responses", ['{"type":"response.future"}', "[DONE]"], None),
    ],
)
async def test_sse_unknown_events_usage_only_and_done_remain_forward_compatible(
    codec: str,
    events: list[str],
    expected_usage: dict | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _provider, _model = _stream_for_codec(codec, _sse_body(*events), monkeypatch)

    chunks = [chunk async for chunk in stream]

    assert [chunk.usage for chunk in chunks if chunk.usage] == (
        [expected_usage] if expected_usage is not None else []
    )


@pytest.mark.asyncio
async def test_copilot_chat_preserves_usage_only_chunk_after_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _provider, _model = _stream_for_codec(
        "copilot-chat",
        _sse_body(
            '{"choices":[{"delta":{"content":"first"},"finish_reason":null}]}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
            '{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
            "[DONE]",
        ),
        monkeypatch,
    )

    chunks = [chunk async for chunk in stream]

    assert [chunk.finish_reason for chunk in chunks if chunk.finish_reason] == ["stop"]
    assert [chunk.usage for chunk in chunks if chunk.usage] == [
        {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    ]


@pytest.mark.asyncio
async def test_copilot_chat_ignores_identical_duplicate_post_terminal_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_event = (
        '{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}'
    )
    stream, _provider, _model = _stream_for_codec(
        "copilot-chat",
        _sse_body(
            '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
            usage_event,
            usage_event,
            "[DONE]",
        ),
        monkeypatch,
    )

    chunks = [chunk async for chunk in stream]

    assert [chunk.usage for chunk in chunks if chunk.usage] == [
        {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    ]


@pytest.mark.asyncio
async def test_copilot_chat_emits_identical_usage_once_across_all_stream_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = '{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}'
    stream, _provider, _model = _stream_for_codec(
        "copilot-chat",
        _sse_body(
            f'{{"choices":[],"usage":{usage}}}',
            f'{{"choices":[{{"delta":{{}},"finish_reason":"stop"}}],"usage":{usage}}}',
            f'{{"choices":[],"usage":{usage}}}',
            "[DONE]",
        ),
        monkeypatch,
    )

    chunks = [chunk async for chunk in stream]

    assert [chunk.finish_reason for chunk in chunks if chunk.finish_reason] == ["stop"]
    assert [chunk.usage for chunk in chunks if chunk.usage] == [
        {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        pytest.param(
            [
                '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
                '{"choices":[],"usage":{"total_tokens":3}}',
                '{"choices":[],"usage":{"total_tokens":999}}',
            ],
            id="conflicting-tail-usage",
        ),
        pytest.param(
            [
                '{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":3}}',
                '{"choices":[],"usage":{"total_tokens":999}}',
            ],
            id="conflicts-with-terminal-usage",
        ),
        pytest.param(
            [
                '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
                '{"type":"future.event","usage":{"total_tokens":3}}',
            ],
            id="missing-explicit-empty-choices",
        ),
        pytest.param(
            [
                '{"choices":[],"usage":{"total_tokens":3}}',
                '{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":999}}',
            ],
            id="pre-terminal-conflicts-with-terminal",
        ),
        pytest.param(
            [
                '{"choices":[],"usage":{"total_tokens":3}}',
                '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
                '{"choices":[],"usage":{"total_tokens":999}}',
            ],
            id="pre-terminal-conflicts-with-tail",
        ),
    ],
)
async def test_copilot_chat_rejects_malformed_post_terminal_usage(
    events: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, provider, model = _stream_for_codec(
        "copilot-chat",
        _sse_body(*events, "[DONE]"),
        monkeypatch,
    )

    with pytest.raises(ProviderError) as exc_info:
        async for _chunk in stream:
            pass

    _assert_protocol_failure(exc_info.value, provider=provider, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize("codec", ["openai-compatible", "copilot-chat"])
async def test_chat_sse_preserves_refusal_delta(
    codec: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _provider, _model = _stream_for_codec(
        codec,
        _sse_body(
            '{"choices":[{"delta":{"refusal":"I cannot help"},"finish_reason":null}]}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "[DONE]",
        ),
        monkeypatch,
    )

    chunks = [chunk async for chunk in stream]

    assert [chunk.refusal for chunk in chunks if chunk.refusal] == ["I cannot help"]


@pytest.mark.asyncio
async def test_responses_stream_malformed_terminal_is_typed_protocol_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, provider, model = _stream_for_codec(
        "copilot-responses",
        _sse_body(
            '{"type":"response.output_text.delta","delta":"first"}',
            '{"type":"response.completed","response":{"status":"' + RAW_MARKER + '"}}',
        ),
        monkeypatch,
    )
    chunks: list[ResponsesStreamChunk] = []

    with pytest.raises(ProviderError) as exc_info:
        async for chunk in stream:
            assert isinstance(chunk, ResponsesStreamChunk)
            chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == ["first"]
    _assert_protocol_failure(exc_info.value, provider=provider, model=model)


@pytest.mark.asyncio
async def test_responses_stream_cancelled_terminal_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _provider, _model = _stream_for_codec(
        "copilot-responses",
        _sse_body('{"type":"response.cancelled","response":{"status":"cancelled","error":null}}'),
        monkeypatch,
    )

    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 1
    assert chunks[0].terminal_outcome is not None
    assert chunks[0].terminal_outcome.response_status is ResponseStatus.CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize("after_first", [False, True], ids=["pre-first", "post-first"])
async def test_responses_stream_cancelled_malformed_schema_is_protocol_failure(
    after_first: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    if after_first:
        events.append('{"type":"response.output_text.delta","delta":"first"}')
    events.append('{"type":"response.cancelled","response":"' + RAW_MARKER + '"}')
    stream, provider, model = _stream_for_codec(
        "copilot-responses", _sse_body(*events), monkeypatch
    )
    chunks: list[ResponsesStreamChunk] = []

    with pytest.raises(ProviderError) as exc_info:
        async for chunk in stream:
            assert isinstance(chunk, ResponsesStreamChunk)
            chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == (["first"] if after_first else [])
    _assert_protocol_failure(exc_info.value, provider=provider, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (429, ProviderFailureKind.RATE_LIMIT, True),
        (400, ProviderFailureKind.UPSTREAM_STATUS, False),
        (503, ProviderFailureKind.UPSTREAM_STATUS, True),
    ],
)
async def test_copilot_responses_stream_status_error_is_typed_and_safe(
    status: int,
    kind: ProviderFailureKind,
    retryable: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=RAW_MARKER.encode())

    provider = CopilotProvider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider.ensure_token = _noop  # type: ignore[method-assign]
    provider._get_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        async for _ in provider.responses_completion_stream(
            ResponsesRequest(model="gpt-5", input="hi", stream=True)
        ):
            pass

    error = exc_info.value
    assert error.kind is kind
    assert error.upstream_status_code == status
    assert error.retryable is retryable
    assert error.provider == provider.name
    assert error.model == "gpt-5"
    assert RAW_MARKER not in str(error)
    assert RAW_MARKER not in caplog.text


@pytest.mark.asyncio
async def test_copilot_persistent_auth_failure_preserves_model_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    provider = CopilotProvider()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider._get_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    provider._refresh_for_auth_status = AsyncMock(return_value=True)  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        await provider._send_with_auth_retry(
            "POST",
            "/chat/completions",
            client=client,
            json={},
            model="gpt-4o",
        )

    assert exc_info.value.kind is ProviderFailureKind.AUTHENTICATION
    assert exc_info.value.model == "gpt-4o"


@pytest.mark.asyncio
async def test_copilot_transport_failure_preserves_model_context() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(RAW_MARKER)

    provider = CopilotProvider()
    clients = [
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ]
    provider._get_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    provider._recycle_client = AsyncMock()  # type: ignore[method-assign]
    provider._get_client = lambda: clients.pop(0)  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        await provider._send_with_auth_retry(
            "POST",
            "/chat/completions",
            json={},
            model="gpt-4o",
        )

    assert exc_info.value.kind is ProviderFailureKind.TRANSPORT
    assert exc_info.value.model == "gpt-4o"
    assert RAW_MARKER not in str(exc_info.value)


@pytest.mark.asyncio
async def test_copilot_responses_stream_529_is_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, request=request)

    provider = CopilotProvider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider.ensure_token = _noop  # type: ignore[method-assign]
    provider._get_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        async for _ in provider.responses_completion_stream(
            ResponsesRequest(model="gpt-5", input="hi", stream=True)
        ):
            pass

    assert exc_info.value.kind is ProviderFailureKind.RATE_LIMIT
    assert exc_info.value.retryable is True
    assert exc_info.value.upstream_status_code == 529
    assert exc_info.value.model == "gpt-5"


@pytest.mark.asyncio
async def test_copilot_token_refresh_529_is_rate_limit() -> None:
    response = httpx.Response(
        529,
        request=httpx.Request("GET", "https://api.github.com/copilot_internal/v2/token"),
    )
    status_error = httpx.HTTPStatusError(
        "rate limited",
        request=response.request,
        response=response,
    )
    provider = CopilotProvider()
    provider.auth_manager.storage.set(
        "github-copilot",
        OAuthCredential(refresh="github-token", access="", expires=0),
    )

    with patch(
        "router_maestro.providers.copilot.get_copilot_token",
        new=AsyncMock(side_effect=status_error),
    ):
        with pytest.raises(ProviderError) as exc_info:
            await provider.ensure_token()

    assert exc_info.value.kind is ProviderFailureKind.RATE_LIMIT
    assert exc_info.value.retryable is True
    assert exc_info.value.upstream_status_code == 529


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"secret":"' + RAW_MARKER + '", BAD}</tool_call>',
        '<tool_call>{"arguments":"' + RAW_MARKER + '"}</tool_call>',
    ],
)
def test_tool_recovery_warning_does_not_log_raw_tool_payload(
    content: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recovered_content, tool_calls = recover_tool_calls_from_content(
        content,
        None,
        "tool_calls",
    )

    assert recovered_content == content
    assert tool_calls is None
    assert RAW_MARKER not in caplog.text


@pytest.mark.asyncio
async def test_openai_model_list_transport_log_does_not_expose_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            raise httpx.ConnectError(RAW_MARKER)

    provider = OpenAIProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    with patch("router_maestro.providers.openai.httpx.AsyncClient", return_value=_FailingClient()):
        models = await provider.list_models()

    assert models
    assert RAW_MARKER not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_cls", [OpenAIProvider, AnthropicProvider, CopilotProvider])
async def test_first_party_authentication_failures_are_typed(provider_cls) -> None:
    provider = provider_cls()
    provider.auth_manager.get_credential = lambda _name: None  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        if isinstance(provider, OpenAIProvider):
            provider._get_api_key()
        elif isinstance(provider, AnthropicProvider):
            provider._get_api_key()
        else:
            await provider.ensure_token()

    assert exc_info.value.kind is ProviderFailureKind.AUTHENTICATION
    assert exc_info.value.status_code == 401
    assert exc_info.value.provider == provider.name


@pytest.mark.asyncio
async def test_default_responses_failure_is_typed_unsupported_operation() -> None:
    provider = OpenAICompatibleProvider(
        name="custom-provider",
        base_url="https://upstream.invalid",
        api_key="secret-token",
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.responses_completion(ResponsesRequest(model="m", input="hi"))

    assert exc_info.value.kind is ProviderFailureKind.UNSUPPORTED_OPERATION
    assert exc_info.value.provider == "custom-provider"
    assert exc_info.value.model == "m"


@pytest.mark.parametrize("status", [ResponseStatus.FAILED, ResponseStatus.CANCELLED])
def test_responses_bridge_business_failure_is_typed(status: ResponseStatus) -> None:
    response = ResponsesResponse(
        content="",
        model="gpt-5",
        terminal_outcome=TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=status,
            error=TerminalError(code="upstream_terminal", message="safe terminal failure"),
        ),
    )

    with pytest.raises(ProviderError) as exc_info:
        responses_response_to_chat_response(response, "gpt-5")

    assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_STATUS
    assert exc_info.value.provider == "github-copilot"
    assert exc_info.value.model == "gpt-5"


def test_responses_bridge_business_failure_does_not_expose_upstream_message() -> None:
    response = ResponsesResponse(
        content="",
        model="gpt-5",
        terminal_outcome=TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.FAILED,
            error=TerminalError(code="upstream_terminal", message=RAW_MARKER),
        ),
    )

    with pytest.raises(ProviderError) as exc_info:
        responses_response_to_chat_response(response, "gpt-5")

    assert RAW_MARKER not in str(exc_info.value)


def test_shared_status_helper_accepts_canonical_provider_and_model_context() -> None:
    response = httpx.Response(
        503,
        request=httpx.Request("POST", "https://upstream.invalid/chat"),
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as caught:
        with pytest.raises(ProviderError) as exc_info:
            BaseProvider._raise_http_status_error(
                "Friendly Label",
                caught,
                get_logger("test.provider_failure"),
                provider="canonical-provider",
                model="model-a",
            )

    assert exc_info.value.provider == "canonical-provider"
    assert exc_info.value.model == "model-a"


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (401, ProviderFailureKind.AUTHENTICATION, False),
        (403, ProviderFailureKind.AUTHENTICATION, False),
        (429, ProviderFailureKind.RATE_LIMIT, True),
        (529, ProviderFailureKind.RATE_LIMIT, True),
        (400, ProviderFailureKind.UPSTREAM_STATUS, False),
        (503, ProviderFailureKind.UPSTREAM_STATUS, True),
    ],
)
def test_shared_http_status_errors_are_typed_and_safe(
    status: int,
    kind: ProviderFailureKind,
    retryable: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = RAW_MARKER.encode()
    response = httpx.Response(
        status,
        content=raw,
        request=httpx.Request("POST", "https://upstream.invalid/chat"),
    )
    cause: httpx.HTTPStatusError | None = None
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as caught:
        cause = caught
        with pytest.raises(ProviderError) as exc_info:
            BaseProvider._raise_http_status_error(
                "Example",
                caught,
                get_logger("test.provider_failure"),
                include_body=True,
            )
    error = exc_info.value
    assert error.kind is kind
    assert error.status_code == status
    assert error.upstream_status_code == status
    assert error.retryable is retryable
    assert error.provider == "Example"
    assert cause is not None
    assert error.cause is cause
    assert RAW_MARKER not in str(error)
    assert RAW_MARKER not in caplog.text


@pytest.mark.parametrize(
    "cause",
    [
        httpx.ReadTimeout(RAW_MARKER),
        httpx.ConnectError(RAW_MARKER),
    ],
)
def test_shared_transport_errors_are_typed_and_safe(cause: httpx.HTTPError) -> None:
    raiser = (
        BaseProvider._raise_timeout_error
        if isinstance(cause, httpx.TimeoutException)
        else BaseProvider._raise_http_error
    )
    with pytest.raises(ProviderError) as exc_info:
        raiser("Example", cause, get_logger("test.provider_failure"))

    error = exc_info.value
    assert error.kind is ProviderFailureKind.TRANSPORT
    assert error.provider == "Example"
    assert error.cause is cause
    assert error.retryable is True
    assert RAW_MARKER not in str(error)


class _StreamProvider:
    def __init__(self, chunks: list[str], error: ProviderError | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.calls = 0

    async def ensure_token(self) -> None:
        return None

    async def stream(self, _request: object) -> AsyncIterator[ChatStreamChunk]:
        self.calls += 1
        for content in self.chunks:
            yield ChatStreamChunk(content=content)
        if self.error is not None:
            raise self.error


class _PreparedStreamProvider:
    def __init__(self, stream: AsyncIterator[ChatStreamChunk | ResponsesStreamChunk]) -> None:
        self.stream = stream

    async def ensure_token(self) -> None:
        return None

    def open(self, _request: object) -> AsyncIterator[ChatStreamChunk | ResponsesStreamChunk]:
        return self.stream


class _ClosableEmptyStream(AsyncIterator[ChatStreamChunk]):
    def __init__(self) -> None:
        self.close_calls = 0

    def __aiter__(self) -> "_ClosableEmptyStream":
        return self

    async def __anext__(self) -> ChatStreamChunk:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1


def _stream_plan(*providers: object) -> RoutePlan:
    candidates: list[RouteCandidate] = []
    operation = Operation.CHAT_STREAM
    features = RequestFeatures()
    for index, provider in enumerate(providers, start=1):
        name = getattr(provider, "name", None) or f"provider-{index}"
        if getattr(provider, "name", None) is None:
            provider.name = name
        ref = ModelRef(name, f"m{index}")
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: CapabilitySupport.SUPPORTED},
        )
        candidates.append(
            RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            )
        )
    return RoutePlan(
        operation,
        features,
        candidates[0],
        tuple(candidates[1:]),
        False,
    )


@pytest.mark.asyncio
async def test_router_empty_stream_is_protocol_failure_and_closed_once() -> None:
    empty = _ClosableEmptyStream()
    primary = _PreparedStreamProvider(empty)
    router = Router.__new__(Router)

    with pytest.raises(ProviderError) as exc_info:
        await router._execute_plan_stream(
            _stream_plan(primary),
            object(),
            True,
            lambda request, _model: request,
            lambda provider, request: provider.open(request),
            "test",
        )

    _assert_protocol_failure(exc_info.value, provider="provider-1", model="m1")
    assert empty.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("codec", "body"),
    [
        (codec, body)
        for codec in ("openai-compatible", "anthropic", "copilot-chat", "copilot-responses")
        for body in (b"", _sse_body("[DONE]"), _sse_body('{"type":"future.event"}'))
    ],
)
async def test_router_falls_back_when_adapter_stream_has_no_canonical_chunk(
    codec: str,
    body: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_stream, _provider_name, _model = _stream_for_codec(codec, body, monkeypatch)
    primary = _PreparedStreamProvider(primary_stream)
    secondary = _StreamProvider(["secondary"])
    router = Router.__new__(Router)

    stream, provider_name = await router._execute_plan_stream(
        _stream_plan(primary, secondary),
        object(),
        True,
        lambda request, _model: request,
        lambda provider, request: (
            provider.open(request)
            if isinstance(provider, _PreparedStreamProvider)
            else provider.stream(request)
        ),
        "test",
    )

    assert provider_name == "provider-2"
    assert [chunk.content async for chunk in stream] == ["secondary"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunk",
    [
        ChatStreamChunk(content="", usage={"total_tokens": 3}),
        ResponsesStreamChunk(
            content="",
            terminal_outcome=TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.COMPLETED,
            ),
        ),
    ],
)
async def test_router_accepts_usage_only_or_terminal_first_canonical_chunk(chunk) -> None:
    primary = _StreamProvider([])

    async def stream(_request):
        primary.calls += 1
        yield chunk

    primary.stream = stream
    router = Router.__new__(Router)

    opened, provider_name = await router._execute_plan_stream(
        _stream_plan(primary),
        object(),
        True,
        lambda request, _model: request,
        lambda provider, request: provider.stream(request),
        "test",
    )

    assert provider_name == "provider-1"
    assert [item async for item in opened] == [chunk]


@pytest.mark.asyncio
async def test_copilot_responses_stream_emits_refusal_deltas_without_done_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _provider, _model = _stream_for_codec(
        "copilot-responses",
        _sse_body(
            '{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"safe "}',
            '{"type":"response.refusal.delta","output_index":0,'
            '"content_index":0,"delta":"refusal"}',
            '{"type":"response.refusal.done","output_index":0,'
            '"content_index":0,"refusal":"safe refusal"}',
            '{"type":"response.completed","response":{"status":"completed"}}',
        ),
        monkeypatch,
    )

    chunks = [chunk async for chunk in stream]

    assert [chunk.refusal for chunk in chunks if chunk.refusal] == ["safe ", "refusal"]
    assert [chunk.content for chunk in chunks if chunk.content] == []


@pytest.mark.asyncio
async def test_copilot_responses_stream_emits_done_only_refusal_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _provider, _model = _stream_for_codec(
        "copilot-responses",
        _sse_body(
            '{"type":"response.refusal.done","output_index":0,'
            '"content_index":0,"refusal":"safe refusal"}',
            '{"type":"response.completed","response":{"status":"completed"}}',
        ),
        monkeypatch,
    )

    chunks = [chunk async for chunk in stream]

    assert [chunk.refusal for chunk in chunks if chunk.refusal] == ["safe refusal"]
    assert [chunk.content for chunk in chunks if chunk.content] == []


@pytest.mark.asyncio
async def test_copilot_responses_stream_deduplicates_each_refusal_part_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, _provider, _model = _stream_for_codec(
        "copilot-responses",
        _sse_body(
            '{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"first"}',
            '{"type":"response.refusal.done","output_index":0,"content_index":0,"refusal":"first"}',
            '{"type":"response.refusal.done","output_index":0,'
            '"content_index":1,"refusal":"second"}',
            '{"type":"response.refusal.done","output_index":0,'
            '"content_index":1,"refusal":"second"}',
            '{"type":"response.completed","response":{"status":"completed"}}',
        ),
        monkeypatch,
    )

    chunks = [chunk async for chunk in stream]

    assert [chunk.refusal for chunk in chunks if chunk.refusal] == ["first", "second"]
    assert [chunk.content for chunk in chunks if chunk.content] == []


@pytest.mark.asyncio
async def test_nonstream_plan_stops_on_secondary_fatal_failure() -> None:
    primary_error = ProviderError("primary", retryable=True)
    secondary_error = ProviderError("secondary", retryable=False)
    calls: list[str] = []

    class _Provider:
        def __init__(self, name: str, result: object) -> None:
            self.name = name
            self.result = result

        async def ensure_token(self) -> None:
            return None

    providers = [
        _Provider("primary", primary_error),
        _Provider("secondary", secondary_error),
        _Provider("tertiary", "success"),
    ]
    candidates = []
    operation = Operation.CHAT
    features = RequestFeatures()
    for provider in providers:
        ref = ModelRef(provider.name, "model")
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: CapabilitySupport.SUPPORTED},
        )
        candidates.append(
            RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            )
        )
    plan = RoutePlan(operation, features, candidates[0], tuple(candidates[1:]), False)
    router = Router.__new__(Router)

    async def call(provider, _request):
        calls.append(provider.name)
        if isinstance(provider.result, ProviderError):
            raise provider.result
        return provider.result

    with pytest.raises(ProviderError) as exc_info:
        await router._execute_plan_nonstream(
            plan, object(), True, lambda request, _model: request, call
        )

    assert exc_info.value is not secondary_error
    assert exc_info.value.__cause__ is secondary_error
    assert calls == ["primary", "secondary"]


@pytest.mark.asyncio
async def test_nonstream_plan_raises_last_meaningful_error_after_retryable_exhaustion() -> None:
    primary_error = ProviderError("primary", retryable=True)
    fallback_errors = [
        ProviderError("secondary", retryable=True),
        ProviderError("tertiary", retryable=True),
    ]

    class _Provider:
        def __init__(self, name: str, error: ProviderError) -> None:
            self.name = name
            self.error = error

        async def ensure_token(self) -> None:
            return None

    providers = [
        _Provider("primary", primary_error),
        _Provider("secondary", fallback_errors[0]),
        _Provider("tertiary", fallback_errors[1]),
    ]
    candidates = []
    operation = Operation.CHAT
    features = RequestFeatures()
    for provider in providers:
        ref = ModelRef(provider.name, "model")
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: CapabilitySupport.SUPPORTED},
        )
        candidates.append(
            RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=CapabilitySupport.SUPPORTED,
            )
        )
    plan = RoutePlan(operation, features, candidates[0], tuple(candidates[1:]), False)
    router = Router.__new__(Router)

    async def call(provider, _request):
        raise provider.error

    with pytest.raises(ProviderError) as exc_info:
        await router._execute_plan_nonstream(
            plan, object(), True, lambda request, _model: request, call
        )

    assert exc_info.value is not fallback_errors[-1]
    assert exc_info.value.__cause__ is fallback_errors[-1]
    assert [attempt.provider for attempt in exc_info.value.attempts] == [
        "primary",
        "secondary",
        "tertiary",
    ]


@pytest.mark.asyncio
async def test_router_falls_back_on_protocol_failure_before_first_canonical_chunk() -> None:
    protocol_error = ProviderError(
        "malformed",
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        status_code=502,
        upstream_status_code=200,
        retryable=True,
    )
    primary = _StreamProvider([], protocol_error)
    secondary = _StreamProvider(["secondary"])
    router = Router.__new__(Router)

    stream, provider_name = await router._execute_plan_stream(
        _stream_plan(primary, secondary),
        object(),
        True,
        lambda request, _model: request,
        lambda provider, request: provider.stream(request),
        "test",
    )

    assert provider_name == "provider-2"
    assert [chunk.content async for chunk in stream] == ["secondary"]
    assert primary.calls == 1
    assert secondary.calls == 1


@pytest.mark.asyncio
async def test_router_never_switches_provider_after_first_canonical_chunk() -> None:
    protocol_error = ProviderError(
        "malformed",
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        status_code=502,
        upstream_status_code=200,
        retryable=True,
    )
    primary = _StreamProvider(["primary"], protocol_error)
    secondary = _StreamProvider(["secondary"])
    router = Router.__new__(Router)
    stream, provider_name = await router._execute_plan_stream(
        _stream_plan(primary, secondary),
        object(),
        True,
        lambda request, _model: request,
        lambda provider, request: provider.stream(request),
        "test",
    )

    contents: list[str] = []
    with pytest.raises(ProviderError) as exc_info:
        async for chunk in stream:
            contents.append(chunk.content)

    assert provider_name == "provider-1"
    assert contents == ["primary"]
    assert exc_info.value is protocol_error
    assert secondary.calls == 0


@pytest.mark.asyncio
async def test_router_model_not_found_is_typed_client_request() -> None:
    router = Router.__new__(Router)
    router._get_auto_route_model = AsyncMock(return_value=None)  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        await router._resolve_provider("router-maestro")

    assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_STATUS
    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_router_explicit_model_not_found_is_typed_client_request() -> None:
    router = Router.__new__(Router)
    router._find_model_in_cache = AsyncMock(return_value=None)  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        await router._resolve_provider("missing/model")

    assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert exc_info.value.status_code == 404
    assert exc_info.value.retryable is False


def test_router_static_capability_failure_is_typed_client_request() -> None:
    class _Provider:
        name = "provider"

    from router_maestro.routing.capabilities import ModelCapabilities
    from router_maestro.routing.route_plan import RouteCandidate, RoutePlan

    ref = ModelRef("provider", "model")
    operation = Operation.CHAT
    features = RequestFeatures()
    capabilities = ModelCapabilities(
        model=ref,
        operations={operation: CapabilitySupport.UNSUPPORTED},
    )
    candidate = RouteCandidate(
        model=ref,
        provider=_Provider(),  # type: ignore[arg-type]
        capabilities=capabilities,
        evaluated_operation=operation,
        evaluated_features=features,
        support=CapabilitySupport.UNSUPPORTED,
    )
    plan = RoutePlan(
        operation=operation,
        features=features,
        primary=candidate,
        fallbacks=(),
        explicit=True,
    )

    with pytest.raises(ProviderError) as exc_info:
        Router._validate_plan_primary(plan)

    assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_router_no_compatible_route_is_typed_client_request() -> None:
    router = Router.__new__(Router)
    router.plan_route = AsyncMock(  # type: ignore[method-assign]
        side_effect=NoCompatibleRouteError("no compatible route")
    )

    with pytest.raises(ProviderError) as exc_info:
        await router._plan_completion_route(
            "router-maestro",
            Operation.CHAT,
            RequestFeatures(),
        )

    assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_router_empty_execution_plan_is_typed_availability_failure() -> None:
    router = Router.__new__(Router)
    ref = ModelRef("provider", "model")

    class _Provider:
        name = "provider"

    from router_maestro.routing.capabilities import ModelCapabilities
    from router_maestro.routing.route_plan import RouteCandidate, RoutePlan

    operation = Operation.CHAT
    features = RequestFeatures()
    capabilities = ModelCapabilities(
        model=ref,
        operations={operation: CapabilitySupport.SUPPORTED},
    )
    candidate = RouteCandidate(
        model=ref,
        provider=_Provider(),  # type: ignore[arg-type]
        capabilities=capabilities,
        evaluated_operation=operation,
        evaluated_features=features,
        support=CapabilitySupport.SUPPORTED,
    )
    plan = RoutePlan(
        operation=operation,
        features=features,
        primary=candidate,
        fallbacks=(),
        explicit=False,
    )
    router._plan_execution_candidates = lambda *_args: ()  # type: ignore[method-assign]

    with pytest.raises(ProviderError) as exc_info:
        await router._execute_plan_nonstream(
            plan,
            object(),
            True,
            lambda request, _model: request,
            AsyncMock(),
        )

    assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_STATUS
    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is True
