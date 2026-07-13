"""Protocol-native client-error envelopes for option rejection."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.config import PrioritiesConfig
from router_maestro.providers import (
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    Feature,
    ModelCapabilities,
    Operation,
    ProviderCapabilities,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.server.routes.anthropic import router as anthropic_router
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.server.routes.gemini import router as gemini_router
from router_maestro.server.routes.responses import router as responses_router
from router_maestro.utils.cache import TTLCache


def _unsupported(parameter: str) -> RequestOptionError:
    return RequestOptionError(
        f"request option '{parameter}' is not supported",
        parameter=parameter,
    )


def _router(error: ProviderError):
    router = MagicMock()
    router.chat_completion = AsyncMock(side_effect=error)
    router.chat_completion_stream = AsyncMock(side_effect=AssertionError("stream must not start"))
    router.validate_chat_request = AsyncMock(side_effect=error)
    router.prepare_chat_completion_stream = AsyncMock(side_effect=error)
    router.plan_chat_completion = AsyncMock(side_effect=error)
    router.prepare_planned_chat_completion = MagicMock(side_effect=error)
    router.get_model_info = AsyncMock(return_value=None)
    router._resolve_provider = AsyncMock(return_value=("test", "m", MagicMock()))
    return router


def _static_capability_router(
    operation: Operation,
    *,
    feature: Feature | None = None,
) -> tuple[Router, MagicMock, MagicMock]:
    primary = MagicMock()
    primary.name = "primary"
    primary.is_authenticated.return_value = True
    primary.capabilities = ProviderCapabilities(operations=frozenset(Operation))
    primary.validate_chat_request = MagicMock()
    primary.validate_responses_request = MagicMock()
    primary.ensure_token = AsyncMock()
    primary.chat_completion = AsyncMock()
    primary.chat_completion_stream = MagicMock()
    primary.responses_completion = AsyncMock()
    primary.responses_completion_stream = MagicMock()

    fallback = MagicMock()
    fallback.name = "fallback"
    fallback.is_authenticated.return_value = True
    fallback.capabilities = ProviderCapabilities(operations=frozenset(Operation))
    fallback.validate_chat_request = MagicMock()
    fallback.validate_responses_request = MagicMock()
    fallback.ensure_token = AsyncMock()
    fallback.chat_completion = AsyncMock()
    fallback.chat_completion_stream = MagicMock()
    fallback.responses_completion = AsyncMock()
    fallback.responses_completion_stream = MagicMock()

    primary_model = ModelInfo(
        id="m",
        name="m",
        provider="primary",
        operation_capabilities={operation.value: feature is not None},
        feature_capabilities=({feature.value: False} if feature is not None else {}),
    )
    fallback_model = ModelInfo(
        id="other",
        name="other",
        provider="fallback",
        operation_capabilities={operation.value: True},
        feature_capabilities=({feature.value: True} if feature is not None else {}),
    )

    router = Router.__new__(Router)
    router.providers = {"primary": primary, "fallback": fallback}
    router._models_cache = {
        "m": ("primary", primary_model),
        "primary/m": ("primary", primary_model),
        "other": ("fallback", fallback_model),
        "fallback/other": ("fallback", fallback_model),
    }
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._models_cache_ttl.set(True)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache.set(PrioritiesConfig(priorities=["fallback/other"]))
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._providers_ttl.set(True)
    return router, primary, fallback


def _assert_no_provider_work(*providers: MagicMock) -> None:
    for provider in providers:
        provider.validate_chat_request.assert_not_called()
        provider.validate_responses_request.assert_not_called()
        provider.ensure_token.assert_not_awaited()
        provider.chat_completion.assert_not_awaited()
        provider.chat_completion_stream.assert_not_called()
        provider.responses_completion.assert_not_awaited()
        provider.responses_completion_stream.assert_not_called()


def _assert_native_capability_error(
    response,
    protocol: str,
    *,
    parameter: str,
) -> None:
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    if protocol == "openai":
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == parameter
        assert body["error"]["code"] == "unsupported_parameter"
    elif protocol == "anthropic":
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
    else:
        assert body["error"]["status"] == "INVALID_ARGUMENT"
        assert body["error"]["details"] == [
            {"reason": "unsupported_parameter", "parameter": parameter}
        ]


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(anthropic_router)
    app.include_router(gemini_router)
    app.include_router(responses_router)
    return TestClient(app)


def test_openai_option_rejection_uses_native_envelope(client: TestClient) -> None:
    with patch(
        "router_maestro.server.routes.chat.get_router",
        return_value=_router(_unsupported("top_k")),
    ):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "request option 'top_k' is not supported",
            "type": "invalid_request_error",
            "param": "top_k",
            "code": "unsupported_parameter",
        }
    }


@pytest.mark.parametrize(
    ("path", "base_payload", "parameter", "value", "router_method"),
    [
        (
            "/api/openai/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "response_format",
            {"type": "json_object"},
            "chat_completion",
        ),
        (
            "/api/openai/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "n",
            2,
            "chat_completion",
        ),
        (
            "/api/openai/v1/responses",
            {"model": "m", "input": "hi"},
            "store",
            True,
            "responses_completion",
        ),
        (
            "/api/openai/v1/responses",
            {"model": "m", "input": "hi"},
            "include",
            ["reasoning.encrypted_content"],
            "responses_completion",
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_unrepresented_openai_semantic_option_is_native_400_before_router_or_sse(
    client: TestClient,
    path: str,
    base_payload: dict,
    parameter: str,
    value: object,
    router_method: str,
    stream: bool,
) -> None:
    model_router = MagicMock()
    model_router.chat_completion = AsyncMock()
    model_router.prepare_chat_completion_stream = AsyncMock()
    model_router.responses_completion = AsyncMock()
    model_router.prepare_responses_completion_stream = AsyncMock()
    patch_target = (
        "router_maestro.server.routes.chat.get_router"
        if router_method == "chat_completion"
        else "router_maestro.server.routes.responses.get_router"
    )
    payload = {**base_payload, "stream": stream, parameter: value}

    with patch(patch_target, return_value=model_router):
        response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "error": {
            "message": f"Unsupported request option '{parameter}'",
            "type": "invalid_request_error",
            "param": parameter,
            "code": "unsupported_parameter",
        }
    }
    assert "data:" not in response.text
    model_router.chat_completion.assert_not_awaited()
    model_router.prepare_chat_completion_stream.assert_not_awaited()
    model_router.responses_completion.assert_not_awaited()
    model_router.prepare_responses_completion_stream.assert_not_awaited()


def test_anthropic_option_rejection_uses_native_envelope(client: TestClient) -> None:
    with patch(
        "router_maestro.server.routes.anthropic.get_router",
        return_value=_router(_unsupported("frequency_penalty")),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "m",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "request option 'frequency_penalty' is not supported",
        },
    }


def test_gemini_option_rejection_uses_native_envelope(client: TestClient) -> None:
    with patch(
        "router_maestro.server.routes.gemini.get_router",
        return_value=_router(_unsupported("topK")),
    ):
        response = client.post(
            "/api/gemini/v1beta/models/m:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": 400,
            "message": "request option 'topK' is not supported",
            "status": "INVALID_ARGUMENT",
            "details": [{"reason": "unsupported_parameter", "parameter": "topK"}],
        }
    }


@pytest.mark.parametrize(
    ("path", "parameter", "payload_option"),
    [
        (
            "/api/gemini/v1beta/models/m:generateContent",
            "unknownOption",
            {"unknownOption": True},
        ),
        (
            "/api/gemini/v1beta/models/m:generateContent",
            "generationConfig.unknownOption",
            {"generationConfig": {"unknownOption": True}},
        ),
        (
            "/api/gemini/v1beta/models/m:streamGenerateContent",
            "unknownOption",
            {"unknownOption": True},
        ),
        (
            "/api/gemini/v1beta/models/m:streamGenerateContent",
            "generationConfig.unknownOption",
            {"generationConfig": {"unknownOption": True}},
        ),
    ],
)
def test_unrepresented_gemini_option_is_native_400_before_router_or_sse(
    client: TestClient,
    path: str,
    parameter: str,
    payload_option: dict,
) -> None:
    model_router = MagicMock()
    model_router.chat_completion = AsyncMock()
    model_router.prepare_chat_completion_stream = AsyncMock()
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        **payload_option,
    }

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "error": {
            "code": 400,
            "message": f"Unsupported request option '{parameter}'",
            "status": "INVALID_ARGUMENT",
            "details": [{"reason": "unsupported_parameter", "parameter": parameter}],
        }
    }
    assert "data:" not in response.text
    model_router.chat_completion.assert_not_awaited()
    model_router.prepare_chat_completion_stream.assert_not_awaited()


def test_unrepresented_gemini_count_tokens_option_is_native_400(client: TestClient) -> None:
    response = client.post(
        "/api/gemini/v1beta/models/m:countTokens",
        json={
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"unknownOption": True},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["details"] == [
        {
            "reason": "unsupported_parameter",
            "parameter": "generationConfig.unknownOption",
        }
    ]


@pytest.mark.parametrize(
    "path",
    [
        "/api/gemini/v1beta/models/m:generateContent",
        "/api/gemini/v1beta/models/m:streamGenerateContent",
        "/api/gemini/v1beta/models/m:countTokens",
    ],
)
@pytest.mark.parametrize(
    ("payload_option", "parameter"),
    [
        (
            {
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": "AUTO",
                        "allowedFunctionNames": ["lookup"],
                    }
                }
            },
            "toolConfig.functionCallingConfig.allowedFunctionNames",
        ),
        (
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "fileData": {
                                    "mimeType": "image/png",
                                    "fileUri": "gs://bucket/image.png",
                                }
                            }
                        ],
                    }
                ]
            },
            "contents[0].parts[0].fileData",
        ),
    ],
)
def test_nested_unrepresented_gemini_option_is_native_400_before_provider_io(
    client: TestClient,
    path: str,
    payload_option: dict,
    parameter: str,
) -> None:
    model_router = MagicMock()
    model_router.chat_completion = AsyncMock()
    model_router.prepare_chat_completion_stream = AsyncMock()
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        **payload_option,
    }

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "error": {
            "code": 400,
            "message": f"Unsupported request option '{parameter}'",
            "status": "INVALID_ARGUMENT",
            "details": [{"reason": "unsupported_parameter", "parameter": parameter}],
        }
    }
    assert "data:" not in response.text
    model_router.chat_completion.assert_not_awaited()
    model_router.prepare_chat_completion_stream.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("parameter", "payload_option"),
    [
        ("frequency_penalty", {"frequency_penalty": 0.2}),
        ("output_config.format", {"output_config": {"format": {"type": "json_schema"}}}),
    ],
)
def test_unrepresented_anthropic_option_is_native_400_before_router_or_sse(
    client: TestClient,
    stream: bool,
    parameter: str,
    payload_option: dict,
) -> None:
    model_router = MagicMock()
    model_router.plan_chat_completion = AsyncMock()
    payload = {
        "model": "m",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        **payload_option,
    }

    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"Unsupported request option '{parameter}'",
        },
    }
    assert "event:" not in response.text
    model_router.plan_chat_completion.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
def test_nested_unrepresented_anthropic_tool_choice_is_native_400_before_provider_io(
    client: TestClient,
    stream: bool,
) -> None:
    model_router = MagicMock()
    model_router.plan_chat_completion = AsyncMock()
    payload = {
        "model": "m",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }

    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        response = client.post("/v1/messages", json=payload)

    parameter = "tool_choice.disable_parallel_tool_use"
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"Unsupported request option '{parameter}'",
        },
    }
    assert "event:" not in response.text
    model_router.plan_chat_completion.assert_not_awaited()


def test_nested_unrepresented_anthropic_count_tokens_option_is_native_400_before_io(
    client: TestClient,
) -> None:
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "lookup",
                "input_schema": {"type": "object"},
                "strict": True,
            }
        ],
    }

    with patch(
        "router_maestro.server.routes.anthropic._resolve_provider_name",
        new_callable=AsyncMock,
    ) as resolve_provider:
        response = client.post("/v1/messages/count_tokens", json=payload)

    parameter = "tools[0].strict"
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"Unsupported request option '{parameter}'",
        },
    }
    resolve_provider.assert_not_awaited()


def test_anthropic_content_vendor_payload_is_not_treated_as_request_option(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["type"] == "message"


def _routing_error() -> ProviderError:
    return ProviderError(
        "Model 'missing' not found",
        status_code=404,
        retryable=False,
        kind=ProviderFailureKind.CLIENT_REQUEST,
    )


@pytest.mark.parametrize(
    ("path", "payload", "patch_target", "expected"),
    [
        (
            "/api/openai/v1/chat/completions",
            {"model": "missing", "messages": [{"role": "user", "content": "hi"}]},
            "router_maestro.server.routes.chat.get_router",
            {"detail": "Model 'missing' not found"},
        ),
        (
            "/api/openai/v1/responses",
            {"model": "missing", "input": "hi"},
            "router_maestro.server.routes.responses.get_router",
            {"detail": "Model 'missing' not found"},
        ),
        (
            "/v1/messages",
            {
                "model": "missing",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "router_maestro.server.routes.anthropic.get_router",
            {"detail": "Model 'missing' not found"},
        ),
        (
            "/api/gemini/v1beta/models/missing:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            "router_maestro.server.routes.gemini.get_router",
            {
                "detail": {
                    "error": {
                        "code": 404,
                        "message": "Model 'missing' not found",
                        "status": "INTERNAL",
                    }
                }
            },
        ),
    ],
)
def test_non_option_client_error_keeps_route_specific_envelope(
    client: TestClient,
    path: str,
    payload: dict,
    patch_target: str,
    expected: dict,
) -> None:
    router = MagicMock()
    router.chat_completion = AsyncMock(side_effect=_routing_error())
    router.responses_completion = AsyncMock(side_effect=_routing_error())
    router.plan_chat_completion = AsyncMock(side_effect=_routing_error())
    router.prepare_planned_chat_completion = MagicMock(side_effect=_routing_error())
    router.get_model_info = AsyncMock(return_value=None)
    router._resolve_provider = AsyncMock(return_value=("test", "missing", MagicMock()))

    with patch(patch_target, return_value=router):
        response = client.post(path, json=payload)

    assert response.status_code == 404
    assert response.json() == expected


@pytest.mark.parametrize(
    ("path", "payload", "patch_target", "protocol", "operation"),
    [
        (
            "/api/openai/v1/chat/completions",
            {"model": "primary/m", "messages": [{"role": "user", "content": "hi"}]},
            "router_maestro.server.routes.chat.get_router",
            "openai",
            Operation.CHAT,
        ),
        (
            "/api/openai/v1/responses",
            {"model": "primary/m", "input": "hi"},
            "router_maestro.server.routes.responses.get_router",
            "openai",
            Operation.RESPONSES,
        ),
        (
            "/v1/messages",
            {
                "model": "primary/m",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "router_maestro.server.routes.anthropic.get_router",
            "anthropic",
            Operation.CHAT,
        ),
        (
            "/api/gemini/v1beta/models/primary/m:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            "router_maestro.server.routes.gemini.get_router",
            "gemini",
            Operation.CHAT,
        ),
    ],
)
def test_explicit_unsupported_operation_uses_native_400_without_model_switch(
    client: TestClient,
    path: str,
    payload: dict,
    patch_target: str,
    protocol: str,
    operation: Operation,
) -> None:
    router, primary, fallback = _static_capability_router(operation)

    with patch(patch_target, return_value=router):
        response = client.post(path, json=payload)

    _assert_native_capability_error(response, protocol, parameter=operation.value)
    _assert_no_provider_work(primary, fallback)


@pytest.mark.parametrize(
    ("path", "payload", "patch_target", "operation"),
    [
        (
            "/api/openai/v1/chat/completions",
            {
                "model": "primary/m",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "router_maestro.server.routes.chat.get_router",
            Operation.CHAT_STREAM,
        ),
        (
            "/api/openai/v1/responses",
            {"model": "primary/m", "input": "hi", "stream": True},
            "router_maestro.server.routes.responses.get_router",
            Operation.RESPONSES_STREAM,
        ),
    ],
)
def test_explicit_unsupported_stream_operation_is_json_before_sse(
    client: TestClient,
    path: str,
    payload: dict,
    patch_target: str,
    operation: Operation,
) -> None:
    router, primary, fallback = _static_capability_router(operation)

    with patch(patch_target, return_value=router):
        response = client.post(path, json=payload)

    _assert_native_capability_error(response, "openai", parameter=operation.value)
    assert "data:" not in response.text
    _assert_no_provider_work(primary, fallback)


@pytest.mark.parametrize(
    ("path", "payload", "patch_target", "protocol"),
    [
        (
            "/api/openai/v1/chat/completions",
            {
                "model": "primary/m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
            "router_maestro.server.routes.chat.get_router",
            "openai",
        ),
        (
            "/api/openai/v1/responses",
            {
                "model": "primary/m",
                "input": "hi",
                "tools": [{"type": "function", "name": "lookup"}],
            },
            "router_maestro.server.routes.responses.get_router",
            "openai",
        ),
        (
            "/v1/messages",
            {
                "model": "primary/m",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            },
            "router_maestro.server.routes.anthropic.get_router",
            "anthropic",
        ),
        (
            "/api/gemini/v1beta/models/primary/m:generateContent",
            {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "tools": [{"functionDeclarations": [{"name": "lookup"}]}],
            },
            "router_maestro.server.routes.gemini.get_router",
            "gemini",
        ),
    ],
)
def test_explicit_unsupported_tools_use_native_400_without_model_switch(
    client: TestClient,
    path: str,
    payload: dict,
    patch_target: str,
    protocol: str,
) -> None:
    router, primary, fallback = _static_capability_router(Operation.CHAT, feature=Feature.TOOLS)
    if protocol == "openai" and path.endswith("/responses"):
        router, primary, fallback = _static_capability_router(
            Operation.RESPONSES,
            feature=Feature.TOOLS,
        )

    with patch(patch_target, return_value=router):
        response = client.post(path, json=payload)

    _assert_native_capability_error(response, protocol, parameter="tools")
    _assert_no_provider_work(primary, fallback)


@pytest.mark.parametrize(
    ("path", "payload", "patch_target", "protocol"),
    [
        (
            "/api/openai/v1/chat/completions",
            {
                "model": "primary/m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            }
                        ],
                    }
                ],
            },
            "router_maestro.server.routes.chat.get_router",
            "openai",
        ),
        (
            "/api/openai/v1/responses",
            {
                "model": "primary/m",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
                        ],
                    }
                ],
            },
            "router_maestro.server.routes.responses.get_router",
            "openai",
        ),
        (
            "/v1/messages",
            {
                "model": "primary/m",
                "max_tokens": 32,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AA==",
                                },
                            }
                        ],
                    }
                ],
            },
            "router_maestro.server.routes.anthropic.get_router",
            "anthropic",
        ),
        (
            "/api/gemini/v1beta/models/primary/m:generateContent",
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"inlineData": {"mimeType": "image/png", "data": "AA=="}}],
                    }
                ]
            },
            "router_maestro.server.routes.gemini.get_router",
            "gemini",
        ),
    ],
)
def test_explicit_unsupported_vision_uses_native_400_without_model_switch(
    client: TestClient,
    path: str,
    payload: dict,
    patch_target: str,
    protocol: str,
) -> None:
    router, primary, fallback = _static_capability_router(Operation.CHAT, feature=Feature.VISION)
    if protocol == "openai" and path.endswith("/responses"):
        router, primary, fallback = _static_capability_router(
            Operation.RESPONSES,
            feature=Feature.VISION,
        )

    with patch(patch_target, return_value=router):
        response = client.post(path, json=payload)

    _assert_native_capability_error(response, protocol, parameter="vision")
    _assert_no_provider_work(primary, fallback)


@pytest.mark.parametrize("effort", ["ultra", "none"])
def test_openai_invalid_reasoning_effort_is_native_400(
    client: TestClient,
    effort: str,
) -> None:
    router = MagicMock()
    router.chat_completion = AsyncMock()
    with patch("router_maestro.server.routes.chat.get_router", return_value=router):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": effort,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "reasoning_effort"
    router.chat_completion.assert_not_awaited()


@pytest.mark.parametrize("effort", ["ultra", "none"])
def test_responses_invalid_reasoning_effort_is_native_400(
    client: TestClient,
    effort: str,
) -> None:
    router = MagicMock()
    router.responses_completion = AsyncMock()
    with patch("router_maestro.server.routes.responses.get_router", return_value=router):
        response = client.post(
            "/api/openai/v1/responses",
            json={"model": "m", "input": "hi", "reasoning": {"effort": effort}},
        )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "reasoning_effort"
    router.responses_completion.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize(
    ("reasoning", "parameter"),
    [
        pytest.param({"summary": "detailed"}, "reasoning.summary", id="summary-only"),
        pytest.param(
            {"effort": "low", "summary": "detailed"},
            "reasoning.summary",
            id="effort-and-summary",
        ),
        pytest.param({"future": "value"}, "reasoning.future", id="unknown-field"),
    ],
)
def test_responses_unrepresented_reasoning_fields_are_native_400_before_router(
    client: TestClient,
    stream: bool,
    reasoning: dict,
    parameter: str,
) -> None:
    router = MagicMock()
    router.responses_completion = AsyncMock()
    router.prepare_responses_completion_stream = AsyncMock()
    with patch("router_maestro.server.routes.responses.get_router", return_value=router):
        response = client.post(
            "/api/openai/v1/responses",
            json={
                "model": "m",
                "input": "hi",
                "stream": stream,
                "reasoning": reasoning,
            },
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["param"] == parameter
    assert "data:" not in response.text
    router.responses_completion.assert_not_awaited()
    router.prepare_responses_completion_stream.assert_not_awaited()


def test_responses_unsupported_tools_use_native_400(client: TestClient) -> None:
    provider = CopilotProvider()

    async def validate_with_copilot(request):
        provider.validate_responses_request(request)
        raise AssertionError("provider validation must reject before transport")

    router = MagicMock()
    router.responses_completion = AsyncMock(side_effect=validate_with_copilot)
    with patch("router_maestro.server.routes.responses.get_router", return_value=router):
        response = client.post(
            "/api/openai/v1/responses",
            json={
                "model": "m",
                "input": "hi",
                "tools": [{"type": "web_search"}],
            },
        )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "tools"
    assert error["code"] == "unsupported_parameter"
    router.responses_completion.assert_awaited_once()


def test_responses_stream_provider_rejection_happens_before_sse_commit(
    client: TestClient,
) -> None:
    router = MagicMock()
    router.validate_responses_request = AsyncMock(side_effect=_unsupported("reasoning_effort"))
    router.prepare_responses_completion_stream = AsyncMock(
        side_effect=_unsupported("reasoning_effort")
    )
    router.responses_completion_stream = AsyncMock(
        side_effect=AssertionError("stream must not start")
    )
    with patch("router_maestro.server.routes.responses.get_router", return_value=router):
        response = client.post(
            "/api/openai/v1/responses",
            json={"model": "m", "input": "hi", "stream": True},
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["param"] == "reasoning_effort"


@pytest.mark.parametrize(
    ("path", "payload", "patch_target"),
    [
        (
            "/api/openai/v1/chat/completions",
            {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            "router_maestro.server.routes.chat.get_router",
        ),
        (
            "/v1/messages",
            {
                "model": "m",
                "stream": True,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "router_maestro.server.routes.anthropic.get_router",
        ),
        (
            "/api/gemini/v1beta/models/m:streamGenerateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            "router_maestro.server.routes.gemini.get_router",
        ),
    ],
)
def test_stream_option_rejection_happens_before_sse_commit(
    client: TestClient,
    path: str,
    payload: dict,
    patch_target: str,
) -> None:
    mocked_router = _router(_unsupported("top_k"))
    with patch(patch_target, return_value=mocked_router):
        response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "data:" not in response.text
    mocked_router.chat_completion_stream.assert_not_awaited()


def _experimental_copilot_router(monkeypatch) -> tuple[Router, CopilotProvider]:
    monkeypatch.setenv("ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API", "1")
    provider = CopilotProvider()
    router = Router.__new__(Router)
    ref = ModelRef("github-copilot", "gpt-5.4")
    operation = Operation.CHAT_STREAM
    features = RequestFeatures(responses_bridge=True)
    capabilities = ModelCapabilities(
        model=ref,
        operations={Operation.RESPONSES_STREAM: CapabilitySupport.SUPPORTED},
    )
    plan = RoutePlan(
        operation=operation,
        features=features,
        primary=RouteCandidate(
            model=ref,
            provider=provider,
            capabilities=capabilities,
            evaluated_operation=Operation.RESPONSES_STREAM,
            evaluated_features=features,
            support=CapabilitySupport.SUPPORTED,
            requested_operation=operation,
        ),
        fallbacks=(),
        explicit=True,
    )
    router._plan_completion_route = AsyncMock(return_value=plan)

    async def prepare(request):
        provider.validate_chat_request(request, stream=True)
        return plan

    router.prepare_chat_completion_stream = AsyncMock(side_effect=prepare)
    router.chat_completion_stream = AsyncMock(side_effect=AssertionError("stream must not start"))
    router.get_model_info = AsyncMock(return_value=None)
    router._resolve_provider = AsyncMock(return_value=("github-copilot", "gpt-5.4", provider))
    return router, provider


def test_anthropic_experimental_responses_rejects_before_sse(client, monkeypatch) -> None:
    router, provider = _experimental_copilot_router(monkeypatch)
    provider.responses_completion_stream = AsyncMock(
        side_effect=AssertionError("provider stream must not start")
    )
    with patch("router_maestro.server.routes.anthropic.get_router", return_value=router):
        response = client.post(
            "/v1/messages",
            json={
                "model": "gpt-5.4",
                "stream": True,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
                "stop_sequences": ["END"],
            },
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "event:" not in response.text
    router.chat_completion_stream.assert_not_awaited()
    provider.responses_completion_stream.assert_not_awaited()


def test_gemini_experimental_responses_rejects_before_sse(client, monkeypatch) -> None:
    router, provider = _experimental_copilot_router(monkeypatch)
    provider.responses_completion_stream = AsyncMock(
        side_effect=AssertionError("provider stream must not start")
    )
    with patch("router_maestro.server.routes.gemini.get_router", return_value=router):
        response = client.post(
            "/api/gemini/v1beta/models/gpt-5.4:streamGenerateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"stopSequences": ["END"]},
            },
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert "data:" not in response.text
    router.chat_completion_stream.assert_not_awaited()
    provider.responses_completion_stream.assert_not_awaited()
