"""Protocol-native client-error envelopes for option rejection."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from router_maestro.config import PrioritiesConfig
from router_maestro.providers import (
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.routing.capabilities import (
    Feature,
    Operation,
    ProviderCapabilities,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.server.app import create_app
from router_maestro.server.middleware import REQUEST_ID_HEADER, verify_api_key
from router_maestro.server.protocols import errors as protocol_errors
from router_maestro.server.routes import chat as chat_routes
from router_maestro.server.routes import gemini as gemini_routes
from router_maestro.server.routes import responses as responses_routes
from router_maestro.server.routes.anthropic import router as anthropic_router
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.server.routes.gemini import router as gemini_router
from router_maestro.server.routes.responses import router as responses_router
from router_maestro.server.schemas.gemini import GeminiGenerateContentRequest
from router_maestro.server.schemas.openai import ChatCompletionRequest
from router_maestro.server.schemas.responses import ResponsesRequest
from router_maestro.utils.cache import TTLCache

_PROTOCOL_SURFACES = (
    "openai_chat",
    "openai_responses",
    "anthropic",
    "gemini",
)


def _native_error(response, surface: str) -> dict:
    body = response.json() if hasattr(response, "json") else json.loads(response.body)
    if surface.startswith("openai"):
        return body["error"]
    if surface == "anthropic":
        assert body["type"] == "error"
        return body["error"]
    return body["error"]


@pytest.mark.parametrize(
    ("status_code", "kind", "openai_type", "anthropic_type", "gemini_status"),
    [
        (
            400,
            ProviderFailureKind.CLIENT_REQUEST,
            "invalid_request_error",
            "invalid_request_error",
            "INVALID_ARGUMENT",
        ),
        (
            401,
            ProviderFailureKind.AUTHENTICATION,
            "authentication_error",
            "authentication_error",
            "UNAUTHENTICATED",
        ),
        (
            429,
            ProviderFailureKind.RATE_LIMIT,
            "rate_limit_error",
            "rate_limit_error",
            "RESOURCE_EXHAUSTED",
        ),
        (502, ProviderFailureKind.UPSTREAM_PROTOCOL, "api_error", "api_error", "INTERNAL"),
    ],
)
@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_typed_provider_error_uses_native_protocol_envelope(
    surface: str,
    status_code: int,
    kind: ProviderFailureKind,
    openai_type: str,
    anthropic_type: str,
    gemini_status: str,
) -> None:
    error = ProviderError(
        "Safe provider failure",
        status_code=status_code,
        retryable=status_code >= 429,
        kind=kind,
    )

    response = protocol_errors.protocol_error_response(error, surface)

    assert response.status_code == status_code
    native = _native_error(response, surface)
    assert native["message"] == "Safe provider failure"
    if surface.startswith("openai"):
        assert native["type"] == openai_type
    elif surface == "anthropic":
        assert native["type"] == anthropic_type
    else:
        assert native["code"] == status_code
        assert native["status"] == gemini_status


@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_request_validation_error_is_native_and_does_not_echo_input(
    surface: str,
) -> None:
    private_marker = "private-validation-input"
    error = RequestValidationError(
        [
            {
                "type": "missing",
                "loc": ("body", "model"),
                "msg": "Field required",
                "input": {"secret": private_marker},
            }
        ]
    )

    response = protocol_errors.protocol_error_response(error, surface)

    assert response.status_code == 422
    native = _native_error(response, surface)
    assert native["message"] == "Invalid request"
    assert private_marker not in response.body.decode()
    if surface.startswith("openai"):
        assert native["param"] == "model"
    elif surface == "gemini":
        assert native["details"] == [{"reason": "invalid_request", "parameter": "model"}]


@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_http_exception_is_native_and_preserves_safe_headers(surface: str) -> None:
    error = StarletteHTTPException(
        status_code=401,
        detail="Missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )

    response = protocol_errors.protocol_error_response(error, surface)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    native = _native_error(response, surface)
    assert native["message"] == "Missing API key"


@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_non_string_http_detail_is_not_reflected(surface: str) -> None:
    private_marker = "private-http-detail"
    error = StarletteHTTPException(
        status_code=400,
        detail={"nested": {"secret": private_marker}},
    )

    response = protocol_errors.protocol_error_response(error, surface)

    assert response.status_code == 400
    assert _native_error(response, surface)["message"] == "Invalid request"
    assert private_marker not in response.body.decode()


@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_unexpected_exception_is_generic_and_native(surface: str) -> None:
    private_marker = "private-unexpected-cause"

    response = protocol_errors.protocol_error_response(RuntimeError(private_marker), surface)

    assert response.status_code == 500
    assert _native_error(response, surface)["message"] == "Internal server error"
    assert private_marker not in response.body.decode()


@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_provider_cause_is_not_exposed_by_native_encoder(surface: str) -> None:
    private_marker = "private-provider-cause"
    error = ProviderError(
        "Safe upstream failure",
        status_code=502,
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        cause=RuntimeError(private_marker),
    )

    response = protocol_errors.protocol_error_response(error, surface)

    assert _native_error(response, surface)["message"] == "Safe upstream failure"
    assert private_marker not in response.body.decode()


@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_provider_signal_maps_to_one_allowlisted_header(surface: str) -> None:
    from router_maestro.providers import ProviderFailureSignal

    error = ProviderError(
        "Copilot API error: 400",
        status_code=400,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
        signal=ProviderFailureSignal.COPILOT_BARE_BAD_REQUEST,
        cause=RuntimeError("private-upstream-body"),
    )

    response = protocol_errors.protocol_error_response(error, surface)

    assert response.headers["X-Router-Maestro-Error-Signal"] == "copilot_bare_bad_request"
    assert "private-upstream-body" not in response.body.decode()
    assert "private-upstream-body" not in str(dict(response.headers))


@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_provider_error_without_signal_emits_no_signal_header(surface: str) -> None:
    error = ProviderError(
        "Ordinary provider failure",
        status_code=400,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )

    response = protocol_errors.protocol_error_response(error, surface)

    assert "X-Router-Maestro-Error-Signal" not in response.headers


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        (
            "openai_chat",
            {
                "error": {
                    "message": "Safe late failure",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        ),
        (
            "openai_responses",
            {"code": "rate_limit_exceeded", "message": "Safe late failure"},
        ),
        (
            "anthropic",
            {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "Safe late failure"},
            },
        ),
        (
            "gemini",
            {
                "error": {
                    "code": 429,
                    "message": "Safe late failure",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [],
                }
            },
        ),
    ],
)
def test_postcommit_error_data_is_protocol_native(surface: str, expected: dict) -> None:
    error = ProviderError(
        "Safe late failure",
        status_code=429,
        kind=ProviderFailureKind.RATE_LIMIT,
    )

    data = protocol_errors.postcommit_error_data(error, surface)

    assert data == expected


def test_openai_chat_postcommit_overload_preserves_error_code() -> None:
    error = ProviderError(
        "Server overloaded, retry",
        status_code=529,
        kind=ProviderFailureKind.RATE_LIMIT,
    )

    data = protocol_errors.postcommit_error_data(error, "openai_chat")

    assert data == {
        "error": {
            "message": "Server overloaded, retry",
            "type": "rate_limit_error",
            "code": "overloaded",
        }
    }


@pytest.mark.parametrize("surface", _PROTOCOL_SURFACES)
def test_postcommit_unexpected_error_is_generic(surface: str) -> None:
    marker = "private-postcommit-error"

    data = protocol_errors.postcommit_error_data(RuntimeError(marker), surface)

    assert marker not in json.dumps(data)
    assert "Internal server error" in json.dumps(data)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/openai/v1/responses", "openai_responses"),
        ("/api/openai/v1/responses/not-found", "openai_responses"),
        ("/api/openai/v1/chat/completions", "openai_chat"),
        ("/api/openai/v1/not-found", "openai_chat"),
        ("/v1/messages", "anthropic"),
        ("/v1/messages/count_tokens", "anthropic"),
        ("/api/anthropic/beta/v1/messages", "anthropic"),
        ("/api/gemini/v1beta/models/m:generateContent", "gemini"),
        ("/api/admin/models", None),
        ("/health", None),
        ("/unmatched", None),
    ],
)
def test_protocol_surface_is_classified_from_path(path: str, expected: str | None) -> None:
    assert protocol_errors.protocol_surface_for_path(path) == expected


_APP_PROTOCOL_CASES = (
    (
        "openai_chat",
        "/api/openai/v1/chat/completions",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    ),
    (
        "openai_responses",
        "/api/openai/v1/responses",
        {"model": "m", "input": "hi"},
    ),
    (
        "anthropic",
        "/v1/messages",
        {
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ),
    (
        "gemini",
        "/api/gemini/v1beta/models/m:generateContent",
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    ),
)


@pytest.fixture
def app_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "server-secret")
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize(("surface", "path", "_payload"), _APP_PROTOCOL_CASES)
def test_app_schema_validation_uses_native_envelope(
    app_client: TestClient,
    surface: str,
    path: str,
    _payload: dict,
) -> None:
    private_marker = "private-validation-body"
    invalid_body = (
        {"contents": private_marker}
        if surface == "gemini"
        else {"unexpected": {"secret": private_marker}}
    )

    response = app_client.post(
        path,
        json=invalid_body,
        headers={"Authorization": "Bearer server-secret"},
    )

    assert response.status_code == 422
    assert _native_error(response, surface)["message"] == "Invalid request"
    assert private_marker not in response.text


@pytest.mark.parametrize(("surface", "path", "payload"), _APP_PROTOCOL_CASES)
@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-secret"}],
    ids=["missing", "invalid"],
)
def test_app_auth_failure_uses_native_envelope_and_preserves_challenge(
    app_client: TestClient,
    surface: str,
    path: str,
    payload: dict,
    headers: dict[str, str],
) -> None:
    response = app_client.post(path, json=payload, headers=headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert _native_error(response, surface)["message"]


@pytest.mark.parametrize(
    ("surface", "path"),
    [
        ("openai_chat", "/api/openai/v1/not-found"),
        ("openai_responses", "/api/openai/v1/responses/not-found"),
        ("anthropic", "/api/anthropic/v1/not-found"),
        ("gemini", "/api/gemini/v1beta/not-found"),
    ],
)
def test_app_namespace_404_uses_native_envelope(
    app_client: TestClient,
    surface: str,
    path: str,
) -> None:
    response = app_client.post(
        path,
        json={},
        headers={"Authorization": "Bearer server-secret"},
    )

    assert response.status_code == 404
    assert _native_error(response, surface)["message"] == "Not Found"
    assert "detail" not in response.json()


@pytest.mark.parametrize(("surface", "path", "_payload"), _APP_PROTOCOL_CASES)
def test_app_wrong_method_uses_native_envelope(
    app_client: TestClient,
    surface: str,
    path: str,
    _payload: dict,
) -> None:
    response = app_client.get(
        path,
        headers={"Authorization": "Bearer server-secret"},
    )

    assert response.status_code == 405
    assert _native_error(response, surface)["message"] == "Method Not Allowed"
    assert "detail" not in response.json()


@pytest.mark.parametrize(
    ("surface", "path"),
    [
        ("openai_chat", "/api/openai/v1/boom"),
        ("openai_responses", "/api/openai/v1/responses/boom"),
        ("anthropic", "/api/anthropic/v1/boom"),
        ("gemini", "/api/gemini/v1beta/boom"),
    ],
)
def test_app_unexpected_failure_is_native_with_one_request_id(
    monkeypatch,
    surface: str,
    path: str,
) -> None:
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "server-secret")
    app = create_app()

    async def boom() -> None:
        raise RuntimeError("private-unexpected-marker")

    app.add_api_route(path, boom, methods=["GET"], dependencies=[Depends(verify_api_key)])
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        path,
        headers={
            "Authorization": "Bearer server-secret",
            REQUEST_ID_HEADER: "req-native-error",
        },
    )

    assert response.status_code == 500
    assert _native_error(response, surface)["message"] == "Internal server error"
    assert "private-unexpected-marker" not in response.text
    assert response.headers.get_list(REQUEST_ID_HEADER) == ["req-native-error"]


def test_app_nonprotocol_provider_error_keeps_plain_500_with_one_request_id(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "server-secret")
    app = create_app()

    async def provider_boom() -> None:
        raise ProviderError(
            "Safe provider failure",
            status_code=502,
            kind=ProviderFailureKind.UPSTREAM_STATUS,
        )

    app.add_api_route("/provider-boom", provider_boom, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        "/provider-boom",
        headers={REQUEST_ID_HEADER: "req-provider-boom"},
    )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers.get_list(REQUEST_ID_HEADER) == ["req-provider-boom"]


def test_app_nonprotocol_responses_keep_existing_default_shapes(app_client: TestClient) -> None:
    health = app_client.get("/health")
    unmatched = app_client.get("/not-a-protocol-path")
    admin = app_client.get("/api/admin/models")

    assert health.status_code == 200
    assert health.json() == {"status": "healthy"}
    assert unmatched.status_code == 404
    assert unmatched.json() == {"detail": "Not Found"}
    assert admin.status_code == 401
    assert admin.json() == {
        "detail": "Missing API key. Use 'Authorization: Bearer <api_key>' header."
    }
    assert admin.headers["WWW-Authenticate"] == "Bearer"


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
def test_unrepresented_openai_semantic_option_reaches_router_not_400(
    client: TestClient,
    path: str,
    base_payload: dict,
    parameter: str,
    value: object,
    router_method: str,
    stream: bool,
) -> None:
    """Options Router-Maestro does not model (``response_format``, ``n``, ``store``,
    ``include``) are NOT rejected at the route. A transparent proxy forwards or
    ignores them, so the request reaches routing instead of a native 400."""
    sentinel = ProviderError(
        "reached routing sentinel",
        status_code=503,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )
    model_router = MagicMock()
    model_router.chat_completion = AsyncMock(side_effect=sentinel)
    model_router.prepare_chat_completion_stream = AsyncMock(side_effect=sentinel)
    model_router.chat_completion_stream = AsyncMock(side_effect=sentinel)
    model_router.responses_completion = AsyncMock(side_effect=sentinel)
    model_router.prepare_responses_completion_stream = AsyncMock(side_effect=sentinel)
    model_router.responses_completion_stream = AsyncMock(side_effect=sentinel)
    patch_target = (
        "router_maestro.server.routes.chat.get_router"
        if router_method == "chat_completion"
        else "router_maestro.server.routes.responses.get_router"
    )
    payload = {**base_payload, "stream": stream, parameter: value}

    with patch(patch_target, return_value=model_router):
        response = client.post(path, json=payload)

    assert response.status_code != 400, response.text
    assert "Unsupported request option" not in response.text


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
def test_unrepresented_gemini_option_reaches_router_not_400(
    client: TestClient,
    path: str,
    parameter: str,
    payload_option: dict,
) -> None:
    """Unknown Gemini options are forwarded/ignored, not rejected at the route."""
    sentinel = ProviderError(
        "reached routing sentinel",
        status_code=503,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )
    model_router = MagicMock()
    model_router.chat_completion = AsyncMock(side_effect=sentinel)
    model_router.prepare_chat_completion_stream = AsyncMock(side_effect=sentinel)
    model_router.chat_completion_stream = AsyncMock(side_effect=sentinel)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        **payload_option,
    }

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(path, json=payload)

    assert response.status_code != 400, response.text
    assert "Unsupported request option" not in response.text


def test_unrepresented_gemini_count_tokens_option_is_ignored_not_400(client: TestClient) -> None:
    """count_tokens estimates locally; an unknown option must not 400 the request."""
    response = client.post(
        "/api/gemini/v1beta/models/m:countTokens",
        json={
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"unknownOption": True},
        },
    )

    assert response.status_code != 400, response.text
    assert "Unsupported request option" not in response.text


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
def test_nested_unrepresented_gemini_option_reaches_router_not_400(
    client: TestClient,
    path: str,
    payload_option: dict,
    parameter: str,
) -> None:
    """Nested Gemini fields Router-Maestro does not model (``allowedFunctionNames``,
    ``fileData``) are forwarded/ignored rather than rejected at the route."""
    sentinel = ProviderError(
        "reached routing sentinel",
        status_code=503,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )
    model_router = MagicMock()
    model_router.chat_completion = AsyncMock(side_effect=sentinel)
    model_router.prepare_chat_completion_stream = AsyncMock(side_effect=sentinel)
    model_router.chat_completion_stream = AsyncMock(side_effect=sentinel)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        **payload_option,
    }

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(path, json=payload)

    assert response.status_code != 400, response.text
    assert "Unsupported request option" not in response.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("parameter", "payload_option"),
    [
        ("frequency_penalty", {"frequency_penalty": 0.2}),
        ("output_config.format", {"output_config": {"format": {"type": "json_schema"}}}),
    ],
)
def test_unrepresented_anthropic_option_reaches_router_not_400(
    client: TestClient,
    stream: bool,
    parameter: str,
    payload_option: dict,
) -> None:
    """Unknown options on the standard Anthropic route are forwarded/ignored,
    not rejected. The request reaches routing instead of a native 400."""
    sentinel = ProviderError(
        "reached routing sentinel",
        status_code=503,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )
    model_router = MagicMock()
    model_router.get_model_info = AsyncMock(return_value=None)
    model_router.chat_completion = AsyncMock(side_effect=sentinel)
    model_router.plan_chat_completion = AsyncMock(side_effect=sentinel)
    model_router.prepare_chat_completion_stream = AsyncMock(side_effect=sentinel)
    model_router.chat_completion_stream = AsyncMock(side_effect=sentinel)
    payload = {
        "model": "m",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        **payload_option,
    }

    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code != 400, response.text
    assert "Unsupported request option" not in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_nested_unrepresented_anthropic_tool_choice_reaches_router_not_400(
    client: TestClient,
    stream: bool,
) -> None:
    """A nested unmodeled tool_choice field is forwarded/ignored, not rejected."""
    sentinel = ProviderError(
        "reached routing sentinel",
        status_code=503,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )
    model_router = MagicMock()
    model_router.get_model_info = AsyncMock(return_value=None)
    model_router.chat_completion = AsyncMock(side_effect=sentinel)
    model_router.plan_chat_completion = AsyncMock(side_effect=sentinel)
    model_router.prepare_chat_completion_stream = AsyncMock(side_effect=sentinel)
    model_router.chat_completion_stream = AsyncMock(side_effect=sentinel)
    payload = {
        "model": "m",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }

    with patch("router_maestro.server.routes.anthropic.get_router", return_value=model_router):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code != 400, response.text
    assert "Unsupported request option" not in response.text


def test_nested_unrepresented_anthropic_count_tokens_option_is_ignored_not_400(
    client: TestClient,
) -> None:
    """count_tokens tolerates an unmodeled nested field instead of returning 400."""
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
        return_value="github-copilot",
    ):
        response = client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code != 400, response.text
    assert "Unsupported request option" not in response.text


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


def _reaches_routing_router() -> MagicMock:
    """A router that raises a sentinel as soon as the route reaches provider work.

    The test asserts the request got PAST the option gate (no 400 option error).
    Any downstream call short-circuits with a distinct sentinel so we can prove
    routing was reached without reproducing the full response-encoding path.
    """
    sentinel = ProviderError(
        "reached routing sentinel",
        status_code=503,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )
    model_router = MagicMock()
    model_router.get_model_info = AsyncMock(return_value=None)
    for method in (
        "chat_completion",
        "plan_chat_completion",
        "prepare_chat_completion_stream",
        "chat_completion_stream",
        "responses_completion",
        "prepare_responses_completion_stream",
        "responses_completion_stream",
    ):
        setattr(model_router, method, AsyncMock(side_effect=sentinel))
    return model_router


@pytest.mark.parametrize(
    ("path", "base_payload", "option", "patch_target"),
    [
        (
            "/api/openai/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            {"context_management": {"enabled": True}},
            "router_maestro.server.routes.chat.get_router",
        ),
        (
            "/v1/messages",
            {
                "model": "m",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
            {"context_management": {"edits": []}},
            "router_maestro.server.routes.anthropic.get_router",
        ),
        (
            "/api/openai/v1/responses",
            {"model": "m", "input": "hi"},
            {"store": True},
            "router_maestro.server.routes.responses.get_router",
        ),
    ],
)
def test_unmodeled_client_option_passes_through_instead_of_400(
    client: TestClient,
    path: str,
    base_payload: dict,
    option: dict,
    patch_target: str,
) -> None:
    """Options Router-Maestro does not model as typed fields (e.g. ``context_management``
    from Claude Code, ``store`` from Codex) must NOT be hard-rejected. A transparent
    proxy forwards or ignores them; it does not 400 on an unknown top-level key."""
    model_router = _reaches_routing_router()
    payload = {**base_payload, **option}

    with patch(patch_target, return_value=model_router):
        response = client.post(path, json=payload)

    # The request must reach routing, not be short-circuited by an option-gate 400.
    assert "Unsupported request option" not in response.text
    assert "is not supported by the beta" not in response.text
    # It got past the gate and hit the sentinel provider failure (502/503), not a 400.
    assert response.status_code != 400, response.text


def _routing_error(
    status_code: int = 404,
    kind: ProviderFailureKind = ProviderFailureKind.CLIENT_REQUEST,
) -> ProviderError:
    return ProviderError(
        "Safe routed failure",
        status_code=status_code,
        retryable=False,
        kind=kind,
    )


@pytest.mark.parametrize(
    ("surface", "path", "payload", "patch_target"),
    [
        (
            "openai_chat",
            "/api/openai/v1/chat/completions",
            {"model": "missing", "messages": [{"role": "user", "content": "hi"}]},
            "router_maestro.server.routes.chat.get_router",
        ),
        (
            "openai_responses",
            "/api/openai/v1/responses",
            {"model": "missing", "input": "hi"},
            "router_maestro.server.routes.responses.get_router",
        ),
        (
            "anthropic",
            "/v1/messages",
            {
                "model": "missing",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "router_maestro.server.routes.anthropic.get_router",
        ),
        (
            "gemini",
            "/api/gemini/v1beta/models/missing:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            "router_maestro.server.routes.gemini.get_router",
        ),
    ],
)
@pytest.mark.parametrize(
    ("status_code", "kind"),
    [
        (400, ProviderFailureKind.CLIENT_REQUEST),
        (401, ProviderFailureKind.AUTHENTICATION),
        (429, ProviderFailureKind.RATE_LIMIT),
        (502, ProviderFailureKind.UPSTREAM_PROTOCOL),
    ],
)
def test_nonstream_provider_error_uses_native_envelope(
    client: TestClient,
    surface: str,
    path: str,
    payload: dict,
    patch_target: str,
    status_code: int,
    kind: ProviderFailureKind,
) -> None:
    error = _routing_error(status_code, kind)
    router = MagicMock()
    router.chat_completion = AsyncMock(side_effect=error)
    router.responses_completion = AsyncMock(side_effect=error)
    router.plan_chat_completion = AsyncMock(side_effect=error)
    router.prepare_planned_chat_completion = MagicMock(side_effect=error)
    router.get_model_info = AsyncMock(return_value=None)
    router._resolve_provider = AsyncMock(return_value=("test", "missing", MagicMock()))

    with patch(patch_target, return_value=router):
        response = client.post(path, json=payload)

    assert response.status_code == status_code
    native = _native_error(response, surface)
    assert native["message"] == "Safe routed failure"
    assert "detail" not in response.json()


@pytest.mark.parametrize(
    ("surface", "path", "payload", "patch_target", "prepare_method", "open_method"),
    [
        (
            "openai_chat",
            "/api/openai/v1/chat/completions",
            {
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "router_maestro.server.routes.chat.get_router",
            "prepare_chat_completion_stream",
            "chat_completion_stream",
        ),
        (
            "openai_responses",
            "/api/openai/v1/responses",
            {"model": "m", "input": "hi", "stream": True},
            "router_maestro.server.routes.responses.get_router",
            "prepare_responses_completion_stream",
            "responses_completion_stream",
        ),
        (
            "gemini",
            "/api/gemini/v1beta/models/m:streamGenerateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            "router_maestro.server.routes.gemini.get_router",
            "prepare_chat_completion_stream",
            "chat_completion_stream",
        ),
    ],
)
@pytest.mark.parametrize(
    ("status_code", "kind", "message"),
    [
        (401, ProviderFailureKind.AUTHENTICATION, "Provider authentication required"),
        (502, ProviderFailureKind.UPSTREAM_PROTOCOL, "Provider returned an empty stream"),
        (503, ProviderFailureKind.UPSTREAM_STATUS, "All fallback providers failed"),
    ],
)
def test_stream_provider_open_failure_is_native_json_before_sse(
    client: TestClient,
    surface: str,
    path: str,
    payload: dict,
    patch_target: str,
    prepare_method: str,
    open_method: str,
    status_code: int,
    kind: ProviderFailureKind,
    message: str,
) -> None:
    router = MagicMock()
    setattr(router, prepare_method, AsyncMock(return_value=object()))
    setattr(
        router,
        open_method,
        AsyncMock(
            side_effect=ProviderError(
                message,
                status_code=status_code,
                retryable=status_code >= 500,
                kind=kind,
            )
        ),
    )

    with patch(patch_target, return_value=router):
        response = client.post(path, json=payload)

    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    assert _native_error(response, surface)["message"] == message
    assert "data:" not in response.text
    getattr(router, open_method).assert_awaited_once()


class _EndpointOwnedStream:
    def __init__(self) -> None:
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_count += 1


def _sse_event_types(text: str) -> list[str]:
    return [
        line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")
    ]


def test_standard_anthropic_open_failure_keeps_eager_ping_compatibility(
    client: TestClient,
) -> None:
    router = MagicMock()
    candidate = MagicMock()
    candidate.model = ModelRef("test", "m")
    route_plan = MagicMock(primary=candidate, prevalidation_fallbacks=())
    prepared_plan = object()
    router.plan_chat_completion = AsyncMock(return_value=route_plan)
    router.prepare_planned_chat_completion = MagicMock(return_value=prepared_plan)
    router.chat_completion_stream = AsyncMock(
        side_effect=ProviderError(
            "Provider authentication required",
            status_code=401,
            kind=ProviderFailureKind.AUTHENTICATION,
        )
    )

    with (
        patch("router_maestro.server.routes.anthropic.get_router", return_value=router),
        patch(
            "router_maestro.server.routes.anthropic._apply_thinking_budget",
            AsyncMock(side_effect=lambda _router, request, _model, **_kwargs: request),
        ),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "m",
                "stream": True,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    event_types = _sse_event_types(response.text)
    assert response.status_code == 200
    assert event_types[0] == "ping"
    assert event_types.count("error") == 1
    assert "message_start" not in event_types
    assert "message_stop" not in event_types


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "gemini"])
async def test_primed_stream_closes_once_if_response_handoff_fails(
    protocol: str,
    monkeypatch,
) -> None:
    stream = _EndpointOwnedStream()
    router = MagicMock()
    handoff_error = RuntimeError("response construction failed")

    if protocol == "chat":
        router.prepare_chat_completion_stream = AsyncMock(return_value=object())
        router.chat_completion_stream = AsyncMock(return_value=(stream, "provider"))
        monkeypatch.setattr(chat_routes, "get_router", lambda: router)
        monkeypatch.setattr(
            chat_routes,
            "sse_streaming_response",
            MagicMock(side_effect=handoff_error),
        )
        call = chat_routes.chat_completions(
            ChatCompletionRequest(
                model="m",
                stream=True,
                messages=[{"role": "user", "content": "hi"}],
            )
        )
    elif protocol == "responses":
        router.prepare_responses_completion_stream = AsyncMock(return_value=object())
        router.responses_completion_stream = AsyncMock(return_value=(stream, "provider"))
        monkeypatch.setattr(responses_routes, "get_router", lambda: router)
        monkeypatch.setattr(
            responses_routes,
            "sse_streaming_response",
            MagicMock(side_effect=handoff_error),
        )
        call = responses_routes.create_response(
            ResponsesRequest(model="m", input="hi", stream=True)
        )
    else:
        router.prepare_chat_completion_stream = AsyncMock(return_value=object())
        router.chat_completion_stream = AsyncMock(return_value=(stream, "provider"))
        monkeypatch.setattr(gemini_routes, "get_router", lambda: router)
        monkeypatch.setattr(
            gemini_routes,
            "sse_streaming_response",
            MagicMock(side_effect=handoff_error),
        )
        call = gemini_routes.stream_generate_content(
            "m",
            GeminiGenerateContentRequest(contents=[{"role": "user", "parts": [{"text": "hi"}]}]),
        )

    with pytest.raises(RuntimeError, match="response construction failed"):
        await call

    assert stream.close_count == 1


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
    "reasoning",
    [
        pytest.param({"summary": "detailed"}, id="summary-only"),
        pytest.param({"effort": "low", "summary": "detailed"}, id="effort-and-summary"),
        pytest.param({"future": "value"}, id="unknown-field"),
        pytest.param({"effort": "low", "context": {"foo": "bar"}}, id="effort-and-context"),
    ],
)
def test_responses_unrepresented_reasoning_fields_reach_router_not_400(
    client: TestClient,
    stream: bool,
    reasoning: dict,
) -> None:
    """Unknown ``reasoning`` siblings (Codex sends ``reasoning.context`` on gpt-5.6,
    ``summary``, etc.) must NOT be rejected. Router-Maestro extracts ``effort`` and
    ignores the rest, so the request reaches routing instead of a native 400."""
    sentinel = ProviderError(
        "reached routing sentinel",
        status_code=503,
        retryable=False,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )
    router = MagicMock()
    router.responses_completion = AsyncMock(side_effect=sentinel)
    router.prepare_responses_completion_stream = AsyncMock(side_effect=sentinel)
    router.responses_completion_stream = AsyncMock(side_effect=sentinel)
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

    assert response.status_code != 400, response.text
    assert "Invalid request option" not in response.text
    assert "Unsupported request option" not in response.text


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
