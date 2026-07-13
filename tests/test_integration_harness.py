"""Tests for the local-only integration test harness."""

import ast
import importlib
import inspect
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from router_maestro.auth import AuthManager, AuthStorage, OAuthCredential
from router_maestro.auth.github_oauth import CopilotTokenResponse
from router_maestro.providers import CopilotProvider, ModelInfo
from router_maestro.routing.capabilities import Operation

ROOT = Path(__file__).resolve().parents[1]


def test_integration_tests_are_outside_default_pytest_tree():
    """Integration tests should not be discovered by the default tests/ run."""
    integration_dir = ROOT / "integration_tests"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert integration_dir.is_dir()
    assert 'testpaths = ["tests"]' in pyproject


def test_makefile_exposes_explicit_integration_test_target():
    """Local live-backend tests should have an explicit Makefile entrypoint."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "integration-test:" in makefile
    assert "uv run pytest integration_tests/ -v" in makefile


def test_integration_model_matrix_defaults_to_full():
    """The default integration suite should cover the full Copilot model matrix."""
    conftest = (ROOT / "integration_tests" / "conftest.py").read_text(encoding="utf-8")

    assert "DEFAULT_MAX_MODEL_MATRIX = 0" in conftest


def test_model_matrix_payload_has_no_optional_feature_knobs():
    """The broad smoke matrix should only validate baseline invocation."""
    conftest = importlib.import_module("integration_tests.conftest")

    payload = conftest.model_matrix_chat_payload("github-copilot/gemini-2.5-pro")

    assert payload["max_tokens"] >= 512
    assert "reasoning_effort" not in payload


def test_anthropic_compat_payload_uses_top_p_without_temperature():
    conftest = importlib.import_module("integration_tests.conftest")

    payload = conftest.anthropic_compat_payload(
        "github-copilot/claude-sonnet-4.6",
        stream=True,
    )

    assert payload["top_p"] == 1
    assert "temperature" not in payload
    assert payload["stream"] is True


def test_gemini_stream_model_identity_accepts_sparse_matching_versions():
    gemini_paths = importlib.import_module("integration_tests.test_live_gemini_paths")
    expected = "github-copilot/gemini-2.5-pro"

    gemini_paths._assert_gemini_stream_model_identity(
        [
            {"candidates": [{"content": {"parts": [{"text": "pong"}]}}]},
            {"modelVersion": expected, "usageMetadata": {"totalTokenCount": 2}},
        ],
        expected,
    )


@pytest.mark.parametrize(
    "final_payload",
    [
        {"usageMetadata": {"totalTokenCount": 2}},
        {
            "modelVersion": "github-copilot/gemini-2.0-flash",
            "usageMetadata": {"totalTokenCount": 2},
        },
    ],
    ids=["missing-final-version", "wrong-final-version"],
)
def test_gemini_stream_model_identity_requires_final_usage_identity(final_payload):
    gemini_paths = importlib.import_module("integration_tests.test_live_gemini_paths")
    expected = "github-copilot/gemini-2.5-pro"

    with pytest.raises(AssertionError, match="final usage payload modelVersion mismatch"):
        gemini_paths._assert_gemini_stream_model_identity(
            [
                {"modelVersion": expected},
                final_payload,
            ],
            expected,
        )


def test_gemini_stream_model_identity_rejects_completely_missing_version():
    gemini_paths = importlib.import_module("integration_tests.test_live_gemini_paths")

    with pytest.raises(AssertionError, match="never reported modelVersion"):
        gemini_paths._assert_gemini_stream_model_identity(
            [{"candidates": [{"content": {"parts": [{"text": "pong"}]}}]}],
            "github-copilot/gemini-2.5-pro",
        )


def test_gemini_stream_model_identity_rejects_mismatch():
    gemini_paths = importlib.import_module("integration_tests.test_live_gemini_paths")
    expected = "github-copilot/gemini-2.5-pro"

    with pytest.raises(AssertionError, match="modelVersion mismatch"):
        gemini_paths._assert_gemini_stream_model_identity(
            [
                {"modelVersion": expected},
                {"modelVersion": "github-copilot/gemini-2.0-flash"},
            ],
            expected,
        )


def test_gemini_model_path_preserves_qualified_public_identity():
    conftest = importlib.import_module("integration_tests.conftest")

    assert (
        conftest.gemini_model_path("github-copilot/shared-model") == "github-copilot/shared-model"
    )
    assert conftest.bare_model("github-copilot/shared-model") == "shared-model"


def test_live_catalog_public_id_preserves_namespaced_upstream_provenance():
    conftest = importlib.import_module("integration_tests.conftest")

    assert (
        conftest._provider_model_id("openrouter", "openrouter/auto") == "openrouter/openrouter/auto"
    )


def test_copilot_models_are_read_from_public_models_endpoint():
    conftest = importlib.import_module("integration_tests.conftest")
    public_id = "github-copilot/catalog-model"
    catalog = {
        public_id: ModelInfo(
            id="catalog-model",
            name="Catalog Model",
            provider="github-copilot",
        )
    }
    client = Mock()
    client.get.return_value = httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"id": public_id, "object": "model", "owned_by": "github-copilot"},
                {"id": "openai/other", "object": "model", "owned_by": "openai"},
            ],
        },
        request=httpx.Request("GET", "http://router.test/api/openai/v1/models"),
    )

    models = conftest.copilot_models.__wrapped__(client, catalog)

    assert models == [public_id]
    client.get.assert_called_once_with("/api/openai/v1/models")


def test_copilot_models_reject_duplicate_public_ids():
    conftest = importlib.import_module("integration_tests.conftest")
    public_id = "github-copilot/catalog-model"
    catalog = {
        public_id: ModelInfo(
            id="catalog-model",
            name="Catalog Model",
            provider="github-copilot",
        )
    }
    client = Mock()
    client.get.return_value = httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"id": public_id, "object": "model", "owned_by": "github-copilot"},
                {"id": public_id, "object": "model", "owned_by": "github-copilot"},
            ],
        },
        request=httpx.Request("GET", "http://router.test/api/openai/v1/models"),
    )

    with pytest.raises(pytest.fail.Exception, match="duplicate Copilot public IDs"):
        conftest.copilot_models.__wrapped__(client, catalog)


def test_copilot_catalog_rejects_duplicate_public_ids(monkeypatch):
    conftest = importlib.import_module("integration_tests.conftest")
    manager = _manager_with_copilot_credential(
        OAuthCredential(
            refresh="github-token",
            access="cached-copilot-token",
            expires=int(time.time()) + 3600,
            api_endpoint="https://copilot.example",
        )
    )
    duplicate = ModelInfo(
        id="catalog-model",
        name="Catalog Model",
        provider="github-copilot",
    )
    provider = CopilotProvider()
    provider.auth_manager = manager
    provider.list_models = AsyncMock(return_value=[duplicate, duplicate])  # type: ignore[method-assign]
    provider.close = AsyncMock()  # type: ignore[method-assign]
    monkeypatch.setattr(conftest, "AuthManager", lambda: manager)
    monkeypatch.setattr(conftest, "CopilotProvider", lambda: provider)

    with pytest.raises(pytest.fail.Exception, match="duplicate public IDs"):
        conftest.copilot_catalog.__wrapped__()


def _manager_with_copilot_credential(credential: OAuthCredential) -> AuthManager:
    manager = AuthManager.__new__(AuthManager)
    manager.storage = AuthStorage()
    manager.storage.set("github-copilot", credential)
    manager.save = Mock()  # type: ignore[method-assign]
    return manager


def _catalog_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": "catalog-model",
                    "name": "Catalog Model",
                    "capabilities": {
                        "type": "chat",
                        "supports": {
                            "reasoning_effort": {
                                "values": [
                                    "none",
                                    {"value": "none"},
                                    {"value": "minimal"},
                                    {"value": "minimal"},
                                ]
                            }
                        },
                    },
                    "supported_endpoints": ["/chat/completions"],
                }
            ]
        },
        request=httpx.Request("GET", "https://copilot.example/models"),
    )


def test_copilot_catalog_reuses_valid_cached_credential_without_mint(monkeypatch):
    conftest = importlib.import_module("integration_tests.conftest")
    manager = _manager_with_copilot_credential(
        OAuthCredential(
            refresh="github-token",
            access="cached-copilot-token",
            expires=int(time.time()) + 3600,
            api_endpoint="https://copilot.example",
        )
    )
    provider = CopilotProvider()
    provider.auth_manager = manager
    provider._send_with_auth_retry = AsyncMock(return_value=_catalog_response())
    provider.close = AsyncMock()  # type: ignore[method-assign]
    mint = AsyncMock(
        return_value=CopilotTokenResponse(
            token="unexpected-new-token",
            expires_at=int(time.time()) + 7200,
            refresh_in=3600,
            api_endpoint="https://unexpected.example",
        )
    )
    monkeypatch.setattr(conftest, "AuthManager", lambda: manager)
    monkeypatch.setattr(conftest, "CopilotProvider", lambda: provider)

    with patch("router_maestro.providers.copilot.get_copilot_token", new=mint):
        catalog = conftest.copilot_catalog.__wrapped__()

    assert list(catalog) == ["github-copilot/catalog-model"]
    assert catalog["github-copilot/catalog-model"].reasoning_effort_values == [
        "none",
        "minimal",
    ]
    assert conftest.model_supports_reasoning(catalog["github-copilot/catalog-model"])
    assert "none" not in conftest.OPENAI_REASONING_EFFORTS
    mint.assert_not_awaited()
    manager.save.assert_not_called()
    assert provider._cached_token == "cached-copilot-token"
    assert provider._token_expires == manager.get_credential("github-copilot").expires
    assert provider._api_base == "https://copilot.example"


def test_copilot_catalog_allows_existing_refresh_for_expired_credential(monkeypatch):
    conftest = importlib.import_module("integration_tests.conftest")
    manager = _manager_with_copilot_credential(
        OAuthCredential(
            refresh="github-token",
            access="expired-copilot-token",
            expires=int(time.time()) - 1,
            api_endpoint="https://copilot.example",
        )
    )
    provider = CopilotProvider()
    provider.auth_manager = manager
    provider._send_with_auth_retry = AsyncMock(return_value=_catalog_response())
    provider.close = AsyncMock()  # type: ignore[method-assign]
    mint = AsyncMock(
        return_value=CopilotTokenResponse(
            token="fresh-copilot-token",
            expires_at=int(time.time()) + 7200,
            refresh_in=3600,
            api_endpoint="https://copilot.example",
        )
    )
    monkeypatch.setattr(conftest, "AuthManager", lambda: manager)
    monkeypatch.setattr(conftest, "CopilotProvider", lambda: provider)

    with patch("router_maestro.providers.copilot.get_copilot_token", new=mint):
        catalog = conftest.copilot_catalog.__wrapped__()

    assert list(catalog) == ["github-copilot/catalog-model"]
    mint.assert_awaited_once()
    assert provider._cached_token == "fresh-copilot-token"
    manager.save.assert_not_called()
    assert provider.auth_manager is not manager
    assert manager.get_credential("github-copilot").access == "expired-copilot-token"
    assert provider.auth_manager.get_credential("github-copilot").access == "fresh-copilot-token"


@pytest.mark.parametrize(
    ("supported_endpoints", "expected_name"),
    [
        (("/responses",), "RESPONSES"),
        (("/chat/completions",), "CHAT"),
        (("/chat/completions", "/responses"), "RESPONSES"),
        (None, "UNKNOWN"),
        (("ws:/responses",), "UNSUPPORTED"),
        (("/v1/messages",), "UNSUPPORTED"),
        (("/future/native",), "UNSUPPORTED"),
        (("", " ", "\t"), "UNSUPPORTED"),
        ((), "UNSUPPORTED"),
    ],
)
def test_live_endpoint_selection_uses_exact_raw_http_endpoints(
    supported_endpoints,
    expected_name,
):
    conftest = importlib.import_module("integration_tests.conftest")
    model = "github-copilot/catalog-model"
    info = ModelInfo(
        id="catalog-model",
        name="Catalog Model",
        provider="github-copilot",
        supported_endpoints=supported_endpoints,
        operation_capabilities={Operation.CHAT: True, Operation.RESPONSES: True},
    )

    endpoint = conftest.select_live_http_endpoint(model, {model: info})

    assert endpoint is getattr(conftest.LiveHttpEndpoint, expected_name, None)


@pytest.mark.parametrize(
    "capabilities",
    [
        {Operation.RESPONSES: True, Operation.CHAT: False},
        {Operation.RESPONSES: False, Operation.CHAT: True},
        {Operation.RESPONSES: True, Operation.CHAT: True},
        {Operation.RESPONSES: False, Operation.CHAT: False},
        {Operation.RESPONSES: False},
        {Operation.CHAT: False},
        {Operation.RESPONSES: None, Operation.CHAT: None},
        {Operation.NATIVE_ANTHROPIC: True},
        {},
    ],
)
def test_live_endpoint_selection_does_not_treat_derived_operations_as_raw_http_contract(
    capabilities,
):
    conftest = importlib.import_module("integration_tests.conftest")
    model = "github-copilot/catalog-model"
    info = ModelInfo(
        id="catalog-model",
        name="Catalog Model",
        provider="github-copilot",
        operation_capabilities=capabilities,
    )

    endpoint = conftest.select_live_http_endpoint(model, {model: info})

    assert endpoint is getattr(conftest.LiveHttpEndpoint, "UNKNOWN", None)


def _chat_matrix_success(model: str = "github-copilot/heuristic-profile") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        request=httpx.Request("POST", "https://router.example/api/openai/v1/chat/completions"),
    )


def _responses_matrix_success(model: str = "github-copilot/unknown-profile") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
        request=httpx.Request("POST", "https://router.example/api/openai/v1/responses"),
    )


def _unsupported_api_response() -> httpx.Response:
    return httpx.Response(
        400,
        json={"error": {"code": "unsupported_api_for_model"}},
        request=httpx.Request("POST", "https://router.example/api/openai/v1/chat/completions"),
    )


def _http_error_response(status_code: int, detail: str, path: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"error": {"code": "upstream_error", "message": detail}},
        request=httpx.Request("POST", f"https://router.example{path}"),
    )


def test_model_matrix_characterizes_unknown_profile_through_chat_first():
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/heuristic-profile"
    info = ModelInfo(
        id="heuristic-profile",
        name="Heuristic Profile",
        provider="github-copilot",
        supported_endpoints=None,
        operation_capabilities={Operation.CHAT: True, Operation.RESPONSES: True},
    )
    client = Mock()
    client.post.return_value = _chat_matrix_success()

    matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    client.post.assert_called_once()
    assert client.post.call_args.args[0] == "/api/openai/v1/chat/completions"


def test_model_matrix_unknown_profile_falls_back_once_to_responses_on_structured_rejection():
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/unknown-profile"
    info = ModelInfo(
        id="unknown-profile",
        name="Unknown Profile",
        provider="github-copilot",
        supported_endpoints=None,
    )
    client = Mock()
    client.post.side_effect = [_unsupported_api_response(), _responses_matrix_success()]

    matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    assert [call.args[0] for call in client.post.call_args_list] == [
        "/api/openai/v1/chat/completions",
        "/api/openai/v1/responses",
    ]


def test_model_matrix_unknown_profile_fails_after_both_transports_reject():
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/unknown-profile"
    info = ModelInfo(
        id="unknown-profile",
        name="Unknown Profile",
        provider="github-copilot",
        supported_endpoints=None,
    )
    client = Mock()
    client.post.side_effect = [_unsupported_api_response(), _unsupported_api_response()]

    with pytest.raises(AssertionError, match="both Chat and Responses explicitly unsupported"):
        matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    assert [call.args[0] for call in client.post.call_args_list] == [
        "/api/openai/v1/chat/completions",
        "/api/openai/v1/responses",
    ]


def test_model_matrix_unknown_profile_reports_both_rejected_transport_responses():
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/unknown-profile"
    info = ModelInfo(
        id="unknown-profile",
        name="Unknown Profile",
        provider="github-copilot",
        supported_endpoints=None,
    )
    chat_rejection = httpx.Response(
        400,
        json={
            "error": {
                "code": "unsupported_api_for_model",
                "message": "chat transport rejected",
            }
        },
        request=httpx.Request("POST", "https://router.example/api/openai/v1/chat/completions"),
    )
    responses_rejection = httpx.Response(
        400,
        json={
            "error": {
                "code": "unsupported_api_for_model",
                "message": "responses transport rejected",
            }
        },
        request=httpx.Request("POST", "https://router.example/api/openai/v1/responses"),
    )
    client = Mock()
    client.post.side_effect = [chat_rejection, responses_rejection]

    with pytest.raises(AssertionError) as exc_info:
        matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    message = str(exc_info.value)
    assert "chat transport rejected" in message
    assert "responses transport rejected" in message


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            400,
            json={
                "error": {
                    "code": "bad_request",
                    "message": "unsupported_api_for_model",
                }
            },
        ),
        httpx.Response(400, text="unsupported_api_for_model"),
        httpx.Response(400, json={"error": {"code": "UNSUPPORTED_API_FOR_MODEL"}}),
        httpx.Response(500, json={"error": {"code": "unsupported_api_for_model"}}),
    ],
)
def test_unsupported_api_detection_requires_exact_structured_400_code(response):
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")

    assert not matrix._is_unsupported_api(response)


def test_model_matrix_unknown_profile_does_not_fallback_on_unstructured_rejection():
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/unknown-profile"
    info = ModelInfo(
        id="unknown-profile",
        name="Unknown Profile",
        provider="github-copilot",
        supported_endpoints=None,
    )
    client = Mock()
    client.post.return_value = httpx.Response(
        400,
        json={"error": {"code": "bad_request", "message": "unsupported_api_for_model"}},
        request=httpx.Request("POST", "https://router.example/api/openai/v1/chat/completions"),
    )

    with pytest.raises(AssertionError, match="HTTP 400"):
        matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    client.post.assert_called_once()


@pytest.mark.parametrize(
    "supported_endpoints",
    [
        (),
        ("", " ", "\t"),
        ("ws:/responses",),
        ("/v1/messages",),
        ("/future/native",),
    ],
)
def test_model_matrix_reports_explicitly_unsupported_profile_without_sending_request(
    supported_endpoints,
):
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/native-only"
    info = ModelInfo(
        id="native-only",
        name="Native Only",
        provider="github-copilot",
        supported_endpoints=supported_endpoints,
    )
    client = Mock()
    client.post.side_effect = AssertionError("matrix sent a request to an unsupported endpoint")

    with pytest.raises(AssertionError, match="no supported HTTP endpoint"):
        matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    client.post.assert_not_called()


def test_model_matrix_declared_chat_does_not_probe_responses_after_rejection():
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/chat-model"
    info = ModelInfo(
        id="chat-model",
        name="Chat Model",
        provider="github-copilot",
        supported_endpoints=("/chat/completions",),
    )
    client = Mock()
    client.post.side_effect = [_unsupported_api_response(), _responses_matrix_success()]

    with pytest.raises(AssertionError) as exc_info:
        matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    client.post.assert_called_once()
    assert "catalog declared Chat support" in str(exc_info.value)
    assert "unsupported_api_for_model" in str(exc_info.value)


def test_model_matrix_treats_selected_endpoint_unsupported_api_as_failure():
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/responses-model"
    info = ModelInfo(
        id="responses-model",
        name="Responses Model",
        provider="github-copilot",
        supported_endpoints=("/responses",),
    )
    client = Mock()
    client.post.return_value = httpx.Response(
        400,
        text='{"error":{"code":"unsupported_api_for_model"}}',
        request=httpx.Request("POST", "https://router.example/api/openai/v1/responses"),
    )

    with pytest.raises(AssertionError, match="catalog declared Responses support"):
        matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    client.post.assert_called_once()
    assert client.post.call_args.args[0] == "/api/openai/v1/responses"


@pytest.mark.parametrize(
    ("supported_endpoint", "response_payload"),
    [
        (
            "/chat/completions",
            {
                "model": "github-copilot/wrong-model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
        (
            "/responses",
            {
                "model": "github-copilot/wrong-model",
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        ),
    ],
)
def test_model_matrix_rejects_selected_model_identity_mismatch(
    supported_endpoint,
    response_payload,
):
    """The public list-to-invoke gate must prove the listed identity executed."""
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/requested-model"
    catalog = {
        model: ModelInfo(
            id="requested-model",
            name="Requested Model",
            provider="github-copilot",
            supported_endpoints=(supported_endpoint,),
        )
    }
    client = Mock()
    client.post.return_value = httpx.Response(
        200,
        json=response_payload,
        request=httpx.Request("POST", f"https://router.example{supported_endpoint}"),
    )

    with pytest.raises(AssertionError, match="model identity"):
        matrix.test_copilot_model_matrix_openai_chat(client, [model], catalog)


@pytest.mark.parametrize(
    ("response", "expected_detail"),
    [
        (httpx.Response(200, text="not-json"), "invalid JSON"),
        (httpx.Response(200, json={}), "missing choices"),
        (httpx.Response(200, json={"choices": []}), "missing first choice"),
        (
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": "invalid",
                },
            ),
            "invalid Chat response",
        ),
    ],
)
def test_model_matrix_aggregates_malformed_chat_success_and_continues(
    response,
    expected_detail,
):
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    malformed_model = "github-copilot/malformed-chat"
    healthy_model = "github-copilot/healthy-chat"
    catalog = {
        model: ModelInfo(
            id=model.rsplit("/", 1)[-1],
            name=model,
            provider="github-copilot",
            supported_endpoints=("/chat/completions",),
        )
        for model in (malformed_model, healthy_model)
    }
    client = Mock()
    client.post.side_effect = [response, _chat_matrix_success(healthy_model)]

    with pytest.raises(AssertionError, match=expected_detail):
        matrix.test_copilot_model_matrix_openai_chat(
            client,
            [malformed_model, healthy_model],
            catalog,
        )

    assert len(client.post.call_args_list) == 2


@pytest.mark.parametrize(
    ("response", "expected_detail"),
    [
        (httpx.Response(200, text="not-json"), "invalid JSON"),
        (httpx.Response(200, json={}), "missing status"),
        (httpx.Response(200, json={"status": "completed"}), "missing usage"),
        (
            httpx.Response(200, json={"status": "completed", "usage": "invalid"}),
            "invalid Responses response",
        ),
    ],
)
def test_model_matrix_aggregates_malformed_responses_success_and_continues(
    response,
    expected_detail,
):
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    malformed_model = "github-copilot/malformed-responses"
    healthy_model = "github-copilot/healthy-responses"
    catalog = {
        model: ModelInfo(
            id=model.rsplit("/", 1)[-1],
            name=model,
            provider="github-copilot",
            supported_endpoints=("/responses",),
        )
        for model in (malformed_model, healthy_model)
    }
    client = Mock()
    client.post.side_effect = [response, _responses_matrix_success(healthy_model)]

    with pytest.raises(AssertionError, match=expected_detail):
        matrix.test_copilot_model_matrix_openai_chat(
            client,
            [malformed_model, healthy_model],
            catalog,
        )

    assert len(client.post.call_args_list) == 2


def test_model_matrix_rejects_unhandled_endpoint_state_before_sending(monkeypatch):
    matrix = importlib.import_module("integration_tests.test_live_model_matrix")
    model = "github-copilot/future-state"
    info = ModelInfo(id="future-state", name="Future State", provider="github-copilot")
    future_state = object()
    monkeypatch.setattr(matrix, "select_live_http_endpoint", lambda *_args: future_state)
    client = Mock()
    client.post.side_effect = AssertionError("unhandled state sent a request")

    with pytest.raises(AssertionError, match="unhandled live HTTP endpoint state"):
        matrix.test_copilot_model_matrix_openai_chat(client, [model], {model: info})

    client.post.assert_not_called()


def test_anthropic_thinking_models_default_to_reasoning_capable_claude(monkeypatch):
    conftest = importlib.import_module("integration_tests.conftest")
    monkeypatch.delenv("RM_INTEGRATION_MAX_REASONING_MODELS", raising=False)
    gpt = "github-copilot/gpt-5.4"
    claude = "github-copilot/claude-sonnet-4.6"
    catalog = {
        gpt: ModelInfo(
            id="gpt-5.4",
            name="GPT 5.4",
            provider="github-copilot",
            reasoning_effort_values=["low"],
        ),
        claude: ModelInfo(
            id="claude-sonnet-4.6",
            name="Claude Sonnet 4.6",
            provider="github-copilot",
            supports_thinking=True,
        ),
    }

    selected = conftest.anthropic_thinking_models.__wrapped__(catalog)

    assert selected == [claude]


def test_anthropic_thinking_models_only_include_reasoning_capable_claude(monkeypatch):
    conftest = importlib.import_module("integration_tests.conftest")
    monkeypatch.setenv("RM_INTEGRATION_MAX_REASONING_MODELS", "0")
    claude = "github-copilot/claude-opus-4.6"
    catalog = {
        claude: ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            feature_capabilities={"reasoning": True},
        ),
        "github-copilot/claude-old": ModelInfo(
            id="claude-old",
            name="Claude Old",
            provider="github-copilot",
        ),
        "github-copilot/claude-opus-4.6-high": ModelInfo(
            id="claude-opus-4.6-high",
            name="Claude Opus 4.6 High",
            provider="github-copilot",
            feature_capabilities={"reasoning": True},
        ),
        "github-copilot/gpt-5.5": ModelInfo(
            id="gpt-5.5",
            name="GPT 5.5",
            provider="github-copilot",
            feature_capabilities={"reasoning": True},
        ),
        "github-copilot/future-reasoner": ModelInfo(
            id="future-reasoner",
            name="Future Reasoner",
            provider="github-copilot",
            reasoning_effort_values=["high"],
        ),
    }

    selected = conftest.anthropic_thinking_models.__wrapped__(catalog)

    assert selected == [claude]


def test_anthropic_gpt5_bridge_models_only_include_http_responses_gpt5(monkeypatch):
    conftest = importlib.import_module("integration_tests.conftest")
    monkeypatch.setenv("RM_INTEGRATION_MAX_REASONING_MODELS", "0")
    gpt = "github-copilot/gpt-5.4"
    catalog = {
        gpt: ModelInfo(
            id="gpt-5.4",
            name="GPT 5.4",
            provider="github-copilot",
            supported_endpoints=("/responses",),
        ),
        "github-copilot/claude-future": ModelInfo(
            id="claude-future",
            name="Claude Future",
            provider="github-copilot",
            supported_endpoints=("/responses",),
        ),
        "github-copilot/future-reasoner": ModelInfo(
            id="future-reasoner",
            name="Future Reasoner",
            provider="github-copilot",
            supported_endpoints=("/responses",),
        ),
        "github-copilot/gpt-5-ws": ModelInfo(
            id="gpt-5-ws",
            name="GPT 5 WS",
            provider="github-copilot",
            supported_endpoints=("ws:/responses",),
        ),
    }

    selected = conftest.anthropic_gpt5_bridge_models.__wrapped__(catalog)

    assert selected == [gpt]


def test_live_reasoning_selection_uses_catalog_metadata_not_model_name():
    conftest = importlib.import_module("integration_tests.conftest")
    capable = ModelInfo(
        id="future-reasoner",
        name="Future Reasoner",
        provider="github-copilot",
        reasoning_effort_values=["low"],
    )
    unsupported_claude = ModelInfo(
        id="claude-old",
        name="Claude Old",
        provider="github-copilot",
        supported_endpoints=("/chat/completions", "/v1/messages"),
        reasoning_effort_values=[],
    )
    catalog_silent_claude = ModelInfo(
        id="claude-silent",
        name="Claude Silent",
        provider="github-copilot",
        supported_endpoints=("/chat/completions", "/v1/messages"),
        operation_capabilities={Operation.CHAT: True, Operation.NATIVE_ANTHROPIC: True},
    )
    explicitly_unsupported = ModelInfo(
        id="claude-feature-disabled",
        name="Claude Feature Disabled",
        provider="github-copilot",
        feature_capabilities={"reasoning": False},
    )
    none_only = ModelInfo(
        id="provider-disabled-reasoning",
        name="Provider Disabled Reasoning",
        provider="github-copilot",
        reasoning_effort_values=["none"],
    )

    assert conftest.model_supports_reasoning(capable)
    assert conftest.model_supports_reasoning(none_only)
    assert not conftest.model_explicitly_rejects_reasoning(none_only)
    assert "none" not in conftest.OPENAI_REASONING_EFFORTS
    assert not conftest.model_supports_reasoning(unsupported_claude)
    assert conftest.model_explicitly_rejects_reasoning(unsupported_claude)
    assert conftest.model_explicitly_rejects_reasoning(explicitly_unsupported)
    assert not conftest.model_explicitly_rejects_reasoning(catalog_silent_claude)


def test_http_failure_assertion_reads_unread_stream_body_safely():
    conftest = importlib.import_module("integration_tests.conftest")
    response = httpx.Response(
        400,
        stream=httpx.ByteStream(b'{"error":"bad"}'),
        request=httpx.Request("POST", "https://example.test"),
    )

    try:
        conftest.assert_http_success(response)
    except AssertionError as error:
        assert "bad" in str(error)
    else:
        raise AssertionError("assert_http_success accepted an error response")


def _scripted_compat_client(
    *responses: httpx.Response,
) -> tuple[httpx.Client, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    scripted = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return next(scripted)

    return (
        httpx.Client(
            base_url="https://router.example",
            transport=httpx.MockTransport(handler),
        ),
        requests,
    )


def test_compat_probe_retries_once_after_exact_bare_400_bytes():
    conftest = importlib.import_module("integration_tests.conftest")
    first = httpx.Response(400, content=b"Bad Request\n")
    success = httpx.Response(200, json={"ok": True})
    client, requests = _scripted_compat_client(first, success)
    sleeper = Mock()

    with client:
        response = conftest._post_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            backoff_seconds=0.01,
            sleep=sleeper,
        )

    assert response is success
    assert len(requests) == 2
    assert first.is_stream_consumed
    assert first.is_closed
    sleeper.assert_called_once_with(0.01)


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (400, b"Bad Request\r\n"),
        (400, b"Bad Request"),
        (400, b" Bad Request\n"),
        (400, b"Bad Request\n "),
        (400, b'{"error":"Bad Request"}'),
        (500, b"Bad Request\n"),
    ],
)
def test_compat_probe_does_not_retry_near_match_or_other_status(status_code, body):
    conftest = importlib.import_module("integration_tests.conftest")
    first = httpx.Response(status_code, content=body)
    unexpected = httpx.Response(200, json={"unexpected": True})
    client, requests = _scripted_compat_client(first, unexpected)
    sleeper = Mock()

    with client:
        response = conftest._post_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            sleep=sleeper,
        )

    assert response is first
    assert len(requests) == 1
    sleeper.assert_not_called()


def test_compat_probe_double_exact_failure_reports_two_bounded_attempts():
    conftest = importlib.import_module("integration_tests.conftest")
    first = httpx.Response(400, content=b"Bad Request\n")
    second = httpx.Response(400, content=b"Bad Request\n")
    client, requests = _scripted_compat_client(first, second)

    with client, pytest.raises(AssertionError) as exc_info:
        conftest._post_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            backoff_seconds=0,
            sleep=lambda _seconds: None,
        )

    message = str(exc_info.value)
    assert "failed after 2 attempts" in message
    assert "attempt 1: HTTP 400" in message
    assert "attempt 2: HTTP 400" in message
    assert message.count("Bad Request") == 2
    assert len(requests) == 2


def test_compat_probe_nonmatching_failure_returns_to_normal_http_assertion():
    conftest = importlib.import_module("integration_tests.conftest")
    failure = httpx.Response(400, content=b"Bad Request\r\n")
    client, requests = _scripted_compat_client(failure)

    with client:
        response = conftest._post_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            sleep=lambda _seconds: None,
        )
        with pytest.raises(AssertionError, match="Bad Request"):
            conftest.assert_http_success(response)

    assert len(requests) == 1


def test_stream_compat_probe_closes_exact_400_then_reopens_once_before_yield():
    conftest = importlib.import_module("integration_tests.conftest")
    first = httpx.Response(400, stream=httpx.ByteStream(b"Bad Request\n"))
    success = httpx.Response(200, stream=httpx.ByteStream(b"data: [DONE]\n\n"))
    client, requests = _scripted_compat_client(first, success)
    sleeper = Mock()

    with client:
        with conftest._stream_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            timeout=180.0,
            backoff_seconds=0.01,
            sleep=sleeper,
        ) as response:
            assert response is success
            assert first.is_stream_consumed
            assert first.is_closed
            assert not response.is_stream_consumed
            assert b"".join(response.iter_bytes()) == b"data: [DONE]\n\n"

    assert len(requests) == 2
    sleeper.assert_called_once_with(0.01)


def test_stream_compat_probe_never_retries_200_before_or_after_body_consumption():
    conftest = importlib.import_module("integration_tests.conftest")
    success = httpx.Response(200, stream=httpx.ByteStream(b"data: first\n\n"))
    unexpected = httpx.Response(200, stream=httpx.ByteStream(b"data: second\n\n"))
    client, requests = _scripted_compat_client(success, unexpected)
    sleeper = Mock()

    with client:
        with conftest._stream_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            timeout=180.0,
            sleep=sleeper,
        ) as response:
            assert not response.is_stream_consumed
            iterator = response.iter_bytes()
            assert next(iterator) == b"data: first\n\n"
            assert len(requests) == 1

    assert len(requests) == 1
    sleeper.assert_not_called()


def test_stream_compat_probe_nonmatching_400_is_read_but_not_retried():
    conftest = importlib.import_module("integration_tests.conftest")
    failure = httpx.Response(400, stream=httpx.ByteStream(b"Bad Request\r\n"))
    unexpected = httpx.Response(200, stream=httpx.ByteStream(b"data: unexpected\n\n"))
    client, requests = _scripted_compat_client(failure, unexpected)

    with client:
        with conftest._stream_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            timeout=180.0,
            sleep=lambda _seconds: None,
        ) as response:
            assert response is failure
            assert response.is_stream_consumed
            with pytest.raises(AssertionError, match="Bad Request"):
                conftest.assert_http_success(response)

    assert len(requests) == 1


def test_stream_compat_probe_double_exact_failure_reports_two_attempts():
    conftest = importlib.import_module("integration_tests.conftest")
    first = httpx.Response(400, stream=httpx.ByteStream(b"Bad Request\n"))
    second = httpx.Response(400, stream=httpx.ByteStream(b"Bad Request\n"))
    client, requests = _scripted_compat_client(first, second)

    with client, pytest.raises(AssertionError) as exc_info:
        with conftest._stream_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            timeout=180.0,
            backoff_seconds=0,
            sleep=lambda _seconds: None,
        ):
            raise AssertionError("double exact retry unexpectedly yielded a response")

    message = str(exc_info.value)
    assert "failed after 2 attempts" in message
    assert "attempt 1: HTTP 400" in message
    assert "attempt 2: HTTP 400" in message
    assert len(requests) == 2
    assert first.is_closed
    assert second.is_closed


class _ReadErrorStream(httpx.SyncByteStream):
    def __init__(self):
        self.closed = False

    def __iter__(self):
        raise httpx.ReadError("compat probe read failed")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("failing_attempt", [1, 2])
def test_stream_compat_probe_closes_response_when_classifier_read_fails(failing_attempt):
    conftest = importlib.import_module("integration_tests.conftest")
    failing_stream = _ReadErrorStream()
    failing = httpx.Response(400, stream=failing_stream)
    responses = (
        (failing,)
        if failing_attempt == 1
        else (httpx.Response(400, content=b"Bad Request\n"), failing)
    )
    client, requests = _scripted_compat_client(*responses)

    with client:
        with pytest.raises(httpx.ReadError, match="compat probe read failed"):
            with conftest._stream_compat_probe_with_exact_400_retry(
                client,
                "/compat",
                json_payload={"probe": True},
                timeout=180.0,
                backoff_seconds=0,
                sleep=lambda _seconds: None,
            ):
                raise AssertionError("read failure unexpectedly yielded a response")
        assert failing_stream.closed

    assert len(requests) == failing_attempt


class _RecordingStreamContext:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.exit_args = None

    def __enter__(self):
        return self.response

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_args = (exc_type, exc_value, traceback)
        self.response.close()
        return False


class _RecordingStreamClient:
    def __init__(self, *responses: httpx.Response):
        self.contexts = [_RecordingStreamContext(response) for response in responses]
        self.calls = 0

    def stream(self, *_args, **_kwargs):
        context = self.contexts[self.calls]
        self.calls += 1
        return context


class _ProbeBodyError(Exception):
    pass


@pytest.mark.parametrize("yield_attempt", [1, 2])
def test_stream_compat_probe_forwards_yield_exception_to_active_context(yield_attempt):
    conftest = importlib.import_module("integration_tests.conftest")
    success = httpx.Response(200, stream=httpx.ByteStream(b"data: event\n\n"))
    responses = (
        (success,)
        if yield_attempt == 1
        else (httpx.Response(400, content=b"Bad Request\n"), success)
    )
    client = _RecordingStreamClient(*responses)

    with pytest.raises(_ProbeBodyError, match="consumer failed") as exc_info:
        with conftest._stream_compat_probe_with_exact_400_retry(
            client,
            "/compat",
            json_payload={"probe": True},
            timeout=180.0,
            backoff_seconds=0,
            sleep=lambda _seconds: None,
        ):
            raise _ProbeBodyError("consumer failed")

    active_context = client.contexts[yield_attempt - 1]
    assert active_context.exit_args is not None
    assert active_context.exit_args[0] is _ProbeBodyError
    assert active_context.exit_args[1] is exc_info.value
    assert active_context.exit_args[2] is not None


def _enclosing_function_name(node, parents) -> str:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(current)
    return "<module>"


def test_exact_400_retry_apis_are_scoped_to_three_fixed_compat_probes():
    public_apis = {
        "post_gemini_generate_content_compat_probe": (
            "test_live_gemini_paths.py",
            "test_gemini_generate_content",
        ),
        "post_openai_chat_compat_probe": (
            "test_live_openai_paths.py",
            "test_openai_chat_completion_non_streaming_returns_usage",
        ),
        "stream_openai_chat_compat_probe": (
            "test_live_openai_paths.py",
            "test_openai_chat_completion_streaming_returns_chunks_and_done",
        ),
    }
    imports: dict[str, list[tuple[str, str | None]]] = {name: [] for name in public_apis}
    references: dict[str, list[tuple[str, str]]] = {name: [] for name in public_apis}
    attribute_references: dict[str, list[str]] = {name: [] for name in public_apis}

    for path in (ROOT / "integration_tests").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "integration_tests.conftest":
                for alias in node.names:
                    if alias.name in public_apis:
                        imports[alias.name].append((path.name, alias.asname))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in public_apis:
                    references[node.id].append((path.name, _enclosing_function_name(node, parents)))
            elif isinstance(node, ast.Attribute) and node.attr in public_apis:
                attribute_references[node.attr].append(path.name)

    for name, expected_location in public_apis.items():
        assert imports[name] == [(expected_location[0], None)]
        assert references[name] == [expected_location]
        assert attribute_references[name] == []

    conftest = importlib.import_module("integration_tests.conftest")
    assert not hasattr(conftest, "post_compat_probe_with_exact_400_retry")
    assert not hasattr(conftest, "stream_compat_probe_with_exact_400_retry")

    source = ast.parse((ROOT / "integration_tests" / "conftest.py").read_text(encoding="utf-8"))
    private_core_calls = {
        "_post_compat_probe_with_exact_400_retry": [],
        "_stream_compat_probe_with_exact_400_retry": [],
    }
    parents = {
        child: parent for parent in ast.walk(source) for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(source):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in private_core_calls
        ):
            private_core_calls[node.func.id].append(_enclosing_function_name(node, parents))

    assert private_core_calls == {
        "_post_compat_probe_with_exact_400_retry": [
            "post_gemini_generate_content_compat_probe",
            "post_openai_chat_compat_probe",
        ],
        "_stream_compat_probe_with_exact_400_retry": [
            "stream_openai_chat_compat_probe",
        ],
    }


def test_reasoning_matrix_payloads_cover_budget_and_effort_controls():
    """Live matrix helpers should exercise thinking budgets and reasoning effort."""
    conftest = importlib.import_module("integration_tests.conftest")

    anthropic = conftest.anthropic_reasoning_payload(
        "github-copilot/claude-sonnet-4.6",
        budget=4096,
        stream=True,
    )
    openai = conftest.openai_reasoning_payload(
        "github-copilot/gpt-5.4",
        effort="high",
        stream=True,
    )

    assert anthropic["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert anthropic["max_tokens"] > 4096
    assert openai["reasoning_effort"] == "high"
    assert openai["stream_options"] == {"include_usage": True}


def test_anthropic_reasoning_result_accepts_empty_content_at_requested_token_limit():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 12, "output_tokens": 2048},
    }

    matrix._assert_anthropic_reasoning_result(
        data,
        requested_max_tokens=2048,
        requested_thinking_budget=None,
    )


def test_anthropic_reasoning_result_accepts_empty_content_at_explicit_thinking_budget():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 12, "output_tokens": 16000},
    }

    matrix._assert_anthropic_reasoning_result(
        data,
        requested_max_tokens=16384,
        requested_thinking_budget=16000,
    )


@pytest.mark.parametrize("requested_max_tokens", [True, 0, -1, 1.5])
def test_anthropic_reasoning_result_rejects_invalid_requested_token_limit(
    requested_max_tokens,
):
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [{"type": "text", "text": "The answer is 42."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 32},
    }

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            data,
            requested_max_tokens=requested_max_tokens,
            requested_thinking_budget=None,
        )


@pytest.mark.parametrize(
    ("field", "present", "value"),
    [
        ("output_tokens", False, None),
        ("output_tokens", True, True),
        ("output_tokens", True, 1.5),
        ("output_tokens", True, -1),
        ("input_tokens", False, None),
        ("input_tokens", True, True),
        ("input_tokens", True, 1.5),
        ("input_tokens", True, -1),
    ],
)
def test_anthropic_reasoning_result_rejects_invalid_usage_token_count(
    field,
    present,
    value,
):
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [{"type": "text", "text": "The answer is 42."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 32},
    }
    if present:
        data["usage"][field] = value
    else:
        del data["usage"][field]

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            data,
            requested_max_tokens=2048,
            requested_thinking_budget=None,
        )


def test_anthropic_reasoning_result_rejects_zero_input_tokens_at_helper_boundary(monkeypatch):
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    monkeypatch.setattr(matrix, "assert_anthropic_usage", lambda usage: None)
    data = {
        "content": [{"type": "text", "text": "The answer is 42."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 32},
    }

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            data,
            requested_max_tokens=2048,
            requested_thinking_budget=None,
        )


def test_anthropic_reasoning_result_accepts_zero_output_tokens_with_visible_content():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [{"type": "text", "text": "The answer is 42."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }

    matrix._assert_anthropic_reasoning_result(
        data,
        requested_max_tokens=1,
        requested_thinking_budget=None,
    )


def test_anthropic_reasoning_result_rejects_zero_output_tokens_for_empty_saturation():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            data,
            requested_max_tokens=1,
            requested_thinking_budget=None,
        )


def test_anthropic_reasoning_result_accepts_positive_integer_contract_values():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [{"type": "text", "text": "The answer is 42."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 32},
    }

    matrix._assert_anthropic_reasoning_result(
        data,
        requested_max_tokens=2048,
        requested_thinking_budget=None,
    )


def test_anthropic_reasoning_result_rejects_empty_end_turn():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 2048},
    }

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            data,
            requested_max_tokens=2048,
            requested_thinking_budget=None,
        )


def test_anthropic_reasoning_result_rejects_empty_content_below_requested_token_limit():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 12, "output_tokens": 2047},
    }

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            data,
            requested_max_tokens=2048,
            requested_thinking_budget=None,
        )


def test_anthropic_reasoning_result_rejects_empty_content_below_explicit_thinking_budget():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 12, "output_tokens": 15999},
    }

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            data,
            requested_max_tokens=16384,
            requested_thinking_budget=16000,
        )


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": "The answer is 42."}],
        [{"type": "thinking", "thinking": "I should calculate this carefully."}],
    ],
)
def test_anthropic_reasoning_result_accepts_visible_text_or_thinking(content):
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": content,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 32},
    }

    matrix._assert_anthropic_reasoning_result(
        data,
        requested_max_tokens=2048,
        requested_thinking_budget=None,
    )


@pytest.mark.parametrize("missing_field", ["usage", "stop_reason"])
def test_anthropic_reasoning_result_rejects_missing_usage_or_terminal(missing_field):
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")
    data = {
        "content": [{"type": "text", "text": "The answer is 42."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 32},
    }
    del data[missing_field]

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            data,
            requested_max_tokens=2048,
            requested_thinking_budget=None,
        )


def test_anthropic_effort_payload_includes_conflicting_budget():
    """Live coverage must exercise effort precedence, not effort in isolation."""
    conftest = importlib.import_module("integration_tests.conftest")

    payload = conftest.anthropic_effort_payload(
        "github-copilot/claude-opus-4.8",
        effort="xhigh",
        budget=1024,
        stream=True,
    )

    assert payload["thinking"] == {"type": "adaptive", "budget_tokens": 1024}
    assert payload["output_config"] == {"effort": "xhigh"}
    assert payload["stream"] is True


def test_integration_tests_include_reasoning_and_gemini_family_matrices():
    """Local live tests should cover the e2e reasoning and Gemini family gaps."""
    integration_dir = ROOT / "integration_tests"

    reasoning = (integration_dir / "test_live_reasoning_matrix.py").read_text(encoding="utf-8")
    gemini = (integration_dir / "test_live_gemini_matrix.py").read_text(encoding="utf-8")

    assert "test_anthropic_claude_thinking_budget_matrix" in reasoning
    assert "test_anthropic_gpt5_responses_bridge_thinking_budget_matrix" in reasoning
    assert "test_openai_chat_reasoning_effort_matrix" in reasoning
    assert "test_anthropic_output_config_effort_precedence" in reasoning
    assert "test_gemini_family_generate_content_matrix" in gemini


def test_responses_top_p_fixture_only_selects_verified_representative():
    conftest = importlib.import_module("integration_tests.conftest")
    arbitrary = "github-copilot/arbitrary-responses-model"
    representative = "github-copilot/gpt-5.3-codex"
    catalog = {
        arbitrary: ModelInfo(
            id="arbitrary-responses-model",
            name="Arbitrary Responses Model",
            provider="github-copilot",
            supported_endpoints=("/responses",),
        ),
        representative: ModelInfo(
            id="gpt-5.3-codex",
            name="GPT 5.3 Codex",
            provider="github-copilot",
            supported_endpoints=("/responses",),
        ),
    }

    selected = conftest.responses_top_p_model.__wrapped__(
        [arbitrary, representative],
        catalog,
    )

    assert selected == representative


def test_bridge_stop_sequences_contract_covers_both_modes_and_native_json_400():
    bridge = importlib.import_module("integration_tests.test_live_anthropic_responses_bridge")
    test = bridge.test_bridge_stop_sequences_returns_native_400
    stream_marks = [
        mark
        for mark in getattr(test, "pytestmark", ())
        if mark.name == "parametrize" and mark.args[0] == "stream"
    ]

    assert stream_marks
    assert tuple(stream_marks[0].args[1]) == (False, True)

    source = inspect.getsource(test)
    assert "response.status_code == 400" in source
    assert 'response.headers["content-type"].startswith("application/json")' in source
    assert 'response.json()["type"] == "error"' in source
    assert 'response.json()["error"]["type"] == "invalid_request_error"' in source
    assert '"event:" not in response.text' in source


def test_integration_harness_documents_existing_config_usage():
    """The harness should say it reuses the user's existing RM configuration."""
    conftest = (ROOT / "integration_tests" / "conftest.py").read_text(encoding="utf-8")

    assert "get_current_context_api_key" in conftest
    assert "ROUTER_MAESTRO_API_KEY" in conftest
    assert "router_maestro.server:app" in conftest


def test_integration_tests_do_not_cover_admin_endpoints():
    """Live integration tests should cover model calls, not admin endpoints."""
    integration_dir = ROOT / "integration_tests"

    for path in integration_dir.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        assert "/api/admin/" not in content, path
