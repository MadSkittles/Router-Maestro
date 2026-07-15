"""Tests for OpenAI chat route option passthrough."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.providers import ChatResponse, ChatStreamChunk, RequestOptionError
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.server.routes.chat import chat_completions
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.server.schemas import ChatCompletionRequest, ChatMessage


class _CapturingRouter:
    def __init__(self):
        self.request = None

    async def chat_completion(self, request):
        self.request = request
        return ChatResponse(content="ok", model=request.model), "test-provider"

    async def prepare_chat_completion_stream(self, request):
        self.request = request
        return object()

    async def chat_completion_stream(self, request, *, prepared_plan=None):
        self.request = request

        async def chunks():
            yield ChatStreamChunk(content="ok", finish_reason="stop")

        return chunks(), "test-provider"


@pytest.mark.anyio
async def test_openai_chat_route_preserves_supported_typed_options(monkeypatch):
    """Accepted OpenAI chat fields should reach typed provider-facing fields."""
    router = _CapturingRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    request = ChatCompletionRequest(
        model="openai/gpt-4o",
        messages=[ChatMessage(role="user", content="Hello")],
        top_p=0.25,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        stop=["END"],
        user="user-123",
        metadata={"trace_id": "trace-123"},
        service_tier="flex",
    )

    response = await chat_completions(request)

    assert response.choices[0].message.content == "ok"
    assert router.request is not None
    assert router.request.top_p == 0.25
    assert router.request.frequency_penalty == 0.1
    assert router.request.presence_penalty == 0.2
    assert router.request.stop == ["END"]
    assert router.request.user == "user-123"
    assert router.request.metadata == {"trace_id": "trace-123"}
    assert router.request.service_tier == "flex"
    assert router.request.extra == {}


@pytest.mark.anyio
async def test_openai_chat_route_preserves_assistant_refusal_history(monkeypatch):
    router = _CapturingRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = await chat_completions(
        ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[
                ChatMessage(role="assistant", content=None, refusal="I cannot help"),
                ChatMessage(role="user", content="Why?"),
            ],
        )
    )

    assert response.choices[0].message.content == "ok"
    assert router.request.messages[0].content is None
    assert router.request.messages[0].refusal == "I cannot help"


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_openai_chat_endpoint_preserves_metadata_and_service_tier(monkeypatch, stream):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    router = _CapturingRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "metadata": {"trace_id": "trace-123"},
            "service_tier": "flex",
        },
    )

    assert response.status_code == 200
    assert router.request is not None
    assert router.request.metadata == {"trace_id": "trace-123"}
    assert router.request.service_tier == "flex"


@pytest.mark.parametrize(
    ("payload_update", "expected"),
    [
        pytest.param({}, None, id="omitted"),
        pytest.param({"temperature": 1.0}, 1.0, id="explicit-default"),
        pytest.param({"temperature": 0.4}, 0.4, id="explicit-custom"),
    ],
)
def test_openai_chat_endpoint_preserves_temperature_presence(
    monkeypatch,
    payload_update,
    expected,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    router = _CapturingRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            **payload_update,
        },
    )

    assert response.status_code == 200
    assert router.request.temperature == expected


@pytest.mark.parametrize(
    "stream_options",
    [
        pytest.param({}, id="default-false"),
        pytest.param({"include_usage": True}, id="include-usage"),
        pytest.param({"include_usage": False}, id="exclude-usage"),
    ],
)
def test_openai_chat_accepts_typed_stream_options_without_provider_passthrough(
    monkeypatch,
    stream_options,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    router = _CapturingRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": stream_options,
        },
    )

    assert response.status_code == 200
    assert router.request is not None
    assert router.request.extra == {}
    assert not hasattr(router.request, "stream_options")


@pytest.mark.parametrize(
    ("stream", "stream_options", "parameter"),
    [
        pytest.param(
            True,
            {"include_usage": "true"},
            "stream_options.include_usage",
            id="string-include-usage",
        ),
        pytest.param(
            True,
            {"include_usage": 1},
            "stream_options.include_usage",
            id="integer-include-usage",
        ),
        pytest.param(
            True,
            {"include_usage": None},
            "stream_options.include_usage",
            id="null-include-usage",
        ),
        pytest.param(
            False,
            {"include_usage": True},
            "stream_options",
            id="nonstream-request",
        ),
    ],
)
def test_openai_chat_rejects_invalid_stream_options_before_router_calls(
    monkeypatch,
    stream,
    stream_options,
    parameter,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app, raise_server_exceptions=False)
    router = _CapturingRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "stream_options": stream_options,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["param"] == parameter
    assert "data:" not in response.text
    assert router.request is None


def test_openai_chat_unknown_stream_option_passes_through(monkeypatch):
    """An unknown nested ``stream_options`` field (e.g. OpenAI's ``include_obfuscation``)
    is forwarded/ignored, not rejected — Router-Maestro is a transparent proxy."""
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app, raise_server_exceptions=False)
    router = _CapturingRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True, "include_obfuscation": True},
        },
    )

    assert response.status_code != 400, response.text
    assert "Unsupported request option" not in response.text
    # The request reached routing rather than being rejected at the option gate.
    assert router.request is not None


class _RejectingOptionRouter:
    def __init__(self, parameter: str, expected) -> None:
        self.parameter = parameter
        self.expected = expected

    def _reject(self, request) -> None:
        assert getattr(request, self.parameter) == self.expected
        raise RequestOptionError(
            f"selected-provider does not support request option '{self.parameter}'",
            provider="selected-provider",
            model=request.model,
            parameter=self.parameter,
        )

    async def chat_completion(self, request):
        self._reject(request)

    async def prepare_chat_completion_stream(self, request):
        self._reject(request)


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        pytest.param("metadata", {"trace_id": "trace-123"}, id="metadata"),
        pytest.param("service_tier", "flex", id="service-tier"),
    ],
)
def test_openai_chat_selected_provider_option_rejection_is_native_400(
    monkeypatch,
    stream,
    parameter,
    value,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app, raise_server_exceptions=False)
    router = _RejectingOptionRouter(parameter, value)
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "selected-provider/model-a",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            parameter: value,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["param"] == parameter
    assert "data:" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "upstream_model",
    ["shared-model", "first/shared-model"],
    ids=["bare-upstream", "already-qualified"],
)
async def test_openai_chat_nonstream_response_qualifies_model_once(
    monkeypatch,
    upstream_model,
):
    class _ModelRouter:
        async def chat_completion(self, _request):
            return ChatResponse(content="ok", model=upstream_model), "first"

    monkeypatch.setattr(
        "router_maestro.server.routes.chat.get_router",
        lambda: _ModelRouter(),
    )

    response = await chat_completions(
        ChatCompletionRequest(
            model="first/shared-model",
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert response.model == "first/shared-model"


@pytest.mark.anyio
async def test_openai_chat_nonstream_emits_refusal_field(monkeypatch):
    class _RefusalRouter:
        async def chat_completion(self, request):
            response = ChatResponse(content=None, model=request.model)
            response.refusal = "I cannot help"
            return response, "first"

    monkeypatch.setattr(
        "router_maestro.server.routes.chat.get_router",
        lambda: _RefusalRouter(),
    )

    response = await chat_completions(
        ChatCompletionRequest(
            model="first/shared-model",
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert response.choices[0].message.content is None
    assert response.choices[0].message.refusal == "I cannot help"


class _ResponsesBridgePreflightRouter:
    """Exercise the real Copilot bridge policy without opening upstream I/O."""

    def __init__(self) -> None:
        self.provider = CopilotProvider()
        self.upstream = AsyncMock(side_effect=AssertionError("upstream must not run"))
        self.chat_calls = 0
        self.prepare_calls = 0
        self.stream_calls = 0

    def _validate(self, request, *, stream: bool) -> None:
        opted_in = replace(request, use_responses_api=True, extra={})
        self.provider.validate_chat_request(opted_in, stream=stream)

    async def chat_completion(self, request):
        self.chat_calls += 1
        self._validate(request, stream=False)
        await self.upstream()

    async def prepare_chat_completion_stream(self, request):
        self.prepare_calls += 1
        self._validate(request, stream=True)
        await self.upstream()

    async def chat_completion_stream(self, request, *, prepared_plan=None):
        self.stream_calls += 1
        await self.upstream()


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_openai_chat_responses_bridge_rejects_explicit_temperature_before_upstream(
    monkeypatch,
    stream,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app, raise_server_exceptions=False)
    router = _ResponsesBridgePreflightRouter()
    monkeypatch.setenv("ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API", "1")
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "temperature": 0.4,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["param"] == "temperature"
    assert "data:" not in response.text
    router.upstream.assert_not_awaited()
    assert router.chat_calls == (0 if stream else 1)
    assert router.prepare_calls == (1 if stream else 0)
    assert router.stream_calls == 0


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("thinking_budget", [1, 1023])
def test_openai_chat_responses_bridge_rejects_tiny_budget_before_upstream(
    monkeypatch,
    stream,
    thinking_budget,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app, raise_server_exceptions=False)
    router = _ResponsesBridgePreflightRouter()
    monkeypatch.setenv("ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API", "1")
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            },
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["param"] == "thinking_budget"
    assert "data:" not in response.text
    router.upstream.assert_not_awaited()
    assert router.chat_calls == (0 if stream else 1)
    assert router.prepare_calls == (1 if stream else 0)
    assert router.stream_calls == 0


class _ThinkingRouteRouter:
    def __init__(self) -> None:
        self.request = None
        self.chat_completion = AsyncMock(side_effect=self._chat_completion)
        self.prepare_chat_completion_stream = AsyncMock(side_effect=self._prepare_stream)
        self.chat_completion_stream = AsyncMock(side_effect=self._chat_completion_stream)

    async def _chat_completion(self, request):
        self.request = request
        return ChatResponse(content="ok", model=request.model), "test-provider"

    async def _prepare_stream(self, request):
        self.request = request
        return object()

    async def _chat_completion_stream(self, request, *, prepared_plan=None):
        async def chunks():
            yield ChatStreamChunk(content="ok", finish_reason="stop")

        return chunks(), "test-provider"


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_openai_chat_preserves_minimal_reasoning_effort(monkeypatch, stream):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    router = _ThinkingRouteRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "reasoning_effort": "minimal",
        },
    )

    assert response.status_code == 200
    assert router.request is not None
    assert router.request.reasoning_effort == "minimal"
    assert router.request.thinking_budget is None


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize(
    ("thinking", "parameter"),
    [
        pytest.param(None, "thinking", id="explicit-null-thinking"),
        pytest.param("enabled", "thinking", id="non-object-string"),
        pytest.param([], "thinking", id="non-object-list"),
        pytest.param({}, "thinking.type", id="missing-type"),
        pytest.param({"type": 7}, "thinking.type", id="non-string-type"),
        pytest.param({"type": "unknown"}, "thinking.type", id="unknown-type"),
        pytest.param(
            {"type": "enabled", "budget_tokens": True},
            "thinking.budget_tokens",
            id="bool-budget",
        ),
        pytest.param(
            {"type": "enabled", "budget_tokens": "1024"},
            "thinking.budget_tokens",
            id="string-budget",
        ),
        pytest.param(
            {"type": "enabled", "budget_tokens": 1.5},
            "thinking.budget_tokens",
            id="fractional-budget",
        ),
        pytest.param(
            {"type": "enabled", "budget_tokens": 0},
            "thinking.budget_tokens",
            id="zero-budget",
        ),
        pytest.param(
            {"type": "enabled", "budget_tokens": -1},
            "thinking.budget_tokens",
            id="negative-budget",
        ),
        pytest.param(
            {"type": "enabled", "budget_tokens": None},
            "thinking.budget_tokens",
            id="explicit-null-budget",
        ),
        pytest.param(
            {"type": "adaptive", "budget_tokens": False},
            "thinking.budget_tokens",
            id="adaptive-bool-budget",
        ),
        pytest.param(
            {"type": "disabled", "budget_tokens": "1024"},
            "thinking.budget_tokens",
            id="disabled-string-budget",
        ),
    ],
)
def test_openai_chat_rejects_malformed_thinking_before_router_calls(
    monkeypatch,
    stream,
    thinking,
    parameter,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app, raise_server_exceptions=False)
    router = _ThinkingRouteRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "thinking": thinking,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["param"] == parameter
    assert "data:" not in response.text
    router.chat_completion.assert_not_awaited()
    router.prepare_chat_completion_stream.assert_not_awaited()
    router.chat_completion_stream.assert_not_awaited()


def test_openai_thinking_with_unknown_sibling_is_accepted():
    from router_maestro.server.schemas.openai import OpenAIThinkingConfig

    # Anthropic's `display` sibling (summarized/omitted) is sent by real clients.
    cfg = OpenAIThinkingConfig.model_validate({"type": "adaptive", "display": "summarized"})
    assert cfg.type == "adaptive"


def test_openai_thinking_invalid_budget_still_rejected():
    from pydantic import ValidationError

    from router_maestro.server.schemas.openai import OpenAIThinkingConfig

    with pytest.raises(ValidationError):
        OpenAIThinkingConfig.model_validate({"type": "enabled", "budget_tokens": -1})



@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize(
    ("thinking", "expected_type", "expected_budget"),
    [
        pytest.param(
            {"type": "enabled", "budget_tokens": 1024},
            "enabled",
            1024,
            id="enabled-positive-budget",
        ),
        pytest.param(
            {"type": "enabled"},
            "enabled",
            None,
            id="enabled-default-budget",
        ),
        pytest.param(
            {"type": "adaptive"},
            "adaptive",
            None,
            id="adaptive-type-only",
        ),
        pytest.param(
            {"type": "adaptive", "budget_tokens": 4096},
            "adaptive",
            4096,
            id="adaptive-positive-budget",
        ),
        pytest.param(
            {"type": "disabled"},
            "disabled",
            None,
            id="disabled-type-only",
        ),
        pytest.param(
            {"type": "disabled", "budget_tokens": 4096},
            "disabled",
            None,
            id="disabled-clears-budget",
        ),
    ],
)
def test_openai_chat_preserves_valid_thinking_contract(
    monkeypatch,
    stream,
    thinking,
    expected_type,
    expected_budget,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    router = _ThinkingRouteRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "thinking": thinking,
        },
    )

    assert response.status_code == 200
    assert router.request is not None
    assert router.request.thinking_type == expected_type
    assert router.request.thinking_budget == expected_budget


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_openai_chat_validates_thinking_even_when_reasoning_effort_takes_precedence(
    monkeypatch,
    stream,
):
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app, raise_server_exceptions=False)
    router = _ThinkingRouteRouter()
    monkeypatch.setattr("router_maestro.server.routes.chat.get_router", lambda: router)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            "reasoning_effort": "high",
            "thinking": {"type": "enabled", "budget_tokens": True},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "thinking.budget_tokens"
    router.chat_completion.assert_not_awaited()
    router.prepare_chat_completion_stream.assert_not_awaited()
    router.chat_completion_stream.assert_not_awaited()
