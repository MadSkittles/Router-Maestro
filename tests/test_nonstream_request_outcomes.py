"""HTTP non-stream routes preserve provider semantics in RequestContext."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.providers import (
    ChatResponse,
    ResponseStatus,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
)
from router_maestro.providers import (
    ResponsesResponse as InternalResponsesResponse,
)
from router_maestro.routing.capabilities import CapabilitySupport
from router_maestro.routing.model_ref import ModelRef
from router_maestro.runtime import RequestContext, RequestContextMiddleware, current_request_context
from router_maestro.server.routes.anthropic import router as anthropic_router
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.server.routes.gemini import router as gemini_router
from router_maestro.server.routes.responses import router as responses_router
from router_maestro.utils.responses_bridge import responses_response_to_chat_response


class _Snapshot:
    revision = "nonstream-outcome-revision"

    def __init__(self) -> None:
        self._config = PrioritiesConfig()

    @property
    def config(self) -> PrioritiesConfig:
        return self._config.model_copy(deep=True)


class _Lease:
    generation_id = 1

    def __init__(self, router, snapshot: _Snapshot) -> None:
        self.router = router
        self.config_snapshot = snapshot
        self.release_count = 0

    async def release(self) -> None:
        self.release_count += 1


class _Owner:
    def __init__(self, lease: _Lease) -> None:
        self.lease = lease

    async def start(self, _snapshot: _Snapshot) -> None:
        return None

    async def acquire(self) -> _Lease:
        return self.lease


class _Repository:
    def __init__(self, snapshot: _Snapshot) -> None:
        self.snapshot = snapshot

    def read(self) -> _Snapshot:
        return self.snapshot


class _CandidateCapabilities:
    max_output_tokens = 8

    def feature(self, _feature) -> CapabilitySupport:
        return CapabilitySupport.UNSUPPORTED


class _Backend:
    def __init__(
        self,
        *,
        chat_response: ChatResponse | None = None,
        responses_response: InternalResponsesResponse | None = None,
        bridge_response: InternalResponsesResponse | None = None,
    ) -> None:
        self.chat_response = chat_response
        self.responses_response = responses_response
        self.bridge_response = bridge_response
        self.contexts: list[RequestContext] = []
        self.candidate = SimpleNamespace(
            model=ModelRef("test-provider", "test-model"),
            capabilities=_CandidateCapabilities(),
        )

    async def plan_chat_completion(self, _request, *, stream: bool):
        assert stream is False
        return SimpleNamespace(primary=self.candidate, prevalidation_fallbacks=())

    def prepare_planned_chat_completion(self, _plan, _request, **_kwargs):
        return object()

    async def chat_completion(self, _request, **_kwargs):
        self.contexts.append(current_request_context())
        if self.bridge_response is not None:
            bridged = responses_response_to_chat_response(
                self.bridge_response,
                "test-model",
                provider="test-provider",
            )
            return bridged, "test-provider"
        assert self.chat_response is not None
        return self.chat_response, "test-provider"

    async def responses_completion(self, _request, **_kwargs):
        self.contexts.append(current_request_context())
        assert self.responses_response is not None
        return self.responses_response, "test-provider"


def _client(
    route: APIRouter,
    backend: _Backend,
) -> tuple[TestClient, _Lease]:
    snapshot = _Snapshot()
    lease = _Lease(backend, snapshot)
    app = FastAPI()
    app.state.runtime_config_repository = _Repository(snapshot)
    app.state.router_owner = _Owner(lease)
    app.include_router(route)
    app.add_middleware(RequestContextMiddleware)
    return TestClient(app), lease


def _incomplete_outcome() -> TerminalOutcome:
    return TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=ResponseStatus.INCOMPLETE,
        finish_reason="length",
        incomplete_details={"reason": "max_output_tokens", "vendor": "preserved"},
    )


def _assert_context(
    backend: _Backend,
    lease: _Lease,
    expected: TerminalOutcome,
) -> None:
    assert len(backend.contexts) == 1
    context = backend.contexts[0]
    assert context.finalized is True
    assert context.status_code == 200
    assert context.outcome == expected
    assert lease.release_count == 1


def test_chat_nonstream_http_200_records_legacy_incomplete_outcome() -> None:
    expected = TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=ResponseStatus.INCOMPLETE,
        finish_reason="length",
    )
    backend = _Backend(
        chat_response=ChatResponse(
            content="partial",
            model="test-model",
            finish_reason="length",
        )
    )
    client, lease = _client(chat_router, backend)

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"
    _assert_context(backend, lease, expected)


def test_responses_nonstream_http_200_records_incomplete_outcome() -> None:
    expected = _incomplete_outcome()
    backend = _Backend(
        responses_response=InternalResponsesResponse(
            content="partial",
            model="test-model",
            terminal_outcome=expected,
        )
    )
    client, lease = _client(responses_router, backend)

    response = client.post(
        "/api/openai/v1/responses",
        json={"model": "test-model", "input": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "incomplete"
    _assert_context(backend, lease, expected)


def test_anthropic_nonstream_http_200_records_canonical_incomplete_outcome() -> None:
    expected = _incomplete_outcome()
    backend = _Backend(
        chat_response=ChatResponse(
            content="partial",
            model="test-model",
            finish_reason="length",
            terminal_outcome=expected,
        )
    )
    client, lease = _client(anthropic_router, backend)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["stop_reason"] == "max_tokens"
    _assert_context(backend, lease, expected)


def test_gemini_nonstream_http_200_records_canonical_incomplete_outcome() -> None:
    expected = _incomplete_outcome()
    backend = _Backend(
        chat_response=ChatResponse(
            content="partial",
            model="test-model",
            finish_reason="length",
            terminal_outcome=expected,
        )
    )
    client, lease = _client(gemini_router, backend)

    response = client.post(
        "/api/gemini/v1beta/models/test-model:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
    )

    assert response.status_code == 200
    assert response.json()["candidates"][0]["finishReason"] == "MAX_TOKENS"
    _assert_context(backend, lease, expected)


@pytest.mark.parametrize("status", [ResponseStatus.FAILED, ResponseStatus.CANCELLED])
def test_native_responses_nonstream_http_200_records_business_terminal_status(
    status: ResponseStatus,
) -> None:
    expected = TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=status,
        error=(
            TerminalError(code="upstream_failed", message="safe failure")
            if status is ResponseStatus.FAILED
            else None
        ),
    )
    backend = _Backend(
        responses_response=InternalResponsesResponse(
            content="partial",
            model="test-model",
            terminal_outcome=expected,
        )
    )
    client, lease = _client(responses_router, backend)

    response = client.post(
        "/api/openai/v1/responses",
        json={"model": "test-model", "input": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == status.value
    _assert_context(backend, lease, expected)


@pytest.mark.parametrize("status", [ResponseStatus.FAILED, ResponseStatus.CANCELLED])
@pytest.mark.parametrize(
    ("route", "path", "payload", "error_key"),
    [
        (
            chat_router,
            "/api/openai/v1/chat/completions",
            {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
            "error",
        ),
        (
            anthropic_router,
            "/v1/messages",
            {
                "model": "test-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello"}],
            },
            "error",
        ),
        (
            gemini_router,
            "/api/gemini/v1beta/models/test-model:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            "error",
        ),
    ],
    ids=["chat", "anthropic", "gemini"],
)
def test_responses_bridge_failure_keeps_protocol_error_semantics(
    status: ResponseStatus,
    route: APIRouter,
    path: str,
    payload: dict,
    error_key: str,
) -> None:
    upstream = InternalResponsesResponse(
        content="partial",
        model="test-model",
        terminal_outcome=TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=status,
            error=(
                TerminalError(code="upstream_failed", message="safe failure")
                if status is ResponseStatus.FAILED
                else None
            ),
        ),
    )
    backend = _Backend(bridge_response=upstream)
    client, lease = _client(route, backend)

    response = client.post(path, json=payload)

    assert response.status_code == 502
    assert error_key in response.json()
    assert len(backend.contexts) == 1
    context = backend.contexts[0]
    assert context.finalized is True
    assert context.status_code == 502
    assert context.outcome is not None
    assert context.outcome.response_status is ResponseStatus.FAILED
    assert lease.release_count == 1
