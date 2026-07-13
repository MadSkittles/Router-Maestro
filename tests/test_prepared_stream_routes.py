"""Streaming routes execute the exact immutable plan they validated."""

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.config import PrioritiesConfig
from router_maestro.providers import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
)
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    ModelCapabilities,
    Operation,
    ProviderCapabilities,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import (
    PreparedChatStream,
    PreparedResponsesStream,
    RouteCandidate,
    RoutePlan,
)
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.server.routes.anthropic import router as anthropic_router
from router_maestro.server.routes.chat import router as chat_router
from router_maestro.server.routes.gemini import router as gemini_router
from router_maestro.server.routes.responses import router as responses_router
from router_maestro.utils.cache import TTLCache


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(anthropic_router)
    app.include_router(gemini_router)
    app.include_router(responses_router)
    return TestClient(app)


def _chat_router(plan: object) -> MagicMock:
    router = MagicMock()
    router.prepare_chat_completion_stream = AsyncMock(return_value=plan)
    router.plan_chat_completion = AsyncMock(return_value=plan)
    router.prepare_planned_chat_completion = MagicMock(return_value=plan)
    router.validate_chat_request = AsyncMock()
    router.get_model_info = AsyncMock(return_value=None)
    router._resolve_provider = AsyncMock(return_value=("test", "m", MagicMock()))
    return router


def _responses_router(plan: object) -> MagicMock:
    router = MagicMock()
    router.prepare_responses_completion_stream = AsyncMock(return_value=plan)
    router.validate_responses_request = AsyncMock()
    return router


def _capture_stream(captured: dict[str, object]):
    async def stream(*_args, prepared_plan=None, **_kwargs) -> AsyncIterator[str]:
        captured["plan"] = prepared_plan
        if False:
            yield ""

    return stream


class _PreparedProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        *,
        validation_error: ProviderError | None = None,
        stream_error: ProviderError | None = None,
        completion_error: ProviderError | None = None,
    ) -> None:
        self.name = name
        self.validation_error = validation_error
        self.stream_error = stream_error
        self.completion_error = completion_error
        self.chat_validations: list[str] = []
        self.responses_validations: list[str] = []
        self.chat_opens: list[str] = []
        self.responses_opens: list[str] = []
        self.chat_validation_requests: list[ChatRequest] = []
        self.responses_validation_requests: list[ResponsesRequest] = []
        self.chat_open_requests: list[ChatRequest] = []
        self.responses_open_requests: list[ResponsesRequest] = []

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        self.chat_opens.append(request.model)
        self.chat_open_requests.append(deepcopy(request))
        if self.completion_error is not None:
            raise self.completion_error
        return ChatResponse(content="ok", model=request.model)

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        self.chat_opens.append(request.model)
        self.chat_open_requests.append(deepcopy(request))
        if self.stream_error is not None:
            raise self.stream_error
        yield ChatStreamChunk(content="ok", finish_reason="stop")

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        self.responses_opens.append(request.model)
        self.responses_open_requests.append(deepcopy(request))
        if self.completion_error is not None:
            raise self.completion_error
        return ResponsesResponse(content="ok", model=request.model)

    async def responses_completion_stream(
        self,
        request: ResponsesRequest,
    ) -> AsyncIterator[ResponsesStreamChunk]:
        self.responses_opens.append(request.model)
        self.responses_open_requests.append(deepcopy(request))
        if self.stream_error is not None:
            raise self.stream_error
        yield ResponsesStreamChunk(content="ok", finish_reason="stop")

    async def list_models(self) -> list[ModelInfo]:
        return []

    def is_authenticated(self) -> bool:
        return True

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            operations=frozenset(
                {
                    Operation.CHAT,
                    Operation.CHAT_STREAM,
                    Operation.RESPONSES,
                    Operation.RESPONSES_STREAM,
                }
            )
        )

    def validate_chat_request(self, request: ChatRequest, *, stream: bool) -> None:
        self.chat_validations.append(request.model)
        self.chat_validation_requests.append(deepcopy(request))
        if self.validation_error is not None:
            raise self.validation_error

    def validate_responses_request(self, request: ResponsesRequest) -> None:
        self.responses_validations.append(request.model)
        self.responses_validation_requests.append(deepcopy(request))
        if self.validation_error is not None:
            raise self.validation_error


def _candidate(provider: BaseProvider, model: str, operation: Operation) -> RouteCandidate:
    ref = ModelRef(provider.name, model)
    features = RequestFeatures()
    capabilities = ModelCapabilities(
        model=ref,
        operations={operation: CapabilitySupport.SUPPORTED},
    )
    return RouteCandidate(
        model=ref,
        provider=provider,
        capabilities=capabilities,
        evaluated_operation=operation,
        evaluated_features=features,
        support=CapabilitySupport.SUPPORTED,
    )


def _unsupported_candidate(
    provider: BaseProvider,
    model: str,
    operation: Operation,
) -> RouteCandidate:
    ref = ModelRef(provider.name, model)
    features = RequestFeatures()
    capabilities = ModelCapabilities(
        model=ref,
        operations={operation: CapabilitySupport.UNSUPPORTED},
    )
    return RouteCandidate(
        model=ref,
        provider=provider,
        capabilities=capabilities,
        evaluated_operation=operation,
        evaluated_features=features,
        support=CapabilitySupport.UNSUPPORTED,
    )


def _plan(
    operation: Operation,
    primary: BaseProvider,
    *fallbacks: BaseProvider,
) -> RoutePlan:
    return RoutePlan(
        operation=operation,
        features=RequestFeatures(),
        primary=_candidate(primary, "primary-model", operation),
        fallbacks=tuple(
            _candidate(provider, f"fallback-{index}-model", operation)
            for index, provider in enumerate(fallbacks, start=1)
        ),
        explicit=False,
    )


def _retryable_open_error() -> ProviderError:
    return ProviderError(
        "primary failed before first chunk",
        status_code=503,
        retryable=True,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )


def _unsupported_top_p() -> RequestOptionError:
    return RequestOptionError("top_p is unsupported", parameter="top_p")


def test_public_plan_prevalidation_keeps_compatible_fallback_slot() -> None:
    primary = _PreparedProvider("primary")
    rejected = _PreparedProvider("rejected", validation_error=_unsupported_top_p())
    retained = _PreparedProvider("retained")
    plan = _plan(Operation.CHAT, primary, rejected, retained)

    validated = Router.prevalidate_plan(
        plan,
        lambda candidate: candidate.provider.validate_chat_request(
            ChatRequest(model=candidate.model.upstream_id, messages=[], top_p=0.5),
            stream=False,
        ),
    )

    assert validated.fallbacks == (plan.fallbacks[1],)


def _planning_router(*providers: _PreparedProvider) -> Router:
    router = Router.__new__(Router)
    router.providers = {provider.name: provider for provider in providers}
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._providers_ttl.set(True)
    priorities: list[str] = []
    for provider in providers:
        model_id = f"{provider.name}-model"
        model = ModelInfo(
            id=model_id,
            name=model_id,
            provider=provider.name,
            operation_capabilities={
                Operation.CHAT.value: True,
                Operation.CHAT_STREAM.value: True,
                Operation.RESPONSES.value: True,
                Operation.RESPONSES_STREAM.value: True,
            },
        )
        router._models_cache[model_id] = (provider.name, model)
        router._models_cache[f"{provider.name}/{model_id}"] = (provider.name, model)
        priorities.append(f"{provider.name}/{model_id}")
    router._models_cache_ttl.set(True)
    router._priorities_cache.set(
        PrioritiesConfig(
            priorities=priorities,
            fallback={"strategy": "priority", "maxRetries": 1},
        )
    )
    return router


async def _execute_planned_request(
    router: Router,
    protocol: str,
    stream: bool,
    *,
    fallback: bool = True,
    use_prepared_plan: bool = True,
) -> tuple[str, PreparedChatStream | PreparedResponsesStream | None]:
    if protocol == "chat":
        request = ChatRequest(
            model="router-maestro",
            messages=[],
            top_p=0.5,
            stream=stream,
        )
        if stream:
            if not use_prepared_plan:
                chunks, provider_name = await router.chat_completion_stream(
                    request,
                    fallback=fallback,
                )
                assert [chunk async for chunk in chunks]
                return provider_name, None
            plan = await router.prepare_chat_completion_stream(request, fallback=fallback)
            chunks, provider_name = await router.chat_completion_stream(
                request,
                fallback=fallback,
                prepared_plan=plan,
            )
            assert [chunk async for chunk in chunks]
            return provider_name, plan
        _response, provider_name = await router.chat_completion(request, fallback=fallback)
        return provider_name, None

    request = ResponsesRequest(
        model="router-maestro",
        input="hi",
        top_p=0.5,
        stream=stream,
    )
    if stream:
        if not use_prepared_plan:
            chunks, provider_name = await router.responses_completion_stream(
                request,
                fallback=fallback,
            )
            assert [chunk async for chunk in chunks]
            return provider_name, None
        plan = await router.prepare_responses_completion_stream(request, fallback=fallback)
        chunks, provider_name = await router.responses_completion_stream(
            request,
            fallback=fallback,
            prepared_plan=plan,
        )
        assert [chunk async for chunk in chunks]
        return provider_name, plan
    _response, provider_name = await router.responses_completion(request, fallback=fallback)
    return provider_name, None


def _provider_activity(
    provider: _PreparedProvider,
    protocol: str,
) -> tuple[list[str], list[str]]:
    if protocol == "chat":
        return provider.chat_validations, provider.chat_opens
    return provider.responses_validations, provider.responses_opens


def _snapshot_request(protocol: str, *, stream: bool = False) -> ChatRequest | ResponsesRequest:
    if protocol == "chat":
        return ChatRequest(
            model="router-maestro",
            messages=[Message(role="user", content="original")],
            top_p=0.5,
            stream=stream,
        )
    return ResponsesRequest(
        model="router-maestro",
        input=[{"role": "user", "content": [{"type": "input_text", "text": "original"}]}],
        top_p=0.5,
        stream=stream,
    )


def _mutate_snapshot_request(request: ChatRequest | ResponsesRequest) -> None:
    request.top_p = 0.9
    request.tools = [{"type": "function", "name": "injected"}]
    if isinstance(request, ChatRequest):
        request.messages[0].content = "mutated"
        return
    assert isinstance(request.input, list)
    request.input[0]["content"][0]["text"] = "mutated"


def _assert_original_snapshot(request: ChatRequest | ResponsesRequest) -> None:
    assert request.top_p == 0.5
    assert request.tools is None
    if isinstance(request, ChatRequest):
        assert request.messages[0].content == "original"
        return
    assert isinstance(request.input, list)
    assert request.input[0]["content"][0]["text"] == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_nonstream_entry_snapshot_survives_mutation_during_plan_await(
    protocol: str,
) -> None:
    provider = _PreparedProvider("primary")
    router = Router.__new__(Router)
    plan_started = asyncio.Event()
    release_plan = asyncio.Event()
    planned_features: list[RequestFeatures] = []

    async def blocked_plan(_model: str, operation: Operation, features: RequestFeatures):
        planned_features.append(features)
        plan_started.set()
        await release_plan.wait()
        return _plan(operation, provider)

    router._plan_completion_route = blocked_plan  # type: ignore[method-assign]
    request = _snapshot_request(protocol)
    if protocol == "chat":
        task = asyncio.create_task(router.chat_completion(request))  # type: ignore[arg-type]
    else:
        task = asyncio.create_task(router.responses_completion(request))  # type: ignore[arg-type]

    await plan_started.wait()
    _mutate_snapshot_request(request)
    release_plan.set()
    await task

    assert planned_features == [RequestFeatures()]
    if protocol == "chat":
        _assert_original_snapshot(provider.chat_validation_requests[0])
        _assert_original_snapshot(provider.chat_open_requests[0])
    else:
        _assert_original_snapshot(provider.responses_validation_requests[0])
        _assert_original_snapshot(provider.responses_open_requests[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_nonstream_fallback_uses_entry_snapshot_after_caller_mutation(
    protocol: str,
) -> None:
    primary = _PreparedProvider("primary")
    fallback = _PreparedProvider("fallback")
    operation = Operation.CHAT if protocol == "chat" else Operation.RESPONSES
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=_plan(operation, primary, fallback))
    primary_started = asyncio.Event()
    release_primary = asyncio.Event()

    async def fail_chat(request: ChatRequest) -> ChatResponse:
        primary.chat_open_requests.append(deepcopy(request))
        primary_started.set()
        await release_primary.wait()
        raise _retryable_open_error()

    async def fail_responses(request: ResponsesRequest) -> ResponsesResponse:
        primary.responses_open_requests.append(deepcopy(request))
        primary_started.set()
        await release_primary.wait()
        raise _retryable_open_error()

    primary.chat_completion = fail_chat  # type: ignore[method-assign]
    primary.responses_completion = fail_responses  # type: ignore[method-assign]
    request = _snapshot_request(protocol)
    if protocol == "chat":
        task = asyncio.create_task(router.chat_completion(request))  # type: ignore[arg-type]
    else:
        task = asyncio.create_task(router.responses_completion(request))  # type: ignore[arg-type]

    await primary_started.wait()
    _mutate_snapshot_request(request)
    release_primary.set()
    await task

    if protocol == "chat":
        _assert_original_snapshot(fallback.chat_open_requests[0])
    else:
        _assert_original_snapshot(fallback.responses_open_requests[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_prepared_stream_captures_entry_snapshot_before_plan_await(
    protocol: str,
) -> None:
    provider = _PreparedProvider("primary")
    router = Router.__new__(Router)
    plan_started = asyncio.Event()
    release_plan = asyncio.Event()
    planned_features: list[RequestFeatures] = []

    async def blocked_plan(_model: str, operation: Operation, features: RequestFeatures):
        planned_features.append(features)
        plan_started.set()
        await release_plan.wait()
        return _plan(operation, provider)

    router._plan_completion_route = blocked_plan  # type: ignore[method-assign]
    request = _snapshot_request(protocol, stream=True)
    if protocol == "chat":
        task = asyncio.create_task(
            router.prepare_chat_completion_stream(request)  # type: ignore[arg-type]
        )
    else:
        task = asyncio.create_task(
            router.prepare_responses_completion_stream(request)  # type: ignore[arg-type]
        )

    await plan_started.wait()
    _mutate_snapshot_request(request)
    release_plan.set()
    prepared = await task

    assert planned_features == [RequestFeatures()]
    _assert_original_snapshot(prepared.request_for_execution())
    if protocol == "chat":
        _assert_original_snapshot(provider.chat_validation_requests[0])
    else:
        _assert_original_snapshot(provider.responses_validation_requests[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_direct_stream_entry_uses_snapshot_when_caller_mutates_during_plan_await(
    protocol: str,
) -> None:
    provider = _PreparedProvider("primary")
    router = Router.__new__(Router)
    plan_started = asyncio.Event()
    release_plan = asyncio.Event()

    async def blocked_plan(_model: str, operation: Operation, _features: RequestFeatures):
        plan_started.set()
        await release_plan.wait()
        return _plan(operation, provider)

    router._plan_completion_route = blocked_plan  # type: ignore[method-assign]
    request = _snapshot_request(protocol, stream=True)
    if protocol == "chat":
        task = asyncio.create_task(
            router.chat_completion_stream(request)  # type: ignore[arg-type]
        )
    else:
        task = asyncio.create_task(
            router.responses_completion_stream(request)  # type: ignore[arg-type]
        )

    await plan_started.wait()
    _mutate_snapshot_request(request)
    release_plan.set()
    stream, provider_name = await task
    assert [chunk async for chunk in stream]

    assert provider_name == "primary"
    if protocol == "chat":
        _assert_original_snapshot(provider.chat_validation_requests[0])
        _assert_original_snapshot(provider.chat_open_requests[0])
    else:
        _assert_original_snapshot(provider.responses_validation_requests[0])
        _assert_original_snapshot(provider.responses_open_requests[0])


def test_openai_chat_stream_reuses_prepared_plan(client: TestClient) -> None:
    plan = object()
    mocked_router = _chat_router(plan)
    captured: dict[str, object] = {}
    with (
        patch("router_maestro.server.routes.chat.get_router", return_value=mocked_router),
        patch("router_maestro.server.routes.chat.stream_response", _capture_stream(captured)),
    ):
        response = client.post(
            "/api/openai/v1/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    mocked_router.prepare_chat_completion_stream.assert_awaited_once()
    mocked_router.validate_chat_request.assert_not_awaited()
    assert captured["plan"] is plan


def test_anthropic_stream_reuses_prepared_plan(client: TestClient) -> None:
    plan = MagicMock()
    mocked_router = _chat_router(plan)
    captured: dict[str, object] = {}
    with (
        patch("router_maestro.server.routes.anthropic.get_router", return_value=mocked_router),
        patch("router_maestro.server.routes.anthropic.stream_response", _capture_stream(captured)),
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
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    mocked_router.plan_chat_completion.assert_awaited_once()
    assert mocked_router.plan_chat_completion.await_args.kwargs == {"stream": True}
    mocked_router.prepare_planned_chat_completion.assert_called_once()
    assert mocked_router.prepare_planned_chat_completion.call_args.args[0] is plan
    mocked_router.prepare_chat_completion_stream.assert_not_awaited()
    mocked_router.validate_chat_request.assert_not_awaited()
    assert captured["plan"] is plan


def test_gemini_stream_reuses_prepared_plan(client: TestClient) -> None:
    plan = object()
    mocked_router = _chat_router(plan)
    captured: dict[str, object] = {}
    with (
        patch("router_maestro.server.routes.gemini.get_router", return_value=mocked_router),
        patch("router_maestro.server.routes.gemini._stream_response", _capture_stream(captured)),
    ):
        response = client.post(
            "/api/gemini/v1beta/models/m:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )

    assert response.status_code == 200
    mocked_router.prepare_chat_completion_stream.assert_awaited_once()
    mocked_router.validate_chat_request.assert_not_awaited()
    assert captured["plan"] is plan


def test_responses_stream_reuses_prepared_plan(client: TestClient) -> None:
    plan = object()
    mocked_router = _responses_router(plan)
    captured: dict[str, object] = {}
    with (
        patch("router_maestro.server.routes.responses.get_router", return_value=mocked_router),
        patch("router_maestro.server.routes.responses.stream_response", _capture_stream(captured)),
    ):
        response = client.post(
            "/api/openai/v1/responses",
            json={"model": "m", "stream": True, "input": "hi"},
        )

    assert response.status_code == 200
    mocked_router.prepare_responses_completion_stream.assert_awaited_once()
    mocked_router.validate_responses_request.assert_not_awaited()
    assert captured["plan"] is plan


@pytest.mark.asyncio
async def test_prepare_chat_stream_filters_option_incompatible_fallbacks() -> None:
    primary = _PreparedProvider("primary")
    rejected = _PreparedProvider("rejected", validation_error=_unsupported_top_p())
    retained = _PreparedProvider("retained")
    plan = _plan(Operation.CHAT_STREAM, primary, rejected, retained)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)
    request = ChatRequest(model="router-maestro", messages=[], top_p=0.5, stream=True)

    prepared = await router.prepare_chat_completion_stream(request)

    router._plan_completion_route.assert_awaited_once()
    assert prepared.plan.primary is plan.primary
    assert prepared.plan.fallbacks == (plan.fallbacks[1],)
    assert primary.chat_validations == ["primary-model"]
    assert rejected.chat_validations == ["fallback-1-model"]
    assert retained.chat_validations == ["fallback-2-model"]


@pytest.mark.asyncio
async def test_prepare_responses_stream_filters_option_incompatible_fallbacks() -> None:
    primary = _PreparedProvider("primary")
    rejected = _PreparedProvider("rejected", validation_error=_unsupported_top_p())
    retained = _PreparedProvider("retained")
    plan = _plan(Operation.RESPONSES_STREAM, primary, rejected, retained)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)
    request = ResponsesRequest(model="router-maestro", input="hi", top_p=0.5, stream=True)

    prepared = await router.prepare_responses_completion_stream(request)

    router._plan_completion_route.assert_awaited_once()
    assert prepared.plan.primary is plan.primary
    assert prepared.plan.fallbacks == (plan.fallbacks[1],)
    assert primary.responses_validations == ["primary-model"]
    assert rejected.responses_validations == ["fallback-1-model"]
    assert retained.responses_validations == ["fallback-2-model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [Operation.CHAT_STREAM, Operation.RESPONSES_STREAM])
async def test_prepare_stream_does_not_swallow_non_option_fallback_validation_error(
    operation: Operation,
) -> None:
    primary = _PreparedProvider("primary")
    fatal = ProviderError(
        "validation bug",
        status_code=500,
        retryable=False,
        kind=ProviderFailureKind.UNKNOWN,
    )
    fallback = _PreparedProvider("fallback", validation_error=fatal)
    plan = _plan(operation, primary, fallback)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)

    with pytest.raises(ProviderError) as exc_info:
        if operation is Operation.CHAT_STREAM:
            await router.prepare_chat_completion_stream(
                ChatRequest(model="router-maestro", messages=[], stream=True)
            )
        else:
            await router.prepare_responses_completion_stream(
                ResponsesRequest(model="router-maestro", input="hi", stream=True)
            )

    assert exc_info.value is fatal


@pytest.mark.asyncio
async def test_prepare_chat_stream_primary_option_error_is_not_filtered() -> None:
    error = _unsupported_top_p()
    primary = _PreparedProvider("primary", validation_error=error)
    fallback = _PreparedProvider("fallback")
    plan = _plan(Operation.CHAT_STREAM, primary, fallback)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)

    with pytest.raises(RequestOptionError) as exc_info:
        await router.prepare_chat_completion_stream(
            ChatRequest(model="router-maestro", messages=[], top_p=0.5, stream=True)
        )

    assert exc_info.value is error
    assert fallback.chat_validations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [Operation.CHAT_STREAM, Operation.RESPONSES_STREAM])
async def test_prepare_stream_rejects_static_unsupported_primary_before_adapter_validation(
    operation: Operation,
) -> None:
    primary = _PreparedProvider("primary")
    plan = RoutePlan(
        operation=operation,
        features=RequestFeatures(),
        primary=_unsupported_candidate(primary, "primary-model", operation),
        fallbacks=(),
        explicit=True,
    )
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)

    with pytest.raises(ProviderError) as exc_info:
        if operation is Operation.CHAT_STREAM:
            await router.prepare_chat_completion_stream(
                ChatRequest(model="primary/primary-model", messages=[], stream=True)
            )
        else:
            await router.prepare_responses_completion_stream(
                ResponsesRequest(model="primary/primary-model", input="hi", stream=True)
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert primary.chat_validations == []
    assert primary.responses_validations == []


@pytest.mark.asyncio
async def test_filtered_chat_plan_never_opens_incompatible_fallback() -> None:
    primary = _PreparedProvider("primary", stream_error=_retryable_open_error())
    rejected = _PreparedProvider("rejected", validation_error=_unsupported_top_p())
    plan = _plan(Operation.CHAT_STREAM, primary, rejected)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)
    request = ChatRequest(model="router-maestro", messages=[], top_p=0.5, stream=True)

    prepared = await router.prepare_chat_completion_stream(request)
    with pytest.raises(ProviderError) as exc_info:
        await router.chat_completion_stream(request, prepared_plan=prepared)

    assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_STATUS
    assert rejected.chat_opens == []
    router._plan_completion_route.assert_awaited_once()


@pytest.mark.asyncio
async def test_filtered_responses_plan_never_opens_incompatible_fallback() -> None:
    primary = _PreparedProvider("primary", stream_error=_retryable_open_error())
    rejected = _PreparedProvider("rejected", validation_error=_unsupported_top_p())
    plan = _plan(Operation.RESPONSES_STREAM, primary, rejected)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)
    request = ResponsesRequest(model="router-maestro", input="hi", top_p=0.5, stream=True)

    prepared = await router.prepare_responses_completion_stream(request)
    with pytest.raises(ProviderError) as exc_info:
        await router.responses_completion_stream(request, prepared_plan=prepared)

    assert exc_info.value.kind is ProviderFailureKind.UPSTREAM_STATUS
    assert rejected.responses_opens == []
    router._plan_completion_route.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [Operation.CHAT, Operation.RESPONSES])
async def test_nonstream_skips_option_incompatible_fallback_before_execution(
    operation: Operation,
) -> None:
    primary = _PreparedProvider("primary", completion_error=_retryable_open_error())
    rejected = _PreparedProvider("rejected", validation_error=_unsupported_top_p())
    retained = _PreparedProvider("retained")
    plan = _plan(operation, primary, rejected, retained)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)

    if operation is Operation.CHAT:
        response, provider_name = await router.chat_completion(
            ChatRequest(model="router-maestro", messages=[], top_p=0.5)
        )
        assert response.model == "fallback-2-model"
        assert primary.chat_validations == ["primary-model"]
        assert rejected.chat_validations == ["fallback-1-model"]
        assert retained.chat_validations == ["fallback-2-model"]
        assert primary.chat_opens == ["primary-model"]
        assert rejected.chat_opens == []
        assert retained.chat_opens == ["fallback-2-model"]
    else:
        response, provider_name = await router.responses_completion(
            ResponsesRequest(model="router-maestro", input="hi", top_p=0.5)
        )
        assert response.model == "fallback-2-model"
        assert primary.responses_validations == ["primary-model"]
        assert rejected.responses_validations == ["fallback-1-model"]
        assert retained.responses_validations == ["fallback-2-model"]
        assert primary.responses_opens == ["primary-model"]
        assert rejected.responses_opens == []
        assert retained.responses_opens == ["fallback-2-model"]

    assert provider_name == "retained"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [Operation.CHAT, Operation.RESPONSES])
async def test_nonstream_primary_option_error_remains_client_failure(
    operation: Operation,
) -> None:
    error = _unsupported_top_p()
    primary = _PreparedProvider("primary", validation_error=error)
    fallback = _PreparedProvider("fallback")
    plan = _plan(operation, primary, fallback)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)

    with pytest.raises(RequestOptionError) as exc_info:
        if operation is Operation.CHAT:
            await router.chat_completion(
                ChatRequest(model="router-maestro", messages=[], top_p=0.5)
            )
        else:
            await router.responses_completion(
                ResponsesRequest(model="router-maestro", input="hi", top_p=0.5)
            )

    assert exc_info.value is error
    assert primary.chat_opens == []
    assert primary.responses_opens == []
    assert fallback.chat_validations == []
    assert fallback.responses_validations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
async def test_fallback_limit_counts_option_compatible_candidates_after_prevalidation(
    protocol: str,
    stream: bool,
) -> None:
    primary = _PreparedProvider(
        "p0",
        completion_error=_retryable_open_error(),
        stream_error=_retryable_open_error(),
    )
    rejected = _PreparedProvider("p1", validation_error=_unsupported_top_p())
    compatible = _PreparedProvider("p2")
    router = _planning_router(primary, rejected, compatible)

    provider_name, prepared = await _execute_planned_request(router, protocol, stream)

    assert provider_name == "p2"
    primary_validations, primary_calls = _provider_activity(primary, protocol)
    rejected_validations, rejected_calls = _provider_activity(rejected, protocol)
    compatible_validations, compatible_calls = _provider_activity(compatible, protocol)
    assert primary_validations == ["p0-model"]
    assert rejected_validations == ["p1-model"]
    assert compatible_validations == ["p2-model"]
    assert primary_calls == ["p0-model"]
    assert rejected_calls == []
    assert compatible_calls == ["p2-model"]
    if prepared is not None:
        assert [candidate.model.qualified_id for candidate in prepared.plan.candidates] == [
            "p0/p0-model",
            "p2/p2-model",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
async def test_first_compatible_fallback_remains_only_executable_retry(
    protocol: str,
    stream: bool,
) -> None:
    primary = _PreparedProvider(
        "p0",
        completion_error=_retryable_open_error(),
        stream_error=_retryable_open_error(),
    )
    first = _PreparedProvider("p1")
    beyond_limit = _PreparedProvider("p2")
    router = _planning_router(primary, first, beyond_limit)

    provider_name, prepared = await _execute_planned_request(router, protocol, stream)

    assert provider_name == "p1"
    first_validations, first_calls = _provider_activity(first, protocol)
    beyond_validations, beyond_calls = _provider_activity(beyond_limit, protocol)
    assert first_validations == ["p1-model"]
    assert first_calls == ["p1-model"]
    assert beyond_validations == []
    assert beyond_calls == []
    if prepared is not None:
        assert [candidate.model.qualified_id for candidate in prepared.plan.candidates] == [
            "p0/p0-model",
            "p1/p1-model",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
async def test_fallback_false_still_attempts_only_primary(
    protocol: str,
    stream: bool,
) -> None:
    primary = _PreparedProvider("p0")
    fallback_validation_error = ProviderError(
        "fallback validator must not run",
        status_code=500,
        retryable=False,
        kind=ProviderFailureKind.UNKNOWN,
    )
    fallback = _PreparedProvider("p1", validation_error=fallback_validation_error)
    router = _planning_router(primary, fallback)

    provider_name, prepared = await _execute_planned_request(
        router,
        protocol,
        stream,
        fallback=False,
        use_prepared_plan=False,
    )

    primary_validations, primary_calls = _provider_activity(primary, protocol)
    fallback_validations, fallback_calls = _provider_activity(fallback, protocol)
    assert provider_name == "p0"
    assert primary_validations == ["p0-model"]
    assert primary_calls == ["p0-model"]
    assert fallback_validations == []
    assert fallback_calls == []
    assert prepared is None


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_primary_only_prepared_stream_plan_cannot_enable_unvalidated_fallback(
    protocol: str,
) -> None:
    primary = _PreparedProvider("p0", stream_error=_retryable_open_error())
    fallback = _PreparedProvider(
        "p1",
        validation_error=ProviderError(
            "fallback validator must not run",
            status_code=500,
            retryable=False,
            kind=ProviderFailureKind.UNKNOWN,
        ),
    )
    router = _planning_router(primary, fallback)

    if protocol == "chat":
        request = ChatRequest(model="router-maestro", messages=[], stream=True)
        prepared = await router.prepare_chat_completion_stream(request, fallback=False)
        with pytest.raises(ProviderError):
            await router.chat_completion_stream(
                request,
                fallback=True,
                prepared_plan=prepared,
            )
    else:
        request = ResponsesRequest(model="router-maestro", input="hi", stream=True)
        prepared = await router.prepare_responses_completion_stream(request, fallback=False)
        with pytest.raises(ProviderError):
            await router.responses_completion_stream(
                request,
                fallback=True,
                prepared_plan=prepared,
            )

    fallback_validations, fallback_calls = _provider_activity(fallback, protocol)
    assert prepared.plan.fallbacks == ()
    assert prepared.plan.prevalidation_fallbacks == ()
    assert prepared.plan.fallback_limit == 0
    assert fallback_validations == []
    assert fallback_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prepared_protocol",
    ["chat", "responses"],
    ids=["chat-plan-into-responses", "responses-plan-into-chat"],
)
async def test_prepared_stream_rejects_cross_protocol_reuse_before_provider_io(
    prepared_protocol: str,
) -> None:
    provider = _PreparedProvider("primary")
    router = _planning_router(provider)

    if prepared_protocol == "chat":
        chat_request = ChatRequest(model="router-maestro", messages=[], stream=True)
        prepared = await router.prepare_chat_completion_stream(chat_request)
        provider.chat_validations.clear()

        with pytest.raises(ProviderError) as exc_info:
            await router.responses_completion_stream(
                ResponsesRequest(model="router-maestro", input="hi", stream=True),
                prepared_plan=prepared,
            )
    else:
        responses_request = ResponsesRequest(model="router-maestro", input="hi", stream=True)
        prepared = await router.prepare_responses_completion_stream(responses_request)
        provider.responses_validations.clear()

        with pytest.raises(ProviderError) as exc_info:
            await router.chat_completion_stream(
                ChatRequest(model="router-maestro", messages=[], stream=True),
                prepared_plan=prepared,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert provider.chat_validations == []
    assert provider.responses_validations == []
    assert provider.chat_opens == []
    assert provider.responses_opens == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["model", "option", "nested"],
)
async def test_prepared_chat_stream_rejects_request_changes_before_provider_io(
    mutation: str,
) -> None:
    provider = _PreparedProvider("primary")
    router = _planning_router(provider)
    request = ChatRequest(
        model="router-maestro",
        messages=[Message(role="user", content=[{"type": "text", "text": "original"}])],
        top_p=0.5,
        stream=True,
    )
    prepared = await router.prepare_chat_completion_stream(request)
    provider.chat_validations.clear()

    if mutation == "model":
        request.model = "different-model"
    elif mutation == "option":
        request.top_p = 0.75
    else:
        assert isinstance(request.messages[0].content, list)
        request.messages[0].content[0]["text"] = "mutated"

    with pytest.raises(ProviderError) as exc_info:
        await router.chat_completion_stream(request, prepared_plan=prepared)

    assert exc_info.value.status_code == 400
    assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert provider.chat_validations == []
    assert provider.chat_opens == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["model", "option", "nested"],
)
async def test_prepared_responses_stream_rejects_request_changes_before_provider_io(
    mutation: str,
) -> None:
    provider = _PreparedProvider("primary")
    router = _planning_router(provider)
    request = ResponsesRequest(
        model="router-maestro",
        input=[{"role": "user", "content": [{"type": "input_text", "text": "original"}]}],
        top_p=0.5,
        stream=True,
    )
    prepared = await router.prepare_responses_completion_stream(request)
    provider.responses_validations.clear()

    if mutation == "model":
        request.model = "different-model"
    elif mutation == "option":
        request.top_p = 0.75
    else:
        assert isinstance(request.input, list)
        request.input[0]["content"][0]["text"] = "mutated"

    with pytest.raises(ProviderError) as exc_info:
        await router.responses_completion_stream(request, prepared_plan=prepared)

    assert exc_info.value.status_code == 400
    assert exc_info.value.kind is ProviderFailureKind.CLIENT_REQUEST
    assert provider.responses_validations == []
    assert provider.responses_opens == []


@pytest.mark.parametrize(
    ("protocol", "path", "payload", "patch_target"),
    [
        (
            "chat",
            "/api/openai/v1/chat/completions",
            {
                "model": "router-maestro",
                "stream": True,
                "top_p": 0.5,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "router_maestro.server.routes.chat.get_router",
        ),
        (
            "anthropic",
            "/v1/messages",
            {
                "model": "router-maestro",
                "stream": True,
                "top_p": 0.5,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "router_maestro.server.routes.anthropic.get_router",
        ),
        (
            "gemini",
            "/api/gemini/v1beta/models/router-maestro:streamGenerateContent",
            {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"topP": 0.5},
            },
            "router_maestro.server.routes.gemini.get_router",
        ),
        (
            "responses",
            "/api/openai/v1/responses",
            {"model": "router-maestro", "stream": True, "top_p": 0.5, "input": "hi"},
            "router_maestro.server.routes.responses.get_router",
        ),
    ],
)
def test_protocol_stream_never_opens_option_incompatible_fallback(
    client: TestClient,
    protocol: str,
    path: str,
    payload: dict,
    patch_target: str,
) -> None:
    operation = Operation.RESPONSES_STREAM if protocol == "responses" else Operation.CHAT_STREAM
    primary = _PreparedProvider("primary", stream_error=_retryable_open_error())
    rejected = _PreparedProvider("rejected", validation_error=_unsupported_top_p())
    plan = _plan(operation, primary, rejected)
    router = Router.__new__(Router)
    router._plan_completion_route = AsyncMock(return_value=plan)
    router.get_model_info = AsyncMock(return_value=None)
    router._resolve_provider = AsyncMock(return_value=("primary", "primary-model", primary))

    with patch(patch_target, return_value=router):
        response = client.post(path, json=payload)

    assert response.status_code == 200
    router._plan_completion_route.assert_awaited_once()
    if protocol == "responses":
        assert rejected.responses_validations == ["fallback-1-model"]
        assert rejected.responses_opens == []
    else:
        assert rejected.chat_validations == ["fallback-1-model"]
        assert rejected.chat_opens == []
