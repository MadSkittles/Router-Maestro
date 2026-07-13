"""Shared fallback-attempt policy contracts for stream and non-stream execution."""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any, Literal

import httpx
import pytest

from router_maestro.providers import (
    ChatRequest,
    ChatStreamChunk,
    Message,
    ProviderError,
    ProviderFailureKind,
    ResponsesRequest,
    ResponsesStreamChunk,
)
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan
from router_maestro.routing.router import Router

Mode = Literal["nonstream", "stream"]


class _TrackedStream(AsyncIterator[ChatStreamChunk]):
    def __init__(
        self,
        contents: list[str] | None = None,
        error: ProviderError | None = None,
    ) -> None:
        self._contents = list(contents or [])
        self._error = error
        self._index = 0
        self._raised = False
        self.close_calls = 0

    def __aiter__(self) -> _TrackedStream:
        return self

    async def __anext__(self) -> ChatStreamChunk:
        if self._index < len(self._contents):
            content = self._contents[self._index]
            self._index += 1
            return ChatStreamChunk(content=content)
        if self._error is not None and not self._raised:
            self._raised = True
            raise self._error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1


class _AttemptProvider:
    def __init__(
        self,
        name: str,
        order: list[str],
        *,
        result: str = "ok",
        error: ProviderError | None = None,
        ensure_error: ProviderError | None = None,
        stream_contents: list[str] | None = None,
    ) -> None:
        self.name = name
        self.order = order
        self.result = result
        self.error = error
        self.ensure_error = ensure_error
        self.ensure_calls = 0
        self.call_count = 0
        self.stream = _TrackedStream(stream_contents or ([result] if error is None else []), error)

    async def ensure_token(self) -> None:
        self.ensure_calls += 1
        self.order.append(self.name)
        if self.ensure_error is not None:
            raise self.ensure_error

    async def complete(self, _request: object) -> str:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.result

    def open_stream(self, _request: object) -> AsyncIterator[ChatStreamChunk]:
        self.call_count += 1
        return self.stream


def _failure(
    message: str,
    *,
    kind: ProviderFailureKind,
    retryable: bool,
    status_code: int = 502,
    upstream_status_code: int | None = None,
    cause: BaseException | None = None,
) -> ProviderError:
    return ProviderError(
        message,
        status_code=status_code,
        retryable=retryable,
        kind=kind,
        upstream_status_code=upstream_status_code,
        cause=cause,
    )


def _plan(
    operation: Operation,
    providers: list[_AttemptProvider],
    *,
    supports: list[CapabilitySupport] | None = None,
    explicit: bool = False,
) -> RoutePlan:
    supports = supports or [CapabilitySupport.SUPPORTED] * len(providers)
    candidates: list[RouteCandidate] = []
    for provider, support in zip(providers, supports, strict=True):
        ref = ModelRef(provider.name, f"{provider.name}-model")
        features = RequestFeatures()
        capabilities = ModelCapabilities(
            model=ref,
            operations={operation: support},
        )
        candidates.append(
            RouteCandidate(
                model=ref,
                provider=provider,
                capabilities=capabilities,
                evaluated_operation=operation,
                evaluated_features=features,
                support=support,
            )
        )
    return RoutePlan(
        operation=operation,
        features=features,
        primary=candidates[0],
        fallbacks=tuple(candidates[1:]),
        explicit=explicit,
    )


async def _execute(
    mode: Mode,
    providers: list[_AttemptProvider],
    *,
    supports: list[CapabilitySupport] | None = None,
    explicit: bool = False,
    fallback: bool = True,
) -> tuple[Any, str]:
    router = Router.__new__(Router)
    operation = Operation.CHAT if mode == "nonstream" else Operation.CHAT_STREAM
    plan = _plan(operation, providers, supports=supports, explicit=explicit)
    if mode == "nonstream":
        return await router._execute_plan_nonstream(
            plan,
            object(),
            fallback,
            lambda request, _model: request,
            lambda provider, request: provider.complete(request),
        )

    stream, provider_name = await router._execute_plan_stream(
        plan,
        object(),
        fallback,
        lambda request, _model: request,
        lambda provider, request: provider.open_stream(request),
        "test",
    )
    return [chunk.content async for chunk in stream], provider_name


def _request_payload(request: ChatRequest | ResponsesRequest) -> dict[str, Any]:
    """Snapshot request data while ignoring the candidate-specific model ID."""
    return {name: deepcopy(value) for name, value in vars(request).items() if name != "model"}


async def _exercise_candidate_request_isolation(
    mode: Mode,
    request: ChatRequest | ResponsesRequest,
    operation: Operation,
    build_request: Any,
    mutate: Any,
) -> tuple[list[ChatRequest | ResponsesRequest], Any, str]:
    """Run a retryable primary mutation through the real RoutePlan executor."""
    order: list[str] = []
    primary = _AttemptProvider("primary", order)
    secondary = _AttemptProvider("secondary", order)
    router = Router.__new__(Router)
    plan = _plan(operation, [primary, secondary])
    seen: list[ChatRequest | ResponsesRequest] = []
    retryable = _failure(
        "primary failed after mutating its candidate request",
        kind=ProviderFailureKind.TRANSPORT,
        retryable=True,
    )

    if mode == "nonstream":

        async def complete(provider: _AttemptProvider, candidate_request: Any) -> str:
            seen.append(candidate_request)
            if provider is primary:
                mutate(candidate_request)
                raise retryable
            return "secondary success"

        result, provider_name = await router._execute_plan_nonstream(
            plan,
            request,
            True,
            build_request,
            complete,
        )
        return seen, result, provider_name

    def open_stream(
        provider: _AttemptProvider,
        candidate_request: Any,
    ) -> AsyncIterator[ChatStreamChunk | ResponsesStreamChunk]:
        async def chunks() -> AsyncIterator[ChatStreamChunk | ResponsesStreamChunk]:
            seen.append(candidate_request)
            if provider is primary:
                mutate(candidate_request)
                raise retryable
            if operation is Operation.CHAT_STREAM:
                yield ChatStreamChunk(content="secondary success")
            else:
                yield ResponsesStreamChunk(content="secondary success")

        return chunks()

    stream, provider_name = await router._execute_plan_stream(
        plan,
        request,
        True,
        build_request,
        open_stream,
        "isolation-test",
    )
    result = [chunk.content async for chunk in stream]
    return seen, result, provider_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "operation"),
    [
        ("nonstream", Operation.CHAT),
        ("stream", Operation.CHAT_STREAM),
    ],
)
async def test_chat_candidate_requests_deeply_isolate_fallback_and_caller(
    mode: Mode,
    operation: Operation,
) -> None:
    request = ChatRequest(
        model="router-maestro",
        messages=[
            Message(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "original-image"},
                    }
                ],
            )
        ],
        stream=mode == "stream",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "original-tool",
                    "parameters": {"properties": {"city": {"enum": ["original-city"]}}},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "original-tool"}},
        stop=["original-stop"],
        stop_sequences=["original-sequence"],
        metadata={"trace": {"tags": ["original-tag"]}},
        provider_extensions={"vendor": {"flags": ["original-flag"]}},
    )
    expected = _request_payload(request)

    def mutate(candidate: ChatRequest) -> None:
        content = candidate.messages[0].content
        assert isinstance(content, list)
        content[0]["image_url"]["url"] = "polluted-image"
        assert candidate.tools is not None
        candidate.tools[0]["function"]["parameters"]["properties"]["city"]["enum"][0] = (
            "polluted-city"
        )
        assert isinstance(candidate.tool_choice, dict)
        candidate.tool_choice["function"]["name"] = "polluted-tool"
        assert isinstance(candidate.stop, list)
        candidate.stop[0] = "polluted-stop"
        assert candidate.stop_sequences is not None
        candidate.stop_sequences[0] = "polluted-sequence"
        assert candidate.metadata is not None
        candidate.metadata["trace"]["tags"][0] = "polluted-tag"
        candidate.provider_extensions["vendor"]["flags"][0] = "polluted-flag"

    router = Router.__new__(Router)
    seen, result, provider_name = await _exercise_candidate_request_isolation(
        mode,
        request,
        operation,
        router._create_request_with_model,
        mutate,
    )

    assert provider_name == "secondary"
    assert result == ("secondary success" if mode == "nonstream" else ["secondary success"])
    assert len(seen) == 2
    primary, fallback = seen
    assert isinstance(primary, ChatRequest)
    assert isinstance(fallback, ChatRequest)
    assert _request_payload(fallback) == expected
    assert _request_payload(request) == expected

    assert primary.messages is not fallback.messages
    assert fallback.messages is not request.messages
    assert primary.messages[0] is not fallback.messages[0]
    assert fallback.messages[0] is not request.messages[0]
    assert primary.messages[0].content is not fallback.messages[0].content
    assert fallback.messages[0].content is not request.messages[0].content
    for field_name in (
        "tools",
        "tool_choice",
        "stop",
        "stop_sequences",
        "metadata",
        "provider_extensions",
        "extra",
    ):
        assert getattr(primary, field_name) is not getattr(fallback, field_name)
        assert getattr(fallback, field_name) is not getattr(request, field_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "operation"),
    [
        ("nonstream", Operation.RESPONSES),
        ("stream", Operation.RESPONSES_STREAM),
    ],
)
async def test_responses_candidate_requests_deeply_isolate_fallback_and_caller(
    mode: Mode,
    operation: Operation,
) -> None:
    request = ResponsesRequest(
        model="router-maestro",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "original-input",
                        "annotations": [{"label": "original-label"}],
                    }
                ],
            }
        ],
        stream=mode == "stream",
        tools=[
            {
                "type": "function",
                "name": "original-tool",
                "parameters": {"properties": {"city": {"enum": ["original-city"]}}},
            }
        ],
        tool_choice={
            "type": "function",
            "name": "original-tool",
            "vendor": {"flags": ["original-choice"]},
        },
        metadata={"trace": {"tags": ["original-tag"]}},
        provider_extensions={"vendor": {"flags": ["original-flag"]}},
    )
    expected = _request_payload(request)

    def mutate(candidate: ResponsesRequest) -> None:
        assert isinstance(candidate.input, list)
        candidate.input[0]["content"][0]["annotations"][0]["label"] = "polluted-label"
        assert candidate.tools is not None
        candidate.tools[0]["parameters"]["properties"]["city"]["enum"][0] = "polluted-city"
        assert isinstance(candidate.tool_choice, dict)
        candidate.tool_choice["vendor"]["flags"][0] = "polluted-choice"
        assert candidate.metadata is not None
        candidate.metadata["trace"]["tags"][0] = "polluted-tag"
        candidate.provider_extensions["vendor"]["flags"][0] = "polluted-flag"

    router = Router.__new__(Router)
    seen, result, provider_name = await _exercise_candidate_request_isolation(
        mode,
        request,
        operation,
        router._create_responses_request_with_model,
        mutate,
    )

    assert provider_name == "secondary"
    assert result == ("secondary success" if mode == "nonstream" else ["secondary success"])
    assert len(seen) == 2
    primary, fallback = seen
    assert isinstance(primary, ResponsesRequest)
    assert isinstance(fallback, ResponsesRequest)
    assert _request_payload(fallback) == expected
    assert _request_payload(request) == expected

    for field_name in (
        "input",
        "tools",
        "tool_choice",
        "metadata",
        "provider_extensions",
    ):
        assert getattr(primary, field_name) is not getattr(fallback, field_name)
        assert getattr(fallback, field_name) is not getattr(request, field_name)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_retryable_primary_advances_to_secondary(mode: Mode) -> None:
    order: list[str] = []
    primary = _AttemptProvider(
        "primary",
        order,
        error=_failure(
            "primary unavailable",
            kind=ProviderFailureKind.TRANSPORT,
            retryable=True,
        ),
    )
    secondary = _AttemptProvider("secondary", order, result="secondary success")

    result, provider_name = await _execute(mode, [primary, secondary])

    assert provider_name == "secondary"
    assert result == ("secondary success" if mode == "nonstream" else ["secondary success"])
    assert order == ["primary", "secondary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_secondary_fatal_stops_without_trying_tertiary(mode: Mode) -> None:
    order: list[str] = []
    primary = _AttemptProvider(
        "primary",
        order,
        error=_failure(
            "primary unavailable",
            kind=ProviderFailureKind.TRANSPORT,
            retryable=True,
        ),
    )
    secondary_error = _failure(
        "bad request",
        kind=ProviderFailureKind.CLIENT_REQUEST,
        retryable=False,
        status_code=400,
    )
    secondary = _AttemptProvider("secondary", order, error=secondary_error)
    tertiary = _AttemptProvider("tertiary", order, result="must not run")

    with pytest.raises(ProviderError) as exc_info:
        await _execute(mode, [primary, secondary, tertiary])

    assert exc_info.value is not secondary_error
    assert exc_info.value.__cause__ is secondary_error
    assert order == ["primary", "secondary"]
    assert tertiary.ensure_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_all_retryable_exhaustion_raises_last_failure_with_read_only_ledger(
    mode: Mode,
) -> None:
    order: list[str] = []
    failures = [
        _failure(
            "safe primary",
            kind=ProviderFailureKind.TRANSPORT,
            retryable=True,
            status_code=504,
        ),
        _failure(
            "safe secondary",
            kind=ProviderFailureKind.RATE_LIMIT,
            retryable=True,
            status_code=429,
            upstream_status_code=429,
        ),
        _failure(
            "safe tertiary",
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            retryable=True,
            status_code=502,
            upstream_status_code=200,
            cause=ValueError("raw-secret-body"),
        ),
    ]
    providers = [
        _AttemptProvider(name, order, error=error)
        for name, error in zip(("primary", "secondary", "tertiary"), failures, strict=True)
    ]

    with pytest.raises(ProviderError) as exc_info:
        await _execute(mode, providers)

    assert exc_info.value is not failures[-1]
    assert exc_info.value.__cause__ is failures[-1]
    assert exc_info.value.kind is failures[-1].kind
    assert exc_info.value.status_code == failures[-1].status_code
    assert order == ["primary", "secondary", "tertiary"]
    attempts = exc_info.value.attempts
    assert isinstance(attempts, tuple)
    assert [record.provider for record in attempts] == order
    assert [record.model for record in attempts] == [
        ModelRef("primary", "primary-model"),
        ModelRef("secondary", "secondary-model"),
        ModelRef("tertiary", "tertiary-model"),
    ]
    assert [record.operation for record in attempts] == [
        Operation.CHAT if mode == "nonstream" else Operation.CHAT_STREAM
    ] * 3
    assert [record.downstream_status_code for record in attempts] == [504, 429, 502]
    assert [record.upstream_status_code for record in attempts] == [None, 429, 200]
    assert [record.failure_kind for record in attempts] == [
        ProviderFailureKind.TRANSPORT,
        ProviderFailureKind.RATE_LIMIT,
        ProviderFailureKind.UPSTREAM_PROTOCOL,
    ]
    assert [record.retryable for record in attempts] == [True, True, True]
    assert not hasattr(attempts[-1], "safe_message")
    assert not hasattr(attempts[-1], "cause")
    assert "raw-secret-body" not in repr(attempts)
    assert "safe tertiary" not in repr(attempts)
    with pytest.raises((AttributeError, FrozenInstanceError)):
        attempts[-1].retryable = False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_authentication_failure_is_fatal_for_both_paths(mode: Mode) -> None:
    order: list[str] = []
    auth_error = _failure(
        "authentication failed",
        kind=ProviderFailureKind.AUTHENTICATION,
        retryable=False,
        status_code=401,
        upstream_status_code=401,
    )
    primary = _AttemptProvider("primary", order, error=auth_error)
    secondary = _AttemptProvider("secondary", order)

    with pytest.raises(ProviderError) as exc_info:
        await _execute(mode, [primary, secondary])

    assert exc_info.value is not auth_error
    assert exc_info.value.__cause__ is auth_error
    assert order == ["primary"]
    assert len(exc_info.value.attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_retryable_upstream_protocol_failure_advances(mode: Mode) -> None:
    order: list[str] = []
    malformed = _failure(
        "malformed upstream response",
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        retryable=True,
        upstream_status_code=200,
    )
    primary = _AttemptProvider("primary", order, error=malformed)
    secondary = _AttemptProvider("secondary", order, result="recovered")

    result, provider_name = await _execute(mode, [primary, secondary])

    assert provider_name == "secondary"
    assert result == ("recovered" if mode == "nonstream" else ["recovered"])
    assert order == ["primary", "secondary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_fallback_false_attempts_only_primary(mode: Mode) -> None:
    order: list[str] = []
    primary_error = _failure(
        "primary unavailable",
        kind=ProviderFailureKind.TRANSPORT,
        retryable=True,
    )
    primary = _AttemptProvider("primary", order, error=primary_error)
    secondary = _AttemptProvider("secondary", order)

    with pytest.raises(ProviderError) as exc_info:
        await _execute(mode, [primary, secondary], fallback=False)

    assert exc_info.value is not primary_error
    assert exc_info.value.__cause__ is primary_error
    assert order == ["primary"]
    assert len(exc_info.value.attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_retryable_ensure_token_failure_uses_same_policy(mode: Mode) -> None:
    order: list[str] = []
    token_error = _failure(
        "token refresh transport failed",
        kind=ProviderFailureKind.TRANSPORT,
        retryable=True,
    )
    primary = _AttemptProvider("primary", order, ensure_error=token_error)
    secondary = _AttemptProvider("secondary", order, result="secondary")

    result, provider_name = await _execute(mode, [primary, secondary])

    assert provider_name == "secondary"
    assert result == ("secondary" if mode == "nonstream" else ["secondary"])
    assert primary.call_count == 0
    assert order == ["primary", "secondary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_retryable_legacy_unknown_failure_remains_compatible(mode: Mode) -> None:
    order: list[str] = []
    legacy = ProviderError("legacy provider failed", status_code=503, retryable=True)
    primary = _AttemptProvider("primary", order, error=legacy)
    secondary = _AttemptProvider("secondary", order, result="secondary")

    result, provider_name = await _execute(mode, [primary, secondary])

    assert provider_name == "secondary"
    assert result == ("secondary" if mode == "nonstream" else ["secondary"])
    assert order == ["primary", "secondary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_execution_never_exceeds_candidates_snapshotted_in_plan(mode: Mode) -> None:
    order: list[str] = []
    primary_error = _failure(
        "primary",
        kind=ProviderFailureKind.TRANSPORT,
        retryable=True,
    )
    secondary_error = _failure(
        "secondary",
        kind=ProviderFailureKind.TRANSPORT,
        retryable=True,
    )
    primary = _AttemptProvider("primary", order, error=primary_error)
    secondary = _AttemptProvider("secondary", order, error=secondary_error)
    unplanned = _AttemptProvider("unplanned", order, result="must not run")

    with pytest.raises(ProviderError) as exc_info:
        await _execute(mode, [primary, secondary])

    assert exc_info.value is not secondary_error
    assert exc_info.value.__cause__ is secondary_error
    assert order == ["primary", "secondary"]
    assert unplanned.ensure_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_auto_unknown_typed_unsupported_advances_as_compatibility_fallback(
    mode: Mode,
) -> None:
    order: list[str] = []
    unsupported = _failure(
        "operation unsupported",
        kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
        retryable=False,
        status_code=501,
    )
    primary = _AttemptProvider("primary", order, error=unsupported)
    secondary = _AttemptProvider("secondary", order, result="compatible")

    result, provider_name = await _execute(
        mode,
        [primary, secondary],
        supports=[CapabilitySupport.UNKNOWN, CapabilitySupport.SUPPORTED],
        explicit=False,
    )

    assert provider_name == "secondary"
    assert result == ("compatible" if mode == "nonstream" else ["compatible"])
    assert order == ["primary", "secondary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
@pytest.mark.parametrize(
    ("primary_support", "explicit"),
    [
        (CapabilitySupport.SUPPORTED, False),
        (CapabilitySupport.UNKNOWN, True),
    ],
)
async def test_runtime_unsupported_never_substitutes_supported_or_explicit_candidate(
    mode: Mode,
    primary_support: CapabilitySupport,
    explicit: bool,
) -> None:
    order: list[str] = []
    unsupported = _failure(
        "operation unsupported",
        kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
        retryable=True,
        status_code=501,
    )
    primary = _AttemptProvider("primary", order, error=unsupported)
    secondary = _AttemptProvider("secondary", order, result="must not run")

    with pytest.raises(ProviderError) as exc_info:
        await _execute(
            mode,
            [primary, secondary],
            supports=[primary_support, CapabilitySupport.SUPPORTED],
            explicit=explicit,
        )

    assert exc_info.value is not unsupported
    assert exc_info.value.__cause__ is unsupported
    assert order == ["primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_explicit_primary_retryable_failure_consumes_full_deduplicated_plan(
    mode: Mode,
) -> None:
    order: list[str] = []
    primary = _AttemptProvider(
        "explicit",
        order,
        error=_failure(
            "explicit transport failed",
            kind=ProviderFailureKind.TRANSPORT,
            retryable=True,
        ),
    )
    first_fallback = _AttemptProvider(
        "priority-one",
        order,
        error=_failure(
            "first fallback failed",
            kind=ProviderFailureKind.RATE_LIMIT,
            retryable=True,
            status_code=429,
        ),
    )
    second_fallback = _AttemptProvider("priority-two", order, result="success")

    result, provider_name = await _execute(
        mode,
        [primary, first_fallback, second_fallback],
        explicit=True,
    )

    assert provider_name == "priority-two"
    assert result == ("success" if mode == "nonstream" else ["success"])
    assert order == ["explicit", "priority-one", "priority-two"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_execution_uses_candidate_objects_without_reloading_mutated_router_state(
    mode: Mode,
) -> None:
    order: list[str] = []
    primary = _AttemptProvider(
        "primary",
        order,
        error=_failure(
            "primary failed",
            kind=ProviderFailureKind.TRANSPORT,
            retryable=True,
        ),
    )
    secondary = _AttemptProvider("secondary", order, result="snapshot success")
    router = Router.__new__(Router)
    operation = Operation.CHAT if mode == "nonstream" else Operation.CHAT_STREAM
    plan = _plan(operation, [primary, secondary])
    router.providers = {}

    def fail_if_reloaded() -> None:
        raise AssertionError("execution reloaded priorities")

    router._get_priorities_config = fail_if_reloaded  # type: ignore[method-assign]
    if mode == "nonstream":
        result, provider_name = await router._execute_plan_nonstream(
            plan,
            object(),
            True,
            lambda request, _model: request,
            lambda provider, request: provider.complete(request),
        )
        assert result == "snapshot success"
    else:
        stream, provider_name = await router._execute_plan_stream(
            plan,
            object(),
            True,
            lambda request, _model: request,
            lambda provider, request: provider.open_stream(request),
            "test",
        )
        assert [chunk.content async for chunk in stream] == ["snapshot success"]

    assert provider_name == "secondary"
    assert order == ["primary", "secondary"]


@pytest.mark.asyncio
async def test_stream_precommit_failure_closes_once_before_fallback() -> None:
    order: list[str] = []
    primary = _AttemptProvider(
        "primary",
        order,
        error=_failure(
            "precommit failed",
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            retryable=True,
        ),
    )
    secondary = _AttemptProvider("secondary", order, result="secondary")

    result, provider_name = await _execute("stream", [primary, secondary])

    assert result == ["secondary"]
    assert provider_name == "secondary"
    assert primary.stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_postcommit_failure_propagates_without_switching_and_closes_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    order: list[str] = []
    postcommit_error = _failure(
        "postcommit-private-message",
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        retryable=True,
        cause=ValueError("postcommit-private-cause"),
    )
    primary = _AttemptProvider(
        "primary",
        order,
        error=postcommit_error,
        stream_contents=["committed"],
    )
    secondary = _AttemptProvider("secondary", order, result="must not run")
    router = Router.__new__(Router)
    plan = _plan(Operation.CHAT_STREAM, [primary, secondary])

    with caplog.at_level(logging.INFO, logger="router_maestro.routing"):
        stream, provider_name = await router._execute_plan_stream(
            plan,
            object(),
            True,
            lambda request, _model: request,
            lambda provider, request: provider.open_stream(request),
            "test",
        )
        contents: list[str] = []
        with pytest.raises(ProviderError) as exc_info:
            async for chunk in stream:
                contents.append(chunk.content)

    assert provider_name == "primary"
    assert contents == ["committed"]
    assert exc_info.value is postcommit_error
    assert order == ["primary"]
    assert secondary.ensure_calls == 0
    assert primary.stream.close_calls == 1
    assert "postcommit-private-message" not in caplog.text
    assert "postcommit-private-cause" not in caplog.text


def test_attempt_ledger_snapshot_is_ordered_and_read_only() -> None:
    attempts_module = importlib.import_module("router_maestro.routing.attempts")
    attempt_ledger_type = attempts_module.AttemptLedger
    attempt_record_type = attempts_module.AttemptRecord
    record = attempt_record_type(
        provider="provider",
        model=ModelRef("provider", "model"),
        operation=Operation.CHAT,
        downstream_status_code=503,
        upstream_status_code=None,
        failure_kind=ProviderFailureKind.UNKNOWN,
        retryable=True,
    )
    ledger = attempt_ledger_type()

    ledger.record(record)
    snapshot = ledger.snapshot()

    assert snapshot == (record,)
    assert ledger.records == snapshot
    with pytest.raises((AttributeError, FrozenInstanceError)):
        snapshot[0].retryable = False


@pytest.mark.asyncio
async def test_concurrent_exhaustions_do_not_share_attempt_ledgers() -> None:
    shared_error = _failure(
        "shared provider failure",
        kind=ProviderFailureKind.TRANSPORT,
        retryable=True,
    )

    async def exhaust(prefix: str):
        order: list[str] = []
        first = _AttemptProvider(f"{prefix}-first", order, error=shared_error)
        last = _AttemptProvider(f"{prefix}-last", order, error=shared_error)
        with pytest.raises(ProviderError) as exc_info:
            await _execute("nonstream", [first, last])
        return exc_info.value, exc_info.value.attempts

    (first_error, first_attempts), (second_error, second_attempts) = await asyncio.gather(
        exhaust("request-a"),
        exhaust("request-b"),
    )

    assert first_error is not shared_error
    assert second_error is not shared_error
    assert first_error is not second_error
    assert [record.provider for record in first_attempts] == [
        "request-a-first",
        "request-a-last",
    ]
    assert [record.provider for record in second_attempts] == [
        "request-b-first",
        "request-b-last",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonstream", "stream"])
async def test_attempt_logs_are_structured_and_do_not_include_error_or_cause_text(
    caplog: pytest.LogCaptureFixture,
    mode: Mode,
) -> None:
    order: list[str] = []
    body_marker = "private-httpx-response-body"
    request = httpx.Request("POST", "https://upstream.example/v1/messages")
    response = httpx.Response(502, text=body_marker, request=request)
    http_error = httpx.HTTPStatusError(
        "private-httpx-status-message",
        request=request,
        response=response,
    )
    primary = _AttemptProvider(
        "primary",
        order,
        error=_failure(
            "safe-but-request-specific-message",
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            retryable=True,
            upstream_status_code=200,
            cause=http_error,
        ),
    )
    secondary = _AttemptProvider("secondary", order, result="success")

    with caplog.at_level(logging.INFO, logger="router_maestro.routing"):
        await _execute(mode, [primary, secondary])

    messages = [record.getMessage() for record in caplog.records]
    failed = next(message for message in messages if "route_attempt_failed" in message)
    assert "attempt=1" in failed
    assert "provider=primary" in failed
    assert "model=primary/primary-model" in failed
    assert "operation=chat" in failed
    assert "kind=upstream_protocol" in failed
    assert "retryable=true" in failed
    assert "decision=fallback" in failed
    assert all("safe-but-request-specific-message" not in message for message in messages)
    assert all("private-httpx-status-message" not in message for message in messages)
    assert all(body_marker not in message for message in messages)
    assert body_marker not in caplog.text
    assert all(body_marker not in repr(record.__dict__) for record in caplog.records)
    terminal_event = "route_attempt_succeeded" if mode == "nonstream" else "route_attempt_selected"
    assert any(terminal_event in message for message in messages)
    if mode == "stream":
        assert not any("route_attempt_succeeded" in message for message in messages)
