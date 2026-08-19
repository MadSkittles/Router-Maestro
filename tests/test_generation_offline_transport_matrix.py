"""Offline contract matrix for every ingress and implemented upstream transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any

import pytest
from _pytest.mark.structures import ParameterSet

from router_maestro.config import FallbackConfig, FallbackStrategy, PrioritiesConfig
from router_maestro.protocols import ConversionMode, WireProtocol
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
)
from router_maestro.providers.bindings import (
    EndpointBinding,
    PreparedAttempt,
    ProviderDialect,
)
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.runtime.reasoning_capsule import ReasoningCapsuleCodec
from router_maestro.server.generation_pipeline import build_generation_pipeline
from router_maestro.utils.cache import TTLCache

_PROVIDER = "matrix-provider"
_UPSTREAM_MODEL = "matrix-model"
_PUBLIC_MODEL = f"{_PROVIDER}/{_UPSTREAM_MODEL}"
_TEXT = "hello from the offline matrix"
_CREATED = 1_723_456_789


def _operations(protocol: WireProtocol) -> frozenset[Operation]:
    return {
        WireProtocol.ANTHROPIC_MESSAGES: frozenset({Operation.NATIVE_ANTHROPIC}),
        WireProtocol.OPENAI_CHAT: frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        WireProtocol.OPENAI_RESPONSES: frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM}),
    }[protocol]


def _endpoint(protocol: WireProtocol) -> str:
    return {
        WireProtocol.ANTHROPIC_MESSAGES: "https://offline.invalid/v1/messages",
        WireProtocol.OPENAI_CHAT: "https://offline.invalid/v1/chat/completions",
        WireProtocol.OPENAI_RESPONSES: "https://offline.invalid/v1/responses",
    }[protocol]


@dataclass(frozen=True, slots=True)
class _IngressCase:
    protocol: WireProtocol
    label: str

    def payload(self, *, stream: bool) -> dict[str, Any]:
        if self.protocol is WireProtocol.ANTHROPIC_MESSAGES:
            return {
                "model": _PUBLIC_MODEL,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": _TEXT}],
                "stream": stream,
            }
        if self.protocol is WireProtocol.OPENAI_CHAT:
            return {
                "model": _PUBLIC_MODEL,
                "messages": [{"role": "user", "content": _TEXT}],
                "max_tokens": 64,
                "stream": stream,
            }
        if self.protocol is WireProtocol.OPENAI_RESPONSES:
            return {
                "model": _PUBLIC_MODEL,
                "input": _TEXT,
                "max_output_tokens": 64,
                "stream": stream,
            }
        return {
            "contents": [{"role": "user", "parts": [{"text": _TEXT}]}],
            "generationConfig": {"maxOutputTokens": 64},
        }


_INGRESS_CASES = (
    _IngressCase(WireProtocol.ANTHROPIC_MESSAGES, "anthropic"),
    _IngressCase(WireProtocol.OPENAI_CHAT, "chat"),
    _IngressCase(WireProtocol.OPENAI_RESPONSES, "responses"),
    _IngressCase(WireProtocol.GEMINI, "gemini"),
)

_UPSTREAMS = (
    (WireProtocol.ANTHROPIC_MESSAGES, "messages"),
    (WireProtocol.OPENAI_CHAT, "chat"),
    (WireProtocol.OPENAI_RESPONSES, "responses"),
)


def _matrix_param(
    ingress: _IngressCase,
    target: WireProtocol,
    target_label: str,
    stream: bool,
) -> ParameterSet:
    return pytest.param(
        ingress,
        target,
        stream,
        id=f"{ingress.label}-to-{target_label}-{'stream' if stream else 'response'}",
    )


_MATRIX = tuple(
    _matrix_param(ingress, target, target_label, stream)
    for ingress, (target, target_label), stream in product(
        _INGRESS_CASES,
        _UPSTREAMS,
        (False, True),
    )
)


class _RecordingDialect(ProviderDialect):
    def __init__(self, protocol: WireProtocol) -> None:
        self.protocol = protocol
        self.prepare_count = 0

    @property
    def id(self) -> str:
        return "matrix-dialect"

    async def prepare_attempt(
        self,
        *,
        binding_id: str,
        protocol: WireProtocol,
        model: ModelRef,
        payload: Mapping[str, Any],
        stream: bool,
        request_context,
    ) -> PreparedAttempt:
        del request_context
        assert protocol is self.protocol
        self.prepare_count += 1
        outbound = deepcopy(dict(payload))
        # Provider dialects own public-to-upstream model rewriting on identity paths.
        outbound["model"] = model.upstream_id
        return PreparedAttempt(
            binding_id=binding_id,
            protocol=protocol,
            model=model,
            url=_endpoint(protocol),
            payload=outbound,
            headers={"x-matrix-binding": binding_id},
            stream=stream,
        )


class _RecordingExecutor:
    def __init__(self, protocol: WireProtocol) -> None:
        self.protocol = protocol
        self.attempts: list[PreparedAttempt] = []

    async def execute(self, attempt: PreparedAttempt) -> Mapping[str, Any]:
        self.attempts.append(attempt)
        return _response(self.protocol)

    def execute_stream(self, attempt: PreparedAttempt) -> AsyncIterator[Mapping[str, Any]]:
        self.attempts.append(attempt)

        async def frames() -> AsyncIterator[Mapping[str, Any]]:
            for frame in _stream_frames(self.protocol):
                yield frame

        return frames()


class _MatrixProvider(BaseProvider):
    name = _PROVIDER

    def __init__(self, target: WireProtocol) -> None:
        self.target = target
        self.dialect = _RecordingDialect(target)
        self.executor = _RecordingExecutor(target)
        self.binding = EndpointBinding(
            id=f"matrix-{target.value}",
            protocol=target,
            capabilities=ProviderCapabilities(operations=_operations(target)),
            dialect=self.dialect,
            executor=self.executor,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.binding.capabilities

    def bindings(self) -> tuple[EndpointBinding, ...]:
        return (self.binding,)

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=_UPSTREAM_MODEL, name=_UPSTREAM_MODEL, provider=self.name)]

    def is_authenticated(self) -> bool:
        return True

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError(f"legacy chat execution was selected for {request.model}")

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        raise AssertionError(f"legacy chat stream was selected for {request.model}")
        if False:  # pragma: no cover - keeps this an async generator
            yield ChatStreamChunk(content="")


def _router(provider: _MatrixProvider) -> Router:
    router = Router.__new__(Router)
    router.providers = {provider.name: provider}
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._model_aliases = None
    router._managed_generation = True
    router._priorities_cache.set(
        PrioritiesConfig(
            priorities=[_PUBLIC_MODEL],
            fallback=FallbackConfig(strategy=FallbackStrategy.NONE, maxRetries=0),
        )
    )
    router._providers_ttl.set(True)
    return router


def _response(protocol: WireProtocol) -> dict[str, Any]:
    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        return {
            "id": "msg_matrix",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": _TEXT}],
            "model": _UPSTREAM_MODEL,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 5},
        }
    if protocol is WireProtocol.OPENAI_CHAT:
        return {
            "id": "chatcmpl_matrix",
            "object": "chat.completion",
            "created": _CREATED,
            "model": _UPSTREAM_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": _TEXT},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        }
    return {
        "id": "resp_matrix",
        "object": "response",
        "created_at": _CREATED,
        "model": _UPSTREAM_MODEL,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_matrix",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _TEXT, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
        "error": None,
        "incomplete_details": None,
    }


def _stream_frames(protocol: WireProtocol) -> tuple[dict[str, Any], ...]:
    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        return (
            {
                "type": "message_start",
                "message": {
                    "id": "msg_matrix",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": _UPSTREAM_MODEL,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 4, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": _TEXT},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        )
    if protocol is WireProtocol.OPENAI_CHAT:
        return (
            {
                "id": "chatcmpl_matrix",
                "object": "chat.completion.chunk",
                "created": _CREATED,
                "model": _UPSTREAM_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": _TEXT},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9,
                },
            },
        )
    base = {
        "id": "resp_matrix",
        "object": "response",
        "created_at": _CREATED,
        "model": _UPSTREAM_MODEL,
        "status": "in_progress",
        "output": [],
        "usage": None,
        "error": None,
        "incomplete_details": None,
    }
    return (
        {"type": "response.created", "response": base},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_matrix",
            "output_index": 0,
            "content_index": 0,
            "delta": _TEXT,
        },
        {
            "type": "response.completed",
            "response": {
                **base,
                "status": "completed",
                "usage": {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
            },
        },
    )


def _assert_outbound_payload(
    payload: Mapping[str, Any],
    *,
    target: WireProtocol,
    stream: bool,
) -> None:
    assert payload["model"] == _UPSTREAM_MODEL
    assert payload.get("stream", False) is stream
    if target is WireProtocol.ANTHROPIC_MESSAGES:
        assert payload["max_tokens"] == 64
        assert payload["messages"][0]["role"] == "user"
    elif target is WireProtocol.OPENAI_CHAT:
        assert payload["max_tokens"] == 64
        assert payload["messages"][0]["role"] == "user"
    else:
        assert payload["max_output_tokens"] == 64
        assert "input" in payload
    assert _TEXT in json.dumps(dict(payload), ensure_ascii=False)


def _assert_nonstream_response(protocol: WireProtocol, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    assert _TEXT in serialized
    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        assert payload["model"] == _PUBLIC_MODEL
        assert payload["stop_reason"] == "end_turn"
    elif protocol is WireProtocol.OPENAI_CHAT:
        assert payload["model"] == _PUBLIC_MODEL
        assert payload["choices"][0]["finish_reason"] == "stop"
    elif protocol is WireProtocol.OPENAI_RESPONSES:
        assert payload["model"] == _PUBLIC_MODEL
        assert payload["status"] == "completed"
    else:
        assert payload["modelVersion"] == _PUBLIC_MODEL
        assert payload["candidates"][0]["finishReason"] == "STOP"


def _is_terminal(protocol: WireProtocol, frame: Mapping[str, Any]) -> bool:
    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        return frame.get("type") == "message_stop"
    if protocol is WireProtocol.OPENAI_RESPONSES:
        return frame.get("type") in {
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
        }
    candidates = frame.get("candidates")
    if protocol is WireProtocol.GEMINI:
        return isinstance(candidates, list) and any(
            isinstance(candidate, Mapping) and candidate.get("finishReason") is not None
            for candidate in candidates
        )
    choices = frame.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, Mapping) and choice.get("finish_reason") is not None
        for choice in choices
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("ingress", "target", "stream"), _MATRIX)
async def test_offline_generation_transport_matrix(
    ingress: _IngressCase,
    target: WireProtocol,
    stream: bool,
) -> None:
    provider = _MatrixProvider(target)
    router = _router(provider)
    source_payload = ingress.payload(stream=stream)
    if ingress.protocol is target:
        source_payload["future_wire_field"] = {"preserved": [1, 2, 3]}
    original_payload = deepcopy(source_payload)
    pipeline = build_generation_pipeline(
        router,
        ReasoningCapsuleCodec(bytes([73]) * 32),
        ingress.protocol,
        source_payload,
        path=f"/matrix/{ingress.label}",
        model=_PUBLIC_MODEL if ingress.protocol is WireProtocol.GEMINI else None,
        stream=stream,
    )

    if stream:
        opened = await pipeline.dispatcher.dispatch_stream(router, pipeline.envelope)
        downstream = [
            frame
            async for frame in pipeline.responses.encode_stream(
                opened,
                pipeline.envelope.runtime,
            )
        ]
        selection = opened.selection.plan
        assert sum(_is_terminal(ingress.protocol, frame) for frame in downstream) == 1
        serialized = json.dumps(downstream, ensure_ascii=False)
        assert _TEXT in serialized
        assert _PUBLIC_MODEL in serialized
    else:
        result = await pipeline.dispatcher.dispatch(router, pipeline.envelope)
        downstream = await pipeline.responses.encode_result(
            result,
            pipeline.envelope.runtime,
        )
        selection = result.selection.plan
        _assert_nonstream_response(ingress.protocol, downstream)

    expected_mode = (
        ConversionMode.IDENTITY if ingress.protocol is target else ConversionMode.SEMANTIC_IR
    )
    assert selection.source_protocol is ingress.protocol
    assert selection.target_protocol is target
    assert selection.binding is provider.binding
    assert selection.binding.id == f"matrix-{target.value}"
    assert selection.conversion_mode is expected_mode
    assert pipeline.envelope.materialization_count == (
        0 if expected_mode is ConversionMode.IDENTITY else 1
    )

    assert provider.dialect.prepare_count == 1
    assert len(provider.executor.attempts) == 1
    attempt = provider.executor.attempts[0]
    assert attempt.binding_id == provider.binding.id
    assert attempt.protocol is target
    assert attempt.model == ModelRef(_PROVIDER, _UPSTREAM_MODEL)
    assert attempt.url == _endpoint(target)
    assert attempt.headers == {"x-matrix-binding": provider.binding.id}
    assert attempt.stream is stream
    _assert_outbound_payload(attempt.payload, target=target, stream=stream)
    if expected_mode is ConversionMode.IDENTITY:
        assert attempt.payload["future_wire_field"] == {"preserved": [1, 2, 3]}

    assert source_payload == original_payload
    assert pipeline.envelope.raw_payload == original_payload


@pytest.mark.asyncio
async def test_anthropic_context_management_noop_reaches_responses_transport() -> None:
    provider = _MatrixProvider(WireProtocol.OPENAI_RESPONSES)
    router = _router(provider)
    source_payload = _IngressCase(WireProtocol.ANTHROPIC_MESSAGES, "anthropic").payload(
        stream=False
    )
    source_payload["context_management"] = {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }
    pipeline = build_generation_pipeline(
        router,
        ReasoningCapsuleCodec(bytes([73]) * 32),
        WireProtocol.ANTHROPIC_MESSAGES,
        source_payload,
        path="/api/anthropic/v1/messages",
    )

    result = await pipeline.dispatcher.dispatch(router, pipeline.envelope)
    downstream = await pipeline.responses.encode_result(result, pipeline.envelope.runtime)

    assert downstream["type"] == "message"
    assert pipeline.envelope.materialization_count == 1
    assert len(provider.executor.attempts) == 1
    attempt = provider.executor.attempts[0]
    assert attempt.protocol is WireProtocol.OPENAI_RESPONSES
    assert "context_management" not in attempt.payload
