"""Contracts for the redesigned virtual Router-Maestro Auto model."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from router_maestro.config import AutoConfig, AutoMode, AutoTaskType, PrioritiesConfig
from router_maestro.protocols import OpenAIChatRuntime, RequestEnvelope, WireProtocol
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ContextWindowOption,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    ProviderFailureSignal,
)
from router_maestro.providers.bindings import EndpointBinding, legacy_endpoint_binding
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.generation_plan import auto_model_info, plan_generation_route
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.server.dispatcher import GenerationDispatcher
from router_maestro.utils.cache import TTLCache


def _chat_binding() -> EndpointBinding:
    return legacy_endpoint_binding(
        binding_id="chat",
        protocol=WireProtocol.OPENAI_CHAT,
        operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    )


class _Provider(BaseProvider):
    def __init__(self, models: tuple[ModelInfo, ...]) -> None:
        self.name = "test"
        self.models = models

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(frozenset({Operation.CHAT, Operation.CHAT_STREAM}))

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="unused", model=request.model)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(content="unused")

    async def list_models(self) -> list[ModelInfo]:
        return list(self.models)

    def is_authenticated(self) -> bool:
        return True

    def bindings(self) -> tuple[EndpointBinding, ...]:
        return (_chat_binding(),)


def _router(config: PrioritiesConfig, models: tuple[ModelInfo, ...]) -> Router:
    router = Router.__new__(Router)
    provider = _Provider(models)
    router.providers = {provider.name: provider}
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache.set(config)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._providers_ttl.set(True)
    router._model_aliases = None
    router._managed_generation = True
    return router


class _Execution:
    def __init__(
        self,
        classifier_task: str,
        *,
        model_actions: Mapping[str, Any] | None = None,
        model_streams: Mapping[str, AsyncIterator[Any] | BaseException] | None = None,
    ) -> None:
        self.classifier_task = classifier_task
        self.model_actions = dict(model_actions or {})
        self.model_streams = dict(model_streams or {})
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.stream_calls: list[tuple[str, Mapping[str, Any]]] = []

    async def execute(self, plan, payload: Mapping[str, Any], *, request_context=None) -> Any:
        del request_context
        self.calls.append((plan.model.qualified_id, payload))
        if plan.model.upstream_id == "router":
            return ChatResponse(
                content=f'{{"task_type":"{self.classifier_task}"}}',
                model=plan.model.upstream_id,
            )
        action = self.model_actions.get(plan.model.upstream_id)
        if isinstance(action, BaseException):
            raise action
        if action is not None:
            return action
        return {"id": "ok", "model": plan.model.upstream_id, "choices": []}

    async def open_stream(self, plan, payload, *, request_context=None):
        del request_context
        self.stream_calls.append((plan.model.qualified_id, payload))
        action = self.model_streams[plan.model.upstream_id]
        if isinstance(action, BaseException):
            raise action
        return action


class _Stream:
    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.closed = False

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> Any:
        if not self.items:
            raise StopAsyncIteration
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


def _context_overflow(model: str) -> ProviderError:
    return ProviderError(
        "Request exceeds the selected model's context window",
        status_code=400,
        retryable=False,
        kind=ProviderFailureKind.CLIENT_REQUEST,
        provider="test",
        model=model,
        signal=ProviderFailureSignal.CONTEXT_WINDOW_EXCEEDED,
    )


def _model(model_id: str, **kwargs: Any) -> ModelInfo:
    return ModelInfo(id=model_id, name=model_id, provider="test", **kwargs)


def test_legacy_non_empty_priorities_migrate_to_strict_chain() -> None:
    config = PrioritiesConfig.model_validate({"priorities": ["test/one", "test/two"]})

    assert config.auto.mode is AutoMode.PRIORITY_CHAIN
    assert config.auto.priority_chain == ["test/one", "test/two"]


def test_priority_chain_profile_rejects_empty_chain_when_configured() -> None:
    with pytest.raises(ValueError, match="priority_chain must contain at least one model"):
        AutoConfig(mode=AutoMode.PRIORITY_CHAIN, priority_chain=[])


@pytest.mark.asyncio
async def test_empty_priority_chain_fails_without_catalog_fallback() -> None:
    # A hand-built compatibility object represents a corrupt legacy/on-disk
    # state; normal config validation prevents saving this shape.
    invalid_auto = AutoConfig.model_construct(mode=AutoMode.PRIORITY_CHAIN, priority_chain=[])
    config = PrioritiesConfig.model_construct(auto=invalid_auto, priorities=[])
    router = _router(config, (_model("one"),))

    with pytest.raises(ProviderError, match="priority chain is empty") as raised:
        await plan_generation_route(router, "router-maestro")

    assert raised.value.status_code == 503
    assert raised.value.parameter == "auto.priority_chain"


@pytest.mark.asyncio
async def test_priority_chain_is_complete_and_ignores_catalog_order() -> None:
    config = PrioritiesConfig(
        auto=AutoConfig(mode=AutoMode.PRIORITY_CHAIN, priority_chain=["test/two", "test/one"])
    )
    router = _router(config, (_model("one"), _model("two"), _model("unconfigured")))

    plan = await plan_generation_route(router, "router-maestro")

    assert [candidate.model for candidate in plan.candidates] == [
        ModelRef("test", "two"),
        ModelRef("test", "one"),
    ]
    assert plan.max_model_switches == 1


@pytest.mark.asyncio
async def test_task_router_classifies_then_executes_configured_task_model() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "task-router",
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {
                        "fast": "test/fast",
                        "general": "test/general",
                        "coding": "test/code",
                        "deep_reasoning": "test/deep",
                    },
                },
            }
        }
    )
    router = _router(
        config,
        tuple(_model(model) for model in ("router", "fast", "general", "code", "deep")),
    )
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "Fix this failing Python test"}],
        },
    )
    execution = _Execution(AutoTaskType.CODING.value)

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.selection.plan.model == ModelRef("test", "code")
    assert [model for model, _payload in execution.calls] == ["test/router", "test/code"]
    classifier = execution.calls[0][1]
    assert classifier["stream"] is False
    assert "tools" not in classifier
    assert "Fix this failing Python test" in classifier["messages"][1]["content"]
    assert classifier["max_tokens"] == 32
    assert classifier["response_format"]["json_schema"]["schema"]["properties"] == {
        "task_type": {
            "type": "string",
            "enum": ["fast", "general", "coding", "deep_reasoning"],
        }
    }


@pytest.mark.asyncio
async def test_task_router_disables_classifier_reasoning_only_for_catalog_none() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {task.value: f"test/{task.value}" for task in AutoTaskType},
                }
            }
        }
    )
    models = (
        _model("router", reasoning_effort_values=["none", "low"]),
        *(_model(task.value) for task in AutoTaskType),
    )
    router = _router(config, models)
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {"model": "router-maestro", "messages": [{"role": "user", "content": "hello"}]},
    )
    execution = _Execution(AutoTaskType.FAST.value)

    await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert execution.calls[0][1]["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_task_router_strictly_filters_unknown_required_capabilities() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "task-router",
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {
                        "fast": "test/plain",
                        "general": "test/tools",
                        "coding": "test/tools",
                        "deep_reasoning": "test/plain",
                    },
                },
            }
        }
    )
    router = _router(
        config,
        (
            _model("router"),
            _model("plain"),
            _model("tools", feature_capabilities={"tools": True}),
        ),
    )
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "Use the tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        },
    )
    execution = _Execution(AutoTaskType.CODING.value)

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.selection.plan.model == ModelRef("test", "tools")
    assert execution.calls == [
        (
            "test/tools",
            envelope.native_payload(),
        )
    ]
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_task_router_excludes_capability_ineligible_tasks_from_classifier() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "task-router",
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {
                        "fast": "test/plain",
                        "general": "test/general-tools",
                        "coding": "test/coding-tools",
                        "deep_reasoning": "test/plain",
                    },
                },
            }
        }
    )
    router = _router(
        config,
        (
            _model("router"),
            _model("plain", feature_capabilities={"tools": False}),
            _model("general-tools", feature_capabilities={"tools": True}),
            _model("coding-tools", feature_capabilities={"tools": True}),
        ),
    )
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "Use the tool to fix this test"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        },
    )
    execution = _Execution(AutoTaskType.CODING.value)

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.selection.plan.model == ModelRef("test", "coding-tools")
    assert [model for model, _payload in execution.calls] == [
        "test/router",
        "test/coding-tools",
    ]
    classifier_schema = execution.calls[0][1]["response_format"]["json_schema"]["schema"]
    assert classifier_schema["properties"]["task_type"]["enum"] == ["general", "coding"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("estimated_tokens", "expected_tasks"),
    [
        (237_999, ["fast", "general", "coding", "deep_reasoning"]),
        (238_000, ["general", "coding", "deep_reasoning"]),
    ],
)
async def test_task_router_applies_context_safety_threshold_at_seventy_percent(
    monkeypatch: pytest.MonkeyPatch,
    estimated_tokens: int,
    expected_tasks: list[str],
) -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "task-router",
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {
                        "fast": "test/gpt-5.4-mini",
                        "general": "test/gpt-5.6-terra",
                        "coding": "test/gpt-5.6-terra",
                        "deep_reasoning": "test/gpt-5.6-sol",
                    },
                },
            }
        }
    )
    router = _router(
        config,
        (
            _model("router"),
            _model(
                "gpt-5.4-mini",
                context_window_options=(
                    ContextWindowOption("default", 272_000, False),
                    ContextWindowOption("long_context", 340_000, True),
                ),
            ),
            _model(
                "gpt-5.6-terra",
                context_window_options=(
                    ContextWindowOption("default", 272_000, True),
                    ContextWindowOption("long_context", 922_000, False),
                ),
            ),
            _model(
                "gpt-5.6-sol",
                context_window_options=(
                    ContextWindowOption("default", 272_000, True),
                    ContextWindowOption("long_context", 922_000, False),
                ),
            ),
        ),
    )
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "Reply OK"}],
        },
    )
    monkeypatch.setattr(
        "router_maestro.utils.tokens.estimate_tokens", lambda _text: estimated_tokens
    )
    execution = _Execution(AutoTaskType.GENERAL.value)

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert envelope.manifest.estimated_input_tokens == estimated_tokens
    assert result.selection.plan.model == ModelRef("test", "gpt-5.6-terra")
    assert [model for model, _payload in execution.calls] == [
        "test/router",
        "test/gpt-5.6-terra",
    ]
    classifier_schema = execution.calls[0][1]["response_format"]["json_schema"]["schema"]
    assert classifier_schema["properties"]["task_type"]["enum"] == expected_tasks


@pytest.mark.asyncio
async def test_task_router_retains_every_largest_context_model_when_none_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "task-router",
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {
                        "fast": "test/fast-340k",
                        "general": "test/general-1m",
                        "coding": "test/coding-1m",
                        "deep_reasoning": "test/deep-1m",
                    },
                },
            }
        }
    )
    router = _router(
        config,
        (
            _model("router"),
            _model("fast-340k", max_prompt_tokens=340_000),
            _model("general-1m", max_prompt_tokens=1_000_000),
            _model("coding-1m", max_prompt_tokens=1_000_000),
            _model("deep-1m", max_prompt_tokens=1_000_000),
        ),
    )
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "Reply OK"}],
        },
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    execution = _Execution(AutoTaskType.CODING.value)

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.selection.plan.model == ModelRef("test", "coding-1m")
    assert [model for model, _payload in execution.calls] == ["test/router", "test/coding-1m"]
    classifier_schema = execution.calls[0][1]["response_format"]["json_schema"]["schema"]
    assert classifier_schema["properties"]["task_type"]["enum"] == [
        "general",
        "coding",
        "deep_reasoning",
    ]


@pytest.mark.asyncio
async def test_task_router_rejects_when_no_hard_compatible_model_remains() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "task-router",
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {task.value: f"test/{task.value}" for task in AutoTaskType},
                },
            }
        }
    )
    router = _router(
        config,
        (
            _model("router"),
            *(_model(task.value, feature_capabilities={"tools": False}) for task in AutoTaskType),
        ),
    )
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )
    execution = _Execution(AutoTaskType.FAST.value)

    with pytest.raises(ProviderError, match="No configured Auto task model supports") as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value.status_code == 400
    assert raised.value.parameter == "model"
    assert execution.calls == []


@pytest.mark.asyncio
async def test_task_router_rejects_invalid_classifier_output() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {task.value: f"test/{task.value}" for task in AutoTaskType},
                }
            }
        }
    )
    models = (_model("router"), *(_model(task.value) for task in AutoTaskType))
    router = _router(config, models)
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {"model": "router-maestro", "messages": [{"role": "user", "content": "hello"}]},
    )

    with pytest.raises(ProviderError, match="invalid task classification") as raised:
        await GenerationDispatcher({}, execution=_Execution("unconfigured")).dispatch(
            router, envelope
        )

    assert raised.value.status_code == 502


@pytest.mark.asyncio
async def test_auto_1m_suffix_is_only_normalized_and_does_not_filter_models() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {
                        "fast": "test/standard",
                        "general": "test/long",
                        "coding": "test/long",
                        "deep_reasoning": "test/standard",
                    },
                }
            }
        }
    )
    router = _router(
        config,
        (
            _model("router"),
            _model(
                "standard",
                context_window_options=(ContextWindowOption("default", 272_000, True),),
            ),
            _model(
                "long",
                context_window_options=(
                    ContextWindowOption("default", 272_000, True),
                    ContextWindowOption("long_context", 922_000, False),
                ),
            ),
        ),
    )
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro[1m]",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    execution = _Execution(AutoTaskType.FAST.value)

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.selection.plan.model == ModelRef("test", "standard")
    assert [model for model, _payload in execution.calls] == ["test/router", "test/standard"]


def _overflow_fallback_router() -> Router:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "task-router",
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {
                        "fast": "test/selected-1m",
                        "general": "test/peer-a-1m",
                        "coding": "test/peer-b-1m",
                        "deep_reasoning": "test/smaller",
                    },
                },
            }
        }
    )
    return _router(
        config,
        (
            _model("router"),
            _model("selected-1m", max_prompt_tokens=1_000_000),
            _model("peer-a-1m", max_prompt_tokens=1_000_000),
            _model("peer-b-1m", max_prompt_tokens=1_000_000),
            _model("smaller", max_prompt_tokens=340_000),
        ),
    )


@pytest.mark.asyncio
async def test_auto_context_overflow_tries_all_tied_largest_models_nonstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {"model": "router-maestro", "messages": [{"role": "user", "content": "hello"}]},
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_actions={
            "selected-1m": _context_overflow("selected-1m"),
            "peer-a-1m": _context_overflow("peer-a-1m"),
            "peer-b-1m": {"id": "ok", "model": "peer-b-1m", "choices": []},
        },
    )

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.selection.plan.model == ModelRef("test", "peer-b-1m")
    assert [model for model, _payload in execution.calls] == [
        "test/router",
        "test/selected-1m",
        "test/peer-a-1m",
        "test/peer-b-1m",
    ]


@pytest.mark.asyncio
async def test_auto_context_overflow_tries_tied_largest_before_first_stream_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    failed = _Stream([_context_overflow("selected-1m")])
    selected = _Stream(
        [
            {
                "id": "chat_1",
                "model": "peer-a-1m",
                "choices": [{"index": 0, "delta": {"content": "ok"}}],
            }
        ]
    )
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_streams={"selected-1m": failed, "peer-a-1m": selected},
    )

    opened = await GenerationDispatcher({}, execution=execution).dispatch_stream(router, envelope)

    assert failed.closed is True
    assert opened.selection.plan.model == ModelRef("test", "peer-a-1m")
    assert [model for model, _payload in execution.stream_calls] == [
        "test/selected-1m",
        "test/peer-a-1m",
    ]
    assert (await anext(opened.frames))["choices"][0]["delta"]["content"] == "ok"


@pytest.mark.asyncio
async def test_auto_context_overflow_tries_strictly_larger_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "task-router",
                "task_router": {
                    "router_model": "test/router",
                    "task_models": {
                        "fast": "test/340k",
                        "general": "test/1m-a",
                        "coding": "test/1m-a",
                        "deep_reasoning": "test/1m-b",
                    },
                },
            }
        }
    )
    router = _router(
        config,
        (
            _model("router"),
            _model("340k", max_prompt_tokens=340_000),
            _model("1m-a", max_prompt_tokens=1_000_000),
            _model("1m-b", max_prompt_tokens=1_000_000),
        ),
    )
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {"model": "router-maestro", "messages": [{"role": "user", "content": "hello"}]},
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 100_000)
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_actions={
            "340k": _context_overflow("340k"),
            "1m-a": {"id": "ok", "model": "1m-a", "choices": []},
        },
    )

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.selection.plan.model == ModelRef("test", "1m-a")
    assert [model for model, _payload in execution.calls] == [
        "test/router",
        "test/340k",
        "test/1m-a",
    ]


@pytest.mark.asyncio
async def test_auto_non_context_client_error_does_not_switch_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {"model": "router-maestro", "messages": [{"role": "user", "content": "hello"}]},
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    ordinary_error = ProviderError(
        "ordinary request error",
        status_code=400,
        retryable=False,
        kind=ProviderFailureKind.CLIENT_REQUEST,
    )
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_actions={"selected-1m": ordinary_error},
    )

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value is ordinary_error
    assert [model for model, _payload in execution.calls] == [
        "test/router",
        "test/selected-1m",
    ]


@pytest.mark.parametrize(
    ("status_code", "kind", "retryable"),
    [
        (429, ProviderFailureKind.RATE_LIMIT, True),
        (503, ProviderFailureKind.UPSTREAM_STATUS, True),
    ],
)
@pytest.mark.asyncio
async def test_auto_context_only_fallback_ignores_other_upstream_failures(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    kind: ProviderFailureKind,
    retryable: bool,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {"model": "router-maestro", "messages": [{"role": "user", "content": "hello"}]},
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    ordinary_error = ProviderError(
        "ordinary upstream error",
        status_code=status_code,
        retryable=retryable,
        kind=kind,
    )
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_actions={"selected-1m": ordinary_error},
    )

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value is ordinary_error
    assert [model for model, _payload in execution.calls] == [
        "test/router",
        "test/selected-1m",
    ]


@pytest.mark.asyncio
async def test_auto_context_only_stream_fallback_ignores_retryable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    ordinary_error = ProviderError(
        "ordinary upstream error",
        status_code=503,
        retryable=True,
        kind=ProviderFailureKind.UPSTREAM_STATUS,
    )
    failed = _Stream([ordinary_error])
    unused = _Stream([])
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_streams={"selected-1m": failed, "peer-a-1m": unused},
    )

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher({}, execution=execution).dispatch_stream(router, envelope)

    assert raised.value is ordinary_error
    assert failed.closed is True
    assert [model for model, _payload in execution.stream_calls] == ["test/selected-1m"]
    assert unused.closed is False


@pytest.mark.asyncio
async def test_auto_stream_never_replays_after_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    committed = _Stream(
        [
            {
                "id": "chat_1",
                "model": "selected-1m",
                "choices": [{"index": 0, "delta": {"content": "first"}}],
            },
            _context_overflow("selected-1m"),
        ]
    )
    unused = _Stream([])
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_streams={"selected-1m": committed, "peer-a-1m": unused},
    )

    opened = await GenerationDispatcher({}, execution=execution).dispatch_stream(router, envelope)
    assert (await anext(opened.frames))["choices"][0]["delta"]["content"] == "first"
    with pytest.raises(ProviderError) as raised:
        await anext(opened.frames)

    assert raised.value.signal is ProviderFailureSignal.CONTEXT_WINDOW_EXCEEDED
    assert [model for model, _payload in execution.stream_calls] == ["test/selected-1m"]
    assert unused.closed is False


@pytest.mark.asyncio
async def test_auto_returns_original_context_overflow_when_largest_candidates_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {"model": "router-maestro", "messages": [{"role": "user", "content": "hello"}]},
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    first = _context_overflow("selected-1m")
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_actions={
            "selected-1m": first,
            "peer-a-1m": _context_overflow("peer-a-1m"),
            "peer-b-1m": _context_overflow("peer-b-1m"),
        },
    )

    with pytest.raises(ProviderError) as raised:
        await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert raised.value is first
    assert raised.value.signal is ProviderFailureSignal.CONTEXT_WINDOW_EXCEEDED


@pytest.mark.asyncio
async def test_auto_overflow_continues_past_later_non_context_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {"model": "router-maestro", "messages": [{"role": "user", "content": "hello"}]},
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_actions={
            "selected-1m": _context_overflow("selected-1m"),
            "peer-a-1m": ProviderError(
                "ordinary request rejection",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            ),
            "peer-b-1m": {"id": "ok", "model": "peer-b-1m", "choices": []},
        },
    )

    result = await GenerationDispatcher({}, execution=execution).dispatch(router, envelope)

    assert result.selection.plan.model == ModelRef("test", "peer-b-1m")
    assert [model for model, _payload in execution.calls] == [
        "test/router",
        "test/selected-1m",
        "test/peer-a-1m",
        "test/peer-b-1m",
    ]


@pytest.mark.asyncio
async def test_auto_stream_overflow_continues_past_later_non_context_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _overflow_fallback_router()
    envelope = RequestEnvelope(
        OpenAIChatRuntime(),
        {
            "model": "router-maestro",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    monkeypatch.setattr("router_maestro.utils.tokens.estimate_tokens", lambda _text: 700_001)
    first = _Stream([_context_overflow("selected-1m")])
    rejected = _Stream(
        [
            ProviderError(
                "ordinary request rejection",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
            )
        ]
    )
    selected = _Stream(
        [
            {
                "id": "chat_1",
                "model": "peer-b-1m",
                "choices": [{"index": 0, "delta": {"content": "ok"}}],
            }
        ]
    )
    execution = _Execution(
        AutoTaskType.FAST.value,
        model_streams={
            "selected-1m": first,
            "peer-a-1m": rejected,
            "peer-b-1m": selected,
        },
    )

    opened = await GenerationDispatcher({}, execution=execution).dispatch_stream(router, envelope)

    assert first.closed is True
    assert rejected.closed is True
    assert opened.selection.plan.model == ModelRef("test", "peer-b-1m")
    assert [model for model, _payload in execution.stream_calls] == [
        "test/selected-1m",
        "test/peer-a-1m",
        "test/peer-b-1m",
    ]


@pytest.mark.asyncio
async def test_auto_metadata_is_union_with_largest_limits() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "priority-chain",
                "priority_chain": ["test/standard", "test/long"],
            }
        }
    )
    router = _router(
        config,
        (
            _model(
                "standard",
                max_prompt_tokens=272_000,
                max_output_tokens=32_000,
                max_context_window_tokens=304_000,
                context_window_options=(ContextWindowOption("default", 272_000, True),),
                feature_capabilities={"tools": True, "vision": False},
                reasoning_effort_values=["none", "low", "high"],
            ),
            _model(
                "long",
                max_prompt_tokens=922_000,
                max_output_tokens=128_000,
                max_context_window_tokens=1_050_000,
                context_window_options=(
                    ContextWindowOption("default", 272_000, True),
                    ContextWindowOption("long_context", 922_000, False),
                ),
                feature_capabilities={"vision": True},
                reasoning_effort_values=["minimal", "medium", "xhigh", "max"],
            ),
        ),
    )

    model = await auto_model_info(router)

    assert model.id == "router-maestro"
    assert model.virtual is True
    assert model.max_prompt_tokens == 922_000
    assert model.max_output_tokens == 128_000
    assert model.max_context_window_tokens == 1_050_000
    assert [
        (option.max_prompt_tokens, option.is_default) for option in model.context_window_options
    ] == [
        (272_000, True),
        (922_000, False),
    ]
    assert model.feature_capabilities == {"tools": True, "vision": True}
    assert model.reasoning_effort_values == [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


@pytest.mark.asyncio
async def test_auto_metadata_uses_largest_window_per_tier() -> None:
    config = PrioritiesConfig.model_validate(
        {
            "auto": {
                "mode": "priority-chain",
                "priority_chain": ["test/standard", "test/larger-default"],
            }
        }
    )
    router = _router(
        config,
        (
            _model(
                "standard",
                context_window_options=(ContextWindowOption("default", 200_000, True),),
            ),
            _model(
                "larger-default",
                context_window_options=(ContextWindowOption("default", 272_000, True),),
            ),
        ),
    )

    model = await auto_model_info(router)

    assert [
        (option.tier, option.max_prompt_tokens, option.is_default)
        for option in model.context_window_options
    ] == [("default", 272_000, True)]
