"""Tests for configurable thinking budget (Feature 4)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.config.priorities import PrioritiesConfig, ThinkingBudgetConfig
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
)
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.server.routes import anthropic as anthropic_route
from router_maestro.server.routes.anthropic import _apply_thinking_budget
from router_maestro.server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicThinkingConfig,
)
from router_maestro.utils.cache import TTLCache
from router_maestro.utils.context_window import resolve_thinking_budget
from router_maestro.utils.reasoning import EFFORT_TO_BUDGET


class TestResolveThinkingBudget:
    """Tests for resolve_thinking_budget priority chain."""

    def _config(self, **kwargs) -> ThinkingBudgetConfig:
        return ThinkingBudgetConfig(**kwargs)

    def test_client_specified_budget(self):
        """Client budget takes priority over everything."""
        budget, ttype = resolve_thinking_budget(
            client_budget=8000,
            client_thinking_type="enabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=16384,
            thinking_config=self._config(default_budget=16000),
            supports_thinking=True,
        )
        assert budget == 8000
        assert ttype == "enabled"

    def test_client_disabled(self):
        """Client disabling thinking returns None."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type="disabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=16384,
            thinking_config=self._config(default_budget=16000, auto_enable=True),
            supports_thinking=True,
        )
        assert budget is None
        assert ttype is None

    def test_client_enabled_no_budget_uses_server_default(self):
        """Client enables thinking without budget → server default applied."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type="enabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(default_budget=16000),
            supports_thinking=True,
        )
        assert budget == 16000
        assert ttype == "enabled"

    def test_client_enabled_no_budget_uses_model_budget(self):
        """Client enables thinking, per-model budget overrides default."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type="enabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(
                default_budget=16000,
                model_budgets={"claude-opus-4.6": 24000},
            ),
            supports_thinking=True,
        )
        assert budget == 24000
        assert ttype == "enabled"

    def test_auto_enable_for_capable_model(self):
        """auto_enable activates thinking for capable models."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type=None,
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(default_budget=16000, auto_enable=True),
            supports_thinking=True,
        )
        assert budget == 16000
        assert ttype == "enabled"

    def test_auto_enable_skipped_for_incapable_model(self):
        """auto_enable does nothing when model doesn't support thinking."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type=None,
            model_id="github-copilot/gpt-4o",
            max_output_tokens=16384,
            thinking_config=self._config(default_budget=16000, auto_enable=True),
            supports_thinking=False,
        )
        assert budget is None
        assert ttype is None

    def test_no_config_no_client_thinking(self):
        """No config and no client thinking → None."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type=None,
            model_id="some-model",
            max_output_tokens=16384,
            thinking_config=None,
            supports_thinking=True,
        )
        assert budget is None
        assert ttype is None

    def test_budget_normalized(self):
        """Budget is clamped to valid range."""
        budget, ttype = resolve_thinking_budget(
            client_budget=500,  # Below 1024 min
            client_thinking_type="enabled",
            model_id="test-model",
            max_output_tokens=16384,
        )
        assert budget == 1024  # Clamped to minimum
        assert ttype == "enabled"

    def test_auto_enable_false_skips(self):
        """When auto_enable=False and client didn't request, no thinking."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type=None,
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(default_budget=16000, auto_enable=False),
            supports_thinking=True,
        )
        assert budget is None
        assert ttype is None

    def test_per_model_budget_with_full_key(self):
        """Per-model budget matched by full provider/model key."""
        budget, ttype = resolve_thinking_budget(
            client_budget=None,
            client_thinking_type="enabled",
            model_id="github-copilot/claude-opus-4.6",
            max_output_tokens=32000,
            thinking_config=self._config(
                default_budget=16000,
                model_budgets={"github-copilot/claude-opus-4.6": 28000},
            ),
            supports_thinking=True,
        )
        assert budget == 28000

    def test_adaptive_thinking_budget_is_preserved_internally(self):
        """Adaptive budget remains an internal signal for provider effort mapping."""
        budget, ttype = resolve_thinking_budget(
            client_budget=10000,
            client_thinking_type="adaptive",
            model_id="test-model",
            max_output_tokens=16384,
        )
        assert budget == 10000
        assert ttype == "adaptive"

    def test_enabled_thinking_removed_without_budget_headroom(self):
        """Enabled thinking is removed atomically when no valid budget can fit."""
        budget, ttype = resolve_thinking_budget(
            client_budget=16000,
            client_thinking_type="enabled",
            model_id="test-model",
            max_output_tokens=1024,
        )
        assert budget is None
        assert ttype is None


class TestChatRequestWithThinking:
    """Tests for ChatRequest.with_thinking()."""

    def _make_request(self) -> ChatRequest:
        return ChatRequest(
            model="test-model",
            messages=[Message(role="user", content="hello")],
            thinking_budget=None,
            thinking_type=None,
        )

    def test_with_thinking_returns_new_request(self):
        """with_thinking returns a new object."""
        req = self._make_request()
        updated = req.with_thinking(thinking_budget=8000, thinking_type="enabled")
        assert updated is not req
        assert updated.thinking_budget == 8000
        assert updated.thinking_type == "enabled"

    def test_with_thinking_preserves_other_fields(self):
        """Non-thinking fields are preserved."""
        req = self._make_request()
        req.temperature = 0.5
        req.max_tokens = 1000
        req.reasoning_effort = "high"
        updated = req.with_thinking(thinking_budget=8000, thinking_type="enabled")
        assert updated.model == "test-model"
        assert updated.temperature == 0.5
        assert updated.max_tokens == 1000
        assert updated.messages == req.messages
        assert updated.reasoning_effort == "high"

    def test_original_not_mutated(self):
        """Original request is not mutated."""
        req = self._make_request()
        req.with_thinking(thinking_budget=8000, thinking_type="enabled")
        assert req.thinking_budget is None
        assert req.thinking_type is None


class _AutoThinkingFallbackProvider(BaseProvider):
    def __init__(self, name: str, *, fail: bool = False):
        self.name = name
        self.fail = fail
        self.requests: list[ChatRequest] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM}))

    def is_authenticated(self) -> bool:
        return True

    async def list_models(self):
        return []

    async def chat_completion(self, request):
        self.requests.append(request)
        if self.fail:
            raise ProviderError(
                "retry",
                status_code=503,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_STATUS,
            )
        return ChatResponse(content="ok", model=request.model)

    async def chat_completion_stream(self, request):
        self.requests.append(request)
        if self.fail:
            raise ProviderError(
                "retry",
                status_code=503,
                retryable=True,
                kind=ProviderFailureKind.UPSTREAM_STATUS,
            )
        yield ChatStreamChunk(content="ok")
        yield ChatStreamChunk(content="", finish_reason="stop")


def _auto_thinking_fallback_router(
    *,
    primary_fail: bool = True,
    fallback_supports_thinking: bool = False,
    fallback_max_output_tokens: int = 32768,
    auto_enable: bool = True,
):
    primary = _AutoThinkingFallbackProvider("primary", fail=primary_fail)
    fallback = _AutoThinkingFallbackProvider("fallback")
    primary_info = ModelInfo(
        id="shared",
        name="Shared",
        provider="primary",
        max_output_tokens=32768,
        supports_thinking=True,
        operation_capabilities={Operation.CHAT.value: True, Operation.CHAT_STREAM.value: True},
        feature_capabilities={"reasoning": True},
    )
    fallback_info = ModelInfo(
        id="shared",
        name="Shared",
        provider="fallback",
        max_output_tokens=fallback_max_output_tokens,
        supports_thinking=fallback_supports_thinking,
        operation_capabilities={Operation.CHAT.value: True, Operation.CHAT_STREAM.value: True},
        feature_capabilities={"reasoning": fallback_supports_thinking},
    )
    router = Router.__new__(Router)
    router.providers = {"primary": primary, "fallback": fallback}
    router._models_cache = {
        "shared": ("primary", primary_info),
        "primary/shared": ("primary", primary_info),
        "fallback/shared": ("fallback", fallback_info),
    }
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._models_cache_ttl.set(True)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    priorities = PrioritiesConfig(
        priorities=["primary/shared", "fallback/shared"],
        fallback={"strategy": "priority", "maxRetries": 1},
        thinking=ThinkingBudgetConfig(default_budget=16000, auto_enable=auto_enable),
    )
    router._priorities_cache.set(priorities)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._providers_ttl.set(True)
    return router, primary, fallback, priorities


class TestAnthropicRouteThinkingBudget:
    """Tests for route-level thinking budget resolution."""

    @pytest.mark.asyncio
    async def test_explicit_disabled_thinking_survives_route_resolution(self, monkeypatch):
        router = MagicMock()
        resolver = MagicMock()
        monkeypatch.setattr(anthropic_route, "resolve_thinking_budget", resolver)
        request = ChatRequest(
            model="claude-opus-4.8",
            messages=[Message(role="user", content="hi")],
            max_tokens=4096,
            thinking_type="disabled",
        )

        result = await _apply_thinking_budget(router, request, request.model)

        assert result is request
        assert result.thinking_type == "disabled"
        resolver.assert_not_called()
        router.get_model_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_disabled_thinking_clears_conflicting_budget(self, monkeypatch):
        from router_maestro.providers.copilot import CopilotProvider

        router = MagicMock()
        resolver = MagicMock()
        monkeypatch.setattr(anthropic_route, "resolve_thinking_budget", resolver)
        request = ChatRequest(
            model="claude-opus-4.8",
            messages=[Message(role="user", content="hi")],
            max_tokens=4096,
            thinking_type="disabled",
            thinking_budget=1024,
        )

        result = await _apply_thinking_budget(router, request, request.model)
        payload = CopilotProvider()._build_chat_payload(result, stream=False)

        assert result.thinking_type == "disabled"
        assert result.thinking_budget is None
        assert "reasoning_effort" not in payload
        assert "thinking_budget" not in payload
        resolver.assert_not_called()
        router.get_model_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_dash_alias_uses_catalog_max_output_for_client_budget(self, monkeypatch):
        """claude-opus-4-6 should use claude-opus-4.6 metadata, not 16k fallback."""

        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: type(
                "Config",
                (),
                {"thinking": ThinkingBudgetConfig(default_budget=16000, auto_enable=False)},
            )(),
        )
        model_info = ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            max_output_tokens=64000,
            supports_thinking=True,
        )
        router = Router.__new__(Router)
        router.providers = {}
        router._models_cache = {
            "claude-opus-4.6": ("github-copilot", model_info),
            "github-copilot/claude-opus-4.6": ("github-copilot", model_info),
        }
        router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._models_cache_ttl.set(True)
        router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        router._fuzzy_cache = {}
        router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._providers_ttl.set(True)
        request = ChatRequest(
            model="claude-opus-4-6",
            messages=[Message(role="user", content="hi")],
            max_tokens=64000,
            thinking_budget=63999,
            thinking_type="enabled",
        )

        result = await _apply_thinking_budget(router, request, "claude-opus-4-6")

        assert result.thinking_budget == EFFORT_TO_BUDGET["max"]
        assert result.thinking_type == "enabled"

    @pytest.mark.asyncio
    async def test_messages_uses_the_planned_candidate_for_thinking_preprocessing(
        self,
        monkeypatch,
    ):
        """A bare alias must not use another provider's catalog limits."""

        class CaptureProvider(BaseProvider):
            def __init__(self, name: str):
                self._name = name
                self.request = None

            @property
            def name(self) -> str:
                return self._name

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(
                    operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM})
                )

            def is_authenticated(self) -> bool:
                return True

            async def ensure_token(self) -> None:
                return None

            async def list_models(self):
                return []

            async def chat_completion(self, request):
                self.request = request
                return ChatResponse(content="ok", model=request.model)

            async def chat_completion_stream(self, request):
                if False:
                    yield request

        class RawRequest:
            headers = {}

            async def body(self) -> bytes:
                return (
                    b'{"model":"shared","max_tokens":32768,'
                    b'"thinking":{"type":"enabled","budget_tokens":8192},'
                    b'"messages":[{"role":"user","content":"hi"}]}'
                )

        first = CaptureProvider("first")
        second = CaptureProvider("second")
        first_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="first",
            max_output_tokens=1024,
            supports_thinking=False,
            operation_capabilities={Operation.CHAT.value: False},
            feature_capabilities={"reasoning": False},
        )
        second_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="second",
            max_output_tokens=32768,
            supports_thinking=True,
            operation_capabilities={Operation.CHAT.value: True},
            feature_capabilities={"reasoning": True},
        )
        router = Router.__new__(Router)
        router.providers = {"first": first, "second": second}
        router._models_cache = {
            "shared": ("first", first_info),
            "first/shared": ("first", first_info),
            "second/shared": ("second", second_info),
        }
        router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._models_cache_ttl.set(True)
        router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        priorities = type(
            "Config",
            (),
            {
                "priorities": ["first/shared", "second/shared"],
                "fallback": type(
                    "Fallback",
                    (),
                    {"strategy": "priority", "maxRetries": 2},
                )(),
                "thinking": ThinkingBudgetConfig(default_budget=16000, auto_enable=False),
                "model_overrides": {},
            },
        )()
        router._priorities_cache.set(priorities)
        router._fuzzy_cache = {}
        router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._providers_ttl.set(True)
        monkeypatch.setattr(anthropic_route, "get_router", lambda: router)
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: priorities,
        )

        response = await anthropic_route.messages(
            AnthropicMessagesRequest(
                model="shared",
                max_tokens=32768,
                messages=[{"role": "user", "content": "hi"}],
                thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=8192),
            ),
            RawRequest(),
        )

        assert response.model == "second/shared"
        assert first.request is None
        assert second.request is not None
        assert second.request.thinking_type == "enabled"
        assert second.request.thinking_budget == 8192

    def test_stream_message_start_estimate_uses_capability_selected_provider(
        self,
        monkeypatch,
    ):
        class StreamProvider(BaseProvider):
            def __init__(self, name: str, operations: frozenset[Operation]):
                self.name = name
                self._capabilities = ProviderCapabilities(operations=operations)

            @property
            def capabilities(self) -> ProviderCapabilities:
                return self._capabilities

            def is_authenticated(self) -> bool:
                return True

            async def list_models(self):
                return []

            async def chat_completion(self, request):
                raise AssertionError("not used")

            async def chat_completion_stream(self, request):
                yield ChatStreamChunk(content="ok")
                yield ChatStreamChunk(content="", finish_reason="stop")

        copilot = StreamProvider("github-copilot", frozenset({Operation.CHAT}))
        anthropic = StreamProvider(
            "anthropic",
            frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        )
        copilot_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="github-copilot",
            operation_capabilities={Operation.CHAT_STREAM.value: False},
            feature_capabilities={"tools": True},
        )
        anthropic_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="anthropic",
            operation_capabilities={Operation.CHAT_STREAM.value: True},
            feature_capabilities={"tools": True},
        )
        model_router = Router.__new__(Router)
        model_router.providers = {
            "github-copilot": copilot,
            "anthropic": anthropic,
        }
        model_router._models_cache = {
            "shared": ("github-copilot", copilot_info),
            "github-copilot/shared": ("github-copilot", copilot_info),
            "anthropic/shared": ("anthropic", anthropic_info),
        }
        model_router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        model_router._models_cache_ttl.set(True)
        model_router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        priorities = PrioritiesConfig(
            priorities=["github-copilot/shared", "anthropic/shared"],
            fallback={"strategy": "priority", "maxRetries": 1},
        )
        model_router._priorities_cache.set(priorities)
        model_router._fuzzy_cache = {}
        model_router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        model_router._providers_ttl.set(True)

        captured = {}
        plan_chat_completion = model_router.plan_chat_completion

        async def capture_plan(request, *, stream):
            plan = await plan_chat_completion(request, stream=stream)
            captured["plan"] = plan
            return plan

        monkeypatch.setattr(model_router, "plan_chat_completion", capture_plan)
        monkeypatch.setattr(anthropic_route, "get_router", lambda: model_router)
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: priorities,
        )

        payload = {
            "model": "shared",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Use the lookup tool"}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look up a value",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
        }
        request = AnthropicMessagesRequest.model_validate(payload)
        app = FastAPI()
        app.include_router(anthropic_route.router)

        response = TestClient(app).post("/v1/messages", json=payload)

        assert response.status_code == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        message_start = next(event for event in events if event.get("type") == "message_start")
        route_plan = captured["plan"]
        assert route_plan.primary.model.provider == "anthropic"
        expected = anthropic_route._estimate_input_tokens(
            request,
            route_plan.primary.model.provider,
        )
        legacy = anthropic_route._estimate_input_tokens(request, "github-copilot")
        assert expected != legacy
        assert message_start["message"]["usage"]["input_tokens"] == expected

    def test_stream_message_start_estimate_follows_precommit_fallback(
        self,
        monkeypatch,
    ):
        class StreamProvider(BaseProvider):
            def __init__(self, name: str, *, fail: bool = False):
                self.name = name
                self.fail = fail

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(
                    operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM})
                )

            def is_authenticated(self) -> bool:
                return True

            async def list_models(self):
                return []

            async def chat_completion(self, request):
                raise AssertionError("not used")

            async def chat_completion_stream(self, request):
                if self.fail:
                    raise ProviderError(
                        "retry",
                        status_code=503,
                        retryable=True,
                        kind=ProviderFailureKind.UPSTREAM_STATUS,
                    )
                yield ChatStreamChunk(content="ok")
                yield ChatStreamChunk(content="", finish_reason="stop")

        copilot = StreamProvider("github-copilot", fail=True)
        anthropic = StreamProvider("anthropic")
        copilot_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="github-copilot",
            operation_capabilities={Operation.CHAT_STREAM.value: True},
            feature_capabilities={"tools": True},
        )
        anthropic_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="anthropic",
            operation_capabilities={Operation.CHAT_STREAM.value: True},
            feature_capabilities={"tools": True},
        )
        model_router = Router.__new__(Router)
        model_router.providers = {
            "github-copilot": copilot,
            "anthropic": anthropic,
        }
        model_router._models_cache = {
            "shared": ("github-copilot", copilot_info),
            "github-copilot/shared": ("github-copilot", copilot_info),
            "anthropic/shared": ("anthropic", anthropic_info),
        }
        model_router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        model_router._models_cache_ttl.set(True)
        model_router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        priorities = PrioritiesConfig(
            priorities=["github-copilot/shared", "anthropic/shared"],
            fallback={"strategy": "priority", "maxRetries": 1},
        )
        model_router._priorities_cache.set(priorities)
        model_router._fuzzy_cache = {}
        model_router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        model_router._providers_ttl.set(True)
        monkeypatch.setattr(anthropic_route, "get_router", lambda: model_router)
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: priorities,
        )

        payload = {
            "model": "shared",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Use the lookup tool"}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look up a value",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
        }
        request = AnthropicMessagesRequest.model_validate(payload)
        app = FastAPI()
        app.include_router(anthropic_route.router)

        response = TestClient(app).post("/v1/messages", json=payload)

        assert response.status_code == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        message_start = next(event for event in events if event.get("type") == "message_start")
        expected = anthropic_route._estimate_input_tokens(request, "anthropic")
        primary = anthropic_route._estimate_input_tokens(request, "github-copilot")
        assert expected != primary
        assert message_start["message"]["model"] == "anthropic/shared"
        assert message_start["message"]["usage"]["input_tokens"] == expected

    @pytest.mark.asyncio
    async def test_runtime_fallback_rebinds_thinking_to_the_actual_candidate_snapshot(
        self,
        monkeypatch,
    ):
        class CandidateProvider(BaseProvider):
            def __init__(self, name: str, *, fail: bool = False):
                self.name = name
                self.fail = fail
                self.requests: list[ChatRequest] = []

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(
                    operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM})
                )

            def is_authenticated(self) -> bool:
                return True

            async def list_models(self):
                return []

            async def chat_completion(self, request):
                self.requests.append(request)
                if self.fail:
                    raise ProviderError(
                        "retry",
                        status_code=503,
                        retryable=True,
                        kind=ProviderFailureKind.UPSTREAM_STATUS,
                    )
                return ChatResponse(content="ok", model=request.model)

            async def chat_completion_stream(self, request):
                raise AssertionError("not used")
                yield

        class RawRequest:
            headers = {}

            async def body(self) -> bytes:
                return (
                    b'{"model":"shared","max_tokens":32768,'
                    b'"thinking":{"type":"enabled","budget_tokens":8192},'
                    b'"messages":[{"role":"user","content":"hi"}]}'
                )

        primary = CandidateProvider("primary", fail=True)
        fallback = CandidateProvider("fallback")
        primary_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="primary",
            max_output_tokens=32768,
            supports_thinking=True,
            operation_capabilities={Operation.CHAT.value: True},
            feature_capabilities={"reasoning": True},
        )
        fallback_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="fallback",
            max_output_tokens=2048,
            supports_thinking=True,
            operation_capabilities={Operation.CHAT.value: True},
            feature_capabilities={"reasoning": True},
        )
        router = Router.__new__(Router)
        router.providers = {"primary": primary, "fallback": fallback}
        router._models_cache = {
            "shared": ("primary", primary_info),
            "primary/shared": ("primary", primary_info),
            "fallback/shared": ("fallback", fallback_info),
        }
        router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._models_cache_ttl.set(True)
        router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        priorities = PrioritiesConfig(
            priorities=["primary/shared", "fallback/shared"],
            fallback={"strategy": "priority", "maxRetries": 1},
            thinking=ThinkingBudgetConfig(default_budget=16000, auto_enable=False),
        )
        router._priorities_cache.set(priorities)
        router._fuzzy_cache = {}
        router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._providers_ttl.set(True)
        monkeypatch.setattr(anthropic_route, "get_router", lambda: router)
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: priorities,
        )

        response = await anthropic_route.messages(
            AnthropicMessagesRequest(
                model="shared",
                max_tokens=32768,
                messages=[{"role": "user", "content": "hi"}],
                thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=8192),
            ),
            RawRequest(),
        )

        assert response.model == "fallback/shared"
        assert primary.requests[0].thinking_budget == 8192
        assert fallback.requests[0].thinking_type == "enabled"
        assert fallback.requests[0].thinking_budget == 2047

    @pytest.mark.asyncio
    async def test_stream_fallback_rebinds_thinking_before_the_first_upstream_chunk(
        self,
        monkeypatch,
    ):
        class StreamProvider(BaseProvider):
            def __init__(self, name: str, *, fail: bool = False):
                self.name = name
                self.fail = fail
                self.requests: list[ChatRequest] = []

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(
                    operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM})
                )

            def is_authenticated(self) -> bool:
                return True

            async def list_models(self):
                return []

            async def chat_completion(self, request):
                raise AssertionError("not used")

            async def chat_completion_stream(self, request):
                self.requests.append(request)
                if self.fail:
                    raise ProviderError(
                        "retry",
                        status_code=503,
                        retryable=True,
                        kind=ProviderFailureKind.UPSTREAM_STATUS,
                    )
                yield ChatStreamChunk(content="ok")
                yield ChatStreamChunk(content="", finish_reason="stop")

        primary = StreamProvider("primary", fail=True)
        fallback = StreamProvider("fallback")
        primary_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="primary",
            max_output_tokens=32768,
            supports_thinking=True,
            operation_capabilities={Operation.CHAT_STREAM.value: True},
            feature_capabilities={"reasoning": True},
        )
        fallback_info = ModelInfo(
            id="shared",
            name="Shared",
            provider="fallback",
            max_output_tokens=2048,
            supports_thinking=True,
            operation_capabilities={Operation.CHAT_STREAM.value: True},
            feature_capabilities={"reasoning": True},
        )
        router = Router.__new__(Router)
        router.providers = {"primary": primary, "fallback": fallback}
        router._models_cache = {
            "shared": ("primary", primary_info),
            "primary/shared": ("primary", primary_info),
            "fallback/shared": ("fallback", fallback_info),
        }
        router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._models_cache_ttl.set(True)
        router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        priorities = PrioritiesConfig(
            priorities=["primary/shared", "fallback/shared"],
            fallback={"strategy": "priority", "maxRetries": 1},
            thinking=ThinkingBudgetConfig(default_budget=16000, auto_enable=False),
        )
        router._priorities_cache.set(priorities)
        router._fuzzy_cache = {}
        router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._providers_ttl.set(True)
        monkeypatch.setattr(anthropic_route, "get_router", lambda: router)
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: priorities,
        )

        class RawRequest:
            headers = {}

            async def body(self) -> bytes:
                return (
                    b'{"model":"shared","stream":true,"max_tokens":32768,'
                    b'"thinking":{"type":"enabled","budget_tokens":8192},'
                    b'"messages":[{"role":"user","content":"hi"}]}'
                )

        response = await anthropic_route.messages(
            AnthropicMessagesRequest(
                model="shared",
                stream=True,
                max_tokens=32768,
                messages=[{"role": "user", "content": "hi"}],
                thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=8192),
            ),
            RawRequest(),
        )
        chunks = [
            chunk.encode() if isinstance(chunk, str) else chunk
            async for chunk in response.body_iterator
        ]
        body = b"".join(chunks).decode()

        assert "message_stop" in body
        assert primary.requests[0].thinking_budget == 8192
        assert fallback.requests[0].thinking_type == "enabled"
        assert fallback.requests[0].thinking_budget == 2047

    @pytest.mark.asyncio
    async def test_auto_enabled_primary_keeps_non_reasoning_fallback(self, monkeypatch):
        router, primary, fallback, priorities = _auto_thinking_fallback_router()
        monkeypatch.setattr(anthropic_route, "get_router", lambda: router)
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: priorities,
        )

        class RawRequest:
            headers = {}

            async def body(self) -> bytes:
                return (
                    b'{"model":"shared","max_tokens":32768,'
                    b'"messages":[{"role":"user","content":"hi"}]}'
                )

        response = await anthropic_route.messages(
            AnthropicMessagesRequest(
                model="shared",
                max_tokens=32768,
                messages=[{"role": "user", "content": "hi"}],
            ),
            RawRequest(),
        )

        assert response.model == "fallback/shared"
        assert primary.requests[0].thinking_type == "enabled"
        assert primary.requests[0].thinking_budget == 16000
        assert fallback.requests[0].thinking_type is None
        assert fallback.requests[0].thinking_budget is None

    @pytest.mark.asyncio
    async def test_unscoped_reasoning_feature_drift_is_rejected(self):
        router, _primary, _fallback, _priorities = _auto_thinking_fallback_router()
        request = ChatRequest(
            model="shared",
            messages=[Message(role="user", content="hi")],
        )
        plan = await router.plan_chat_completion(request, stream=False)
        enhanced = request.with_thinking(thinking_budget=16000, thinking_type="enabled")

        with pytest.raises(ProviderError, match="features do not match"):
            router.prepare_planned_chat_completion(plan, enhanced)

    @pytest.mark.asyncio
    async def test_explicit_client_reasoning_cannot_be_removed_during_preparation(self):
        router, _primary, _fallback, _priorities = _auto_thinking_fallback_router()
        request = ChatRequest(
            model="shared",
            messages=[Message(role="user", content="hi")],
            thinking_budget=16000,
            thinking_type="enabled",
        )
        plan = await router.plan_chat_completion(request, stream=False)
        stripped = request.with_thinking(thinking_budget=None, thinking_type=None)

        with pytest.raises(RequestOptionError, match="features do not match") as exc_info:
            router.prepare_planned_chat_completion(plan, stripped)

        assert exc_info.value.parameter == "thinking_budget"

    @pytest.mark.asyncio
    async def test_unscoped_fallback_reasoning_enhancement_removes_that_fallback(self):
        router, _primary, _fallback, _priorities = _auto_thinking_fallback_router()
        request = ChatRequest(
            model="shared",
            messages=[Message(role="user", content="hi")],
        )
        plan = await router.plan_chat_completion(request, stream=False)
        primary_request = request.with_thinking(
            thinking_budget=16000,
            thinking_type="enabled",
        )

        prepared = router.prepare_planned_chat_completion(
            plan,
            primary_request,
            candidate_requests={plan.primary.model: primary_request},
        )

        assert prepared.plan.fallbacks == ()

    @pytest.mark.asyncio
    async def test_fallback_snapshot_cannot_hide_real_client_feature_drift(self):
        router, _primary, _fallback, _priorities = _auto_thinking_fallback_router()
        request = ChatRequest(
            model="shared",
            messages=[Message(role="user", content="hi")],
        )
        plan = await router.plan_chat_completion(request, stream=False)
        fallback_request = ChatRequest(
            model="shared",
            messages=[Message(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "drifted"}}],
        )

        with pytest.raises(ProviderError, match="features do not match"):
            router.prepare_planned_chat_completion(
                plan,
                request,
                candidate_requests={
                    plan.primary.model: request,
                    plan.prevalidation_fallbacks[0].model: fallback_request,
                },
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_explicit_reasoning_skips_fallback_that_loses_thinking_headroom(
        self,
        stream,
    ):
        router, _primary, fallback, _priorities = _auto_thinking_fallback_router(
            primary_fail=False,
            fallback_supports_thinking=True,
            fallback_max_output_tokens=1024,
            auto_enable=False,
        )
        request = ChatRequest(
            model="shared",
            messages=[Message(role="user", content="hi")],
            max_tokens=32768,
            stream=stream,
            thinking_budget=8192,
            thinking_type="enabled",
        )
        plan = await router.plan_chat_completion(request, stream=stream)
        candidate_requests = {
            candidate.model: await _apply_thinking_budget(
                router,
                request,
                request.model,
                candidate=candidate,
            )
            for candidate in (plan.primary, *plan.prevalidation_fallbacks)
        }

        prepared = router.prepare_planned_chat_completion(
            plan,
            candidate_requests[plan.primary.model],
            candidate_requests=candidate_requests,
        )

        assert prepared.plan.fallbacks == ()
        assert candidate_requests[plan.primary.model].thinking_type == "enabled"
        assert candidate_requests[plan.primary.model].thinking_budget == 8192
        fallback_request = candidate_requests[plan.prevalidation_fallbacks[0].model]
        assert fallback_request.thinking_type is None
        assert fallback_request.thinking_budget is None
        assert fallback.requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_scoped_primary_that_loses_thinking_headroom_still_proceeds(
        self,
        stream,
    ):
        """A scoped primary whose thinking is disabled for lack of output
        headroom must proceed rather than 400.

        When the requested ``max_tokens`` is too small to fit any thinking
        budget, ``_apply_thinking_budget`` disables thinking for that candidate.
        The primary is the only executable route, and a non-reasoning request
        always runs on a reasoning-capable route, so preparation must succeed.
        A scoped fallback that loses the same headroom is still dropped.
        """
        router, _primary, _fallback, _priorities = _auto_thinking_fallback_router(
            primary_fail=False,
            fallback_supports_thinking=True,
            auto_enable=False,
        )
        request = ChatRequest(
            model="shared",
            messages=[Message(role="user", content="hi")],
            max_tokens=200,
            stream=stream,
            thinking_budget=8192,
            thinking_type="enabled",
        )
        plan = await router.plan_chat_completion(request, stream=stream)
        candidate_requests = {
            candidate.model: await _apply_thinking_budget(
                router,
                request,
                request.model,
                candidate=candidate,
            )
            for candidate in (plan.primary, *plan.prevalidation_fallbacks)
        }

        prepared = router.prepare_planned_chat_completion(
            plan,
            candidate_requests[plan.primary.model],
            candidate_requests=candidate_requests,
        )

        assert prepared.plan.fallbacks == ()
        assert candidate_requests[plan.primary.model].thinking_type is None
        assert candidate_requests[plan.primary.model].thinking_budget is None

    @pytest.mark.asyncio
    async def test_auto_enabled_primary_keeps_non_reasoning_stream_fallback(self, monkeypatch):
        router, primary, fallback, priorities = _auto_thinking_fallback_router()
        monkeypatch.setattr(anthropic_route, "get_router", lambda: router)
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: priorities,
        )

        class RawRequest:
            headers = {}

            async def body(self) -> bytes:
                return (
                    b'{"model":"shared","stream":true,"max_tokens":32768,'
                    b'"messages":[{"role":"user","content":"hi"}]}'
                )

        response = await anthropic_route.messages(
            AnthropicMessagesRequest(
                model="shared",
                stream=True,
                max_tokens=32768,
                messages=[{"role": "user", "content": "hi"}],
            ),
            RawRequest(),
        )
        chunks = [
            chunk.encode() if isinstance(chunk, str) else chunk
            async for chunk in response.body_iterator
        ]
        body = b"".join(chunks).decode()

        assert "message_stop" in body
        assert primary.requests[0].thinking_type == "enabled"
        assert primary.requests[0].thinking_budget == 16000
        assert fallback.requests[0].thinking_type is None
        assert fallback.requests[0].thinking_budget is None

    @pytest.mark.asyncio
    async def test_explicit_effort_clears_budget_without_resolving_server_default(
        self, monkeypatch
    ):
        """Effort is authoritative and bypasses all client/server budget resolution."""
        router = MagicMock()
        resolver = MagicMock()
        monkeypatch.setattr(anthropic_route, "resolve_thinking_budget", resolver)
        request = ChatRequest(
            model="claude-opus-4.8",
            messages=[Message(role="user", content="hi")],
            max_tokens=64000,
            thinking_budget=4096,
            thinking_type="adaptive",
            reasoning_effort="xhigh",
        )

        result = await _apply_thinking_budget(router, request, "claude-opus-4-8")

        assert result.thinking_budget is None
        assert result.thinking_type == "adaptive"
        assert result.reasoning_effort == "xhigh"
        router.get_model_info.assert_not_called()
        resolver.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_effort_preserves_required_enabled_budget(self, monkeypatch):
        """Effort may coexist with the budget required by manual enabled thinking."""
        router = MagicMock()
        router.get_model_info = AsyncMock(return_value=None)
        resolver = MagicMock()
        resolver.return_value = (4096, "enabled")
        monkeypatch.setattr(anthropic_route, "resolve_thinking_budget", resolver)
        request = ChatRequest(
            model="claude-opus-4.6",
            messages=[Message(role="user", content="hi")],
            max_tokens=64000,
            thinking_budget=4096,
            thinking_type="enabled",
            reasoning_effort="xhigh",
        )

        result = await _apply_thinking_budget(router, request, "claude-opus-4-6")

        assert result.thinking_budget == 4096
        assert result.thinking_type == "enabled"
        assert result.reasoning_effort == "xhigh"
        resolver.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_effort_fills_missing_enabled_budget(self, monkeypatch):
        """Manual enabled thinking remains valid when the client omits its budget."""
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: type(
                "Config",
                (),
                {"thinking": ThinkingBudgetConfig(default_budget=16000, auto_enable=False)},
            )(),
        )
        model_info = ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            max_output_tokens=64000,
            supports_thinking=True,
        )
        router = Router.__new__(Router)
        router.providers = {}
        router._models_cache = {
            "claude-opus-4.6": ("github-copilot", model_info),
            "github-copilot/claude-opus-4.6": ("github-copilot", model_info),
        }
        router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._models_cache_ttl.set(True)
        router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
        router._fuzzy_cache = {}
        router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
        router._providers_ttl.set(True)
        request = ChatRequest(
            model="claude-opus-4.6",
            messages=[Message(role="user", content="hi")],
            max_tokens=64000,
            thinking_type="enabled",
            reasoning_effort="xhigh",
        )

        result = await _apply_thinking_budget(router, request, "claude-opus-4-6")

        assert result.thinking_budget == 16000
        assert result.thinking_type == "enabled"
        assert result.reasoning_effort == "xhigh"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("client_budget", "expected_budget"),
        [(500, 1024), (63999, EFFORT_TO_BUDGET["max"])],
    )
    async def test_explicit_effort_normalizes_enabled_budget(
        self, monkeypatch, client_budget, expected_budget
    ):
        """Manual budgets remain within Anthropic's accepted token range."""
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: type(
                "Config",
                (),
                {"thinking": ThinkingBudgetConfig(default_budget=16000, auto_enable=False)},
            )(),
        )
        model_info = ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            max_output_tokens=64000,
            supports_thinking=True,
        )
        router = MagicMock()
        router.get_model_info = AsyncMock(return_value=model_info)
        request = ChatRequest(
            model="claude-opus-4.6",
            messages=[Message(role="user", content="hi")],
            max_tokens=64000,
            thinking_budget=client_budget,
            thinking_type="enabled",
            reasoning_effort="xhigh",
        )

        result = await _apply_thinking_budget(router, request, "claude-opus-4-6")

        assert result.thinking_budget == expected_budget
        assert result.thinking_type == "enabled"
        assert result.reasoning_effort == "xhigh"

    @pytest.mark.asyncio
    async def test_enabled_budget_is_capped_by_request_max_tokens(self, monkeypatch):
        """Client max_tokens is the wire-level upper bound for manual thinking."""
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: type(
                "Config",
                (),
                {"thinking": ThinkingBudgetConfig(default_budget=16000, auto_enable=False)},
            )(),
        )
        model_info = ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            max_output_tokens=64000,
            supports_thinking=True,
        )
        router = MagicMock()
        router.get_model_info = AsyncMock(return_value=model_info)
        request = ChatRequest(
            model="claude-opus-4.6",
            messages=[Message(role="user", content="hi")],
            max_tokens=4096,
            thinking_budget=16000,
            thinking_type="enabled",
            reasoning_effort="xhigh",
        )

        result = await _apply_thinking_budget(router, request, "claude-opus-4-6")

        assert result.thinking_budget == 4095
        assert result.thinking_type == "enabled"

    @pytest.mark.asyncio
    async def test_enabled_thinking_is_removed_without_budget_headroom(self, monkeypatch):
        """Never emit enabled thinking when max_tokens cannot fit the minimum budget."""
        monkeypatch.setattr(
            "router_maestro.config.load_priorities_config",
            lambda: type(
                "Config",
                (),
                {"thinking": ThinkingBudgetConfig(default_budget=16000, auto_enable=False)},
            )(),
        )
        model_info = ModelInfo(
            id="claude-opus-4.6",
            name="Claude Opus 4.6",
            provider="github-copilot",
            max_output_tokens=64000,
            supports_thinking=True,
        )
        router = MagicMock()
        router.get_model_info = AsyncMock(return_value=model_info)
        request = ChatRequest(
            model="claude-opus-4.6",
            messages=[Message(role="user", content="hi")],
            max_tokens=1024,
            thinking_budget=16000,
            thinking_type="enabled",
            reasoning_effort="xhigh",
        )

        result = await _apply_thinking_budget(router, request, "claude-opus-4-6")

        assert result.thinking_budget is None
        assert result.thinking_type is None

    @pytest.mark.asyncio
    async def test_messages_logs_inbound_thinking_metadata(self, monkeypatch):
        """Inbound logs include raw thinking metadata needed to debug client budgets."""

        class RawRequest:
            headers = {}

            async def body(self) -> bytes:
                return (
                    b'{"model":"claude-opus-4-6","max_tokens":64000,'
                    b'"thinking":{"type":"enabled","budget_tokens":63999},'
                    b'"messages":[{"role":"user","content":"hi"}]}'
                )

        request = AnthropicMessagesRequest(
            model="test",
            max_tokens=64000,
            messages=[{"role": "user", "content": "hi"}],
            thinking=AnthropicThinkingConfig(type="enabled", budget_tokens=63999),
        )
        log = MagicMock()
        monkeypatch.setattr(anthropic_route, "logger", log)

        await anthropic_route.messages(request, RawRequest())

        log.info.assert_called_once()
        _message, model, stream, max_tokens, thinking, raw_thinking, effort = (
            log.info.call_args.args
        )
        assert model == "test"
        assert stream is False
        assert max_tokens == 64000
        assert thinking == {"type": "enabled", "budget_tokens": 63999}
        assert raw_thinking == {"type": "enabled", "budget_tokens": 63999}
        assert effort is None
