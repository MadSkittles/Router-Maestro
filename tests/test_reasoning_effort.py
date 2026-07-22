"""Tests for reasoning_effort end-to-end passthrough."""

from __future__ import annotations

import logging

import pytest

from router_maestro.providers.base import (
    ChatRequest,
    Message,
    ProviderError,
    ProviderFailureKind,
    ResponsesRequest,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.utils import reasoning
from router_maestro.utils.reasoning import (
    EFFORT_TO_BUDGET,
    VALID_EFFORTS,
    budget_to_effort,
    downgrade_for_upstream,
    effort_to_budget,
    pick_closest_effort,
)


class _StubOpenAIProvider(OpenAIChatProvider):
    name = "stub"

    def _get_headers(self) -> dict[str, str]:
        return {}

    async def list_models(self):  # pragma: no cover - unused
        return []

    def is_authenticated(self) -> bool:  # pragma: no cover - unused
        return True


def _stub_request(**kwargs) -> ChatRequest:
    return ChatRequest(
        model="stub-model",
        messages=[Message(role="user", content="hi")],
        **kwargs,
    )


class TestEffortMapping:
    @pytest.mark.parametrize(
        "effort,expected",
        [
            ("low", 1024),
            ("medium", 4096),
            ("high", 8192),
            ("xhigh", 16384),
            ("max", 32768),
            ("minimal", None),
            ("HIGH", 8192),
            ("MAX", 32768),
            (None, None),
            ("bogus", None),
        ],
    )
    def test_effort_to_budget(self, effort, expected):
        assert effort_to_budget(effort) == expected

    def test_budget_to_effort_picks_highest_fitting(self):
        assert budget_to_effort(None) is None
        assert budget_to_effort(100) is None
        assert budget_to_effort(1024) == "low"
        assert budget_to_effort(4096) == "medium"
        assert budget_to_effort(8000) == "medium"
        assert budget_to_effort(8192) == "high"
        assert budget_to_effort(16000) == "high"
        assert budget_to_effort(EFFORT_TO_BUDGET["xhigh"]) == "xhigh"
        assert budget_to_effort(EFFORT_TO_BUDGET["max"]) == "max"
        assert budget_to_effort(64000) == "max"

    def test_minimal_is_ordered_below_low_without_an_implicit_budget(self):
        assert reasoning.EFFORT_ORDER == (
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )
        assert VALID_EFFORTS == reasoning.EFFORT_ORDER
        assert "minimal" not in EFFORT_TO_BUDGET
        assert effort_to_budget("minimal") is None
        assert budget_to_effort(1023) is None

    def test_picker_supports_minimal_and_rejects_unknown_desired_tiers(self):
        assert pick_closest_effort("minimal", ["minimal", "low"]) == "minimal"
        assert pick_closest_effort("low", ["minimal", "future"]) == "minimal"
        assert pick_closest_effort("minimal", ["low"]) is None
        assert pick_closest_effort("low", ["none"]) is None
        assert pick_closest_effort("low", ["none", "low"]) == "low"
        assert pick_closest_effort("minimal", ["none", "low"]) is None
        assert pick_closest_effort("ultra", ["low", "high"]) is None
        assert pick_closest_effort("low", ["future"]) is None

    def test_resolve_effort_within_catalog_clamps_up_when_below_floor(self):
        from router_maestro.utils.reasoning import resolve_effort_within_catalog

        # At/below desired: behaves like pick_closest_effort.
        assert resolve_effort_within_catalog("high", ["low", "medium", "high"]) == "high"
        assert resolve_effort_within_catalog("xhigh", ["low", "medium"]) == "medium"
        # Below every available tier: clamp UP to the lowest available.
        assert resolve_effort_within_catalog("low", ["medium", "high"]) == "medium"
        assert resolve_effort_within_catalog("minimal", ["high"]) == "high"
        # Unknown desired tier still resolves to the lowest available tier.
        assert resolve_effort_within_catalog("ultra", ["low", "high"]) == "low"
        # Empty / no valid tier -> None.
        assert resolve_effort_within_catalog("low", []) is None
        assert resolve_effort_within_catalog("low", ["bogus"]) is None

    def test_downgrade_for_upstream(self):
        assert downgrade_for_upstream(None) is None
        assert downgrade_for_upstream("low") == "low"
        assert downgrade_for_upstream("xhigh") == "high"
        assert downgrade_for_upstream("max") == "high"
        assert downgrade_for_upstream("garbage") is None


class TestOpenAIPayloadEffort:
    def _provider(self) -> _StubOpenAIProvider:
        return _StubOpenAIProvider(base_url="http://stub", logger=logging.getLogger("stub"))

    def test_payload_includes_native_effort(self):
        payload = self._provider()._build_payload(
            _stub_request(reasoning_effort="medium"), stream=False
        )
        assert payload["reasoning_effort"] == "medium"

    def test_payload_passes_through_minimal_as_a_native_effort(self):
        payload = self._provider()._build_payload(
            _stub_request(reasoning_effort="minimal"), stream=False
        )
        assert payload["reasoning_effort"] == "minimal"

    def test_xhigh_downgraded_with_warning(self, caplog):
        provider = self._provider()
        with caplog.at_level(logging.WARNING, logger="stub"):
            payload = provider._build_payload(_stub_request(reasoning_effort="xhigh"), stream=False)
        assert payload["reasoning_effort"] == "high"
        assert any("xhigh" in rec.message for rec in caplog.records)

    def test_payload_derives_effort_from_budget(self):
        payload = self._provider()._build_payload(
            _stub_request(thinking_budget=16384), stream=False
        )
        assert payload["reasoning_effort"] == "high"

    def test_payload_omits_effort_when_unset(self):
        payload = self._provider()._build_payload(_stub_request(), stream=False)
        assert "reasoning_effort" not in payload


class TestCopilotResponsesPayloadEffort:
    @staticmethod
    def _seed_catalog(provider: CopilotProvider, model_id: str, allowed: list[str]) -> None:
        """Prime the model cache so ``_catalog_effort_values`` returns ``allowed``."""
        from router_maestro.providers.base import ModelInfo

        provider._models_ttl_cache.set(
            [
                ModelInfo(
                    id=model_id,
                    name=model_id,
                    provider="github-copilot",
                    reasoning_effort_values=allowed,
                )
            ]
        )

    def test_responses_payload_writes_reasoning(self):
        provider = CopilotProvider()
        self._seed_catalog(provider, "gpt-5", ["low", "medium", "high"])
        req = ResponsesRequest(model="gpt-5", input="hi", reasoning_effort="high")
        payload = provider._build_responses_payload(req)
        assert payload["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_responses_payload_preserves_xhigh_when_catalog_advertises_it(self):
        # Regression: chat path is catalog-driven and preserves ``xhigh`` for
        # gpt-5.5 (catalog: ['none','low','medium','high','xhigh']). Responses
        # path used to unconditionally downgrade — codex+gpt-5.5 silently ran
        # at ``high`` while chat ran at ``xhigh``. Both paths must agree.
        provider = CopilotProvider()
        self._seed_catalog(provider, "gpt-5.5", ["none", "low", "medium", "high", "xhigh"])
        req = ResponsesRequest(model="gpt-5.5", input="hi", reasoning_effort="xhigh")
        payload = provider._build_responses_payload(req)
        assert payload["reasoning"] == {"effort": "xhigh", "summary": "auto"}

    def test_responses_payload_picks_closest_when_xhigh_unsupported(self):
        # gpt-5-mini's catalog tops out at ``high``. Asking for ``xhigh`` must
        # land on ``high`` (not silently fall back to nothing).
        provider = CopilotProvider()
        self._seed_catalog(provider, "gpt-5-mini", ["low", "medium", "high"])
        req = ResponsesRequest(model="gpt-5-mini", input="hi", reasoning_effort="xhigh")
        payload = provider._build_responses_payload(req)
        assert payload["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_responses_payload_clamps_to_catalog_for_clamped_models(self):
        # claude-opus-4.7's catalog only advertises ``medium``. Asking for
        # ``xhigh`` must clamp down to ``medium`` — same as the chat path.
        provider = CopilotProvider()
        self._seed_catalog(provider, "claude-opus-4.7", ["medium"])
        req = ResponsesRequest(model="claude-opus-4.7", input="hi", reasoning_effort="xhigh")
        payload = provider._build_responses_payload(req)
        assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}

    def test_responses_payload_rejects_when_catalog_says_no_reasoning(self):
        # Explicit reasoning must not be silently dropped.
        provider = CopilotProvider()
        self._seed_catalog(provider, "gpt-4o", [])
        req = ResponsesRequest(model="gpt-4o", input="hi", reasoning_effort="high")
        with pytest.raises(ProviderError) as caught:
            provider._build_responses_payload(req)
        assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert caught.value.parameter == "reasoning_effort"

    def test_responses_payload_preserves_known_supported_tier_when_catalog_cold(self):
        # Static capability knowledge keeps cold and warm behavior consistent.
        provider = CopilotProvider()
        # No _seed_catalog call → cache is cold.
        req = ResponsesRequest(model="gpt-5.5", input="hi", reasoning_effort="xhigh")
        payload = provider._build_responses_payload(req)
        assert payload["reasoning"] == {"effort": "xhigh", "summary": "auto"}

    def test_responses_payload_cold_and_warm_known_supported_tiers_match(self):
        cold = CopilotProvider()._build_responses_payload(
            ResponsesRequest(model="gpt-5.4", input="hi", reasoning_effort="xhigh")
        )
        warm_provider = CopilotProvider()
        self._seed_catalog(
            warm_provider,
            "gpt-5.4",
            ["low", "medium", "high", "xhigh"],
        )
        warm = warm_provider._build_responses_payload(
            ResponsesRequest(model="gpt-5.4", input="hi", reasoning_effort="xhigh")
        )

        assert (
            cold["reasoning"]
            == warm["reasoning"]
            == {
                "effort": "xhigh",
                "summary": "auto",
            }
        )

    def test_responses_payload_omits_when_unset(self):
        provider = CopilotProvider()
        self._seed_catalog(provider, "gpt-5", ["low", "medium", "high", "xhigh"])
        payload = provider._build_responses_payload(ResponsesRequest(model="gpt-5", input="hi"))
        assert "reasoning" not in payload


class TestChatRouteSchemaPassthrough:
    """The OpenAI Chat schema must accept reasoning_effort and thinking."""

    def test_schema_accepts_reasoning_effort(self):
        from router_maestro.server.schemas.openai import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )
        assert req.reasoning_effort == "high"

    def test_internal_request_accepts_minimal_reasoning_effort(self):
        request = _stub_request(reasoning_effort="minimal")

        assert request.reasoning_effort == "minimal"

    def test_internal_request_rejects_provider_catalog_none_sentinel(self):
        with pytest.raises(ValueError, match="reasoning_effort"):
            _stub_request(reasoning_effort="none")

    def test_omitted_chat_reasoning_effort_remains_unset(self):
        from router_maestro.server.schemas.openai import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert request.reasoning_effort is None
        assert "reasoning_effort" not in request.model_fields_set

    def test_schema_accepts_thinking_passthrough(self):
        from router_maestro.server.schemas.openai import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 12000},
        )
        assert req.thinking == {"type": "enabled", "budget_tokens": 12000}


class TestResponsesSchemaPassthrough:
    def test_schema_accepts_reasoning(self):
        from router_maestro.server.schemas.responses import (
            ResponsesRequest as ResponsesSchema,
        )

        req = ResponsesSchema(model="m", input="hi", reasoning={"effort": "xhigh"})
        assert req.reasoning == {"effort": "xhigh"}

    def test_schema_accepts_minimal_reasoning(self):
        from router_maestro.server.schemas.responses import (
            ResponsesReasoningConfig,
        )

        reasoning = ResponsesReasoningConfig.model_validate({"effort": "minimal"})

        assert reasoning.effort == "minimal"

    def test_schema_rejects_provider_catalog_none_sentinel(self):
        from pydantic import ValidationError

        from router_maestro.server.schemas.responses import ResponsesReasoningConfig

        with pytest.raises(ValidationError):
            ResponsesReasoningConfig.model_validate({"effort": "none"})

    def test_omitted_responses_reasoning_remains_unset(self):
        from router_maestro.server.schemas.responses import (
            ResponsesRequest as ResponsesSchema,
        )

        request = ResponsesSchema(model="m", input="hi")

        assert request.reasoning is None
        assert "reasoning" not in request.model_fields_set


def test_omitted_anthropic_output_config_remains_unset() -> None:
    from router_maestro.server.schemas.anthropic import AnthropicMessagesRequest

    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert request.output_config is None
    assert "output_config" not in request.model_fields_set
