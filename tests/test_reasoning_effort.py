"""Tests for reasoning_effort end-to-end passthrough."""

from __future__ import annotations

import logging

import pytest

from router_maestro.providers.base import ChatRequest, Message, ResponsesRequest
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.utils.reasoning import (
    EFFORT_TO_BUDGET,
    budget_to_effort,
    downgrade_for_upstream,
    effort_to_budget,
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
            ("HIGH", 8192),
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

    def test_downgrade_for_upstream(self):
        assert downgrade_for_upstream(None) is None
        assert downgrade_for_upstream("low") == "low"
        assert downgrade_for_upstream("xhigh") == "high"
        assert downgrade_for_upstream("garbage") is None


class TestOpenAIPayloadEffort:
    def _provider(self) -> _StubOpenAIProvider:
        return _StubOpenAIProvider(base_url="http://stub", logger=logging.getLogger("stub"))

    def test_payload_includes_native_effort(self):
        payload = self._provider()._build_payload(
            _stub_request(reasoning_effort="medium"), stream=False
        )
        assert payload["reasoning_effort"] == "medium"

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
    def test_responses_payload_writes_reasoning(self):
        provider = CopilotProvider()
        req = ResponsesRequest(model="gpt-5", input="hi", reasoning_effort="high")
        payload = provider._build_responses_payload(req)
        assert payload["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_responses_payload_downgrades_xhigh(self):
        provider = CopilotProvider()
        req = ResponsesRequest(model="gpt-5", input="hi", reasoning_effort="xhigh")
        payload = provider._build_responses_payload(req)
        assert payload["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_responses_payload_omits_when_unset(self):
        provider = CopilotProvider()
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
