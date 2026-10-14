"""Disabled efforts are accepted without requiring positive thinking capabilities."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    OpenAIChatRuntime,
    OpenAIResponsesRuntime,
    ReasoningConfig,
)
from router_maestro.providers.base import ChatRequest, ChatResponse, Message, ResponsesRequest
from router_maestro.routing.capabilities import RequestFeatures
from router_maestro.server.routes import chat


@pytest.mark.parametrize("effort", ["off", "none", "OFF", "NONE"])
def test_chat_http_accepts_disabled_effort_and_normalizes_without_enabling_thinking(
    monkeypatch, effort
):
    app = FastAPI()
    app.include_router(chat.router)
    model_router = AsyncMock()
    model_router.chat_completion.return_value = (ChatResponse(content="ok", model="m"), "stub")
    monkeypatch.setattr(chat, "get_router", lambda: model_router)
    response = TestClient(app).post(
        "/api/openai/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": effort,
        },
    )

    assert response.status_code == 200
    request = model_router.chat_completion.await_args.args[0]
    assert request.reasoning_effort == effort.lower()
    assert request.thinking_type == "disabled"
    assert request.thinking_budget is None
    assert RequestFeatures.for_chat(request).reasoning is False


@pytest.mark.parametrize("effort", ["off", "none", "OFF", "NONE"])
def test_internal_chat_and_responses_disabled_efforts_are_not_positive_features(effort):
    request = ChatRequest(
        model="m", messages=[Message(role="user", content="hi")], reasoning_effort=effort
    )
    assert request.reasoning_effort == effort.lower()
    features = RequestFeatures.for_chat(request)
    assert features.reasoning is False
    assert features.reasoning_parameter is None
    responses = RequestFeatures.for_responses(
        ResponsesRequest(model="m", input="hi", reasoning_effort=effort)
    )
    assert responses.reasoning is False
    assert responses.reasoning_parameter is None


@pytest.mark.parametrize("runtime", [OpenAIChatRuntime, OpenAIResponsesRuntime])
@pytest.mark.parametrize("effort", ["off", "none", "OFF", "NONE"])
@pytest.mark.asyncio
async def test_disabled_protocol_semantics_survive_cross_protocol_conversion(runtime, effort):
    source = runtime()
    payload = (
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": effort}
        if runtime is OpenAIChatRuntime
        else {"model": "m", "input": "hi", "reasoning": {"effort": effort}}
    )
    assert source.inspect_request(payload).reasoning is False
    semantic = await source.decode_request(payload)
    assert semantic.reasoning == ReasoningConfig(enabled=False)
    responses = await OpenAIResponsesRuntime().encode_request(semantic)
    assert responses["reasoning"]["effort"] == "none"
    chat_payload = await OpenAIChatRuntime().encode_request(semantic)
    assert chat_payload["reasoning_effort"] == "none"
    assert chat_payload["thinking"] == {"type": "disabled"}
    assert OpenAIChatRuntime().inspect_request(chat_payload).reasoning is False
    anthropic = await AnthropicMessagesRuntime().encode_request(
        replace(semantic, max_output_tokens=1024)
    )
    assert anthropic.get("output_config") is None
    assert anthropic.get("thinking", {}).get("type") in {None, "disabled"}


@pytest.mark.parametrize("effort", ["off", "none"])
def test_disabled_effort_wins_over_positive_budget_in_capability_inspection(effort):
    request = ChatRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
        reasoning_effort=effort,
        thinking_type="enabled",
        thinking_budget=4096,
    )
    assert RequestFeatures.for_chat(request).reasoning is False


def test_positive_thinking_budget_and_native_anthropic_are_unchanged():
    request = ChatRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
        thinking_type="enabled",
        thinking_budget=4096,
    )
    features = RequestFeatures.for_chat(request)
    assert features.reasoning is True
    assert features.reasoning_parameter == "thinking_budget"
    assert (
        RequestFeatures.for_anthropic_native(
            {"thinking": {"type": "enabled", "budget_tokens": 4096}}
        ).reasoning
        is True
    )
    assert (
        RequestFeatures.for_anthropic_native({"output_config": {"effort": "off"}}).reasoning
        is False
    )
