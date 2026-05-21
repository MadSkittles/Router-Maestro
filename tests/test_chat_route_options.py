"""Tests for OpenAI chat route option passthrough."""

import pytest

from router_maestro.providers import ChatResponse
from router_maestro.server.routes.chat import chat_completions
from router_maestro.server.schemas import ChatCompletionRequest, ChatMessage


class _CapturingRouter:
    def __init__(self):
        self.request = None

    async def rewrite_to_reasoning_variant(self, request):
        return request

    async def chat_completion(self, request):
        self.request = request
        return ChatResponse(content="ok", model=request.model), "test-provider"


@pytest.mark.anyio
async def test_openai_chat_route_preserves_supported_extra_options(monkeypatch):
    """Accepted OpenAI chat fields should reach provider-facing ChatRequest.extra."""
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
    )

    response = await chat_completions(request)

    assert response.choices[0].message.content == "ok"
    assert router.request is not None
    assert router.request.extra == {
        "top_p": 0.25,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "stop": ["END"],
        "user": "user-123",
    }
