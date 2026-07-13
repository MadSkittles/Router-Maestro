"""Public stream responses identify the provider/model actually selected by Router."""

import json
from collections.abc import AsyncIterator

import pytest

from router_maestro.providers import (
    ChatRequest,
    ChatStreamChunk,
    Message,
    ResponsesRequest,
    ResponsesStreamChunk,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.server.routes.anthropic import stream_response as anthropic_stream_response
from router_maestro.server.routes.chat import stream_response as chat_stream_response
from router_maestro.server.routes.gemini import _stream_response as gemini_stream_response
from router_maestro.server.routes.responses import stream_response as responses_stream_response


class _SelectedStream:
    def __init__(self, chunks, selected_model: ModelRef) -> None:
        self.selected_model = selected_model
        self._iterator = self._iterate(chunks)

    @staticmethod
    async def _iterate(chunks):
        for chunk in chunks:
            yield chunk

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await anext(self._iterator)

    async def aclose(self) -> None:
        await self._iterator.aclose()


class _Router:
    def __init__(self, chunks, *, protocol: str) -> None:
        selected = ModelRef("second", "shared-model")
        self.stream = _SelectedStream(chunks, selected)
        self.protocol = protocol

    async def chat_completion_stream(
        self,
        _request,
        fallback: bool = True,
        *,
        prepared_plan=None,
    ) -> tuple[AsyncIterator[ChatStreamChunk], str]:
        assert self.protocol == "chat"
        return self.stream, "second"

    async def responses_completion_stream(
        self,
        _request,
        fallback: bool = True,
        *,
        prepared_plan=None,
    ) -> tuple[AsyncIterator[ResponsesStreamChunk], str]:
        assert self.protocol == "responses"
        return self.stream, "second"


def _json_data(frames: list[str]) -> list[dict]:
    events = []
    for frame in frames:
        for line in frame.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line.removeprefix("data: ")))
    return events


def _chat_request() -> ChatRequest:
    return ChatRequest(
        model="router-maestro",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )


@pytest.mark.asyncio
async def test_openai_chat_stream_uses_selected_model_identity() -> None:
    router = _Router(
        [
            ChatStreamChunk(content="ok"),
            ChatStreamChunk(content="", finish_reason="stop"),
        ],
        protocol="chat",
    )

    frames = [frame async for frame in chat_stream_response(router, _chat_request())]
    events = _json_data(frames)

    assert {event["model"] for event in events if "model" in event} == {"second/shared-model"}


@pytest.mark.asyncio
async def test_openai_responses_stream_uses_selected_model_identity() -> None:
    router = _Router(
        [
            ResponsesStreamChunk(content="ok"),
            ResponsesStreamChunk(content="", finish_reason="stop"),
        ],
        protocol="responses",
    )
    request = ResponsesRequest(model="router-maestro", input="hi", stream=True)

    frames = [
        frame
        async for frame in responses_stream_response(
            router,
            request,
            request_id="req-identity",
            start_time=0.0,
        )
    ]
    events = _json_data(frames)
    response_models = {
        event["response"]["model"] for event in events if isinstance(event.get("response"), dict)
    }

    assert response_models == {"second/shared-model"}


@pytest.mark.asyncio
async def test_anthropic_stream_uses_selected_model_identity() -> None:
    router = _Router(
        [
            ChatStreamChunk(content="ok"),
            ChatStreamChunk(content="", finish_reason="stop"),
        ],
        protocol="chat",
    )

    frames = [
        frame
        async for frame in anthropic_stream_response(
            router,
            _chat_request(),
            "router-maestro",
        )
    ]
    events = _json_data(frames)
    start = next(event for event in events if event.get("type") == "message_start")

    assert start["message"]["model"] == "second/shared-model"


@pytest.mark.asyncio
async def test_gemini_stream_uses_selected_model_identity() -> None:
    router = _Router(
        [
            ChatStreamChunk(content="ok"),
            ChatStreamChunk(content="", finish_reason="stop"),
        ],
        protocol="chat",
    )

    frames = [
        frame
        async for frame in gemini_stream_response(
            router,
            _chat_request(),
            "router-maestro",
        )
    ]
    events = _json_data(frames)

    assert {event["modelVersion"] for event in events if "modelVersion" in event} == {
        "second/shared-model"
    }
