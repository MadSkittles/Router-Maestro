"""Focused tests for canonical Chat to downstream Anthropic reduction."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ProviderFailureKind,
    ResponseStatus,
    TerminalOutcome,
    TransportTermination,
    unexpected_eof_outcome,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.server.protocols.anthropic_reducer import (
    AnthropicReducer,
    AnthropicStreamProtocolError,
    build_anthropic_response,
)
from router_maestro.server.routes.anthropic import router as anthropic_router
from router_maestro.server.routes.anthropic import stream_response as anthropic_stream_response


def _tool_call(
    *,
    index: int | None = None,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> dict:
    tool: dict = {"function": {}}
    if index is not None:
        tool["index"] = index
    if tool_id is not None:
        tool["id"] = tool_id
    if name is not None:
        tool["function"]["name"] = name
    if arguments is not None:
        tool["function"]["arguments"] = arguments
    return tool


def _reconstruct_blocks(events: list[dict]) -> list[dict]:
    blocks: dict[int, dict] = {}
    for event in events:
        event_type = event["type"]
        if event_type == "content_block_start":
            blocks[event["index"]] = dict(event["content_block"])
            continue
        if event_type != "content_block_delta":
            continue
        block = blocks[event["index"]]
        delta = event["delta"]
        if delta["type"] == "thinking_delta":
            block["thinking"] += delta["thinking"]
        elif delta["type"] == "signature_delta":
            block["signature"] = block.get("signature", "") + delta["signature"]
        elif delta["type"] == "text_delta":
            block["text"] += delta["text"]
        elif delta["type"] == "input_json_delta":
            block["input"] = json.loads(delta["partial_json"])
    return [blocks[index] for index in sorted(blocks)]


def test_nonstream_builds_ordered_blocks_usage_and_terminal() -> None:
    downstream = build_anthropic_response(
        ChatResponse(
            content="answer",
            refusal="cannot help",
            model="upstream-model",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            thinking="consider",
            thinking_signature="opaque",
            tool_calls=[
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"router"}'},
                }
            ],
        ),
        response_id="msg_test",
        model="provider/upstream-model",
    )

    assert [block.model_dump(exclude_none=True) for block in downstream.content] == [
        {"type": "thinking", "thinking": "consider", "signature": "opaque"},
        {"type": "text", "text": "answer"},
        {"type": "text", "text": "cannot help"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "lookup",
            "input": {"q": "router"},
        },
    ]
    assert downstream.stop_reason == "tool_use"
    assert downstream.usage.model_dump(exclude_none=True) == {
        "input_tokens": 5,
        "output_tokens": 7,
    }


def test_stream_and_nonstream_share_content_and_terminal_rules() -> None:
    response = ChatResponse(
        content="answer",
        model="upstream-model",
        finish_reason="tool_calls",
        usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        thinking="consider",
        thinking_signature="opaque",
        tool_calls=[
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"router"}'},
            }
        ],
    )
    nonstream = build_anthropic_response(
        response,
        response_id="msg_test",
        model="provider/upstream-model",
    )
    reducer = AnthropicReducer(
        response_id="msg_test",
        model="provider/upstream-model",
        estimated_input_tokens=99,
    )

    events = reducer.start()
    for chunk in [
        ChatStreamChunk(content="", thinking="con"),
        ChatStreamChunk(content="", thinking="sider", thinking_signature="opa"),
        ChatStreamChunk(content="", thinking_signature="que"),
        ChatStreamChunk(content="answer"),
        ChatStreamChunk(
            content="",
            tool_calls=[
                _tool_call(
                    index=0,
                    tool_id="toolu_1",
                    name="lookup",
                    arguments='{"q":',
                )
            ],
        ),
        ChatStreamChunk(
            content="",
            tool_calls=[_tool_call(index=0, arguments='"router"}')],
        ),
        ChatStreamChunk(
            content="",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        ),
    ]:
        events.extend(reducer.reduce(chunk))

    assert _reconstruct_blocks(events) == [
        block.model_dump(exclude_none=True) for block in nonstream.content
    ]
    terminal = next(event for event in events if event["type"] == "message_delta")
    assert terminal["delta"]["stop_reason"] == nonstream.stop_reason == "tool_use"
    assert terminal["usage"]["input_tokens"] == nonstream.usage.input_tokens == 5
    assert terminal["usage"]["output_tokens"] == nonstream.usage.output_tokens == 7
    assert [event["type"] for event in events].count("message_stop") == 1


def test_tools_are_transactional_and_flush_in_explicit_index_order() -> None:
    reducer = AnthropicReducer(response_id="msg_test", model="provider/model")
    reducer.start()

    assert (
        reducer.reduce(
            ChatStreamChunk(
                content="",
                tool_calls=[_tool_call(index=7, tool_id="tool-a", name="alpha", arguments='{"a":')],
            )
        )
        == []
    )
    assert (
        reducer.reduce(
            ChatStreamChunk(
                content="",
                tool_calls=[
                    _tool_call(index=2, tool_id="tool-b", name="beta", arguments='{"b":2}')
                ],
            )
        )
        == []
    )
    assert (
        reducer.reduce(
            ChatStreamChunk(
                content="",
                tool_calls=[_tool_call(index=7, arguments="1}")],
            )
        )
        == []
    )

    events = reducer.reduce(ChatStreamChunk(content="", finish_reason="tool_calls"))

    assert [block["id"] for block in _reconstruct_blocks(events)] == ["tool-b", "tool-a"]


def test_malformed_tool_terminal_is_transactional() -> None:
    reducer = AnthropicReducer(response_id="msg_test", model="provider/model")
    reducer.start()
    assert (
        reducer.reduce(
            ChatStreamChunk(
                content="",
                tool_calls=[_tool_call(index=0, tool_id="tool-a", name="alpha", arguments="{")],
            )
        )
        == []
    )

    with pytest.raises(AnthropicStreamProtocolError, match="valid JSON object") as caught:
        reducer.reduce(ChatStreamChunk(content="", finish_reason="tool_calls"))

    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert reducer.state.message_complete is False
    assert reducer.state.content_block_open is False


@pytest.mark.parametrize(
    ("tool_call", "match"),
    [
        ({"id": "tool-a", "function": []}, "function must be an object"),
        (
            {"id": 7, "function": {"name": "alpha", "arguments": "{}"}},
            "id must be a string",
        ),
        (
            {"id": "tool-a", "function": {"arguments": "{}"}},
            "missing name",
        ),
        (
            {"id": "tool-a", "function": {"name": "alpha", "arguments": "{"}},
            "valid JSON object",
        ),
        (
            {"id": "tool-a", "function": {"name": "alpha", "arguments": "[]"}},
            "JSON object",
        ),
    ],
    ids=["function", "id", "name", "truncated-json", "non-object-json"],
)
def test_nonstream_complete_tool_validation_matches_stream_terminal(
    tool_call: dict,
    match: str,
) -> None:
    response = ChatResponse(
        content=None,
        model="upstream-model",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )

    with pytest.raises(AnthropicStreamProtocolError, match=match):
        build_anthropic_response(
            response,
            response_id="msg_test",
            model="provider/upstream-model",
        )


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": True},
        {"completion_tokens": "1"},
        {"prompt_tokens": -1},
    ],
)
def test_stream_usage_must_contain_nonnegative_integer_counts(usage: dict) -> None:
    reducer = AnthropicReducer(response_id="msg_test", model="provider/model")
    reducer.start()

    with pytest.raises(AnthropicStreamProtocolError, match="usage .* non-negative integer"):
        reducer.reduce(ChatStreamChunk(content="", usage=usage))


@pytest.mark.parametrize(
    "usage",
    [{"prompt_tokens": True}, {"completion_tokens": "1"}, {"prompt_tokens": -1}],
)
def test_nonstream_usage_uses_the_same_typed_validation(usage: dict) -> None:
    with pytest.raises(AnthropicStreamProtocolError, match="usage .* non-negative integer"):
        build_anthropic_response(
            ChatResponse(
                content="answer",
                model="upstream-model",
                finish_reason="stop",
                usage=usage,
            ),
            response_id="msg_test",
            model="provider/model",
        )


@pytest.mark.parametrize(
    "field",
    ["content", "refusal", "thinking", "thinking_signature"],
)
def test_nonstream_canonical_scalars_must_be_strings(field: str) -> None:
    response = ChatResponse(
        content="answer",
        model="upstream-model",
        finish_reason="stop",
    )
    setattr(response, field, 7)

    with pytest.raises(AnthropicStreamProtocolError, match=f"{field} must be a string or null"):
        build_anthropic_response(
            response,
            response_id="msg_test",
            model="provider/model",
        )


@pytest.mark.parametrize(
    "field",
    ["content", "refusal", "thinking", "thinking_signature"],
)
def test_stream_canonical_scalars_fail_before_reducer_state_mutation(field: str) -> None:
    reducer = AnthropicReducer(response_id="msg_test", model="provider/model")
    chunk = ChatStreamChunk(content="", finish_reason="stop")
    setattr(chunk, field, 7)

    with pytest.raises(AnthropicStreamProtocolError, match=f"{field} must be a string or null"):
        reducer.reduce(chunk)

    assert reducer.state.message_start_sent is False
    assert reducer.state.message_complete is False
    assert reducer.state.content_block_open is False


def test_nonstream_route_surfaces_malformed_canonical_scalar_as_typed_502() -> None:
    response = ChatResponse(
        content=7,  # type: ignore[arg-type]
        model="upstream-model",
        finish_reason="stop",
    )
    primary = SimpleNamespace(model=ModelRef("fake-provider", "upstream-model"))
    plan = SimpleNamespace(primary=primary, prevalidation_fallbacks=())

    class FakeRouter:
        async def plan_chat_completion(self, _request, *, stream):
            assert stream is False
            return plan

        def prepare_planned_chat_completion(self, route_plan, request, *, candidate_requests):
            assert route_plan is plan
            assert candidate_requests == {primary.model: request}
            return request

        async def chat_completion(self, request, *, prepared_plan):
            assert prepared_plan is request
            return response, "fake-provider"

    async def passthrough_budget(_router, request, _original_model, *, candidate):
        assert candidate is primary
        return request

    app = FastAPI()
    app.include_router(anthropic_router)
    with (
        patch("router_maestro.server.routes.anthropic.get_router", return_value=FakeRouter()),
        patch(
            "router_maestro.server.routes.anthropic._apply_thinking_budget",
            side_effect=passthrough_budget,
        ),
    ):
        route_response = TestClient(app).post(
            "/api/anthropic/v1/messages",
            json={
                "model": "fake-provider/upstream-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert route_response.status_code == 502
    assert route_response.json() == {"detail": "content must be a string or null"}


@pytest.mark.asyncio
async def test_stream_route_emits_one_error_and_no_success_for_malformed_scalar() -> None:
    async def chunks():
        yield ChatStreamChunk(content=7, finish_reason="stop")  # type: ignore[arg-type]

    class FakeRouter:
        async def chat_completion_stream(self, _request, **_kwargs):
            return chunks(), "fake-provider"

    frames = [
        frame
        async for frame in anthropic_stream_response(
            FakeRouter(),  # type: ignore[arg-type]
            ChatRequest(
                model="upstream-model",
                messages=[Message(role="user", content="hi")],
                stream=True,
            ),
            "upstream-model",
        )
    ]
    event_types = [
        line.removeprefix("event: ")
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("event: ")
    ]

    assert event_types.count("error") == 1
    assert "message_stop" not in event_types


@pytest.mark.parametrize(
    ("outcome", "expected_stop"),
    [
        (
            TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.COMPLETED,
                finish_reason="stop",
            ),
            "end_turn",
        ),
        (
            TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.INCOMPLETE,
                finish_reason="length",
                incomplete_details={"reason": "max_output_tokens"},
            ),
            "max_tokens",
        ),
    ],
)
def test_canonical_terminal_outcome_is_projected_directly(
    outcome: TerminalOutcome,
    expected_stop: str,
) -> None:
    reducer = AnthropicReducer(response_id="msg_test", model="provider/model")

    events = reducer.reduce(ChatStreamChunk(content="", terminal_outcome=outcome))

    terminal = next(event for event in events if event["type"] == "message_delta")
    assert terminal["delta"]["stop_reason"] == expected_stop
    assert events[-1] == {"type": "message_stop"}


def test_unexpected_eof_outcome_never_flushes_tools_or_success() -> None:
    reducer = AnthropicReducer(response_id="msg_test", model="provider/model")
    reducer.start()
    reducer.reduce(
        ChatStreamChunk(
            content="",
            tool_calls=[_tool_call(index=0, tool_id="tool-a", name="alpha", arguments="{")],
        )
    )

    with pytest.raises(AnthropicStreamProtocolError, match="non-success terminal"):
        reducer.reduce(ChatStreamChunk(content="", terminal_outcome=unexpected_eof_outcome()))

    assert reducer.state.message_complete is False
    assert reducer.state.content_block_open is False


def test_unknown_canonical_finish_reason_is_not_guessed() -> None:
    reducer = AnthropicReducer(response_id="msg_test", model="provider/model")

    with pytest.raises(AnthropicStreamProtocolError, match="non-success terminal"):
        reducer.reduce(ChatStreamChunk(content="", finish_reason="future_reason"))


def test_nonstream_route_surfaces_malformed_canonical_tool_as_typed_502() -> None:
    response = ChatResponse(
        content=None,
        model="upstream-model",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "tool-a",
                "type": "function",
                "function": {"name": "alpha", "arguments": "{"},
            }
        ],
    )
    primary = SimpleNamespace(model=ModelRef("fake-provider", "upstream-model"))
    plan = SimpleNamespace(primary=primary, prevalidation_fallbacks=())

    class FakeRouter:
        async def plan_chat_completion(self, _request, *, stream):
            assert stream is False
            return plan

        def prepare_planned_chat_completion(self, route_plan, request, *, candidate_requests):
            assert route_plan is plan
            assert candidate_requests == {primary.model: request}
            return request

        async def chat_completion(self, request, *, prepared_plan):
            assert prepared_plan is request
            return response, "fake-provider"

    async def passthrough_budget(_router, request, _original_model, *, candidate):
        assert candidate is primary
        return request

    app = FastAPI()
    app.include_router(anthropic_router)
    with (
        patch(
            "router_maestro.server.routes.anthropic.get_router",
            return_value=FakeRouter(),
        ),
        patch(
            "router_maestro.server.routes.anthropic._apply_thinking_budget",
            side_effect=passthrough_budget,
        ),
    ):
        route_response = TestClient(app).post(
            "/api/anthropic/v1/messages",
            json={
                "model": "fake-provider/upstream-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert route_response.status_code == 502
    assert route_response.json() == {"detail": "tool call arguments must be a valid JSON object"}


def test_no_events_after_message_complete() -> None:
    reducer = AnthropicReducer(response_id="msg_test", model="provider/model")
    reducer.start()
    reducer.reduce(ChatStreamChunk(content="done", finish_reason="stop"))

    assert reducer.reduce(ChatStreamChunk(content="late")) == []
