from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest

from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    ProtocolRuntime,
    RequestManifest,
    SemanticEvent,
    SemanticEventType,
    SemanticRequest,
    SemanticResponse,
    TerminalMetadata,
    TextContent,
    Usage,
    UsageMode,
    WireProtocol,
)
from router_maestro.protocols.openai_chat import OpenAIChatRuntime
from router_maestro.protocols.openai_responses import OpenAIResponsesRuntime
from router_maestro.providers.base import (
    BaseProvider,
    ChatResponse,
    ChatStreamChunk,
    ProviderError,
    ProviderFailureKind,
    ResponsesResponse,
    ResponsesStreamChunk,
)
from router_maestro.providers.bindings import legacy_endpoint_binding
from router_maestro.routing.capabilities import Operation
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.transport_plan import FlowCandidate, TransportPlan
from router_maestro.server.dispatcher import (
    DispatchResult,
    DispatchSelection,
    OpenedDispatchStream,
)
from router_maestro.server.response_bridge import GenerationResponseBridge


class _Runtime:
    def __init__(
        self,
        protocol: WireProtocol,
        *,
        decoded_response: SemanticResponse | None = None,
    ) -> None:
        self.protocol = protocol
        self.decoded_response = decoded_response
        self.decode_calls: list[Mapping[str, Any]] = []
        self.encoded_responses: list[SemanticResponse] = []

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        del payload
        raise AssertionError("inspect_request was not expected")

    async def decode_request(self, payload: Mapping[str, Any]) -> SemanticRequest:
        del payload
        raise AssertionError("decode_request was not expected")

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        del request
        raise AssertionError("encode_request was not expected")

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        self.decode_calls.append(payload)
        if self.decoded_response is None:
            raise AssertionError("decode_response was not expected")
        return self.decoded_response

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        self.encoded_responses.append(response)
        return {
            "id": response.id,
            "model": response.model,
            "output_count": len(response.output),
        }

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        del payload
        raise AssertionError("decode_stream_event was not expected")

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        del event
        raise AssertionError("encode_stream_event was not expected")


class _TrackedIterator:
    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.index = 0
        self.closed = False

    def __aiter__(self) -> _TrackedIterator:
        return self

    async def __anext__(self) -> object:
        if self.closed or self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


class _StreamDecoder:
    def __init__(self, identifier: int) -> None:
        self.identifier = identifier
        self.started = False

    def decode(self, payload: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        events = []
        if not self.started:
            self.started = True
            events.append(
                SemanticEvent(
                    type=SemanticEventType.RESPONSE_STARTED,
                    response_id=f"upstream-{self.identifier}",
                    metadata={"model": "upstream-model"},
                )
            )
        text = payload.get("text")
        if isinstance(text, str):
            events.append(SemanticEvent(type=SemanticEventType.TEXT_DELTA, delta=text))
        if payload.get("terminal") is True:
            events.append(
                SemanticEvent(
                    type=SemanticEventType.TERMINAL,
                    terminal=TerminalMetadata(response_status="completed"),
                )
            )
        return tuple(events)


class _StreamEncoder:
    def __init__(self, identifier: int) -> None:
        self.identifier = identifier
        self.events: list[SemanticEvent] = []

    def encode(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        self.events.append(event)
        return (
            {
                "encoder": self.identifier,
                "type": event.type.value,
                "model": event.metadata.get("model"),
                "response_id": event.response_id,
                "delta": event.delta,
            },
        )


class _StreamFactories:
    def __init__(self) -> None:
        self.decoders: list[_StreamDecoder] = []
        self.encoders: list[_StreamEncoder] = []

    def decoder(self, runtime: ProtocolRuntime, plan: TransportPlan) -> _StreamDecoder:
        del runtime, plan
        decoder = _StreamDecoder(len(self.decoders))
        self.decoders.append(decoder)
        return decoder

    def encoder(
        self,
        runtime: ProtocolRuntime,
        plan: TransportPlan,
        response_id: str,
        public_model: str,
    ) -> _StreamEncoder:
        del runtime, plan, response_id, public_model
        encoder = _StreamEncoder(len(self.encoders))
        self.encoders.append(encoder)
        return encoder


def _plan(
    source: WireProtocol,
    target: WireProtocol,
    *,
    binding_id: str = "binding",
) -> TransportPlan:
    operations = {
        WireProtocol.ANTHROPIC_MESSAGES: frozenset({Operation.NATIVE_ANTHROPIC}),
        WireProtocol.OPENAI_CHAT: frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
        WireProtocol.OPENAI_RESPONSES: frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM}),
        WireProtocol.GEMINI: frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
    }[target]
    binding = legacy_endpoint_binding(
        binding_id=binding_id,
        protocol=target,
        operations=operations,
    )
    return TransportPlan(
        model=ModelRef("provider", "upstream-model"),
        provider=cast(BaseProvider, SimpleNamespace(name="provider")),
        candidate=FlowCandidate.for_binding(source_protocol=source, binding=binding),
    )


def _result(plan: TransportPlan, value: object) -> DispatchResult:
    return DispatchResult(value=value, selection=DispatchSelection(plan))


def _opened(plan: TransportPlan, source: _TrackedIterator) -> OpenedDispatchStream:
    return OpenedDispatchStream(frames=source, selection=DispatchSelection(plan))


async def _collect(stream) -> list[Mapping[str, Any]]:
    return [item async for item in stream]


def _fail_decoder_factory(
    runtime: ProtocolRuntime,
    plan: TransportPlan,
) -> NoReturn:
    del runtime, plan
    raise AssertionError("semantic stream decoder was not expected")


def _fail_encoder_factory(
    runtime: ProtocolRuntime,
    plan: TransportPlan,
    response_id: str,
    public_model: str,
) -> NoReturn:
    del runtime, plan, response_id, public_model
    raise AssertionError("semantic stream encoder was not expected")


@pytest.mark.asyncio
async def test_identity_mapping_is_copy_on_write_and_never_enters_semantic_ir() -> None:
    plan = _plan(WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_CHAT)
    runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    raw = {
        "id": "chatcmpl-1",
        "model": "upstream-model",
        "choices": [],
        "future_extension": {"preserve": True},
    }
    bridge = GenerationResponseBridge({WireProtocol.OPENAI_CHAT: runtime})

    encoded = await bridge.encode_result(_result(plan, raw), runtime)

    assert encoded == {
        **raw,
        "model": "provider/upstream-model",
        "object": "chat.completion",
    }
    assert encoded is not raw
    assert raw["model"] == "upstream-model"
    assert "object" not in raw
    assert runtime.decode_calls == []
    assert runtime.encoded_responses == []


@pytest.mark.asyncio
async def test_chat_identity_stream_projects_chunk_object_without_semantic_ir() -> None:
    plan = _plan(WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_CHAT)
    runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    raw = {
        "model": "upstream-model",
        "choices": [{"index": 0, "finish_reason": "stop"}],
        "future_extension": {"preserve": True},
    }
    source = _TrackedIterator(raw)

    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_CHAT: runtime},
        stream_decoder_factory=_fail_decoder_factory,
        stream_encoder_factory=_fail_encoder_factory,
    )

    frames = await _collect(bridge.encode_stream(_opened(plan, source), runtime))

    assert frames == [
        {
            **raw,
            "model": "provider/upstream-model",
            "object": "chat.completion.chunk",
        }
    ]
    assert "object" not in raw
    assert runtime.decode_calls == []
    assert runtime.encoded_responses == []


@pytest.mark.asyncio
async def test_cross_mapping_uses_target_decoder_and_ingress_encoder() -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    target = _Runtime(
        WireProtocol.OPENAI_RESPONSES,
        decoded_response=SemanticResponse(
            id="resp-upstream",
            model="upstream-model",
            output=(TextContent("hello"),),
        ),
    )
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    bridge = GenerationResponseBridge({WireProtocol.OPENAI_RESPONSES: target})
    raw = {"id": "resp-upstream", "model": "upstream-model", "output": []}

    encoded = await bridge.encode_result(_result(plan, raw), ingress)

    assert target.decode_calls == [raw]
    assert ingress.encoded_responses[0].model == "provider/upstream-model"
    assert ingress.encoded_responses[0].id == "resp-upstream"
    assert encoded["model"] == "provider/upstream-model"


@pytest.mark.asyncio
async def test_predecoded_cross_response_is_reused_without_target_decode() -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    target = _Runtime(WireProtocol.OPENAI_RESPONSES)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    semantic = SemanticResponse(
        id="resp-upstream",
        model="upstream-model",
        output=(TextContent("hello"),),
    )

    encoded = await GenerationResponseBridge({WireProtocol.OPENAI_RESPONSES: target}).encode_result(
        _result(plan, semantic), ingress
    )

    assert target.decode_calls == []
    assert ingress.encoded_responses[0].model == "provider/upstream-model"
    assert encoded["model"] == "provider/upstream-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        ChatResponse(content="chat", model="wrong-model"),
        ResponsesResponse(content="responses", model="wrong-model"),
    ],
    ids=["legacy-chat", "legacy-responses"],
)
async def test_legacy_response_dtos_use_helpers_and_selected_public_model(value) -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    bridge = GenerationResponseBridge({}, response_id_factory=lambda protocol: "msg-rm")

    encoded = await bridge.encode_result(_result(plan, value), ingress)

    semantic = ingress.encoded_responses[0]
    assert semantic.id == "msg-rm"
    assert semantic.model == "provider/upstream-model"
    assert encoded["model"] == "provider/upstream-model"


@pytest.mark.asyncio
async def test_cross_streams_create_isolated_decoder_and_encoder_state() -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    target = _Runtime(WireProtocol.OPENAI_RESPONSES)
    factories = _StreamFactories()
    response_ids = iter(("msg-one", "msg-two"))
    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_RESPONSES: target},
        stream_decoder_factory=factories.decoder,
        stream_encoder_factory=factories.encoder,
        response_id_factory=lambda protocol: next(response_ids),
    )
    source_one = _TrackedIterator({"text": "one"}, {"terminal": True})
    source_two = _TrackedIterator({"text": "two"}, {"terminal": True})
    stream_one = bridge.encode_stream(_opened(plan, source_one), ingress).__aiter__()
    stream_two = bridge.encode_stream(_opened(plan, source_two), ingress).__aiter__()

    first_one = await anext(stream_one)
    first_two = await anext(stream_two)
    rest_one = [item async for item in stream_one]
    rest_two = [item async for item in stream_two]

    assert len(factories.decoders) == 2
    assert len(factories.encoders) == 2
    assert {item["encoder"] for item in [first_one, *rest_one]} == {0}
    assert {item["encoder"] for item in [first_two, *rest_two]} == {1}
    assert {item["model"] for item in [first_one, *rest_one, first_two, *rest_two]} == {
        "provider/upstream-model"
    }
    assert source_one.closed is True
    assert source_two.closed is True


@pytest.mark.asyncio
async def test_cross_stream_reuses_primed_decoder_without_double_decode() -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    ingress = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    factories = _StreamFactories()
    calls: list[Mapping[str, Any]] = []

    class _PrimedDecoder:
        def __init__(self) -> None:
            self.started = False

        def decode(self, payload: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
            calls.append(payload)
            phase = payload.get("phase")
            if phase == "first":
                if self.started:
                    raise AssertionError("first frame was decoded twice")
                self.started = True
                return (
                    SemanticEvent(
                        type=SemanticEventType.RESPONSE_STARTED,
                        response_id="resp-upstream",
                        metadata={"model": "upstream-model"},
                    ),
                    SemanticEvent(type=SemanticEventType.TEXT_DELTA, delta="one"),
                )
            if phase != "second" or not self.started:
                raise AssertionError("second frame did not continue the primed decoder state")
            return (
                SemanticEvent(type=SemanticEventType.TEXT_DELTA, delta="two"),
                SemanticEvent(
                    type=SemanticEventType.TERMINAL,
                    terminal=TerminalMetadata(response_status="completed"),
                ),
            )

    first = {"phase": "first"}
    second = {"phase": "second"}
    decoder = _PrimedDecoder()
    first_events = decoder.decode(first)
    source = _TrackedIterator(first, second)

    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES)},
        stream_decoder_factory=_fail_decoder_factory,
        stream_encoder_factory=factories.encoder,
    )
    opened = OpenedDispatchStream(
        frames=source,
        selection=DispatchSelection(plan),
        semantic_decoder=decoder,
        first_events=first_events,
    )

    frames = await _collect(bridge.encode_stream(opened, ingress))

    assert calls == [first, second]
    assert [frame["type"] for frame in frames] == [
        "response_started",
        "text_delta",
        "text_delta",
        "terminal",
    ]
    assert len(factories.decoders) == 0
    assert len(factories.encoders) == 1
    assert source.closed is True


@pytest.mark.asyncio
async def test_identity_stream_qualifies_nested_model_without_semantic_state() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_RESPONSES)
    runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    created = {
        "type": "response.created",
        "response": {"id": "resp-1", "model": "upstream-model"},
    }
    completed = {
        "type": "response.completed",
        "response": {"id": "resp-1", "model": "upstream-model"},
    }
    source = _TrackedIterator(created, completed)

    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_RESPONSES: runtime},
        stream_decoder_factory=_fail_decoder_factory,
        stream_encoder_factory=_fail_encoder_factory,
    )

    frames = await _collect(bridge.encode_stream(_opened(plan, source), runtime))

    assert [frame["response"]["model"] for frame in frames] == [
        "provider/upstream-model",
        "provider/upstream-model",
    ]
    assert created["response"]["model"] == "upstream-model"
    assert completed["response"]["model"] == "upstream-model"
    assert runtime.decode_calls == []
    assert runtime.encoded_responses == []
    assert source.closed is True


@pytest.mark.asyncio
async def test_responses_identity_done_is_terminal_without_synthetic_eof() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_RESPONSES)
    runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    done = {
        "type": "response.done",
        "response": {
            "id": "resp-1",
            "model": "upstream-model",
            "status": "completed",
            "output": [],
        },
    }
    source = _TrackedIterator(done)

    frames = await _collect(
        GenerationResponseBridge({WireProtocol.OPENAI_RESPONSES: runtime}).encode_stream(
            _opened(plan, source),
            runtime,
        )
    )

    assert frames == [
        {
            **done,
            "response": {**done["response"], "model": "provider/upstream-model"},
        }
    ]
    assert source.closed is True


@pytest.mark.asyncio
async def test_gemini_identity_safety_feedback_is_terminal_without_synthetic_eof() -> None:
    plan = _plan(WireProtocol.GEMINI, WireProtocol.GEMINI)
    runtime = _Runtime(WireProtocol.GEMINI)
    blocked = {
        "modelVersion": "upstream-model",
        "promptFeedback": {"blockReason": "SAFETY"},
    }
    source = _TrackedIterator(blocked)

    frames = await _collect(
        GenerationResponseBridge({WireProtocol.GEMINI: runtime}).encode_stream(
            _opened(plan, source),
            runtime,
        )
    )

    assert frames == [{**blocked, "modelVersion": "provider/upstream-model"}]
    assert source.closed is True


@pytest.mark.asyncio
async def test_clean_eof_without_terminal_emits_unexpected_eof_and_closes_source() -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    factories = _StreamFactories()
    source = _TrackedIterator({"text": "partial"})
    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES)},
        stream_decoder_factory=factories.decoder,
        stream_encoder_factory=factories.encoder,
    )

    frames = await _collect(
        bridge.encode_stream(
            _opened(plan, source),
            _Runtime(WireProtocol.ANTHROPIC_MESSAGES),
        )
    )

    assert [frame["type"] for frame in frames[-2:]] == ["error", "terminal"]
    assert source.closed is True


@pytest.mark.asyncio
async def test_identity_clean_eof_emits_protocol_native_error_without_semantic_state() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_RESPONSES)
    runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    source = _TrackedIterator(
        {
            "type": "response.created",
            "response": {"id": "resp-1", "model": "upstream-model"},
        }
    )

    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_RESPONSES: runtime},
        stream_decoder_factory=_fail_decoder_factory,
        stream_encoder_factory=_fail_encoder_factory,
    )

    frames = await _collect(bridge.encode_stream(_opened(plan, source), runtime))

    assert frames[-1]["type"] == "response.incomplete"
    assert frames[-1]["sequence_number"] == 0
    assert frames[-1]["response"] == {
        "id": "resp-1",
        "object": "response",
        "created_at": frames[-1]["response"]["created_at"],
        "model": "provider/upstream-model",
        "status": "incomplete",
        "output": [],
        "usage": None,
        "incomplete_details": {"reason": "unexpected_eof"},
        "error": None,
    }
    assert source.closed is True


@pytest.mark.parametrize("protocol", list(WireProtocol))
@pytest.mark.asyncio
async def test_empty_identity_stream_uses_native_unexpected_eof_without_semantic_state(
    protocol: WireProtocol,
) -> None:
    plan = _plan(protocol, protocol)
    runtime = _Runtime(protocol)
    source = _TrackedIterator()

    bridge = GenerationResponseBridge(
        {protocol: runtime},
        stream_decoder_factory=_fail_decoder_factory,
        stream_encoder_factory=_fail_encoder_factory,
    )

    frames = await _collect(bridge.encode_stream(_opened(plan, source), runtime))

    assert len(frames) == 1
    frame = frames[0]
    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        assert frame["type"] == "error"
        assert frame["error"]["type"] == "api_error"
        assert frame["error"]["code"] == "unexpected_eof"
    elif protocol is WireProtocol.OPENAI_RESPONSES:
        assert frame["type"] == "response.incomplete"
        assert frame["sequence_number"] == 0
        assert frame["response"]["id"].startswith("resp_")
        assert frame["response"]["object"] == "response"
        assert frame["response"]["model"] == "provider/upstream-model"
        assert frame["response"]["output"] == []
        assert frame["response"]["incomplete_details"] == {"reason": "unexpected_eof"}
    elif protocol is WireProtocol.OPENAI_CHAT:
        assert frame["error"]["type"] == "unexpected_eof"
        assert frame["error"]["code"] == "unexpected_eof"
    else:
        assert frame["error"]["code"] == 502
        assert frame["error"]["status"] == "INTERNAL"
        assert frame["error"]["details"] == [{"reason": "unexpected_eof"}]
    assert source.closed is True


@pytest.mark.asyncio
async def test_chat_identity_stream_allows_one_standard_usage_chunk_after_terminal() -> None:
    plan = _plan(WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_CHAT)
    runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    terminal = {
        "model": "upstream-model",
        "choices": [{"index": 0, "finish_reason": "stop"}],
    }
    usage = {
        "model": "upstream-model",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    source = _TrackedIterator(terminal, usage)

    frames = await _collect(
        GenerationResponseBridge({WireProtocol.OPENAI_CHAT: runtime}).encode_stream(
            _opened(plan, source),
            runtime,
        )
    )

    assert len(frames) == 2
    assert frames[1]["choices"] == []
    assert frames[1]["usage"]["total_tokens"] == 3
    assert source.closed is True


@pytest.mark.asyncio
async def test_chat_identity_stream_rejects_usage_tail_when_terminal_has_usage() -> None:
    plan = _plan(WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_CHAT)
    runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    source = _TrackedIterator(
        {
            "choices": [{"index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
        {"choices": [], "usage": {"total_tokens": 999}},
    )

    with pytest.raises(ProviderError, match="after its terminal"):
        await _collect(
            GenerationResponseBridge({WireProtocol.OPENAI_CHAT: runtime}).encode_stream(
                _opened(plan, source),
                runtime,
            )
        )

    assert source.closed is True


@pytest.mark.asyncio
async def test_identity_stream_rejects_non_usage_frame_after_terminal() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_RESPONSES)
    runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    source = _TrackedIterator(
        {"type": "response.completed", "response": {"model": "upstream-model"}},
        {"type": "response.output_text.delta", "delta": "late"},
    )

    with pytest.raises(ProviderError, match="after its terminal"):
        await _collect(
            GenerationResponseBridge({WireProtocol.OPENAI_RESPONSES: runtime}).encode_stream(
                _opened(plan, source),
                runtime,
            )
        )

    assert source.closed is True


@pytest.mark.asyncio
async def test_chat_identity_stream_rejects_malformed_delta_before_eof() -> None:
    plan = _plan(WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_CHAT)
    source = _TrackedIterator(
        {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
        {"choices": [{"delta": "malformed", "finish_reason": None}]},
    )
    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_CHAT: _Runtime(WireProtocol.OPENAI_CHAT)}
    )
    opened = OpenedDispatchStream(source, DispatchSelection(plan))

    stream = bridge.encode_stream(opened, _Runtime(WireProtocol.OPENAI_CHAT))
    assert await anext(stream) == {
        "choices": [{"delta": {"content": "partial"}, "finish_reason": None}],
        "object": "chat.completion.chunk",
    }
    with pytest.raises(ProviderError) as caught:
        await anext(stream)

    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert "malformed upstream response" in str(caught.value)
    assert source.closed is True


@pytest.mark.asyncio
async def test_chat_identity_stream_rejects_second_post_terminal_usage_chunk() -> None:
    plan = _plan(WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_CHAT)
    runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    usage = {"choices": [], "usage": {"total_tokens": 3}}
    source = _TrackedIterator(
        {"choices": [{"finish_reason": "stop"}]},
        usage,
        usage,
    )

    with pytest.raises(ProviderError, match="after its terminal"):
        await _collect(
            GenerationResponseBridge({WireProtocol.OPENAI_CHAT: runtime}).encode_stream(
                _opened(plan, source),
                runtime,
            )
        )

    assert source.closed is True


@pytest.mark.asyncio
async def test_duplicate_terminal_is_rejected_and_source_is_closed() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_RESPONSES)
    runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    source = _TrackedIterator(
        {"type": "response.completed", "response": {"model": "upstream-model"}},
        {"type": "response.failed", "response": {"model": "upstream-model"}},
    )
    bridge = GenerationResponseBridge({WireProtocol.OPENAI_RESPONSES: runtime})

    with pytest.raises(ProviderError) as caught:
        await _collect(bridge.encode_stream(_opened(plan, source), runtime))

    assert "multiple terminal" in caught.value.safe_message
    assert source.closed is True


@pytest.mark.asyncio
async def test_postcommit_source_error_is_not_replayed_and_closes_source() -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    factories = _StreamFactories()
    source_error = RuntimeError("disconnect")
    source = _TrackedIterator({"text": "committed"}, source_error)
    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_RESPONSES: _Runtime(WireProtocol.OPENAI_RESPONSES)},
        stream_decoder_factory=factories.decoder,
        stream_encoder_factory=factories.encoder,
    )
    delivered = []

    with pytest.raises(RuntimeError, match="disconnect"):
        async for frame in bridge.encode_stream(
            _opened(plan, source),
            _Runtime(WireProtocol.ANTHROPIC_MESSAGES),
        ):
            delivered.append(frame)

    assert delivered
    assert len(factories.decoders) == 1
    assert len(factories.encoders) == 1
    assert source.closed is True


@pytest.mark.asyncio
async def test_legacy_responses_stream_uses_selected_model_and_one_terminal() -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    factories = _StreamFactories()
    source = _TrackedIterator(
        ResponsesStreamChunk(content="hello"),
        ResponsesStreamChunk(content="", finish_reason="stop"),
    )
    bridge = GenerationResponseBridge(
        {},
        stream_decoder_factory=factories.decoder,
        stream_encoder_factory=factories.encoder,
    )

    frames = await _collect(
        bridge.encode_stream(
            _opened(plan, source),
            _Runtime(WireProtocol.ANTHROPIC_MESSAGES),
        )
    )

    assert [frame["type"] for frame in frames] == ["text_delta", "terminal"]
    assert {frame["model"] for frame in frames} == {"provider/upstream-model"}
    assert source.closed is True


@pytest.mark.asyncio
async def test_real_responses_to_anthropic_stream_rewrites_identity_and_terminates() -> None:
    plan = _plan(WireProtocol.ANTHROPIC_MESSAGES, WireProtocol.OPENAI_RESPONSES)
    target = OpenAIResponsesRuntime(
        provider_name="provider",
        binding_id="binding",
    )
    ingress = AnthropicMessagesRuntime()
    response = {
        "id": "resp-upstream",
        "object": "response",
        "created_at": 1,
        "model": "upstream-model",
        "status": "in_progress",
        "output": [],
        "usage": None,
        "error": None,
        "incomplete_details": None,
    }
    completed = {
        **response,
        "status": "completed",
        "usage": {
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    }
    source = _TrackedIterator(
        {"type": "response.created", "response": response},
        {
            "type": "response.output_text.delta",
            "item_id": "msg-upstream",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
        },
        {"type": "response.completed", "response": completed},
    )
    bridge = GenerationResponseBridge(
        {WireProtocol.OPENAI_RESPONSES: target},
        response_id_factory=lambda protocol: "msg-downstream",
    )

    frames = await _collect(bridge.encode_stream(_opened(plan, source), ingress))

    assert frames[0]["type"] == "message_start"
    assert frames[0]["message"]["id"] == "msg-downstream"
    assert frames[0]["message"]["model"] == "provider/upstream-model"
    assert any(
        frame.get("type") == "content_block_delta" and frame.get("delta", {}).get("text") == "hello"
        for frame in frames
    )
    assert frames[-2]["usage"] == {"input_tokens": 2, "output_tokens": 1}
    assert frames[-1] == {"type": "message_stop"}
    assert source.closed is True


def _anthropic_partial_usage_source() -> _TrackedIterator:
    return _TrackedIterator(
        {
            "type": "message_start",
            "message": {
                "id": "msg-upstream",
                "type": "message",
                "role": "assistant",
                "model": "upstream-model",
                "content": [],
                "usage": {"input_tokens": 5},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    )


@pytest.mark.asyncio
async def test_anthropic_partial_usage_snapshots_merge_for_responses_stream() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.ANTHROPIC_MESSAGES)
    target = AnthropicMessagesRuntime()
    ingress = OpenAIResponsesRuntime()
    source = _anthropic_partial_usage_source()
    bridge = GenerationResponseBridge(
        {WireProtocol.ANTHROPIC_MESSAGES: target},
        response_id_factory=lambda protocol: "resp-downstream",
    )

    frames = await _collect(bridge.encode_stream(_opened(plan, source), ingress))

    assert frames[-1]["type"] == "response.completed"
    assert frames[-1]["response"]["usage"] == {
        "input_tokens": 5,
        "output_tokens": 7,
    }
    assert sum(frame["type"] == "response.completed" for frame in frames) == 1
    assert source.closed is True


@pytest.mark.asyncio
async def test_anthropic_partial_usage_snapshots_merge_for_chat_stream() -> None:
    plan = _plan(WireProtocol.OPENAI_CHAT, WireProtocol.ANTHROPIC_MESSAGES)
    target = AnthropicMessagesRuntime()
    ingress = OpenAIChatRuntime()
    source = _anthropic_partial_usage_source()
    bridge = GenerationResponseBridge(
        {WireProtocol.ANTHROPIC_MESSAGES: target},
        response_id_factory=lambda protocol: "chatcmpl-downstream",
    )

    frames = await _collect(bridge.encode_stream(_opened(plan, source), ingress))

    usage_frames = [frame for frame in frames if frame.get("usage") is not None]
    assert usage_frames[0]["usage"] == {"prompt_tokens": 5}
    assert usage_frames[1]["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
    }
    terminal_frames = [
        frame
        for frame in frames
        if any(choice.get("finish_reason") is not None for choice in frame.get("choices", []))
    ]
    assert len(terminal_frames) == 1
    assert source.closed is True


@pytest.mark.asyncio
async def test_stream_usage_merge_honors_delta_snapshot_missing_and_zero() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.ANTHROPIC_MESSAGES)
    ingress = _Runtime(WireProtocol.OPENAI_RESPONSES)
    factories = _StreamFactories()

    class _UsageDecoder:
        def decode(self, payload: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
            event_type = payload["event"]
            if event_type == "usage":
                return (SemanticEvent(type=SemanticEventType.USAGE, usage=payload["usage"]),)
            return (
                SemanticEvent(
                    type=SemanticEventType.TERMINAL,
                    terminal=TerminalMetadata(response_status="completed"),
                ),
            )

    source = _TrackedIterator(
        {
            "event": "usage",
            "usage": Usage(
                input_tokens=5,
                output_tokens=4,
                total_tokens=9,
                cached_input_tokens=3,
            ),
        },
        {
            "event": "usage",
            "usage": Usage(input_tokens=0, reasoning_tokens=0),
        },
        {
            "event": "usage",
            "usage": Usage(
                mode=UsageMode.DELTA,
                input_tokens=2,
                output_tokens=1,
                total_tokens=3,
                cached_input_tokens=0,
                reasoning_tokens=2,
            ),
        },
        {"event": "terminal"},
    )
    bridge = GenerationResponseBridge(
        {WireProtocol.ANTHROPIC_MESSAGES: _Runtime(WireProtocol.ANTHROPIC_MESSAGES)},
        stream_decoder_factory=lambda runtime, plan: _UsageDecoder(),
        stream_encoder_factory=factories.encoder,
    )

    frames = await _collect(bridge.encode_stream(_opened(plan, source), ingress))

    usages = [
        event.usage
        for event in factories.encoders[0].events
        if event.type is SemanticEventType.USAGE
    ]
    assert usages == [
        Usage(
            input_tokens=5,
            output_tokens=4,
            total_tokens=9,
            cached_input_tokens=3,
        ),
        Usage(
            input_tokens=0,
            output_tokens=4,
            total_tokens=9,
            cached_input_tokens=3,
            reasoning_tokens=0,
        ),
        Usage(
            input_tokens=2,
            output_tokens=5,
            total_tokens=12,
            cached_input_tokens=3,
            reasoning_tokens=2,
        ),
    ]
    assert [frame["type"] for frame in frames] == ["usage", "usage", "usage", "terminal"]
    assert source.closed is True


@pytest.mark.asyncio
async def test_cross_stream_accepts_usage_only_chunk_after_legacy_chat_terminal() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_CHAT)
    ingress = OpenAIResponsesRuntime()
    source = _TrackedIterator(
        ChatStreamChunk(content="hello"),
        ChatStreamChunk(content="", finish_reason="stop"),
        ChatStreamChunk(
            content="",
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        ),
    )
    bridge = GenerationResponseBridge({}, response_id_factory=lambda protocol: "resp-downstream")

    frames = await _collect(bridge.encode_stream(_opened(plan, source), ingress))

    assert frames[-1]["type"] == "response.completed"
    assert frames[-1]["response"]["usage"] == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }
    assert frames[-1]["response"]["output"][0]["content"] == [
        {"type": "output_text", "text": "hello"}
    ]
    assert source.closed is True


@pytest.mark.asyncio
async def test_cross_stream_rejects_legacy_usage_tail_when_terminal_has_usage() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_CHAT)
    ingress = OpenAIResponsesRuntime()
    source = _TrackedIterator(
        ChatStreamChunk(content="", finish_reason="stop", usage={"total_tokens": 3}),
        ChatStreamChunk(content="", usage={"total_tokens": 999}),
    )

    with pytest.raises(ProviderError, match="after its terminal"):
        await _collect(GenerationResponseBridge({}).encode_stream(_opened(plan, source), ingress))

    assert source.closed is True


@pytest.mark.asyncio
async def test_cross_stream_rejects_second_legacy_usage_tail() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_CHAT)
    ingress = OpenAIResponsesRuntime()
    usage = ChatStreamChunk(content="", usage={"total_tokens": 3})
    source = _TrackedIterator(
        ChatStreamChunk(content="", finish_reason="stop"),
        usage,
        usage,
    )

    with pytest.raises(ProviderError, match="after its terminal"):
        await _collect(GenerationResponseBridge({}).encode_stream(_opened(plan, source), ingress))

    assert source.closed is True


@pytest.mark.asyncio
async def test_cross_stream_rejects_non_usage_legacy_tail_atomically() -> None:
    plan = _plan(WireProtocol.OPENAI_RESPONSES, WireProtocol.OPENAI_CHAT)
    ingress = OpenAIResponsesRuntime()
    source = _TrackedIterator(
        ChatStreamChunk(content="", finish_reason="stop"),
        ChatStreamChunk(content="late", usage={"total_tokens": 3}),
    )
    stream = GenerationResponseBridge({}).encode_stream(_opened(plan, source), ingress)

    with pytest.raises(ProviderError, match="after its terminal"):
        await anext(stream)

    assert source.closed is True
