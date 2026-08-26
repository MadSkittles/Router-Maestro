"""Focused invariants for the lazy protocol foundation."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from router_maestro.protocols import (
    DuplicateProtocolRuntimeError,
    FrozenJsonValue,
    MessageRole,
    OpaqueState,
    OpenAIChatRuntime,
    ProtocolRuntime,
    ProtocolRuntimeNotFoundError,
    ProtocolRuntimeRegistry,
    RefusalContent,
    RequestEnvelope,
    RequestManifest,
    SemanticEvent,
    SemanticEventType,
    SemanticMessage,
    SemanticRequest,
    SemanticResponse,
    TerminalMetadata,
    TextContent,
    ToolDefinition,
    UnsupportedProtocolOperationError,
    Usage,
    UsageMode,
    WireProtocol,
)


class _Runtime(ProtocolRuntime):
    def __init__(self, protocol: WireProtocol) -> None:
        self.protocol = protocol
        self.decode_calls = 0

    def inspect_request(self, payload) -> RequestManifest:
        return RequestManifest(
            protocol=self.protocol,
            model=payload.get("model"),
            stream=payload.get("stream", False),
        )

    async def decode_request(self, payload) -> SemanticRequest:
        self.decode_calls += 1
        await asyncio.sleep(0.01)
        payload["messages"][0]["content"] = "decoder mutation"
        return SemanticRequest(
            model=payload["model"],
            input=(
                SemanticMessage(
                    role=MessageRole.USER,
                    content=(TextContent(payload["messages"][0]["content"]),),
                ),
            ),
        )

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        del request
        raise UnsupportedProtocolOperationError(self.protocol, "encode_request")

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        del payload
        raise UnsupportedProtocolOperationError(self.protocol, "decode_response")

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        del response
        raise UnsupportedProtocolOperationError(self.protocol, "encode_response")

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        del payload
        raise UnsupportedProtocolOperationError(self.protocol, "decode_stream_event")

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        del event
        raise UnsupportedProtocolOperationError(self.protocol, "encode_stream_event")


def _payload() -> dict[str, Any]:
    return {
        "model": "example-model",
        "stream": True,
        "messages": [{"role": "user", "content": "original"}],
    }


def test_provider_first_import_does_not_cycle_through_protocol_codecs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from router_maestro.providers import ModelInfo; "
            "from router_maestro.protocols import OpenAIChatRuntime; "
            "assert ModelInfo and OpenAIChatRuntime",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_optional_representability_hook_does_not_expand_runtime_contract() -> None:
    runtime = OpenAIChatRuntime()

    assert not hasattr(runtime, "request_representability")
    assert isinstance(runtime, ProtocolRuntime)

    registry = ProtocolRuntimeRegistry()
    registry.register(runtime)
    assert registry.get(WireProtocol.OPENAI_CHAT) is runtime


def test_identity_access_never_materializes_semantic_ir() -> None:
    runtime = _Runtime(WireProtocol.OPENAI_RESPONSES)
    envelope = RequestEnvelope(runtime, _payload())

    assert envelope.protocol is WireProtocol.OPENAI_RESPONSES
    assert envelope.manifest.model == "example-model"
    assert envelope.native_payload()["messages"][0]["content"] == "original"
    assert envelope.materialization_count == 0
    assert runtime.decode_calls == 0


def test_envelope_preserves_isolated_transport_context_without_ir() -> None:
    runtime = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    query = {"beta": "true"}
    headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
    envelope = RequestEnvelope(
        runtime,
        _payload(),
        path="/v1/messages",
        query=query,
        headers=headers,
    )

    query["beta"] = "false"
    headers["anthropic-beta"] = "mutated"
    returned_headers = envelope.headers
    returned_headers["anthropic-beta"] = "returned mutation"

    assert envelope.path == "/v1/messages"
    assert envelope.query == {"beta": "true"}
    assert envelope.headers == {"anthropic-beta": "interleaved-thinking-2025-05-14"}
    assert envelope.model == "example-model"
    assert envelope.stream is True
    assert envelope.materialization_count == 0
    assert runtime.decode_calls == 0


@pytest.mark.asyncio
async def test_repeated_and_concurrent_ir_access_materializes_exactly_once() -> None:
    runtime = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)
    envelope = RequestEnvelope(runtime, _payload())

    results = await asyncio.gather(*(envelope.semantic_ir() for _ in range(20)))
    repeated = await envelope.semantic_ir()

    assert all(result is results[0] for result in results)
    assert repeated is results[0]
    assert runtime.decode_calls == 1
    assert envelope.materialization_count == 1


def test_semantic_ir_single_flight_is_safe_across_event_loop_threads() -> None:
    runtime = _Runtime(WireProtocol.OPENAI_CHAT)
    envelope = RequestEnvelope(runtime, _payload())

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: asyncio.run(envelope.semantic_ir()), range(4)))

    assert all(result is results[0] for result in results)
    assert runtime.decode_calls == 1
    assert envelope.materialization_count == 1


def test_caller_and_returned_payload_mutation_are_isolated() -> None:
    payload = _payload()
    envelope = RequestEnvelope(_Runtime(WireProtocol.GEMINI), payload)

    payload["messages"][0]["content"] = "caller mutation"
    returned = envelope.raw_payload
    returned["messages"][0]["content"] = "returned mutation"

    assert envelope.raw_payload["messages"][0]["content"] == "original"
    assert envelope.materialization_count == 0


@pytest.mark.asyncio
async def test_decoder_mutation_cannot_change_the_preserved_native_payload() -> None:
    envelope = RequestEnvelope(_Runtime(WireProtocol.OPENAI_CHAT), _payload())

    await envelope.semantic_ir()

    assert envelope.raw_payload["messages"][0]["content"] == "original"


@pytest.mark.parametrize(
    "protocol",
    [
        WireProtocol.ANTHROPIC_MESSAGES,
        WireProtocol.OPENAI_CHAT,
        WireProtocol.OPENAI_RESPONSES,
        WireProtocol.GEMINI,
    ],
)
def test_registry_registers_and_resolves_every_wire_protocol(protocol: WireProtocol) -> None:
    registry = ProtocolRuntimeRegistry()
    runtime = _Runtime(protocol)

    registry.register(runtime)

    assert registry.get(protocol) is runtime
    assert protocol in registry
    assert registry.snapshot()[protocol] is runtime


def test_registry_rejects_duplicates_and_requires_explicit_replacement() -> None:
    registry = ProtocolRuntimeRegistry()
    original = _Runtime(WireProtocol.OPENAI_CHAT)
    replacement = _Runtime(WireProtocol.OPENAI_CHAT)
    registry.register(original)

    with pytest.raises(DuplicateProtocolRuntimeError) as error:
        registry.register(replacement)

    assert error.value.protocol is WireProtocol.OPENAI_CHAT
    assert registry.replace(replacement) is original
    assert registry.get(WireProtocol.OPENAI_CHAT) is replacement


def test_registry_missing_lookup_raises_typed_error() -> None:
    registry = ProtocolRuntimeRegistry()

    with pytest.raises(ProtocolRuntimeNotFoundError) as error:
        registry.get(WireProtocol.GEMINI)

    assert error.value.protocol is WireProtocol.GEMINI


@pytest.mark.asyncio
async def test_default_optional_runtime_hook_reports_unsupported_operation() -> None:
    runtime = _Runtime(WireProtocol.ANTHROPIC_MESSAGES)

    with pytest.raises(UnsupportedProtocolOperationError) as error:
        await runtime.encode_request(SemanticRequest(model="example-model"))

    assert error.value.protocol is WireProtocol.ANTHROPIC_MESSAGES
    assert error.value.operation == "encode_request"


def test_semantic_models_freeze_nested_json_and_opaque_provenance() -> None:
    schema = {"type": "object", "required": ["city"]}
    tool = ToolDefinition(name="weather", input_schema=schema)
    state = OpaqueState(
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_model="gpt-example",
        item_id="reasoning-1",
        blob=b"opaque",
    )
    schema["required"].append("unit")

    assert tool.input_schema["required"] == ("city",)
    with pytest.raises(TypeError):
        tool.input_schema["new"] = True  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        state.item_id = "detached"  # type: ignore[misc]


def test_usage_distinguishes_missing_zero_delta_and_snapshot() -> None:
    missing = Usage()
    zero_delta = Usage(mode=UsageMode.DELTA, input_tokens=0, output_tokens=0)

    assert missing.mode is UsageMode.SNAPSHOT
    assert missing.input_tokens is None
    assert zero_delta.mode is UsageMode.DELTA
    assert zero_delta.input_tokens == 0


def test_manifest_freezes_reasoning_affinity_hints() -> None:
    capsules = ["rmr1.key.payload"]
    manifest = RequestManifest(
        protocol=WireProtocol.ANTHROPIC_MESSAGES,
        reasoning=True,
        reasoning_capsules=cast(tuple[str, ...], capsules),
        opaque_continuation=True,
    )
    capsules.append("rmr1.other.payload")

    assert manifest.reasoning_capsules == ("rmr1.key.payload",)
    assert manifest.opaque_continuation is True


def test_extended_request_and_event_fidelity_fields_are_immutable() -> None:
    structured_output = {"type": "json_schema", "schema": {"type": "object"}}
    extensions = {"vendor": {"option": [1]}}
    request = SemanticRequest(
        model="example-model",
        input=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(RefusalContent(refusal="cannot comply"),),
                item_id="message-1",
                status="completed",
            ),
        ),
        structured_output=structured_output,
        provider_extensions=cast(Mapping[str, FrozenJsonValue], extensions),
        explicit_fields=cast(
            frozenset[str],
            {"structured_output", "provider_extensions"},
        ),
    )
    event = SemanticEvent(
        type=SemanticEventType.TERMINAL,
        sequence=7,
        terminal=TerminalMetadata(response_status="completed", transport_status=200),
    )
    structured_output["schema"]["type"] = "array"
    extensions["vendor"]["option"].append(2)

    assert request.structured_output is not None
    schema_value = request.structured_output["schema"]
    assert isinstance(schema_value, Mapping)
    assert schema_value["type"] == "object"
    vendor_value = request.provider_extensions["vendor"]
    assert isinstance(vendor_value, Mapping)
    assert vendor_value["option"] == (1,)
    assert request.explicit_fields == frozenset({"structured_output", "provider_extensions"})
    assert event.sequence == 7
    assert event.terminal is not None
    assert event.terminal.transport_status == 200
