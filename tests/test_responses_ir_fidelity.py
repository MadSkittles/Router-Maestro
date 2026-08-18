"""Focused Semantic IR fidelity contracts for the Responses wire protocol."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    FileContent,
    GeminiRuntime,
    MessageRole,
    OpenAIChatRuntime,
    OpenAIResponsesRuntime,
    ProtocolDecodeError,
    ProtocolRepresentabilityError,
    ProtocolRuntime,
    SemanticEvent,
    SemanticEventType,
    SemanticMessage,
    SemanticRequest,
    TerminalMetadata,
    TextContent,
    ToolCall,
    ToolResult,
    Usage,
    UsageMode,
)


def _response(*, usage: object) -> dict[str, object]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 10,
        "model": "gpt-example",
        "status": "completed",
        "output": [],
        "usage": usage,
        "error": None,
        "incomplete_details": None,
    }


@pytest.mark.asyncio
async def test_responses_tool_call_result_kinds_and_namespaces_round_trip() -> None:
    runtime = OpenAIResponsesRuntime()
    input_items = [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_function",
            "name": "lookup",
            "namespace": "crm",
            "arguments": '{"query":"rm"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "id": "fco_1",
            "call_id": "call_function",
            "output": "function result",
        },
        {
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_custom",
            "name": "code_exec",
            "namespace": "sandbox",
            "input": "print('hello')",
            "status": "completed",
        },
        {
            "type": "custom_tool_call_output",
            "id": "ctco_1",
            "call_id": "call_custom",
            "namespace": "sandbox",
            "output": [
                {"type": "input_text", "text": "custom result"},
                {"type": "input_file", "file_id": "file_1"},
            ],
        },
        {
            "type": "tool_search_call",
            "id": "tsc_1",
            "call_id": "call_search",
            "execution": "client",
            "status": "completed",
            "arguments": {"paths": ["crm"]},
        },
        {
            "type": "tool_search_output",
            "id": "tso_1",
            "call_id": "call_search",
            "execution": "client",
            "status": "completed",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup_order",
                    "description": "Look up an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                    },
                }
            ],
        },
    ]

    semantic = await runtime.decode_request({"model": "gpt-example", "input": input_items})

    function_call, function_result = semantic.input[0:2]
    custom_call, custom_result = semantic.input[2:4]
    search_call, search_result = semantic.input[4:6]
    assert isinstance(function_call, ToolCall)
    assert isinstance(function_result, ToolResult)
    assert (function_call.kind, function_call.namespace) == ("function", "crm")
    assert (function_result.kind, function_result.namespace) == ("function", "crm")
    assert isinstance(custom_call, ToolCall)
    assert isinstance(custom_result, ToolResult)
    assert (custom_call.kind, custom_call.namespace) == ("custom", "sandbox")
    assert (custom_result.kind, custom_result.namespace) == ("custom", "sandbox")
    assert custom_result.content == (
        TextContent("custom result"),
        FileContent(source="file_1", source_kind="file_id"),
    )
    assert isinstance(search_call, ToolCall)
    assert isinstance(search_result, ToolResult)
    assert (search_call.kind, search_result.kind) == ("tool_search", "tool_search")
    assert isinstance(search_result.structured_content, Mapping)
    assert search_result.structured_content["execution"] == "client"
    assert search_result.structured_content["status"] == "completed"

    encoded = await runtime.encode_request(semantic)

    assert encoded["input"] == input_items


@pytest.mark.asyncio
async def test_responses_orphan_function_output_keeps_compatible_defaults() -> None:
    runtime = OpenAIResponsesRuntime()
    semantic = await runtime.decode_request(
        {
            "model": "gpt-example",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_orphan",
                    "output": "ok",
                }
            ],
        }
    )

    result = semantic.input[0]
    assert isinstance(result, ToolResult)
    assert result.kind == "function"
    assert result.namespace is None
    assert (await runtime.encode_request(semantic))["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_orphan",
            "output": "ok",
        }
    ]


@pytest.mark.asyncio
async def test_responses_result_type_must_match_preceding_call() -> None:
    runtime = OpenAIResponsesRuntime()

    with pytest.raises(ProtocolDecodeError) as raised:
        await runtime.decode_request(
            {
                "model": "gpt-example",
                "input": [
                    {
                        "type": "custom_tool_call",
                        "call_id": "call_1",
                        "name": "code_exec",
                        "input": "print(1)",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "1",
                    },
                ],
            }
        )

    assert raised.value.path == "input[1].type"


@pytest.mark.asyncio
async def test_responses_custom_output_namespace_round_trips_and_matches_call() -> None:
    runtime = OpenAIResponsesRuntime()
    orphan = await runtime.decode_request(
        {
            "model": "gpt-example",
            "input": [
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "namespace": "sandbox",
                    "output": "ok",
                }
            ],
        }
    )

    result = orphan.input[0]
    assert isinstance(result, ToolResult)
    assert (result.kind, result.namespace) == ("custom", "sandbox")
    assert (await runtime.encode_request(orphan))["input"] == [
        {
            "type": "custom_tool_call_output",
            "call_id": "call_1",
            "namespace": "sandbox",
            "output": "ok",
        }
    ]

    with pytest.raises(ProtocolDecodeError) as mismatch:
        await runtime.decode_request(
            {
                "model": "gpt-example",
                "input": [
                    {
                        "type": "custom_tool_call",
                        "call_id": "call_1",
                        "name": "code_exec",
                        "namespace": "sandbox",
                        "input": "print(1)",
                    },
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call_1",
                        "namespace": "other",
                        "output": "1",
                    },
                ],
            }
        )
    assert mismatch.value.path == "input[1].namespace"


@pytest.mark.asyncio
async def test_responses_tool_result_rejects_error_and_function_output_namespace() -> None:
    runtime = OpenAIResponsesRuntime()

    with pytest.raises(ProtocolRepresentabilityError) as error_raised:
        await runtime.encode_request(
            SemanticRequest(
                model="gpt-example",
                input=(ToolResult(call_id="call_1", content=(TextContent("bad"),), is_error=True),),
            )
        )
    assert error_raised.value.path == "input[0].is_error"

    with pytest.raises(ProtocolRepresentabilityError) as namespace_raised:
        await runtime.encode_request(
            SemanticRequest(
                model="gpt-example",
                input=(
                    ToolResult(
                        call_id="call_1",
                        content=(TextContent("ok"),),
                        kind="function",
                        namespace="sandbox",
                    ),
                ),
            )
        )
    assert namespace_raised.value.path == "input[0].namespace"

    with pytest.raises(ProtocolDecodeError) as wire_namespace:
        await runtime.decode_request(
            {
                "model": "gpt-example",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "namespace": "sandbox",
                        "output": "ok",
                    }
                ],
            }
        )
    assert wire_namespace.value.path == "input[0].namespace"


@pytest.mark.asyncio
async def test_responses_usage_preserves_missing_zero_and_snapshot_mode() -> None:
    runtime = OpenAIResponsesRuntime()

    missing = await runtime.decode_response(_response(usage=None))
    zero = await runtime.decode_response(
        _response(
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
        )
    )

    assert missing.usage is None
    assert zero.usage == Usage(
        mode=UsageMode.SNAPSHOT,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cached_input_tokens=0,
        reasoning_tokens=0,
    )
    assert (await runtime.encode_response(zero))["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def test_responses_stream_preserves_event_order_indices_and_partial_usage() -> None:
    runtime = OpenAIResponsesRuntime()
    decoder = runtime.new_stream_decoder(sequence_start=9)
    started = decoder.decode(
        {
            "type": "response.created",
            "response": {
                "id": "resp_1",
                "model": "gpt-example",
                "status": "in_progress",
                "output": [],
                "usage": None,
            },
        }
    )
    later_output = decoder.decode(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_4",
            "output_index": 4,
            "content_index": 2,
            "delta": "later",
        }
    )
    earlier_output = decoder.decode(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 1,
            "content_index": 0,
            "delta": "earlier",
        }
    )
    completed = decoder.decode(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-example",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
    )
    events = started + later_output + earlier_output + completed

    assert [event.sequence for event in events] == [9, 10, 11, 12, 13]
    assert [event.type for event in events] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.TEXT_DELTA,
        SemanticEventType.TEXT_DELTA,
        SemanticEventType.USAGE,
        SemanticEventType.TERMINAL,
    ]
    assert [event.output_index for event in events[1:3]] == [4, 1]
    assert [event.content_index for event in events[1:3]] == [2, 0]
    assert events[3].usage == Usage(
        mode=UsageMode.SNAPSHOT,
        input_tokens=0,
        output_tokens=0,
    )


def test_responses_stream_rejects_delta_usage_projection() -> None:
    encoder = OpenAIResponsesRuntime().new_stream_encoder(model="gpt-example", response_id="resp_1")
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.USAGE,
            usage=Usage(mode=UsageMode.DELTA, output_tokens=1),
        )
    )

    with pytest.raises(ProtocolRepresentabilityError) as raised:
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.TERMINAL,
                terminal=TerminalMetadata(
                    finish_reason="stop",
                    response_status="completed",
                ),
            )
        )

    assert raised.value.path == "event.usage.mode"


def _cross_protocol_runtime(name: str) -> ProtocolRuntime:
    if name == "anthropic":
        return AnthropicMessagesRuntime()
    if name == "chat":
        return OpenAIChatRuntime()
    if name == "gemini":
        return GeminiRuntime()
    raise AssertionError(f"unexpected runtime {name!r}")


def _cross_protocol_request(name: str, result: ToolResult) -> SemanticRequest:
    item: object = result
    if name == "gemini":
        item = SemanticMessage(
            role=MessageRole.TOOL,
            name="lookup",
            content=(result,),
        )
    return SemanticRequest(
        model="example-model",
        input=(item,),
        max_output_tokens=64 if name == "anthropic" else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_name", "path_prefix"),
    [
        ("anthropic", "input[0].content[0]"),
        ("chat", "input[0].content"),
        ("gemini", "input[0].content[0]"),
    ],
)
@pytest.mark.parametrize(
    ("result", "field"),
    [
        (ToolResult(call_id="call_1", kind="custom"), "kind"),
        (ToolResult(call_id="call_1", namespace="crm"), "namespace"),
    ],
)
async def test_cross_protocol_tool_results_reject_kind_and_namespace_silent_drop(
    runtime_name: str,
    path_prefix: str,
    result: ToolResult,
    field: str,
) -> None:
    runtime = _cross_protocol_runtime(runtime_name)

    with pytest.raises(ProtocolRepresentabilityError) as raised:
        await runtime.encode_request(_cross_protocol_request(runtime_name, result))

    assert raised.value.path == f"{path_prefix}.{field}"


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_name", ["anthropic", "chat", "gemini"])
async def test_cross_protocol_function_tool_result_path_remains_supported(
    runtime_name: str,
) -> None:
    runtime = _cross_protocol_runtime(runtime_name)
    request = _cross_protocol_request(
        runtime_name,
        ToolResult(call_id="call_1", content=(TextContent("ok"),)),
    )

    encoded = await runtime.encode_request(request)

    assert encoded
