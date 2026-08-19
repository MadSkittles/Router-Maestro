"""Focused Anthropic Messages semantic-runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    ImageContent,
    MessageRole,
    OpaqueState,
    OpenAIResponsesRuntime,
    ProtocolDecodeError,
    ProtocolRepresentabilityError,
    ReasoningSummary,
    RefusalContent,
    SemanticEvent,
    SemanticEventType,
    SemanticMessage,
    SemanticRequest,
    SemanticResponse,
    TerminalMetadata,
    TextContent,
    ToolCall,
    ToolResult,
    Usage,
    WireProtocol,
)


def _request(*messages: dict, **overrides: object) -> dict:
    payload: dict = {
        "model": "gpt-example",
        "max_tokens": 128,
        "messages": list(messages) or [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


def _assistant_reasoning(block: dict) -> dict:
    return {"role": "assistant", "content": [block]}


def test_manifest_finds_features_and_capsules_without_invoking_decoder() -> None:
    decode_calls = 0

    def decode_capsule(*_args, **_kwargs):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("inspect_request must not decrypt capsules")

    payload = _request(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
                },
                {
                    "type": "document",
                    "source": {"type": "text", "data": "notes"},
                },
            ],
        },
        _assistant_reasoning(
            {"type": "thinking", "thinking": "trace", "signature": "rmr2x.native"}
        ),
        _assistant_reasoning({"type": "redacted_thinking", "data": "rmr1.key.payload"}),
        _assistant_reasoning({"type": "redacted_thinking", "data": "rmr27.key.payload"}),
        stream=True,
        tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto", "disable_parallel_tool_use": False},
    )
    runtime = AnthropicMessagesRuntime(decode_opaque_state=decode_capsule)

    manifest = runtime.inspect_request(payload)

    assert manifest.model == "gpt-example"
    assert manifest.stream is True
    assert manifest.tools is True
    assert manifest.images is True
    assert manifest.files is True
    assert manifest.reasoning is True
    assert manifest.parallel_tools is True
    assert manifest.reasoning_capsules == ("rmr1.key.payload", "rmr27.key.payload")
    assert manifest.opaque_continuation is True
    assert decode_calls == 0


@pytest.mark.asyncio
async def test_request_decodes_tools_media_options_and_tool_result_error() -> None:
    runtime = AnthropicMessagesRuntime()
    payload = _request(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": 1}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "failed",
                    "is_error": True,
                }
            ],
        },
        system=[{"type": "text", "text": "system"}],
        tools=[
            {
                "name": "lookup",
                "description": "Lookup a value",
                "input_schema": {"type": "object", "properties": {"q": {"type": "integer"}}},
                "strict": True,
            }
        ],
        tool_choice={"type": "tool", "name": "lookup", "disable_parallel_tool_use": True},
        thinking={"type": "enabled", "budget_tokens": 32},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        },
        stop_sequences=["done"],
        top_k=8,
    )

    semantic = await runtime.decode_request(payload)

    assert semantic.input[0] == SemanticMessage(
        role=MessageRole.SYSTEM,
        content=(TextContent("system"),),
    )
    user = semantic.input[1]
    assert isinstance(user, SemanticMessage)
    assert isinstance(user.content[1], ImageContent)
    assistant = semantic.input[2]
    assert isinstance(assistant, SemanticMessage)
    assert assistant.content[0] == ToolCall(
        call_id="call_1",
        item_id="call_1",
        name="lookup",
        arguments={"q": 1},
    )
    tool_message = semantic.input[3]
    assert isinstance(tool_message, SemanticMessage)
    assert tool_message.role is MessageRole.TOOL
    tool_result = tool_message.content[0]
    assert isinstance(tool_result, ToolResult)
    assert tool_result.call_id == "call_1"
    assert tool_result.is_error is True
    assert semantic.tool_choice is not None
    assert semantic.tool_choice.mode == "function"
    assert semantic.tool_choice.name == "lookup"
    assert semantic.parallel_tool_calls is False
    assert semantic.reasoning is not None
    assert semantic.reasoning.budget_tokens == 32
    assert semantic.reasoning.effort == "high"
    assert semantic.structured_output == {"type": "json_schema", "schema": {"type": "object"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context_management",
    [
        None,
        {},
        {"edits": []},
        {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        {"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "all"}}]},
    ],
)
async def test_context_management_exact_noops_decode_for_cross_protocol_requests(
    context_management: object,
) -> None:
    runtime = AnthropicMessagesRuntime()

    semantic = await runtime.decode_request(_request(context_management=context_management))
    encoded = await OpenAIResponsesRuntime().encode_request(semantic)

    assert encoded["model"] == "gpt-example"
    assert "context_management" not in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context_management",
    [
        {
            "edits": [
                {
                    "type": "clear_thinking_20251015",
                    "keep": {"type": "thinking_turns", "value": 1},
                }
            ]
        },
        {"edits": [{"type": "clear_tool_uses_20250919"}]},
        {
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": 100_000},
                    "keep": {"type": "tool_uses", "value": 3},
                    "clear_at_least": {"type": "input_tokens", "value": 1_000},
                    "exclude_tools": ["keep_me"],
                    "clear_tool_inputs": ["discard_input_for_me"],
                }
            ]
        },
        {
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "clear_at_least": None,
                    "exclude_tools": None,
                    "clear_tool_inputs": None,
                }
            ]
        },
        {"edits": [{"type": "clear_thinking_20251015", "keep": "all", "future": True}]},
        {"edits": [{"type": "future_edit_20260101", "future": {"shape": True}}]},
    ],
)
async def test_active_or_unknown_context_management_is_target_unrepresentable(
    context_management: object,
) -> None:
    semantic = await AnthropicMessagesRuntime().decode_request(
        _request(context_management=context_management)
    )

    assert "context_management" in semantic.provider_extensions
    with pytest.raises(ProtocolRepresentabilityError) as raised:
        await OpenAIResponsesRuntime().encode_request(semantic)

    assert raised.value.path == "context_management"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context_management", "parameter"),
    [
        ([], "context_management"),
        ({"edits": {}}, "context_management.edits"),
        ({"edits": ["clear"]}, "context_management.edits[0]"),
        ({"edits": [{"type": 1}]}, "context_management.edits[0].type"),
        (
            {"edits": [{"type": "clear_thinking_20251015", "keep": 7}]},
            "context_management.edits[0].keep",
        ),
        (
            {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "tool_uses", "value": 1},
                    }
                ]
            },
            "context_management.edits[0].keep.type",
        ),
        (
            {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": "1"},
                    }
                ]
            },
            "context_management.edits[0].keep.value",
        ),
        (
            {
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "input_tokens", "value": False},
                    }
                ]
            },
            "context_management.edits[0].trigger.value",
        ),
        (
            {
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "exclude_tools": ["valid", 3],
                    }
                ]
            },
            "context_management.edits[0].exclude_tools[1]",
        ),
        (
            {
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "keep": "all",
                    }
                ]
            },
            "context_management.edits[0].keep",
        ),
        (
            {
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "clear_tool_inputs": ["valid", 3],
                    }
                ]
            },
            "context_management.edits[0].clear_tool_inputs[1]",
        ),
    ],
)
async def test_malformed_context_management_is_an_ingress_decode_error(
    context_management: object,
    parameter: str,
) -> None:
    with pytest.raises(ProtocolDecodeError) as raised:
        await AnthropicMessagesRuntime().decode_request(
            _request(context_management=context_management)
        )

    assert raised.value.path == parameter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            _request(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello", "cache_control": None}],
                }
            ),
            id="explicit-null",
        ),
        pytest.param(
            _request(
                system=[
                    {
                        "type": "text",
                        "text": "system",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            id="system-text",
        ),
        pytest.param(
            _request(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ),
            id="message-text",
        ),
        pytest.param(
            _request(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AA==",
                            },
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ),
            id="image",
        ),
        pytest.param(
            _request(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "text", "data": "notes"},
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ),
            id="document",
        ),
        pytest.param(
            _request(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "lookup",
                            "input": {},
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ),
            id="tool-use",
        ),
        pytest.param(
            _request(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "done",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ),
            id="tool-result",
        ),
        pytest.param(
            _request(
                tools=[
                    {
                        "name": "lookup",
                        "input_schema": {"type": "object"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            id="tool-definition",
        ),
    ],
)
async def test_standard_ephemeral_cache_control_is_an_advisory_cross_protocol_noop(
    payload: dict,
) -> None:
    semantic = await AnthropicMessagesRuntime().decode_request(payload)

    assert semantic.provider_extensions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cache_control", "error_type", "parameter"),
    [
        (
            {"type": "ephemeral", "ttl": "1h"},
            ProtocolRepresentabilityError,
            "messages[0].content[0].cache_control.ttl",
        ),
        (
            {"type": "ephemeral", "ttl": 60},
            ProtocolDecodeError,
            "messages[0].content[0].cache_control.ttl",
        ),
        (
            {"type": "ephemeral", "ttl": "forever"},
            ProtocolDecodeError,
            "messages[0].content[0].cache_control.ttl",
        ),
        (
            {"type": "persistent"},
            ProtocolRepresentabilityError,
            "messages[0].content[0].cache_control.type",
        ),
        ({}, ProtocolDecodeError, "messages[0].content[0].cache_control.type"),
        ({"type": 1}, ProtocolDecodeError, "messages[0].content[0].cache_control.type"),
        ("ephemeral", ProtocolDecodeError, "messages[0].content[0].cache_control"),
    ],
)
async def test_nonstandard_cache_control_is_rejected_with_exact_path(
    cache_control: object,
    error_type: type[ProtocolDecodeError] | type[ProtocolRepresentabilityError],
    parameter: str,
) -> None:
    payload = _request(
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello", "cache_control": cache_control}],
        }
    )

    with pytest.raises(error_type) as raised:
        await AnthropicMessagesRuntime().decode_request(payload)

    assert raised.value.path == parameter


@pytest.mark.asyncio
async def test_native_reasoning_block_preserves_complete_raw_object_round_trip() -> None:
    block = {
        "type": "thinking",
        "thinking": "summary",
        "signature": "native-signature",
        "future_sibling": {"nested": [1, 2]},
    }
    runtime = AnthropicMessagesRuntime()

    semantic = await runtime.decode_request(_request(_assistant_reasoning(block)))
    message = semantic.input[0]
    assert isinstance(message, SemanticMessage)
    reasoning = message.content[0]
    assert isinstance(reasoning, ReasoningSummary)
    assert reasoning.opaque_state is not None
    assert reasoning.opaque_state.origin_provider is None
    assert isinstance(reasoning.opaque_state.blob, Mapping)
    assert reasoning.opaque_state.blob["future_sibling"] == {"nested": (1, 2)}

    encoded = await runtime.encode_request(semantic)

    assert encoded["messages"][0]["content"][0] == block


@pytest.mark.asyncio
@pytest.mark.parametrize("signature", ["rmr1.bad.value", "rmr2.bad.value"])
async def test_rmr_capsule_without_decoder_context_fails_closed(signature: str) -> None:
    runtime = AnthropicMessagesRuntime()

    with pytest.raises(ProtocolDecodeError, match="requires decoder context") as raised:
        await runtime.decode_request(
            _request(
                _assistant_reasoning(
                    {"type": "thinking", "thinking": "summary", "signature": signature}
                )
            )
        )

    assert raised.value.path == "messages[0].content[0]"


@pytest.mark.asyncio
async def test_invalid_capsule_error_is_sanitized() -> None:
    def reject_capsule(*_args, **_kwargs):
        raise ValueError("provider-secret")

    runtime = AnthropicMessagesRuntime(decode_opaque_state=reject_capsule)

    with pytest.raises(ProtocolDecodeError) as raised:
        await runtime.decode_request(
            _request(_assistant_reasoning({"type": "redacted_thinking", "data": "rmr1.bad"}))
        )

    assert "invalid Router-Maestro reasoning capsule" in str(raised.value)
    assert "provider-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_foreign_opaque_state_requires_and_uses_capsule_encoder() -> None:
    state = OpaqueState(
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_model="gpt-example",
        item_id="rs_1",
        blob="encrypted-content",
        origin_binding="copilot-openai-responses",
    )
    semantic = SemanticRequest(
        model="gpt-example",
        max_output_tokens=64,
        input=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(ReasoningSummary("summary", opaque_state=state),),
            ),
        ),
    )

    with pytest.raises(ProtocolRepresentabilityError, match="capsule encoder context"):
        await AnthropicMessagesRuntime().encode_request(semantic)

    calls = []

    def encode_capsule(
        state: OpaqueState,
        *,
        protocol: WireProtocol,
        model: str,
        item_id: str,
    ) -> str:
        calls.append((state, protocol, model, item_id))
        return "rmr1.key.payload"

    encoded = await AnthropicMessagesRuntime(encode_opaque_state=encode_capsule).encode_request(
        semantic
    )

    assert encoded["messages"][0]["content"][0]["signature"] == "rmr1.key.payload"
    assert calls == [(state, WireProtocol.ANTHROPIC_MESSAGES, "gpt-example", "rs_1")]


@pytest.mark.asyncio
async def test_responses_refusal_in_request_history_fails_closed_for_anthropic() -> None:
    semantic = await OpenAIResponsesRuntime().decode_request(
        {
            "model": "gpt-example",
            "max_output_tokens": 64,
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
                }
            ],
        }
    )

    with pytest.raises(ProtocolRepresentabilityError, match="no refusal content carrier") as raised:
        await AnthropicMessagesRuntime().encode_request(semantic)

    assert raised.value.path == "input[0].content[0]"


@pytest.mark.asyncio
async def test_responses_refusal_response_projects_text_with_refusal_stop_reason() -> None:
    refusal = "I cannot help with that."
    semantic = await OpenAIResponsesRuntime().decode_response(
        {
            "id": "resp_refusal",
            "object": "response",
            "created_at": 1,
            "model": "gpt-example",
            "status": "incomplete",
            "output": [
                {
                    "type": "message",
                    "id": "msg_refusal",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": refusal}],
                }
            ],
            "usage": {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
            "error": None,
            "incomplete_details": {"reason": "content_filter"},
        }
    )

    encoded = await AnthropicMessagesRuntime().encode_response(semantic)

    assert encoded["content"] == [{"type": "text", "text": refusal}]
    assert encoded["stop_reason"] == "refusal"


@pytest.mark.asyncio
async def test_nonstream_refusal_rejects_conflicting_explicit_finish_reason() -> None:
    semantic = SemanticResponse(
        id="msg_refusal",
        model="gpt-example",
        output=(RefusalContent("I cannot help with that."),),
        terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
    )

    with pytest.raises(ProtocolRepresentabilityError, match="conflicts with refusal") as raised:
        await AnthropicMessagesRuntime().encode_response(semantic)

    assert raised.value.path == "response.terminal.finish_reason"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "unknown"])
async def test_nonstream_rejects_non_success_terminal_without_error(status: str) -> None:
    semantic = SemanticResponse(
        id="msg_1",
        model="gpt-example",
        output=(TextContent("partial"),),
        usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
        terminal=TerminalMetadata(response_status=status),
    )

    with pytest.raises(ProtocolRepresentabilityError) as raised:
        await AnthropicMessagesRuntime().encode_response(semantic)

    assert raised.value.path == "response.terminal.response_status"


@pytest.mark.asyncio
async def test_nonstream_incomplete_terminal_keeps_max_tokens_projection() -> None:
    semantic = SemanticResponse(
        id="msg_1",
        model="gpt-example",
        output=(TextContent("partial"),),
        usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
        terminal=TerminalMetadata(finish_reason="length", response_status="incomplete"),
    )

    encoded = await AnthropicMessagesRuntime().encode_response(semantic)

    assert encoded["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_response_round_trip_preserves_usage_reasoning_and_terminal_fields() -> None:
    payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "summary",
                "signature": "native-signature",
                "future_sibling": True,
            },
            {"type": "text", "text": "done"},
        ],
        "model": "gpt-example",
        "stop_reason": "stop_sequence",
        "stop_sequence": "END",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
            "service_tier": "standard_only",
        },
    }
    runtime = AnthropicMessagesRuntime()

    semantic = await runtime.decode_response(payload)

    assert semantic.usage is not None
    assert semantic.usage.total_tokens == 14
    assert semantic.usage.cached_input_tokens == 3
    assert semantic.terminal is not None
    assert semantic.terminal.finish_reason == "stop"
    assert semantic.terminal.stop_sequence == "END"
    assert await runtime.encode_response(semantic) == payload


def test_stream_decoder_maps_full_messages_lifecycle_and_one_terminal() -> None:
    decoder = AnthropicMessagesRuntime().new_stream_decoder(sequence_start=10)
    frames = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "gpt-example",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]

    events = tuple(event for frame in frames for event in decoder.decode(frame))

    assert [event.sequence for event in events] == list(range(10, 17))
    assert [event.type for event in events] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.USAGE,
        SemanticEventType.OUTPUT_ITEM,
        SemanticEventType.TEXT_DELTA,
        SemanticEventType.OUTPUT_ITEM,
        SemanticEventType.USAGE,
        SemanticEventType.TERMINAL,
    ]
    assert all(event.response_id == "msg_1" for event in events)
    assert events[4].metadata["output_item_done"] is True
    assert events[-1].terminal == TerminalMetadata(
        finish_reason="stop",
        response_status="completed",
    )
    assert decoder.finish_eof() == ()
    with pytest.raises(ProtocolDecodeError, match="after terminal"):
        decoder.decode({"type": "message_stop"})


def test_stream_decoder_reassembles_split_capsule_before_invoking_hook() -> None:
    calls = []

    def decode_capsule(value, *, protocol, model, item_id):
        calls.append((value, protocol, model, item_id))
        return OpaqueState(
            origin_protocol=WireProtocol.OPENAI_RESPONSES,
            origin_provider="github-copilot",
            origin_model=model,
            item_id="rs_1",
            blob="encrypted",
            origin_binding="copilot-responses",
        )

    decoder = AnthropicMessagesRuntime(decode_opaque_state=decode_capsule).new_stream_decoder()
    decoder.decode(
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "gpt-example",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        }
    )
    decoder.decode(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }
    )
    decoder.decode(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "rmr1.key."},
        }
    )
    decoder.decode(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "payload"},
        }
    )

    stop_events = decoder.decode({"type": "content_block_stop", "index": 0})

    assert calls == [
        (
            "rmr1.key.payload",
            WireProtocol.ANTHROPIC_MESSAGES,
            "gpt-example",
            "anthropic-thinking-0",
        )
    ]
    assert len(stop_events) == 2
    item = stop_events[0].item
    assert isinstance(item, ReasoningSummary)
    assert item.opaque_state is not None
    assert item.opaque_state.item_id == "rs_1"
    assert stop_events[1].metadata["output_item_done"] is True


def test_stream_error_and_unexpected_eof_have_exactly_one_terminal() -> None:
    error_decoder = AnthropicMessagesRuntime().new_stream_decoder()
    error_events = error_decoder.decode(
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "retry"},
        }
    )
    assert [event.type for event in error_events] == [
        SemanticEventType.ERROR,
        SemanticEventType.TERMINAL,
    ]
    assert error_decoder.finish_eof() == ()

    eof_events = AnthropicMessagesRuntime().new_stream_decoder().finish_eof()
    assert [event.type for event in eof_events] == [
        SemanticEventType.ERROR,
        SemanticEventType.TERMINAL,
    ]
    assert eof_events[-1].terminal is not None
    assert eof_events[-1].terminal.error_code == "unexpected_eof"


def test_stream_encoder_uses_capsule_hook_and_closes_one_terminal() -> None:
    state = OpaqueState(
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_model="gpt-example",
        item_id="rs_1",
        blob="encrypted",
        origin_binding="copilot-responses",
    )
    calls = []

    def encode_capsule(
        state: OpaqueState,
        *,
        protocol: WireProtocol,
        model: str,
        item_id: str,
    ) -> str:
        calls.append((state, protocol, model, item_id))
        return "rmr1.key.payload"

    encoder = AnthropicMessagesRuntime(encode_opaque_state=encode_capsule).new_stream_encoder()
    started = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.RESPONSE_STARTED,
            response_id="msg_1",
            metadata={"model": "gpt-example"},
        )
    )
    reasoning = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.OUTPUT_ITEM,
            response_id="msg_1",
            output_index=0,
            item_id="rs_1",
            item=ReasoningSummary("trace", opaque_state=state),
            metadata={"output_item_done": True},
        )
    )
    terminal = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            response_id="msg_1",
            terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
        )
    )

    assert started[0]["type"] == "message_start"
    assert reasoning[0]["content_block"] == {
        "type": "thinking",
        "thinking": "trace",
        "signature": "rmr1.key.payload",
    }
    assert reasoning[-1] == {"type": "content_block_stop", "index": 0}
    assert [frame["type"] for frame in terminal] == ["message_delta", "message_stop"]
    assert calls == [(state, WireProtocol.ANTHROPIC_MESSAGES, "gpt-example", "rs_1")]
    with pytest.raises(ProtocolRepresentabilityError, match="after terminal"):
        encoder.encode(SemanticEvent(type=SemanticEventType.TEXT_DELTA, delta="late"))


def test_stream_encoder_preserves_usage_totals_when_reasoning_breakdown_is_present() -> None:
    encoder = AnthropicMessagesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="msg_1",
    )
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.USAGE,
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                reasoning_tokens=3,
            ),
        )
    )

    frames = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
        )
    )

    assert frames[-2] == {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    assert frames[-1] == {"type": "message_stop"}


@pytest.mark.parametrize("arguments", [None, "", "   "])
def test_stream_encoder_emits_empty_object_for_zero_argument_tool_call(
    arguments: str | None,
) -> None:
    encoder = AnthropicMessagesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="msg_1",
    )

    argument_frames = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TOOL_ARGUMENTS_DELTA,
            output_index=0,
            item_id="toolu_1",
            delta=arguments,
            metadata={"call_id": "toolu_1", "name": "lookup"},
        )
    )
    terminal_frames = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(
                finish_reason="tool_calls",
                response_status="completed",
            ),
        )
    )
    frames = argument_frames + terminal_frames

    deltas = [
        frame["delta"]["partial_json"]
        for frame in frames
        if frame["type"] == "content_block_delta" and frame["delta"]["type"] == "input_json_delta"
    ]
    assert deltas == ["{}"]
    assert [frame["type"] for frame in frames].count("message_stop") == 1


def test_stream_encoder_preserves_whitespace_after_tool_json_has_started() -> None:
    encoder = AnthropicMessagesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="msg_1",
    )
    frames = []
    for fragment in ("   ", '{"value":"', "   ", 'x"}'):
        frames.extend(
            encoder.encode(
                SemanticEvent(
                    type=SemanticEventType.TOOL_ARGUMENTS_DELTA,
                    output_index=0,
                    item_id="toolu_1",
                    delta=fragment,
                    metadata={"call_id": "toolu_1", "name": "lookup"},
                )
            )
        )
    frames.extend(
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.TERMINAL,
                terminal=TerminalMetadata(
                    finish_reason="tool_calls",
                    response_status="completed",
                ),
            )
        )
    )

    deltas = [
        frame["delta"]["partial_json"]
        for frame in frames
        if frame["type"] == "content_block_delta" and frame["delta"]["type"] == "input_json_delta"
    ]
    assert deltas == ['{"value":"', "   ", 'x"}']
    assert "".join(deltas) == '{"value":"   x"}'


def test_stream_encoder_reuses_tool_identity_for_followup_argument_deltas() -> None:
    encoder = AnthropicMessagesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="msg_1",
    )

    first = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TOOL_ARGUMENTS_DELTA,
            output_index=0,
            item_id="call_1",
            delta='{"location"',
            metadata={"call_id": "call_1", "name": "get_weather"},
        )
    )
    second = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TOOL_ARGUMENTS_DELTA,
            output_index=0,
            delta=':"Paris"}',
        )
    )

    tool_start = next(frame for frame in first if frame["type"] == "content_block_start")
    assert tool_start["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {},
    }
    assert first[-1]["delta"]["partial_json"] == '{"location"'
    assert second == (
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": ':"Paris"}'},
        },
    )


def test_responses_tool_stream_uses_call_id_for_one_anthropic_tool_block() -> None:
    responses = OpenAIResponsesRuntime().new_stream_decoder()
    anthropic = AnthropicMessagesRuntime().new_stream_encoder()
    item_id = "fc_67f2d8a9737c4e17a8f89d57"
    call_id = "call_Pw3vQCRnTQkzvU5s"
    tool_item = {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": "lookup_weather",
        "arguments": '{"city":"Paris"}',
        "status": "completed",
    }
    response = {
        "id": "resp_8f3c",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.2",
        "status": "completed",
        "output": [tool_item],
        "usage": {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
        "error": None,
        "incomplete_details": None,
    }
    upstream_frames = (
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": [], "usage": None},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**tool_item, "arguments": "", "status": "in_progress"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "output_index": 0,
            "delta": '{"city":"',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "output_index": 0,
            "delta": 'Paris"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": item_id,
            "output_index": 0,
            "arguments": tool_item["arguments"],
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": tool_item,
        },
        {"type": "response.completed", "response": response},
    )

    downstream_frames = []
    for upstream_frame in upstream_frames:
        for event in responses.decode(upstream_frame):
            downstream_frames.extend(anthropic.encode(event))

    tool_starts = [
        frame
        for frame in downstream_frames
        if frame["type"] == "content_block_start" and frame["content_block"]["type"] == "tool_use"
    ]
    assert [frame["content_block"] for frame in tool_starts] == [
        {
            "type": "tool_use",
            "id": call_id,
            "name": "lookup_weather",
            "input": {},
        }
    ]
    argument_deltas = [
        frame["delta"]["partial_json"]
        for frame in downstream_frames
        if frame["type"] == "content_block_delta" and frame["delta"]["type"] == "input_json_delta"
    ]
    assert "".join(argument_deltas) == tool_item["arguments"]
    assert [frame["type"] for frame in downstream_frames].count("content_block_stop") == 1
    assert [frame["type"] for frame in downstream_frames].count("message_stop") == 1
    assert (
        next(frame for frame in downstream_frames if frame["type"] == "message_delta")["delta"][
            "stop_reason"
        ]
        == "tool_use"
    )
    assert anthropic.terminal is True


def test_responses_refusal_stream_projects_text_with_refusal_terminal() -> None:
    responses = OpenAIResponsesRuntime().new_stream_decoder()
    anthropic = AnthropicMessagesRuntime().new_stream_encoder()
    refusal = "I cannot help with that."
    message = {
        "type": "message",
        "id": "msg_refusal",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": refusal}],
    }
    response = {
        "id": "resp_refusal",
        "object": "response",
        "created_at": 1,
        "model": "gpt-example",
        "status": "incomplete",
        "output": [message],
        "usage": {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
        "error": None,
        "incomplete_details": {"reason": "content_filter"},
    }
    upstream_frames = (
        {
            "type": "response.created",
            "response": {
                **response,
                "status": "in_progress",
                "output": [],
                "usage": None,
                "incomplete_details": None,
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "item_id": message["id"],
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "refusal", "refusal": ""},
        },
        {
            "type": "response.refusal.delta",
            "item_id": message["id"],
            "output_index": 0,
            "content_index": 0,
            "delta": refusal,
        },
        {
            "type": "response.refusal.done",
            "item_id": message["id"],
            "output_index": 0,
            "content_index": 0,
            "refusal": refusal,
        },
        {
            "type": "response.content_part.done",
            "item_id": message["id"],
            "output_index": 0,
            "content_index": 0,
            "part": message["content"][0],
        },
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.incomplete", "response": response},
    )

    downstream_frames = []
    for upstream_frame in upstream_frames:
        for event in responses.decode(upstream_frame):
            downstream_frames.extend(anthropic.encode(event))

    text_starts = [
        frame
        for frame in downstream_frames
        if frame["type"] == "content_block_start" and frame["content_block"]["type"] == "text"
    ]
    assert len(text_starts) == 1
    text_deltas = [
        frame["delta"]["text"]
        for frame in downstream_frames
        if frame["type"] == "content_block_delta" and frame["delta"]["type"] == "text_delta"
    ]
    assert "".join(text_deltas) == refusal
    assert [frame["type"] for frame in downstream_frames].count("content_block_stop") == 1
    assert [frame["type"] for frame in downstream_frames].count("message_stop") == 1
    terminal = next(frame for frame in downstream_frames if frame["type"] == "message_delta")
    assert terminal["delta"]["stop_reason"] == "refusal"
    assert anthropic.terminal is True


def test_stream_refusal_rejects_conflicting_explicit_finish_reason() -> None:
    encoder = AnthropicMessagesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="msg_refusal",
    )
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.OUTPUT_ITEM,
            output_index=0,
            item_id="msg_refusal_item",
            item=RefusalContent("I cannot help with that."),
        )
    )

    with pytest.raises(ProtocolRepresentabilityError, match="conflicts with refusal") as raised:
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.TERMINAL,
                terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
            )
        )

    assert raised.value.path == "event.terminal.finish_reason"


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("failed", "Upstream response failed"),
        ("cancelled", "Upstream response was cancelled"),
        ("unknown", "Upstream response ended with an unknown status"),
    ],
)
def test_stream_projects_non_success_terminal_without_error(
    status: str,
    message: str,
) -> None:
    encoder = AnthropicMessagesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="msg_1",
    )
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.RESPONSE_STARTED,
            response_id="msg_1",
            metadata={"model": "gpt-example"},
        )
    )

    frames = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(response_status=status),
        )
    )

    assert frames == (
        {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        },
    )
    assert encoder.finish_eof() == ()


def test_stream_incomplete_terminal_keeps_max_tokens_and_one_message_stop() -> None:
    encoder = AnthropicMessagesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="msg_1",
    )
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.RESPONSE_STARTED,
            response_id="msg_1",
            metadata={"model": "gpt-example"},
        )
    )

    frames = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(finish_reason="length", response_status="incomplete"),
        )
    )

    assert frames[-2]["delta"]["stop_reason"] == "max_tokens"
    assert [frame["type"] for frame in frames].count("message_stop") == 1
    assert encoder.finish_eof() == ()


def test_stream_encoder_eof_is_one_error_frame() -> None:
    encoder = AnthropicMessagesRuntime().new_stream_encoder()

    assert encoder.finish_eof() == (
        {
            "type": "error",
            "error": {
                "type": "unexpected_eof",
                "message": "Semantic event stream ended before terminal event",
            },
        },
    )
    assert encoder.finish_eof() == ()
