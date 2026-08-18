"""Focused Gemini semantic-runtime and streaming contracts."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    FileContent,
    GeminiRuntime,
    ImageContent,
    MessageRole,
    OpaqueState,
    OpaqueStateDecodeHook,
    OpaqueStateEncodeHook,
    OpenAIChatRuntime,
    OpenAIResponsesRuntime,
    ProtocolDecodeError,
    ProtocolRepresentabilityError,
    ReasoningSummary,
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

_MODEL = "gemini-example"


def _request(*contents: dict, **overrides: object) -> dict:
    payload: dict = {
        "contents": list(contents) or [{"role": "user", "parts": [{"text": "hello"}]}],
    }
    payload.update(overrides)
    return payload


def _runtime(
    *,
    stream: bool = False,
    origin_provider: str | None = None,
    decode_opaque_state: OpaqueStateDecodeHook | None = None,
    encode_opaque_state: OpaqueStateEncodeHook | None = None,
) -> GeminiRuntime:
    return GeminiRuntime(
        default_model=_MODEL,
        stream=stream,
        origin_provider=origin_provider,
        decode_opaque_state=decode_opaque_state,
        encode_opaque_state=encode_opaque_state,
    )


def test_manifest_classifies_media_and_capsules_without_decoding() -> None:
    decode_calls = 0

    def decode_capsule(*_args, **_kwargs):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("inspect_request must not decrypt capsules")

    payload = _request(
        {
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "image/png", "data": "AA=="}},
                {"fileData": {"mimeType": "application/pdf", "fileUri": "files/1"}},
            ],
        },
        {
            "role": "model",
            "parts": [
                {
                    "text": "trace",
                    "thought": True,
                    "thoughtSignature": "rmr1.key.payload",
                },
                {
                    "text": "future",
                    "thought": True,
                    "thoughtSignature": "rmr42.key.payload",
                },
                {
                    "text": "native",
                    "thought": True,
                    "thoughtSignature": "rmr2x.native",
                },
            ],
        },
        tools=[{"functionDeclarations": [{"name": "lookup"}]}],
        generationConfig={"thinkingConfig": {"includeThoughts": True}},
    )
    runtime = _runtime(stream=True, decode_opaque_state=decode_capsule)

    manifest = runtime.inspect_request(payload)

    assert manifest.model == _MODEL
    assert manifest.stream is True
    assert manifest.tools is True
    assert manifest.images is True
    assert manifest.files is True
    assert manifest.reasoning is True
    assert manifest.reasoning_capsules == ("rmr1.key.payload", "rmr42.key.payload")
    assert manifest.opaque_continuation is True
    assert decode_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "path"),
    [
        (_request(unknown=True), "unknown"),
        (
            _request({"role": "user", "parts": [{"text": "hello", "future": True}]}),
            "contents[0].parts[0].future",
        ),
        (
            _request(generationConfig={"future": True}),
            "generationConfig.future",
        ),
        (
            _request(tools=[{"functionDeclarations": [{"name": "lookup", "future": True}]}]),
            "tools[0].functionDeclarations[0].future",
        ),
        (
            _request(toolConfig={"functionCallingConfig": {"mode": "AUTO", "future": True}}),
            "toolConfig.functionCallingConfig.future",
        ),
    ],
)
async def test_unknown_request_fields_fail_closed(payload: dict, path: str) -> None:
    with pytest.raises(ProtocolDecodeError) as raised:
        await _runtime().decode_request(payload)

    assert raised.value.path == path


@pytest.mark.asyncio
async def test_missing_function_response_id_binds_one_prior_matching_call() -> None:
    semantic = await _runtime().decode_request(
        _request(
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "lookup", "args": {"q": 1}}}],
            },
            {
                "role": "user",
                "parts": [{"functionResponse": {"name": "lookup", "response": {"value": 2}}}],
            },
        )
    )

    call_message = semantic.input[0]
    result_message = semantic.input[1]
    assert isinstance(call_message, SemanticMessage)
    assert isinstance(result_message, SemanticMessage)
    call = call_message.content[0]
    result = result_message.content[0]
    assert isinstance(call, ToolCall)
    assert isinstance(result, ToolResult)
    assert call.call_id == "gemini-call-0-0"
    assert result.call_id == call.call_id
    assert result.structured_content == {"value": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_count", [0, 2])
async def test_missing_function_response_id_rejects_ambiguous_history(
    prior_count: int,
) -> None:
    contents = [
        {
            "role": "model",
            "parts": [
                {"functionCall": {"name": "lookup", "args": {"q": index}}}
                for index in range(prior_count)
            ],
        },
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "lookup", "response": {"value": 2}}}],
        },
    ]

    with pytest.raises(ProtocolDecodeError, match=f"found {prior_count}") as raised:
        await _runtime().decode_request(_request(*contents))

    assert raised.value.path == "contents[1].parts[0].functionResponse.id"


@pytest.mark.asyncio
async def test_function_responses_consume_prior_calls() -> None:
    payload = _request(
        {
            "role": "model",
            "parts": [
                {"functionCall": {"id": "call-1", "name": "lookup", "args": {}}},
                {"functionCall": {"id": "call-2", "name": "lookup", "args": {}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": "call-1",
                        "name": "lookup",
                        "response": {"value": 1},
                    }
                },
                {
                    "functionResponse": {
                        "name": "lookup",
                        "response": {"value": 2},
                    }
                },
            ],
        },
    )

    semantic = await _runtime().decode_request(payload)

    first_result = semantic.input[1]
    second_result = semantic.input[2]
    assert isinstance(first_result, SemanticMessage)
    assert isinstance(second_result, SemanticMessage)
    assert isinstance(first_result.content[0], ToolResult)
    assert isinstance(second_result.content[0], ToolResult)
    assert first_result.content[0].call_id == "call-1"
    assert second_result.content[0].call_id == "call-2"


@pytest.mark.asyncio
async def test_function_response_cannot_reuse_consumed_implicit_call() -> None:
    payload = _request(
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "lookup", "args": {}}}],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "lookup", "response": {"value": 1}}},
                {"functionResponse": {"name": "lookup", "response": {"value": 2}}},
            ],
        },
    )

    with pytest.raises(ProtocolDecodeError, match="found 0") as raised:
        await _runtime().decode_request(payload)

    assert raised.value.path == "contents[1].parts[1].functionResponse.id"


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_count", [0, 2])
async def test_candidate_count_other_than_one_is_rejected(candidate_count: int) -> None:
    payload = _request(generationConfig={"candidateCount": candidate_count})

    with pytest.raises(ProtocolDecodeError, match="must equal 1") as raised:
        await _runtime().decode_request(payload)

    assert raised.value.path == "generationConfig.candidateCount"


@pytest.mark.asyncio
async def test_single_candidate_is_representable_in_all_upstream_transports() -> None:
    semantic = await _runtime().decode_request(
        _request(generationConfig={"candidateCount": 1, "maxOutputTokens": 64})
    )

    anthropic = await AnthropicMessagesRuntime().encode_request(semantic)
    chat = await OpenAIChatRuntime().encode_request(semantic)
    responses = await OpenAIResponsesRuntime().encode_request(semantic)

    assert "candidate_count" not in anthropic
    assert "candidate_count" not in chat
    assert "candidate_count" not in responses


@pytest.mark.asyncio
async def test_validated_tool_mode_is_rejected_at_exact_path() -> None:
    payload = _request(toolConfig={"functionCallingConfig": {"mode": "VALIDATED"}})

    with pytest.raises(ProtocolDecodeError, match="no exact cross-protocol mapping") as raised:
        await _runtime().decode_request(payload)

    assert raised.value.path == "toolConfig.functionCallingConfig.mode"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"safetySettings": []}, "safetySettings"),
        ({"cachedContent": "cachedContents/1"}, "cachedContent"),
        ({"tools": [{"googleSearch": {}}]}, "tools[0].googleSearch"),
    ],
)
async def test_nonportable_gemini_features_fail_closed(overrides: dict, path: str) -> None:
    with pytest.raises(ProtocolDecodeError) as raised:
        await _runtime().decode_request(_request(**overrides))

    assert raised.value.path == path


@pytest.mark.asyncio
async def test_function_call_arguments_must_be_an_object() -> None:
    payload = _request(
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "lookup", "args": "invalid"}}],
        }
    )

    with pytest.raises(ProtocolDecodeError) as raised:
        await _runtime().decode_request(payload)

    assert raised.value.path == "contents[0].parts[0].functionCall.args"


@pytest.mark.asyncio
async def test_inline_image_projects_to_data_url_for_openai_transports() -> None:
    semantic = await _runtime().decode_request(
        _request(
            {
                "role": "user",
                "parts": [{"inlineData": {"mimeType": "image/png", "data": "AA=="}}],
            }
        )
    )

    chat = await OpenAIChatRuntime().encode_request(semantic)
    responses = await OpenAIResponsesRuntime().encode_request(semantic)

    assert chat["messages"][0]["content"][0]["image_url"]["url"] == ("data:image/png;base64,AA==")
    assert responses["input"][0]["content"][0]["image_url"] == ("data:image/png;base64,AA==")


@pytest.mark.asyncio
async def test_non_image_inline_data_is_not_representable_in_openai_transports() -> None:
    semantic = await _runtime().decode_request(
        _request(
            {
                "role": "user",
                "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "AA=="}}],
            }
        )
    )

    with pytest.raises(ProtocolRepresentabilityError):
        await OpenAIChatRuntime().encode_request(semantic)
    with pytest.raises(ProtocolRepresentabilityError):
        await OpenAIResponsesRuntime().encode_request(semantic)


@pytest.mark.asyncio
async def test_request_round_trip_preserves_media_tools_and_structured_output() -> None:
    payload = _request(
        {
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "image/png", "data": "AA=="}},
                {
                    "fileData": {
                        "mimeType": "application/pdf",
                        "fileUri": "files/report",
                        "displayName": "report.pdf",
                    }
                },
            ],
        },
        tools=[
            {
                "functionDeclarations": [
                    {
                        "name": "lookup",
                        "description": "Lookup",
                        "parametersJsonSchema": {
                            "type": "OBJECT",
                            "properties": {"q": {"type": "STRING"}},
                        },
                    }
                ]
            }
        ],
        toolConfig={"functionCallingConfig": {"mode": "ANY"}},
        generationConfig={
            "maxOutputTokens": 64,
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "OBJECT",
                "properties": {"answer": {"type": "STRING"}},
            },
            "thinkingConfig": {"thinkingBudget": -1, "includeThoughts": True},
        },
    )
    runtime = _runtime()

    semantic = await runtime.decode_request(payload)

    message = semantic.input[0]
    assert isinstance(message, SemanticMessage)
    assert isinstance(message.content[0], ImageContent)
    assert isinstance(message.content[1], FileContent)
    assert semantic.structured_output == {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
    }
    assert semantic.reasoning is not None
    assert semantic.reasoning.effort == "adaptive"
    encoded = await runtime.encode_request(semantic)
    assert encoded["contents"] == payload["contents"]
    assert encoded["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert encoded["generationConfig"]["responseJsonSchema"] == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("signature", ["rmr1.bad.value", "rmr2.bad.value"])
async def test_capsule_requires_context_and_decoder_errors_are_sanitized(
    signature: str,
) -> None:
    payload = _request(
        {
            "role": "model",
            "parts": [{"text": "trace", "thought": True, "thoughtSignature": signature}],
        }
    )

    with pytest.raises(ProtocolDecodeError, match="requires decoder context"):
        await _runtime().decode_request(payload)

    def reject_capsule(*_args, **_kwargs):
        raise ValueError("provider-secret")

    with pytest.raises(ProtocolDecodeError) as raised:
        await _runtime(decode_opaque_state=reject_capsule).decode_request(payload)

    assert "invalid Router-Maestro reasoning capsule" in str(raised.value)
    assert "provider-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_foreign_reasoning_uses_capsule_encoder() -> None:
    state = OpaqueState(
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_model=_MODEL,
        item_id="rs_1",
        blob="encrypted",
        origin_binding="copilot-responses",
    )
    request = SemanticRequest(
        model=_MODEL,
        input=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(ReasoningSummary("trace", opaque_state=state),),
            ),
        ),
    )

    with pytest.raises(ProtocolRepresentabilityError, match="capsule encoder context"):
        await _runtime().encode_request(request)

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

    encoded = await _runtime(encode_opaque_state=encode_capsule).encode_request(request)

    assert encoded["contents"][0]["parts"][0]["thoughtSignature"] == "rmr1.key.payload"
    assert calls == [(state, WireProtocol.GEMINI, _MODEL, "rs_1")]


@pytest.mark.asyncio
async def test_response_round_trip_preserves_reasoning_usage_and_terminal() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "trace",
                            "thought": True,
                            "thoughtSignature": "native-signature",
                        },
                        {"text": "done"},
                    ],
                },
                "finishReason": "MAX_TOKENS",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "totalTokenCount": 14,
            "cachedContentTokenCount": 3,
            "thoughtsTokenCount": 2,
        },
        "modelVersion": _MODEL,
    }
    runtime = _runtime()

    semantic = await runtime.decode_response(payload)

    output = semantic.output[0]
    assert isinstance(output, SemanticMessage)
    reasoning = output.content[0]
    assert isinstance(reasoning, ReasoningSummary)
    assert reasoning.opaque_state is not None
    assert reasoning.opaque_state.blob == "native-signature"
    assert semantic.usage == Usage(
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
        cached_input_tokens=3,
        reasoning_tokens=2,
    )
    assert semantic.terminal == TerminalMetadata(
        finish_reason="length",
        response_status="incomplete",
    )
    assert await runtime.encode_response(semantic) == payload


@pytest.mark.asyncio
async def test_safety_only_response_is_a_business_terminal() -> None:
    payload = {
        "promptFeedback": {
            "blockReason": "SAFETY",
            "blockReasonMessage": "blocked by policy",
            "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}],
        },
        "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3},
        "modelVersion": _MODEL,
    }

    semantic = await _runtime().decode_response(payload)

    assert semantic.output == ()
    assert semantic.terminal == TerminalMetadata(
        finish_reason="content_filter",
        response_status="incomplete",
        transport_termination="explicit_terminal",
        incomplete_details={"reason": "content_filter"},
    )
    prompt_feedback = semantic.metadata["gemini_prompt_feedback"]
    assert isinstance(prompt_feedback, Mapping)
    assert prompt_feedback["blockReason"] == "SAFETY"
    assert semantic.usage == Usage(input_tokens=3, total_tokens=3)


def test_stream_decoder_preserves_thought_signature_and_emits_one_terminal() -> None:
    decoder = _runtime().new_stream_decoder(sequence_start=10)

    first = decoder.decode(
        {
            "modelVersion": _MODEL,
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": "trace",
                                "thought": True,
                                "thoughtSignature": "native-signature",
                            }
                        ],
                    },
                    "index": 0,
                }
            ],
        }
    )
    final = decoder.decode(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "done"}]},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 3,
                "totalTokenCount": 5,
            },
        }
    )
    events = first + final

    assert [event.sequence for event in events] == list(range(10, 16))
    assert [event.type for event in events] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.REASONING_DELTA,
        SemanticEventType.OUTPUT_ITEM,
        SemanticEventType.TEXT_DELTA,
        SemanticEventType.USAGE,
        SemanticEventType.TERMINAL,
    ]
    opaque_item = events[2].item
    assert isinstance(opaque_item, ReasoningSummary)
    assert opaque_item.text == ""
    assert opaque_item.opaque_state is not None
    assert opaque_item.opaque_state.blob == "native-signature"
    assert decoder.finish_eof() == ()
    with pytest.raises(ProtocolDecodeError, match="after terminal"):
        decoder.decode({"candidates": []})


def test_safety_only_stream_frame_emits_one_business_terminal() -> None:
    decoder = _runtime().new_stream_decoder()

    events = decoder.decode(
        {
            "modelVersion": _MODEL,
            "promptFeedback": {
                "blockReason": "SAFETY",
                "blockReasonMessage": "blocked by policy",
            },
        }
    )

    assert [event.type for event in events] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.TERMINAL,
    ]
    assert events[-1].terminal == TerminalMetadata(
        finish_reason="content_filter",
        response_status="incomplete",
        transport_termination="explicit_terminal",
        incomplete_details={"reason": "content_filter"},
    )
    prompt_feedback = events[-1].metadata["gemini_prompt_feedback"]
    assert isinstance(prompt_feedback, Mapping)
    assert prompt_feedback["blockReason"] == "SAFETY"
    assert decoder.finish_eof() == ()


def test_stream_error_and_unexpected_eof_have_exactly_one_terminal() -> None:
    error_decoder = _runtime().new_stream_decoder()
    error_events = error_decoder.decode(
        {"error": {"code": 503, "status": "UNAVAILABLE", "message": "retry"}}
    )
    assert [event.type for event in error_events] == [
        SemanticEventType.ERROR,
        SemanticEventType.TERMINAL,
    ]
    assert error_decoder.finish_eof() == ()

    eof_events = _runtime().new_stream_decoder().finish_eof()
    assert [event.type for event in eof_events] == [
        SemanticEventType.ERROR,
        SemanticEventType.TERMINAL,
    ]
    assert eof_events[-1].terminal is not None
    assert eof_events[-1].terminal.error_code == "unexpected_eof"


def test_stream_encoder_uses_capsule_hook_and_rejects_post_terminal_event() -> None:
    state = OpaqueState(
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_model=_MODEL,
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

    encoder = _runtime(encode_opaque_state=encode_capsule).new_stream_encoder()
    assert (
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.RESPONSE_STARTED,
                metadata={"model": _MODEL},
            )
        )
        == ()
    )
    reasoning_frame = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.OUTPUT_ITEM,
            output_index=0,
            item=ReasoningSummary("trace", opaque_state=state),
        )
    )[0]
    terminal_frame = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
        )
    )[0]

    assert reasoning_frame["candidates"][0]["content"]["parts"][0] == {
        "text": "trace",
        "thought": True,
        "thoughtSignature": "rmr1.key.payload",
    }
    assert terminal_frame["candidates"][0]["finishReason"] == "STOP"
    assert calls == [(state, WireProtocol.GEMINI, _MODEL, "rs_1")]
    with pytest.raises(ProtocolRepresentabilityError, match="after terminal"):
        encoder.encode(SemanticEvent(type=SemanticEventType.TEXT_DELTA, delta="late"))


def test_stream_encoder_ignores_empty_output_lifecycle_events() -> None:
    encoder = _runtime().new_stream_encoder()

    assert (
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.RESPONSE_STARTED,
                metadata={"model": _MODEL},
            )
        )
        == ()
    )
    assert (
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.OUTPUT_ITEM,
                output_index=0,
                metadata={"output_item_type": "text"},
            )
        )
        == ()
    )
    text_frame = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TEXT_DELTA,
            output_index=0,
            delta="hello",
        )
    )
    assert (
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.OUTPUT_ITEM,
                output_index=0,
                metadata={"output_item_type": "text", "output_item_done": True},
            )
        )
        == ()
    )
    terminal_frame = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
        )
    )

    assert text_frame == (
        {
            "modelVersion": _MODEL,
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "hello"}]},
                    "index": 0,
                }
            ],
        },
    )
    assert terminal_frame[0]["candidates"][0]["finishReason"] == "STOP"
    with pytest.raises(ProtocolRepresentabilityError, match="after terminal"):
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.OUTPUT_ITEM,
                output_index=0,
                metadata={"output_item_done": True},
            )
        )


def test_stream_encoder_buffers_tool_deltas_until_gemini_can_emit_json_object() -> None:
    encoder = _runtime().new_stream_encoder()
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.RESPONSE_STARTED,
            metadata={"model": _MODEL},
        )
    )

    assert (
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.TOOL_ARGUMENTS_DELTA,
                output_index=2,
                item_id="call_1",
                delta="",
                metadata={"call_id": "call_1", "name": "lookup"},
            )
        )
        == ()
    )
    for delta in ('{"city"', ':"Shanghai"}'):
        assert (
            encoder.encode(
                SemanticEvent(
                    type=SemanticEventType.TOOL_ARGUMENTS_DELTA,
                    output_index=2,
                    delta=delta,
                )
            )
            == ()
        )

    call_frame, usage_frame = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.USAGE,
            usage=Usage(input_tokens=3, output_tokens=4, total_tokens=7),
        )
    )
    terminal_frame = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(
                finish_reason="tool_calls",
                response_status="completed",
            ),
        )
    )[0]

    assert call_frame == {
        "modelVersion": _MODEL,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "lookup",
                                "args": {"city": "Shanghai"},
                            }
                        }
                    ],
                },
                "index": 0,
            }
        ],
    }
    assert usage_frame["modelVersion"] == _MODEL
    assert usage_frame["usageMetadata"]["totalTokenCount"] == 7
    assert terminal_frame["modelVersion"] == _MODEL
    assert terminal_frame["candidates"][0]["finishReason"] == "STOP"


def test_stream_encoder_flushes_anthropic_tool_lifecycle_without_duplication() -> None:
    encoder = _runtime().new_stream_encoder()

    assert (
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.OUTPUT_ITEM,
                output_index=1,
                item_id="call_1",
                item=ToolCall(call_id="call_1", name="lookup"),
                metadata={"model": _MODEL},
            )
        )
        == ()
    )
    assert (
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.TOOL_ARGUMENTS_DELTA,
                output_index=1,
                item_id="call_1",
                delta='{"q":1}',
                metadata={"name": "lookup"},
            )
        )
        == ()
    )
    frames = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.OUTPUT_ITEM,
            output_index=1,
            item_id="call_1",
            metadata={
                "output_item_type": "tool_use",
                "output_item_done": True,
            },
        )
    )

    assert len(frames) == 1
    assert frames[0]["candidates"][0]["content"]["parts"] == [
        {
            "functionCall": {
                "id": "call_1",
                "name": "lookup",
                "args": {"q": 1},
            }
        }
    ]
    terminal = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
        )
    )
    assert len(terminal) == 1


def test_stream_encoder_rejects_incomplete_tool_json_at_terminal() -> None:
    encoder = _runtime().new_stream_encoder()
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TOOL_ARGUMENTS_DELTA,
            output_index=0,
            delta='{"city":"Shanghai',
            metadata={"model": _MODEL, "call_id": "call_1", "name": "lookup"},
        )
    )

    with pytest.raises(ProtocolRepresentabilityError, match="must contain valid JSON"):
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.TERMINAL,
                terminal=TerminalMetadata(
                    finish_reason="length",
                    response_status="incomplete",
                ),
            )
        )


def test_stream_encoder_still_rejects_unsupported_nonempty_output_item() -> None:
    encoder = _runtime().new_stream_encoder()

    with pytest.raises(ProtocolRepresentabilityError, match="unsupported output ToolResult"):
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.OUTPUT_ITEM,
                output_index=0,
                item=ToolResult(call_id="call_1", structured_content={"answer": 1}),
                metadata={"model": _MODEL},
            )
        )


def test_stream_encoder_eof_is_one_error_frame() -> None:
    encoder = _runtime().new_stream_encoder()

    assert encoder.finish_eof() == (
        {
            "error": {
                "code": "unexpected_eof",
                "message": "Semantic event stream ended before terminal event",
            }
        },
    )
    assert encoder.finish_eof() == ()


def test_stream_encoder_projects_cross_protocol_unexpected_eof_as_gemini_internal() -> None:
    encoder = _runtime().new_stream_encoder()

    assert encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(
                error_code="unexpected_eof",
                error_message="Upstream stream ended before an explicit terminal event",
                response_status="failed",
                transport_termination="unexpected_eof",
            ),
        )
    ) == (
        {
            "error": {
                "code": 502,
                "message": "Upstream stream ended before an explicit terminal event",
                "status": "INTERNAL",
                "details": [{"reason": "unexpected_eof"}],
            }
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "unknown"])
async def test_nonstream_rejects_non_success_terminal_without_error(status: str) -> None:
    response = SemanticResponse(
        id="response_1",
        model=_MODEL,
        output=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(TextContent("partial"),),
            ),
        ),
        terminal=TerminalMetadata(response_status=status),
    )

    with pytest.raises(ProtocolRepresentabilityError) as raised:
        await _runtime().encode_response(response)

    assert raised.value.path == "response.terminal.response_status"


@pytest.mark.parametrize(
    ("status", "code", "gemini_status", "reason", "message"),
    [
        ("failed", 502, "INTERNAL", "upstream_error", "Upstream response failed"),
        (
            "cancelled",
            499,
            "CANCELLED",
            "upstream_cancelled",
            "Upstream response was cancelled",
        ),
        (
            "unknown",
            502,
            "INTERNAL",
            "upstream_status_unknown",
            "Upstream response ended with an unknown status",
        ),
    ],
)
def test_stream_projects_non_success_terminal_without_error(
    status: str,
    code: int,
    gemini_status: str,
    reason: str,
    message: str,
) -> None:
    encoder = _runtime().new_stream_encoder()

    frames = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(response_status=status),
        )
    )

    assert frames == (
        {
            "error": {
                "code": code,
                "message": message,
                "status": gemini_status,
                "details": [{"reason": reason}],
            }
        },
    )
    assert encoder.finish_eof() == ()


@pytest.mark.asyncio
async def test_incomplete_terminal_keeps_max_tokens_projection() -> None:
    terminal = TerminalMetadata(finish_reason="length", response_status="incomplete")
    response = SemanticResponse(
        id="response_1",
        model=_MODEL,
        output=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(TextContent("partial"),),
            ),
        ),
        terminal=terminal,
    )

    encoded = await _runtime().encode_response(response)
    encoder = _runtime().new_stream_encoder()
    stream_frames = encoder.encode(
        SemanticEvent(type=SemanticEventType.TERMINAL, terminal=terminal)
    )

    assert encoded["candidates"][0]["finishReason"] == "MAX_TOKENS"
    assert len(stream_frames) == 1
    assert stream_frames[0]["candidates"][0]["finishReason"] == "MAX_TOKENS"
