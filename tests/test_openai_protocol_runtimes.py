"""Focused OpenAI Chat/Responses lazy-runtime contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest

from router_maestro.protocols import (
    FrozenJsonValue,
    GeminiRuntime,
    MessageRole,
    OpaqueState,
    OpenAIChatRuntime,
    OpenAIResponsesRuntime,
    ProtocolDecodeError,
    ProtocolRepresentabilityError,
    ReasoningConfig,
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
    chat_response_to_semantic,
    responses_response_to_semantic,
    semantic_to_chat_response,
    semantic_to_responses_response,
)
from router_maestro.providers.base import (
    ChatResponse,
    ResponsesResponse,
    ResponseStatus,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
)


def _responses_response(**overrides: object) -> dict:
    payload = {
        "id": "resp_1",
        "object": "response",
        "created_at": 10,
        "model": "gpt-example",
        "status": "completed",
        "output": [],
        "usage": {
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
        },
        "error": None,
        "incomplete_details": None,
    }
    payload.update(overrides)
    return payload


def _responses_message(text: str = "hello") -> dict:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


_RESPONSES_TOOL_CALLS = (
    {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{}",
        "status": "completed",
    },
    {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_1",
        "name": "shell",
        "input": "pwd",
        "status": "completed",
    },
    {
        "type": "tool_search_call",
        "id": "tsc_1",
        "call_id": "call_1",
        "name": "tool_search",
        "arguments": {},
        "execution": "client",
        "status": "completed",
    },
)


def test_responses_manifest_shallowly_finds_continuation_without_decoding() -> None:
    runtime = OpenAIResponsesRuntime()
    payload = {
        "model": "gpt-example",
        "previous_response_id": "resp_previous",
        "parallel_tool_calls": True,
        "input": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "rmr1.key.payload",
            },
            {
                "type": "reasoning",
                "id": "rs_2",
                "summary": [],
                "encrypted_content": "rmr12.key.payload",
            },
            {
                "type": "reasoning",
                "id": "rs_native",
                "summary": [],
                "encrypted_content": "rmr2x.native",
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "rmr1.not-a-top-level-reasoning-carrier",
                    }
                ],
            },
        ],
    }

    manifest = runtime.inspect_request(payload)

    assert manifest.model == "gpt-example"
    assert manifest.reasoning is True
    assert manifest.parallel_tools is True
    assert manifest.reasoning_capsules == ("rmr1.key.payload", "rmr12.key.payload")
    assert manifest.previous_response_id == "resp_previous"
    assert manifest.opaque_continuation is True


def test_chat_manifest_shallowly_finds_explicit_parallel_tools() -> None:
    runtime = OpenAIChatRuntime()

    manifest = runtime.inspect_request(
        {
            "model": "gpt-example",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "parallel_tool_calls": True,
        }
    )

    assert manifest.tools is True
    assert manifest.parallel_tools is True


@pytest.mark.asyncio
async def test_chat_tool_result_error_projection_round_trips() -> None:
    runtime = OpenAIChatRuntime()
    semantic = SemanticRequest(
        model="gpt-example",
        input=(
            SemanticMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResult(
                        call_id="call_1",
                        content=(TextContent("command failed"),),
                        is_error=True,
                    ),
                ),
            ),
        ),
    )

    encoded = await runtime.encode_request(semantic)
    tool_message = encoded["messages"][0]
    assert json.loads(tool_message["content"]) == {
        "$router_maestro": {"type": "tool_result", "version": 1},
        "is_error": True,
        "output": "command failed",
    }

    decoded = await runtime.decode_request(encoded)
    message = decoded.input[0]
    assert isinstance(message, SemanticMessage)
    result = message.content[0]
    assert isinstance(result, ToolResult)
    assert result.content == (TextContent("command failed"),)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_chat_tool_result_projection_escapes_literal_envelopes() -> None:
    runtime = OpenAIChatRuntime()
    literal = json.dumps(
        {
            "$router_maestro": {"type": "tool_result", "version": 1},
            "is_error": True,
            "output": "literal",
        },
        separators=(",", ":"),
    )
    semantic = SemanticRequest(
        model="gpt-example",
        input=(
            SemanticMessage(
                role=MessageRole.TOOL,
                content=(ToolResult(call_id="call_1", content=(TextContent(literal),)),),
            ),
        ),
    )

    encoded = await runtime.encode_request(semantic)
    projected = json.loads(encoded["messages"][0]["content"])
    assert projected == {
        "$router_maestro": {"type": "tool_result", "version": 1},
        "is_error": False,
        "output": literal,
    }

    decoded = await runtime.decode_request(encoded)
    message = decoded.input[0]
    assert isinstance(message, SemanticMessage)
    result = message.content[0]
    assert isinstance(result, ToolResult)
    assert result.content == (TextContent(literal),)
    assert result.is_error is False


@pytest.mark.asyncio
async def test_chat_tool_result_projection_rejects_unknown_version() -> None:
    runtime = OpenAIChatRuntime()
    output = json.dumps(
        {
            "$router_maestro": {"type": "tool_result", "version": 2},
            "is_error": True,
            "output": "bad",
        },
        separators=(",", ":"),
    )

    with pytest.raises(ProtocolDecodeError) as raised:
        await runtime.decode_request(
            {
                "model": "gpt-example",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": output,
                    }
                ],
            }
        )

    assert raised.value.path == "messages[0].content"


@pytest.mark.asyncio
async def test_previous_response_id_is_identity_only_with_exact_error_path() -> None:
    runtime = OpenAIResponsesRuntime()

    with pytest.raises(ProtocolDecodeError) as raised:
        await runtime.decode_request(
            {
                "model": "gpt-example",
                "input": "hello",
                "previous_response_id": "resp_previous",
            }
        )

    assert raised.value.path == "previous_response_id"


@pytest.mark.asyncio
async def test_chat_and_responses_wire_response_round_trip() -> None:
    chat_payload = {
        "id": "chat_1",
        "object": "chat.completion",
        "created": 10,
        "model": "gpt-example",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    chat_runtime = OpenAIChatRuntime()
    chat_semantic = await chat_runtime.decode_response(chat_payload)
    assert await chat_runtime.encode_response(chat_semantic) == chat_payload

    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "plan"}],
        "encrypted_content": "opaque",
        "future_sibling": {"nested": [1, 2]},
    }
    responses_payload = _responses_response(output=[reasoning])
    responses_runtime = OpenAIResponsesRuntime(
        provider_name="github-copilot",
        binding_id="copilot-responses",
    )
    responses_semantic = await responses_runtime.decode_response(responses_payload)
    preserved = responses_semantic.output[0]
    assert isinstance(preserved, ReasoningSummary)
    assert preserved.opaque_state is not None
    assert isinstance(preserved.opaque_state.blob, Mapping)
    assert preserved.opaque_state.blob["future_sibling"] == {"nested": (1, 2)}
    assert await responses_runtime.encode_response(responses_semantic) == responses_payload


@pytest.mark.asyncio
async def test_responses_output_text_accepts_empty_standard_logprobs_only() -> None:
    runtime = OpenAIResponsesRuntime()
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "hello",
                "annotations": [],
                "logprobs": [],
            }
        ],
    }

    semantic = await runtime.decode_response(_responses_response(output=[message]))

    assert semantic.output == (
        SemanticMessage(
            role=MessageRole.ASSISTANT,
            content=(TextContent("hello"),),
            item_id="msg_1",
            status="completed",
        ),
    )
    message["content"][0]["logprobs"] = [{"token": "hello"}]
    with pytest.raises(ProtocolDecodeError) as raised:
        await runtime.decode_response(_responses_response(output=[message]))
    assert raised.value.path == "response.output[0].content[0].logprobs"


@pytest.mark.asyncio
async def test_chat_response_accepts_declared_standard_metadata_but_rejects_unknowns() -> None:
    runtime = OpenAIChatRuntime()
    payload = {
        "id": "chat_1",
        "object": "chat.completion",
        "created": 10,
        "model": "gpt-example",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "system_fingerprint": "fp_example",
        "service_tier": "default",
    }

    semantic = await runtime.decode_response(payload)

    assert semantic.output == (
        SemanticMessage(role=MessageRole.ASSISTANT, content=(TextContent("hello"),)),
    )
    assert semantic.usage == Usage(
        input_tokens=2,
        output_tokens=1,
        total_tokens=3,
        cached_input_tokens=0,
    )
    with pytest.raises(ProtocolDecodeError) as raised:
        await runtime.decode_response({**payload, "future_response_field": True})
    assert raised.value.path == "response.future_response_field"

    invalid_usage = {
        **payload,
        "usage": {
            **payload["usage"],
            "completion_tokens_details": {"future_tokens": 1},
        },
    }
    with pytest.raises(ProtocolDecodeError) as usage_raised:
        await runtime.decode_response(invalid_usage)
    assert usage_raised.value.path == "usage.completion_tokens_details.future_tokens"


@pytest.mark.asyncio
async def test_chat_response_decodes_copilot_reasoning_aliases_without_losing_usage() -> None:
    payload = {
        "id": "chat_1",
        "object": "chat.completion",
        "created": 10,
        "model": "gemini-3.7-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "pong",
                    "reasoning_text": "model reasoning",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 16,
            "completion_tokens": 1,
            "total_tokens": 158,
            "prompt_tokens_details": {"cached_tokens": 0},
            "reasoning_tokens": 141,
        },
    }

    semantic = await OpenAIChatRuntime().decode_response(payload)

    assert semantic.output == (
        SemanticMessage(
            role=MessageRole.ASSISTANT,
            content=(TextContent("pong"), ReasoningSummary("model reasoning")),
        ),
    )
    assert semantic.usage == Usage(
        input_tokens=16,
        output_tokens=1,
        total_tokens=158,
        cached_input_tokens=0,
        reasoning_tokens=141,
    )


@pytest.mark.asyncio
async def test_responses_response_accepts_only_declared_nonsemantic_echo_fields() -> None:
    runtime = OpenAIResponsesRuntime()
    payload = _responses_response(
        instructions="Be concise",
        max_output_tokens=64,
        output_text="hello",
        parallel_tool_calls=True,
        previous_response_id=None,
        prompt_cache_retention=None,
        reasoning={"effort": "low", "summary": "auto"},
        safety_identifier="safe-id",
        service_tier="default",
        temperature=None,
        text={"format": {"type": "text"}},
        tool_choice="auto",
        tools=[],
        top_p=1,
        truncation="disabled",
        output=[
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }
        ],
    )

    semantic = await runtime.decode_response(payload)
    encoded = await runtime.encode_response(semantic)

    assert isinstance(semantic.output[0], SemanticMessage)
    assert semantic.output[0].content == (TextContent("hello"),)
    assert not set(payload) - {
        "instructions",
        "max_output_tokens",
        "output_text",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_p",
        "truncation",
        *encoded,
    }
    assert "phase" not in encoded["output"][0]

    with pytest.raises(ProtocolDecodeError) as raised:
        await runtime.decode_response({**payload, "future_response_field": True})
    assert raised.value.path == "response.future_response_field"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_call",
    _RESPONSES_TOOL_CALLS,
    ids=("function", "custom", "tool-search"),
)
async def test_responses_completed_tool_output_uses_tool_calls_finish_reason(
    tool_call: dict,
) -> None:
    semantic = await OpenAIResponsesRuntime().decode_response(
        _responses_response(output=[tool_call])
    )

    assert semantic.terminal is not None
    assert semantic.terminal.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_responses_completed_text_output_uses_stop_finish_reason() -> None:
    semantic = await OpenAIResponsesRuntime().decode_response(
        _responses_response(output=[_responses_message()])
    )

    assert semantic.terminal is not None
    assert semantic.terminal.finish_reason == "stop"


@pytest.mark.asyncio
async def test_responses_request_maps_anthropic_budget_to_effort_tier() -> None:
    runtime = OpenAIResponsesRuntime()
    request = SemanticRequest(
        model="gpt-example",
        input=(SemanticMessage(role=MessageRole.USER, content=(TextContent("hello"),)),),
        reasoning=ReasoningConfig(enabled=True, budget_tokens=1024),
    )

    payload = await runtime.encode_request(request)

    assert payload["reasoning"] == {"effort": "low"}

    with pytest.raises(ProtocolRepresentabilityError) as raised:
        await runtime.encode_request(
            replace(request, reasoning=ReasoningConfig(enabled=True, budget_tokens=100))
        )
    assert raised.value.path == "reasoning.budget_tokens"


@pytest.mark.asyncio
async def test_chat_response_projects_responses_message_id_and_status() -> None:
    responses_runtime = OpenAIResponsesRuntime()
    semantic = await responses_runtime.decode_response(
        {
            "id": "resp_1",
            "model": "gpt-example",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            "error": None,
            "incomplete_details": None,
        }
    )

    encoded = await OpenAIChatRuntime().encode_response(semantic)

    assert encoded["id"] == "resp_1"
    assert encoded["model"] == "gpt-example"
    assert encoded["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ]
    assert encoded["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }


@pytest.mark.asyncio
async def test_real_openai_envelope_metadata_projects_across_response_protocols() -> None:
    chat_payload = {
        "id": "chat_1",
        "object": "chat.completion",
        "created": 10,
        "model": "gpt-example",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    chat_semantic = await OpenAIChatRuntime().decode_response(chat_payload)

    responses_projection = await OpenAIResponsesRuntime().encode_response(chat_semantic)

    assert responses_projection["object"] == "response"
    assert responses_projection["created_at"] == 10
    assert responses_projection["status"] == "completed"

    responses_semantic = await OpenAIResponsesRuntime().decode_response(
        _responses_response(output=[_responses_message()])
    )

    chat_projection = await OpenAIChatRuntime().encode_response(responses_semantic)

    assert chat_projection["object"] == "chat.completion"
    assert chat_projection["created"] == 10
    assert chat_projection["choices"][0]["finish_reason"] == "stop"

    gemini_from_chat = await GeminiRuntime().encode_response(chat_semantic)
    gemini_from_responses = await GeminiRuntime().encode_response(responses_semantic)
    assert gemini_from_chat["modelVersion"] == "gpt-example"
    assert gemini_from_responses["modelVersion"] == "gpt-example"
    assert gemini_from_chat["candidates"][0]["finishReason"] == "STOP"
    assert gemini_from_responses["candidates"][0]["finishReason"] == "STOP"


@pytest.mark.asyncio
async def test_chat_response_message_projection_still_rejects_opaque_reasoning() -> None:
    state = OpaqueState(
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_model="gpt-example",
        origin_binding="copilot-responses",
        item_id="rs_1",
        blob={"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
    )
    response = SemanticResponse(
        id="resp_1",
        model="gpt-example",
        output=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(ReasoningSummary("plan", opaque_state=state),),
                item_id="msg_1",
                status="completed",
            ),
        ),
    )

    with pytest.raises(ProtocolRepresentabilityError, match="opaque reasoning state"):
        await OpenAIChatRuntime().encode_response(response)


@pytest.mark.asyncio
async def test_chat_response_message_projection_round_trips_tool_namespace() -> None:
    response = SemanticResponse(
        id="resp_1",
        model="gpt-example",
        output=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(
                    ToolCall(
                        call_id="call_1",
                        name="lookup",
                        arguments={"query": "rm"},
                        namespace="mcp",
                    ),
                ),
                item_id="msg_1",
                status="completed",
            ),
        ),
    )

    runtime = OpenAIChatRuntime()
    encoded = await runtime.encode_response(response)
    raw_name = encoded["choices"][0]["message"]["tool_calls"][0]["function"]["name"]

    assert raw_name != "lookup"
    decoded = await runtime.decode_response(encoded)
    decoded_message = cast(SemanticMessage, decoded.output[0])
    call = decoded_message.content[0]
    assert isinstance(call, ToolCall)
    assert (call.namespace, call.name) == ("mcp", "lookup")


@pytest.mark.asyncio
async def test_responses_codex_controls_and_namespace_encode_to_chat() -> None:
    responses = OpenAIResponsesRuntime()
    semantic = await responses.decode_request(
        {
            "model": "gemini-test",
            "input": [
                {"role": "developer", "content": "Follow project instructions."},
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "prompt_cache_key": "cache-key",
            "client_metadata": {"thread_id": "thread-1"},
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__qmd",
                    "description": "QMD tools",
                    "tools": [
                        {
                            "type": "function",
                            "name": "status",
                            "description": "Get status",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert semantic.metadata["prompt_cache_key"] == "cache-key"
    assert semantic.metadata["client_metadata"] == {"thread_id": "thread-1"}
    assert cast(SemanticMessage, semantic.input[0]).role is MessageRole.SYSTEM
    assert semantic.tools[0].namespace == "mcp__qmd"

    chat = await OpenAIChatRuntime().encode_request(semantic)
    encoded_name = chat["tools"][0]["function"]["name"]
    assert encoded_name != "status"

    decoded_chat = await OpenAIChatRuntime().decode_request(chat)
    assert (decoded_chat.tools[0].namespace, decoded_chat.tools[0].name) == (
        "mcp__qmd",
        "status",
    )


@pytest.mark.asyncio
async def test_copilot_chat_reasoning_opaque_round_trips_for_bound_runtime() -> None:
    runtime = OpenAIChatRuntime(
        origin_provider="github-copilot",
        origin_binding="copilot-openai-chat",
        default_model="gemini-test",
        allow_reasoning_opaque=True,
    )
    semantic = await runtime.decode_response(
        {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "gemini-test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_text": "plan",
                        "reasoning_opaque": "provider-state",
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": None,
        }
    )
    response_message = cast(SemanticMessage, semantic.output[0])
    reasoning = next(
        part for part in response_message.content if isinstance(part, ReasoningSummary)
    )
    assert isinstance(reasoning, ReasoningSummary)
    assert reasoning.opaque_state is not None
    assert reasoning.opaque_state.origin_binding == "copilot-openai-chat"

    request = SemanticRequest(
        model="gemini-test",
        input=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(reasoning,),
            ),
        ),
    )
    replay = await runtime.encode_request(request)
    assert replay["messages"][0]["reasoning_opaque"] == "provider-state"
    assert replay["messages"][0]["reasoning_text"] == "plan"


@pytest.mark.asyncio
async def test_chat_response_message_projection_does_not_drop_semantic_name() -> None:
    response = SemanticResponse(
        id="resp_1",
        model="gpt-example",
        output=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(TextContent("hello"),),
                name="named-assistant",
                item_id="msg_1",
                status="completed",
            ),
        ),
    )

    with pytest.raises(ProtocolRepresentabilityError, match="semantic name"):
        await OpenAIChatRuntime().encode_response(response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_provider", "target_binding", "target_model", "path_suffix"),
    [
        ("other", "copilot-responses", "gpt-example", "origin_provider"),
        ("github-copilot", "other-binding", "gpt-example", "origin_binding"),
        ("github-copilot", "copilot-responses", "other-model", "origin_model"),
    ],
)
async def test_responses_opaque_replay_requires_exact_provenance(
    target_provider: str,
    target_binding: str,
    target_model: str,
    path_suffix: str,
) -> None:
    state = OpaqueState(
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_model="gpt-example",
        origin_binding="copilot-responses",
        item_id="rs_1",
        blob=cast(
            FrozenJsonValue,
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "encrypted_content": "opaque",
            },
        ),
    )
    request = SemanticRequest(
        model=target_model,
        input=(ReasoningSummary("plan", opaque_state=state),),
    )
    runtime = OpenAIResponsesRuntime(
        provider_name=target_provider,
        binding_id=target_binding,
    )

    with pytest.raises(ProtocolRepresentabilityError) as raised:
        await runtime.encode_request(request)

    assert raised.value.path.endswith(path_suffix)


@pytest.mark.asyncio
async def test_malformed_responses_arguments_report_exact_path() -> None:
    runtime = OpenAIResponsesRuntime()

    with pytest.raises(ProtocolDecodeError) as raised:
        await runtime.decode_request(
            {
                "model": "gpt-example",
                "input": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": "{",
                    }
                ],
            }
        )

    assert raised.value.path == "input[0].arguments"


def test_legacy_chat_rejects_non_chat_or_structured_opaque_state() -> None:
    state = OpaqueState(
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_model="gpt-example",
        item_id="rs_1",
        blob=cast(
            FrozenJsonValue,
            {"type": "reasoning", "id": "rs_1", "summary": []},
        ),
    )
    response = SemanticResponse(
        id="chat_1",
        model="gpt-example",
        output=(ReasoningSummary("plan", opaque_state=state),),
        terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
    )

    with pytest.raises(ProtocolRepresentabilityError, match="Chat-origin"):
        semantic_to_chat_response(response)


def test_chat_stream_runtime_orders_payload_usage_and_terminal() -> None:
    decoder = OpenAIChatRuntime().new_stream_decoder(sequence_start=4)
    events = decoder.decode(
        {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "created": 10,
            "model": "gpt-example",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        }
    )

    assert [event.type for event in events] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.TEXT_DELTA,
        SemanticEventType.USAGE,
        SemanticEventType.TERMINAL,
    ]
    assert [event.sequence for event in events] == [4, 5, 6, 7]
    assert events[-1].terminal is not None
    assert events[-1].terminal.transport_termination == "explicit_terminal"
    assert decoder.finish_eof() == ()


def test_chat_stream_runtime_allows_one_usage_only_tail_after_terminal() -> None:
    decoder = OpenAIChatRuntime(default_model="gpt-example").new_stream_decoder(sequence_start=4)
    terminal = decoder.decode(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    usage = decoder.decode(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        }
    )

    assert [event.type for event in terminal] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.TEXT_DELTA,
        SemanticEventType.TERMINAL,
    ]
    assert [event.type for event in usage] == [SemanticEventType.USAGE]
    assert usage[0].sequence == 7
    assert usage[0].usage == Usage(input_tokens=2, output_tokens=1, total_tokens=3)
    assert decoder.finish_eof() == ()


def test_chat_stream_runtime_rejects_non_usage_frame_after_terminal() -> None:
    decoder = OpenAIChatRuntime(default_model="gpt-example").new_stream_decoder()
    decoder.decode({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})

    with pytest.raises(ProtocolDecodeError, match="after terminal"):
        decoder.decode(
            {"choices": [{"index": 0, "delta": {"content": "late"}, "finish_reason": None}]}
        )


def test_chat_stream_runtime_rejects_second_usage_tail_after_terminal() -> None:
    decoder = OpenAIChatRuntime(default_model="gpt-example").new_stream_decoder()
    decoder.decode({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    usage = {
        "choices": [],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    decoder.decode(usage)

    with pytest.raises(ProtocolDecodeError, match="after terminal"):
        decoder.decode(usage)


def test_chat_stream_runtime_rejects_usage_tail_when_terminal_already_has_usage() -> None:
    decoder = OpenAIChatRuntime(default_model="gpt-example").new_stream_decoder()
    decoder.decode(
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
    )

    with pytest.raises(ProtocolDecodeError, match="after terminal"):
        decoder.decode(
            {
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 999},
            }
        )


def test_chat_stream_runtime_rejects_usage_tail_after_error_terminal() -> None:
    decoder = OpenAIChatRuntime(default_model="gpt-example").new_stream_decoder()
    decoder.decode({"error": {"code": "upstream_error", "message": "failed"}})

    with pytest.raises(ProtocolDecodeError, match="after terminal"):
        decoder.decode(
            {
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )


def test_chat_stream_runtime_eof_closes_usage_tail_window() -> None:
    decoder = OpenAIChatRuntime(default_model="gpt-example").new_stream_decoder()
    decoder.decode({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    assert decoder.finish_eof() == ()

    with pytest.raises(ProtocolDecodeError, match="after terminal"):
        decoder.decode(
            {
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )


def test_chat_stream_runtime_decodes_copilot_reasoning_and_final_usage_aliases() -> None:
    decoder = OpenAIChatRuntime(
        origin_provider="github-copilot",
        default_model="gemini-3.7-flash",
    ).new_stream_decoder()

    reasoning = decoder.decode(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_text": "model reasoning",
                    },
                    "finish_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )
    final = decoder.decode(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 16,
                "completion_tokens": 1,
                "total_tokens": 158,
                "reasoning_tokens": 141,
            },
        }
    )

    assert [event.type for event in reasoning] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.REASONING_DELTA,
        SemanticEventType.USAGE,
    ]
    assert final[0].type is SemanticEventType.USAGE
    assert final[0].usage == Usage(
        input_tokens=16,
        output_tokens=1,
        total_tokens=158,
        reasoning_tokens=141,
    )


def test_chat_stream_provider_context_supplies_omitted_wire_identity() -> None:
    decoder = OpenAIChatRuntime(
        origin_provider="github-copilot",
        default_model="gpt-example",
    ).new_stream_decoder()

    events = decoder.decode(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_1",
                                "function": {"name": "lookup"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )

    assert [event.type for event in events] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.TOOL_ARGUMENTS_DELTA,
    ]
    assert events[0].metadata["model"] == "gpt-example"
    assert events[1].delta == ""
    assert events[1].metadata == {"call_id": "toolu_1", "name": "lookup"}

    with pytest.raises(ProtocolDecodeError, match="stream.id"):
        OpenAIChatRuntime().new_stream_decoder().decode(
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        )


def test_responses_stream_runtime_preserves_reasoning_state_and_terminal_order() -> None:
    runtime = OpenAIResponsesRuntime(
        provider_name="github-copilot",
        binding_id="copilot-responses",
    )
    decoder = runtime.new_stream_decoder(sequence_start=20)
    created = decoder.decode(
        {
            "type": "response.created",
            "response": _responses_response(status="in_progress", usage=None),
        }
    )
    reasoning_delta = decoder.decode(
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "plan",
        }
    )
    raw_reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "plan"}],
        "encrypted_content": "opaque",
        "future_sibling": True,
    }
    reasoning_done = decoder.decode(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": raw_reasoning,
        }
    )
    text = decoder.decode(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 1,
            "content_index": 0,
            "delta": "answer",
        }
    )
    completed = decoder.decode(
        {
            "type": "response.completed",
            "response": _responses_response(output=[raw_reasoning]),
        }
    )
    events = created + reasoning_delta + reasoning_done + text + completed

    assert [event.sequence for event in events] == list(range(20, 26))
    assert events[-2].type is SemanticEventType.USAGE
    assert events[-1].type is SemanticEventType.TERMINAL
    state_event = reasoning_done[0]
    assert isinstance(state_event.item, ReasoningSummary)
    assert state_event.item_id == "rs_1"
    assert state_event.item.opaque_state is not None
    assert state_event.item.opaque_state.origin_provider == "github-copilot"
    assert state_event.item.opaque_state.origin_binding == "copilot-responses"
    opaque_blob = state_event.item.opaque_state.blob
    assert isinstance(opaque_blob, Mapping)
    assert opaque_blob["future_sibling"] is True
    assert decoder.finish_eof() == ()


@pytest.mark.parametrize(
    ("output", "expected_finish_reason"),
    [
        ([_RESPONSES_TOOL_CALLS[0]], "tool_calls"),
        ([_responses_message()], "stop"),
    ],
    ids=("tool-call", "text"),
)
def test_responses_stream_completed_derives_finish_reason_from_output(
    output: list[dict],
    expected_finish_reason: str,
) -> None:
    decoder = OpenAIResponsesRuntime().new_stream_decoder()

    events = decoder.decode(
        {
            "type": "response.completed",
            "response": _responses_response(output=output),
        }
    )

    assert events[-1].type is SemanticEventType.TERMINAL
    assert events[-1].terminal is not None
    assert events[-1].terminal.finish_reason == expected_finish_reason


def test_responses_stream_rejects_changed_response_id_by_default() -> None:
    decoder = OpenAIResponsesRuntime().new_stream_decoder()
    decoder.decode(
        {
            "type": "response.created",
            "response": _responses_response(id="resp_first", status="in_progress", usage=None),
        }
    )

    with pytest.raises(ProtocolDecodeError) as raised:
        decoder.decode(
            {
                "type": "response.in_progress",
                "response": _responses_response(
                    id="resp_changed",
                    status="in_progress",
                    usage=None,
                ),
            }
        )

    assert raised.value.path == "stream.response.id"


def test_responses_stream_can_correlate_provider_obfuscated_envelope_ids() -> None:
    decoder = OpenAIResponsesRuntime(
        provider_name="github-copilot",
        binding_id="copilot-openai-responses",
        allow_per_event_response_ids=True,
    ).new_stream_decoder()
    created = decoder.decode(
        {
            "type": "response.created",
            "response": _responses_response(id="opaque_first", status="in_progress", usage=None),
        }
    )
    in_progress = decoder.decode(
        {
            "type": "response.in_progress",
            "response": _responses_response(
                id="opaque_second",
                status="in_progress",
                usage=None,
            ),
        }
    )
    completed = decoder.decode(
        {
            "type": "response.completed",
            "response": _responses_response(id="opaque_terminal"),
        }
    )

    assert in_progress == ()
    assert {event.response_id for event in created + completed} == {"opaque_first"}
    assert completed[-1].type is SemanticEventType.TERMINAL


def test_responses_stream_preserves_validated_content_part_lifecycle_order() -> None:
    decoder = OpenAIResponsesRuntime().new_stream_decoder(sequence_start=3)
    started = decoder.decode(
        {
            "type": "response.created",
            "response": _responses_response(status="in_progress", usage=None),
        }
    )
    added = decoder.decode(
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            },
            "sequence_number": 4,
        }
    )
    done = decoder.decode(
        {
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "hello",
                "annotations": [],
                "logprobs": [],
            },
            "sequence_number": 8,
        }
    )

    assert [event.sequence for event in started + added + done] == [3, 4, 5, 6]
    assert added[0].item is None
    assert added[0].item_id == "msg_1"
    assert added[0].metadata["content_part_added"] is True
    assert done[0].type is SemanticEventType.TEXT_DELTA
    assert done[0].item_id == "msg_1"
    assert done[0].delta == "hello"
    assert done[1].item_id == "msg_1"
    assert done[1].metadata["content_part_done"] is True
    content_part = done[1].metadata["content_part"]
    assert isinstance(content_part, Mapping)
    assert content_part["text"] == "hello"
    assert done[1].metadata["provenance_only"] is True


def test_responses_stream_content_part_lifecycle_rejects_unmodeled_fields() -> None:
    decoder = OpenAIResponsesRuntime().new_stream_decoder()

    with pytest.raises(ProtocolDecodeError) as raised:
        decoder.decode(
            {
                "type": "response.content_part.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": [{"type": "url_citation"}],
                },
            }
        )

    assert raised.value.path == "stream.part.annotations"


def test_responses_stream_reasoning_added_is_provenance_until_done() -> None:
    decoder = OpenAIResponsesRuntime(
        provider_name="github-copilot",
        binding_id="copilot-openai-responses",
        defer_intermediate_item_ids=True,
    ).new_stream_decoder()
    decoder.decode(
        {
            "type": "response.created",
            "response": _responses_response(status="in_progress", usage=None),
        }
    )
    added = decoder.decode(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "ephemeral",
                "summary": [],
                "encrypted_content": "ephemeral-state",
            },
        }
    )
    delta = decoder.decode(
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "ephemeral-delta",
            "output_index": 0,
            "summary_index": 0,
            "delta": "plan",
        }
    )
    summary_done = decoder.decode(
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "ephemeral-summary-done",
            "output_index": 0,
            "summary_index": 0,
            "text": "plan",
        }
    )
    done = decoder.decode(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "canonical",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "encrypted_content": "canonical-state",
            },
        }
    )

    assert added[0].item is None
    assert added[0].item_id is None
    assert added[0].metadata["provenance_only"] is True
    assert delta[0].type is SemanticEventType.REASONING_DELTA
    assert delta[0].item_id is None
    assert summary_done[0].item_id is None
    assert summary_done[0].metadata["content_part_done"] is True
    assert isinstance(done[0].item, ReasoningSummary)
    assert done[0].item_id == "canonical"
    assert done[0].item.opaque_state is not None
    assert done[0].item.opaque_state.item_id == "canonical"


@pytest.mark.parametrize(
    ("frame", "path"),
    [
        (
            {
                "type": "response.function_call_arguments.delta",
                "item_id": 7,
                "output_index": 0,
                "delta": "{}",
            },
            "stream.item_id",
        ),
        (
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": 7,
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "",
                },
            },
            "stream.item.id",
        ),
    ],
)
def test_responses_stream_deferred_item_ids_still_validate_wire_types(
    frame: dict[str, object],
    path: str,
) -> None:
    decoder = OpenAIResponsesRuntime(
        provider_name="github-copilot",
        binding_id="copilot-openai-responses",
        defer_intermediate_item_ids=True,
    ).new_stream_decoder()

    with pytest.raises(ProtocolDecodeError) as raised:
        decoder.decode(frame)

    assert raised.value.path == path


def test_responses_stream_message_snapshots_emit_only_missing_suffix_before_terminal() -> None:
    decoder = OpenAIResponsesRuntime().new_stream_decoder()
    response = _responses_response(status="in_progress", usage=None)
    decoder.decode({"type": "response.created", "response": response})
    decoder.decode(
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }
    )
    delta = decoder.decode(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "he",
        }
    )
    text_done = decoder.decode(
        {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": "hello",
        }
    )
    part_done = decoder.decode(
        {
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "hello", "annotations": []},
        }
    )
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
    }
    item_done = decoder.decode(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": message,
        }
    )
    terminal = decoder.decode(
        {
            "type": "response.completed",
            "response": _responses_response(output=[message]),
        }
    )

    text_events = [
        event
        for event in delta + text_done + part_done + item_done
        if event.type is SemanticEventType.TEXT_DELTA
    ]
    assert "".join(event.delta or "" for event in text_events) == "hello"
    assert item_done[-1].item is None
    assert item_done[-1].metadata["output_item_done"] is True
    assert [event.type for event in terminal] == [
        SemanticEventType.USAGE,
        SemanticEventType.TERMINAL,
    ]


def test_responses_stream_rejects_conflicting_message_snapshot() -> None:
    decoder = OpenAIResponsesRuntime().new_stream_decoder()
    decoder.decode(
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
        }
    )

    with pytest.raises(ProtocolDecodeError) as raised:
        decoder.decode(
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "goodbye",
            }
        )

    assert raised.value.path == "stream.text"


@pytest.mark.parametrize("runtime", [OpenAIChatRuntime(), OpenAIResponsesRuntime()])
def test_openai_stream_decoders_emit_one_error_terminal_pair_on_eof(runtime) -> None:
    decoder = runtime.new_stream_decoder()

    events = decoder.finish_eof()

    assert [event.type for event in events] == [
        SemanticEventType.ERROR,
        SemanticEventType.TERMINAL,
    ]
    assert events[-1].terminal is not None
    assert events[-1].terminal.transport_termination == "unexpected_eof"
    assert decoder.finish_eof() == ()


def test_openai_stream_encoders_finish_eof_once() -> None:
    chat = OpenAIChatRuntime().new_stream_encoder(model="gpt-example", response_id="chat_1")
    responses = OpenAIResponsesRuntime().new_stream_encoder(
        model="gpt-example", response_id="resp_1"
    )

    assert chat.finish_eof()[0]["error"]["code"] == "unexpected_eof"
    assert responses.finish_eof()[0]["type"] == "error"
    assert chat.finish_eof() == ()
    assert responses.finish_eof() == ()


def test_legacy_nonstream_terminal_outcome_round_trips_fully() -> None:
    outcome = TerminalOutcome(
        transport=TransportTermination.UNEXPECTED_EOF,
        response_status=ResponseStatus.UNKNOWN,
        incomplete_details=None,
        error=TerminalError(code="unexpected_eof", message="truncated"),
    )
    chat = ChatResponse(
        content="partial",
        model="gpt-example",
        finish_reason=None,  # type: ignore[arg-type]
        terminal_outcome=outcome,
    )
    chat_semantic = chat_response_to_semantic(
        chat,
        response_id="chat_1",
        origin_provider="github-copilot",
    )

    assert chat_semantic.terminal is not None
    assert chat_semantic.terminal.transport_termination == "unexpected_eof"
    assert semantic_to_chat_response(chat_semantic).terminal_outcome == outcome

    responses = ResponsesResponse(
        content="partial",
        model="gpt-example",
        finish_reason=None,
        terminal_outcome=outcome,
    )
    responses_semantic = responses_response_to_semantic(
        responses,
        response_id="resp_1",
        origin_provider="github-copilot",
        origin_binding="copilot-responses",
    )

    assert responses_semantic.terminal is not None
    assert responses_semantic.terminal.transport_termination == "unexpected_eof"
    assert semantic_to_responses_response(responses_semantic).terminal_outcome == outcome


def test_terminal_metadata_freezes_incomplete_details() -> None:
    details = {"reason": "vendor_limit", "nested": {"values": [1]}}
    terminal = TerminalMetadata(
        response_status="incomplete",
        transport_termination="explicit_terminal",
        incomplete_details=details,
    )
    details["nested"]["values"].append(2)

    assert terminal.incomplete_details == {
        "reason": "vendor_limit",
        "nested": {"values": (1,)},
    }
    assert isinstance(terminal.incomplete_details, Mapping)


def test_responses_stream_encoder_emits_usage_on_terminal_frame() -> None:
    encoder = OpenAIResponsesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="resp_1",
    )
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.RESPONSE_STARTED,
            response_id="resp_1",
            metadata={"model": "gpt-example"},
        )
    )
    assert (
        encoder.encode(
            SemanticEvent(
                type=SemanticEventType.USAGE,
                usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
            )
        )
        == ()
    )
    terminal = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(
                finish_reason="stop",
                response_status="completed",
                transport_termination="explicit_terminal",
            ),
        )
    )

    assert terminal[0]["type"] == "response.completed"
    assert terminal[0]["response"]["usage"]["total_tokens"] == 3


def test_responses_stream_encoder_opens_tool_item_from_chat_delta() -> None:
    encoder = OpenAIResponsesRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="resp_1",
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

    assert [frame["type"] for frame in first] == [
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.delta",
    ]
    assert first[1]["item"]["name"] == "get_weather"
    assert second[0]["type"] == "response.function_call_arguments.delta"


def test_chat_stream_encoder_emits_standard_chunks_and_terminal_last() -> None:
    encoder = OpenAIChatRuntime().new_stream_encoder(model="gpt-example", response_id="chat_1")
    started = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.RESPONSE_STARTED,
            response_id="chat_1",
            metadata={"model": "gpt-example"},
        )
    )
    text = encoder.encode(SemanticEvent(type=SemanticEventType.TEXT_DELTA, delta="hello"))
    terminal = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
        )
    )

    assert started[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert text[0]["choices"][0]["delta"] == {"content": "hello"}
    assert terminal[-1]["choices"][0]["finish_reason"] == "stop"
    with pytest.raises(ProtocolRepresentabilityError, match="after terminal"):
        encoder.encode(SemanticEvent(type=SemanticEventType.TEXT_DELTA, delta="late"))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "unknown"])
async def test_chat_nonstream_rejects_non_success_terminal_without_error(status: str) -> None:
    response = SemanticResponse(
        id="chat_1",
        model="gpt-example",
        output=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(TextContent("partial"),),
            ),
        ),
        terminal=TerminalMetadata(response_status=status),
    )

    with pytest.raises(ProtocolRepresentabilityError) as raised:
        await OpenAIChatRuntime().encode_response(response)

    assert raised.value.path == "response.terminal.response_status"


@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        ("failed", "upstream_error", "Upstream response failed"),
        ("cancelled", "upstream_cancelled", "Upstream response was cancelled"),
        (
            "unknown",
            "upstream_status_unknown",
            "Upstream response ended with an unknown status",
        ),
    ],
)
def test_chat_stream_projects_non_success_terminal_without_error(
    status: str,
    code: str,
    message: str,
) -> None:
    encoder = OpenAIChatRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="chat_1",
    )

    frames = encoder.encode(
        SemanticEvent(
            type=SemanticEventType.TERMINAL,
            terminal=TerminalMetadata(response_status=status),
        )
    )

    assert frames == ({"error": {"code": code, "message": message}},)
    assert encoder.finish_eof() == ()


@pytest.mark.asyncio
async def test_chat_incomplete_terminal_keeps_length_projection() -> None:
    terminal = TerminalMetadata(finish_reason="length", response_status="incomplete")
    response = SemanticResponse(
        id="chat_1",
        model="gpt-example",
        output=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(TextContent("partial"),),
            ),
        ),
        terminal=terminal,
    )

    encoded = await OpenAIChatRuntime().encode_response(response)
    encoder = OpenAIChatRuntime().new_stream_encoder(
        model="gpt-example",
        response_id="chat_1",
    )
    encoder.encode(
        SemanticEvent(
            type=SemanticEventType.RESPONSE_STARTED,
            response_id="chat_1",
            metadata={"model": "gpt-example"},
        )
    )
    stream_frames = encoder.encode(
        SemanticEvent(type=SemanticEventType.TERMINAL, terminal=terminal)
    )

    assert encoded["choices"][0]["finish_reason"] == "length"
    assert len(stream_frames) == 1
    assert stream_frames[0]["choices"][0]["finish_reason"] == "length"


def test_semantic_response_can_hold_message_for_chat_projection() -> None:
    response = SemanticResponse(
        id="chat_1",
        model="gpt-example",
        output=(
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(TextContent("hello"),),
            ),
        ),
        terminal=TerminalMetadata(finish_reason="stop", response_status="completed"),
    )

    assert semantic_to_chat_response(response).content == "hello"
