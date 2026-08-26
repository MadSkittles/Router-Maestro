"""Contracts for the semantic-to-legacy Chat provider bridge."""

from __future__ import annotations

import json

import pytest

from router_maestro.protocols import (
    FileContent,
    ImageContent,
    MessageRole,
    OpaqueState,
    ProtocolRepresentabilityError,
    ReasoningConfig,
    ReasoningSummary,
    SemanticEventType,
    SemanticMessage,
    SemanticRequest,
    TextContent,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolResult,
    WireProtocol,
    semantic_events_from_legacy_chat_chunk,
    semantic_request_to_legacy_chat,
    semantic_response_from_legacy_chat,
)
from router_maestro.providers.base import ChatResponse, ChatStreamChunk


def test_request_bridge_preserves_portable_messages_tools_media_and_options() -> None:
    request = SemanticRequest(
        model="gpt-example",
        input=(
            SemanticMessage(
                role=MessageRole.SYSTEM,
                content=(TextContent("system"),),
            ),
            SemanticMessage(
                role=MessageRole.USER,
                content=(
                    TextContent("inspect"),
                    ImageContent(
                        source="AA==",
                        media_type="image/png",
                        detail="high",
                        source_kind="base64",
                    ),
                    FileContent(
                        source="notes",
                        filename="notes.txt",
                        media_type="text/plain",
                        source_kind="text",
                    ),
                ),
            ),
            SemanticMessage(
                role=MessageRole.ASSISTANT,
                content=(ToolCall(call_id="call_1", name="lookup", arguments={"q": 1}),),
            ),
            SemanticMessage(
                role=MessageRole.TOOL,
                content=(ToolResult(call_id="call_1", content=(TextContent("value"),)),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="lookup",
                description="Lookup",
                input_schema={"type": "object"},
                strict=True,
            ),
        ),
        stream=True,
        max_output_tokens=128,
        temperature=0.2,
        top_p=0.9,
        top_k=8,
        stop_sequences=("done",),
        tool_choice=ToolChoice("function", name="lookup"),
        reasoning=ReasoningConfig(enabled=True, effort="high", budget_tokens=32),
        structured_output={"type": "json_schema", "schema": {"type": "object"}},
        response_mime_type="application/json",
        metadata={"request": "one"},
        service_tier="priority",
        candidate_count=1,
        provider_extensions={"vendor": True},
    )

    bridged = semantic_request_to_legacy_chat(request)

    assert bridged.model == "gpt-example"
    assert bridged.stream is True
    assert bridged.max_tokens == 128
    assert bridged.messages[0].content == "system"
    assert bridged.messages[1].content == [
        {"type": "text", "text": "inspect"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}},
        {
            "type": "document",
            "source": {
                "type": "text",
                "data": "notes",
                "media_type": "text/plain",
            },
            "title": "notes.txt",
        },
    ]
    assert bridged.messages[2].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":1}'},
        }
    ]
    assert bridged.messages[3].tool_call_id == "call_1"
    assert bridged.messages[3].content == "value"
    assert bridged.tool_choice == {"type": "function", "function": {"name": "lookup"}}
    assert bridged.thinking_budget == 32
    assert bridged.thinking_type == "enabled"
    assert bridged.reasoning_effort == "high"
    assert bridged.output_format == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }
    assert bridged.provider_extensions == {"vendor": True}


def test_request_bridge_projects_error_tool_results_for_chat_fallback() -> None:
    request = SemanticRequest(
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

    bridged = semantic_request_to_legacy_chat(request)

    content = bridged.messages[0].content
    assert isinstance(content, str)
    assert json.loads(content) == {
        "$router_maestro": {"type": "tool_result", "version": 1},
        "is_error": True,
        "output": "command failed",
    }


@pytest.mark.parametrize("parallel_tool_calls", [False, True])
def test_explicit_parallel_tool_calls_is_rejected(parallel_tool_calls: bool) -> None:
    request = SemanticRequest(
        model="gpt-example",
        parallel_tool_calls=parallel_tool_calls,
    )

    with pytest.raises(ProtocolRepresentabilityError) as raised:
        semantic_request_to_legacy_chat(request)

    assert raised.value.path == "parallel_tool_calls"


def test_response_bridge_attaches_complete_reasoning_provenance() -> None:
    response = ChatResponse(
        content="done",
        model="gpt-example",
        finish_reason="length",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
        thinking="trace",
        thinking_signature="encrypted",
        thinking_id="rs_1",
    )

    semantic = semantic_response_from_legacy_chat(
        response,
        response_id="resp_1",
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_binding="copilot-responses",
    )

    assert semantic.id == "resp_1"
    assert semantic.usage is not None
    assert semantic.usage.cached_input_tokens == 3
    assert semantic.usage.reasoning_tokens == 2
    assert semantic.terminal is not None
    assert semantic.terminal.finish_reason == "length"
    assert semantic.terminal.response_status == "incomplete"
    message = semantic.output[0]
    assert isinstance(message, SemanticMessage)
    reasoning = message.content[1]
    assert reasoning == ReasoningSummary(
        "trace",
        opaque_state=OpaqueState(
            origin_protocol=WireProtocol.OPENAI_RESPONSES,
            origin_provider="github-copilot",
            origin_model="gpt-example",
            item_id="rs_1",
            blob="encrypted",
            origin_binding="copilot-responses",
        ),
    )


def test_chunk_bridge_emits_reasoning_state_text_tools_usage_then_terminal() -> None:
    chunk = ChatStreamChunk(
        content="done",
        finish_reason="tool_calls",
        usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        tool_calls=[
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":'},
            }
        ],
        thinking="trace",
        thinking_signature="encrypted",
        thinking_id="rs_1",
    )

    events = semantic_events_from_legacy_chat_chunk(
        chunk,
        response_id="resp_1",
        model="gpt-example",
        origin_protocol=WireProtocol.OPENAI_RESPONSES,
        origin_provider="github-copilot",
        origin_binding="copilot-responses",
        sequence_start=20,
    )

    assert [event.type for event in events] == [
        SemanticEventType.REASONING_DELTA,
        SemanticEventType.OUTPUT_ITEM,
        SemanticEventType.TEXT_DELTA,
        SemanticEventType.OUTPUT_ITEM,
        SemanticEventType.TOOL_ARGUMENTS_DELTA,
        SemanticEventType.USAGE,
        SemanticEventType.TERMINAL,
    ]
    assert [event.sequence for event in events] == list(range(20, 27))
    assert all(event.response_id == "resp_1" for event in events)
    opaque = events[1].item
    assert isinstance(opaque, ReasoningSummary)
    assert opaque.opaque_state is not None
    assert opaque.opaque_state.origin_binding == "copilot-responses"
    call = events[3].item
    assert call == ToolCall(call_id="call_1", name="lookup")
    assert events[4].delta == '{"q":'
    assert events[-1].terminal is not None
    assert events[-1].terminal.finish_reason == "tool_calls"


def test_chunk_opaque_reasoning_requires_model_and_item_id() -> None:
    chunk = ChatStreamChunk(
        content="",
        thinking_signature="encrypted",
        thinking_id="rs_1",
    )

    with pytest.raises(ProtocolRepresentabilityError, match="model context"):
        semantic_events_from_legacy_chat_chunk(chunk)

    chunk.thinking_id = None
    with pytest.raises(ProtocolRepresentabilityError, match="item ID"):
        semantic_events_from_legacy_chat_chunk(chunk, model="gpt-example")
