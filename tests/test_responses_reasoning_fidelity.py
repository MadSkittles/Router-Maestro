"""Lossless reasoning-state contracts for legacy Responses provider DTOs."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest

from router_maestro.protocols import (
    OpaqueState,
    ReasoningSummary,
    SemanticResponse,
    WireProtocol,
)
from router_maestro.protocols.openai_responses import (
    responses_chunk_to_semantic_events,
    responses_response_to_semantic,
    semantic_events_to_responses_chunks,
    semantic_to_responses_response,
)
from router_maestro.providers.base import ResponsesRequest, ResponsesResponse, ResponsesStreamChunk
from router_maestro.providers.bindings import COPILOT_OPENAI_RESPONSES_BINDING
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.providers.copilot_support.responses_codec import CopilotResponsesCodec
from router_maestro.runtime.reasoning_capsule import ReasoningCapsuleCodec
from router_maestro.server.protocols.runtime_factory import ProtocolRuntimeFactory


def _raw_reasoning_item() -> dict:
    return {
        "type": "reasoning",
        "id": "rs_lossless",
        "status": "completed",
        "summary": [
            {
                "type": "summary_text",
                "text": "plan",
                "future_part_field": {"rank": 1},
            }
        ],
        "encrypted_content": "opaque-state",
        "future_sibling": {"nested": [1, {"preserved": True}]},
    }


def _factory() -> ProtocolRuntimeFactory:
    return ProtocolRuntimeFactory(
        ReasoningCapsuleCodec(bytes([41]) * 32),
        {
            (
                "github-copilot",
                COPILOT_OPENAI_RESPONSES_BINDING,
            ): WireProtocol.OPENAI_RESPONSES
        },
    )


def _reasoning_state(response: ResponsesResponse) -> tuple[SemanticResponse, OpaqueState]:
    semantic = responses_response_to_semantic(
        response,
        response_id="resp_lossless",
        origin_provider="github-copilot",
        origin_binding=COPILOT_OPENAI_RESPONSES_BINDING,
    )
    reasoning = semantic.output[0]
    assert isinstance(reasoning, ReasoningSummary)
    assert reasoning.opaque_state is not None
    return semantic, reasoning.opaque_state


def test_legacy_response_full_reasoning_item_survives_capsule_and_replay() -> None:
    raw_item = _raw_reasoning_item()
    response = ResponsesResponse(
        content="",
        model="gpt-5",
        thinking="plan",
        thinking_id="rs_lossless",
        thinking_signature="opaque-state",
        reasoning_item=raw_item,
    )
    semantic, state = _reasoning_state(response)

    capsule = _factory().encode_opaque_state(
        state,
        protocol=WireProtocol.ANTHROPIC_MESSAGES,
        model="gpt-5",
        item_id="rs_lossless",
    )
    restored = _factory().decode_opaque_state(
        capsule,
        protocol=WireProtocol.ANTHROPIC_MESSAGES,
        model="ignored-public-model",
        item_id="ignored-carrier-id",
    )
    replay_semantic = replace(
        semantic,
        output=(ReasoningSummary("plan", opaque_state=restored),),
    )
    replay = semantic_to_responses_response(replay_semantic)

    assert replay.reasoning_item == raw_item
    assert replay.thinking_id == "rs_lossless"
    assert replay.thinking_signature == "opaque-state"
    assert replay.reasoning_item is not None
    assert replay.reasoning_item["future_sibling"]["nested"][1]["preserved"] is True


def test_legacy_stream_chunk_prefers_complete_reasoning_item_for_opaque_state() -> None:
    raw_item = _raw_reasoning_item()
    chunk = ResponsesStreamChunk(
        content="",
        thinking_id="rs_lossless",
        thinking_signature="opaque-state",
        output_index=2,
        output_item_type="reasoning",
        output_item_done=True,
        reasoning_item=raw_item,
    )

    events = responses_chunk_to_semantic_events(
        chunk,
        response_id="resp_lossless",
        model="gpt-5",
        origin_provider="github-copilot",
        origin_binding=COPILOT_OPENAI_RESPONSES_BINDING,
    )

    assert len(events) == 1
    assert isinstance(events[0].item, ReasoningSummary)
    assert events[0].item.text == ""
    assert events[0].item.opaque_state is not None
    blob = events[0].item.opaque_state.blob
    assert isinstance(blob, Mapping)
    future_sibling = blob["future_sibling"]
    assert isinstance(future_sibling, Mapping)
    nested = future_sibling["nested"]
    assert isinstance(nested, tuple)
    preserved_item = nested[1]
    assert isinstance(preserved_item, Mapping)
    assert preserved_item["preserved"] is True
    replay = semantic_events_to_responses_chunks(events)
    assert len(replay) == 1
    assert replay[0].reasoning_item == raw_item


@pytest.mark.asyncio
async def test_copilot_nonstream_response_carries_complete_reasoning_item() -> None:
    raw_item = _raw_reasoning_item()
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    upstream = httpx.Response(
        200,
        json={
            "id": "resp_upstream",
            "model": "gpt-5",
            "status": "completed",
            "output": [raw_item],
        },
        request=httpx.Request("POST", "https://api.githubcopilot.com/responses"),
    )
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=upstream
    )

    response = await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))

    assert response.reasoning_item == raw_item
    assert response.thinking == "plan"
    assert response.thinking_id == "rs_lossless"
    assert response.thinking_signature == "opaque-state"


async def _sse_lines(*events: dict) -> AsyncIterator[str]:
    for event in events:
        yield f"data: {json.dumps(event)}"


@pytest.mark.asyncio
async def test_copilot_stream_done_chunk_carries_complete_reasoning_item() -> None:
    raw_item = _raw_reasoning_item()
    codec = CopilotResponsesCodec()
    lines = _sse_lines(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": raw_item,
        },
        {
            "type": "response.completed",
            "response": {"status": "completed", "usage": None},
        },
    )

    chunks = [
        chunk
        async for chunk in codec.decode_stream(
            lines,
            ResponsesRequest(model="gpt-5", input="hi", stream=True),
        )
    ]
    done = next(chunk for chunk in chunks if chunk.output_item_done)

    assert done.reasoning_item == raw_item
    assert done.thinking_id == "rs_lossless"
    assert done.thinking_signature == "opaque-state"
    assert done.reasoning_item is not None
    assert done.reasoning_item["future_sibling"]["nested"][1]["preserved"] is True
