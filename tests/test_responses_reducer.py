"""Pure Responses reducer characterization tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from router_maestro.providers.base import (
    ProviderError,
    ProviderFailureKind,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
    exception_outcome,
)
from router_maestro.server import protocols
from router_maestro.server.protocols.responses_reducer import (
    ResponsesReducer,
    ResponsesSnapshot,
    build_output_items,
)


def _id_factory() -> Callable[[str], str]:
    counters: dict[str, int] = {}

    def generate(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-test-{counters[prefix]}"

    return generate


def _base_response() -> dict[str, Any]:
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 123,
        "model": "provider/model",
        "error": None,
        "incomplete_details": None,
    }


def test_protocol_package_exports_only_cross_module_production_api() -> None:
    assert protocols.__all__ == [
        "ResponsesReducer",
        "build_nonstream_snapshot",
        "client_error_response",
        "unrepresented_option_error",
    ]


def _reduce(
    chunks: list[ResponsesStreamChunk],
) -> tuple[list[dict[str, Any]], ResponsesSnapshot]:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    events = list(reducer.start().events)
    for chunk in chunks:
        step = reducer.feed(chunk)
        events.extend(step.events)
    if reducer.snapshot is None:
        step = reducer.finish()
        events.extend(step.events)
    assert reducer.snapshot is not None
    return events, reducer.snapshot


def test_text_usage_and_completed_snapshot_are_pure_payloads() -> None:
    events, snapshot = _reduce(
        [
            ResponsesStreamChunk(content="hel"),
            ResponsesStreamChunk(content="lo"),
            ResponsesStreamChunk(
                content="",
                usage={
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 1},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
                finish_reason="stop",
            ),
        ]
    )

    assert all(isinstance(event, dict) for event in events)
    text_deltas = [
        event["delta"] for event in events if event["type"] == "response.output_text.delta"
    ]
    assert text_deltas == ["hel", "lo"]
    assert snapshot.outcome.response_status is ResponseStatus.COMPLETED
    assert snapshot.response["output"][0]["content"] == [
        {"type": "output_text", "text": "hello", "annotations": []}
    ]
    assert snapshot.response["usage"] == {
        "input_tokens": 3,
        "input_tokens_details": {"cached_tokens": 1},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 5,
    }


def test_latest_upstream_usage_snapshot_wins() -> None:
    _events, snapshot = _reduce(
        [
            ResponsesStreamChunk(
                content="x",
                usage={"input_tokens": 4, "output_tokens": 1},
            ),
            ResponsesStreamChunk(
                content="",
                usage={"input_tokens": 4, "output_tokens": 2},
                finish_reason="stop",
            ),
        ]
    )

    assert snapshot.response["usage"] == {
        "input_tokens": 4,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 6,
    }


def test_reasoning_identity_signature_and_interleaved_message_are_preserved() -> None:
    events, snapshot = _reduce(
        [
            ResponsesStreamChunk(content="", thinking="plan", thinking_id="rs-upstream"),
            ResponsesStreamChunk(content="answer"),
            ResponsesStreamChunk(
                content="",
                thinking_id="rs-upstream",
                thinking_signature="encrypted",
            ),
            ResponsesStreamChunk(content="", finish_reason="stop"),
        ]
    )

    assert snapshot.response["output"] == [
        {
            "type": "reasoning",
            "id": "rs-upstream",
            "summary": [{"type": "summary_text", "text": "plan"}],
            "encrypted_content": "encrypted",
        },
        {
            "type": "message",
            "id": "msg-test-1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer", "annotations": []}],
            "status": "completed",
        },
    ]
    reasoning_events = [
        event
        for event in events
        if event["type"].startswith("response.reasoning_summary")
        or (
            event["type"] in {"response.output_item.added", "response.output_item.done"}
            and event.get("item", {}).get("type") == "reasoning"
        )
    ]
    reasoning_ids = {
        event.get("item_id", event.get("item", {}).get("id")) for event in reasoning_events
    }
    assert reasoning_ids == {"rs-upstream"}


@pytest.mark.parametrize(
    ("tool_call", "expected_item"),
    [
        (
            ResponsesToolCall(
                call_id="call-fn",
                name="lookup",
                arguments='{"q":"x"}',
                namespace="mcp",
            ),
            {
                "type": "function_call",
                "id": "fc-test-1",
                "call_id": "call-fn",
                "name": "lookup",
                "arguments": '{"q":"x"}',
                "status": "completed",
                "namespace": "mcp",
            },
        ),
        (
            ResponsesToolCall(
                call_id="call-custom",
                name="apply_patch",
                arguments="patch body",
                kind="custom",
            ),
            {
                "type": "custom_tool_call",
                "id": "ctc-test-1",
                "call_id": "call-custom",
                "name": "apply_patch",
                "input": "patch body",
                "status": "completed",
            },
        ),
        (
            ResponsesToolCall(
                call_id="call-search",
                name="tool_search",
                arguments='{"query":"x"}',
                kind="tool_search",
            ),
            {
                "type": "tool_search_call",
                "call_id": "call-search",
                "execution": "client",
                "status": "completed",
                "arguments": {"query": "x"},
            },
        ),
    ],
)
def test_tool_call_kinds_emit_canonical_items(
    tool_call: ResponsesToolCall,
    expected_item: dict[str, Any],
) -> None:
    events, snapshot = _reduce(
        [
            ResponsesStreamChunk(content="", tool_call=tool_call),
            ResponsesStreamChunk(content="", finish_reason="stop"),
        ]
    )

    assert snapshot.response["output"] == [expected_item]
    assert (
        next(event["item"] for event in events if event["type"] == "response.output_item.done")
        == expected_item
    )


@pytest.mark.parametrize(
    ("outcome", "event_type", "status"),
    [
        (
            TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.INCOMPLETE,
                incomplete_details={"reason": "max_output_tokens"},
            ),
            "response.incomplete",
            "incomplete",
        ),
        (
            TerminalOutcome(
                transport=TransportTermination.EXPLICIT_TERMINAL,
                response_status=ResponseStatus.FAILED,
                error=TerminalError(code="upstream_failed", message="safe failure"),
            ),
            "response.failed",
            "failed",
        ),
    ],
)
def test_terminal_snapshots_preserve_incomplete_and_failure(
    outcome: TerminalOutcome,
    event_type: str,
    status: str,
) -> None:
    events, snapshot = _reduce(
        [
            ResponsesStreamChunk(content="partial"),
            ResponsesStreamChunk(content="", terminal_outcome=outcome),
        ]
    )

    assert events[-1]["type"] == event_type
    assert snapshot.response["status"] == status
    assert snapshot.response["output"][0]["content"][0]["text"] == "partial"


def test_indexed_interleaving_is_reduced_in_source_item_order() -> None:
    events, snapshot = _reduce(
        [
            ResponsesStreamChunk(
                content="B",
                output_index=1,
                content_index=0,
                output_item_type="message",
                output_item_done=True,
            ),
            ResponsesStreamChunk(
                content="A",
                output_index=0,
                content_index=0,
                output_item_type="message",
                output_item_done=True,
            ),
            ResponsesStreamChunk(content="", finish_reason="stop"),
        ]
    )

    assert [item["content"][0]["text"] for item in snapshot.response["output"]] == ["A", "B"]
    assert [
        event["output_index"] for event in events if event["type"] == "response.output_item.done"
    ] == [0, 1]


def test_nonstream_and_stream_share_output_item_construction() -> None:
    tool_calls = [
        ResponsesToolCall(
            call_id="call-fn",
            name="lookup",
            arguments="{}",
            namespace="mcp",
        ),
        ResponsesToolCall(
            call_id="call-custom",
            name="apply_patch",
            arguments="patch",
            kind="custom",
        ),
        ResponsesToolCall(
            call_id="call-search",
            name="tool_search",
            arguments='{"query":"x"}',
            kind="tool_search",
        ),
    ]
    nonstream = build_output_items(
        ResponsesResponse(
            content="answer",
            model="provider/model",
            thinking="plan",
            thinking_id="rs-upstream",
            thinking_signature="encrypted",
            tool_calls=tool_calls,
        ),
        id_factory=_id_factory(),
    )
    _events, snapshot = _reduce(
        [
            ResponsesStreamChunk(
                content="",
                thinking="plan",
                thinking_id="rs-upstream",
                thinking_signature="encrypted",
            ),
            ResponsesStreamChunk(content="answer"),
            *(ResponsesStreamChunk(content="", tool_call=tool_call) for tool_call in tool_calls),
            ResponsesStreamChunk(content="", finish_reason="stop"),
        ]
    )

    assert snapshot.response["output"] == nonstream


@pytest.mark.parametrize("thinking", [None, "plan"], ids=["signature-only", "thinking-signature"])
def test_nonstream_reasoning_signature_requires_upstream_id(thinking: str | None) -> None:
    response = ResponsesResponse(
        content="",
        model="provider/model",
        thinking=thinking,
        thinking_signature="encrypted",
    )

    with pytest.raises(ProviderError) as caught:
        build_output_items(response, id_factory=_id_factory())

    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert str(caught.value) == "Reasoning signature is missing its upstream item id"


def test_nonstream_reasoning_signature_preserves_real_upstream_id() -> None:
    output = build_output_items(
        ResponsesResponse(
            content="",
            model="provider/model",
            thinking="plan",
            thinking_id="rs-upstream",
            thinking_signature="encrypted",
        ),
        id_factory=_id_factory(),
    )

    assert output == [
        {
            "type": "reasoning",
            "id": "rs-upstream",
            "summary": [{"type": "summary_text", "text": "plan"}],
            "encrypted_content": "encrypted",
        }
    ]


def test_terminate_closes_partial_output_and_preserves_usage() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    reducer.feed(
        ResponsesStreamChunk(
            content="partial",
            usage={"input_tokens": 2, "output_tokens": 1},
        )
    )
    outcome = exception_outcome("guard limit", code="overloaded")

    result = reducer.terminate(
        outcome,
        wire_error={"type": "server_error", "message": "Overloaded: please retry"},
    )

    assert result.snapshot is not None
    terminal = [event for event in result.events if event["type"] == "response.failed"]
    assert len(terminal) == 1
    assert terminal[0]["response"] == result.snapshot.response
    assert result.snapshot.response["output"][0]["content"] == [
        {"type": "output_text", "text": "partial", "annotations": []}
    ]
    assert result.snapshot.response["usage"] == {
        "input_tokens": 2,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 3,
    }
    assert result.snapshot.response["error"] == {
        "type": "server_error",
        "message": "Overloaded: please retry",
    }


def test_terminate_is_idempotent_and_emits_one_terminal() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    outcome = exception_outcome("failed", code="server_error")

    first = reducer.terminate(outcome)
    second = reducer.terminate(outcome)

    assert [event["type"] for event in first.events].count("response.failed") == 1
    assert second.events == ()
    assert second.snapshot is first.snapshot


def test_terminate_materializes_deferred_mixed_chunk_before_failure() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    reducer.feed(
        ResponsesStreamChunk(
            content="answer",
            thinking="plan",
            usage={"input_tokens": 4, "output_tokens": 2},
        )
    )

    result = reducer.terminate(exception_outcome("boom", code="provider_error"))

    assert result.snapshot is not None
    terminal = [event for event in result.events if event["type"] == "response.failed"]
    assert len(terminal) == 1
    assert not any(event["type"] == "response.completed" for event in result.events)
    assert [item["type"] for item in result.snapshot.response["output"]] == [
        "reasoning",
        "message",
    ]
    assert result.snapshot.response["output"][0]["summary"] == [
        {"type": "summary_text", "text": "plan"}
    ]
    assert result.snapshot.response["output"][1]["content"] == [
        {"type": "output_text", "text": "answer", "annotations": []}
    ]
    assert result.snapshot.response["usage"] == {
        "input_tokens": 4,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 6,
    }


def test_terminate_ignores_terminal_metadata_on_deferred_payload() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    with pytest.raises(ProviderError):
        reducer.feed(
            ResponsesStreamChunk(
                content="answer",
                thinking="plan",
                thinking_signature="invalid-without-id",
                usage={"input_tokens": 4, "output_tokens": 2},
                finish_reason="stop",
            )
        )

    result = reducer.terminate(exception_outcome("boom", code="upstream_protocol_error"))

    assert result.snapshot is not None
    assert result.snapshot.response["status"] == "failed"
    assert [event["type"] for event in result.events].count("response.failed") == 1
    assert not any(event["type"] == "response.completed" for event in result.events)
    assert [item["type"] for item in result.snapshot.response["output"]] == [
        "reasoning",
        "message",
    ]
    assert result.snapshot.response["usage"]["total_tokens"] == 6


def test_terminate_finalizes_contiguous_indexed_output_in_source_order() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    reducer.feed(
        ResponsesStreamChunk(
            content="future",
            output_index=1,
            content_index=0,
            output_item_type="message",
            output_item_done=True,
        )
    )
    reducer.feed(
        ResponsesStreamChunk(
            content="active",
            output_index=0,
            content_index=0,
            output_item_type="message",
        )
    )

    result = reducer.terminate(exception_outcome("boom", code="provider_error"))

    assert result.snapshot is not None
    assert [event["type"] for event in result.events].count("response.failed") == 1
    assert [item["content"][0]["text"] for item in result.snapshot.response["output"]] == [
        "active",
        "future",
    ]


def test_terminate_drops_unreachable_indexed_future_after_gap() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    reducer.feed(
        ResponsesStreamChunk(
            content="future-two",
            output_index=2,
            content_index=0,
            output_item_type="message",
            output_item_done=True,
        )
    )
    reducer.feed(
        ResponsesStreamChunk(
            content="active-zero",
            output_index=0,
            content_index=0,
            output_item_type="message",
        )
    )
    outcome = exception_outcome("provider failed", code="provider_error")

    result = reducer.terminate(outcome)

    assert result.snapshot is not None
    assert result.snapshot.outcome is outcome
    assert result.snapshot.response["status"] == "failed"
    assert [event["type"] for event in result.events].count("response.failed") == 1
    assert [item["content"][0]["text"] for item in result.snapshot.response["output"]] == [
        "active-zero"
    ]
    assert reducer.indexed_outputs.buckets == {}
    assert reducer.indexed_outputs.future_chunk_count == 0
    assert reducer.indexed_outputs.future_payload_bytes == 0


def test_terminate_drops_invalid_indexed_suffix_and_keeps_external_failure() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    reducer.feed(
        ResponsesStreamChunk(
            content="future-invalid",
            output_index=1,
            content_index=1,
            output_item_type="message",
            output_item_done=True,
        )
    )
    reducer.feed(
        ResponsesStreamChunk(
            content="safe-prefix",
            output_index=0,
            content_index=0,
            output_item_type="message",
        )
    )
    outcome = exception_outcome("provider failed", code="provider_error")

    result = reducer.terminate(outcome)

    assert result.snapshot is not None
    assert result.snapshot.outcome is outcome
    assert [event["type"] for event in result.events].count("response.failed") == 1
    assert [item["content"][0]["text"] for item in result.snapshot.response["output"]] == [
        "safe-prefix"
    ]


def test_terminate_drops_invalid_deferred_payload_and_keeps_external_failure() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    reducer.feed(
        ResponsesStreamChunk(
            content="invalid-message",
            content_index=1,
            thinking="safe-reasoning",
            thinking_id="rs-safe",
            usage={"input_tokens": 3, "output_tokens": 1},
        )
    )
    outcome = exception_outcome("provider failed", code="provider_error")

    result = reducer.terminate(outcome)

    assert result.snapshot is not None
    assert result.snapshot.outcome is outcome
    assert [event["type"] for event in result.events].count("response.failed") == 1
    assert result.snapshot.response["output"] == [
        {
            "type": "reasoning",
            "id": "rs-safe",
            "summary": [{"type": "summary_text", "text": "safe-reasoning"}],
        }
    ]
    assert result.snapshot.response["usage"] is None


def test_terminate_drops_deferred_payload_on_unexpected_replay_error(monkeypatch) -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    reducer.feed(
        ResponsesStreamChunk(
            content="unsafe-message",
            thinking="safe-reasoning",
            thinking_id="rs-safe",
        )
    )
    monkeypatch.setattr(
        reducer.state,
        "bind_source_content_index",
        lambda _index: (_ for _ in ()).throw(RuntimeError("unexpected replay failure")),
    )
    outcome = exception_outcome("provider failed", code="provider_error")

    result = reducer.terminate(outcome)

    assert result.snapshot is not None
    assert result.snapshot.outcome is outcome
    assert [event["type"] for event in result.events].count("response.failed") == 1
    assert result.snapshot.response["output"] == [
        {
            "type": "reasoning",
            "id": "rs-safe",
            "summary": [{"type": "summary_text", "text": "safe-reasoning"}],
        }
    ]


def test_reducer_copies_base_response_and_binds_model_explicitly() -> None:
    base_response = _base_response()
    reducer = ResponsesReducer(base_response, id_factory=_id_factory())

    base_response["model"] = "external-mutation"
    assert reducer.base_response["model"] == "provider/model"

    reducer.bind_model("selected/model")

    assert base_response["model"] == "external-mutation"
    assert reducer.start().events[0]["response"]["model"] == "selected/model"


def test_terminal_snapshot_copies_output_list_from_mutable_state() -> None:
    reducer = ResponsesReducer(_base_response(), id_factory=_id_factory())
    reducer.feed(ResponsesStreamChunk(content="safe"))

    result = reducer.terminate(exception_outcome("failed", code="provider_error"))
    assert result.snapshot is not None
    terminal = next(event for event in result.events if event["type"] == "response.failed")

    reducer.state.output_items.append({"type": "unsafe-late-mutation"})

    assert [item["type"] for item in result.snapshot.response["output"]] == ["message"]
    assert [item["type"] for item in terminal["response"]["output"]] == ["message"]
