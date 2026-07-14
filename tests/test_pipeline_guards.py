"""Tests for the stream pipeline guards."""

import importlib
from types import SimpleNamespace

import pytest

import router_maestro.pipeline as pipeline_api
import router_maestro.runtime.request_context as request_context_module
from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.pipeline.beta_strip import strip_beta_tokens
from router_maestro.pipeline.leak_guard import LeakGuard
from router_maestro.pipeline.request_pipeline import RequestPipeline
from router_maestro.pipeline.runaway_guard import RunawayGuard
from router_maestro.providers.base import ChatStreamChunk, ResponsesStreamChunk, ResponsesToolCall


class _RecordingGuard:
    def __init__(self, *, abort_text: str | None = None):
        self.abort_text = abort_text
        self.calls: list[tuple[str, object]] = []

    def feed_chunk(self, chunk: object) -> str | None:
        self.calls.append(("chunk", chunk))
        return None

    def feed_text(self, text: str) -> str | None:
        self.calls.append(("text", text))
        if text == self.abort_text:
            return "recording_guard:abort"
        return None


def test_guard_compatibility_imports_remain_available():
    guards_module = importlib.import_module("router_maestro.pipeline.guards")

    assert pipeline_api.StreamGuard is guards_module.StreamGuard
    assert pipeline_api.build_guards is guards_module.build_guards
    assert pipeline_api.guarded_stream is guards_module.guarded_stream
    assert guards_module.GuardAbortError.__module__ == "router_maestro.pipeline.guards"


def test_superseded_raw_frame_guard_is_not_a_second_production_entrypoint():
    leak_guard_module = importlib.import_module("router_maestro.pipeline.leak_guard")

    assert not hasattr(leak_guard_module, "RawFrameLeakGuard")


def test_guard_chain_dispatches_in_order_and_stops_after_first_abort():
    first = _RecordingGuard(abort_text="stop")
    second = _RecordingGuard()
    chain = pipeline_api.GuardChain([first, second])
    chunk = ChatStreamChunk(content="stop")

    result = chain.feed_chunk(chunk)

    assert result == "recording_guard:abort"
    assert first.calls == [("chunk", chunk), ("text", "stop")]
    assert second.calls == []


def test_legacy_custom_guard_receives_only_content_text():
    guard = _RecordingGuard()
    chain = pipeline_api.GuardChain([guard])
    chunk = ResponsesStreamChunk(
        content="content",
        refusal="refusal",
        thinking="thinking",
    )

    assert chain.feed_chunk(chunk) is None
    assert guard.calls == [("chunk", chunk), ("text", "content")]


@pytest.mark.parametrize("field", ["content", "refusal", "thinking"])
def test_guard_chain_leak_scans_each_human_visible_text_field(field):
    leak_guard = LeakGuard()
    chain = pipeline_api.GuardChain([leak_guard])
    payload = {"content": "", "refusal": None, "thinking": None}
    payload[field] = '<channel source="guard-test">visible</channel>'
    chunk = ResponsesStreamChunk(**payload)

    result = chain.feed_chunk(chunk)

    assert result == "response_leak:control_envelope:channel"


def test_guard_chain_separates_visible_leak_text_from_opaque_counted_payloads():
    leak_guard = LeakGuard()
    runaway_guard = RunawayGuard(max_bytes=10_000, max_deltas=50_000)
    chain = pipeline_api.GuardChain([leak_guard, runaway_guard])
    chunk = ResponsesStreamChunk(
        content="content",
        refusal="refusal",
        thinking="thinking",
        thinking_id="<tick>opaque-id</tick>",
        thinking_signature="<tick>opaque-signature</tick>",
        tool_call=ResponsesToolCall(
            call_id="call-1",
            name="lookup",
            arguments="<tick>structured-arguments</tick>",
        ),
    )

    result = chain.feed_chunk(chunk)

    assert result is None
    assert leak_guard.accumulated_text == "content"
    counted_payloads = (
        chunk.content,
        chunk.refusal,
        chunk.thinking,
        chunk.thinking_id,
        chunk.thinking_signature,
        chunk.tool_call.arguments,
    )
    assert runaway_guard._total_bytes == sum(
        len(payload.encode("utf-8")) for payload in counted_payloads if payload is not None
    )
    assert runaway_guard._delta_count == 1


def test_guard_chain_recovers_invoke_from_content_but_not_refusal_or_thinking():
    leak_guard = LeakGuard(allowed_tool_names={"Read"})
    chain = pipeline_api.GuardChain([leak_guard])
    invoke = '<invoke name="Read"><parameter name="file_path">/tmp/x</parameter></invoke>'

    assert (
        chain.feed_chunk(
            ResponsesStreamChunk(
                content="",
                refusal=invoke,
                thinking=invoke,
            )
        )
        is None
    )
    assert leak_guard.check_invoke_at_finish() is None

    assert chain.feed_chunk(ResponsesStreamChunk(content=invoke)) is None
    recovered = leak_guard.check_invoke_at_finish()
    assert recovered is not None
    assert [tool_call["function"]["name"] for tool_call in recovered] == ["Read"]


def test_leak_guard_scanner_state_is_isolated_between_visible_text_kinds():
    leak_guard = LeakGuard()
    chain = pipeline_api.GuardChain([leak_guard])

    assert chain.feed_chunk(ResponsesStreamChunk(content="```")) is None
    result = chain.feed_chunk(
        ResponsesStreamChunk(
            content="",
            thinking="<tick>private control</tick>",
        )
    )

    assert result == "response_leak:control_envelope:tick"


def test_leak_guard_never_joins_control_envelope_across_visible_text_kinds():
    leak_guard = LeakGuard()
    chain = pipeline_api.GuardChain([leak_guard])

    assert chain.feed_chunk(ResponsesStreamChunk(content="<tick>content")) is None
    result = chain.feed_chunk(
        ResponsesStreamChunk(
            content="",
            thinking="thinking</tick>",
        )
    )

    assert result is None


def test_request_pipeline_normalizes_legacy_guard_list_to_guard_chain():
    guard = _RecordingGuard()
    pipeline = RequestPipeline(
        request_id="req-guard-list",
        guards=[guard],
        leak_guard=None,
        audit=None,
        config=PrioritiesConfig(),
    )
    chunk = ChatStreamChunk(content="visible")

    assert pipeline.feed_stream(chunk) is None
    assert isinstance(pipeline._guard_chain, pipeline_api.GuardChain)
    assert guard.calls == [("chunk", chunk), ("text", "visible")]


def test_request_pipeline_builds_guards_from_bound_request_config(monkeypatch):
    config = PrioritiesConfig.model_validate(
        {
            "guards": {
                "leak_guard": {"enabled": False},
                "runaway_guard": {"enabled": False},
            }
        }
    )
    context = SimpleNamespace(config=config, audit=None, pipeline=None)
    monkeypatch.setattr(
        request_context_module,
        "get_current_request_context",
        lambda: context,
    )

    def fail_config_reload():
        pytest.fail("RequestPipeline reloaded config instead of using the request snapshot")

    monkeypatch.setattr(
        "router_maestro.pipeline.request_pipeline.load_priorities_config",
        fail_config_reload,
    )

    pipeline = RequestPipeline.create(
        request_id="req-snapshot-guards",
        model="github-copilot/gpt-4o",
    )

    assert context.pipeline is pipeline
    assert isinstance(pipeline._guard_chain, pipeline_api.GuardChain)
    assert pipeline.feed_stream(ChatStreamChunk(content="<tick>visible</tick>")) is None


def test_build_guards_retains_legacy_list_api():
    guards = pipeline_api.build_guards(
        model="github-copilot/gpt-4o",
        leak_guard_enabled=True,
        runaway_guard_enabled=True,
    )

    assert isinstance(guards, list)
    assert [type(guard) for guard in guards] == [LeakGuard, RunawayGuard]


@pytest.mark.asyncio
async def test_guarded_stream_retains_abort_tuple_behavior_without_wrapper_warning(caplog):
    guard = _RecordingGuard(abort_text="stop")
    chunks = [
        ChatStreamChunk(content="first"),
        ChatStreamChunk(content="stop"),
        ChatStreamChunk(content="unreachable"),
    ]

    async def inner():
        for chunk in chunks:
            yield chunk

    with caplog.at_level("WARNING", logger="router_maestro.pipeline.guards"):
        results = [
            item
            async for item in pipeline_api.guarded_stream(
                inner(),
                [guard],
            )
        ]

    assert results == [(chunks[0], None), (chunks[1], "recording_guard:abort")]
    assert caplog.records == []


class TestLeakGuard:
    """Tests for the LeakGuard."""

    def test_no_leak_normal_text(self):
        guard = LeakGuard(allowed_tool_names={"Read", "Bash"})
        result = guard.feed_text("Hello, this is a normal response with no leaks.")
        assert result is None

    def test_invoke_leak_not_abort(self):
        """Invoke leaks should NOT trigger abort — they are recovered at finish."""
        guard = LeakGuard(allowed_tool_names={"Read"})
        text = '<invoke name="Read"><parameter name="file_path">/tmp/x</parameter></invoke>'
        result = guard.feed_text(text)
        assert result is None  # No abort for invoke

    def test_invoke_recovery_at_finish(self):
        """Invoke leaks should be recoverable at stream finish."""
        guard = LeakGuard(allowed_tool_names={"Read"})
        guard.feed_text(
            '<invoke name="Read"><parameter name="file_path">/tmp/x</parameter></invoke>'
        )
        recovered = guard.check_invoke_at_finish()
        assert recovered is not None
        assert len(recovered) == 1
        assert recovered[0]["function"]["name"] == "Read"

    def test_invoke_unknown_tool_no_recovery(self):
        """Invoke for unknown tool should not recover."""
        guard = LeakGuard(allowed_tool_names={"Bash"})
        guard.feed_text(
            '<invoke name="Read"><parameter name="file_path">/tmp/x</parameter></invoke>'
        )
        recovered = guard.check_invoke_at_finish()
        assert recovered is None

    def test_control_envelope_task_notification_abort(self):
        """Task notification envelope should trigger abort."""
        guard = LeakGuard()
        text = (
            "<task-notification><task-id>123</task-id><summary>done</summary></task-notification>"
        )
        result = guard.feed_text(text)
        assert result is not None
        assert "control_envelope:task-notification" in result

    def test_control_envelope_teammate_message_abort(self):
        guard = LeakGuard()
        text = '<teammate-message teammate_id="abc">hello</teammate-message>'
        result = guard.feed_text(text)
        assert result is not None
        assert "control_envelope:teammate-message" in result

    def test_control_envelope_channel_abort(self):
        guard = LeakGuard()
        text = '<channel source="main">data</channel>'
        result = guard.feed_text(text)
        assert result is not None
        assert "control_envelope:channel" in result

    def test_control_envelope_cross_session_abort(self):
        guard = LeakGuard()
        text = '<cross-session-message from="other">hi</cross-session-message>'
        result = guard.feed_text(text)
        assert result is not None
        assert "control_envelope:cross-session-message" in result

    def test_control_envelope_tick_abort(self):
        guard = LeakGuard()
        text = "<tick>content here</tick>"
        result = guard.feed_text(text)
        assert result is not None
        assert "control_envelope:tick" in result

    def test_empty_tick_no_abort(self):
        """Empty <tick></tick> should not trigger (no content)."""
        guard = LeakGuard()
        result = guard.feed_text("<tick></tick>")
        assert result is None

    def test_control_envelope_in_code_fence_no_abort(self):
        """Control envelopes inside code fences should not trigger."""
        guard = LeakGuard()
        text = (
            "```\n<task-notification><task-id>123</task-id>"
            "<summary>example</summary></task-notification>\n```"
        )
        result = guard.feed_text(text)
        assert result is None

    def test_incremental_feeding(self):
        """Guard should work when text is fed incrementally (simulating streaming)."""
        guard = LeakGuard()
        parts = [
            "<task-noti",
            "fication>",
            "<task-id>1</task-id>",
            "<summary>x</summary>",
            "</task-notification>",
        ]
        result = None
        for part in parts:
            result = guard.feed_text(part)
            if result:
                break
        assert result is not None
        assert "task-notification" in result

    def test_tripped_stays_tripped(self):
        """Once tripped, subsequent feeds return the same reason."""
        guard = LeakGuard()
        text = "<task-notification><task-id>1</task-id><summary>x</summary></task-notification>"
        r1 = guard.feed_text(text)
        r2 = guard.feed_text("more text")
        assert r1 == r2


class TestRunawayGuard:
    """Tests for the RunawayGuard."""

    def test_normal_stream_no_trip(self):
        guard = RunawayGuard(max_bytes=10_000_000, max_deltas=50_000)
        for i in range(100):
            chunk = ChatStreamChunk(content=f"Normal chunk {i} with some content")
            result = guard.feed_chunk(chunk)
            assert result is None

    def test_max_bytes_trip(self):
        guard = RunawayGuard(max_bytes=1000, max_deltas=50_000)
        chunk = ChatStreamChunk(content="x" * 500)
        assert guard.feed_chunk(chunk) is None
        chunk2 = ChatStreamChunk(content="x" * 600)
        result = guard.feed_chunk(chunk2)
        assert result is not None
        assert "max_bytes_exceeded" in result

    def test_tiny_fragments_trip(self):
        guard = RunawayGuard(max_bytes=10_000_000, max_deltas=100, min_avg_bytes=2.0)
        for i in range(101):
            chunk = ChatStreamChunk(content="x")
            result = guard.feed_chunk(chunk)
            if result:
                assert "tiny_fragments" in result
                return
        pytest.fail("Guard should have tripped on tiny fragments")

    def test_tiny_fragments_no_trip_if_avg_ok(self):
        guard = RunawayGuard(max_bytes=10_000_000, max_deltas=100, min_avg_bytes=2.0)
        for i in range(150):
            chunk = ChatStreamChunk(content="x" * 10)
            result = guard.feed_chunk(chunk)
            assert result is None

    def test_feed_text_noop(self):
        guard = RunawayGuard()
        assert guard.feed_text("anything") is None

    def test_counts_content_and_all_tool_arguments_once_per_chunk(self):
        guard = RunawayGuard(max_bytes=10, max_deltas=50_000)
        chunk = ChatStreamChunk(
            content="abc",
            tool_calls=[
                {"function": {"arguments": "de"}},
                {"function": {"arguments": "fghi"}},
            ],
        )

        result = guard.feed_chunk(chunk)

        assert result is None
        assert guard._total_bytes == 9
        assert guard._delta_count == 1
        assert guard.feed_text(chunk.content) is None
        assert guard._total_bytes == 9
        assert guard._delta_count == 1

    def test_tool_arguments_use_utf8_bytes_and_trip_max_bytes(self):
        guard = RunawayGuard(max_bytes=5, max_deltas=50_000)
        chunk = ChatStreamChunk(
            content="",
            tool_calls=[{"function": {"arguments": "猫猫"}}],
        )

        result = guard.feed_chunk(chunk)

        assert result == "runaway_guard:max_bytes_exceeded:6>5"
        assert guard._delta_count == 1

    def test_empty_tool_chunk_still_counts_one_delta(self):
        guard = RunawayGuard(max_bytes=10_000_000, max_deltas=50_000)

        assert guard.feed_chunk(ChatStreamChunk(content="", tool_calls=[{}])) is None

        assert guard._total_bytes == 0
        assert guard._delta_count == 1

    def test_complete_responses_tool_call_arguments_count_once(self):
        guard = RunawayGuard(max_bytes=5, max_deltas=50_000)
        chunk = ResponsesStreamChunk(
            content="",
            tool_call=ResponsesToolCall(call_id="call-1", name="alpha", arguments="abcdef"),
        )

        result = guard.feed_chunk(chunk)

        assert result == "runaway_guard:max_bytes_exceeded:6>5"
        assert guard._delta_count == 1

    @pytest.mark.parametrize("field", ["thinking", "refusal"])
    def test_responses_payload_fields_use_utf8_bytes_and_trip_max_bytes(self, field):
        guard = RunawayGuard(max_bytes=5, max_deltas=50_000)
        chunk = ResponsesStreamChunk(content="", **{field: "猫猫"})

        result = guard.feed_chunk(chunk)

        assert result == "runaway_guard:max_bytes_exceeded:6>5"
        assert guard._total_bytes == 6
        assert guard._delta_count == 1

    def test_reasoning_bytes_at_limit_are_allowed_and_next_fragment_trips(self):
        guard = RunawayGuard(max_bytes=6, max_deltas=50_000)

        assert guard.feed_chunk(ResponsesStreamChunk(content="", thinking="猫")) is None
        assert guard.feed_chunk(ResponsesStreamChunk(content="", thinking="猫")) is None
        result = guard.feed_chunk(ResponsesStreamChunk(content="", thinking="猫"))

        assert result == "runaway_guard:max_bytes_exceeded:9>6"
        assert guard._delta_count == 3

    def test_control_and_usage_only_chunks_do_not_count_as_payload_deltas(self):
        guard = RunawayGuard(max_bytes=10, max_deltas=1, min_avg_bytes=100.0)

        assert guard.feed_chunk(ResponsesStreamChunk(content="", provenance_only=True)) is None
        assert (
            guard.feed_chunk(
                ResponsesStreamChunk(
                    content="",
                    usage={"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
                )
            )
            is None
        )
        assert guard.feed_chunk(ResponsesStreamChunk(content="x")) is None

        assert guard._delta_count == 1
        assert guard._total_bytes == 1

    @pytest.mark.parametrize(
        ("field", "value", "expected_bytes"),
        [
            ("thinking_signature", "ABCDEF", 6),
            ("thinking_signature", "猫猫", 6),
            ("thinking_id", "rs-猫", 6),
        ],
    )
    def test_reasoning_identity_payload_fields_trip_utf8_byte_limit(
        self, field, value, expected_bytes
    ):
        guard = RunawayGuard(max_bytes=5, max_deltas=50_000)

        result = guard.feed_chunk(ResponsesStreamChunk(content="", **{field: value}))

        assert result == f"runaway_guard:max_bytes_exceeded:{expected_bytes}>5"
        assert guard._total_bytes == expected_bytes
        assert guard._delta_count == 1

    def test_combined_reasoning_fields_count_once_and_sum_all_bytes(self):
        guard = RunawayGuard(max_bytes=10, max_deltas=50_000)

        result = guard.feed_chunk(
            ResponsesStreamChunk(
                content="",
                thinking="abc",
                thinking_id="rs-1",
                thinking_signature="WXYZ",
            )
        )

        assert result == "runaway_guard:max_bytes_exceeded:11>10"
        assert guard._total_bytes == 11
        assert guard._delta_count == 1

    def test_opaque_wire_payload_overrides_typed_field_byte_accounting(self):
        guard = RunawayGuard(max_bytes=10_000, max_deltas=50_000)
        chunk = ChatStreamChunk(
            content="visible",
            thinking="private",
            opaque_payload="exact-wire-frame",
        )

        assert guard.feed_chunk(chunk) is None
        assert guard._total_bytes == len(b"exact-wire-frame")
        assert guard._delta_count == 1


class TestBetaStrip:
    """Tests for beta header token stripping."""

    def test_no_patterns_passthrough(self):
        assert strip_beta_tokens("token-a,token-b", []) == "token-a,token-b"

    def test_none_header(self):
        assert strip_beta_tokens(None, ["foo"]) is None

    def test_empty_header(self):
        assert strip_beta_tokens("", ["foo"]) is None

    def test_exact_match_strip(self):
        result = strip_beta_tokens(
            "output-128k-2025-02-19,context-1m-2025-08-07", ["output-128k-2025-02-19"]
        )
        assert result == "context-1m-2025-08-07"

    def test_wildcard_strip(self):
        result = strip_beta_tokens(
            "output-128k-2025-02-19,context-1m-2025-08-07", ["output-128k-*"]
        )
        assert result == "context-1m-2025-08-07"

    def test_strip_all_returns_none(self):
        result = strip_beta_tokens("output-128k-2025-02-19", ["output-128k-*"])
        assert result is None

    def test_multiple_patterns(self):
        result = strip_beta_tokens(
            "output-128k-2025-02-19,advisor-tool-2025-04-02,context-1m-2025-08-07",
            ["output-128k-*", "advisor-tool-*"],
        )
        assert result == "context-1m-2025-08-07"

    def test_no_match_passthrough(self):
        result = strip_beta_tokens("context-1m-2025-08-07", ["output-128k-*"])
        assert result == "context-1m-2025-08-07"

    def test_spaces_in_tokens_handled(self):
        result = strip_beta_tokens("token-a , token-b", ["token-a"])
        assert result == "token-b"
