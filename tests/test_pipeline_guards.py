"""Tests for the stream pipeline guards."""

import pytest

from router_maestro.pipeline.leak_guard import LeakGuard, RawFrameLeakGuard
from router_maestro.pipeline.runaway_guard import RunawayGuard
from router_maestro.pipeline.beta_strip import strip_beta_tokens
from router_maestro.providers.base import ChatStreamChunk


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
        guard.feed_text('<invoke name="Read"><parameter name="file_path">/tmp/x</parameter></invoke>')
        recovered = guard.check_invoke_at_finish()
        assert recovered is not None
        assert len(recovered) == 1
        assert recovered[0]["function"]["name"] == "Read"

    def test_invoke_unknown_tool_no_recovery(self):
        """Invoke for unknown tool should not recover."""
        guard = LeakGuard(allowed_tool_names={"Bash"})
        guard.feed_text('<invoke name="Read"><parameter name="file_path">/tmp/x</parameter></invoke>')
        recovered = guard.check_invoke_at_finish()
        assert recovered is None

    def test_control_envelope_task_notification_abort(self):
        """Task notification envelope should trigger abort."""
        guard = LeakGuard()
        text = "<task-notification><task-id>123</task-id><summary>done</summary></task-notification>"
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
        text = '```\n<task-notification><task-id>123</task-id><summary>example</summary></task-notification>\n```'
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


class TestRawFrameLeakGuard:
    """Tests for the passthrough-route leak guard variant."""

    def test_text_delta_detected(self):
        import json

        guard = RawFrameLeakGuard()
        data = json.dumps({
            "delta": {"type": "text_delta", "text": "<tick>content</tick>"}
        })
        result = guard.feed_frame("content_block_delta", data)
        assert result is not None
        assert "tick" in result

    def test_non_delta_event_ignored(self):
        guard = RawFrameLeakGuard()
        result = guard.feed_frame("message_start", '{"message":{}}')
        assert result is None

    def test_thinking_delta_scanned(self):
        import json

        guard = RawFrameLeakGuard()
        data = json.dumps({
            "delta": {
                "type": "thinking_delta",
                "thinking": '<channel source="x">leak</channel>',
            }
        })
        result = guard.feed_frame("content_block_delta", data)
        assert result is not None


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


class TestBetaStrip:
    """Tests for beta header token stripping."""

    def test_no_patterns_passthrough(self):
        assert strip_beta_tokens("token-a,token-b", []) == "token-a,token-b"

    def test_none_header(self):
        assert strip_beta_tokens(None, ["foo"]) is None

    def test_empty_header(self):
        assert strip_beta_tokens("", ["foo"]) is None

    def test_exact_match_strip(self):
        result = strip_beta_tokens("output-128k-2025-02-19,context-1m-2025-08-07", ["output-128k-2025-02-19"])
        assert result == "context-1m-2025-08-07"

    def test_wildcard_strip(self):
        result = strip_beta_tokens("output-128k-2025-02-19,context-1m-2025-08-07", ["output-128k-*"])
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
