"""Tests for streaming chunk token accumulation (Feature 3)."""

from router_maestro.providers import ChatStreamChunk
from router_maestro.server.protocols.anthropic_reducer import AnthropicReducer
from router_maestro.server.schemas.anthropic import AnthropicStreamState


def _make_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    tool_calls: list | None = None,
    chunk_id: str = "chatcmpl-test",
) -> ChatStreamChunk:
    """Build a minimal canonical streaming chunk."""
    return ChatStreamChunk(
        content=content or "",
        finish_reason=finish_reason,
        usage=usage,
        tool_calls=tool_calls,
    )


def _reduce(chunk: ChatStreamChunk, state: AnthropicStreamState, model: str) -> list[dict]:
    return AnthropicReducer(response_id="chatcmpl-test", model=model, state=state).reduce(chunk)


class TestStreamingAccumulation:
    """Tests for usage accumulation in streaming translation."""

    def test_single_usage_chunk(self):
        """Single usage chunk is accumulated correctly."""
        state = AnthropicStreamState(estimated_input_tokens=100)
        chunk = _make_chunk(
            content="hello",
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )
        _reduce(chunk, state, "test-model")

        assert state.accumulated_prompt_tokens == 50
        assert state.accumulated_completion_tokens == 10
        assert state.last_usage is not None

    def test_multiple_usage_chunks_accumulate_max(self):
        """Multiple usage chunks keep the maximum (cumulative totals)."""
        state = AnthropicStreamState(estimated_input_tokens=100)

        # First chunk with partial usage
        chunk1 = _make_chunk(
            content="hi",
            usage={"prompt_tokens": 50, "completion_tokens": 5},
        )
        _reduce(chunk1, state, "test-model")
        assert state.accumulated_completion_tokens == 5

        # Second chunk with higher cumulative total
        chunk2 = _make_chunk(
            content=" there",
            usage={"prompt_tokens": 50, "completion_tokens": 15},
        )
        _reduce(chunk2, state, "test-model")
        assert state.accumulated_completion_tokens == 15
        assert state.accumulated_prompt_tokens == 50

    def test_completion_tokens_details_tracked(self):
        """completion_tokens_details from usage is stored."""
        state = AnthropicStreamState()
        chunk = _make_chunk(
            content="x",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        )
        _reduce(chunk, state, "test-model")
        assert state.completion_tokens_details == {"reasoning_tokens": 3}

    def test_prompt_tokens_details_tracked(self):
        """prompt_tokens_details from usage is stored."""
        state = AnthropicStreamState()
        chunk = _make_chunk(
            content="x",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        )
        _reduce(chunk, state, "test-model")
        assert state.prompt_tokens_details == {"cached_tokens": 8}

    def test_finish_uses_accumulated_tokens(self):
        """Finish event uses accumulated tokens instead of last chunk."""
        state = AnthropicStreamState(estimated_input_tokens=100)

        # Send content with usage
        chunk1 = _make_chunk(
            content="hello",
            usage={"prompt_tokens": 200, "completion_tokens": 30},
        )
        _reduce(chunk1, state, "test-model")

        # Finish chunk without usage
        finish_chunk = _make_chunk(finish_reason="stop")
        events = _reduce(finish_chunk, state, "test-model")

        # Find message_delta event
        delta_events = [e for e in events if e.get("type") == "message_delta"]
        assert len(delta_events) == 1
        usage = delta_events[0]["usage"]
        assert usage["input_tokens"] == 200
        assert usage["output_tokens"] == 30

    def test_finish_fallback_to_last_usage(self):
        """When no accumulation, falls back to last_usage."""
        state = AnthropicStreamState(estimated_input_tokens=100)

        # Content chunk without usage
        chunk1 = _make_chunk(content="hello")
        _reduce(chunk1, state, "test-model")

        # Finish chunk with usage
        finish_chunk = _make_chunk(
            finish_reason="stop",
            usage={"prompt_tokens": 150, "completion_tokens": 20},
        )
        events = _reduce(finish_chunk, state, "test-model")

        delta_events = [e for e in events if e.get("type") == "message_delta"]
        assert len(delta_events) == 1
        usage = delta_events[0]["usage"]
        assert usage["input_tokens"] == 150
        assert usage["output_tokens"] == 20

    def test_finish_fallback_to_estimated(self):
        """When no usage at all, falls back to estimated_input_tokens."""
        state = AnthropicStreamState(estimated_input_tokens=500)

        # Content chunk
        chunk1 = _make_chunk(content="hello")
        _reduce(chunk1, state, "test-model")

        # Finish with no usage
        finish_chunk = _make_chunk(finish_reason="stop")
        events = _reduce(finish_chunk, state, "test-model")

        delta_events = [e for e in events if e.get("type") == "message_delta"]
        assert len(delta_events) == 1
        usage = delta_events[0]["usage"]
        assert usage["input_tokens"] == 500
        assert usage["output_tokens"] == 0

    def test_zero_tokens_not_accumulated(self):
        """Zero-value tokens don't overwrite existing accumulation."""
        state = AnthropicStreamState()

        chunk1 = _make_chunk(
            content="x",
            usage={"prompt_tokens": 100, "completion_tokens": 10},
        )
        _reduce(chunk1, state, "test-model")

        # Chunk with zero values should not reduce accumulated
        chunk2 = _make_chunk(
            content="y",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )
        _reduce(chunk2, state, "test-model")

        assert state.accumulated_prompt_tokens == 100
        assert state.accumulated_completion_tokens == 10
