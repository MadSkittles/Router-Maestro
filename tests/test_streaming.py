"""Tests for the shared SSE streaming utility."""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest

from router_maestro.server.streaming import (
    SSE_HEADERS,
    resilient_sse_generator,
    sse_streaming_response,
)


async def _async_gen_from_list(items: list[str]) -> AsyncGenerator[str, None]:
    """Helper: yield items from a list as an async generator."""
    for item in items:
        yield item


async def _slow_gen(delay: float, items: list[str]) -> AsyncGenerator[str, None]:
    """Helper: yield items with a delay between each."""
    for item in items:
        await asyncio.sleep(delay)
        yield item


async def _failing_gen(
    error: Exception, items_before: list[str] | None = None
) -> AsyncGenerator[str, None]:
    """Helper: yield some items then raise an error."""
    for item in items_before or []:
        yield item
    raise error


async def _collect(gen: AsyncGenerator[str, None]) -> list[str]:
    """Collect all items from an async generator."""
    result = []
    async for item in gen:
        result.append(item)
    return result


@pytest.mark.asyncio
async def test_resilient_generator_passes_through_data():
    """Normal data should pass through unchanged."""
    items = ["data: chunk1\n\n", "data: chunk2\n\n", "data: [DONE]\n\n"]
    inner = _async_gen_from_list(items)

    result = await _collect(resilient_sse_generator(inner))

    assert result == items


@pytest.mark.asyncio
async def test_resilient_generator_yields_keepalive_on_timeout():
    """A keepalive comment should be emitted when no data arrives within the interval."""
    # Use a very short keepalive for testing
    with patch("router_maestro.server.streaming.SSE_KEEPALIVE_INTERVAL", 0.05):
        # Reimport to pick up the patched constant — but since we use
        # asyncio.wait(timeout=SSE_KEEPALIVE_INTERVAL) and the constant is
        # read at call time, patching the module-level constant works.

        async def slow_single_item() -> AsyncGenerator[str, None]:
            await asyncio.sleep(0.15)  # 3x the keepalive interval
            yield "data: final\n\n"

        result = await _collect(resilient_sse_generator(slow_single_item()))

    # Should have at least one keepalive before the final data
    keepalives = [r for r in result if r == ": keepalive\n\n"]
    data_items = [r for r in result if r.startswith("data:")]

    assert len(keepalives) >= 1, f"Expected keepalive comments, got: {result}"
    assert data_items == ["data: final\n\n"]


@pytest.mark.asyncio
async def test_resilient_generator_handles_cancelled_error():
    """CancelledError should be logged and re-raised."""

    async def cancelled_gen() -> AsyncGenerator[str, None]:
        yield "data: first\n\n"
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _collect(resilient_sse_generator(cancelled_gen()))


@pytest.mark.asyncio
async def test_resilient_generator_logs_connection_reset():
    """ConnectionResetError should be caught and logged."""
    inner = _failing_gen(ConnectionResetError("peer reset"), ["data: ok\n\n"])

    with patch("router_maestro.server.streaming.logger") as mock_logger:
        result = await _collect(resilient_sse_generator(inner))

    # Should have yielded the first item before the error
    assert "data: ok\n\n" in result
    mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_resilient_generator_logs_generic_exception():
    """Generic exceptions should be caught and logged with traceback."""
    inner = _failing_gen(RuntimeError("boom"), ["data: ok\n\n"])

    with patch("router_maestro.server.streaming.logger") as mock_logger:
        result = await _collect(resilient_sse_generator(inner))

    assert "data: ok\n\n" in result
    mock_logger.error.assert_called_once()
    # Verify exc_info=True was passed
    _, kwargs = mock_logger.error.call_args
    assert kwargs.get("exc_info") is True


def test_sse_streaming_response_has_correct_headers():
    """The response should include all required SSE headers."""
    inner = _async_gen_from_list(["data: test\n\n"])
    resp = sse_streaming_response(inner)

    for key, value in SSE_HEADERS.items():
        assert resp.headers.get(key) == value


def test_sse_streaming_response_has_correct_media_type():
    """The response media type should be text/event-stream."""
    inner = _async_gen_from_list(["data: test\n\n"])
    resp = sse_streaming_response(inner)

    assert resp.media_type == "text/event-stream"
