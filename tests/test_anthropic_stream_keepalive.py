"""Tests for the Anthropic-route SSE keepalive frame.

Claude Code's stream consumer cancels at ~15s when only SSE comments
(``: keepalive\\n\\n``) flow between the eager ``message_start`` event and
the first upstream chunk.  The Anthropic route compensates by passing
real ``event: ping`` frames as the keepalive shape — this module verifies
that wiring stays in place and that the OpenAI route is not affected.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest

from router_maestro.server.routes.anthropic import ANTHROPIC_PING_FRAME
from router_maestro.server.streaming import (
    SSE_KEEPALIVE_COMMENT,
    resilient_sse_generator,
)


async def _slow_then_done(delay: float, item: str) -> AsyncGenerator[str, None]:
    """Sleep ``delay`` seconds, then emit ``item`` and finish."""
    await asyncio.sleep(delay)
    yield item


async def _collect(gen: AsyncGenerator[str, None]) -> list[str]:
    return [item async for item in gen]


def test_anthropic_ping_frame_is_valid_protocol_event():
    """The constant must be a real Anthropic ``ping`` SSE event, not a comment."""
    assert ANTHROPIC_PING_FRAME.startswith("event: ping\n")
    assert "data: " in ANTHROPIC_PING_FRAME
    assert ANTHROPIC_PING_FRAME.endswith("\n\n")
    # data line must be parseable JSON with type=ping
    data_line = next(line for line in ANTHROPIC_PING_FRAME.split("\n") if line.startswith("data: "))
    payload = json.loads(data_line[len("data: ") :])
    assert payload == {"type": "ping"}


@pytest.mark.asyncio
async def test_keepalive_frame_overrides_default_comment():
    """When ``keepalive_frame`` is supplied, it replaces the SSE comment."""
    with patch("router_maestro.server.streaming.SSE_KEEPALIVE_INTERVAL", 0.05):
        result = await _collect(
            resilient_sse_generator(
                _slow_then_done(0.2, "data: final\n\n"),
                keepalive_frame=ANTHROPIC_PING_FRAME,
            )
        )

    pings = [r for r in result if r == ANTHROPIC_PING_FRAME]
    comments = [r for r in result if r == SSE_KEEPALIVE_COMMENT]
    assert len(pings) >= 1, f"Expected ping frames, got: {result}"
    assert comments == [], "Anthropic keepalive must not emit SSE comments"
    assert "data: final\n\n" in result


@pytest.mark.asyncio
async def test_default_keepalive_is_still_sse_comment():
    """Routes that don't override keep the SSE comment (OpenAI/Codex regression)."""
    with patch("router_maestro.server.streaming.SSE_KEEPALIVE_INTERVAL", 0.05):
        result = await _collect(resilient_sse_generator(_slow_then_done(0.2, "data: final\n\n")))

    comments = [r for r in result if r == SSE_KEEPALIVE_COMMENT]
    assert len(comments) >= 1
    pings = [r for r in result if r.startswith("event: ping")]
    assert pings == []


@pytest.mark.asyncio
async def test_anthropic_route_passes_ping_frame_to_streaming_response():
    """``stream`` requests must wire ``ANTHROPIC_PING_FRAME`` into the response.

    Guards against accidentally regressing to the default SSE comment when
    refactoring ``routes/anthropic.py``.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from router_maestro.server.routes.anthropic import router as anthropic_router

    captured: dict = {}

    real_sse = pytest.importorskip("router_maestro.server.routes.anthropic")
    original = real_sse.sse_streaming_response

    def _spy(generator, keepalive_frame=SSE_KEEPALIVE_COMMENT):
        captured["keepalive_frame"] = keepalive_frame
        return original(generator, keepalive_frame=keepalive_frame)

    app = FastAPI()
    app.include_router(anthropic_router)

    with patch.object(real_sse, "sse_streaming_response", side_effect=_spy):
        client = TestClient(app)
        # The ``test`` model short-circuits before hitting any provider, so
        # we don't need to mock the router.
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "stream": True,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    assert captured.get("keepalive_frame") == ANTHROPIC_PING_FRAME
