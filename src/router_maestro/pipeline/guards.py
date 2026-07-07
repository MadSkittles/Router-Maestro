"""Stream guard protocol and pipeline wrapper."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Protocol

from router_maestro.providers.base import ChatStreamChunk
from router_maestro.utils import get_logger

logger = get_logger("pipeline.guards")


class StreamGuard(Protocol):
    """Protocol for streaming response guards.

    A guard inspects each chunk and can signal abort by returning an abort
    reason string. Guards are stateful per-stream (created fresh per request).
    """

    def feed_chunk(self, chunk: ChatStreamChunk) -> str | None:
        """Process a chunk. Return abort reason to stop the stream, else None."""
        ...

    def feed_text(self, text: str) -> str | None:
        """Process raw text content. Return abort reason if triggered."""
        ...


class GuardAbortError(Exception):
    """Raised internally when a guard trips."""

    def __init__(self, reason: str, guard_name: str):
        self.reason = reason
        self.guard_name = guard_name
        super().__init__(f"{guard_name}: {reason}")


async def guarded_stream(
    inner: AsyncIterator[ChatStreamChunk],
    guards: list[StreamGuard],
) -> AsyncGenerator[tuple[ChatStreamChunk, str | None], None]:
    """Wrap a provider stream with guards.

    Yields (chunk, abort_reason) tuples. When a guard trips, the final
    tuple has a non-None abort_reason and iteration stops. The caller is
    responsible for emitting the appropriate error event for its API surface.
    """
    async for chunk in inner:
        abort_reason: str | None = None
        for guard in guards:
            reason = guard.feed_chunk(chunk)
            if reason:
                abort_reason = reason
                break
            if chunk.content:
                reason = guard.feed_text(chunk.content)
                if reason:
                    abort_reason = reason
                    break

        if abort_reason:
            logger.warning(
                "Stream guard triggered: reason=%s model=%s",
                abort_reason,
                chunk.model or "unknown",
            )
            yield chunk, abort_reason
            return

        yield chunk, None


def build_guards(
    *,
    model: str,
    tool_names: set[str] | None = None,
    leak_guard_enabled: bool = True,
    runaway_guard_enabled: bool = True,
    runaway_max_bytes: int = 10_000_000,
    runaway_max_deltas: int = 50_000,
) -> list[StreamGuard]:
    """Build the appropriate guard chain based on configuration."""
    from router_maestro.pipeline.leak_guard import LeakGuard
    from router_maestro.pipeline.runaway_guard import RunawayGuard

    guards: list[StreamGuard] = []

    if leak_guard_enabled:
        guards.append(LeakGuard(allowed_tool_names=tool_names))

    if runaway_guard_enabled:
        guards.append(RunawayGuard(max_bytes=runaway_max_bytes, max_deltas=runaway_max_deltas))

    return guards
