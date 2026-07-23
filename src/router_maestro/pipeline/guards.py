"""Canonical stream-guard dispatch and compatibility helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from enum import StrEnum
from typing import Protocol, TypeVar

from router_maestro.config.priorities import GuardsConfig
from router_maestro.providers.base import ChatStreamChunk, ResponsesStreamChunk

GuardChunk = ChatStreamChunk | ResponsesStreamChunk


class GuardTextKind(StrEnum):
    """Classification for human-visible text carried by a stream chunk."""

    CONTENT = "content"
    REFUSAL = "refusal"
    THINKING = "thinking"


class StreamGuard(Protocol):
    """Contract implemented by stateful, per-request stream guards."""

    def feed_chunk(self, chunk: GuardChunk) -> str | None:
        """Inspect one typed chunk and return an abort reason when tripped."""
        ...

    def feed_text(self, text: str) -> str | None:
        """Inspect one human-visible text delta and return an abort reason."""
        ...


class GuardAbortError(Exception):
    """Raised internally when a guard trips."""

    def __init__(self, reason: str, guard_name: str):
        self.reason = reason
        self.guard_name = guard_name
        super().__init__(f"{guard_name}: {reason}")


_GuardT = TypeVar("_GuardT", bound=StreamGuard)


class GuardChain:
    """Own ordered dispatch across the guards for one response stream.

    Chunk-aware guards see the complete typed payload once. Human-readable
    ``content``, ``refusal``, and ``thinking`` deltas are then sent to that
    guard's text scanner exactly once. Opaque signatures, reasoning IDs, and
    structured tool arguments deliberately remain chunk-only payloads.
    """

    __slots__ = ("_guards",)

    def __init__(self, guards: Iterable[StreamGuard] = ()) -> None:
        self._guards = tuple(guards)

    @classmethod
    def from_config(
        cls,
        config: GuardsConfig,
        *,
        model: str,
        tool_names: set[str] | None = None,
    ) -> GuardChain:
        """Build from the guards section of an already captured config snapshot."""
        return cls(
            build_guards(
                model=model,
                tool_names=tool_names,
                leak_guard_enabled=config.leak_guard.enabled,
                runaway_guard_enabled=config.runaway_guard.enabled,
                runaway_max_bytes=config.runaway_guard.max_bytes,
                runaway_max_deltas=config.runaway_guard.max_deltas,
            )
        )

    def find(self, guard_type: type[_GuardT]) -> _GuardT | None:
        """Return the first guard of ``guard_type``, if the chain contains one."""
        for guard in self._guards:
            if isinstance(guard, guard_type):
                return guard
        return None

    def feed_chunk(self, chunk: GuardChunk) -> str | None:
        """Dispatch one chunk in guard order, stopping at the first abort."""
        visible_text = tuple(
            (kind, value)
            for kind in GuardTextKind
            if isinstance((value := getattr(chunk, kind.value, None)), str) and value
        )
        for guard in self._guards:
            reason = guard.feed_chunk(chunk)
            if reason:
                return reason
            feed_visible_text = getattr(guard, "feed_visible_text", None)
            for kind, text in visible_text:
                if callable(feed_visible_text):
                    reason = feed_visible_text(text, kind)
                elif kind is GuardTextKind.CONTENT:
                    # Preserve the legacy StreamGuard contract: third-party
                    # guards historically received only ``chunk.content``.
                    reason = guard.feed_text(text)
                else:
                    continue
                if reason:
                    return reason
        return None

    def feed_text(self, text: str) -> str | None:
        """Dispatch raw human-visible text in guard order."""
        for guard in self._guards:
            reason = guard.feed_text(text)
            if reason:
                return reason
        return None


async def guarded_stream(
    inner: AsyncIterator[ChatStreamChunk],
    guards: Iterable[StreamGuard] | GuardChain,
) -> AsyncGenerator[tuple[ChatStreamChunk, str | None]]:
    """Yield legacy ``(chunk, abort_reason)`` tuples through ``GuardChain``.

    This compatibility wrapper remains available for external callers. New
    Router-Maestro code should own a chain through ``RequestPipeline``.
    """
    chain = guards if isinstance(guards, GuardChain) else GuardChain(guards)
    async for chunk in inner:
        abort_reason = chain.feed_chunk(chunk)
        yield chunk, abort_reason
        if abort_reason:
            return


def build_guards(
    *,
    model: str,
    tool_names: set[str] | None = None,
    leak_guard_enabled: bool = True,
    runaway_guard_enabled: bool = True,
    runaway_max_bytes: int = 10_000_000,
    runaway_max_deltas: int = 50_000,
) -> list[StreamGuard]:
    """Build the legacy guard list without loading runtime configuration.

    ``model`` remains in the compatibility signature for third-party callers;
    the current guards are model-independent. New internal code should use
    :meth:`GuardChain.from_config` with its request-scoped config snapshot.
    """
    from router_maestro.pipeline.leak_guard import LeakGuard
    from router_maestro.pipeline.runaway_guard import RunawayGuard

    guards: list[StreamGuard] = []
    if leak_guard_enabled:
        guards.append(LeakGuard(allowed_tool_names=tool_names))
    if runaway_guard_enabled:
        guards.append(
            RunawayGuard(
                max_bytes=runaway_max_bytes,
                max_deltas=runaway_max_deltas,
            )
        )
    return guards
