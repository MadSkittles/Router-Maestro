"""Runaway Guard: volume circuit-breaker for degenerate generation.

Detects streams that produce an excessive number of tiny fragments without
progressing toward completion — a sign of a model stuck in a degenerate
generation loop. Aborts the stream to avoid wasting resources and hanging
the client.

Trip conditions (both must hold):
- delta_count > max_deltas AND average bytes/delta < min_avg_bytes
  (detecting infinite tiny fragment streams)
- OR total_bytes > max_bytes (absolute ceiling regardless of pattern)
"""

from router_maestro.utils import get_logger

logger = get_logger("pipeline.runaway_guard")


class RunawayGuard:
    """Volume circuit-breaker for streaming responses."""

    def __init__(
        self,
        max_bytes: int = 10_000_000,
        max_deltas: int = 50_000,
        min_avg_bytes: float = 1.5,
    ):
        self._max_bytes = max_bytes
        self._max_deltas = max_deltas
        self._min_avg_bytes = min_avg_bytes
        self._total_bytes = 0
        self._delta_count = 0

    def feed_chunk(self, chunk) -> str | None:
        """Feed a chunk and check volume thresholds."""
        opaque_payload = getattr(chunk, "opaque_payload", None)
        if isinstance(opaque_payload, str) and opaque_payload:
            return self._record_payloads([opaque_payload])

        payloads: list[str] = []
        has_payload_event = False
        for field in (
            "content",
            "refusal",
            "thinking",
            "thinking_id",
            "thinking_signature",
        ):
            value = getattr(chunk, field, None)
            if isinstance(value, str) and value:
                payloads.append(value)
                has_payload_event = True
        tool_calls = getattr(chunk, "tool_calls", None) or []
        if tool_calls:
            has_payload_event = True
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                payloads.append(arguments)

        complete_tool_call = getattr(chunk, "tool_call", None)
        if complete_tool_call is not None:
            has_payload_event = True
        arguments = getattr(complete_tool_call, "arguments", None)
        if isinstance(arguments, str):
            payloads.append(arguments)

        if not has_payload_event:
            return None
        return self._record_payloads(payloads)

    def _record_payloads(self, payloads: list[str]) -> str | None:
        """Record one logical emitted payload event and apply circuit-breaker limits."""
        self._delta_count += 1
        self._total_bytes += sum(len(payload.encode("utf-8")) for payload in payloads)

        if self._total_bytes > self._max_bytes:
            reason = f"runaway_guard:max_bytes_exceeded:{self._total_bytes}>{self._max_bytes}"
            logger.warning(
                "Runaway guard tripped: total_bytes=%d > max=%d (deltas=%d)",
                self._total_bytes,
                self._max_bytes,
                self._delta_count,
            )
            return reason

        if (
            self._delta_count > self._max_deltas
            and self._delta_count > 0
            and (self._total_bytes / self._delta_count) < self._min_avg_bytes
        ):
            avg = self._total_bytes / self._delta_count
            reason = f"runaway_guard:tiny_fragments:deltas={self._delta_count},avg_bytes={avg:.1f}"
            logger.warning(
                "Runaway guard tripped: %d deltas with avg %.1f bytes/delta",
                self._delta_count,
                avg,
            )
            return reason

        return None

    def feed_text(self, text: str) -> str | None:
        """No-op — RunawayGuard operates on chunk-level metrics only."""
        return None
