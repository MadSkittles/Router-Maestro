"""Unified request processing pipeline.

Every route creates a `RequestPipeline` at the start of a request. The
pipeline provides:
  - Stream guards (leak detection, runaway protection)
  - Audit tracing (per-request capture when enabled)
  - Configuration access (guards config, beta strip patterns)

Usage in a route:
    pipeline = RequestPipeline.create(request_id, model, tool_names)
    pipeline.record_inbound(method, path, headers, body)
    ...
    async for chunk in stream:
        abort = pipeline.feed_stream(chunk)
        if abort:
            yield error_event(abort)
            break
    ...
    pipeline.finish(wire_status=200, outcome=terminal_outcome)
"""

from __future__ import annotations

from typing import Any

from router_maestro.config import load_priorities_config
from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.pipeline.leak_guard import LeakGuard
from router_maestro.pipeline.runaway_guard import RunawayGuard
from router_maestro.providers.base import TerminalOutcome
from router_maestro.utils import get_logger
from router_maestro.utils.audit import AuditTrace, get_trace_dir, is_tracing_enabled

logger = get_logger("pipeline")


class RequestPipeline:
    """Per-request processing pipeline.

    Encapsulates guards and audit tracing. Create one per request,
    feed it stream chunks, and call finish() with the wire status and outcome.
    """

    __slots__ = (
        "_guards",
        "_audit",
        "_leak_guard",
        "_config",
        "_finished",
        "_outcome",
        "_wire_status",
        "_request_id",
    )

    def __init__(
        self,
        request_id: str,
        guards: list,
        leak_guard: LeakGuard | None,
        audit: AuditTrace | None,
        config: PrioritiesConfig,
    ):
        self._request_id = request_id
        self._guards = guards
        self._leak_guard = leak_guard
        self._audit = audit
        self._config = config
        self._finished = False
        self._outcome: TerminalOutcome | None = None
        self._wire_status: int | None = None

    @classmethod
    def create(
        cls,
        request_id: str,
        model: str,
        tool_names: set[str] | None = None,
    ) -> RequestPipeline:
        """Factory: build a pipeline from current configuration."""
        config = load_priorities_config()
        guards_cfg = config.guards

        guards: list = []
        leak_guard: LeakGuard | None = None

        if guards_cfg.leak_guard.enabled:
            leak_guard = LeakGuard(allowed_tool_names=tool_names)
            guards.append(leak_guard)

        if guards_cfg.runaway_guard.enabled:
            guards.append(
                RunawayGuard(
                    max_bytes=guards_cfg.runaway_guard.max_bytes,
                    max_deltas=guards_cfg.runaway_guard.max_deltas,
                )
            )

        audit: AuditTrace | None = None
        if is_tracing_enabled(config.audit.enabled):
            trace_dir = get_trace_dir(config.audit.trace_dir)
            audit = AuditTrace(request_id, trace_dir)

        return cls(
            request_id=request_id,
            guards=guards,
            leak_guard=leak_guard,
            audit=audit,
            config=config,
        )

    @property
    def audit(self) -> AuditTrace | None:
        return self._audit

    @property
    def config(self) -> PrioritiesConfig:
        return self._config

    @property
    def leak_guard(self) -> LeakGuard | None:
        return self._leak_guard

    @property
    def outcome(self) -> TerminalOutcome | None:
        """Return the semantic terminal outcome after finalization."""
        return self._outcome

    @property
    def wire_status(self) -> int | None:
        """Return the HTTP status recorded by the first finalization."""
        return self._wire_status

    @property
    def beta_strip_patterns(self) -> list[str]:
        return self._config.beta_strip

    def record_inbound(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: Any,
    ) -> None:
        """Record the inbound client request (if tracing)."""
        if self._audit:
            self._audit.record_inbound(method, path, headers, body)

    def record_upstream(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any,
    ) -> None:
        """Record the upstream request (if tracing)."""
        if self._audit:
            self._audit.record_upstream(method, url, headers, body)

    def record_upstream_response(
        self,
        status: int,
        headers: dict[str, str],
        body: Any = None,
        *,
        stream_summary: str | None = None,
    ) -> None:
        """Record the upstream response (if tracing)."""
        if self._audit:
            self._audit.record_upstream_response(
                status, headers, body, stream_summary=stream_summary
            )

    def feed_stream(self, chunk) -> str | None:
        """Feed a stream chunk through all guards.

        Returns an abort reason string if any guard trips, else None.
        Works with both ChatStreamChunk and ResponsesStreamChunk.
        """
        for guard in self._guards:
            reason = guard.feed_chunk(chunk)
            if reason:
                return reason
            content = getattr(chunk, "content", None)
            if content:
                reason = guard.feed_text(content)
                if reason:
                    return reason
        return None

    def feed_text(self, text: str) -> str | None:
        """Feed raw text (for passthrough routes that don't use ChatStreamChunk)."""
        for guard in self._guards:
            reason = guard.feed_text(text)
            if reason:
                return reason
        return None

    def check_invoke_at_finish(self) -> list[dict] | None:
        """Check for recoverable invoke leaks at stream end."""
        if self._leak_guard:
            return self._leak_guard.check_invoke_at_finish()
        return None

    def finish(
        self,
        *,
        wire_status: int,
        outcome: TerminalOutcome,
        body_summary: str | None = None,
    ) -> None:
        """Finalize once with the actual wire status and semantic outcome."""
        if self._finished:
            if self._wire_status != wire_status or self._outcome != outcome:
                logger.error(
                    "Request pipeline conflicting finalization: request_id=%s "
                    "first_wire_status=%s second_wire_status=%s "
                    "first_outcome=%r second_outcome=%r",
                    self._request_id,
                    self._wire_status,
                    wire_status,
                    self._outcome,
                    outcome,
                )
            return
        self._finished = True
        self._wire_status = wire_status
        self._outcome = outcome
        if self._audit:
            self._audit.record_outbound(
                wire_status,
                body_summary=body_summary,
                outcome=outcome,
            )
            self._audit.flush()
