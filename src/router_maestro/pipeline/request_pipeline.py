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
from router_maestro.pipeline.guards import GuardChain, StreamGuard
from router_maestro.pipeline.leak_guard import LeakGuard
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
        "_guard_chain",
        "_audit",
        "_leak_guard",
        "_config",
        "_finished",
        "_outcome",
        "_wire_status",
        "_request_id",
        "_defer_flush",
    )

    def __init__(
        self,
        request_id: str,
        guards: list[StreamGuard] | GuardChain,
        leak_guard: LeakGuard | None,
        audit: AuditTrace | None,
        config: PrioritiesConfig,
        *,
        defer_flush: bool = False,
    ):
        self._request_id = request_id
        self._guard_chain = guards if isinstance(guards, GuardChain) else GuardChain(guards)
        self._leak_guard = leak_guard
        self._audit = audit
        self._config = config
        self._finished = False
        self._outcome: TerminalOutcome | None = None
        self._wire_status: int | None = None
        self._defer_flush = defer_flush

    @classmethod
    def create(
        cls,
        request_id: str,
        model: str,
        tool_names: set[str] | None = None,
    ) -> RequestPipeline:
        """Factory: build a pipeline from current configuration."""
        context = None
        try:
            from router_maestro.runtime.request_context import get_current_request_context

            context = get_current_request_context()
        except ImportError:
            pass
        if context is not None and context.pipeline is not None:
            return context.pipeline

        config = context.config if context is not None else load_priorities_config()
        guard_chain = GuardChain.from_config(
            config.guards,
            model=model,
            tool_names=tool_names,
        )
        leak_guard = guard_chain.find(LeakGuard)

        audit = context.audit if context is not None else None
        if context is None and is_tracing_enabled(config.audit.enabled):
            trace_dir = get_trace_dir(config.audit.trace_dir)
            audit = AuditTrace(request_id, trace_dir)

        pipeline = cls(
            request_id=request_id,
            guards=guard_chain,
            leak_guard=leak_guard,
            audit=audit,
            config=config,
            defer_flush=context is not None,
        )
        if context is not None:
            context.pipeline = pipeline
        return pipeline

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
        return self._guard_chain.feed_chunk(chunk)

    def feed_text(self, text: str) -> str | None:
        """Feed raw text (for passthrough routes that don't use ChatStreamChunk)."""
        return self._guard_chain.feed_text(text)

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
        if self._audit and not self._defer_flush:
            self._audit.record_outbound(
                wire_status,
                body_summary=body_summary,
                outcome=outcome,
            )
            self._audit.flush()
