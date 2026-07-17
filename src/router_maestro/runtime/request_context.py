"""One immutable configuration and Router lease for an inference request."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.providers.base import (
    ResponseStatus,
    TerminalOutcome,
    TransportTermination,
    client_cancelled_outcome,
    exception_outcome,
    unexpected_eof_outcome,
)
from router_maestro.utils.audit import AuditTrace, get_trace_dir, is_tracing_enabled

if TYPE_CHECKING:
    from router_maestro.pipeline.request_pipeline import RequestPipeline


class RuntimeConfigSnapshot(Protocol):
    """Task 15 snapshot surface consumed by request ownership."""

    revision: str

    @property
    def config(self) -> PrioritiesConfig: ...


class RuntimeConfigRepository(Protocol):
    def read(self) -> RuntimeConfigSnapshot: ...


class RouterLease(Protocol):
    generation_id: int
    router: Any
    config_snapshot: object | None

    async def release(self) -> None: ...


class RouterOwner(Protocol):
    async def acquire(self) -> RouterLease: ...


_current_request_context: ContextVar[RequestContext | None] = ContextVar(
    "router_maestro_request_context",
    default=None,
)


def get_current_request_context() -> RequestContext | None:
    """Return the bound inference context, if this task has one."""
    return _current_request_context.get()


def current_request_context() -> RequestContext:
    """Return the bound inference context or raise outside request execution."""
    context = get_current_request_context()
    if context is None:
        raise LookupError("No Router-Maestro request context is active")
    return context


def _completed_outcome() -> TerminalOutcome:
    return TerminalOutcome(
        transport=TransportTermination.EXPLICIT_TERMINAL,
        response_status=ResponseStatus.COMPLETED,
    )


def _http_outcome(status: int | None) -> TerminalOutcome:
    if status is not None and status >= 400:
        return TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=ResponseStatus.FAILED,
        )
    return _completed_outcome()


async def _await_cleanup(task: asyncio.Task[None]) -> None:
    """Defer caller cancellation until a resource cleanup task reaches its terminal state."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


@dataclass(slots=True)
class RequestContext:
    """Own request-scoped configuration, Router resources, pipeline, and audit."""

    request_id: str
    config_snapshot: RuntimeConfigSnapshot
    config: PrioritiesConfig
    lease: RouterLease
    revision: str = field(init=False)
    generation_id: int = field(init=False)
    router: Any = field(init=False)
    pipeline: RequestPipeline | None = None
    audit: AuditTrace | None = None
    stream_committed: bool = False
    outcome: TerminalOutcome | None = None
    status_code: int | None = None
    _finalized: bool = False
    _finalize_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _inbound_recorded: bool = False
    _request_method: str = ""
    _request_path: str = ""
    _request_headers: dict[str, str] = field(default_factory=dict, repr=False)
    _request_body: bytearray = field(default_factory=bytearray, repr=False)

    def __post_init__(self) -> None:
        self.revision = self.config_snapshot.revision
        self.generation_id = self.lease.generation_id
        self.router = self.lease.router

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        config_snapshot: RuntimeConfigSnapshot,
        lease: RouterLease,
    ) -> RequestContext:
        """Capture the snapshot config once and construct its optional audit trace."""
        config = config_snapshot.config
        audit = None
        if is_tracing_enabled(config.audit.enabled):
            audit = AuditTrace(request_id, get_trace_dir(config.audit.trace_dir))
        return cls(
            request_id=request_id,
            config_snapshot=config_snapshot,
            config=config,
            lease=lease,
            audit=audit,
        )

    @property
    def finalized(self) -> bool:
        return self._finalized

    def bind_request(self, scope: Scope) -> None:
        """Capture immutable inbound metadata before route execution."""
        self._request_method = scope.get("method", "")
        self._request_path = scope.get("path", "")
        self._request_headers = {
            name.decode("latin-1"): value.decode("latin-1")
            for name, value in scope.get("headers", ())
        }

    def append_request_body(self, body: bytes, *, complete: bool) -> None:
        self._request_body.extend(body)
        if complete:
            self.record_inbound()

    def record_inbound(self) -> None:
        if self._inbound_recorded or self.audit is None:
            return
        self._inbound_recorded = True
        self.audit.record_inbound(
            self._request_method,
            self._request_path,
            self._request_headers,
            bytes(self._request_body),
        )

    def request_header(self, name: str) -> str | None:
        lowered = name.lower()
        for header_name, value in self._request_headers.items():
            if header_name.lower() == lowered:
                return value
        return None

    async def finalize(
        self,
        *,
        outcome: TerminalOutcome | None = None,
        wire_status: int | None = None,
        body_summary: str | None = None,
    ) -> None:
        """Flush audit and release the generation lease exactly once."""
        async with self._finalize_lock:
            if self._finalized:
                return
            if wire_status is not None:
                self.status_code = wire_status
            pipeline_outcome = self.pipeline.outcome if self.pipeline is not None else None
            resolved_outcome = (
                outcome or self.outcome or pipeline_outcome or _http_outcome(self.status_code)
            )
            self.outcome = resolved_outcome
            status = self.status_code or 500

            async def finish() -> None:
                try:
                    self.record_inbound()
                    if self.pipeline is not None:
                        if self.pipeline.outcome is None:
                            self.pipeline.finish(
                                wire_status=status,
                                outcome=resolved_outcome,
                                body_summary=body_summary,
                            )
                        if self.audit is not None:
                            self.audit.record_outbound(
                                status,
                                body_summary=body_summary,
                                outcome=resolved_outcome,
                            )
                    elif self.audit is not None:
                        self.audit.record_outbound(
                            status,
                            body_summary=body_summary,
                            outcome=resolved_outcome,
                        )
                    if self.audit is not None:
                        await self.audit.flush_async()
                finally:
                    await self.lease.release()
                    self._finalized = True

            await _await_cleanup(asyncio.create_task(finish()))


def is_inference_path(path: str) -> bool:
    """Return whether a path executes or tokenizes an inference request."""
    if path in {
        "/api/openai/v1/chat/completions",
        "/api/openai/v1/responses",
        "/api/openai/beta/v1/responses",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/api/anthropic/v1/messages",
        "/api/anthropic/v1/messages/count_tokens",
        "/api/anthropic/beta/v1/messages",
        "/api/anthropic/beta/v1/messages/count_tokens",
    }:
        return True
    return path.startswith("/api/gemini/v1beta/models/") and path.endswith(
        (":generateContent", ":streamGenerateContent", ":countTokens")
    )


class RequestContextMiddleware:
    """Bind resource finalization to ASGI response-body completion."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_inference_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        app = scope.get("app")
        state = getattr(app, "state", None)
        repository = cast(RuntimeConfigRepository, getattr(state, "runtime_config_repository"))
        owner = cast(RouterOwner, getattr(state, "router_owner"))
        snapshot = repository.read()
        start = getattr(owner, "start", None)
        if callable(start):
            await start(snapshot)
        lease = await owner.acquire()
        request_id = scope.setdefault("state", {}).get("request_id", "")
        lease_snapshot = getattr(lease, "config_snapshot", None)
        context = RequestContext.create(
            request_id=request_id,
            config_snapshot=(
                cast(RuntimeConfigSnapshot, lease_snapshot)
                if lease_snapshot is not None
                else snapshot
            ),
            lease=lease,
        )
        context.bind_request(scope)
        scope["state"]["request_context"] = context
        token: Token[RequestContext | None] = _current_request_context.set(context)
        final_body_sent = False
        client_disconnected = False

        async def receive_with_context() -> Message:
            nonlocal client_disconnected
            message = await receive()
            if message["type"] == "http.request":
                context.append_request_body(
                    message.get("body", b""),
                    complete=not message.get("more_body", False),
                )
            elif message["type"] == "http.disconnect":
                client_disconnected = True
            return message

        async def send_with_context(message: Message) -> None:
            nonlocal final_body_sent
            if message["type"] == "http.response.start":
                context.status_code = message["status"]
                context.stream_committed = True
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                final_body_sent = True
                await context.finalize(
                    outcome=client_cancelled_outcome() if client_disconnected else None,
                    wire_status=context.status_code,
                )

        try:
            await self.app(scope, receive_with_context, send_with_context)
            if not final_body_sent:
                await context.finalize(
                    outcome=(
                        client_cancelled_outcome()
                        if client_disconnected
                        else unexpected_eof_outcome()
                    ),
                    wire_status=context.status_code,
                    body_summary="ASGI application returned before the final response body",
                )
        except asyncio.CancelledError:
            await context.finalize(
                outcome=client_cancelled_outcome(),
                wire_status=context.status_code,
            )
            raise
        except (ClientDisconnect, OSError):
            await context.finalize(
                outcome=client_cancelled_outcome(),
                wire_status=context.status_code,
            )
            raise
        except Exception as error:
            await context.finalize(
                outcome=exception_outcome(str(error)),
                wire_status=context.status_code,
                body_summary=str(error),
            )
            raise
        finally:
            _current_request_context.reset(token)
