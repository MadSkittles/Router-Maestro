"""Pure-ASGI request ownership and finalization contracts."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse

from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.providers.base import (
    ResponseStatus,
    TransportTermination,
    client_cancelled_outcome,
)
from router_maestro.providers.copilot_support.transport import CopilotTransport
from router_maestro.runtime.request_context import (
    RequestContext,
    RequestContextMiddleware,
    current_request_context,
)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    revision: str
    value: str

    @property
    def config(self) -> PrioritiesConfig:
        return PrioritiesConfig(priorities=[self.value])


class _Repository:
    def __init__(self, snapshot: _Snapshot) -> None:
        self.snapshot = snapshot
        self.read_count = 0

    def read(self) -> _Snapshot:
        self.read_count += 1
        return self.snapshot


class _Lease:
    def __init__(
        self,
        generation_id: int,
        router: object,
        config_snapshot: object | None = None,
    ) -> None:
        self.generation_id = generation_id
        self.router = router
        self.config_snapshot = config_snapshot
        self.release_count = 0

    async def release(self) -> None:
        self.release_count += 1


class _Owner:
    def __init__(self, lease: _Lease) -> None:
        self.lease = lease
        self.acquire_count = 0

    async def acquire(self) -> _Lease:
        self.acquire_count += 1
        return self.lease


def _scope(repository: _Repository, owner: _Owner, path: str = "/v1/messages") -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
        "state": {"request_id": "req-1"},
        "app": SimpleNamespace(
            state=SimpleNamespace(
                runtime_config_repository=repository,
                router_owner=owner,
            )
        ),
    }


async def _receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


async def _discard_send(_message: dict[str, Any]) -> None:
    return None


@pytest.mark.asyncio
async def test_endpoint_return_does_not_finalize_before_final_body() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(11, object())
    owner = _Owner(lease)
    endpoint_returned = asyncio.Event()
    allow_final_body = asyncio.Event()
    sent: list[dict[str, Any]] = []
    captured: list[RequestContext] = []

    async def app(scope, receive, send) -> None:
        context = current_request_context()
        captured.append(context)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        endpoint_returned.set()
        await allow_final_body.wait()
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    middleware = RequestContextMiddleware(app)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    task = asyncio.create_task(middleware(_scope(repository, owner), _receive, send))
    await endpoint_returned.wait()

    assert lease.release_count == 0
    assert captured[0].finalized is False

    allow_final_body.set()
    await task
    assert lease.release_count == 1
    assert captured[0].finalized is True
    assert captured[0].stream_committed is True
    assert captured[0].outcome is not None
    assert captured[0].outcome.transport is TransportTermination.EXPLICIT_TERMINAL
    assert captured[0].outcome.response_status is ResponseStatus.COMPLETED


@pytest.mark.asyncio
async def test_exception_before_response_finalizes_and_releases_once() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(1, object())
    owner = _Owner(lease)
    captured: list[RequestContext] = []

    async def app(scope, receive, send) -> None:
        captured.append(current_request_context())
        raise RuntimeError("boom")

    middleware = RequestContextMiddleware(app)
    with pytest.raises(RuntimeError, match="boom"):
        await middleware(_scope(repository, owner), _receive, _discard_send)

    assert lease.release_count == 1
    assert captured[0].outcome is not None
    assert captured[0].outcome.transport is TransportTermination.EXCEPTION
    await captured[0].finalize()
    assert lease.release_count == 1


@pytest.mark.asyncio
async def test_cancelled_request_records_cancel_and_releases_once() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(1, object())
    owner = _Owner(lease)
    started = asyncio.Event()
    captured: list[RequestContext] = []

    async def app(scope, receive, send) -> None:
        captured.append(current_request_context())
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        RequestContextMiddleware(app)(
            _scope(repository, owner),
            _receive,
            _discard_send,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert lease.release_count == 1
    assert captured[0].outcome == client_cancelled_outcome()


@pytest.mark.asyncio
async def test_disconnect_message_records_client_cancel_when_app_returns_normally() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(1, object())
    owner = _Owner(lease)
    captured: list[RequestContext] = []

    async def receive_disconnect() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def app(scope, receive, send) -> None:
        captured.append(current_request_context())
        assert await receive() == {"type": "http.disconnect"}

    await RequestContextMiddleware(app)(
        _scope(repository, owner),
        receive_disconnect,
        _discard_send,
    )

    assert lease.release_count == 1
    assert captured[0].outcome == client_cancelled_outcome()


@pytest.mark.asyncio
async def test_streaming_response_disconnect_listener_records_client_cancel() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(1, object())
    owner = _Owner(lease)
    captured: list[RequestContext] = []

    async def receive_disconnect() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def stream():
        await asyncio.Event().wait()
        yield b"unreachable"

    async def app(scope, receive, send) -> None:
        captured.append(current_request_context())
        await StreamingResponse(stream())(scope, receive, send)

    scope = _scope(repository, owner)
    scope["asgi"] = {"spec_version": "2.3"}
    await RequestContextMiddleware(app)(scope, receive_disconnect, _discard_send)

    assert lease.release_count == 1
    assert captured[0].outcome == client_cancelled_outcome()


@pytest.mark.asyncio
async def test_streaming_response_send_disconnect_records_client_cancel() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(1, object())
    owner = _Owner(lease)
    captured: list[RequestContext] = []

    async def stream():
        yield b"chunk"

    async def app(scope, receive, send) -> None:
        captured.append(current_request_context())
        await StreamingResponse(stream())(scope, receive, send)

    async def disconnected_send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client socket closed")

    scope = _scope(repository, owner)
    scope["asgi"] = {"spec_version": "2.4"}
    with pytest.raises(ClientDisconnect):
        await RequestContextMiddleware(app)(scope, _receive, disconnected_send)

    assert lease.release_count == 1
    assert captured[0].outcome == client_cancelled_outcome()


@pytest.mark.asyncio
async def test_cancel_during_finalize_waits_for_release_before_propagating() -> None:
    class BlockingLease(_Lease):
        def __init__(self) -> None:
            super().__init__(1, object())
            self.release_started = asyncio.Event()
            self.finish_release = asyncio.Event()

        async def release(self) -> None:
            self.release_started.set()
            await self.finish_release.wait()
            self.release_count += 1

    snapshot = _Snapshot("rev", "config")
    lease = BlockingLease()
    context = RequestContext(
        request_id="req",
        config_snapshot=snapshot,
        config=snapshot.config,
        lease=lease,
    )
    finalize_task = asyncio.create_task(context.finalize(wire_status=200))
    await lease.release_started.wait()

    finalize_task.cancel()
    await asyncio.sleep(0)
    finalize_task.cancel()
    await asyncio.sleep(0)
    assert not finalize_task.done()
    assert context.finalized is False

    lease.finish_release.set()
    with pytest.raises(asyncio.CancelledError):
        await finalize_task
    assert context.finalized is True
    assert lease.release_count == 1
    await context.finalize(wire_status=200)
    assert lease.release_count == 1


@pytest.mark.asyncio
async def test_unexpected_eof_finalizes_when_app_returns_without_final_body() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(1, object())
    owner = _Owner(lease)
    captured: list[RequestContext] = []

    async def app(scope, receive, send) -> None:
        captured.append(current_request_context())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})

    await RequestContextMiddleware(app)(
        _scope(repository, owner),
        _receive,
        _discard_send,
    )

    assert lease.release_count == 1
    assert captured[0].outcome is not None
    assert captured[0].outcome.transport is TransportTermination.UNEXPECTED_EOF


@pytest.mark.asyncio
async def test_request_captures_one_snapshot_and_generation_for_entire_body() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    router_a = object()
    lease = _Lease(41, router_a)
    owner = _Owner(lease)
    captured: list[tuple[str, int, object, list[str]]] = []

    async def app(scope, receive, send) -> None:
        context = current_request_context()
        captured.append(
            (
                context.revision,
                context.generation_id,
                context.router,
                context.config.priorities,
            )
        )
        repository.snapshot = _Snapshot("rev-b", "config-b")
        owner.lease = _Lease(42, object())
        captured.append(
            (
                context.revision,
                context.generation_id,
                context.router,
                context.config.priorities,
            )
        )
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    await RequestContextMiddleware(app)(
        _scope(repository, owner),
        _receive,
        _discard_send,
    )

    assert repository.read_count == 1
    assert owner.acquire_count == 1
    assert captured == [
        ("rev-a", 41, router_a, ["config-a"]),
        ("rev-a", 41, router_a, ["config-a"]),
    ]


@pytest.mark.asyncio
async def test_request_uses_the_snapshot_owned_by_the_leased_generation() -> None:
    repository = _Repository(_Snapshot("repo-new", "config-new"))
    generation_snapshot = _Snapshot("generation-old", "config-old")
    lease = _Lease(41, object(), generation_snapshot)
    owner = _Owner(lease)
    captured: list[tuple[str, list[str]]] = []

    async def app(scope, receive, send) -> None:
        context = current_request_context()
        captured.append((context.revision, context.config.priorities))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    await RequestContextMiddleware(app)(
        _scope(repository, owner),
        _receive,
        _discard_send,
    )

    assert repository.read_count == 1
    assert captured == [("generation-old", ["config-old"])]


@pytest.mark.asyncio
async def test_non_inference_path_does_not_capture_snapshot_or_lease() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(1, object())
    owner = _Owner(lease)

    async def app(scope, receive, send) -> None:
        with pytest.raises(LookupError):
            current_request_context()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    await RequestContextMiddleware(app)(
        _scope(repository, owner, path="/health"),
        _receive,
        _discard_send,
    )

    assert repository.read_count == 0
    assert owner.acquire_count == 0
    assert lease.release_count == 0


@pytest.mark.asyncio
async def test_beta_header_strip_uses_captured_snapshot() -> None:
    config = PrioritiesConfig(priorities=[])
    config.beta_strip = ["remove-*", "exact"]

    @dataclass(frozen=True, slots=True)
    class Snapshot:
        revision: str = "beta-revision"

        @property
        def config(self) -> PrioritiesConfig:
            return config.model_copy(deep=True)

    repository = _Repository(_Snapshot("ignored", "ignored"))
    repository.snapshot = Snapshot()  # type: ignore[assignment]
    lease = _Lease(1, object())
    owner = _Owner(lease)
    observed: list[str | None] = []

    async def app(scope, receive, send) -> None:
        auth = SimpleNamespace(cached_token="token", provider_name="github-copilot")
        observed.append(CopilotTransport(auth).headers().get("anthropic-beta"))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    scope = _scope(repository, owner)
    scope["headers"] = [
        (b"anthropic-beta", b"keep,remove-one,exact"),
    ]
    await RequestContextMiddleware(app)(scope, _receive, _discard_send)

    assert observed == ["keep"]


@pytest.mark.asyncio
async def test_pipeline_outbound_audit_is_deferred_until_context_finalize() -> None:
    from router_maestro.pipeline.request_pipeline import RequestPipeline
    from router_maestro.providers.base import unexpected_eof_outcome

    class AuditSpy:
        def __init__(self) -> None:
            self.outbound_count = 0
            self.flush_count = 0

        def record_outbound(self, *_args, **_kwargs) -> None:
            self.outbound_count += 1

        def record_inbound(self, *_args, **_kwargs) -> None:
            return None

        async def flush_async(self) -> None:
            self.flush_count += 1

    audit = AuditSpy()
    snapshot = _Snapshot("rev", "config")
    lease = _Lease(1, object())
    context = RequestContext(
        request_id="req",
        config_snapshot=snapshot,
        config=snapshot.config,
        lease=lease,
        audit=audit,  # type: ignore[arg-type]
    )
    context.pipeline = RequestPipeline(
        request_id="req",
        guards=[],
        leak_guard=None,
        audit=audit,  # type: ignore[arg-type]
        config=context.config,
        defer_flush=True,
    )
    outcome = unexpected_eof_outcome()

    context.pipeline.finish(wire_status=200, outcome=outcome)
    assert audit.outbound_count == 0

    await context.finalize(wire_status=200)
    assert audit.outbound_count == 1
    assert audit.flush_count == 1


@pytest.mark.asyncio
async def test_early_auth_response_still_finalizes_captured_inference_context() -> None:
    repository = _Repository(_Snapshot("rev-a", "config-a"))
    lease = _Lease(1, object())
    owner = _Owner(lease)
    auth_rejection_context: list[RequestContext] = []

    async def auth_rejection(scope, receive, send) -> None:
        auth_rejection_context.append(current_request_context())
        await send({"type": "http.response.start", "status": 401, "headers": []})
        await send({"type": "http.response.body", "body": b"no", "more_body": False})

    await RequestContextMiddleware(auth_rejection)(
        _scope(repository, owner),
        _receive,
        _discard_send,
    )

    assert repository.read_count == 1
    assert owner.acquire_count == 1
    assert lease.release_count == 1
    context = auth_rejection_context[0]
    assert context.outcome is not None
    assert context.outcome.transport is TransportTermination.EXPLICIT_TERMINAL
    assert context.outcome.response_status is ResponseStatus.FAILED
