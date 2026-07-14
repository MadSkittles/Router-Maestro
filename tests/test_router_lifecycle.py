"""Router generation and model-catalog lifecycle contracts."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.providers.copilot_support.catalog import CopilotCatalog
from router_maestro.routing.router import Router, RouterOwner
from router_maestro.runtime.request_context import RequestContext, get_current_request_context
from router_maestro.server.app import lifespan


class _ClosingProvider(BaseProvider):
    name = "test"

    def __init__(self) -> None:
        self.close_count = 0

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        if False:
            yield ChatStreamChunk()

    async def list_models(self) -> list[ModelInfo]:
        return []

    def is_authenticated(self) -> bool:
        return True

    async def close(self) -> None:
        self.close_count += 1


@dataclass
class _GenerationRouter:
    label: str
    provider: _ClosingProvider

    @property
    def providers(self) -> dict[str, BaseProvider]:
        return {"test": self.provider}


class _RouterFactory:
    def __init__(self) -> None:
        self.routers: list[_GenerationRouter] = []
        self.fail_for: set[str] = set()

    def __call__(self, snapshot: object | None = None) -> _GenerationRouter:
        label = str(snapshot)
        if label in self.fail_for:
            raise RuntimeError(f"cannot build {label}")
        router = _GenerationRouter(label, _ClosingProvider())
        self.routers.append(router)
        return router


@pytest.mark.asyncio
async def test_stream_lease_keeps_retired_generation_open_until_release() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("A")

    lease_a = await owner.acquire()
    generation_a = lease_a.generation_id
    router_a = lease_a.router

    generation_b = await owner.rebuild("B")
    lease_b = await owner.acquire()

    assert generation_b != generation_a
    assert lease_b.generation_id == generation_b
    assert lease_b.router is factory.routers[1]
    assert router_a.provider.close_count == 0

    await lease_a.release()
    assert router_a.provider.close_count == 1

    await lease_b.release()
    await owner.close()
    assert factory.routers[1].provider.close_count == 1


@pytest.mark.asyncio
async def test_lease_release_is_idempotent_and_async_context_managed() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("A")
    lease = await owner.acquire()
    await owner.rebuild("B")

    async with lease as entered:
        assert entered is lease
    await lease.release()

    assert factory.routers[0].provider.close_count == 1
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_lease_holder_releases_retired_generation_once() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("A")
    entered = asyncio.Event()
    hold = asyncio.Event()

    async def serve() -> None:
        async with await owner.acquire():
            entered.set()
            await hold.wait()

    task = asyncio.create_task(serve())
    await entered.wait()
    await owner.rebuild("B")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert factory.routers[0].provider.close_count == 1
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_release_finishes_reference_drop_before_propagating_cancel() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("A")
    lease = await owner.acquire()
    generation = owner._generations[lease.generation_id]

    await owner._state_lock.acquire()
    release_task = asyncio.create_task(lease.release())
    await asyncio.sleep(0)
    release_task.cancel()
    await asyncio.sleep(0)
    release_task.cancel()
    await asyncio.sleep(0)
    owner._state_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await release_task
    assert lease.released is True
    assert generation.references == 0
    await asyncio.wait_for(owner.close(), timeout=1)


@pytest.mark.asyncio
async def test_failed_rebuild_preserves_active_generation() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    generation_a = await owner.start("A")
    factory.fail_for.add("broken")

    with pytest.raises(RuntimeError, match="cannot build broken"):
        await owner.rebuild("broken")

    lease = await owner.acquire()
    assert lease.generation_id == generation_a
    assert lease.router is factory.routers[0]
    assert factory.routers[0].provider.close_count == 0
    await lease.release()
    await owner.close()


@pytest.mark.asyncio
async def test_rebuild_without_snapshot_inherits_active_generation_snapshot() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("snapshot-a")

    await owner.rebuild()
    lease = await owner.acquire()

    assert lease.config_snapshot == "snapshot-a"
    assert lease.router.label == "snapshot-a"
    await lease.release()
    await owner.close()


@pytest.mark.asyncio
async def test_acquire_rebuilds_stale_managed_generation_without_mutating_old_lease() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("snapshot-a")
    old_lease = await owner.acquire()
    stale = True

    def needs_reload() -> bool:
        return stale

    old_lease.router.needs_provider_config_reload = needs_reload  # type: ignore[attr-defined]
    refreshed_lease = await owner.acquire()
    stale = False

    assert refreshed_lease.generation_id != old_lease.generation_id
    assert refreshed_lease.config_snapshot == "snapshot-a"
    assert refreshed_lease.router is factory.routers[1]
    assert old_lease.router is factory.routers[0]
    assert old_lease.router.provider.close_count == 0

    await refreshed_lease.release()
    await old_lease.release()
    assert factory.routers[0].provider.close_count == 1
    await owner.close()


@pytest.mark.asyncio
async def test_concurrent_acquires_share_one_provider_config_refresh() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("snapshot-a")
    factory.routers[0].needs_provider_config_reload = lambda: True  # type: ignore[attr-defined]

    leases = await asyncio.gather(*(owner.acquire() for _ in range(10)))

    assert len(factory.routers) == 2
    assert len({lease.generation_id for lease in leases}) == 1
    for lease in leases:
        await lease.release()
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_build_closes_candidate_after_factory_finishes() -> None:
    routers: list[_GenerationRouter] = []
    candidate_created = asyncio.Event()
    finish_build = asyncio.Event()
    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    class BlockingRouter(_GenerationRouter):
        async def close(self) -> None:
            close_started.set()
            await finish_close.wait()
            await self.provider.close()

    async def factory(snapshot: object | None) -> _GenerationRouter:
        router = (
            BlockingRouter(str(snapshot), _ClosingProvider())
            if snapshot == "B"
            else _GenerationRouter(str(snapshot), _ClosingProvider())
        )
        routers.append(router)
        if snapshot == "B":
            candidate_created.set()
            await finish_build.wait()
        return router

    owner = RouterOwner(factory)
    await owner.start("A")
    rebuild_task = asyncio.create_task(owner.rebuild("B"))
    await candidate_created.wait()

    rebuild_task.cancel()
    finish_build.set()
    await close_started.wait()
    rebuild_task.cancel()
    await asyncio.sleep(0)
    finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await rebuild_task

    assert routers[1].provider.close_count == 1
    lease = await owner.acquire()
    assert lease.router is routers[0]
    await lease.release()
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_install_closes_built_candidate() -> None:
    factory = _RouterFactory()
    candidate_built = asyncio.Event()
    finish_build = asyncio.Event()

    async def build(snapshot: object | None) -> _GenerationRouter:
        router = factory(snapshot)
        if snapshot == "B":
            candidate_built.set()
            await finish_build.wait()
        return router

    owner = RouterOwner(build)
    await owner.start("A")
    rebuild_task = asyncio.create_task(owner.rebuild("B"))
    await candidate_built.wait()
    await owner._state_lock.acquire()
    finish_build.set()
    await asyncio.sleep(0)
    rebuild_task.cancel()
    owner._state_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await rebuild_task
    assert factory.routers[1].provider.close_count == 1
    lease = await owner.acquire()
    assert lease.router is factory.routers[0]
    await lease.release()
    await owner.close()


@pytest.mark.asyncio
async def test_before_swap_failure_closes_candidate_without_replacing_active() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    generation_a = await owner.start("A")

    def reject_swap() -> object:
        raise RuntimeError("CAS conflict")

    with pytest.raises(RuntimeError, match="CAS conflict"):
        await owner.rebuild("replacement", before_swap=reject_swap)

    assert factory.routers[1].provider.close_count == 1
    lease = await owner.acquire()
    assert lease.generation_id == generation_a
    assert lease.router is factory.routers[0]
    await lease.release()
    await owner.close()


@pytest.mark.asyncio
async def test_before_swap_return_becomes_installed_generation_snapshot() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("A")
    committed_snapshot = object()
    observed_active: list[_GenerationRouter] = []

    def commit() -> object:
        assert owner._active is not None
        observed_active.append(owner._active.router)
        return committed_snapshot

    await owner.rebuild("replacement", before_swap=commit)
    lease = await owner.acquire()

    assert observed_active == [factory.routers[0]]
    assert lease.router.label == "replacement"
    assert lease.config_snapshot is committed_snapshot
    await lease.release()
    await owner.close()


@pytest.mark.asyncio
async def test_rebuild_returns_committed_before_retired_generation_finishes_closing() -> None:
    close_started = asyncio.Event()
    finish_close = asyncio.Event()
    routers: list[_GenerationRouter] = []

    class BlockingRouter(_GenerationRouter):
        async def close(self) -> None:
            close_started.set()
            await finish_close.wait()
            await self.provider.close()

    def build(snapshot: object | None) -> _GenerationRouter:
        if snapshot == "A":
            router = BlockingRouter("A", _ClosingProvider())
        else:
            router = _GenerationRouter(str(snapshot), _ClosingProvider())
        routers.append(router)
        return router

    owner = RouterOwner(build)
    await owner.start("A")
    rebuild_task = asyncio.create_task(owner.rebuild("B"))
    await close_started.wait()
    await asyncio.sleep(0)

    assert rebuild_task.done()
    generation_b = rebuild_task.result()
    lease = await owner.acquire()
    assert lease.generation_id == generation_b
    assert lease.router is routers[1]
    assert routers[0].provider.close_count == 0

    close_owner = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    assert not close_owner.done()
    finish_close.set()
    await lease.release()
    await close_owner
    assert routers[0].provider.close_count == 1


@pytest.mark.asyncio
async def test_lease_carries_the_snapshot_that_built_its_generation() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("snapshot-a")
    lease_a = await owner.acquire()
    await owner.rebuild("snapshot-b")
    lease_b = await owner.acquire()

    assert lease_a.config_snapshot == "snapshot-a"
    assert lease_b.config_snapshot == "snapshot-b"

    await lease_a.release()
    await lease_b.release()
    await owner.close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_active_lease_then_closes_exactly_once() -> None:
    factory = _RouterFactory()
    owner = RouterOwner(factory)
    await owner.start("A")
    lease = await owner.acquire()

    close_task = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    assert factory.routers[0].provider.close_count == 0

    await lease.release()
    await close_task
    await owner.close()
    assert factory.routers[0].provider.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        await owner.acquire()


@pytest.mark.asyncio
async def test_cancelled_shutdown_finishes_provider_close_before_propagating() -> None:
    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    class BlockingProvider(_ClosingProvider):
        async def close(self) -> None:
            close_started.set()
            await finish_close.wait()
            await super().close()

    router = _GenerationRouter("A", BlockingProvider())
    owner = RouterOwner(lambda _snapshot: router)
    await owner.start("A")

    close_task = asyncio.create_task(owner.close())
    await close_started.wait()
    concurrent_close = asyncio.create_task(owner.close())
    close_task.cancel()
    await asyncio.sleep(0)

    assert not close_task.done()
    assert not concurrent_close.done()
    assert router.provider.close_count == 0

    finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    await concurrent_close

    assert router.provider.close_count == 1
    await owner.close()
    assert router.provider.close_count == 1


@pytest.mark.asyncio
async def test_cancelled_shutdown_finishes_every_generation_close() -> None:
    close_started = {label: asyncio.Event() for label in ("A", "B", "C")}
    finish_close = {label: asyncio.Event() for label in ("A", "B", "C")}
    routers: list[_GenerationRouter] = []

    class BlockingProvider(_ClosingProvider):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        async def close(self) -> None:
            close_started[self.label].set()
            await finish_close[self.label].wait()
            await super().close()

    def factory(snapshot: object | None) -> _GenerationRouter:
        label = str(snapshot)
        router = _GenerationRouter(label, BlockingProvider(label))
        routers.append(router)
        return router

    owner = RouterOwner(factory)
    await owner.start("A")
    await owner.rebuild("B")
    await close_started["A"].wait()
    await owner.rebuild("C")
    await close_started["B"].wait()

    close_task = asyncio.create_task(owner.close())
    await close_started["C"].wait()
    close_task.cancel()
    await asyncio.sleep(0)
    close_task.cancel()
    await asyncio.sleep(0)

    assert not close_task.done()
    for event in finish_close.values():
        event.set()

    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert [router.provider.close_count for router in routers] == [1, 1, 1]
    assert owner._closed is True
    assert owner._generations == {}
    await owner.close()
    assert [router.provider.close_count for router in routers] == [1, 1, 1]


@pytest.mark.asyncio
async def test_router_close_retries_only_provider_interrupted_by_cancellation(monkeypatch) -> None:
    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    class BlockingProvider(_ClosingProvider):
        attempts = 0

        async def close(self) -> None:
            self.attempts += 1
            close_started.set()
            await finish_close.wait()
            await super().close()

    monkeypatch.setattr(Router, "_load_providers", lambda self: None)
    router = Router()
    closed_first = _ClosingProvider()
    interrupted = BlockingProvider()
    router.providers = {"first": closed_first, "interrupted": interrupted}

    close_task = asyncio.create_task(router.close())
    await close_started.wait()
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert closed_first.close_count == 1
    assert interrupted.close_count == 0
    assert interrupted.attempts == 1
    assert router._closed is False

    finish_close.set()
    await router.close()
    await router.close()

    assert closed_first.close_count == 1
    assert interrupted.close_count == 1
    assert interrupted.attempts == 2
    assert router._closed is True


@pytest.mark.asyncio
async def test_lifespan_startup_cancellation_releases_prewarm_lease() -> None:
    routers: list[_GenerationRouter] = []
    prewarm_started = asyncio.Event()

    class PrewarmRouter(_GenerationRouter):
        async def list_models(self) -> list[ModelInfo]:
            prewarm_started.set()
            await asyncio.Event().wait()
            return []

    def build(value: object | None) -> PrewarmRouter:
        router = PrewarmRouter(str(value), _ClosingProvider())
        routers.append(router)
        return router

    owner = RouterOwner(build)
    snapshot = object()
    repository = type("Repository", (), {"read": lambda self: snapshot})()
    state = type(
        "State",
        (),
        {"runtime_config_repository": repository, "router_owner": owner},
    )()
    app = type("App", (), {"state": state})()
    manager = lifespan(app)
    startup = asyncio.create_task(manager.__aenter__())
    await prewarm_started.wait()

    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup

    assert routers[0].provider.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        await owner.acquire()


def _catalog_response(model_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"id": model_id}]},
        request=httpx.Request("GET", "https://catalog.invalid/models"),
    )


async def _noop_token() -> None:
    return None


def _normalize(_model: Any) -> tuple[str, ...] | None:
    return None


def _operations(_model: Any) -> dict[str, bool]:
    return {}


def _protocol_error(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("valid catalog must not raise a protocol error")


@pytest.mark.asyncio
async def test_concurrent_cold_catalog_requests_share_one_refresh() -> None:
    catalog = CopilotCatalog()
    send_count = 0
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def send(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal send_count
        send_count += 1
        send_started.set()
        await release_send.wait()
        return _catalog_response("fresh")

    calls = [
        asyncio.create_task(
            catalog.list_models(
                provider_name="github-copilot",
                ensure_token=_noop_token,
                send=send,
                normalize_endpoints=_normalize,
                derive_operations=_operations,
                raise_protocol_error=_protocol_error,
            )
        )
        for _ in range(10)
    ]
    await send_started.wait()
    assert send_count == 1
    release_send.set()

    results = await asyncio.gather(*calls)
    assert send_count == 1
    assert [[model.id for model in result] for result in results] == [["fresh"]] * 10
    assert len({id(result) for result in results}) == 10


@pytest.mark.asyncio
async def test_stale_catalog_is_served_while_one_refresh_runs() -> None:
    catalog = CopilotCatalog()
    stale = [ModelInfo(id="stale", name="Stale", provider="github-copilot")]
    catalog.models_ttl_cache.set(stale)
    catalog.models_ttl_cache._timestamp = 0
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    send_count = 0

    async def send(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal send_count
        send_count += 1
        send_started.set()
        await release_send.wait()
        return _catalog_response("fresh")

    first = await catalog.list_models(
        provider_name="github-copilot",
        ensure_token=_noop_token,
        send=send,
        normalize_endpoints=_normalize,
        derive_operations=_operations,
        raise_protocol_error=_protocol_error,
    )
    await send_started.wait()
    second = await catalog.list_models(
        provider_name="github-copilot",
        ensure_token=_noop_token,
        send=send,
        normalize_endpoints=_normalize,
        derive_operations=_operations,
        raise_protocol_error=_protocol_error,
    )

    assert [model.id for model in first] == ["stale"]
    assert [model.id for model in second] == ["stale"]
    assert first is not second
    assert send_count == 1

    release_send.set()
    refreshed = await catalog.list_models(
        force_refresh=True,
        provider_name="github-copilot",
        ensure_token=_noop_token,
        send=send,
        normalize_endpoints=_normalize,
        derive_operations=_operations,
        raise_protocol_error=_protocol_error,
    )
    assert [model.id for model in refreshed] == ["fresh"]
    assert send_count == 1


@pytest.mark.asyncio
async def test_stale_catalog_refresh_runs_without_request_context() -> None:
    catalog = CopilotCatalog()
    stale = [ModelInfo(id="stale", name="Stale", provider="github-copilot")]
    catalog.models_ttl_cache.set(stale)
    catalog.models_ttl_cache._timestamp = 0
    refresh_contexts: list[RequestContext | None] = []
    refresh_finished = asyncio.Event()

    async def send(*_args: Any, **_kwargs: Any) -> httpx.Response:
        refresh_contexts.append(get_current_request_context())
        refresh_finished.set()
        return _catalog_response("fresh")

    from router_maestro.runtime import request_context as request_context_module

    token = request_context_module._current_request_context.set(object())  # type: ignore[arg-type]
    try:
        models = await catalog.list_models(
            provider_name="github-copilot",
            ensure_token=_noop_token,
            send=send,
            normalize_endpoints=_normalize,
            derive_operations=_operations,
            raise_protocol_error=_protocol_error,
        )
    finally:
        request_context_module._current_request_context.reset(token)

    assert [model.id for model in models] == ["stale"]
    await refresh_finished.wait()
    await catalog.aclose()
    assert refresh_contexts == [None]


@pytest.mark.asyncio
async def test_cold_catalog_refresh_inherits_awaiting_request_context() -> None:
    catalog = CopilotCatalog()
    request_context = object()
    refresh_contexts: list[object | None] = []

    async def send(*_args: Any, **_kwargs: Any) -> httpx.Response:
        refresh_contexts.append(get_current_request_context())
        return _catalog_response("fresh")

    from router_maestro.runtime import request_context as request_context_module

    token = request_context_module._current_request_context.set(request_context)  # type: ignore[arg-type]
    try:
        models = await catalog.list_models(
            provider_name="github-copilot",
            ensure_token=_noop_token,
            send=send,
            normalize_endpoints=_normalize,
            derive_operations=_operations,
            raise_protocol_error=_protocol_error,
        )
    finally:
        request_context_module._current_request_context.reset(token)

    assert [model.id for model in models] == ["fresh"]
    assert refresh_contexts == [request_context]
    await catalog.aclose()


@pytest.mark.asyncio
async def test_catalog_aclose_cancels_and_awaits_active_refresh() -> None:
    catalog = CopilotCatalog()
    stale = [ModelInfo(id="stale", name="Stale", provider="github-copilot")]
    catalog.models_ttl_cache.set(stale)
    catalog.models_ttl_cache._timestamp = 0
    refresh_started = asyncio.Event()
    refresh_finished = asyncio.Event()

    async def send(*_args: Any, **_kwargs: Any) -> httpx.Response:
        refresh_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            refresh_finished.set()
        raise AssertionError("unreachable")

    await catalog.list_models(
        provider_name="github-copilot",
        ensure_token=_noop_token,
        send=send,
        normalize_endpoints=_normalize,
        derive_operations=_operations,
        raise_protocol_error=_protocol_error,
    )
    await refresh_started.wait()

    await catalog.aclose()

    assert refresh_finished.is_set()
    assert catalog._refresh_task is None


@pytest.mark.asyncio
async def test_copilot_provider_closes_catalog_before_transport() -> None:
    provider = CopilotProvider()
    close_order: list[str] = []

    async def close_catalog() -> None:
        close_order.append("catalog")

    async def close_transport() -> None:
        close_order.append("transport")

    provider._catalog.aclose = close_catalog  # type: ignore[method-assign]
    provider._transport.close = close_transport  # type: ignore[method-assign]

    await provider.close()

    assert close_order == ["catalog", "transport"]


def test_ttl_cache_uses_monotonic_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from router_maestro.utils import cache as cache_module
    from router_maestro.utils.cache import TTLCache

    monotonic = 100.0
    wall_clock = 1_000.0
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: monotonic)
    monkeypatch.setattr(cache_module.time, "time", lambda: wall_clock)
    cache = TTLCache[str](10)
    cache.set("value")

    wall_clock = -50_000.0
    monotonic = 109.0
    assert cache.get() == "value"

    monotonic = 111.0
    assert cache.get() is None
    assert cache.peek() == "value"


def test_catalog_effort_lookup_returns_a_defensive_copy() -> None:
    catalog = CopilotCatalog()
    catalog.models_ttl_cache.set(
        [
            ModelInfo(
                id="model",
                name="Model",
                provider="github-copilot",
                reasoning_effort_values=["low", "high"],
            )
        ]
    )

    values = catalog.effort_values("model")
    assert values is not None
    values.append("mutated")

    assert catalog.effort_values("model") == ["low", "high"]


def test_router_custom_provider_construction_delegates_credential_policy(monkeypatch) -> None:
    provider_config = object()
    repository = object()
    expected_provider = object()
    observed: dict[str, object] = {}

    class FakeRepository:
        def __new__(cls):
            return repository

    def create(provider_name, config, *, credential_repository):
        observed.update(
            provider_name=provider_name,
            config=config,
            credential_repository=credential_repository,
        )
        return expected_provider

    monkeypatch.setattr("router_maestro.auth.repository.CredentialRepository", FakeRepository)
    monkeypatch.setattr("router_maestro.providers.custom_factory.create_custom_provider", create)
    router = Router.__new__(Router)

    provider = router._create_custom_provider("local-llm", provider_config)

    assert provider is expected_provider
    assert observed == {
        "provider_name": "local-llm",
        "config": provider_config,
        "credential_repository": repository,
    }
