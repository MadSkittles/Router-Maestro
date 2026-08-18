"""Focused lifecycle contracts for the shared provider HTTP executor."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import NoReturn
from unittest.mock import Mock, patch

import httpx
import pytest

from router_maestro.protocols import WireProtocol
from router_maestro.providers.bindings import PreparedAttempt
from router_maestro.providers.http_executor import ProviderHttpClientPool, SharedHttpExecutor
from router_maestro.providers.openai_compat import OpenAICompatibleProvider
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.router import Router


def _attempt(*, stream: bool) -> PreparedAttempt:
    return PreparedAttempt(
        binding_id="test-binding",
        protocol=WireProtocol.OPENAI_CHAT,
        model=ModelRef(provider="test-provider", upstream_id="test-model"),
        url="https://provider.example/generate",
        payload={"model": "test-model", "stream": stream},
        stream=stream,
    )


@pytest.mark.asyncio
async def test_openai_binding_reuses_provider_client_until_close() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": len(requests)})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        name="custom-test",
        base_url="https://provider.example",
        api_key="secret",
    )
    binding = provider.bindings()[0]
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="test-model"),
        payload={"messages": [{"role": "user", "content": "hello"}]},
        stream=False,
    )

    with patch(
        "router_maestro.providers.openai_base.httpx.AsyncClient",
        return_value=client,
    ) as client_factory:
        assert binding.executor is not None
        assert await binding.executor.execute(attempt) == {"ok": 1}
        assert await binding.executor.execute(attempt) == {"ok": 2}

    assert client_factory.call_count == 1
    assert len(requests) == 2
    assert client.is_closed is False

    await provider.close()
    await provider.close()
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_router_close_closes_provider_owned_pool() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200)))
    provider = OpenAICompatibleProvider(
        name="custom-test",
        base_url="https://provider.example",
        api_key="secret",
    )
    with patch(
        "router_maestro.providers.openai_base.httpx.AsyncClient",
        return_value=client,
    ):
        assert provider._http_client_pool.get_client() is client
    router = Router.__new__(Router)
    router.providers = {provider.name: provider}
    router._close_lock = asyncio.Lock()
    router._closed_provider_ids = set()
    router._closed = False

    await router.close()
    await router.close()

    assert client.is_closed is True


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self, frame: Mapping[str, object]) -> None:
        self._first = f"data: {json.dumps(frame)}\n\n".encode()
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield self._first
        self.waiting.set()
        await self.release.wait()

    async def aclose(self) -> None:
        self.closed = True
        self.release.set()


@pytest.mark.asyncio
async def test_stream_cancellation_closes_response_but_keeps_pool_reusable() -> None:
    upstream = _BlockingStream({"delta": "first"})
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=upstream))
    )
    pool = ProviderHttpClientPool(lambda: client)
    executor = SharedHttpExecutor(client_pool=pool)
    iterator = executor.execute_stream(_attempt(stream=True))

    assert await anext(iterator) == {"delta": "first"}

    async def read_next_frame() -> Mapping[str, object]:
        return await anext(iterator)

    next_frame = asyncio.create_task(read_next_frame())
    await upstream.waiting.wait()
    next_frame.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_frame

    assert upstream.closed is True
    assert client.is_closed is False

    await pool.close()
    assert client.is_closed is True


class _ProjectedStatusError(RuntimeError):
    pass


class _HookExecutor(SharedHttpExecutor):
    def _project_payload(
        self,
        payload: dict[str, object],
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> Mapping[str, object]:
        return {
            **payload,
            "projected_model": attempt.model.upstream_id,
            "projected_stream": stream,
        }

    def _raise_status(
        self,
        error: httpx.HTTPStatusError,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        raise _ProjectedStatusError(
            f"{attempt.model.upstream_id}:{error.response.status_code}:{stream}"
        ) from error


@pytest.mark.asyncio
async def test_shared_executor_delegates_response_projection_and_status_policy() -> None:
    responses = iter(
        (
            httpx.Response(200, json={"value": "kept"}),
            httpx.Response(429, json={"error": "slow down"}),
        )
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: next(responses)))
    pool = ProviderHttpClientPool(lambda: client)
    executor = _HookExecutor(client_pool=pool)
    attempt = _attempt(stream=False)

    assert await executor.execute(attempt) == {
        "value": "kept",
        "projected_model": "test-model",
        "projected_stream": False,
    }
    with pytest.raises(_ProjectedStatusError, match="test-model:429:False"):
        await executor.execute(attempt)

    await pool.close()


@pytest.mark.asyncio
async def test_shared_executor_records_one_audit_pair_per_attempt() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"value": "ok"}))
    )
    pool = ProviderHttpClientPool(lambda: client)
    executor = SharedHttpExecutor(client_pool=pool)
    attempt = _attempt(stream=False)
    audit = Mock()

    with patch(
        "router_maestro.providers.http_executor._request_audit",
        return_value=audit,
    ):
        assert await executor.execute(attempt) == {"value": "ok"}

    audit.record_upstream.assert_called_once_with(
        "POST",
        "https://provider.example/generate",
        {},
        {"model": "test-model", "stream": False},
    )
    audit.record_upstream_response.assert_called_once()
    assert audit.record_upstream_response.call_args.args[0] == 200

    await pool.close()
