"""Regression tests for Copilot HTTP client rotation and stream lifetimes."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from router_maestro.providers.copilot_support.auth_session import CopilotAuthSession
from router_maestro.providers.copilot_support.transport import CopilotTransport


class _BlockingLineStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield b"first\n"
        await self.release.wait()
        yield b"second\n"

    async def aclose(self) -> None:
        self.closed = True
        self.release.set()


@pytest.mark.asyncio
async def test_aged_client_rotation_waits_for_active_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _BlockingLineStream()
    old_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, request=request, stream=upstream)
        )
    )
    new_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    )
    auth = CopilotAuthSession()
    transport = CopilotTransport(auth)
    transport.client = old_client
    transport.client_created_at = time.monotonic()
    monkeypatch.setattr(
        "router_maestro.providers.copilot_support.transport.httpx.AsyncClient",
        lambda **_kwargs: new_client,
    )

    async def keep_token(_path: str, _status: int) -> bool:
        return False

    async with transport.stream_with_auth_retry(
        "/chat/completions",
        json={},
        headers_kwargs={},
        get_headers=lambda **_kwargs: {},
        refresh_for_auth_status=keep_token,
    ) as response:
        lines = response.aiter_lines()
        assert await anext(lines) == "first"

        transport.client_created_at = time.monotonic() - transport.client_max_age - 1
        assert transport.get_client() is new_client
        await asyncio.sleep(0)
        assert old_client.is_closed is False

        upstream.release.set()
        assert await anext(lines) == "second"

    assert old_client.is_closed is True
    assert upstream.closed is True
    assert new_client.is_closed is False

    await transport.close()
    assert new_client.is_closed is True


@pytest.mark.asyncio
async def test_aged_idle_client_closes_after_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request)

    old_client = httpx.AsyncClient(transport=httpx.MockTransport(ok))
    new_client = httpx.AsyncClient(transport=httpx.MockTransport(ok))
    transport = CopilotTransport(CopilotAuthSession())
    transport.client = old_client
    transport.client_created_at = time.monotonic() - transport.client_max_age - 1
    monkeypatch.setattr(
        "router_maestro.providers.copilot_support.transport.httpx.AsyncClient",
        lambda **_kwargs: new_client,
    )

    assert transport.get_client() is new_client
    await asyncio.sleep(0)

    assert old_client.is_closed is True
    assert new_client.is_closed is False

    await transport.close()
    assert new_client.is_closed is True


@pytest.mark.asyncio
async def test_aged_client_rotation_waits_for_active_non_stream_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def blocking_response(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await release_request.wait()
        return httpx.Response(200, json={"ok": True}, request=request)

    old_client = httpx.AsyncClient(transport=httpx.MockTransport(blocking_response))
    new_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    )
    transport = CopilotTransport(CopilotAuthSession())
    transport.client = old_client
    transport.client_created_at = time.monotonic()
    monkeypatch.setattr(
        "router_maestro.providers.copilot_support.transport.httpx.AsyncClient",
        lambda **_kwargs: new_client,
    )

    async def keep_token(_path: str, _status: int) -> bool:
        return False

    request = asyncio.create_task(
        transport.send_with_auth_retry(
            "POST",
            "/chat/completions",
            json={},
            headers_kwargs={},
            get_headers=lambda **_kwargs: {},
            refresh_for_auth_status=keep_token,
        )
    )
    await request_started.wait()

    transport.client_created_at = time.monotonic() - transport.client_max_age - 1
    assert transport.get_client() is new_client
    await asyncio.sleep(0)
    assert old_client.is_closed is False

    release_request.set()
    response = await request

    assert response.json() == {"ok": True}
    assert old_client.is_closed is True
    assert new_client.is_closed is False

    await transport.close()
    assert new_client.is_closed is True


@pytest.mark.asyncio
async def test_retired_client_closes_after_last_concurrent_lease() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    )
    transport = CopilotTransport(CopilotAuthSession())
    transport.client = client
    transport.client_created_at = time.monotonic()

    first_lease = transport.lease_client()
    second_lease = transport.lease_client()
    assert await first_lease.__aenter__() is client
    assert await second_lease.__aenter__() is client

    await transport.recycle_client(client)
    assert client.is_closed is False

    await first_lease.__aexit__(None, None, None)
    assert client.is_closed is False

    await second_lease.__aexit__(None, None, None)
    assert client.is_closed is True
