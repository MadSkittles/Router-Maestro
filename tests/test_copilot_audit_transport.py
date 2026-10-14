"""Audit HTTP failures before Copilot streaming responses are committed."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest

from router_maestro.providers.base import ProviderError
from router_maestro.providers.copilot_support.auth_session import CopilotAuthSession
from router_maestro.providers.copilot_support.transport import CopilotTransport
from router_maestro.utils.audit import AuditTrace


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes, *, error: BaseException | None = None) -> None:
        self.body = body
        self.error = error
        self.reads = 0
        self.closes = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.reads += 1
        if self.error is not None:
            raise self.error
        yield self.body

    async def aclose(self) -> None:
        self.closes += 1


class _TrackedClient(httpx.AsyncClient):
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = iter(responses)
        self.stream_exits = 0
        super().__init__(transport=httpx.MockTransport(lambda _request: next(self.responses)))

    @asynccontextmanager
    async def stream(self, *args, **kwargs) -> AsyncIterator[httpx.Response]:
        try:
            async with super().stream(*args, **kwargs) as response:
                yield response
        finally:
            self.stream_exits += 1


@pytest.fixture
def audited_transport(monkeypatch, tmp_path):
    trace = AuditTrace("copilot-stream", tmp_path)
    monkeypatch.setattr(
        "router_maestro.providers.copilot_support.transport._request_audit", lambda: trace
    )
    return CopilotTransport(CopilotAuthSession()), trace


def _open_stream(transport: CopilotTransport, client: httpx.AsyncClient, **kwargs):
    return transport.stream_with_auth_retry(
        "/responses",
        json={"model": "test-model", "stream": True},
        headers_kwargs={},
        get_client=lambda: client,
        get_headers=lambda **_kwargs: {"Authorization": "Bearer request-secret"},
        **kwargs,
    )


def _response_records(trace: AuditTrace) -> list[dict]:
    return [record for name, record in trace._records.items() if name.startswith("upstream_resp")]


@pytest.mark.parametrize("status", [302, 400, 422, 429, 500])
async def test_http_failure_body_is_audited_once_and_remains_readable(
    audited_transport, tmp_path, caplog, status
):
    transport, trace = audited_transport
    payload = {
        "error": {
            "message": "Unsupported value: 'none'; supported values: 'low', 'medium'.",
            "type": "invalid_request_error",
            "access_token": "response-secret",
            "nested": [{"clientSecret": "nested-secret"}],
        }
    }
    raw_body = json.dumps(payload).encode()
    upstream = _TrackedStream(raw_body)
    async with _TrackedClient(
        [
            httpx.Response(
                status,
                stream=upstream,
                headers={"Content-Type": "application/json", "X-Api-Key": "header-secret"},
            )
        ]
    ) as client:
        async with _open_stream(transport, client) as response:
            assert response.content == raw_body
            assert await response.aread() == raw_body
            assert response.json() == payload
            assert transport._client_leases == {client: 1}
        assert client.stream_exits == 1
        assert transport._client_leases == {}
        assert client.is_closed is False

    assert upstream.reads == 1
    assert upstream.closes == 1
    trace.flush()
    record = json.loads((tmp_path / "copilot-stream" / "upstream_resp.json").read_text())
    assert record["status"] == status
    assert "stream_summary" not in record
    assert record["body"]["error"]["message"] == payload["error"]["message"]
    assert record["body"]["error"]["access_token"] == "***"
    assert record["body"]["error"]["nested"] == [{"clientSecret": "***"}]
    assert record["headers"]["x-api-key"] == "***"
    assert len(_response_records(trace)) == 1
    assert not (tmp_path / "copilot-stream" / "upstream_resp_2.json").exists()
    serialized = json.dumps(trace._records)
    for secret in ("request-secret", "response-secret", "nested-secret", "header-secret"):
        assert secret not in serialized
        assert secret not in caplog.text
    assert payload["error"]["message"] not in caplog.text


@pytest.mark.parametrize(
    "raw_body",
    [b'{"error":{"token":"malformed-secret"', b"\xff\x00binary-secret", b"", b"text-secret"],
)
async def test_non_json_failure_keeps_only_byte_count_in_audit(audited_transport, raw_body):
    transport, trace = audited_transport
    upstream = _TrackedStream(raw_body)
    async with _TrackedClient([httpx.Response(400, stream=upstream)]) as client:
        async with _open_stream(transport, client) as response:
            assert await response.aread() == raw_body
    (record,) = _response_records(trace)
    assert record["body"] == f"<{len(raw_body)} bytes>"
    assert "stream_summary" not in record
    assert upstream.reads == 1
    assert upstream.closes == 1


@pytest.mark.parametrize("status", [200, 201, 204])
async def test_success_is_not_pre_read_by_audit(audited_transport, status):
    transport, trace = audited_transport
    sse = b'data: {"type":"response.completed"}\n\n'
    upstream = _TrackedStream(sse)
    async with _TrackedClient([httpx.Response(status, stream=upstream)]) as client:
        async with _open_stream(transport, client) as response:
            assert upstream.reads == 0
            assert response.is_stream_consumed is False
            (record,) = _response_records(trace)
            assert record["stream_summary"] == "stream opened"
            assert "body" not in record
            assert [chunk async for chunk in response.aiter_bytes()] == [sse]
        assert client.stream_exits == 1
        assert transport._client_leases == {}
    assert upstream.reads == 1
    assert upstream.closes == 1


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_retry_audits_error_before_refresh_and_leaves_success_unread(
    audited_transport, status
):
    transport, trace = audited_transport
    failure = _TrackedStream(b'{"error":{"message":"expired","token":"auth-secret"}}')
    success = _TrackedStream(b"data: [DONE]\n\n")
    async with _TrackedClient(
        [httpx.Response(status, stream=failure), httpx.Response(200, stream=success)]
    ) as client:

        async def refresh(path, upstream_status):
            assert path == "/responses"
            assert upstream_status == status
            (record,) = _response_records(trace)
            assert record["body"] == {"error": {"message": "expired", "token": "***"}}
            assert failure.closes == 1
            assert client.stream_exits == 1
            return True

        refresh_mock = AsyncMock(side_effect=refresh)
        async with _open_stream(
            transport, client, refresh_for_auth_status=refresh_mock
        ) as response:
            assert response.status_code == 200
            assert success.reads == 0
            assert await response.aread() == success.body
        refresh_mock.assert_awaited_once_with("/responses", status)
        assert client.stream_exits == 2
        assert transport._client_leases == {}
    assert failure.reads == failure.closes == 1
    assert success.reads == success.closes == 1
    assert [record["status"] for record in _response_records(trace)] == [status, 200]
    assert len([name for name in trace._records if name.startswith("upstream")]) == 4


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("refresh_succeeds", [False, True])
async def test_auth_rejection_records_each_attempt_and_closes_exactly_once(
    audited_transport, status, refresh_succeeds
):
    transport, trace = audited_transport
    attempts = 2 if refresh_succeeds else 1
    streams = [
        _TrackedStream(json.dumps({"error": {"message": f"rejected-{index}"}}).encode())
        for index in range(attempts)
    ]
    refresh = AsyncMock(return_value=refresh_succeeds)
    async with _TrackedClient(
        [httpx.Response(status, stream=stream) for stream in streams]
    ) as client:
        with pytest.raises(ProviderError, match="authentication rejected") as caught:
            async with _open_stream(transport, client, refresh_for_auth_status=refresh):
                pytest.fail("Authentication failures must not be yielded to caller")
        assert caught.value.status_code == status
        assert client.stream_exits == attempts
        assert transport._client_leases == {}
        assert client.is_closed is False
    refresh.assert_awaited_once_with("/responses", status)
    assert len(_response_records(trace)) == attempts
    assert [record["body"]["error"]["message"] for record in _response_records(trace)] == [
        f"rejected-{index}" for index in range(attempts)
    ]
    assert all(stream.reads == stream.closes == 1 for stream in streams)


async def test_refresh_failure_preserves_audit_and_releases_lease(audited_transport):
    transport, trace = audited_transport
    upstream = _TrackedStream(b'{"error":{"message":"expired"}}')
    async with _TrackedClient([httpx.Response(401, stream=upstream)]) as client:
        with pytest.raises(RuntimeError, match="refresh failed"):
            async with _open_stream(
                transport,
                client,
                refresh_for_auth_status=AsyncMock(side_effect=RuntimeError("refresh failed")),
            ):
                pytest.fail("Refresh failure must propagate")
        assert client.stream_exits == 1
        assert transport._client_leases == {}
    assert upstream.closes == 1
    assert len(_response_records(trace)) == 1


async def test_caller_failure_closes_audited_error_response(audited_transport):
    transport, trace = audited_transport
    upstream = _TrackedStream(b'{"error":{"message":"invalid request"}}')
    async with _TrackedClient([httpx.Response(400, stream=upstream)]) as client:
        with pytest.raises(ProviderError, match="Copilot API error: 400"):
            async with _open_stream(transport, client) as response:
                assert await response.aread() == upstream.body
                raise ProviderError("Copilot API error: 400", status_code=400)
        assert client.stream_exits == 1
        assert transport._client_leases == {}
    assert upstream.closes == 1
    assert len(_response_records(trace)) == 1


async def test_error_body_capture_keeps_retired_client_leased(audited_transport):
    transport, trace = audited_transport
    read_started = asyncio.Event()
    release_read = asyncio.Event()

    class BlockingErrorStream(_TrackedStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            read_started.set()
            await release_read.wait()
            async for chunk in super().__aiter__():
                yield chunk

    upstream = BlockingErrorStream(b'{"error":{"message":"invalid request"}}')
    async with _TrackedClient([httpx.Response(400, stream=upstream)]) as client:

        async def consume():
            async with _open_stream(transport, client) as response:
                assert await response.aread() == upstream.body
                assert client.is_closed is False

        request = asyncio.create_task(consume())
        await read_started.wait()
        try:
            await transport.recycle_client(client)
            assert client.is_closed is False
            assert transport._client_leases == {client: 1}
        finally:
            release_read.set()
            await request
        assert client.is_closed is True
        assert transport._client_leases == {}
        assert client.stream_exits == 1
    assert upstream.closes == 1
    assert len(_response_records(trace)) == 1


@pytest.mark.parametrize("status", [400, 401])
@pytest.mark.parametrize(
    "error_class",
    [httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.DecodingError],
)
async def test_body_read_failure_records_safe_metadata_and_does_not_block_auth(
    audited_transport, caplog, status, error_class
):
    transport, trace = audited_transport
    upstream = _TrackedStream(b"", error=error_class("sensitive-error-text"))
    refresh = AsyncMock(return_value=False)
    async with _TrackedClient([httpx.Response(status, stream=upstream)]) as client:
        if status == 401:
            with pytest.raises(ProviderError, match="authentication rejected"):
                async with _open_stream(transport, client, refresh_for_auth_status=refresh):
                    pytest.fail("Authentication failure must propagate")
            refresh.assert_awaited_once_with("/responses", 401)
        else:
            async with _open_stream(transport, client) as response:
                assert response.status_code == 400
                assert response.content == b""
                assert await response.aread() == b""
                assert response.request.url.path == "/responses"
        assert client.stream_exits == 1
        assert transport._client_leases == {}
    (record,) = _response_records(trace)
    assert record["stream_summary"] == f"error body unavailable ({error_class.__name__})"
    assert "body" not in record
    assert "sensitive-error-text" not in json.dumps(trace._records)
    assert "sensitive-error-text" not in caplog.text
    assert upstream.closes == 1


@pytest.mark.parametrize("status", [401, 403])
async def test_unreadable_auth_body_still_retries_with_one_record_per_response(
    audited_transport, status
):
    transport, trace = audited_transport
    failure = _TrackedStream(b"", error=httpx.ReadError("private-error-message"))
    success = _TrackedStream(b"data: [DONE]\n\n")
    refresh = AsyncMock(return_value=True)
    async with _TrackedClient(
        [httpx.Response(status, stream=failure), httpx.Response(200, stream=success)]
    ) as client:
        async with _open_stream(transport, client, refresh_for_auth_status=refresh) as response:
            assert success.reads == 0
            assert await response.aread() == success.body
        assert client.stream_exits == 2
        assert transport._client_leases == {}
    refresh.assert_awaited_once_with("/responses", status)
    first, second = _response_records(trace)
    assert first["stream_summary"] == "error body unavailable (ReadError)"
    assert second["stream_summary"] == "stream opened"
    assert failure.closes == success.closes == 1


@pytest.mark.parametrize("status", [400, 422, 429, 500])
async def test_audit_disabled_does_not_pre_read_error(monkeypatch, status):
    monkeypatch.setattr(
        "router_maestro.providers.copilot_support.transport._request_audit", lambda: None
    )
    transport = CopilotTransport(CopilotAuthSession())
    upstream = _TrackedStream(b'{"error":{"message":"caller only"}}')
    async with _TrackedClient([httpx.Response(status, stream=upstream)]) as client:
        async with _open_stream(transport, client) as response:
            assert upstream.reads == 0
            assert await response.aread() == upstream.body
            assert await response.aread() == upstream.body
            assert response.content == upstream.body
            assert response.status_code == status
    assert upstream.closes == 1


async def test_cancelled_error_read_closes_response_and_releases_lease(audited_transport):
    transport, trace = audited_transport
    upstream = _TrackedStream(b"", error=asyncio.CancelledError())
    async with _TrackedClient([httpx.Response(400, stream=upstream)]) as client:
        with pytest.raises(asyncio.CancelledError):
            async with _open_stream(transport, client):
                pytest.fail("Cancellation must propagate")
        assert client.stream_exits == 1
        assert transport._client_leases == {}
    assert upstream.closes == 1
