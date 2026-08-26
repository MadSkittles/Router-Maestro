"""Shared pooled HTTP execution for protocol-native provider bindings."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, NoReturn

import httpx

from router_maestro.providers.base import TIMEOUT_NON_STREAMING, TIMEOUT_STREAMING
from router_maestro.providers.bindings import PreparedAttempt


def _request_audit():
    from router_maestro.runtime import get_current_request_context

    context = get_current_request_context()
    return context.audit if context is not None else None


class ProviderHttpClientPool:
    """Lazily own one reusable ``AsyncClient`` for a provider instance."""

    def __init__(self, client_factory: Callable[[], httpx.AsyncClient]) -> None:
        self._client_factory = client_factory
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    @property
    def client(self) -> httpx.AsyncClient | None:
        """Return the current client without creating one."""
        return self._client

    def get_client(self) -> httpx.AsyncClient:
        """Return the provider-owned client, creating it on first use."""
        if self._closed:
            raise RuntimeError("provider HTTP client pool is closed")
        if self._client is None or self._client.is_closed:
            self._client = self._client_factory()
        return self._client

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[httpx.AsyncClient]:
        """Yield the reusable client without transferring close ownership."""
        yield self.get_client()

    async def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout: Any,
    ) -> httpx.Response:
        """Send one request through the shared provider client."""
        return await self.get_client().request(
            method,
            url,
            json=dict(payload) if payload is not None else None,
            headers=dict(headers),
            timeout=timeout,
        )

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: Any,
    ) -> AsyncIterator[httpx.Response]:
        """Open one stream while keeping the provider client reusable."""
        async with self.get_client().stream(
            method,
            url,
            json=dict(payload),
            headers=dict(headers),
            timeout=timeout,
        ) as response:
            yield response

    async def close(self) -> None:
        """Close the owned client exactly once."""
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()


class SharedHttpExecutor:
    """Provider-neutral JSON/SSE response lifecycle for endpoint bindings.

    Subclasses retain provider policy through validation, status/error, frame
    filtering, and response projection hooks. Providers with an ordinary HTTP
    endpoint share ``ProviderHttpClientPool``; Copilot overrides the two send
    hooks so its existing auth-retry transport and HTTP/2 pool remain intact.
    """

    def __init__(
        self,
        *,
        client_pool: ProviderHttpClientPool | None = None,
        transport_records_audit: bool = False,
    ) -> None:
        self._client_pool = client_pool
        self._transport_records_audit = transport_records_audit

    async def execute(self, attempt: PreparedAttempt) -> Mapping[str, Any]:
        """Execute one non-stream request and return projected JSON."""
        self._validate_attempt(attempt, stream=False)
        audit = _request_audit()
        if audit is not None and not self._transport_records_audit:
            audit.record_upstream(
                attempt.method,
                attempt.url,
                dict(attempt.headers),
                dict(attempt.payload),
            )

        try:
            response = await self._send(attempt, timeout=TIMEOUT_NON_STREAMING)
            if audit is not None and not self._transport_records_audit:
                audit.record_upstream_response(
                    response.status_code,
                    dict(response.headers),
                    response.content,
                )
            response.raise_for_status()
            data = self._decode_json_object(response, attempt)
            return self._project_payload(data, attempt, stream=False)
        except httpx.HTTPStatusError as error:
            self._raise_status(error, attempt, stream=False)
        except httpx.TimeoutException as error:
            self._raise_timeout(error, attempt, stream=False)
        except httpx.HTTPError as error:
            self._raise_http_error(error, attempt, stream=False)
        raise AssertionError("provider HTTP error hook returned unexpectedly")

    async def execute_stream(
        self,
        attempt: PreparedAttempt,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Execute one SSE request and close its response on EOF or cancellation."""
        self._validate_attempt(attempt, stream=True)
        audit = _request_audit()
        if audit is not None and not self._transport_records_audit:
            audit.record_upstream(
                attempt.method,
                attempt.url,
                dict(attempt.headers),
                dict(attempt.payload),
            )

        try:
            async with self._open_stream(attempt, timeout=TIMEOUT_STREAMING) as response:
                if response.status_code >= 400:
                    with contextlib.suppress(Exception):
                        await response.aread()
                if audit is not None and not self._transport_records_audit:
                    if response.status_code >= 400:
                        audit.record_upstream_response(
                            response.status_code,
                            dict(response.headers),
                            response.content,
                        )
                    else:
                        audit.record_upstream_response(
                            response.status_code,
                            dict(response.headers),
                            stream_summary="stream opened",
                        )
                response.raise_for_status()
                async for frame in self._iter_sse_data(response.aiter_lines(), attempt):
                    yield frame
        except httpx.HTTPStatusError as error:
            self._raise_status(error, attempt, stream=True)
        except httpx.TimeoutException as error:
            self._raise_timeout(error, attempt, stream=True)
        except httpx.HTTPError as error:
            self._raise_http_error(error, attempt, stream=True)

    async def _send(self, attempt: PreparedAttempt, *, timeout: Any) -> httpx.Response:
        pool = self._required_client_pool()
        return await pool.request(
            attempt.method,
            attempt.url,
            payload=attempt.payload,
            headers=attempt.headers,
            timeout=timeout,
        )

    def _open_stream(
        self,
        attempt: PreparedAttempt,
        *,
        timeout: Any,
    ) -> AbstractAsyncContextManager[httpx.Response]:
        pool = self._required_client_pool()
        return pool.stream(
            attempt.method,
            attempt.url,
            payload=attempt.payload,
            headers=attempt.headers,
            timeout=timeout,
        )

    async def _iter_sse_data(
        self,
        lines: AsyncIterator[str],
        attempt: PreparedAttempt,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async for line in lines:
            if not line.startswith("data:"):
                continue
            raw_data = line[5:].strip()
            if not raw_data or self._skip_raw_sse_data(raw_data, attempt):
                continue
            try:
                frame = json.loads(raw_data)
                if not isinstance(frame, dict):
                    raise TypeError("SSE data must be a JSON object")
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self._raise_protocol_error(error, attempt)
            if self._skip_sse_frame(frame, attempt):
                continue
            yield self._project_payload(frame, attempt, stream=True)

    def _decode_json_object(
        self,
        response: httpx.Response,
        attempt: PreparedAttempt,
    ) -> dict[str, Any]:
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("response must be a JSON object")
            return data
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._raise_protocol_error(error, attempt)

    def _required_client_pool(self) -> ProviderHttpClientPool:
        if self._client_pool is None:
            raise RuntimeError("HTTP executor has no client pool or custom transport hook")
        return self._client_pool

    def _validate_attempt(self, attempt: PreparedAttempt, *, stream: bool) -> None:
        if attempt.stream is not stream:
            raise ValueError("attempt stream mode does not match execution")

    def _skip_raw_sse_data(self, data: str, attempt: PreparedAttempt) -> bool:
        del data, attempt
        return False

    def _skip_sse_frame(
        self,
        frame: Mapping[str, Any],
        attempt: PreparedAttempt,
    ) -> bool:
        del frame, attempt
        return False

    def _project_payload(
        self,
        payload: dict[str, Any],
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> Mapping[str, Any]:
        del attempt, stream
        return payload

    def _raise_status(
        self,
        error: httpx.HTTPStatusError,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        del attempt, stream
        raise error

    def _raise_timeout(
        self,
        error: httpx.TimeoutException,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        del attempt, stream
        raise error

    def _raise_http_error(
        self,
        error: httpx.HTTPError,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        del attempt, stream
        raise error

    def _raise_protocol_error(self, error: Exception, attempt: PreparedAttempt) -> NoReturn:
        del attempt
        raise error


__all__ = ["ProviderHttpClientPool", "SharedHttpExecutor"]
