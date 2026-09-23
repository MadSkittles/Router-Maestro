"""HTTP transport policy for the GitHub Copilot provider."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any
from uuid import uuid4

import httpx

from router_maestro.auth.github_oauth import (
    COPILOT_API_VERSION,
    COPILOT_EDITOR_VERSION,
    COPILOT_PLUGIN_VERSION,
    COPILOT_USER_AGENT,
)
from router_maestro.pipeline.beta_strip import strip_beta_tokens
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    Message,
    ProviderError,
    ProviderFailureKind,
)
from router_maestro.providers.copilot_support.auth_session import (
    AUTH_RETRY_STATUSES,
    CopilotAuthSession,
)
from router_maestro.utils import get_logger

logger = get_logger("providers.copilot.transport")


def _request_audit():
    from router_maestro.runtime import get_current_request_context

    context = get_current_request_context()
    return context.audit if context is not None else None


class CopilotTransport:
    """Own pooled HTTP/2 clients, headers, retries, and stream lifetimes."""

    client_max_age = 300

    def __init__(self, auth: CopilotAuthSession) -> None:
        self.auth = auth
        self.client: httpx.AsyncClient | None = None
        self.client_created_at = 0.0
        self._client_leases: dict[httpx.AsyncClient, int] = {}
        self._retired_clients: set[httpx.AsyncClient] = set()
        self._client_close_tasks: dict[httpx.AsyncClient, asyncio.Task[None]] = {}

    def url(self, path: str) -> str:
        return f"{self.auth.api_base.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def chat_initiator(messages: list[Message] | None) -> str:
        if not messages:
            return "user"
        return (
            "agent"
            if any(message.role in ("assistant", "tool") for message in messages)
            else "user"
        )

    @staticmethod
    def responses_initiator(response_input: str | list[dict[str, Any]] | None) -> str:
        if isinstance(response_input, str) or not response_input:
            return "user"
        for item in response_input:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if not role or (isinstance(role, str) and role.lower() == "assistant"):
                return "agent"
        return "user"

    def headers(
        self,
        vision_request: bool = False,
        *,
        messages: list[Message] | None = None,
        response_input: str | list[dict[str, Any]] | None = None,
        intent: str = "conversation-panel",
    ) -> dict[str, str]:
        if not self.auth.cached_token:
            raise ProviderError(
                "No valid token available",
                status_code=401,
                kind=ProviderFailureKind.AUTHENTICATION,
                provider=self.auth.provider_name,
            )
        headers = {
            "Authorization": f"Bearer {self.auth.cached_token}",
            "Content-Type": "application/json",
            "Editor-Version": COPILOT_EDITOR_VERSION,
            "Editor-Plugin-Version": COPILOT_PLUGIN_VERSION,
            "Copilot-Integration-Id": "vscode-chat",
            "User-Agent": COPILOT_USER_AGENT,
            "OpenAI-Intent": intent,
            "X-GitHub-Api-Version": COPILOT_API_VERSION,
            "X-Request-Id": str(uuid4()),
            "X-Vscode-User-Agent-Library-Version": "electron-fetch",
        }
        if response_input is not None:
            headers["X-Initiator"] = self.responses_initiator(response_input)
        elif messages is not None:
            headers["X-Initiator"] = self.chat_initiator(messages)
        if vision_request:
            headers["Copilot-Vision-Request"] = "true"
        from router_maestro.runtime import get_current_request_context

        context = get_current_request_context()
        if context is not None:
            anthropic_beta = strip_beta_tokens(
                context.request_header("anthropic-beta"),
                context.config.beta_strip,
            )
            if anthropic_beta is not None:
                headers["anthropic-beta"] = anthropic_beta
        return headers

    def get_client(self) -> httpx.AsyncClient:
        now = time.monotonic()
        client = self.client
        if (
            client is not None
            and not client.is_closed
            and self.client_created_at > 0
            and now - self.client_created_at >= self.client_max_age
        ):
            self._retire_client(client)
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
                http2=True,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
            self.client_created_at = now
        return self.client

    @contextlib.asynccontextmanager
    async def lease_client(
        self,
        get_client: Callable[[], httpx.AsyncClient] | None = None,
    ) -> AsyncIterator[httpx.AsyncClient]:
        """Keep one client alive until the request or stream releases it."""
        client = (get_client or self.get_client)()
        self._client_leases[client] = self._client_leases.get(client, 0) + 1
        try:
            yield client
        finally:
            remaining = self._client_leases[client] - 1
            if remaining:
                self._client_leases[client] = remaining
            else:
                self._client_leases.pop(client, None)
                if client in self._retired_clients:
                    task = self._schedule_client_close(client)
                    if task is not None:
                        await asyncio.shield(task)

    def _retire_client(self, client: httpx.AsyncClient) -> asyncio.Task[None] | None:
        """Stop assigning new work to a client and close it once it is idle."""
        if self.client is client:
            self.client = None
            self.client_created_at = 0.0
        self._retired_clients.add(client)
        if self._client_leases.get(client, 0) == 0:
            return self._schedule_client_close(client)
        return None

    def _schedule_client_close(self, client: httpx.AsyncClient) -> asyncio.Task[None] | None:
        if client.is_closed:
            self._retired_clients.discard(client)
            return None
        task = self._client_close_tasks.get(client)
        if task is None:
            task = asyncio.create_task(self._close_retired_client(client))
            self._client_close_tasks[client] = task
        return task

    async def _close_retired_client(self, client: httpx.AsyncClient) -> None:
        try:
            with contextlib.suppress(Exception):
                await client.aclose()
        finally:
            self._retired_clients.discard(client)
            self._client_close_tasks.pop(client, None)

    async def recycle_client(self, client: httpx.AsyncClient | None = None) -> None:
        target = client or self.client
        if target is None:
            return
        task = self._retire_client(target)
        if task is not None:
            await asyncio.shield(task)

    async def close(self) -> None:
        clients = set(self._retired_clients)
        clients.update(self._client_close_tasks)
        if self.client is not None:
            clients.add(self.client)
        self.client = None
        self.client_created_at = 0.0
        for client in clients:
            task = self._client_close_tasks.get(client)
            if task is not None:
                await asyncio.shield(task)
            elif not client.is_closed:
                await client.aclose()
        self._retired_clients.clear()

    async def send_with_auth_retry(
        self,
        method: str,
        path: str,
        *,
        client: httpx.AsyncClient | None = None,
        json: dict | None = None,
        headers_kwargs: dict | None = None,
        timeout: Any = TIMEOUT_NON_STREAMING,
        model: str | None = None,
        get_client: Callable[[], httpx.AsyncClient] | None = None,
        get_headers: Callable[..., dict[str, str]] | None = None,
        recycle_client: Callable[[httpx.AsyncClient | None], Awaitable[None]] | None = None,
        refresh_for_auth_status: Callable[[str, int], Awaitable[bool]] | None = None,
        raise_auth_failure: Callable[..., None] | None = None,
    ) -> httpx.Response:
        get_client = get_client or self.get_client
        get_headers = get_headers or self.headers
        recycle_client = recycle_client or self.recycle_client
        refresh_for_auth_status = refresh_for_auth_status or self.auth.refresh_for_auth_status
        raise_auth_failure = raise_auth_failure or self.auth.raise_auth_failure
        use_managed_client = client is None
        headers_kwargs = headers_kwargs or {}
        for attempt in range(2):
            client_context = (
                self.lease_client(get_client)
                if use_managed_client
                else contextlib.nullcontext(client)
            )
            async with client_context as active_client:
                assert active_client is not None
                headers = get_headers(**headers_kwargs)
                audit = _request_audit()
                if audit is not None:
                    audit.record_upstream(method, self.url(path), headers, json)
                try:
                    if method == "GET":
                        response = await active_client.get(
                            self.url(path),
                            headers=headers,
                            timeout=timeout,
                        )
                    else:
                        response = await active_client.post(
                            self.url(path),
                            json=json,
                            headers=headers,
                            timeout=timeout,
                        )
                except (
                    httpx.RemoteProtocolError,
                    httpx.PoolTimeout,
                    httpx.ConnectError,
                ) as error:
                    if attempt == 0:
                        logger.warning(
                            "Connection error on %s, recycling client (%s)",
                            path,
                            type(error).__name__,
                        )
                        if use_managed_client:
                            await recycle_client(active_client)
                        use_managed_client = True
                        continue
                    raise ProviderError(
                        f"Connection failed after retry ({type(error).__name__})",
                        status_code=502,
                        retryable=True,
                        kind=ProviderFailureKind.TRANSPORT,
                        provider=self.auth.provider_name,
                        model=model,
                        cause=error,
                    ) from error
                if audit is not None:
                    audit.record_upstream_response(
                        response.status_code,
                        dict(response.headers),
                        response.content,
                    )
                if attempt == 0 and await refresh_for_auth_status(path, response.status_code):
                    continue
                if response.status_code in AUTH_RETRY_STATUSES:
                    raise_auth_failure(path, response.status_code, model=model)
                return response
        return response

    @contextlib.asynccontextmanager
    async def stream_with_auth_retry(
        self,
        path: str,
        *,
        json: dict,
        headers_kwargs: dict,
        model: str | None = None,
        get_client: Callable[[], httpx.AsyncClient] | None = None,
        get_headers: Callable[..., dict[str, str]] | None = None,
        recycle_client: Callable[[httpx.AsyncClient | None], Awaitable[None]] | None = None,
        refresh_for_auth_status: Callable[[str, int], Awaitable[bool]] | None = None,
        raise_auth_failure: Callable[..., None] | None = None,
    ) -> AsyncIterator[httpx.Response]:
        get_client = get_client or self.get_client
        get_headers = get_headers or self.headers
        recycle_client = recycle_client or self.recycle_client
        refresh_for_auth_status = refresh_for_auth_status or self.auth.refresh_for_auth_status
        raise_auth_failure = raise_auth_failure or self.auth.raise_auth_failure
        for attempt in range(2):
            async with self.lease_client(get_client) as client:
                headers = get_headers(**headers_kwargs)
                audit = _request_audit()
                if audit is not None:
                    audit.record_upstream("POST", self.url(path), headers, json)
                try:
                    cm: AbstractAsyncContextManager[httpx.Response] = client.stream(
                        "POST",
                        self.url(path),
                        json=json,
                        headers=headers,
                    )
                    response = await cm.__aenter__()
                except (
                    httpx.RemoteProtocolError,
                    httpx.PoolTimeout,
                    httpx.ConnectError,
                ) as error:
                    if attempt == 0:
                        logger.warning(
                            "Stream connection error on %s, recycling client (%s)",
                            path,
                            type(error).__name__,
                        )
                        await recycle_client(client)
                        continue
                    raise ProviderError(
                        f"Stream connection failed after retry ({type(error).__name__})",
                        status_code=502,
                        retryable=True,
                        kind=ProviderFailureKind.TRANSPORT,
                        provider=self.auth.provider_name,
                        model=model,
                        cause=error,
                    ) from error
                if audit is not None:
                    audit.record_upstream_response(
                        response.status_code,
                        dict(response.headers),
                        stream_summary="stream opened",
                    )
                if attempt == 0 and response.status_code in AUTH_RETRY_STATUSES:
                    with contextlib.suppress(Exception):
                        await response.aread()
                    await cm.__aexit__(None, None, None)
                    if await refresh_for_auth_status(path, response.status_code):
                        continue
                if response.status_code in AUTH_RETRY_STATUSES:
                    with contextlib.suppress(Exception):
                        await response.aread()
                    await cm.__aexit__(None, None, None)
                    raise_auth_failure(path, response.status_code, model=model)
                try:
                    yield response
                finally:
                    await cm.__aexit__(None, None, None)
                return
