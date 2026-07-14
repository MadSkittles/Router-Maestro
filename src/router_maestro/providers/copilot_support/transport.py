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
            "Editor-Version": "vscode/1.95.0",
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "Copilot-Integration-Id": "vscode-chat",
            "User-Agent": "GitHubCopilotChat/0.26.7",
            "OpenAI-Intent": "conversation-panel",
            "X-GitHub-Api-Version": "2025-04-01",
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
        now = time.time()
        needs_recycle = (
            self.client is not None
            and not self.client.is_closed
            and self.client_created_at > 0
            and now - self.client_created_at >= self.client_max_age
        )
        if needs_recycle:
            asyncio.ensure_future(self.client.aclose())
            self.client = None
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

    async def recycle_client(self) -> None:
        if self.client and not self.client.is_closed:
            with contextlib.suppress(Exception):
                await self.client.aclose()
        self.client = None

    async def close(self) -> None:
        if self.client and not self.client.is_closed:
            await self.client.aclose()
        self.client = None

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
        recycle_client: Callable[[], Awaitable[None]] | None = None,
        refresh_for_auth_status: Callable[[str, int], Awaitable[bool]] | None = None,
        raise_auth_failure: Callable[..., None] | None = None,
    ) -> httpx.Response:
        get_client = get_client or self.get_client
        get_headers = get_headers or self.headers
        recycle_client = recycle_client or self.recycle_client
        refresh_for_auth_status = refresh_for_auth_status or self.auth.refresh_for_auth_status
        raise_auth_failure = raise_auth_failure or self.auth.raise_auth_failure
        active_client = client or get_client()
        headers_kwargs = headers_kwargs or {}
        for attempt in range(2):
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
            except (httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.ConnectError) as error:
                if attempt == 0:
                    logger.warning(
                        "Connection error on %s, recycling client (%s)",
                        path,
                        type(error).__name__,
                    )
                    await recycle_client()
                    active_client = get_client()
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
        recycle_client: Callable[[], Awaitable[None]] | None = None,
        refresh_for_auth_status: Callable[[str, int], Awaitable[bool]] | None = None,
        raise_auth_failure: Callable[..., None] | None = None,
    ) -> AsyncIterator[httpx.Response]:
        get_client = get_client or self.get_client
        get_headers = get_headers or self.headers
        recycle_client = recycle_client or self.recycle_client
        refresh_for_auth_status = refresh_for_auth_status or self.auth.refresh_for_auth_status
        raise_auth_failure = raise_auth_failure or self.auth.raise_auth_failure
        client = get_client()
        for attempt in range(2):
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
            except (httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.ConnectError) as error:
                if attempt == 0:
                    logger.warning(
                        "Stream connection error on %s, recycling client (%s)",
                        path,
                        type(error).__name__,
                    )
                    await recycle_client()
                    client = get_client()
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
