"""Request-scoped audit coverage for provider model-catalog transports."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from router_maestro.providers.openai import OpenAIProvider
from router_maestro.providers.openai_compat import OpenAICompatibleProvider
from router_maestro.runtime import request_context as request_context_module


class _AuditSpy:
    def __init__(self) -> None:
        self.upstream: list[tuple] = []
        self.responses: list[tuple] = []

    def record_upstream(self, *args) -> None:
        self.upstream.append(args)

    def record_upstream_response(self, *args) -> None:
        self.responses.append(args)


def _provider(kind: str):
    if kind == "openai":
        provider = OpenAIProvider(base_url="https://catalog.invalid/v1")
        provider._get_headers = lambda: {  # type: ignore[method-assign]
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
        }
        return provider
    return OpenAICompatibleProvider(
        name="custom",
        base_url="https://catalog.invalid/v1",
        api_key="secret",
    )


def _client_patch(kind: str) -> str:
    module = "openai" if kind == "openai" else "openai_compat"
    return f"router_maestro.providers.{module}.httpx.AsyncClient"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["openai", "compatible"])
async def test_model_catalog_success_is_recorded_in_request_audit(kind: str) -> None:
    provider = _provider(kind)
    audit = _AuditSpy()
    response = httpx.Response(
        200,
        json={"data": [{"id": "gpt-audit"}]},
        headers={"x-request-id": "catalog-1"},
        request=httpx.Request("GET", "https://catalog.invalid/v1/models"),
    )

    with patch(_client_patch(kind)) as client_cls:
        client = AsyncMock()
        client.get.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client_cls.return_value = client
        token = request_context_module._current_request_context.set(  # type: ignore[attr-defined]
            type("Context", (), {"audit": audit})()
        )
        try:
            await provider.list_models()
        finally:
            request_context_module._current_request_context.reset(token)  # type: ignore[attr-defined]

    assert audit.upstream == [
        (
            "GET",
            "https://catalog.invalid/v1/models",
            provider._get_headers(),
            None,
        )
    ]
    assert audit.responses == [
        (
            200,
            {
                "x-request-id": "catalog-1",
                "content-length": "29",
                "content-type": "application/json",
            },
            b'{"data":[{"id":"gpt-audit"}]}',
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["openai", "compatible"])
async def test_model_catalog_http_error_response_is_recorded(kind: str) -> None:
    provider = _provider(kind)
    audit = _AuditSpy()
    response = httpx.Response(
        503,
        json={"error": "unavailable"},
        request=httpx.Request("GET", "https://catalog.invalid/v1/models"),
    )

    with patch(_client_patch(kind)) as client_cls:
        client = AsyncMock()
        client.get.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client_cls.return_value = client
        token = request_context_module._current_request_context.set(  # type: ignore[attr-defined]
            type("Context", (), {"audit": audit})()
        )
        try:
            models = await provider.list_models()
        finally:
            request_context_module._current_request_context.reset(token)  # type: ignore[attr-defined]

    if kind == "openai":
        assert models
    else:
        assert models == []
    assert len(audit.upstream) == 1
    assert audit.responses == [
        (
            503,
            {"content-length": "23", "content-type": "application/json"},
            b'{"error":"unavailable"}',
        )
    ]
