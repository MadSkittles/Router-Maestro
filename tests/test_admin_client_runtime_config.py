"""CLI admin client contract for runtime-config compare-and-swap."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from router_maestro.cli.client import AdminClient, AdminConfigConflictError


@pytest.mark.asyncio
async def test_admin_client_patch_sends_complete_config_and_revision():
    captured = {}

    async def patch_request(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(
            200,
            json={**kwargs["json"], "revision": "b" * 64},
            request=httpx.Request("PATCH", url),
        )

    client = AdminClient("http://router.test", "secret")
    config = {
        "priorities": ["provider/model"],
        "fallback": {"strategy": "priority", "maxRetries": 2},
        "model_overrides": {},
        "thinking": {"default_budget": 16000, "auto_enable": False, "model_budgets": {}},
        "guards": {
            "leak_guard": {"enabled": True},
            "runaway_guard": {"enabled": True, "max_bytes": 10000000, "max_deltas": 50000},
        },
        "beta_strip": ["beta-*"],
        "audit": {"enabled": False, "trace_dir": None},
    }

    with patch("router_maestro.cli.client.httpx.AsyncClient") as async_client:
        async_client.return_value.__aenter__.return_value.patch = AsyncMock(
            side_effect=patch_request
        )
        result = await client.patch_runtime_config(config=config, revision="a" * 64)

    assert captured["url"] == "http://router.test/api/admin/priorities"
    assert captured["json"] == {**config, "revision": "a" * 64}
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert result["revision"] == "b" * 64


@pytest.mark.asyncio
async def test_admin_client_maps_revision_conflict():
    request = httpx.Request("PATCH", "http://router.test/api/admin/priorities")
    response = httpx.Response(
        409,
        json={
            "detail": {
                "code": "config_revision_conflict",
                "current_revision": "c" * 64,
            }
        },
        headers={"ETag": f'"{"c" * 64}"'},
        request=request,
    )
    client = AdminClient("http://router.test", "secret")

    with patch("router_maestro.cli.client.httpx.AsyncClient") as async_client:
        async_client.return_value.__aenter__.return_value.patch = AsyncMock(return_value=response)
        with pytest.raises(AdminConfigConflictError) as exc_info:
            await client.patch_runtime_config(config={}, revision="a" * 64)

    assert exc_info.value.current_revision == "c" * 64


@pytest.mark.asyncio
async def test_admin_client_lists_typed_auth_provider_definitions():
    captured = {}

    async def get_request(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(
            200,
            json={
                "providers": [
                    {
                        "provider": "remote-custom",
                        "display_name": "Remote Custom",
                        "auth_type": "api",
                        "credential_required": True,
                        "source": "custom",
                        "api_key_env": "REMOTE_CUSTOM_API_KEY",
                    }
                ]
            },
            request=httpx.Request("GET", url),
        )

    client = AdminClient("http://router.test", "secret")
    with patch("router_maestro.cli.client.httpx.AsyncClient") as async_client:
        async_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=get_request)
        definitions = await client.list_auth_providers()

    assert captured["url"] == "http://router.test/api/admin/auth/providers"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert len(definitions) == 1
    assert definitions[0].provider == "remote-custom"
    assert definitions[0].credential_required is True
