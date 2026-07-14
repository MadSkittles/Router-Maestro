"""Tests for admin route side effects."""

from dataclasses import dataclass

import pytest
from starlette.responses import Response

from router_maestro.auth.storage import OAuthCredential
from router_maestro.config.repository import RuntimeConfigRepository
from router_maestro.providers import ModelInfo as ProviderModelInfo
from router_maestro.server.routes import admin
from router_maestro.server.schemas.admin import PrioritiesUpdateRequest


@dataclass
class _AccessToken:
    access_token: str


@dataclass
class _CopilotToken:
    token: str
    expires_at: int
    api_endpoint: str | None


class _StorageSpy:
    def __init__(self):
        self.saved: dict[str, OAuthCredential] = {}

    def set(self, provider: str, credential: OAuthCredential) -> None:
        self.saved[provider] = credential


class _AuthManagerSpy:
    instances: list["_AuthManagerSpy"] = []

    def __init__(self):
        self.storage = _StorageSpy()
        self.save_called = False
        _AuthManagerSpy.instances.append(self)

    def save(self) -> None:
        self.save_called = True


class _OAuthSessionsSpy:
    def __init__(self):
        self.updates: list[dict] = []

    async def update_session_status(self, session_id: str, **kwargs) -> None:
        self.updates.append({"session_id": session_id, **kwargs})


async def _fake_poll_access_token(client, device_code, interval, timeout):
    return _AccessToken("gh-access")


async def _fake_get_copilot_token(client, access_token):
    return _CopilotToken(
        token="copilot-token",
        expires_at=12345,
        api_endpoint="https://api.enterprise.githubcopilot.com",
    )


@pytest.mark.asyncio
async def test_oauth_completion_preserves_copilot_api_endpoint(monkeypatch):
    sessions = _OAuthSessionsSpy()
    reset_calls: list[None] = []
    _AuthManagerSpy.instances.clear()

    monkeypatch.setattr(admin, "AuthManager", _AuthManagerSpy)
    monkeypatch.setattr(admin, "oauth_sessions", sessions)
    monkeypatch.setattr(admin, "reset_router", lambda: reset_calls.append(None))
    monkeypatch.setattr(admin, "poll_access_token", _fake_poll_access_token)
    monkeypatch.setattr(admin, "get_copilot_token", _fake_get_copilot_token)

    await admin._poll_oauth_completion("sess-1", "device-code", 1)

    manager = _AuthManagerSpy.instances[0]
    credential = manager.storage.saved["github-copilot"]
    assert credential.api_endpoint == "https://api.enterprise.githubcopilot.com"
    assert manager.save_called is True
    assert reset_calls == [None]
    assert sessions.updates[0]["status"] == "complete"


@pytest.mark.asyncio
async def test_update_priorities_rebuilds_router_after_cas(tmp_path):
    repository = RuntimeConfigRepository(tmp_path / "priorities.json")
    initial = repository.read()

    class _Owner:
        snapshots = []

        async def rebuild(self, *, config_snapshot=None):
            self.snapshots.append(config_snapshot)
            return 2

    owner = _Owner()
    wire_response = Response()
    request = PrioritiesUpdateRequest(
        **initial.config.model_dump(mode="json"),
        revision=initial.revision,
    )
    request.priorities = ["github-copilot/gpt-4o"]

    response = await admin.update_priorities(
        request,
        wire_response,
        repository,
        owner,
    )

    assert response.priorities == ["github-copilot/gpt-4o"]
    assert repository.read().config.priorities == ["github-copilot/gpt-4o"]
    assert [snapshot.revision for snapshot in owner.snapshots] == [response.revision]


@pytest.mark.asyncio
async def test_admin_models_returns_unique_provider_qualified_public_ids(monkeypatch):
    class _ModelRouter:
        async def list_models(self):
            return [
                ProviderModelInfo(id="shared-model", name="Shared", provider="first"),
                ProviderModelInfo(id="shared-model", name="Shared", provider="second"),
            ]

    monkeypatch.setattr(admin, "get_router", lambda: _ModelRouter())

    response = await admin.list_models()

    assert [model.id for model in response.models] == [
        "first/shared-model",
        "second/shared-model",
    ]
    assert [model.provider for model in response.models] == ["first", "second"]
