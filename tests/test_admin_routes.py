"""Tests for admin route side effects."""

import asyncio
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks
from starlette.responses import Response

from router_maestro.auth.discovery import ProviderAuthSource
from router_maestro.auth.repository import CredentialRepository
from router_maestro.auth.storage import ApiKeyCredential, AuthType, Credential, OAuthCredential
from router_maestro.config.providers import CustomProviderConfig, ProvidersConfig
from router_maestro.config.repository import RuntimeConfigRepository
from router_maestro.providers import ModelInfo as ProviderModelInfo
from router_maestro.server.routes import admin
from router_maestro.server.schemas.admin import LoginRequest, PrioritiesUpdateRequest


@dataclass
class _AccessToken:
    access_token: str


@dataclass
class _CopilotToken:
    token: str
    expires_at: int
    api_endpoint: str | None


class _CredentialRepositorySpy:
    def __init__(self):
        self.saved: dict[str, Credential] = {}

    def update_provider(self, provider: str, credential: Credential) -> None:
        self.saved[provider] = credential

    def remove_provider(self, provider: str) -> bool:
        return self.saved.pop(provider, None) is not None

    def get_provider(self, provider: str):
        return self.saved.get(provider)

    def compare_and_swap_provider(self, provider: str, *, expected, replacement) -> bool:
        if self.saved.get(provider) != expected:
            return False
        if replacement is None:
            self.saved.pop(provider, None)
        else:
            self.saved[provider] = replacement
        return True


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
    repository = _CredentialRepositorySpy()

    class _Owner:
        snapshots = []

        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            self.snapshots.append(config_snapshot)

    owner = _Owner()
    runtime_snapshot = SimpleNamespace(revision="runtime-revision")
    runtime_repository = SimpleNamespace(read=lambda: runtime_snapshot)

    monkeypatch.setattr(admin, "oauth_sessions", sessions)
    monkeypatch.setattr(admin, "poll_access_token", _fake_poll_access_token)
    monkeypatch.setattr(admin, "get_copilot_token", _fake_get_copilot_token)

    await admin._poll_oauth_completion(
        "sess-1",
        "device-code",
        1,
        owner,
        repository,
        runtime_repository,
    )

    credential = repository.saved["github-copilot"]
    assert credential.api_endpoint == "https://api.enterprise.githubcopilot.com"
    assert owner.snapshots == [runtime_snapshot]
    assert sessions.updates[0]["status"] == "complete"


@pytest.mark.asyncio
async def test_cancelled_oauth_completion_compensates_late_credential_write(
    monkeypatch,
    tmp_path,
):
    sessions = _OAuthSessionsSpy()
    repository = CredentialRepository(tmp_path / "auth.json")
    original = OAuthCredential(refresh="old-refresh", access="old-access", expires=100)
    repository.update_provider("github-copilot", original)
    update_started = threading.Event()
    allow_update = threading.Event()
    update_finished = threading.Event()
    original_update = repository.update_provider

    def blocking_update(provider, credential):
        update_started.set()
        if not allow_update.wait(timeout=5):
            raise TimeoutError("test did not release credential update")
        try:
            original_update(provider, credential)
        finally:
            update_finished.set()

    class _Owner:
        rebuild_calls = 0

        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            self.rebuild_calls += 1

    owner = _Owner()
    monkeypatch.setattr(repository, "update_provider", blocking_update)
    monkeypatch.setattr(admin, "oauth_sessions", sessions)
    monkeypatch.setattr(admin, "poll_access_token", _fake_poll_access_token)
    monkeypatch.setattr(admin, "get_copilot_token", _fake_get_copilot_token)

    poll_task = asyncio.create_task(
        admin._poll_oauth_completion(
            "sess-cancelled",
            "device-code",
            1,
            owner,
            repository,
            SimpleNamespace(read=lambda: SimpleNamespace(revision="runtime-revision")),
        )
    )
    assert await asyncio.to_thread(update_started.wait, 1)

    poll_task.cancel()
    await asyncio.sleep(0)
    allow_update.set()
    assert await asyncio.to_thread(update_finished.wait, 1)

    with pytest.raises(asyncio.CancelledError):
        await poll_task

    assert repository.get_provider("github-copilot") == original
    assert owner.rebuild_calls == 0
    assert sessions.updates == []


@pytest.mark.asyncio
async def test_cancelled_oauth_compensation_ignores_repeated_cancellation_and_concurrent_write(
    monkeypatch,
    tmp_path,
):
    sessions = _OAuthSessionsSpy()
    repository = CredentialRepository(tmp_path / "auth.json")
    original = OAuthCredential(refresh="old-refresh", access="old-access", expires=100)
    concurrent = OAuthCredential(
        refresh="concurrent-refresh",
        access="concurrent-access",
        expires=200,
    )
    repository.update_provider("github-copilot", original)
    update_started = threading.Event()
    allow_update = threading.Event()
    update_finished = threading.Event()
    allow_update_return = threading.Event()
    compensation_started = threading.Event()
    allow_compensation = threading.Event()
    compensation_finished = threading.Event()
    original_update = repository.update_provider
    original_compare_and_swap = repository.compare_and_swap_provider

    def blocking_update(provider, credential):
        update_started.set()
        if not allow_update.wait(timeout=5):
            raise TimeoutError("test did not release credential update")
        original_update(provider, credential)
        update_finished.set()
        if not allow_update_return.wait(timeout=5):
            raise TimeoutError("test did not release credential update return")

    def blocking_compare_and_swap(provider, *, expected, replacement):
        compensation_started.set()
        if not allow_compensation.wait(timeout=5):
            raise TimeoutError("test did not release credential compensation")
        try:
            return original_compare_and_swap(
                provider,
                expected=expected,
                replacement=replacement,
            )
        finally:
            compensation_finished.set()

    class _Owner:
        rebuild_calls = 0

        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            self.rebuild_calls += 1

    owner = _Owner()
    monkeypatch.setattr(repository, "update_provider", blocking_update)
    monkeypatch.setattr(repository, "compare_and_swap_provider", blocking_compare_and_swap)
    monkeypatch.setattr(admin, "oauth_sessions", sessions)
    monkeypatch.setattr(admin, "poll_access_token", _fake_poll_access_token)
    monkeypatch.setattr(admin, "get_copilot_token", _fake_get_copilot_token)

    poll_task = asyncio.create_task(
        admin._poll_oauth_completion(
            "sess-repeated-cancel",
            "device-code",
            1,
            owner,
            repository,
            SimpleNamespace(read=lambda: SimpleNamespace(revision="runtime-revision")),
        )
    )
    assert await asyncio.to_thread(update_started.wait, 1)

    poll_task.cancel()
    await asyncio.sleep(0)
    allow_update.set()
    assert await asyncio.to_thread(update_finished.wait, 1)
    await asyncio.to_thread(
        CredentialRepository.update_provider,
        repository,
        "github-copilot",
        concurrent,
    )
    allow_update_return.set()
    assert await asyncio.to_thread(compensation_started.wait, 1)

    poll_task.cancel()
    await asyncio.sleep(0)
    poll_task.cancel()
    await asyncio.sleep(0)
    assert not poll_task.done()

    allow_compensation.set()
    assert await asyncio.to_thread(compensation_finished.wait, 1)
    with pytest.raises(asyncio.CancelledError):
        await poll_task

    assert repository.get_provider("github-copilot") == concurrent
    assert owner.rebuild_calls == 0
    assert sessions.updates == []


@pytest.mark.asyncio
async def test_oauth_login_background_task_receives_current_router_owner(monkeypatch):
    owner = object()
    runtime_repository = object()
    background_tasks = BackgroundTasks()
    device_code = SimpleNamespace(
        device_code="device-code",
        user_code="user-code",
        verification_uri="https://github.example/device",
        expires_in=900,
        interval=5,
    )
    session = SimpleNamespace(session_id="session-id")

    async def fake_request_device_code(_client):
        return device_code

    async def fake_create_session(**_kwargs):
        return session

    monkeypatch.setattr(admin, "request_device_code", fake_request_device_code)
    monkeypatch.setattr(admin.oauth_sessions, "create_session", fake_create_session)

    await admin.login(
        LoginRequest(provider="github-copilot"),
        background_tasks,
        owner,
        runtime_repository,
    )

    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is admin._poll_oauth_completion
    assert background_tasks.tasks[0].args[3] is owner
    assert background_tasks.tasks[0].args[5] is runtime_repository


@pytest.mark.asyncio
async def test_api_key_login_and_logout_rebuild_router(monkeypatch):
    class _Owner:
        snapshots = []

        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            self.snapshots.append(config_snapshot)

    owner = _Owner()
    credential_repository = _CredentialRepositorySpy()
    runtime_snapshot = SimpleNamespace(revision="runtime-revision")
    runtime_repository = SimpleNamespace(read=lambda: runtime_snapshot)
    monkeypatch.setattr(admin, "CredentialRepository", lambda: credential_repository)

    login_result = await admin.login(
        LoginRequest(provider="openai", api_key="secret"),
        BackgroundTasks(),
        owner,
        runtime_repository,
    )
    logout_result = await admin.logout("openai", owner, runtime_repository)

    assert login_result == {"success": True, "provider": "openai"}
    assert logout_result == {"success": True, "provider": "openai"}
    assert owner.snapshots == [runtime_snapshot, runtime_snapshot]
    assert credential_repository.saved == {}


@pytest.mark.asyncio
async def test_api_key_login_compensates_only_its_credential_when_rebuild_fails(
    monkeypatch,
    tmp_path,
):
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("openai", ApiKeyCredential(key="old-key"))

    class _Owner:
        snapshots = []

        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            self.snapshots.append(config_snapshot)
            raise RuntimeError("router build failed")

    runtime_snapshot = RuntimeConfigRepository(tmp_path / "priorities.json").read()
    manager_type = admin.AuthManager
    monkeypatch.setattr(admin, "CredentialRepository", lambda: repository)
    monkeypatch.setattr(admin, "AuthManager", lambda *_args: manager_type(repository))

    with pytest.raises(RuntimeError, match="router build failed"):
        await admin.login(
            LoginRequest(provider="openai", api_key="new-key"),
            BackgroundTasks(),
            _Owner(),
            SimpleNamespace(read=lambda: runtime_snapshot),
        )

    assert repository.get_provider("openai") == ApiKeyCredential(key="old-key")


@pytest.mark.asyncio
async def test_api_key_login_compensation_never_overwrites_concurrent_credential(
    monkeypatch,
    tmp_path,
):
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("openai", ApiKeyCredential(key="old-key"))

    class _Owner:
        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            repository.update_provider("openai", ApiKeyCredential(key="concurrent-key"))
            raise RuntimeError("router build failed")

    runtime_snapshot = RuntimeConfigRepository(tmp_path / "priorities.json").read()
    manager_type = admin.AuthManager
    monkeypatch.setattr(admin, "CredentialRepository", lambda: repository)
    monkeypatch.setattr(admin, "AuthManager", lambda *_args: manager_type(repository))

    with pytest.raises(RuntimeError, match="router build failed"):
        await admin.login(
            LoginRequest(provider="openai", api_key="new-key"),
            BackgroundTasks(),
            _Owner(),
            SimpleNamespace(read=lambda: runtime_snapshot),
        )

    assert repository.get_provider("openai") == ApiKeyCredential(key="concurrent-key")


@pytest.mark.asyncio
async def test_logout_restores_removed_credential_when_rebuild_fails(monkeypatch, tmp_path):
    repository = CredentialRepository(tmp_path / "auth.json")
    original = ApiKeyCredential(key="old-key")
    repository.update_provider("openai", original)

    class _Owner:
        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            raise RuntimeError("router build failed")

    runtime_snapshot = RuntimeConfigRepository(tmp_path / "priorities.json").read()
    manager_type = admin.AuthManager
    monkeypatch.setattr(admin, "CredentialRepository", lambda: repository)
    monkeypatch.setattr(admin, "AuthManager", lambda *_args: manager_type(repository))

    with pytest.raises(RuntimeError, match="router build failed"):
        await admin.logout(
            "openai",
            _Owner(),
            SimpleNamespace(read=lambda: runtime_snapshot),
        )

    assert repository.get_provider("openai") == original


@pytest.mark.asyncio
async def test_logout_compensation_never_overwrites_concurrent_credential(monkeypatch, tmp_path):
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("openai", ApiKeyCredential(key="old-key"))

    class _Owner:
        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            repository.update_provider("openai", ApiKeyCredential(key="concurrent-key"))
            raise RuntimeError("router build failed")

    runtime_snapshot = RuntimeConfigRepository(tmp_path / "priorities.json").read()
    manager_type = admin.AuthManager
    monkeypatch.setattr(admin, "CredentialRepository", lambda: repository)
    monkeypatch.setattr(admin, "AuthManager", lambda *_args: manager_type(repository))

    with pytest.raises(RuntimeError, match="router build failed"):
        await admin.logout(
            "openai",
            _Owner(),
            SimpleNamespace(read=lambda: runtime_snapshot),
        )

    assert repository.get_provider("openai") == ApiKeyCredential(key="concurrent-key")


@pytest.mark.asyncio
async def test_oauth_completion_restores_previous_credential_when_rebuild_fails(
    monkeypatch,
    tmp_path,
):
    sessions = _OAuthSessionsSpy()
    repository = CredentialRepository(tmp_path / "auth.json")
    original = OAuthCredential(refresh="old-refresh", access="old-access", expires=100)
    repository.update_provider("github-copilot", original)

    class _Owner:
        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            raise RuntimeError("router build failed")

    runtime_snapshot = RuntimeConfigRepository(tmp_path / "priorities.json").read()
    monkeypatch.setattr(admin, "oauth_sessions", sessions)
    monkeypatch.setattr(admin, "poll_access_token", _fake_poll_access_token)
    monkeypatch.setattr(admin, "get_copilot_token", _fake_get_copilot_token)

    await admin._poll_oauth_completion(
        "sess-failed",
        "device-code",
        1,
        _Owner(),
        repository,
        SimpleNamespace(read=lambda: runtime_snapshot),
    )

    assert repository.get_provider("github-copilot") == original
    assert sessions.updates[-1]["status"] == "error"


@pytest.mark.asyncio
async def test_oauth_compensation_never_overwrites_concurrent_credential(monkeypatch, tmp_path):
    sessions = _OAuthSessionsSpy()
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider(
        "github-copilot",
        OAuthCredential(refresh="old-refresh", access="old-access", expires=100),
    )
    concurrent = OAuthCredential(
        refresh="concurrent-refresh",
        access="concurrent-access",
        expires=200,
    )

    class _Owner:
        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            repository.update_provider("github-copilot", concurrent)
            raise RuntimeError("router build failed")

    runtime_snapshot = RuntimeConfigRepository(tmp_path / "priorities.json").read()
    monkeypatch.setattr(admin, "oauth_sessions", sessions)
    monkeypatch.setattr(admin, "poll_access_token", _fake_poll_access_token)
    monkeypatch.setattr(admin, "get_copilot_token", _fake_get_copilot_token)

    await admin._poll_oauth_completion(
        "sess-concurrent",
        "device-code",
        1,
        _Owner(),
        repository,
        SimpleNamespace(read=lambda: runtime_snapshot),
    )

    assert repository.get_provider("github-copilot") == concurrent
    assert sessions.updates[-1]["status"] == "error"


def test_auth_provider_discovery_uses_server_provider_configuration(monkeypatch):
    monkeypatch.setattr(
        admin,
        "load_providers_config",
        lambda: ProvidersConfig(
            providers={
                "ollama": CustomProviderConfig(
                    baseURL="http://localhost:11434/v1",
                    options={"allow_unauthenticated": True},
                )
            }
        ),
    )

    response = admin.list_auth_providers()

    assert [provider.provider for provider in response.providers] == [
        "github-copilot",
        "openai",
        "anthropic",
        "ollama",
    ]
    assert response.providers[-1].auth_type is AuthType.API_KEY
    assert response.providers[-1].source is ProviderAuthSource.CUSTOM
    assert response.providers[-1].credential_required is False
    assert response.providers[-1].api_key_env == "OLLAMA_API_KEY"


@pytest.mark.asyncio
async def test_update_priorities_rebuilds_router_after_cas(tmp_path):
    repository = RuntimeConfigRepository(tmp_path / "priorities.json")
    initial = repository.read()

    class _Owner:
        snapshots = []
        installed_snapshots = []

        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            self.snapshots.append(config_snapshot)
            installed_snapshot = before_swap() if before_swap is not None else config_snapshot
            self.installed_snapshots.append(installed_snapshot)
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
    assert [snapshot.revision for snapshot in owner.installed_snapshots] == [response.revision]


@pytest.mark.asyncio
async def test_admin_models_returns_unique_provider_qualified_public_ids(monkeypatch):
    class _ModelRouter:
        async def list_models(self):
            return [
                ProviderModelInfo(id="shared-model", name="Shared", provider="first"),
                ProviderModelInfo(id="shared-model", name="Shared", provider="second"),
            ]

    class _Lease:
        router = _ModelRouter()
        released = False

        async def release(self):
            self.released = True

    class _Owner:
        lease = _Lease()

        async def acquire(self):
            return self.lease

    owner = _Owner()

    response = await admin.list_models(owner)

    assert [model.id for model in response.models] == [
        "first/shared-model",
        "second/shared-model",
    ]
    assert [model.provider for model in response.models] == ["first", "second"]
    assert owner.lease.released is True


@pytest.mark.asyncio
async def test_admin_model_refresh_rebuilds_generation_and_releases_lease_after_failure():
    events = []

    class _ModelRouter:
        async def list_models(self):
            events.append("list")
            raise RuntimeError("catalog unavailable")

    class _Lease:
        router = _ModelRouter()
        released = False

        async def release(self):
            self.released = True

    class _Owner:
        lease = _Lease()
        snapshots = []

        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            events.append("rebuild")
            self.snapshots.append(config_snapshot)

        async def acquire(self):
            events.append("acquire")
            return self.lease

    owner = _Owner()
    runtime_snapshot = SimpleNamespace(revision="runtime-revision")

    with pytest.raises(Exception) as exc_info:
        await admin.refresh_models(owner, SimpleNamespace(read=lambda: runtime_snapshot))

    assert getattr(exc_info.value, "status_code", None) == 500
    assert events == ["rebuild", "acquire", "list"]
    assert owner.snapshots == [runtime_snapshot]
    assert owner.lease.released is True


@pytest.mark.asyncio
async def test_admin_model_refresh_does_not_acquire_when_rebuild_fails():
    class _Owner:
        acquired = False

        async def rebuild(self, config_snapshot=None, *, before_swap=None):
            raise RuntimeError("router build failed")

        async def acquire(self):
            self.acquired = True
            raise AssertionError("old generation must not be queried after rebuild failure")

    owner = _Owner()
    runtime_snapshot = SimpleNamespace(revision="runtime-revision")

    with pytest.raises(Exception) as exc_info:
        await admin.refresh_models(owner, SimpleNamespace(read=lambda: runtime_snapshot))

    assert getattr(exc_info.value, "status_code", None) == 500
    assert owner.acquired is False
