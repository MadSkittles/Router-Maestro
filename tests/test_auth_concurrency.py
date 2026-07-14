"""Concurrency contracts for provider credential persistence."""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from router_maestro.auth.manager import AuthManager
from router_maestro.auth.repository import CredentialRepository
from router_maestro.auth.storage import ApiKeyCredential, AuthStorage, OAuthCredential
from router_maestro.providers.copilot_support.auth_session import CopilotAuthSession


def _oauth(access: str = "copilot-old") -> OAuthCredential:
    return OAuthCredential(
        refresh="github-refresh",
        access=access,
        expires=0,
        api_endpoint="https://api.githubcopilot.com",
    )


def test_update_provider_reads_latest_before_patching(tmp_path) -> None:
    path = tmp_path / "auth.json"
    first = CredentialRepository(path)
    second = CredentialRepository(path)
    first.update_provider("github-copilot", _oauth())

    stale = first.read()
    second.update_provider("openai", ApiKeyCredential(key="openai-key"))
    stale.set("github-copilot", _oauth("copilot-new"))
    first.update_provider("github-copilot", stale.get("github-copilot"))

    stored = first.read()
    assert stored.get("openai") == ApiKeyCredential(key="openai-key")
    assert stored.get("github-copilot") == _oauth("copilot-new")


def test_repository_instances_serialize_concurrent_provider_updates(tmp_path) -> None:
    path = tmp_path / "nested" / ".." / "auth.json"
    repositories = [CredentialRepository(path), CredentialRepository(path.resolve())]
    barrier = threading.Barrier(3)

    def update(index: int) -> None:
        barrier.wait()
        repositories[index].update_provider(
            ("openai", "anthropic")[index],
            ApiKeyCredential(key=f"key-{index}"),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(update, index) for index in range(2)]
        barrier.wait()
        for future in futures:
            future.result()

    stored = CredentialRepository(path).read()
    assert stored.get("openai") == ApiKeyCredential(key="key-0")
    assert stored.get("anthropic") == ApiKeyCredential(key="key-1")


def test_remove_provider_preserves_unrelated_credentials(tmp_path) -> None:
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("openai", ApiKeyCredential(key="openai-key"))
    repository.update_provider("anthropic", ApiKeyCredential(key="anthropic-key"))

    assert repository.remove_provider("openai") is True

    stored = repository.read()
    assert stored.get("openai") is None
    assert stored.get("anthropic") == ApiKeyCredential(key="anthropic-key")


def test_remove_missing_provider_does_not_rewrite_file(tmp_path, monkeypatch) -> None:
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("openai", ApiKeyCredential(key="openai-key"))
    original_save = AuthStorage.save
    save_calls: list[object] = []

    def recording_save(storage: AuthStorage, path) -> None:
        save_calls.append(path)
        original_save(storage, path)

    monkeypatch.setattr(AuthStorage, "save", recording_save)

    assert repository.remove_provider("missing") is False
    assert save_calls == []


def test_repository_reads_and_credentials_are_defensive_copies(tmp_path) -> None:
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("openai", ApiKeyCredential(key="original"))

    snapshot = repository.read()
    snapshot.set("anthropic", ApiKeyCredential(key="in-memory-only"))
    credential = repository.get_provider("openai")
    assert isinstance(credential, ApiKeyCredential)
    credential.key = "mutated"

    assert repository.get_provider("anthropic") is None
    assert repository.get_provider("openai") == ApiKeyCredential(key="original")


def test_manager_observes_credentials_written_after_construction(tmp_path) -> None:
    path = tmp_path / "auth.json"
    manager = AuthManager(CredentialRepository(path))

    CredentialRepository(path).update_provider("openai", ApiKeyCredential(key="latest"))

    assert manager.get_credential("openai") == ApiKeyCredential(key="latest")
    assert manager.list_authenticated() == ["openai"]


@pytest.mark.skipif(
    not hasattr(os, "fchmod"),
    reason="POSIX permission bits are not applicable on this platform",
)
def test_atomic_provider_update_preserves_owner_only_permissions(tmp_path) -> None:
    path = tmp_path / "auth.json"
    repository = CredentialRepository(path)

    repository.update_provider("openai", ApiKeyCredential(key="secret"))

    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_copilot_refresh_preserves_concurrent_provider_login(tmp_path) -> None:
    repository = CredentialRepository(tmp_path / "auth.json")
    repository.update_provider("github-copilot", _oauth())
    session = CopilotAuthSession(repository)

    async def mint(_client, _refresh):
        repository.update_provider("openai", ApiKeyCredential(key="openai-key"))
        return SimpleNamespace(
            token="copilot-new",
            expires_at=2**31,
            api_endpoint="https://api.githubcopilot.com",
        )

    await session.ensure_token(mint=mint)

    stored = repository.read()
    assert stored.get("openai") == ApiKeyCredential(key="openai-key")
    assert stored.get("github-copilot") == _oauth("copilot-new").model_copy(
        update={"expires": 2**31}
    )


@pytest.mark.asyncio
async def test_copilot_persists_refreshed_credential_off_event_loop(tmp_path) -> None:
    class RecordingRepository(CredentialRepository):
        def __init__(self, path) -> None:
            super().__init__(path)
            self.update_threads: list[int] = []

        def update_provider(self, provider, credential) -> None:
            self.update_threads.append(threading.get_ident())
            super().update_provider(provider, credential)

    repository = RecordingRepository(tmp_path / "auth.json")
    repository.update_provider("github-copilot", _oauth())
    repository.update_threads.clear()
    session = CopilotAuthSession(repository)
    event_loop_thread = threading.get_ident()

    async def mint(_client, _refresh):
        await asyncio.sleep(0)
        return SimpleNamespace(
            token="copilot-new",
            expires_at=2**31,
            api_endpoint=None,
        )

    await session.ensure_token(mint=mint)

    assert len(repository.update_threads) == 1
    assert repository.update_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_copilot_refresh_persists_to_injected_manager_repository(tmp_path) -> None:
    original_repository = CredentialRepository(tmp_path / "original-auth.json")
    injected_repository = CredentialRepository(tmp_path / "injected-auth.json")
    injected_repository.update_provider("github-copilot", _oauth())
    session = CopilotAuthSession(original_repository)
    session.auth_manager = AuthManager(injected_repository)

    async def mint(_client, _refresh):
        return SimpleNamespace(
            token="copilot-new",
            expires_at=2**31,
            api_endpoint=None,
        )

    await session.ensure_token(mint=mint)

    assert injected_repository.get_provider("github-copilot") == _oauth("copilot-new").model_copy(
        update={"expires": 2**31}
    )
    assert original_repository.get_provider("github-copilot") is None
