"""Admin HTTP contract for versioned runtime configuration."""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.config.repository import RuntimeConfigRepository
from router_maestro.server.routes import admin


class _RouterOwnerSpy:
    def __init__(self, *, before_swap_hook=None) -> None:
        self.snapshots = []
        self.installed_snapshots = []
        self.before_swap_calls = 0
        self.before_swap_hook = before_swap_hook

    async def rebuild(self, config_snapshot=None, *, before_swap=None):
        self.snapshots.append(config_snapshot)
        if self.before_swap_hook is not None:
            self.before_swap_hook()
        installed_snapshot = config_snapshot
        if before_swap is not None:
            self.before_swap_calls += 1
            committed_snapshot = before_swap()
            if committed_snapshot is not None:
                installed_snapshot = committed_snapshot
        self.installed_snapshots.append(installed_snapshot)
        return len(self.snapshots) + 1


class _FailingRouterOwner(_RouterOwnerSpy):
    async def rebuild(self, config_snapshot=None, *, before_swap=None):
        self.snapshots.append(config_snapshot)
        raise RuntimeError("router build failed")


def _client(tmp_path):
    application = FastAPI()
    repository = RuntimeConfigRepository(tmp_path / "priorities.json")
    owner = _RouterOwnerSpy()
    application.state.runtime_config_repository = repository
    application.state.router_owner = owner
    application.include_router(admin.router, dependencies=[Depends(lambda: None)])
    return TestClient(application), repository, owner


def test_admin_get_returns_complete_config_revision_and_etag(tmp_path):
    client, repository, _owner = _client(tmp_path)
    repository.write_compat(
        PrioritiesConfig(
            priorities=["github-copilot/gpt-5"],
            model_overrides={"gpt-5": {"max_output_tokens": 8192}},
            beta_strip=["prompt-caching-*"],
            audit={"enabled": True, "trace_dir": "/tmp/traces"},
        )
    )

    response = client.get("/api/admin/priorities")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "priorities",
        "fallback",
        "model_overrides",
        "thinking",
        "guards",
        "beta_strip",
        "audit",
        "revision",
    }
    assert body["model_overrides"]["gpt-5"]["max_output_tokens"] == 8192
    assert body["beta_strip"] == ["prompt-caching-*"]
    assert body["audit"] == {"enabled": True, "trace_dir": "/tmp/traces"}
    assert response.headers["etag"] == f'"{body["revision"]}"'


def test_admin_patch_builds_candidate_then_commits_with_generation_swap(tmp_path):
    client, repository, owner = _client(tmp_path)
    initial = repository.read()
    payload = initial.config.model_dump(mode="json")
    payload["priorities"] = ["github-copilot/gpt-5"]
    payload["revision"] = initial.revision

    response = client.patch("/api/admin/priorities", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["revision"] != initial.revision
    assert repository.read().config.priorities == ["github-copilot/gpt-5"]
    assert [snapshot.revision for snapshot in owner.snapshots] == [body["revision"]]
    assert [snapshot.revision for snapshot in owner.installed_snapshots] == [body["revision"]]
    assert owner.before_swap_calls == 1
    assert response.headers["etag"] == f'"{body["revision"]}"'


def test_admin_stale_patch_returns_conflict_without_write_or_rebuild(tmp_path):
    client, repository, owner = _client(tmp_path)
    stale = repository.read()
    current = repository.compare_and_swap(
        expected_revision=stale.revision,
        replacement=PrioritiesConfig(priorities=["provider/current"]),
    )
    payload = stale.config.model_dump(mode="json")
    payload["priorities"] = ["provider/stale"]
    payload["revision"] = stale.revision

    response = client.patch("/api/admin/priorities", json=payload)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "config_revision_conflict"
    assert response.json()["detail"]["current_revision"] == current.revision
    assert response.headers["etag"] == f'"{current.revision}"'
    assert repository.read().revision == current.revision
    assert owner.snapshots == []


def test_admin_noop_patch_does_not_rebuild(tmp_path):
    client, repository, owner = _client(tmp_path)
    current = repository.read()
    payload = current.config.model_dump(mode="json")
    payload["revision"] = current.revision

    response = client.patch("/api/admin/priorities", json=payload)

    assert response.status_code == 200
    assert response.json()["revision"] == current.revision
    assert owner.snapshots == []


def test_admin_put_alias_requires_revision(tmp_path):
    client, _repository, owner = _client(tmp_path)

    response = client.put(
        "/api/admin/priorities",
        json={"priorities": ["provider/model"], "fallback": {"maxRetries": 2}},
    )

    assert response.status_code == 422
    assert owner.snapshots == []


def test_admin_patch_build_failure_never_commits_candidate(tmp_path):
    application = FastAPI()
    repository = RuntimeConfigRepository(tmp_path / "priorities.json")
    owner = _FailingRouterOwner()
    application.state.runtime_config_repository = repository
    application.state.router_owner = owner
    application.include_router(admin.router)
    client = TestClient(application, raise_server_exceptions=False)
    initial = repository.read()
    payload = initial.config.model_dump(mode="json")
    payload["priorities"] = ["provider/rejected"]
    payload["revision"] = initial.revision

    response = client.patch("/api/admin/priorities", json=payload)

    assert response.status_code == 500
    assert repository.read() == initial
    assert owner.before_swap_calls == 0


def test_admin_patch_cas_conflict_during_swap_returns_409_and_preserves_writer(tmp_path):
    application = FastAPI()
    repository = RuntimeConfigRepository(tmp_path / "priorities.json")
    initial = repository.read()
    concurrent = PrioritiesConfig(priorities=["provider/concurrent"])

    def write_concurrently() -> None:
        current = repository.read()
        repository.compare_and_swap(
            expected_revision=current.revision,
            replacement=concurrent,
        )

    owner = _RouterOwnerSpy(before_swap_hook=write_concurrently)
    application.state.runtime_config_repository = repository
    application.state.router_owner = owner
    application.include_router(admin.router)
    client = TestClient(application, raise_server_exceptions=False)
    payload = initial.config.model_dump(mode="json")
    payload["priorities"] = ["provider/rejected"]
    payload["revision"] = initial.revision

    response = client.patch("/api/admin/priorities", json=payload)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "config_revision_conflict"
    assert repository.read().config == concurrent
    assert owner.before_swap_calls == 1
    assert owner.installed_snapshots == []
