"""Contract tests for versioned runtime configuration snapshots."""

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from router_maestro.config import repository as repository_module
from router_maestro.config import settings
from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.config.repository import (
    RuntimeConfigConflictError,
    RuntimeConfigRepository,
    RuntimeConfigSnapshot,
    canonical_runtime_config_bytes,
    runtime_config_revision,
)


@pytest.fixture
def priorities_path(tmp_path):
    return tmp_path / "priorities.json"


@pytest.fixture
def repositories(priorities_path):
    return (
        RuntimeConfigRepository(priorities_path),
        RuntimeConfigRepository(priorities_path),
    )


def _config(*priorities: str, max_retries: int = 2) -> PrioritiesConfig:
    return PrioritiesConfig(
        priorities=list(priorities),
        fallback={"maxRetries": max_retries},
    )


def test_revision_is_sha256_of_canonical_validated_json():
    config = PrioritiesConfig(
        priorities=["custom/模型"],
        model_overrides={
            "custom/模型": {
                "max_prompt_tokens": None,
                "max_output_tokens": 4096,
            }
        },
    )
    validated = PrioritiesConfig.model_validate(config.model_dump(mode="json"))
    payload = validated.model_dump(
        mode="json",
        exclude_none=False,
        exclude_defaults=False,
    )
    expected = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    canonical_json = canonical_runtime_config_bytes(config)
    revision = runtime_config_revision(canonical_json)

    assert canonical_json == expected
    assert "模型".encode() in canonical_json
    assert json.loads(canonical_json)["audit"]["trace_dir"] is None
    assert revision == hashlib.sha256(expected).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", revision)


def test_semantically_equal_formatting_key_order_and_defaults_share_revision(priorities_path):
    repository = RuntimeConfigRepository(priorities_path)
    priorities_path.write_text(
        '{ "priorities" : [ "github-copilot/gpt-5" ] }\n',
        encoding="utf-8",
    )
    minimal = repository.read()

    expanded = minimal.config.model_dump(
        mode="json",
        exclude_none=False,
        exclude_defaults=False,
    )
    reordered = dict(reversed(expanded.items()))
    reordered["fallback"] = dict(reversed(reordered["fallback"].items()))
    priorities_path.write_text(json.dumps(reordered, indent=7), encoding="utf-8")
    explicit_defaults = repository.read()

    assert explicit_defaults.revision == minimal.revision
    assert explicit_defaults.canonical_json == minimal.canonical_json


def test_revision_ignores_mtime_only_changes(priorities_path):
    repository = RuntimeConfigRepository(priorities_path)
    snapshot = repository.write_compat(_config("github-copilot/gpt-5"))
    stat = priorities_path.stat()

    os.utime(
        priorities_path,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )

    assert repository.read().revision == snapshot.revision


def test_revision_keeps_list_order_significant():
    first = canonical_runtime_config_bytes(_config("provider/a", "provider/b"))
    reversed_order = canonical_runtime_config_bytes(_config("provider/b", "provider/a"))

    assert runtime_config_revision(first) != runtime_config_revision(reversed_order)


def test_snapshot_survives_backing_file_replacement(repositories):
    first_repository, second_repository = repositories
    original = first_repository.write_compat(_config("provider/original"))
    original_bytes = original.canonical_json

    replacement = second_repository.write_compat(_config("provider/replacement"))

    assert original.config.priorities == ["provider/original"]
    assert original.canonical_json == original_bytes
    assert original.revision == runtime_config_revision(original_bytes)
    assert replacement.revision != original.revision
    assert first_repository.read().revision == replacement.revision


def test_snapshot_config_property_returns_defensive_models(priorities_path):
    snapshot = RuntimeConfigRepository(priorities_path).write_compat(
        _config("provider/original", max_retries=3)
    )
    original_bytes = snapshot.canonical_json
    first = snapshot.config
    second = snapshot.config

    first.add_priority("provider", "mutated")
    first.fallback.maxRetries = 9

    assert first is not second
    assert first.fallback is not second.fallback
    assert second.priorities == ["provider/original"]
    assert second.fallback.maxRetries == 3
    assert snapshot.config.priorities == ["provider/original"]
    assert snapshot.canonical_json == original_bytes
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = "mutable"  # type: ignore[misc]


def test_compare_and_swap_writes_replacement_and_new_revision(priorities_path):
    repository = RuntimeConfigRepository(priorities_path)
    initial = repository.read()
    replacement = _config("github-copilot/gpt-5", max_retries=4)

    updated = repository.compare_and_swap(
        expected_revision=initial.revision,
        replacement=replacement,
    )

    assert updated.revision != initial.revision
    assert updated.revision == runtime_config_revision(updated.canonical_json)
    assert updated.config == replacement
    assert repository.read() == updated
    assert json.loads(priorities_path.read_bytes()) == json.loads(updated.canonical_json)


def test_compare_and_swap_rejects_stale_revision_without_writing(
    priorities_path,
    monkeypatch,
):
    first_repository = RuntimeConfigRepository(priorities_path)
    second_repository = RuntimeConfigRepository(priorities_path)
    stale = first_repository.write_compat(_config("provider/base"))
    current = second_repository.compare_and_swap(
        expected_revision=stale.revision,
        replacement=_config("provider/current"),
    )
    persisted_before_conflict = priorities_path.read_bytes()
    writer_calls = []

    def record_writer(*args, **kwargs):
        writer_calls.append((args, kwargs))

    monkeypatch.setattr(repository_module, "write_json_owner_only", record_writer)

    with pytest.raises(RuntimeConfigConflictError) as exc_info:
        first_repository.compare_and_swap(
            expected_revision=stale.revision,
            replacement=_config("provider/stale-write"),
        )

    assert exc_info.value.expected_revision == stale.revision
    assert exc_info.value.current_revision == current.revision
    assert writer_calls == []
    assert priorities_path.read_bytes() == persisted_before_conflict


def test_two_repositories_with_same_revision_have_exactly_one_cas_winner(repositories):
    first_repository, second_repository = repositories
    initial = first_repository.write_compat(_config("provider/base"))
    barrier = threading.Barrier(2)

    def attempt(repository, replacement):
        barrier.wait(timeout=5)
        try:
            return repository.compare_and_swap(
                expected_revision=initial.revision,
                replacement=replacement,
            )
        except RuntimeConfigConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(attempt, first_repository, _config("provider/first")),
            executor.submit(attempt, second_repository, _config("provider/second")),
        ]
        outcomes = [future.result(timeout=5) for future in futures]

    winners = [outcome for outcome in outcomes if isinstance(outcome, RuntimeConfigSnapshot)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, RuntimeConfigConflictError)]
    assert len(winners) == 1
    assert len(conflicts) == 1
    assert conflicts[0].expected_revision == initial.revision
    assert conflicts[0].current_revision == winners[0].revision
    assert first_repository.read().revision == winners[0].revision
    assert first_repository.read().config.priorities in [
        ["provider/first"],
        ["provider/second"],
    ]


def test_noop_compare_and_swap_skips_atomic_write(priorities_path, monkeypatch):
    repository = RuntimeConfigRepository(priorities_path)
    current = repository.write_compat(_config("provider/unchanged"))

    def fail_writer(*args, **kwargs):
        pytest.fail(f"no-op CAS unexpectedly wrote config: {args!r} {kwargs!r}")

    monkeypatch.setattr(repository_module, "write_json_owner_only", fail_writer)

    unchanged = repository.compare_and_swap(
        expected_revision=current.revision,
        replacement=current.config,
    )

    assert unchanged == current


def test_repository_instances_share_lock_for_resolved_path_aliases(priorities_path):
    nested = priorities_path.parent / "nested"
    nested.mkdir()
    alias = nested / ".." / priorities_path.name

    direct_repository = RuntimeConfigRepository(priorities_path)
    alias_repository = RuntimeConfigRepository(alias)

    assert direct_repository._lock is alias_repository._lock


def test_missing_file_initializes_valid_default_snapshot(priorities_path):
    repository = RuntimeConfigRepository(priorities_path)

    snapshot = repository.read()

    expected = PrioritiesConfig.get_default()
    assert priorities_path.exists()
    assert snapshot.config == expected
    assert snapshot.canonical_json == canonical_runtime_config_bytes(expected)
    assert snapshot.revision == runtime_config_revision(snapshot.canonical_json)


def test_compat_write_uses_same_path_lock(priorities_path, monkeypatch):
    first_repository = RuntimeConfigRepository(priorities_path)
    second_repository = RuntimeConfigRepository(priorities_path)
    real_writer = repository_module.write_json_owner_only
    writer_called = threading.Event()
    write_started = threading.Event()

    def observed_writer(path, data):
        writer_called.set()
        real_writer(path, data)

    def write_from_second_repository():
        write_started.set()
        return second_repository.write_compat(_config("provider/replacement"))

    monkeypatch.setattr(repository_module, "write_json_owner_only", observed_writer)
    first_repository._lock.acquire()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(write_from_second_repository)
        assert write_started.wait(timeout=5)
        assert not writer_called.wait(timeout=0.1)
    finally:
        first_repository._lock.release()

    try:
        snapshot = future.result(timeout=5)
    finally:
        executor.shutdown(wait=True)

    assert writer_called.is_set()
    assert snapshot.config.priorities == ["provider/replacement"]


def test_settings_priorities_wrappers_return_defensive_models_and_keep_last_writer_wins(
    priorities_path,
    monkeypatch,
):
    monkeypatch.setattr(settings, "PRIORITIES_FILE", priorities_path)
    settings.save_priorities_config(_config("provider/base"))

    first = settings.load_priorities_config()
    second = settings.load_priorities_config()
    first.add_priority("provider", "first")

    assert second.priorities == ["provider/base"]

    settings.save_priorities_config(first)
    second.add_priority("provider", "second")
    settings.save_priorities_config(second)

    assert settings.load_priorities_config() == second
    assert settings.load_priorities_config().priorities == ["provider/base", "provider/second"]


def test_settings_priorities_wrappers_share_repository_lock(priorities_path, monkeypatch):
    monkeypatch.setattr(settings, "PRIORITIES_FILE", priorities_path)
    repository = RuntimeConfigRepository(priorities_path)
    real_writer = repository_module.write_json_owner_only
    writer_called = threading.Event()
    save_started = threading.Event()

    def observed_writer(path, data):
        writer_called.set()
        real_writer(path, data)

    def save_through_settings():
        save_started.set()
        settings.save_priorities_config(_config("provider/settings"))

    monkeypatch.setattr(repository_module, "write_json_owner_only", observed_writer)
    repository._lock.acquire()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(save_through_settings)
        assert save_started.wait(timeout=5)
        assert not writer_called.wait(timeout=0.1)
    finally:
        repository._lock.release()

    try:
        future.result(timeout=5)
    finally:
        executor.shutdown(wait=True)

    assert writer_called.is_set()
    assert repository.read().config.priorities == ["provider/settings"]


def test_settings_priorities_save_preserves_atomic_persistence(priorities_path, monkeypatch):
    monkeypatch.setattr(settings, "PRIORITIES_FILE", priorities_path)
    settings.save_priorities_config(_config("provider/original"))
    original = priorities_path.read_bytes()

    def fail_after_partial_write(data, file, *args, **kwargs):
        file.write('{"partial":')
        file.flush()
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(settings.json, "dump", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="simulated serialization failure"):
        settings.save_priorities_config(_config("provider/replacement"))

    assert priorities_path.read_bytes() == original
    assert list(priorities_path.parent.glob("*.tmp")) == []
