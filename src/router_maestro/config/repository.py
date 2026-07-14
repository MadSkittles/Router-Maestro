"""Versioned runtime configuration snapshots and persistence."""

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from router_maestro.config.paths import PRIORITIES_FILE
from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.config.settings import load_config, write_json_owner_only

_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCKS_BY_PATH: dict[str, threading.RLock] = {}


@dataclass(frozen=True, slots=True)
class RuntimeConfigSnapshot:
    """An immutable serialized runtime configuration at one content revision."""

    revision: str
    canonical_json: bytes = field(repr=False)

    @property
    def config(self) -> PrioritiesConfig:
        """Return a fresh validated model for this snapshot."""
        return PrioritiesConfig.model_validate_json(self.canonical_json)


class RuntimeConfigConflictError(RuntimeError):
    """Raised when a compare-and-swap uses a stale revision."""

    expected_revision: str
    current_revision: str

    def __init__(
        self,
        *,
        expected_revision: str,
        current_revision: str,
    ) -> None:
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        super().__init__(
            "Runtime configuration revision conflict: "
            f"expected {expected_revision}, current {current_revision}"
        )


def canonical_runtime_config_bytes(config: PrioritiesConfig) -> bytes:
    """Serialize validated runtime configuration into stable canonical JSON."""
    validated = PrioritiesConfig.model_validate(config.model_dump(mode="json"))
    payload = validated.model_dump(
        mode="json",
        exclude_none=False,
        exclude_defaults=False,
    )
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def runtime_config_revision(canonical_json: bytes) -> str:
    """Return the SHA-256 content revision for canonical configuration bytes."""
    return hashlib.sha256(canonical_json).hexdigest()


def _lock_key(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _lock_for_path(path: Path) -> threading.RLock:
    key = _lock_key(path)
    with _LOCK_REGISTRY_GUARD:
        return _LOCKS_BY_PATH.setdefault(key, threading.RLock())


def _snapshot(config: PrioritiesConfig) -> RuntimeConfigSnapshot:
    canonical_json = canonical_runtime_config_bytes(config)
    return RuntimeConfigSnapshot(
        revision=runtime_config_revision(canonical_json),
        canonical_json=canonical_json,
    )


class RuntimeConfigRepository:
    """Coordinate in-process reads and atomic writes of runtime configuration."""

    def __init__(self, path: Path = PRIORITIES_FILE) -> None:
        self.path = path
        self._lock = _lock_for_path(path)

    def read(self) -> RuntimeConfigSnapshot:
        """Read and validate the latest runtime configuration snapshot."""
        with self._lock:
            config = load_config(self.path, PrioritiesConfig, PrioritiesConfig.get_default)
            return _snapshot(config)

    def prepare(self, replacement: PrioritiesConfig) -> RuntimeConfigSnapshot:
        """Validate and canonicalize a candidate without persisting it."""
        return _snapshot(replacement)

    def compare_and_swap(
        self,
        *,
        expected_revision: str,
        replacement: PrioritiesConfig,
    ) -> RuntimeConfigSnapshot:
        """Persist replacement only when expected_revision is still current."""
        replacement_snapshot = _snapshot(replacement)
        with self._lock:
            current = self.read()
            if current.revision != expected_revision:
                raise RuntimeConfigConflictError(
                    expected_revision=expected_revision,
                    current_revision=current.revision,
                )
            if current.canonical_json == replacement_snapshot.canonical_json:
                return current
            write_json_owner_only(
                self.path,
                json.loads(replacement_snapshot.canonical_json),
            )
            return replacement_snapshot

    def write_compat(self, replacement: PrioritiesConfig) -> RuntimeConfigSnapshot:
        """Persist replacement with legacy last-writer-wins behavior."""
        replacement_snapshot = _snapshot(replacement)
        with self._lock:
            write_json_owner_only(
                self.path,
                json.loads(replacement_snapshot.canonical_json),
            )
            return replacement_snapshot
