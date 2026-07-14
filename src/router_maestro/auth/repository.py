"""Atomic, single-provider credential persistence."""

from __future__ import annotations

import threading
from pathlib import Path

from router_maestro.auth.storage import AuthStorage, Credential
from router_maestro.config.paths import AUTH_FILE

_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _lock_for(path: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(path, threading.RLock())


class CredentialRepository:
    """Read the latest auth file and atomically patch one provider at a time."""

    def __init__(self, path: Path = AUTH_FILE) -> None:
        self.path = _canonical_path(path)
        self._lock = _lock_for(self.path)

    def read(self) -> AuthStorage:
        """Return a defensive snapshot of the latest credential file."""
        with self._lock:
            return AuthStorage.load(self.path).model_copy(deep=True)

    def get_provider(self, provider: str) -> Credential | None:
        """Return a defensive copy of one provider credential."""
        credential = self.read().get(provider)
        return credential.model_copy(deep=True) if credential is not None else None

    def list_providers(self) -> list[str]:
        """Return provider names from the latest credential file."""
        return self.read().list_providers()

    def update_provider(self, provider: str, credential: Credential) -> None:
        """Read latest, replace one provider, and atomically persist the result."""
        with self._lock:
            storage = AuthStorage.load(self.path)
            storage.set(provider, credential.model_copy(deep=True))
            storage.save(self.path)

    def remove_provider(self, provider: str) -> bool:
        """Read latest and remove one provider without disturbing other entries."""
        with self._lock:
            storage = AuthStorage.load(self.path)
            if not storage.remove(provider):
                return False
            storage.save(self.path)
            return True
