"""Authenticated envelopes for provider-owned opaque reasoning state."""

import base64
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from router_maestro.config.paths import REASONING_CAPSULE_KEYS_FILE

REASONING_CAPSULE_KEY_ENV: Final = "ROUTER_MAESTRO_REASONING_CAPSULE_KEY"
REASONING_CAPSULE_PREVIOUS_KEYS_ENV: Final = "ROUTER_MAESTRO_REASONING_CAPSULE_PREVIOUS_KEYS"

_CAPSULE_VERSION: Final = "rmr1"
_KEY_FILE_VERSION: Final = 1
_OPAQUE_STATE_VERSION: Final = 1
_KEY_BYTES: Final = 32
_NONCE_BYTES: Final = 12
_PAYLOAD_FIELDS: Final = frozenset({"provider", "model", "transport", "item_id", "opaque_state"})
_CAPSULE_ERROR_MESSAGE: Final = "Invalid reasoning capsule"
_KEY_ERROR_MESSAGE: Final = "Invalid reasoning capsule key configuration"


class ReasoningCapsuleError(ValueError):
    """Raised when a capsule cannot be authenticated or validated."""


class ReasoningCapsuleKeyError(RuntimeError):
    """Raised when reasoning capsule key configuration is unusable."""


@dataclass(frozen=True, slots=True)
class ReasoningCapsulePayload:
    """Provider state and the provenance required to use it safely."""

    provider: str
    model: str
    transport: str
    item_id: str
    opaque_state: str


def _thaw_opaque_state(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_opaque_state(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_opaque_state(item) for item in value]
    return value


def serialize_opaque_state(value: Any) -> str:
    """Encode provider-owned state for authenticated capsule storage."""
    if isinstance(value, bytes):
        payload = {
            "version": _OPAQUE_STATE_VERSION,
            "kind": "bytes",
            "value": base64.urlsafe_b64encode(value).decode("ascii"),
        }
    else:
        payload = {
            "version": _OPAQUE_STATE_VERSION,
            "kind": "json",
            "value": _thaw_opaque_state(value),
        }
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        raise ValueError("opaque reasoning state is not serializable") from None


def deserialize_opaque_state(value: str) -> Any:
    """Decode authenticated provider state after capsule provenance checks."""
    raw = json.loads(value)
    if not isinstance(raw, dict) or set(raw) != {"version", "kind", "value"}:
        raise ValueError
    if raw["version"] != _OPAQUE_STATE_VERSION:
        raise ValueError
    if raw["kind"] == "json":
        return raw["value"]
    if raw["kind"] == "bytes" and isinstance(raw["value"], str):
        return base64.b64decode(raw["value"], altchars=b"-_", validate=True)
    raise ValueError


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    if not value or any(character.isspace() for character in value):
        raise ValueError
    padding = "=" * (-len(value) % 4)
    decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    if _b64url_encode(decoded) != value:
        raise ValueError
    return decoded


def _decode_key(value: str) -> bytes:
    try:
        key = _b64url_decode(value)
    except (TypeError, ValueError) as exc:
        raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE) from exc
    if len(key) != _KEY_BYTES:
        raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE)
    return key


def _key_id(key: bytes) -> str:
    """Return a stable, non-secret identifier suitable for key rotation."""
    return _b64url_encode(hashlib.sha256(key).digest()[:12])


class ReasoningCapsuleCodec:
    """Seal and authenticate opaque provider state with a rotating key ring."""

    def __init__(self, current_key: bytes, previous_keys: Iterable[bytes] = ()) -> None:
        if not isinstance(current_key, bytes) or len(current_key) != _KEY_BYTES:
            raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE)

        self._current_kid = _key_id(current_key)
        keys: dict[str, bytes] = {self._current_kid: current_key}
        for key in previous_keys:
            if not isinstance(key, bytes) or len(key) != _KEY_BYTES:
                raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE)
            kid = _key_id(key)
            existing = keys.get(kid)
            if existing is not None and existing != key:
                raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE)
            keys[kid] = key
        self._keys = keys

    @property
    def current_key_id(self) -> str:
        """Identifier of the only key used to seal new capsules."""
        return self._current_kid

    def seal(self, payload: ReasoningCapsulePayload) -> str:
        """Encrypt a canonical payload using the current key."""
        if not isinstance(payload, ReasoningCapsulePayload):
            raise TypeError("payload must be a ReasoningCapsulePayload")
        values = asdict(payload)
        if any(not isinstance(value, str) for value in values.values()):
            raise TypeError("reasoning capsule payload fields must be strings")

        aad = f"{_CAPSULE_VERSION}.{self._current_kid}".encode("ascii")
        plaintext = json.dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        nonce = os.urandom(_NONCE_BYTES)
        encrypted = AESGCM(self._keys[self._current_kid]).encrypt(nonce, plaintext, aad)
        return f"{_CAPSULE_VERSION}.{self._current_kid}.{_b64url_encode(nonce + encrypted)}"

    def unseal(
        self,
        capsule: str,
        *,
        expected_provider: str,
        expected_model: str,
        expected_transport: str,
        expected_item_id: str | None = None,
    ) -> ReasoningCapsulePayload:
        """Authenticate a capsule and enforce its expected provenance."""
        payload = self.unseal_for_routing(capsule)
        if (
            payload.provider != expected_provider
            or payload.model != expected_model
            or payload.transport != expected_transport
            or (expected_item_id is not None and payload.item_id != expected_item_id)
        ):
            raise ReasoningCapsuleError(_CAPSULE_ERROR_MESSAGE)
        return payload

    def unseal_for_routing(self, capsule: str) -> ReasoningCapsulePayload:
        """Authenticate a capsule and return the affinity that routing must freeze.

        This method is intentionally distinct from :meth:`unseal`: callers may
        use it only to select the exact provider/model/binding recorded in the
        capsule.  The selected attempt must still call :meth:`unseal` with that
        expected provenance before restoring opaque provider state.
        """
        try:
            version, kid, encoded = capsule.split(".")
            if version != _CAPSULE_VERSION:
                raise ValueError
            key = self._keys.get(kid)
            if key is None or _key_id(key) != kid:
                raise ValueError

            packed = _b64url_decode(encoded)
            if len(packed) <= _NONCE_BYTES:
                raise ValueError
            nonce, encrypted = packed[:_NONCE_BYTES], packed[_NONCE_BYTES:]
            aad = f"{version}.{kid}".encode("ascii")
            plaintext = AESGCM(key).decrypt(nonce, encrypted, aad)
            raw_payload = json.loads(plaintext)
            return self._validate_payload(raw_payload)
        except ReasoningCapsuleError:
            raise
        except (
            AttributeError,
            InvalidTag,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            raise ReasoningCapsuleError(_CAPSULE_ERROR_MESSAGE) from None

    @staticmethod
    def _validate_payload(raw_payload: object) -> ReasoningCapsulePayload:
        if not isinstance(raw_payload, dict) or set(raw_payload) != _PAYLOAD_FIELDS:
            raise ValueError
        if any(not isinstance(value, str) for value in raw_payload.values()):
            raise ValueError
        return ReasoningCapsulePayload(**raw_payload)


def _parse_previous_keys(value: str | None) -> list[bytes]:
    if value is None:
        return []
    encoded_keys = value.split(",")
    if not encoded_keys or any(not encoded_key for encoded_key in encoded_keys):
        raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE)
    return [_decode_key(encoded_key) for encoded_key in encoded_keys]


def _codec_from_encoded_keys(
    current_key: str,
    previous_keys: Iterable[str],
) -> ReasoningCapsuleCodec:
    return ReasoningCapsuleCodec(
        _decode_key(current_key),
        (_decode_key(key) for key in previous_keys),
    )


def _read_key_file(path: Path) -> ReasoningCapsuleCodec:
    try:
        if os.name != "nt" and path.stat().st_mode & 0o077:
            raise ValueError
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or set(data) != {
            "version",
            "current_key",
            "previous_keys",
        }:
            raise ValueError
        if data["version"] != _KEY_FILE_VERSION:
            raise ValueError
        current_key = data["current_key"]
        previous_keys = data["previous_keys"]
        if not isinstance(current_key, str) or not isinstance(previous_keys, list):
            raise ValueError
        if any(not isinstance(key, str) for key in previous_keys):
            raise ValueError
        return _codec_from_encoded_keys(current_key, previous_keys)
    except ReasoningCapsuleKeyError:
        raise
    except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE) from exc


def _create_key_file(path: Path) -> ReasoningCapsuleCodec:
    """Create the first key file without allowing concurrent writers to diverge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_key = _b64url_encode(os.urandom(_KEY_BYTES))
    data = {
        "version": _KEY_FILE_VERSION,
        "current_key": encoded_key,
        "previous_keys": [],
    }

    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    fdopen_took_fd = False
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fdopen_took_fd = True
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return _read_key_file(path)
        return _codec_from_encoded_keys(encoded_key, [])
    except ReasoningCapsuleKeyError:
        raise
    except OSError as exc:
        raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE) from exc
    finally:
        if not fdopen_took_fd:
            with suppress(OSError):
                os.close(fd)
        with suppress(OSError):
            temporary_path.unlink()


def load_reasoning_capsule_codec(
    *,
    environ: Mapping[str, str] | None = None,
    key_file: Path | None = None,
) -> ReasoningCapsuleCodec:
    """Load keys from the environment or an owner-only XDG fallback file.

    Environment keys are unpadded URL-safe base64. Previous keys are a
    comma-separated list and are used only for decryption.
    """
    source = os.environ if environ is None else environ
    if REASONING_CAPSULE_KEY_ENV in source:
        return ReasoningCapsuleCodec(
            _decode_key(source[REASONING_CAPSULE_KEY_ENV]),
            _parse_previous_keys(source.get(REASONING_CAPSULE_PREVIOUS_KEYS_ENV)),
        )

    if REASONING_CAPSULE_PREVIOUS_KEYS_ENV in source:
        raise ReasoningCapsuleKeyError(_KEY_ERROR_MESSAGE)

    path = REASONING_CAPSULE_KEYS_FILE if key_file is None else key_file
    if path.exists():
        return _read_key_file(path)
    return _create_key_file(path)


__all__ = [
    "REASONING_CAPSULE_KEY_ENV",
    "REASONING_CAPSULE_PREVIOUS_KEYS_ENV",
    "ReasoningCapsuleCodec",
    "ReasoningCapsuleError",
    "ReasoningCapsuleKeyError",
    "ReasoningCapsulePayload",
    "deserialize_opaque_state",
    "load_reasoning_capsule_codec",
    "serialize_opaque_state",
]
