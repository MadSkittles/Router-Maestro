"""Tests for authenticated provider reasoning-state capsules."""

import base64
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from router_maestro.config.paths import REASONING_CAPSULE_KEYS_FILE
from router_maestro.runtime.reasoning_capsule import (
    REASONING_CAPSULE_KEY_ENV,
    REASONING_CAPSULE_PREVIOUS_KEYS_ENV,
    ReasoningCapsuleCodec,
    ReasoningCapsuleError,
    ReasoningCapsuleKeyError,
    ReasoningCapsulePayload,
    load_reasoning_capsule_codec,
)

_ROOT = Path(__file__).resolve().parents[1]


def _encoded_key(byte: int) -> str:
    return base64.urlsafe_b64encode(bytes([byte]) * 32).rstrip(b"=").decode("ascii")


def _payload() -> ReasoningCapsulePayload:
    return ReasoningCapsulePayload(
        provider="github-copilot",
        model="gpt-5.4",
        transport="responses",
        item_id="rs_123",
        opaque_state="opaque-状态",
    )


def _unseal(codec: ReasoningCapsuleCodec, capsule: str) -> ReasoningCapsulePayload:
    return codec.unseal(
        capsule,
        expected_provider="github-copilot",
        expected_model="gpt-5.4",
        expected_transport="responses",
    )


def _tamper(capsule: str) -> str:
    version, kid, encoded = capsule.split(".")
    packed = bytearray(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    packed[-1] ^= 1
    changed = base64.urlsafe_b64encode(packed).rstrip(b"=").decode("ascii")
    return f"{version}.{kid}.{changed}"


def test_capsule_round_trip_uses_versioned_aead_wire_format(monkeypatch) -> None:
    key = bytes([7]) * 32
    nonce = bytes([9]) * 12
    monkeypatch.setattr(os, "urandom", lambda length: nonce if length == 12 else b"x" * length)
    codec = ReasoningCapsuleCodec(key)

    capsule = codec.seal(_payload())

    version, kid, encoded = capsule.split(".")
    assert version == "rmr1"
    assert kid == codec.current_key_id
    assert capsule.startswith(f"rmr1.{codec.current_key_id}.")
    assert "opaque-状态" not in capsule

    packed = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    assert packed[:12] == nonce
    plaintext = AESGCM(key).decrypt(
        nonce,
        packed[12:],
        f"rmr1.{kid}".encode("ascii"),
    )
    assert plaintext == (
        b'{"item_id":"rs_123","model":"gpt-5.4","opaque_state":"opaque-'
        b'\xe7\x8a\xb6\xe6\x80\x81","provider":"github-copilot","transport":"responses"}'
    )
    assert _unseal(codec, capsule) == _payload()


def test_each_seal_uses_a_fresh_nonce() -> None:
    codec = ReasoningCapsuleCodec(bytes([3]) * 32)

    first = codec.seal(_payload())
    second = codec.seal(_payload())

    assert first != second
    assert _unseal(codec, first) == _payload()
    assert _unseal(codec, second) == _payload()


def test_invalid_version_unknown_key_and_tamper_share_safe_typed_error() -> None:
    codec = ReasoningCapsuleCodec(bytes([4]) * 32)
    capsule = codec.seal(_payload())
    _, kid, encoded = capsule.split(".")
    invalid_capsules = [
        f"rmr2.{kid}.{encoded}",
        f"rmr1.unknown-key.{encoded}",
        _tamper(capsule),
        "not-a-capsule",
        None,
    ]

    for invalid in invalid_capsules:
        with pytest.raises(ReasoningCapsuleError) as caught:
            _unseal(codec, invalid)  # type: ignore[arg-type]
        assert str(caught.value) == "Invalid reasoning capsule"


@pytest.mark.parametrize("compose_name", ["docker-compose.yml", "docker-compose.dev.yml"])
def test_compose_forwards_capsule_keys_without_injecting_empty_values(
    compose_name: str,
) -> None:
    source = (_ROOT / compose_name).read_text(encoding="utf-8")

    for variable in (REASONING_CAPSULE_KEY_ENV, REASONING_CAPSULE_PREVIOUS_KEYS_ENV):
        assert f"\n      - {variable}\n" in source
        assert f"{variable}=" not in source


@pytest.mark.parametrize(
    ("overrides", "expected_item_id"),
    [
        ({"expected_provider": "openai"}, None),
        ({"expected_model": "gpt-5.3"}, None),
        ({"expected_transport": "chat"}, None),
        ({}, "rs_other"),
    ],
)
def test_capsule_rejects_provenance_mismatch(overrides, expected_item_id) -> None:
    codec = ReasoningCapsuleCodec(bytes([5]) * 32)
    capsule = codec.seal(_payload())
    expected = {
        "expected_provider": "github-copilot",
        "expected_model": "gpt-5.4",
        "expected_transport": "responses",
    }
    expected.update(overrides)

    with pytest.raises(ReasoningCapsuleError, match="^Invalid reasoning capsule$"):
        codec.unseal(capsule, expected_item_id=expected_item_id, **expected)


def test_rotation_decrypts_old_capsules_but_seals_only_with_current_key() -> None:
    old_key = bytes([6]) * 32
    new_key = bytes([8]) * 32
    old_codec = ReasoningCapsuleCodec(old_key)
    old_capsule = old_codec.seal(_payload())
    rotated = ReasoningCapsuleCodec(new_key, [old_key])

    assert _unseal(rotated, old_capsule) == _payload()

    new_capsule = rotated.seal(_payload())
    assert new_capsule.startswith(f"rmr1.{rotated.current_key_id}.")
    assert not new_capsule.startswith(f"rmr1.{old_codec.current_key_id}.")
    with pytest.raises(ReasoningCapsuleError):
        _unseal(old_codec, new_capsule)


def test_authenticated_affinity_can_freeze_routing_before_exact_unseal() -> None:
    codec = ReasoningCapsuleCodec(bytes([7]) * 32)
    capsule = codec.seal(_payload())

    affinity = codec.unseal_for_routing(capsule)

    assert affinity == _payload()
    assert _unseal(codec, capsule) == affinity


def test_affinity_read_rejects_tampering_without_exposing_payload() -> None:
    codec = ReasoningCapsuleCodec(bytes([9]) * 32)
    capsule = codec.seal(_payload())

    with pytest.raises(ReasoningCapsuleError, match="^Invalid reasoning capsule$"):
        codec.unseal_for_routing(_tamper(capsule))


def test_key_id_is_stable_for_the_same_key() -> None:
    key = bytes([10]) * 32

    assert ReasoningCapsuleCodec(key).current_key_id == ReasoningCapsuleCodec(key).current_key_id


def test_environment_keys_take_precedence_over_a_corrupt_file(tmp_path) -> None:
    key_file = tmp_path / "reasoning-capsule-keys.json"
    original = b"corrupt-fallback-must-not-be-read"
    key_file.write_bytes(original)

    codec = load_reasoning_capsule_codec(
        environ={REASONING_CAPSULE_KEY_ENV: _encoded_key(11)},
        key_file=key_file,
    )

    assert _unseal(codec, codec.seal(_payload())) == _payload()
    assert key_file.read_bytes() == original


def test_environment_rotation_uses_previous_keys_for_decryption() -> None:
    old_codec = load_reasoning_capsule_codec(environ={REASONING_CAPSULE_KEY_ENV: _encoded_key(12)})
    old_capsule = old_codec.seal(_payload())

    rotated = load_reasoning_capsule_codec(
        environ={
            REASONING_CAPSULE_KEY_ENV: _encoded_key(13),
            REASONING_CAPSULE_PREVIOUS_KEYS_ENV: f"{_encoded_key(12)},{_encoded_key(14)}",
        }
    )

    assert _unseal(rotated, old_capsule) == _payload()
    assert rotated.seal(_payload()).startswith(f"rmr1.{rotated.current_key_id}.")


@pytest.mark.parametrize(
    "environ",
    [
        {REASONING_CAPSULE_KEY_ENV: "secret-not-a-valid-key"},
        {REASONING_CAPSULE_PREVIOUS_KEYS_ENV: _encoded_key(15)},
        {
            REASONING_CAPSULE_KEY_ENV: _encoded_key(16),
            REASONING_CAPSULE_PREVIOUS_KEYS_ENV: "secret-not-a-valid-previous-key",
        },
    ],
)
def test_invalid_environment_configuration_fails_closed(environ, tmp_path) -> None:
    key_file = tmp_path / "reasoning-capsule-keys.json"
    valid_fallback = {
        "version": 1,
        "current_key": _encoded_key(17),
        "previous_keys": [],
    }
    key_file.write_text(json.dumps(valid_fallback), encoding="utf-8")

    with pytest.raises(
        ReasoningCapsuleKeyError,
        match="^Invalid reasoning capsule key configuration$",
    ) as caught:
        load_reasoning_capsule_codec(environ=environ, key_file=key_file)

    rendered = "".join(traceback.format_exception(caught.value))
    assert "secret-not-a-valid" not in rendered
    assert json.loads(key_file.read_text(encoding="utf-8")) == valid_fallback


def test_xdg_fallback_is_generated_once_and_reused(tmp_path) -> None:
    key_file = tmp_path / "data" / "router-maestro" / "reasoning-capsule-keys.json"

    first = load_reasoning_capsule_codec(environ={}, key_file=key_file)
    capsule = first.seal(_payload())
    first_contents = key_file.read_bytes()
    second = load_reasoning_capsule_codec(environ={}, key_file=key_file)

    assert _unseal(second, capsule) == _payload()
    assert second.current_key_id == first.current_key_id
    assert key_file.read_bytes() == first_contents
    document = json.loads(first_contents)
    assert set(document) == {"version", "current_key", "previous_keys"}
    assert document["version"] == 1
    assert document["previous_keys"] == []
    current_key = base64.urlsafe_b64decode(document["current_key"] + "=")
    assert len(current_key) == 32
    assert REASONING_CAPSULE_KEYS_FILE.name == "reasoning-capsule-keys.json"


@pytest.mark.skipif(
    not hasattr(os, "fchmod"),
    reason="POSIX permission bits are not applicable on this platform",
)
def test_generated_fallback_key_file_is_owner_only(tmp_path) -> None:
    key_file = tmp_path / "reasoning-capsule-keys.json"

    load_reasoning_capsule_codec(environ={}, key_file=key_file)

    assert key_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX permission bits are not applicable on this platform",
)
def test_existing_fallback_with_group_or_other_access_is_rejected(tmp_path) -> None:
    key_file = tmp_path / "reasoning-capsule-keys.json"
    key_file.write_text(
        json.dumps({"version": 1, "current_key": _encoded_key(20), "previous_keys": []}),
        encoding="utf-8",
    )
    key_file.chmod(0o640)

    with pytest.raises(ReasoningCapsuleKeyError):
        load_reasoning_capsule_codec(environ={}, key_file=key_file)


def test_concurrent_first_creation_converges_on_one_key(tmp_path) -> None:
    key_file = tmp_path / "reasoning-capsule-keys.json"
    worker_count = 12
    barrier = threading.Barrier(worker_count)

    def load_after_barrier() -> ReasoningCapsuleCodec:
        barrier.wait()
        return load_reasoning_capsule_codec(environ={}, key_file=key_file)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        codecs = list(executor.map(lambda _: load_after_barrier(), range(worker_count)))

    assert len({codec.current_key_id for codec in codecs}) == 1
    capsule = codecs[0].seal(_payload())
    assert all(_unseal(codec, capsule) == _payload() for codec in codecs)
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        json.dumps({"version": 2, "current_key": _encoded_key(18), "previous_keys": []}),
        json.dumps({"version": 1, "current_key": "invalid", "previous_keys": []}),
        json.dumps({"version": 1, "current_key": _encoded_key(18), "previous_keys": "bad"}),
    ],
)
def test_invalid_fallback_file_fails_closed_without_replacement(contents, tmp_path) -> None:
    key_file = tmp_path / "reasoning-capsule-keys.json"
    key_file.write_text(contents, encoding="utf-8")
    if os.name != "nt":
        key_file.chmod(0o600)
    original = key_file.read_bytes()

    with pytest.raises(ReasoningCapsuleKeyError) as caught:
        load_reasoning_capsule_codec(environ={}, key_file=key_file)

    assert str(caught.value) == "Invalid reasoning capsule key configuration"
    assert key_file.read_bytes() == original


def test_capsule_failures_do_not_expose_ciphertext_or_opaque_state() -> None:
    codec = ReasoningCapsuleCodec(bytes([19]) * 32)
    opaque_state = "provider-secret-opaque-state"
    payload = ReasoningCapsulePayload(
        provider="github-copilot",
        model="gpt-5.4",
        transport="responses",
        item_id="rs_secret",
        opaque_state=opaque_state,
    )
    capsule = codec.seal(payload)

    with pytest.raises(ReasoningCapsuleError) as caught:
        codec.unseal(
            _tamper(capsule),
            expected_provider=payload.provider,
            expected_model=payload.model,
            expected_transport=payload.transport,
        )

    rendered = "".join(traceback.format_exception(caught.value))
    assert capsule not in rendered
    assert opaque_state not in rendered
