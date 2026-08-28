"""Provider-independent strict helpers for concrete wire runtimes."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Never

from router_maestro.protocols.models import FrozenJsonValue, WireProtocol
from router_maestro.protocols.runtime import (
    ProtocolDecodeError,
    ProtocolRepresentabilityError,
)

_REASONING_CAPSULE_CARRIER = re.compile(r"^rmr[0-9]+\.")


def is_reasoning_capsule_carrier(value: object) -> bool:
    """Return whether a wire value uses Router-Maestro's reserved capsule prefix."""
    return isinstance(value, str) and _REASONING_CAPSULE_CARRIER.match(value) is not None


def reject(protocol: WireProtocol, parameter: str, reason: str) -> Never:
    raise ProtocolRepresentabilityError(protocol, parameter, reason)


def decode_reject(protocol: WireProtocol, parameter: str, reason: str) -> Never:
    raise ProtocolDecodeError(protocol, parameter, reason)


def require_mapping(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        decode_reject(protocol, parameter, "must be an object")
    for key in value:
        if not isinstance(key, str):
            decode_reject(protocol, parameter, "object keys must be strings")
    return value


def require_list(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        decode_reject(protocol, parameter, "must be an array")
    return value


def require_string(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
    allow_empty: bool = True,
) -> str:
    if not isinstance(value, str):
        decode_reject(protocol, parameter, "must be a string")
    if not allow_empty and not value:
        decode_reject(protocol, parameter, "must be a non-empty string")
    return value


def optional_string(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> str | None:
    if value is None:
        return None
    return require_string(value, protocol=protocol, parameter=parameter)


def optional_bool(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        decode_reject(protocol, parameter, "must be a boolean")
    return value


def optional_int(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        decode_reject(protocol, parameter, "must be an integer")
    return value


def optional_number(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> int | float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        decode_reject(protocol, parameter, "must be a number")
    return value


def reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    protocol: WireProtocol,
    parameter: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        suffix = f".{unknown[0]}" if parameter else unknown[0]
        decode_reject(protocol, f"{parameter}{suffix}", "field is not representable")


def thaw_json(value: FrozenJsonValue | object) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [thaw_json(item) for item in value]
    return value


def has_typed_block(value: object, block_types: set[str]) -> bool:
    if isinstance(value, Mapping):
        block_type = value.get("type")
        if isinstance(block_type, str) and block_type in block_types:
            return True
        return any(has_typed_block(item, block_types) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(has_typed_block(item, block_types) for item in value)
    return False


__all__ = [
    "decode_reject",
    "has_typed_block",
    "is_reasoning_capsule_carrier",
    "optional_bool",
    "optional_int",
    "optional_number",
    "optional_string",
    "reject",
    "reject_unknown_keys",
    "require_list",
    "require_mapping",
    "require_string",
    "thaw_json",
]
