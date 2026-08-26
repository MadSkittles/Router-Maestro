"""Reversible namespaced-tool projection for function-only transports."""

from __future__ import annotations

import base64
import re

_COMPACT_PREFIX = "rmn2_"
_LEGACY_PREFIX = "rmns_"
_MAX_FUNCTION_NAME_LENGTH = 64
_PORTABLE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")


def encode_namespaced_tool_name(namespace: str, name: str) -> str:
    """Encode one namespace/name pair as a portable function identifier."""
    if not namespace or not name:
        raise ValueError("tool namespace and name must be non-empty")

    if _PORTABLE_IDENTIFIER.fullmatch(namespace) and _PORTABLE_IDENTIFIER.fullmatch(name):
        if len(namespace) > 0xFF:
            raise ValueError("tool namespace is too long")
        encoded = f"{_COMPACT_PREFIX}{len(namespace):02x}_{namespace}{name}"
        if len(encoded) > _MAX_FUNCTION_NAME_LENGTH:
            raise ValueError("namespaced tool name exceeds the function-name limit")
        return encoded

    namespace_token = _encode(namespace)
    name_token = _encode(name)
    if len(namespace_token) > 0xFF:
        raise ValueError("tool namespace is too long")
    encoded = f"{_LEGACY_PREFIX}{len(namespace_token):02x}_{namespace_token}{name_token}"
    if len(encoded) > _MAX_FUNCTION_NAME_LENGTH:
        raise ValueError("namespaced tool name exceeds the function-name limit")
    return encoded


def decode_namespaced_tool_name(value: str) -> tuple[str, str] | None:
    """Restore an identifier created by :func:`encode_namespaced_tool_name`."""
    compact = _split_payload(value, _COMPACT_PREFIX)
    if compact is not None:
        namespace, name = compact
        if _PORTABLE_IDENTIFIER.fullmatch(namespace) and _PORTABLE_IDENTIFIER.fullmatch(name):
            return namespace, name
        return None

    legacy = _split_payload(value, _LEGACY_PREFIX)
    if legacy is None:
        return None
    namespace_token, name_token = legacy
    try:
        namespace = _decode(namespace_token)
        name = _decode(name_token)
    except (UnicodeDecodeError, ValueError):
        return None
    if not namespace or not name:
        return None
    return namespace, name


def _split_payload(value: str, prefix: str) -> tuple[str, str] | None:
    if not value.startswith(prefix) or len(value) < len(prefix) + 4:
        return None
    length_start = len(prefix)
    separator = length_start + 2
    if value[separator] != "_":
        return None
    try:
        namespace_length = int(value[length_start:separator], 16)
    except ValueError:
        return None
    payload = value[separator + 1 :]
    if namespace_length <= 0 or namespace_length >= len(payload):
        return None
    return payload[:namespace_length], payload[namespace_length:]


def _encode(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(f"{value}{padding}", altchars=b"-_", validate=True).decode()


__all__ = ["decode_namespaced_tool_name", "encode_namespaced_tool_name"]
