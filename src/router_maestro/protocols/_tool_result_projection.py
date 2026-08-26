"""Lossless tool-result error projection for protocols without an error flag."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast, overload

_MARKER_KEY = "$router_maestro"
_MARKER_TYPE = "tool_result"
_VERSION = 1
_PROJECTION_KEYS = frozenset({_MARKER_KEY, "is_error", "output"})
_MARKER_KEYS = frozenset({"type", "version"})


class ToolResultProjectionError(ValueError):
    """A reserved tool-result projection uses an unsupported version."""


@overload
def project_tool_result_output(output: str, *, is_error: bool) -> str: ...


@overload
def project_tool_result_output(output: object, *, is_error: bool) -> object: ...


def project_tool_result_output(output: object, *, is_error: bool) -> object:
    """Encode error state, escaping literal values that collide with the reserved envelope."""
    if not is_error and _projection_candidate(output) is None:
        return output
    return json.dumps(
        {
            _MARKER_KEY: {"type": _MARKER_TYPE, "version": _VERSION},
            "is_error": is_error,
            "output": output,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def unproject_tool_result_output(output: object) -> tuple[object, bool]:
    """Decode at most one projection layer and return the original output plus error state."""
    projection = _projection_candidate(output)
    if projection is None:
        return output, False
    marker = cast(Mapping[str, object], projection[_MARKER_KEY])
    version = marker["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != _VERSION:
        raise ToolResultProjectionError(f"unsupported tool-result projection version {version!r}")
    return projection["output"], cast(bool, projection["is_error"])


def _projection_candidate(output: object) -> Mapping[str, object] | None:
    if not isinstance(output, str):
        return None
    try:
        decoded = cast(object, json.loads(output))
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping) or set(decoded) != _PROJECTION_KEYS:
        return None
    marker = decoded.get(_MARKER_KEY)
    if (
        not isinstance(marker, Mapping)
        or set(marker) != _MARKER_KEYS
        or marker.get("type") != _MARKER_TYPE
        or not isinstance(decoded.get("is_error"), bool)
    ):
        return None
    return cast(Mapping[str, object], decoded)


__all__ = [
    "ToolResultProjectionError",
    "project_tool_result_output",
    "unproject_tool_result_output",
]
