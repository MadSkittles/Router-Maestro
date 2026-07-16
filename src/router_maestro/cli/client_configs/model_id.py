"""Structural model-family detection and per-vendor official-id spelling.

The optional "official model id" feature writes a vendor's native id (e.g.
``gpt-4.1``, ``claude-opus-4-6``) instead of the internal ``provider/upstream``
form. The conversion is derived from naming conventions — NOT a per-model lookup
table — so new models work without code changes. Each client binds the family it
considers native and the spelling function for that vendor.

Vendors disagree on spelling: Anthropic uses dashes in version numbers
(``claude-opus-4.6`` -> ``claude-opus-4-6``), while OpenAI and Google keep dots
(``gpt-4.1`` stays, ``gemini-2.5-pro`` stays). That divergence is exactly why
the conversion is bound per client rather than shared.
"""

from __future__ import annotations

import re
from enum import StrEnum

from router_maestro.utils.model_match import normalize_model_identity


class ModelFamily(StrEnum):
    """Vendor family a model id belongs to, by naming convention."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    UNKNOWN = "unknown"


# OpenAI reasoning "o-series": o1/o3/o4 followed by a boundary (end, '-', or a
# digit like o1 vs a hypothetical o10). Excludes words like "olmo"/"orca".
_O_SERIES = re.compile(r"^o[1-9]\d*(?:-|$)")


def detect_family(bare_id: str) -> ModelFamily:
    """Classify a bare (un-prefixed) model id into its vendor family."""
    s = bare_id.lower()
    if s.startswith("claude"):
        return ModelFamily.ANTHROPIC
    if s.startswith("gpt") or s.startswith("codex") or _O_SERIES.match(s):
        return ModelFamily.OPENAI
    if s.startswith("gemini"):
        return ModelFamily.GOOGLE
    return ModelFamily.UNKNOWN


def to_anthropic_official(bare_id: str) -> str:
    """Anthropic official spelling: lowercase, dots become dashes.

    ``normalize_model_identity`` also preserves date suffixes, which are part of
    Anthropic's official dated ids (``claude-sonnet-4-20250514``).
    """
    return normalize_model_identity(bare_id)


def to_openai_official(bare_id: str) -> str:
    """OpenAI official spelling: the bare id as-is (dots kept)."""
    return bare_id


def to_gemini_official(bare_id: str) -> str:
    """Google/Gemini official spelling: the bare id as-is (dots kept)."""
    return bare_id
