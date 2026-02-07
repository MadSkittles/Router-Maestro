"""Model ID parsing and sorting utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from router_maestro.providers.base import ModelInfo

# Known variant suffixes (order doesn't matter for matching)
_KNOWN_VARIANTS = frozenset({"preview", "latest", "turbo", "mini"})

# Pattern: 8-digit date at the end (YYYYMMDD)
_DATE_SUFFIX_PLAIN = re.compile(r"-(\d{8})$")
# Pattern: YYYY-MM-DD at the end
_DATE_SUFFIX_DASHED = re.compile(r"-(\d{4})-(\d{2})-(\d{2})$")


@dataclass(frozen=True)
class ParsedModelId:
    """Parsed components of a model ID."""

    family: str
    variant: str
    version: int
    raw_id: str


def parse_model_id(model_id: str) -> ParsedModelId:
    """Parse a model ID into its components.

    Extraction order:
    1. Extract date suffix (YYYYMMDD or YYYY-MM-DD) from the end
    2. Extract known variant suffix from the end
    3. Remaining part becomes the family

    Unknown suffixes are kept in the family to avoid misclassification.

    Args:
        model_id: Raw model ID string (e.g. "gpt-4o-mini-2024-07-18")

    Returns:
        ParsedModelId with family, variant, version, and raw_id
    """
    remaining = model_id
    version = 0
    variant = ""

    # Step 1: Extract date suffix
    match = _DATE_SUFFIX_DASHED.search(remaining)
    if match:
        version = int(match.group(1) + match.group(2) + match.group(3))
        remaining = remaining[: match.start()]
    else:
        match = _DATE_SUFFIX_PLAIN.search(remaining)
        if match:
            version = int(match.group(1))
            remaining = remaining[: match.start()]

    # Step 2: Extract known variant suffix
    # Check if the last segment (after the last hyphen) is a known variant
    last_hyphen = remaining.rfind("-")
    if last_hyphen > 0:
        candidate = remaining[last_hyphen + 1 :]
        if candidate in _KNOWN_VARIANTS:
            variant = candidate
            remaining = remaining[:last_hyphen]

    family = remaining

    return ParsedModelId(
        family=family,
        variant=variant,
        version=version,
        raw_id=model_id,
    )


# Splits a string into alternating text and numeric segments for natural sorting.
# Matches integers or dotted version strings like "4.10.2".
_NATURAL_SORT_RE = re.compile(r"(\d+(?:\.\d+)*)")


def _natural_sort_key(
    s: str, descending: bool = False
) -> list[tuple[int, tuple[int, ...], str]]:
    """Split a string into segments for natural (human) sort order.

    Each segment becomes a tuple of (type_flag, version_tuple, text_value):
    - Numeric segments (e.g. "4.10"): (0, (4, 10), "")
      Dotted numbers are split into integer tuples so that "4.10" > "4.2".
    - Text segments: (1, (), text)

    When descending=True, integer components are negated so larger numbers sort first.
    """
    parts = _NATURAL_SORT_RE.split(s)
    key: list[tuple[int, tuple[int, ...], str]] = []
    for part in parts:
        if not part:
            continue
        if _NATURAL_SORT_RE.fullmatch(part):
            components = [int(x) for x in part.split(".")]
            # Pad to 3 components so "5" compares as "5.0.0" against "5.2"
            while len(components) < 3:
                components.append(0)
            ints = tuple(-x if descending else x for x in components)
            key.append((0, ints, ""))
        else:
            key.append((1, (), part))
    return key


_NaturalKey = list[tuple[int, tuple[int, ...], str]]


def _sort_key(
    model: ModelInfo,
) -> tuple[str, _NaturalKey, _NaturalKey, int]:
    """Generate a sort key for a model.

    Sort order: (provider, name_natural_desc, variant_natural, -version)
    - provider: alphabetical grouping
    - name: natural sort on display name, descending on numeric segments (newer first)
      Uses model.name instead of parsed model.id because display names have
      consistently formatted version numbers (e.g. "4.1" vs the ID "41").
    - variant: natural sort ascending (base model "" before variants)
    - -version: descending (newer date-versioned first)
    """
    parsed = parse_model_id(model.id)
    return (
        model.provider,
        _natural_sort_key(model.name, descending=True),
        _natural_sort_key(parsed.variant),
        -parsed.version,
    )


def sort_models(models: list[ModelInfo]) -> list[ModelInfo]:
    """Sort models by provider, family, variant, and version (descending).

    Returns a new list without modifying the input.

    Args:
        models: List of ModelInfo to sort

    Returns:
        New sorted list of ModelInfo
    """
    return sorted(models, key=_sort_key)
