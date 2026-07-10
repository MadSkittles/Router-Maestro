"""Reasoning effort ↔ thinking budget mapping.

OpenAI-style ``reasoning_effort`` (``"low"``/``"medium"``/``"high"``) and
Anthropic-style ``thinking.budget_tokens`` (integer) are normalised through
this module so that every entry-route and every provider speaks the same
language.

``"xhigh"`` and ``"max"`` are Router-Maestro extensions above OpenAI's spec —
when sent to an upstream that does not accept them, providers downgrade to
the highest tier the upstream supports.
"""

from __future__ import annotations

EFFORT_TO_BUDGET: dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}

VALID_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Effort tiers eligible for transparent variant rewriting (e.g. claude-opus-4.7
# → claude-opus-4.7-high). low/medium are passed through as reasoning_effort.
VARIANT_EFFORTS: tuple[str, ...] = ("high", "xhigh", "max")

# Effort levels that vanilla OpenAI / Copilot upstreams accept directly.
UPSTREAM_NATIVE_EFFORTS: tuple[str, ...] = ("low", "medium", "high")


def pick_closest_effort(desired: str, allowed: list[str]) -> str | None:
    """Pick the closest catalog-supported effort, preferring the next higher tier."""
    if not allowed:
        return None
    if desired in allowed:
        return desired
    try:
        target = VALID_EFFORTS.index(desired)
    except ValueError:
        return max(
            allowed,
            key=lambda value: VALID_EFFORTS.index(value) if value in VALID_EFFORTS else -1,
        )
    higher = [
        value for value in allowed if value in VALID_EFFORTS and VALID_EFFORTS.index(value) > target
    ]
    if higher:
        return min(higher, key=VALID_EFFORTS.index)
    lower = [
        value for value in allowed if value in VALID_EFFORTS and VALID_EFFORTS.index(value) < target
    ]
    if lower:
        return max(lower, key=VALID_EFFORTS.index)
    return allowed[0]


def effort_to_budget(effort: str | None) -> int | None:
    if effort is None:
        return None
    return EFFORT_TO_BUDGET.get(effort.lower())


def budget_to_effort(budget: int | None) -> str | None:
    """Approximate inverse mapping.

    Picks the highest defined effort whose budget is ≤ the requested one.
    Returns ``None`` when ``budget`` is ``None`` or below the smallest tier.
    """
    if budget is None:
        return None
    best: str | None = None
    best_val = -1
    for name, val in EFFORT_TO_BUDGET.items():
        if val <= budget and val > best_val:
            best, best_val = name, val
    return best


def downgrade_for_upstream(effort: str | None) -> str | None:
    """Map ``xhigh``/``max`` → ``high`` for upstreams that reject extensions."""
    if effort is None:
        return None
    if effort in UPSTREAM_NATIVE_EFFORTS:
        return effort
    if effort in ("xhigh", "max"):
        return "high"
    return None
