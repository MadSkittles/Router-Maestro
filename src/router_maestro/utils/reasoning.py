"""Reasoning effort ↔ thinking budget mapping.

OpenAI-style ``reasoning_effort`` (``"minimal"``/``"low"``/``"medium"``/
``"high"``) and
Anthropic-style ``thinking.budget_tokens`` (integer) are normalised through
this module so that every entry-route and every provider speaks the same
language.

``"xhigh"`` and ``"max"`` are Router-Maestro extensions above OpenAI's spec —
when sent to an upstream that does not accept them, providers downgrade to
the highest tier the upstream supports.
"""

from __future__ import annotations

# ``minimal`` is an explicit upstream tier, but there is no documented token
# budget equivalent. Keep the ordinal request domain separate from this partial
# budget mapping so a small Anthropic budget is never guessed to mean minimal.
EFFORT_TO_BUDGET: dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}

EFFORT_ORDER: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")
VALID_EFFORTS: tuple[str, ...] = EFFORT_ORDER

# Effort levels that vanilla OpenAI / Copilot upstreams accept directly.
UPSTREAM_NATIVE_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")


def pick_closest_effort(desired: str, allowed: list[str]) -> str | None:
    """Pick the highest catalog-supported effort that does not exceed desired."""
    if desired not in EFFORT_ORDER:
        return None
    valid_allowed = [value for value in allowed if value in EFFORT_ORDER]
    if desired in valid_allowed:
        return desired
    target = EFFORT_ORDER.index(desired)
    lower = [value for value in valid_allowed if EFFORT_ORDER.index(value) < target]
    if lower:
        return max(lower, key=EFFORT_ORDER.index)
    return None


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
    """Preserve native tiers and map ``xhigh``/``max`` to ``high``."""
    if effort is None:
        return None
    if effort in UPSTREAM_NATIVE_EFFORTS:
        return effort
    if effort in ("xhigh", "max"):
        return "high"
    return None
