"""Reasoning effort ↔ thinking budget mapping.

OpenAI-style ``reasoning_effort`` (``"low"``/``"medium"``/``"high"``) and
Anthropic-style ``thinking.budget_tokens`` (integer) are normalised through
this module so that every entry-route and every provider speaks the same
language.

``"xhigh"`` is a Router-Maestro extension above OpenAI's spec — when sent to
an upstream that does not accept it, providers should downgrade to ``"high"``.
"""

from __future__ import annotations

EFFORT_TO_BUDGET: dict[str, int] = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 24000,
}

VALID_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")

# Effort levels that vanilla OpenAI / Copilot upstreams accept directly.
UPSTREAM_NATIVE_EFFORTS: tuple[str, ...] = ("low", "medium", "high")


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
    """Map ``xhigh`` → ``high`` for upstreams that reject the extension."""
    if effort is None:
        return None
    if effort in UPSTREAM_NATIVE_EFFORTS:
        return effort
    if effort == "xhigh":
        return "high"
    return None
