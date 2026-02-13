"""Context window budget calculations.

Implements Copilot Chat's formula for calculating effective output token
limits and usable prompt space based on model capabilities.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    """Computed context budget for a model."""

    max_prompt_tokens: int
    max_output_tokens: int
    context_window: int


def calculate_context_budget(
    max_prompt_tokens: int | None,
    max_output_tokens: int | None,
    max_context_window_tokens: int | None,
) -> ContextBudget | None:
    """Calculate context budget using Copilot Chat's formula.

    The effective output is capped at 15% of max_prompt_tokens (matching
    Copilot Chat's ``chatEndpoint.ts`` logic), then the usable prompt
    space is derived by subtracting that from the context window.

    Returns None if max_prompt_tokens is unknown.
    """
    if max_prompt_tokens is None:
        return None

    effective_output = min(
        max_output_tokens or 4096,
        int(max_prompt_tokens * 0.15),
    )
    context_window = max_context_window_tokens or (effective_output + max_prompt_tokens)
    usable_prompt = max(0, min(max_prompt_tokens, context_window - effective_output))

    return ContextBudget(
        max_prompt_tokens=usable_prompt,
        max_output_tokens=effective_output,
        context_window=context_window,
    )


def normalize_thinking_budget(
    budget: int | None,
    max_output_tokens: int,
    min_budget: int = 1024,
    max_budget: int = 32000,
) -> int | None:
    """Clamp thinking budget per Copilot Chat constraints.

    Returns None if budget is None (thinking not requested).
    """
    if budget is None:
        return None
    if max_output_tokens <= 1:
        return min_budget
    return max(min_budget, min(budget, min(max_budget, max_output_tokens - 1)))
