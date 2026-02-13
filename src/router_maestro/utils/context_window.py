"""Context window budget calculations.

Implements Copilot Chat's formula for calculating effective output token
limits and usable prompt space based on model capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router_maestro.config.priorities import ThinkingBudgetConfig


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


def resolve_thinking_budget(
    client_budget: int | None,
    client_thinking_type: str | None,
    model_id: str,
    max_output_tokens: int,
    thinking_config: ThinkingBudgetConfig | None = None,
    supports_thinking: bool = False,
) -> tuple[int | None, str | None]:
    """Resolve effective thinking budget.

    Priority:
    1. Client-specified budget → normalize and return
    2. Client explicitly disabled → (None, None)
    3. Per-model config budget → normalize and return
    4. Default budget (if auto_enable) → normalize and return
    5. (None, None) — thinking not enabled

    Returns:
        Tuple of (budget_tokens, thinking_type) or (None, None).
    """
    # 1. Client explicitly requested thinking with a budget
    if client_thinking_type in ("enabled", "adaptive") and client_budget is not None:
        return normalize_thinking_budget(client_budget, max_output_tokens), client_thinking_type

    # 2. Client explicitly disabled thinking
    if client_thinking_type == "disabled":
        return None, None

    # 3. Client requested thinking but without a budget — use server defaults
    if client_thinking_type in ("enabled", "adaptive"):
        budget = _resolve_server_budget(model_id, thinking_config)
        if budget is not None:
            return normalize_thinking_budget(budget, max_output_tokens), client_thinking_type
        # No server default — pass through client's request without budget
        return None, client_thinking_type

    # 4. Client didn't specify thinking at all — check auto_enable
    if thinking_config is not None and thinking_config.auto_enable and supports_thinking:
        budget = _resolve_server_budget(model_id, thinking_config)
        if budget is not None:
            return normalize_thinking_budget(budget, max_output_tokens), "enabled"

    # 5. No thinking
    return None, None


def _resolve_server_budget(
    model_id: str,
    thinking_config: ThinkingBudgetConfig | None,
) -> int | None:
    """Look up server-side thinking budget for a model."""
    if thinking_config is None:
        return None

    # Check per-model budgets (try full key, then bare model name)
    if model_id in thinking_config.model_budgets:
        return thinking_config.model_budgets[model_id]

    # Strip provider prefix for matching
    bare = model_id.split("/", 1)[1] if "/" in model_id else model_id
    if bare in thinking_config.model_budgets:
        return thinking_config.model_budgets[bare]

    return thinking_config.default_budget
