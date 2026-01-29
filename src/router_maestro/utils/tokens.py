"""Token estimation utilities."""

from typing import Literal

# Approximate characters per token for English text
CHARS_PER_TOKEN = 4

AnthropicStopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses a rough approximation of ~4 characters per token for English text.
    This provides an estimate for context display before actual usage is known.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // CHARS_PER_TOKEN


def estimate_tokens_from_char_count(char_count: int) -> int:
    """Estimate token count from character count.

    Args:
        char_count: Number of characters

    Returns:
        Estimated token count
    """
    return char_count // CHARS_PER_TOKEN


def map_openai_stop_reason_to_anthropic(
    openai_reason: str | None,
) -> AnthropicStopReason | None:
    """Map OpenAI finish reason to Anthropic stop reason.

    Args:
        openai_reason: OpenAI finish reason (stop, length, tool_calls, content_filter)

    Returns:
        Anthropic stop reason (end_turn, max_tokens, tool_use)
    """
    if openai_reason is None:
        return None
    mapping: dict[str, AnthropicStopReason] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_reason, "end_turn")
