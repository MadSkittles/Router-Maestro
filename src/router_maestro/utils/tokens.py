"""Token estimation utilities."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from router_maestro.server.schemas.anthropic import (
        AnthropicMessage,
        AnthropicTextBlock,
        AnthropicTool,
    )

# Approximate characters per token for English text
# Using 3 instead of 4 for more conservative estimation
CHARS_PER_TOKEN = 3

# Per-message overhead for role markers, separators, and special tokens
MESSAGE_OVERHEAD_TOKENS = 4

# Structure overhead multiplier for JSON framing, special tokens, etc.
STRUCTURE_OVERHEAD_MULTIPLIER = 1.25

AnthropicStopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses a rough approximation of ~3 characters per token for English text.
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


def _count_system_chars(system: str | list[AnthropicTextBlock] | None) -> int:
    """Count characters in system prompt.

    Args:
        system: System prompt as string or list of text blocks

    Returns:
        Total character count
    """
    if system is None:
        return 0
    if isinstance(system, str):
        return len(system)
    # List of AnthropicTextBlock
    return sum(len(block.text) for block in system)


def _count_message_chars(message: AnthropicMessage) -> int:
    """Count characters in a single message.

    Handles text blocks, thinking blocks, tool_use blocks, and tool_result blocks.

    Args:
        message: An Anthropic user or assistant message

    Returns:
        Total character count
    """
    content = message.content
    if isinstance(content, str):
        return len(content)

    total = 0
    for block in content:
        # Text blocks
        if hasattr(block, "text") and block.text:
            total += len(block.text)
        # Thinking blocks
        if hasattr(block, "thinking") and block.thinking:
            total += len(block.thinking)
        # Tool use blocks
        if hasattr(block, "name") and block.name:
            total += len(block.name)
        if hasattr(block, "input") and block.input:
            try:
                total += len(json.dumps(block.input))
            except Exception:
                pass
        # Tool result blocks - handle content field
        if hasattr(block, "content") and hasattr(block, "tool_use_id"):
            tool_content = block.content
            if isinstance(tool_content, str):
                total += len(tool_content)
            elif isinstance(tool_content, list):
                for tc in tool_content:
                    if hasattr(tc, "text") and tc.text:
                        total += len(tc.text)
    return total


def _count_tools_chars(tools: list[AnthropicTool] | None) -> int:
    """Count characters in tool definitions.

    Args:
        tools: List of tool definitions

    Returns:
        Total character count
    """
    if not tools:
        return 0
    total = 0
    for tool in tools:
        total += len(tool.name)
        if tool.description:
            total += len(tool.description)
        try:
            total += len(json.dumps(tool.input_schema))
        except Exception:
            pass
    return total


def estimate_anthropic_request_tokens(
    system: str | list | None,
    messages: list,
    tools: list | None = None,
) -> int:
    """Estimate total tokens for an Anthropic-style request.

    This is the centralized estimation function that accounts for:
    - System prompt content
    - Message content (text, tool_use, tool_result, thinking blocks)
    - Per-message overhead (role markers, separators)
    - Tool definitions
    - Structure overhead (JSON framing, special tokens)

    Args:
        system: System prompt (string or list of text blocks)
        messages: List of AnthropicMessage objects
        tools: List of AnthropicTool objects

    Returns:
        Estimated token count with overhead applied
    """
    total_chars = 0

    # Count system prompt
    total_chars += _count_system_chars(system)

    # Count messages
    message_count = len(messages)
    for msg in messages:
        total_chars += _count_message_chars(msg)

    # Count tools
    total_chars += _count_tools_chars(tools)

    # Calculate base tokens
    base_tokens = total_chars // CHARS_PER_TOKEN

    # Add per-message overhead
    base_tokens += message_count * MESSAGE_OVERHEAD_TOKENS

    # Apply structure overhead multiplier
    return int(base_tokens * STRUCTURE_OVERHEAD_MULTIPLIER)


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
