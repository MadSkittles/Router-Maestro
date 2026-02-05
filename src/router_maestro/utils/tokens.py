"""Token counting utilities using tiktoken."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import tiktoken

if TYPE_CHECKING:
    from router_maestro.server.schemas.anthropic import (
        AnthropicMessage,
        AnthropicTextBlock,
        AnthropicTool,
    )

# Default encoding for Claude models (cl100k_base is compatible)
DEFAULT_ENCODING = "cl100k_base"

# Per-message overhead for role markers, separators, and special tokens
MESSAGE_OVERHEAD_TOKENS = 4

AnthropicStopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]


# =============================================================================
# Tiktoken Encoder (cached for performance)
# =============================================================================


@lru_cache(maxsize=4)
def get_encoding(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Get a tiktoken encoding, cached for performance.

    Args:
        encoding_name: Name of the encoding (default: cl100k_base)

    Returns:
        Tiktoken encoding object
    """
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding name

    Returns:
        Exact token count
    """
    if not text:
        return 0
    encoding = get_encoding(encoding_name)
    return len(encoding.encode(text))


# =============================================================================
# Legacy estimation functions (kept for backward compatibility)
# =============================================================================

# Approximate characters per token (legacy, kept for backward compatibility)
CHARS_PER_TOKEN = 3

# Structure overhead multiplier (legacy, no longer used with tiktoken)
STRUCTURE_OVERHEAD_MULTIPLIER = 1.25


def estimate_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    This function now uses tiktoken for accurate counting instead of
    character-based estimation.

    Args:
        text: The text to count tokens for

    Returns:
        Token count
    """
    return count_tokens(text)


def estimate_tokens_from_char_count(char_count: int) -> int:
    """Estimate token count from character count.

    .. deprecated::
        This function uses inaccurate character-based estimation.
        Use count_tokens() with actual text for accurate results.

    Args:
        char_count: Number of characters

    Returns:
        Estimated token count
    """
    return char_count // CHARS_PER_TOKEN


# =============================================================================
# Content extraction helpers
# =============================================================================


def _extract_system_text(system: str | list[AnthropicTextBlock] | None) -> str:
    """Extract text from system prompt.

    Args:
        system: System prompt as string or list of text blocks

    Returns:
        Combined text content
    """
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    # List of AnthropicTextBlock
    return "\n".join(block.text for block in system)


def _extract_message_text(message: AnthropicMessage | dict) -> str:
    """Extract text from a single message.

    Handles text blocks, thinking blocks, tool_use blocks, and tool_result blocks.

    Args:
        message: An Anthropic user or assistant message

    Returns:
        Combined text content
    """
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""

    parts = []
    for block in content:
        if isinstance(block, dict):
            block_text = block.get("text")
            if block_text:
                parts.append(block_text)
            block_thinking = block.get("thinking")
            if block_thinking:
                parts.append(block_thinking)
            block_name = block.get("name")
            if block_name:
                parts.append(block_name)
            block_input = block.get("input")
            if block_input:
                try:
                    parts.append(json.dumps(block_input))
                except Exception:
                    pass
            if "tool_use_id" in block:
                tool_content = block.get("content")
                if isinstance(tool_content, str):
                    parts.append(tool_content)
                elif isinstance(tool_content, list):
                    for tool_block in tool_content:
                        if isinstance(tool_block, dict):
                            tool_text = tool_block.get("text")
                            if tool_text:
                                parts.append(tool_text)
                        elif hasattr(tool_block, "text"):
                            tool_block_text = getattr(tool_block, "text", None)
                            if tool_block_text:
                                parts.append(tool_block_text)
            continue
        # Text blocks
        block_text = getattr(block, "text", None)
        if block_text:
            parts.append(block_text)
        # Thinking blocks
        block_thinking = getattr(block, "thinking", None)
        if block_thinking:
            parts.append(block_thinking)
        # Tool use blocks
        block_name = getattr(block, "name", None)
        if block_name:
            parts.append(block_name)
        block_input = getattr(block, "input", None)
        if block_input:
            try:
                parts.append(json.dumps(block_input))
            except Exception:
                pass
        # Tool result blocks - handle content field
        if hasattr(block, "tool_use_id"):
            tool_content = getattr(block, "content", None)
            if isinstance(tool_content, str):
                parts.append(tool_content)
            elif isinstance(tool_content, list):
                for tc in tool_content:
                    tc_text = getattr(tc, "text", None)
                    if tc_text:
                        parts.append(tc_text)
    return "\n".join(parts)


def _extract_tools_text(tools: list[AnthropicTool | dict] | None) -> str:
    """Extract text from tool definitions.

    Args:
        tools: List of tool definitions

    Returns:
        Combined text content
    """
    if not tools:
        return ""
    parts = []
    for tool in tools:
        if isinstance(tool, dict):
            tool_name = tool.get("name")
            if tool_name:
                parts.append(tool_name)
            tool_description = tool.get("description")
            if tool_description:
                parts.append(tool_description)
            tool_input_schema = tool.get("input_schema")
        else:
            parts.append(tool.name)
            if tool.description:
                parts.append(tool.description)
            tool_input_schema = tool.input_schema
        if tool_input_schema:
            try:
                parts.append(json.dumps(tool_input_schema))
            except Exception:
                pass
    return "\n".join(parts)


# =============================================================================
# Main token counting function
# =============================================================================


def count_anthropic_request_tokens(
    system: str | list | None,
    messages: list,
    tools: list | None = None,
    model: str | None = None,
) -> int:
    """Count tokens for an Anthropic-style request using tiktoken.

    This function provides accurate token counting by using tiktoken
    (cl100k_base encoding) which is compatible with Claude models.

    Args:
        system: System prompt (string or list of text blocks)
        messages: List of AnthropicMessage objects
        tools: List of AnthropicTool objects
        model: Model name/ID (currently unused, reserved for future use)

    Returns:
        Token count for the request
    """
    total_tokens = 0

    # Count system prompt tokens
    system_text = _extract_system_text(system)
    if system_text:
        total_tokens += count_tokens(system_text)

    # Count message tokens
    for msg in messages:
        message_text = _extract_message_text(msg)
        if message_text:
            total_tokens += count_tokens(message_text)
        # Add per-message overhead for role markers and separators
        total_tokens += MESSAGE_OVERHEAD_TOKENS

    # Count tool definition tokens
    tools_text = _extract_tools_text(tools)
    if tools_text:
        total_tokens += count_tokens(tools_text)

    return total_tokens


# Alias for backward compatibility
def estimate_anthropic_request_tokens(
    system: str | list | None,
    messages: list,
    tools: list | None = None,
    model: str | None = None,
) -> int:
    """Count tokens for an Anthropic-style request.

    This is an alias for count_anthropic_request_tokens() for backward
    compatibility. The function now uses tiktoken for accurate counting.

    Args:
        system: System prompt (string or list of text blocks)
        messages: List of AnthropicMessage objects
        tools: List of AnthropicTool objects
        model: Model name/ID (currently unused, reserved for future use)

    Returns:
        Token count for the request
    """
    return count_anthropic_request_tokens(system, messages, tools, model)


# =============================================================================
# Legacy calibration (no longer needed with tiktoken, kept for reference)
# =============================================================================

# Note: The calibration logic has been removed since tiktoken provides
# accurate token counts. The calibration was only needed to correct
# the character-based estimation which had significant errors.


def calibrate_tokens(
    base_tokens: int,
    is_input: bool = True,
    model: str | None = None,
) -> int:
    """Legacy calibration function (now a no-op).

    With tiktoken providing accurate token counts, calibration is no longer
    needed. This function is kept for backward compatibility and simply
    returns the input value.

    Args:
        base_tokens: Token count
        is_input: Ignored (was for input vs output calibration)
        model: Ignored (was for model-specific calibration)

    Returns:
        Same token count (no calibration applied)
    """
    return base_tokens


# =============================================================================
# Stop reason mapping
# =============================================================================


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
