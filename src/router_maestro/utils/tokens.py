"""Token estimation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
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


# =============================================================================
# Token Calibration (ported from agent-maestro)
# =============================================================================
#
# Linear regression model: calibrated = slope * base_tokens + base_offset
#
# These coefficients were derived from actual API usage data to correct
# the estimation to match real token counts from the Anthropic API.
# =============================================================================


@dataclass
class CalibrationCoefficients:
    """Linear regression coefficients for token calibration."""

    slope: float
    base_offset: int  # Integer baseline adjustment for token count accuracy


@dataclass
class TokenRangeCalibration:
    """Calibration parameters for different token size ranges."""

    small: CalibrationCoefficients  # < 10K tokens
    medium: CalibrationCoefficients  # 10K - 50K tokens
    large: CalibrationCoefficients  # 50K - 100K tokens
    xlarge: CalibrationCoefficients  # >= 100K tokens


@dataclass
class TokenCalibrationConfig:
    """Calibration config for input and output tokens."""

    input: TokenRangeCalibration
    output: TokenRangeCalibration


# Calibration parameters optimized for different token size ranges
# These are based on linear regression from actual API usage data
CALIBRATION_CONFIG: dict[str, TokenCalibrationConfig] = {
    "default": TokenCalibrationConfig(
        input=TokenRangeCalibration(
            small=CalibrationCoefficients(slope=1.065, base_offset=-120),
            medium=CalibrationCoefficients(slope=1.082, base_offset=1300),
            large=CalibrationCoefficients(slope=1.05, base_offset=2000),
            xlarge=CalibrationCoefficients(slope=1.05, base_offset=1500),
        ),
        output=TokenRangeCalibration(
            # Use same parameters for all output token ranges due to high variability
            small=CalibrationCoefficients(slope=0.67, base_offset=170),
            medium=CalibrationCoefficients(slope=0.67, base_offset=170),
            large=CalibrationCoefficients(slope=0.67, base_offset=170),
            xlarge=CalibrationCoefficients(slope=0.67, base_offset=170),
        ),
    ),
    "opus": TokenCalibrationConfig(
        input=TokenRangeCalibration(
            small=CalibrationCoefficients(slope=1.1, base_offset=0),
            medium=CalibrationCoefficients(slope=1.1, base_offset=1500),
            large=CalibrationCoefficients(slope=1.12, base_offset=1500),
            xlarge=CalibrationCoefficients(slope=1.14, base_offset=1500),
        ),
        output=TokenRangeCalibration(
            small=CalibrationCoefficients(slope=1.0, base_offset=150),
            medium=CalibrationCoefficients(slope=1.0, base_offset=150),
            large=CalibrationCoefficients(slope=1.0, base_offset=150),
            xlarge=CalibrationCoefficients(slope=1.0, base_offset=150),
        ),
    ),
}


def _get_calibration_config(model: str | None) -> TokenCalibrationConfig:
    """Get calibration config based on model name.

    Args:
        model: Model name/ID

    Returns:
        Calibration config for the model
    """
    if model and "opus" in model.lower():
        return CALIBRATION_CONFIG["opus"]
    return CALIBRATION_CONFIG["default"]


def _get_calibration_coefficients(
    tokens: int,
    is_input: bool,
    config: TokenCalibrationConfig,
) -> CalibrationCoefficients:
    """Select calibration coefficients based on token count and type.

    Args:
        tokens: Base token count
        is_input: True for input tokens, False for output tokens
        config: Calibration config to use

    Returns:
        Calibration coefficients for the token range
    """
    range_config = config.input if is_input else config.output

    # Thresholds (9K, 45K, 90K) approximate actual API thresholds (10K, 50K, 100K)
    if tokens < 9000:
        return range_config.small
    elif tokens < 45000:
        return range_config.medium
    elif tokens < 90000:
        return range_config.large
    else:
        return range_config.xlarge


def calibrate_tokens(
    base_tokens: int,
    is_input: bool = True,
    model: str | None = None,
) -> int:
    """Calibrate token count to fit actual API usage.

    Uses linear regression coefficients optimized from actual API usage data
    to correct the base estimation to match real token counts.

    Args:
        base_tokens: Base token count from estimation
        is_input: True for input tokens, False for output tokens
        model: Model name/ID to select appropriate calibration

    Returns:
        Calibrated token count matching actual API usage
    """
    if base_tokens <= 0:
        return 0

    config = _get_calibration_config(model)
    coefficients = _get_calibration_coefficients(base_tokens, is_input, config)

    # Apply calibration: calibrated = slope × base + base_offset
    calibrated = coefficients.slope * base_tokens + coefficients.base_offset

    # Ensure we return at least 1 for non-zero input
    return max(1, round(calibrated))


# =============================================================================
# Base Token Estimation Functions
# =============================================================================


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
    model: str | None = None,
) -> int:
    """Estimate total tokens for an Anthropic-style request.

    This is the centralized estimation function that accounts for:
    - System prompt content
    - Message content (text, tool_use, tool_result, thinking blocks)
    - Per-message overhead (role markers, separators)
    - Tool definitions
    - Structure overhead (JSON framing, special tokens)
    - Model-specific calibration (Opus vs other models)

    Args:
        system: System prompt (string or list of text blocks)
        messages: List of AnthropicMessage objects
        tools: List of AnthropicTool objects
        model: Model name/ID for calibration selection

    Returns:
        Calibrated token count matching actual API usage
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
    base_tokens = int(base_tokens * STRUCTURE_OVERHEAD_MULTIPLIER)

    # Apply model-specific calibration
    return calibrate_tokens(base_tokens, is_input=True, model=model)


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
