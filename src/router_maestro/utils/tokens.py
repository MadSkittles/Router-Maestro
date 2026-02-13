"""Token counting utilities using tiktoken.

Aligned with VS Code Copilot Chat's token counting approach for accurate
estimation compatible with OpenAI/GitHub Copilot token limits.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import tiktoken

from router_maestro.utils.token_config import DEFAULT_CONFIG, TokenCountingConfig

if TYPE_CHECKING:
    from router_maestro.server.schemas.anthropic import AnthropicTextBlock

# =============================================================================
# Constants (aligned with VS Code Copilot Chat BPETokenizer)
# =============================================================================

DEFAULT_ENCODING = "cl100k_base"

# Per-message overhead: special tokens like <|im_start|>role<|im_sep|>
TOKENS_PER_MESSAGE = 3

# Extra token cost when a message has a 'name' field
TOKENS_PER_NAME = 1

# Base tokens for the assistant reply priming (<|im_start|>assistant<|message|>)
TOKENS_PER_COMPLETION = 3

# Base overhead when any tools are present in the request
BASE_TOOL_TOKENS = 16

# Per-tool overhead for each tool definition
TOKENS_PER_TOOL = 8

# Safety multiplier for tool definition token counts
TOOL_DEFINITION_MULTIPLIER = 1.1

# Safety multiplier for tool_calls content blocks
TOOL_CALLS_MULTIPLIER = 1.5

# Regex pattern matching O-series models that use o200k_base encoding
_O_SERIES_PATTERN = re.compile(r"\bo[1-9]\b|\bo[1-9]-|\bo[1-9][0-9]*-")

# Legacy constants (kept for backward compatibility)
CHARS_PER_TOKEN = 3

AnthropicStopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]


# =============================================================================
# Tiktoken Encoder (cached for performance)
# =============================================================================


@lru_cache(maxsize=4)
def get_encoding(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Get a tiktoken encoding, cached for performance."""
    return tiktoken.get_encoding(encoding_name)


def _select_encoding(model: str | None) -> str:
    """Select the appropriate tiktoken encoding based on the model.

    O-series models (o1, o3, o4-mini, etc.) use o200k_base.
    All other models (Claude, GPT-4, etc.) use cl100k_base.
    """
    if model and _O_SERIES_PATTERN.search(model):
        return "o200k_base"
    return DEFAULT_ENCODING


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    encoding = get_encoding(encoding_name)
    return len(encoding.encode(text))


# =============================================================================
# Legacy estimation functions (kept for backward compatibility)
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (legacy alias)."""
    return count_tokens(text)


def estimate_tokens_from_char_count(char_count: int) -> int:
    """Estimate token count from character count (deprecated)."""
    return char_count // CHARS_PER_TOKEN


# =============================================================================
# Image token calculation
# =============================================================================


def calculate_image_token_cost(width: int, height: int, detail: str | None = None) -> int:
    """Calculate token cost for an image based on its dimensions.

    Follows the OpenAI vision token calculation:
    - low detail: fixed 85 tokens
    - high detail: scale to fit 2048x2048, then scale shortest side to 768,
      then count 512x512 tiles at 170 tokens each + 85 base
    """
    if detail == "low":
        return 85

    # Scale to fit within 2048x2048
    if width > 2048 or height > 2048:
        scale_factor = 2048 / max(width, height)
        width = round(width * scale_factor)
        height = round(height * scale_factor)

    # Scale so the shortest side is at least 768
    if min(width, height) > 0:
        scale_factor = 768 / min(width, height)
        width = round(width * scale_factor)
        height = round(height * scale_factor)

    # Count 512x512 tiles
    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return tiles * 170 + 85


# =============================================================================
# Recursive token counting for message objects
# =============================================================================


def _count_object_tokens(obj: Any, encoding_name: str) -> int:
    """Recursively count tokens for an object, including keys.

    Used for tool definitions where both keys and values are tokenized.
    """
    if obj is None:
        return 0

    if isinstance(obj, str):
        return count_tokens(obj, encoding_name)

    if isinstance(obj, bool):
        return 1

    if isinstance(obj, (int, float)):
        return count_tokens(str(obj), encoding_name)

    if isinstance(obj, list):
        total = 0
        for item in obj:
            total += _count_object_tokens(item, encoding_name)
        return total

    if isinstance(obj, dict):
        total = 0
        for key, value in obj.items():
            if value is None:
                continue
            total += count_tokens(key, encoding_name)
            total += _count_object_tokens(value, encoding_name)
        return total

    # Pydantic models or other objects with attributes
    if hasattr(obj, "__dict__"):
        return _count_object_tokens(
            {k: v for k, v in obj.__dict__.items() if not k.startswith("_")},
            encoding_name,
        )

    return count_tokens(str(obj), encoding_name)


def _count_message_object_tokens(
    obj: Any,
    encoding_name: str,
    config: TokenCountingConfig | None = None,
) -> int:
    """Recursively count tokens for a message object's values.

    Unlike _count_object_tokens, this does NOT tokenize the keys themselves
    (matching VS Code Copilot Chat's countMessageObjectTokens behavior).
    Special handling:
    - 'name' key adds config.tokens_per_name extra token
    - 'tool_calls' key values are multiplied by config.tool_calls_multiplier
    - image_url content is calculated based on image dimensions
    """
    cfg = config if config is not None else DEFAULT_CONFIG

    if obj is None:
        return 0

    if isinstance(obj, str):
        return count_tokens(obj, encoding_name)

    if isinstance(obj, bool):
        return 1

    if isinstance(obj, (int, float)):
        return count_tokens(str(obj), encoding_name)

    if isinstance(obj, list):
        total = 0
        for item in obj:
            total += _count_message_object_tokens(item, encoding_name, cfg)
        return total

    if isinstance(obj, dict):
        total = 0
        for key, value in obj.items():
            if value is None:
                continue

            # Handle image_url type content blocks
            if key == "image_url" and isinstance(value, dict):
                detail = value.get("detail", "auto")
                # Without actual image dimensions, use a conservative default
                # In practice, the API would provide dimensions or a URL
                if detail == "low":
                    total += 85
                else:
                    # Default high-detail cost for unknown dimensions
                    total += 765  # ~4 tiles (2x2) * 170 + 85
                continue

            new_tokens = _count_message_object_tokens(value, encoding_name, cfg)

            # tool_calls get a safety multiplier
            if key == "tool_calls":
                new_tokens = int(new_tokens * cfg.tool_calls_multiplier)

            total += new_tokens

            # name fields cost an extra token
            if key == "name" and value is not None:
                total += cfg.tokens_per_name

        return total

    # Pydantic models or other objects with attributes
    if hasattr(obj, "__dict__"):
        return _count_message_object_tokens(
            {k: v for k, v in obj.__dict__.items() if not k.startswith("_")},
            encoding_name,
            cfg,
        )

    return count_tokens(str(obj), encoding_name)


# =============================================================================
# Content extraction and conversion helpers
# =============================================================================


def _message_to_dict(message: Any) -> dict:
    """Convert a message (dict or Pydantic model) to a plain dict."""
    if isinstance(message, dict):
        return message

    result: dict[str, Any] = {}
    if hasattr(message, "role"):
        result["role"] = message.role

    content = getattr(message, "content", None)
    if content is not None:
        if isinstance(content, str):
            result["content"] = content
        elif isinstance(content, list):
            result["content"] = [_block_to_dict(b) for b in content]
        else:
            result["content"] = content

    # Copy other relevant fields
    for field in ("name", "tool_calls"):
        val = getattr(message, field, None)
        if val is not None:
            result[field] = val

    return result


def _block_to_dict(block: Any) -> dict:
    """Convert a content block (dict or Pydantic model) to a plain dict."""
    if isinstance(block, dict):
        return block

    result: dict[str, Any] = {}
    for attr in (
        "type",
        "text",
        "thinking",
        "id",
        "name",
        "input",
        "tool_use_id",
        "content",
        "image_url",
    ):
        val = getattr(block, attr, None)
        if val is not None:
            if attr == "content" and isinstance(val, list):
                result[attr] = [_block_to_dict(b) for b in val]
            elif attr == "input" and isinstance(val, dict):
                result[attr] = val
            else:
                result[attr] = val
    return result


def _extract_system_text(system: str | list[AnthropicTextBlock] | None) -> str:
    """Extract text from system prompt."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "\n".join(block.text if hasattr(block, "text") else str(block) for block in system)


def _tool_to_dict(tool: Any) -> dict:
    """Convert a tool definition (dict or Pydantic model) to a plain dict."""
    if isinstance(tool, dict):
        return tool
    result: dict[str, Any] = {}
    for attr in ("name", "description", "input_schema"):
        val = getattr(tool, attr, None)
        if val is not None:
            result[attr] = val
    return result


# =============================================================================
# Per-message token counting with tool_use multiplier
# =============================================================================


def _count_message_tokens(
    msg_dict: dict,
    encoding_name: str,
    config: TokenCountingConfig | None = None,
) -> int:
    """Count tokens for a single message dict.

    Applies the config.tool_calls_multiplier to tool_use blocks in Anthropic format
    (where they appear as content blocks) and to tool_calls in OpenAI format.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    total = 0

    for key, value in msg_dict.items():
        if value is None:
            continue

        # OpenAI-format tool_calls at the message level
        if key == "tool_calls":
            tc_tokens = _count_message_object_tokens(value, encoding_name, cfg)
            total += int(tc_tokens * cfg.tool_calls_multiplier)
            continue

        # Anthropic-format: content is a list that may contain tool_use blocks
        if key == "content" and isinstance(value, list):
            for block in value:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    # Apply multiplier to tool_use blocks
                    block_tokens = _count_message_object_tokens(block, encoding_name, cfg)
                    total += int(block_tokens * cfg.tool_calls_multiplier)
                else:
                    total += _count_message_object_tokens(block, encoding_name, cfg)
            continue

        new_tokens = _count_message_object_tokens(value, encoding_name, cfg)
        total += new_tokens

        if key == "name" and value is not None:
            total += cfg.tokens_per_name

    return total


# =============================================================================
# Main token counting function
# =============================================================================


def count_anthropic_request_tokens(
    system: str | list | None,
    messages: list,
    tools: list | None = None,
    model: str | None = None,
    config: TokenCountingConfig | None = None,
) -> int:
    """Count tokens for an Anthropic-style request using tiktoken.

    Implements recursive key-value token counting aligned with VS Code
    Copilot Chat's BPETokenizer approach, including:
    - Per-message overhead (config.tokens_per_message, default 3)
    - Name field overhead (config.tokens_per_name, default 1)
    - Completion priming overhead (config.tokens_per_completion, default 3)
    - Tool calls multiplier (config.tool_calls_multiplier, default 1.5x)
    - Tool definition overhead (config.base_tool_tokens + config.tokens_per_tool
      per tool, scaled by config.tool_definition_multiplier)
    - Model-specific encoding selection (o200k_base for O-series)

    When *config* is ``None``, uses ``DEFAULT_CONFIG`` (Copilot-aligned) for
    full backward compatibility.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    encoding_name = _select_encoding(model)
    total_tokens = 0

    # Outer base: one tokens_per_message for the overall message sequence
    if messages:
        total_tokens += cfg.tokens_per_message

    # Count system prompt tokens
    system_text = _extract_system_text(system)
    if system_text:
        total_tokens += count_tokens(system_text, encoding_name)

    # Count message tokens
    for msg in messages:
        msg_dict = _message_to_dict(msg)
        total_tokens += cfg.tokens_per_message
        total_tokens += _count_message_tokens(msg_dict, encoding_name, cfg)

    # Completion priming overhead
    if messages:
        total_tokens += cfg.tokens_per_completion

    # Count tool definition tokens
    if tools:
        tool_tokens = cfg.base_tool_tokens
        for tool in tools:
            tool_dict = _tool_to_dict(tool)
            tool_tokens += cfg.tokens_per_tool
            tool_tokens += _count_object_tokens(
                {
                    "name": tool_dict.get("name"),
                    "description": tool_dict.get("description"),
                    "parameters": tool_dict.get("input_schema"),
                },
                encoding_name,
            )
        total_tokens += int(tool_tokens * cfg.tool_definition_multiplier)

    return total_tokens


# =============================================================================
# Stop reason mapping
# =============================================================================


def map_openai_stop_reason_to_anthropic(
    openai_reason: str | None,
) -> AnthropicStopReason | None:
    """Map OpenAI finish reason to Anthropic stop reason."""
    if openai_reason is None:
        return None
    mapping: dict[str, AnthropicStopReason] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_reason, "end_turn")
