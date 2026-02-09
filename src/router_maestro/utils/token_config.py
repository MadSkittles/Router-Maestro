"""Provider-aware token counting configuration.

Different providers have different tokenization overhead and context window
sizes. The constants in this module parameterize the token counting logic
per provider so that estimates are accurate regardless of which upstream
provider serves the request.

- Copilot config: calibrated to match VS Code Copilot Chat's inflated counts
  (128k context, higher safety multipliers).
- Anthropic config: native Anthropic API with no inflation (200k context).
- OpenAI config: standard OpenAI overhead (no inflated multipliers).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("router_maestro.utils.token_config")


@dataclass(frozen=True)
class TokenCountingConfig:
    """Parameterises the token counting constants by provider.

    Each field corresponds to a constant previously hardcoded in ``tokens.py``.
    Frozen for thread-safety and to prevent accidental mutation.
    """

    # Per-message overhead: special tokens like <|im_start|>role<|im_sep|>
    tokens_per_message: int = 3

    # Extra token cost when a message has a 'name' field
    tokens_per_name: int = 1

    # Base tokens for the assistant reply priming
    tokens_per_completion: int = 3

    # Base overhead when any tools are present in the request
    base_tool_tokens: int = 16

    # Per-tool overhead for each tool definition
    tokens_per_tool: int = 8

    # Safety multiplier for tool definition token counts
    tool_definition_multiplier: float = 1.1

    # Safety multiplier for tool_calls content blocks
    tool_calls_multiplier: float = 1.5


# ---------------------------------------------------------------------------
# Pre-built configs per provider
# ---------------------------------------------------------------------------

COPILOT_CONFIG = TokenCountingConfig(
    tokens_per_message=3,
    tokens_per_name=1,
    tokens_per_completion=3,
    base_tool_tokens=16,
    tokens_per_tool=8,
    tool_definition_multiplier=1.1,
    tool_calls_multiplier=1.5,
)

ANTHROPIC_CONFIG = TokenCountingConfig(
    tokens_per_message=3,
    tokens_per_name=1,
    tokens_per_completion=3,
    base_tool_tokens=0,
    tokens_per_tool=8,
    tool_definition_multiplier=1.0,
    tool_calls_multiplier=1.0,
)

OPENAI_CONFIG = TokenCountingConfig(
    tokens_per_message=3,
    tokens_per_name=1,
    tokens_per_completion=3,
    base_tool_tokens=8,
    tokens_per_tool=8,
    tool_definition_multiplier=1.0,
    tool_calls_multiplier=1.0,
)

# Default config preserves backward compatibility (= Copilot-aligned)
DEFAULT_CONFIG = COPILOT_CONFIG

# Mapping from provider name prefix to config
_PROVIDER_CONFIG_MAP: dict[str, TokenCountingConfig] = {
    "github-copilot": COPILOT_CONFIG,
    "anthropic": ANTHROPIC_CONFIG,
    "openai": OPENAI_CONFIG,
}


def get_config_for_provider(provider_name: str | None) -> TokenCountingConfig:
    """Return the appropriate ``TokenCountingConfig`` for a provider.

    Args:
        provider_name: Provider name (e.g. ``"github-copilot"``, ``"anthropic"``).
            When ``None``, returns ``DEFAULT_CONFIG`` (Copilot-aligned).

    Returns:
        The matching config, falling back to ``DEFAULT_CONFIG`` for unknown providers.
    """
    if provider_name is None:
        return DEFAULT_CONFIG
    return _PROVIDER_CONFIG_MAP.get(provider_name, DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Anthropic upstream token counting API
# ---------------------------------------------------------------------------


async def count_tokens_via_anthropic_api(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Call Anthropic's ``/messages/count_tokens`` API for an exact count.

    Args:
        base_url: Anthropic API base URL (e.g. ``"https://api.anthropic.com/v1"``).
            This should include the ``/v1`` path segment, matching the provider's
            ``base_url`` attribute.
        api_key: Anthropic API key.
        model: Model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        messages: List of message dicts in Anthropic format.
        system: Optional system prompt (string or list of text blocks).
        tools: Optional list of tool definition dicts.

    Returns:
        The exact input token count from Anthropic's API.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
        httpx.RequestError: On connection failures.
    """
    url = f"{base_url.rstrip('/')}/messages/count_tokens"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if system is not None:
        payload["system"] = system
    if tools is not None:
        payload["tools"] = tools

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    input_tokens: int = data["input_tokens"]
    logger.debug(
        "Anthropic API count_tokens returned %d for model=%s",
        input_tokens,
        model,
    )
    return input_tokens
