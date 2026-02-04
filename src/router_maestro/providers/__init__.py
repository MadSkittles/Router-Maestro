"""Providers module for router-maestro."""

from router_maestro.providers.anthropic import AnthropicProvider
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ModelInfo,
    ProviderError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponsesToolCall,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.providers.openai import OpenAIProvider
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    # Base classes
    "BaseProvider",
    "ProviderError",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "ChatStreamChunk",
    "ModelInfo",
    "ResponsesRequest",
    "ResponsesResponse",
    "ResponsesStreamChunk",
    "ResponsesToolCall",
    "OpenAIChatProvider",
    # Providers
    "CopilotProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenAICompatibleProvider",
]
