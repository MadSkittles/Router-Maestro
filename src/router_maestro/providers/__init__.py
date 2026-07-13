"""Providers module for router-maestro."""

from router_maestro.providers.anthropic import AnthropicProvider
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    TIMEOUT_STREAMING,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
    client_cancelled_outcome,
    exception_outcome,
    finish_reason_for_outcome,
    resolve_terminal_outcome,
    unexpected_eof_outcome,
)
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.providers.openai import OpenAIProvider
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    # Constants
    "TIMEOUT_NON_STREAMING",
    "TIMEOUT_STREAMING",
    # Base classes
    "BaseProvider",
    "ProviderError",
    "ProviderFailureKind",
    "RequestOptionError",
    "ResponseStatus",
    "TerminalError",
    "TerminalOutcome",
    "TransportTermination",
    "resolve_terminal_outcome",
    "unexpected_eof_outcome",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "ChatStreamChunk",
    "client_cancelled_outcome",
    "exception_outcome",
    "finish_reason_for_outcome",
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
