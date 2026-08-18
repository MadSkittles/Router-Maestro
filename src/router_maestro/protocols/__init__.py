"""Lazy protocol conversion contracts."""

from importlib import import_module
from typing import TYPE_CHECKING

from router_maestro.protocols.envelope import RequestEnvelope
from router_maestro.protocols.models import (
    ContentBlock,
    ConversionMode,
    FileContent,
    FrozenJsonValue,
    ImageContent,
    JsonScalar,
    MessageContent,
    MessageRole,
    OpaqueState,
    ReasoningConfig,
    ReasoningSummary,
    RefusalContent,
    RepresentabilityReport,
    RequestManifest,
    SemanticEvent,
    SemanticEventType,
    SemanticItem,
    SemanticMessage,
    SemanticRequest,
    SemanticResponse,
    TerminalMetadata,
    TextContent,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolResult,
    Usage,
    UsageMode,
    WireProtocol,
)
from router_maestro.protocols.runtime import (
    DuplicateProtocolRuntimeError,
    OpaqueStateDecodeHook,
    OpaqueStateEncodeHook,
    ProtocolDecodeError,
    ProtocolRepresentabilityError,
    ProtocolRuntime,
    ProtocolRuntimeNotFoundError,
    ProtocolRuntimeRegistry,
    UnsupportedProtocolOperationError,
    check_request_representability,
)

if TYPE_CHECKING:
    from router_maestro.protocols.anthropic import (
        AnthropicMessagesRuntime,
        AnthropicStreamDecoder,
        AnthropicStreamEncoder,
    )
    from router_maestro.protocols.gemini import (
        GeminiRuntime,
        GeminiStreamDecoder,
        GeminiStreamEncoder,
    )
    from router_maestro.protocols.legacy import (
        semantic_events_from_legacy_chat_chunk,
        semantic_request_to_legacy_chat,
        semantic_response_from_legacy_chat,
    )
    from router_maestro.protocols.openai_chat import (
        OpenAIChatRuntime,
        OpenAIChatStreamDecoder,
        OpenAIChatStreamEncoder,
        chat_chunk_to_semantic_events,
        chat_request_to_semantic,
        chat_response_to_semantic,
        semantic_events_to_chat_chunks,
        semantic_to_chat_request,
        semantic_to_chat_response,
    )
    from router_maestro.protocols.openai_responses import (
        OpenAIResponsesRuntime,
        OpenAIResponsesStreamDecoder,
        OpenAIResponsesStreamEncoder,
        responses_chunk_to_semantic_events,
        responses_request_to_semantic,
        responses_response_to_semantic,
        semantic_events_to_responses_chunks,
        semantic_to_responses_request,
        semantic_to_responses_response,
    )

_LAZY_EXPORTS = {
    "AnthropicMessagesRuntime": ("router_maestro.protocols.anthropic", "AnthropicMessagesRuntime"),
    "AnthropicStreamDecoder": ("router_maestro.protocols.anthropic", "AnthropicStreamDecoder"),
    "AnthropicStreamEncoder": ("router_maestro.protocols.anthropic", "AnthropicStreamEncoder"),
    "GeminiRuntime": ("router_maestro.protocols.gemini", "GeminiRuntime"),
    "GeminiStreamDecoder": ("router_maestro.protocols.gemini", "GeminiStreamDecoder"),
    "GeminiStreamEncoder": ("router_maestro.protocols.gemini", "GeminiStreamEncoder"),
    "OpenAIChatRuntime": ("router_maestro.protocols.openai_chat", "OpenAIChatRuntime"),
    "OpenAIChatStreamDecoder": (
        "router_maestro.protocols.openai_chat",
        "OpenAIChatStreamDecoder",
    ),
    "OpenAIChatStreamEncoder": (
        "router_maestro.protocols.openai_chat",
        "OpenAIChatStreamEncoder",
    ),
    "OpenAIResponsesRuntime": (
        "router_maestro.protocols.openai_responses",
        "OpenAIResponsesRuntime",
    ),
    "OpenAIResponsesStreamDecoder": (
        "router_maestro.protocols.openai_responses",
        "OpenAIResponsesStreamDecoder",
    ),
    "OpenAIResponsesStreamEncoder": (
        "router_maestro.protocols.openai_responses",
        "OpenAIResponsesStreamEncoder",
    ),
    "chat_chunk_to_semantic_events": (
        "router_maestro.protocols.openai_chat",
        "chat_chunk_to_semantic_events",
    ),
    "chat_request_to_semantic": (
        "router_maestro.protocols.openai_chat",
        "chat_request_to_semantic",
    ),
    "chat_response_to_semantic": (
        "router_maestro.protocols.openai_chat",
        "chat_response_to_semantic",
    ),
    "responses_chunk_to_semantic_events": (
        "router_maestro.protocols.openai_responses",
        "responses_chunk_to_semantic_events",
    ),
    "responses_request_to_semantic": (
        "router_maestro.protocols.openai_responses",
        "responses_request_to_semantic",
    ),
    "responses_response_to_semantic": (
        "router_maestro.protocols.openai_responses",
        "responses_response_to_semantic",
    ),
    "semantic_events_from_legacy_chat_chunk": (
        "router_maestro.protocols.legacy",
        "semantic_events_from_legacy_chat_chunk",
    ),
    "semantic_events_to_chat_chunks": (
        "router_maestro.protocols.openai_chat",
        "semantic_events_to_chat_chunks",
    ),
    "semantic_events_to_responses_chunks": (
        "router_maestro.protocols.openai_responses",
        "semantic_events_to_responses_chunks",
    ),
    "semantic_request_to_legacy_chat": (
        "router_maestro.protocols.legacy",
        "semantic_request_to_legacy_chat",
    ),
    "semantic_response_from_legacy_chat": (
        "router_maestro.protocols.legacy",
        "semantic_response_from_legacy_chat",
    ),
    "semantic_to_chat_request": (
        "router_maestro.protocols.openai_chat",
        "semantic_to_chat_request",
    ),
    "semantic_to_chat_response": (
        "router_maestro.protocols.openai_chat",
        "semantic_to_chat_response",
    ),
    "semantic_to_responses_request": (
        "router_maestro.protocols.openai_responses",
        "semantic_to_responses_request",
    ),
    "semantic_to_responses_response": (
        "router_maestro.protocols.openai_responses",
        "semantic_to_responses_response",
    ),
}


def __getattr__(name: str) -> object:
    """Load concrete codecs only when requested, avoiding provider import cycles."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "AnthropicMessagesRuntime",
    "AnthropicStreamDecoder",
    "AnthropicStreamEncoder",
    "ContentBlock",
    "ConversionMode",
    "DuplicateProtocolRuntimeError",
    "FileContent",
    "FrozenJsonValue",
    "GeminiRuntime",
    "GeminiStreamDecoder",
    "GeminiStreamEncoder",
    "ImageContent",
    "JsonScalar",
    "MessageContent",
    "MessageRole",
    "OpaqueState",
    "OpaqueStateDecodeHook",
    "OpaqueStateEncodeHook",
    "OpenAIChatRuntime",
    "OpenAIChatStreamDecoder",
    "OpenAIChatStreamEncoder",
    "OpenAIResponsesRuntime",
    "OpenAIResponsesStreamDecoder",
    "OpenAIResponsesStreamEncoder",
    "ProtocolDecodeError",
    "ProtocolRepresentabilityError",
    "ProtocolRuntime",
    "ProtocolRuntimeNotFoundError",
    "ProtocolRuntimeRegistry",
    "RefusalContent",
    "ReasoningConfig",
    "ReasoningSummary",
    "RepresentabilityReport",
    "RequestEnvelope",
    "RequestManifest",
    "SemanticEvent",
    "SemanticEventType",
    "SemanticItem",
    "SemanticMessage",
    "SemanticRequest",
    "SemanticResponse",
    "chat_chunk_to_semantic_events",
    "check_request_representability",
    "chat_request_to_semantic",
    "chat_response_to_semantic",
    "responses_chunk_to_semantic_events",
    "responses_request_to_semantic",
    "responses_response_to_semantic",
    "semantic_events_from_legacy_chat_chunk",
    "semantic_events_to_chat_chunks",
    "semantic_events_to_responses_chunks",
    "semantic_request_to_legacy_chat",
    "semantic_response_from_legacy_chat",
    "semantic_to_chat_request",
    "semantic_to_chat_response",
    "semantic_to_responses_request",
    "semantic_to_responses_response",
    "TerminalMetadata",
    "TextContent",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "ToolResult",
    "UnsupportedProtocolOperationError",
    "Usage",
    "UsageMode",
    "WireProtocol",
]
