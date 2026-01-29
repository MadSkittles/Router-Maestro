"""Anthropic API-compatible schemas."""

from typing import Literal

from pydantic import BaseModel, Field

# Request types


class AnthropicTextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class AnthropicImageSource(BaseModel):
    """Image source for base64 encoded images."""

    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class AnthropicImageBlock(BaseModel):
    """Image content block."""

    type: Literal["image"] = "image"
    source: AnthropicImageSource


class AnthropicToolResultContentBlock(BaseModel):
    """Content block within tool result (text or image)."""

    type: Literal["text", "image"]
    text: str | None = None
    source: AnthropicImageSource | None = None


class AnthropicToolResultBlock(BaseModel):
    """Tool result content block."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[AnthropicToolResultContentBlock]
    is_error: bool | None = None


class AnthropicToolUseBlock(BaseModel):
    """Tool use content block."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


class AnthropicThinkingBlock(BaseModel):
    """Thinking content block."""

    type: Literal["thinking"] = "thinking"
    thinking: str


AnthropicUserContentBlock = AnthropicTextBlock | AnthropicImageBlock | AnthropicToolResultBlock
AnthropicAssistantContentBlock = AnthropicTextBlock | AnthropicToolUseBlock | AnthropicThinkingBlock


class AnthropicUserMessage(BaseModel):
    """User message."""

    role: Literal["user"] = "user"
    content: str | list[AnthropicUserContentBlock]


class AnthropicAssistantMessage(BaseModel):
    """Assistant message."""

    role: Literal["assistant"] = "assistant"
    content: str | list[AnthropicAssistantContentBlock]


AnthropicMessage = AnthropicUserMessage | AnthropicAssistantMessage


class AnthropicTool(BaseModel):
    """Tool definition."""

    name: str
    description: str | None = None
    input_schema: dict


class AnthropicToolChoice(BaseModel):
    """Tool choice configuration."""

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None


class AnthropicThinkingConfig(BaseModel):
    """Thinking configuration."""

    type: Literal["enabled"] = "enabled"
    budget_tokens: int | None = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    system: str | list[AnthropicTextBlock] | None = None
    metadata: dict | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None
    thinking: AnthropicThinkingConfig | None = None
    service_tier: Literal["auto", "standard_only"] | None = None


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic count_tokens API request (max_tokens not required)."""

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicTextBlock] | None = None
    tools: list[AnthropicTool] | None = None


# Response types


class AnthropicUsage(BaseModel):
    """Token usage information."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    service_tier: Literal["standard", "priority", "batch"] | None = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicAssistantContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"]
        | None
    )
    stop_sequence: str | None = None
    usage: AnthropicUsage


# Streaming event types


class AnthropicMessageStartEvent(BaseModel):
    """Message start event."""

    type: Literal["message_start"] = "message_start"
    message: dict  # Partial AnthropicMessagesResponse


class AnthropicContentBlockStartEvent(BaseModel):
    """Content block start event."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: dict


class AnthropicContentBlockDeltaEvent(BaseModel):
    """Content block delta event."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: dict


class AnthropicContentBlockStopEvent(BaseModel):
    """Content block stop event."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class AnthropicMessageDeltaEvent(BaseModel):
    """Message delta event."""

    type: Literal["message_delta"] = "message_delta"
    delta: dict
    usage: dict | None = None


class AnthropicMessageStopEvent(BaseModel):
    """Message stop event."""

    type: Literal["message_stop"] = "message_stop"


class AnthropicPingEvent(BaseModel):
    """Ping event."""

    type: Literal["ping"] = "ping"


class AnthropicErrorEvent(BaseModel):
    """Error event."""

    type: Literal["error"] = "error"
    error: dict


AnthropicStreamEvent = (
    AnthropicMessageStartEvent
    | AnthropicContentBlockStartEvent
    | AnthropicContentBlockDeltaEvent
    | AnthropicContentBlockStopEvent
    | AnthropicMessageDeltaEvent
    | AnthropicMessageStopEvent
    | AnthropicPingEvent
    | AnthropicErrorEvent
)


class AnthropicStreamState(BaseModel):
    """State for tracking streaming translation."""

    message_start_sent: bool = False
    content_block_index: int = 0
    content_block_open: bool = False
    tool_calls: dict[int, dict] = Field(default_factory=dict)
    estimated_input_tokens: int = 0  # Estimated input tokens from request
    last_usage: dict | None = None  # Track the latest usage from stream chunks
    message_complete: bool = False  # Track if message_stop was sent
