"""OpenAI-compatible API schemas."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessageFunction(BaseModel):
    """Function details in a tool call."""

    name: str
    arguments: str


class ChatMessageToolCall(BaseModel):
    """A tool call in an assistant message."""

    id: str
    type: str = "function"
    function: ChatMessageFunction


class ChatMessage(BaseModel):
    """A message in the chat."""

    role: str
    content: str | list | None = None
    refusal: str | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    tool_call_id: str | None = None


class OpenAIThinkingConfig(BaseModel):
    """Strict Anthropic-style thinking extension accepted by Chat Completions."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["enabled", "adaptive", "disabled"]
    budget_tokens: Annotated[int, Field(strict=True, gt=0)] | None = None

    @field_validator("budget_tokens", mode="before")
    @classmethod
    def reject_explicit_null_budget(cls, value):
        """Distinguish an omitted default budget from an explicit null value."""
        if value is None:
            raise ValueError("budget_tokens must be a positive integer when provided")
        return value


class ChatCompletionStreamOptions(BaseModel):
    """Client-facing encoding options for a Chat Completions stream."""

    # Retain unknown nested options rather than raising FastAPI's 422; a
    # transparent proxy forwards or ignores options it does not model.
    model_config = ConfigDict(extra="allow")

    include_usage: Annotated[bool, Field(strict=True)] = False


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    # Preserve unknown top-level options rather than rejecting them: Router-
    # Maestro is a transparent proxy, so options it does not model are ignored
    # (or forwarded by the provider) instead of failing the request. ``extra=
    # allow`` keeps them addressable; the default ``extra=ignore`` would also
    # work, while ``extra=forbid`` would wrongly 422 on unknown client options.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = None
    stream: bool = False
    # Keep the raw value until the route validates it into
    # ``ChatCompletionStreamOptions`` and can encode failures as native 400s.
    stream_options: Any | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
    service_tier: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    # OpenAI-style reasoning effort, including the catalog-advertised "minimal" tier.
    # Router-Maestro additionally accepts "xhigh"/"max" (downgraded when needed).
    reasoning_effort: str | None = None
    # Anthropic-style passthrough; some SDKs forward {"type": "...", "budget_tokens": N}
    # Keep the raw value at the FastAPI boundary so the route can encode every
    # invalid extension shape as an OpenAI-native 400 instead of a generic 422.
    thinking: Any | None = None

    @field_validator("temperature", mode="before")
    @classmethod
    def reject_explicit_null_temperature(cls, value):
        if value is None:
            raise ValueError("temperature must be a number when provided")
        return value


class ChatCompletionChoice(BaseModel):
    """A choice in the chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str | None


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A choice in a streaming chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A chunk in streaming response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: ChatCompletionUsage | None = None


class ModelObject(BaseModel):
    """A model object."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str
    max_prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    max_context_window_tokens: int | None = None
    supports_thinking: bool | None = None
    supports_vision: bool | None = None


class ModelList(BaseModel):
    """List of models."""

    object: str = "list"
    data: list[ModelObject]


class ErrorDetail(BaseModel):
    """Error detail."""

    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: ErrorDetail
