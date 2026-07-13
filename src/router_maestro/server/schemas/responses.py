"""OpenAI Responses API schemas for Codex models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================================
# Input Types
# ============================================================================


class ResponsesInputTextContent(BaseModel):
    """Text content block in input message."""

    type: str = "input_text"
    text: str


class ResponsesInputMessage(BaseModel):
    """An input message for the Responses API.

    Supports both simple string content and array of content blocks.
    """

    type: str = "message"
    role: str
    content: str | list[ResponsesInputTextContent | dict[str, Any]]


class ResponsesFunctionCallInput(BaseModel):
    """A function call from a previous assistant response.

    ``extra="allow"`` is critical: Codex echoes back upstream-emitted fields
    (notably ``namespace`` for MCP-routed tools like ``kusto/execute_query``)
    that this schema doesn't enumerate. Without ``allow``, Pydantic silently
    drops them at the FastAPI request boundary and Copilot CAPI 400s the next
    turn with ``Missing namespace for function_call 'X'`` (v0.3.8/9/10 bug).
    """

    model_config = ConfigDict(extra="allow")

    type: str = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str
    status: str = "completed"
    # MCP namespace echoed back from a prior turn — required by Copilot CAPI
    # when the original ``function_call`` was emitted with a namespace, e.g.
    # ``mcp__kusto_mcp__``. Declared explicitly so it survives ``model_dump``.
    namespace: str | None = None


class ResponsesFunctionCallOutput(BaseModel):
    """Output/result from a function call execution."""

    model_config = ConfigDict(extra="allow")

    type: str = "function_call_output"
    call_id: str
    output: str


# ============================================================================
# Tool Definitions
# ============================================================================


class ResponsesFunctionParameters(BaseModel):
    """JSON Schema for function parameters."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ResponsesFunctionTool(BaseModel):
    """A function tool definition."""

    type: str = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ResponsesToolChoiceFunction(BaseModel):
    """Specific function tool choice."""

    type: str = "function"
    name: str


class ResponsesReasoningConfig(BaseModel):
    """Reasoning controls represented by Router-Maestro in Phase 1."""

    model_config = ConfigDict(extra="forbid")

    effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None = None


# ============================================================================
# Request
# ============================================================================


class ResponsesRequest(BaseModel):
    """Request for the Responses API (used by Codex models).

    The input can be:
    - A simple string prompt
    - A list of message objects with role and content
    - A list including function_call and function_call_output items
    """

    # Retain unknown top-level options for the route's native unsupported-option
    # response instead of silently dropping them or returning FastAPI's 422.
    model_config = ConfigDict(extra="allow")

    model: str
    input: (
        str
        | list[
            ResponsesInputMessage
            | ResponsesFunctionCallInput
            | ResponsesFunctionCallOutput
            | dict[str, Any]
        ]
    )
    stream: bool = False
    instructions: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] | None = None
    service_tier: str | None = None
    max_output_tokens: int | None = None
    # Tool support
    tools: list[ResponsesFunctionTool | dict[str, Any]] | None = None
    tool_choice: str | ResponsesToolChoiceFunction | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    # OpenAI Responses reasoning shape, including the catalog-advertised
    # "minimal" tier. Router-Maestro also accepts "xhigh"/"max" extensions.
    # Keep the raw boundary value so route-level validation can return the
    # OpenAI-native 400 envelope instead of FastAPI's generic 422 body.
    reasoning: Any | None = None

    @field_validator("temperature", mode="before")
    @classmethod
    def reject_explicit_null_temperature(cls, value):
        if value is None:
            raise ValueError("temperature must be a number when provided")
        return value


# ============================================================================
# Output Types
# ============================================================================


class ResponsesOutputText(BaseModel):
    """Text content in the response output."""

    type: str = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponsesRefusalOutput(BaseModel):
    """Refusal content in the response output."""

    type: str = "refusal"
    refusal: str


class ResponsesMessageOutput(BaseModel):
    """A message output in the response."""

    type: str = "message"
    id: str
    role: str = "assistant"
    status: str = "completed"
    content: list[ResponsesOutputText | ResponsesRefusalOutput]


class ResponsesFunctionCallOutput(BaseModel):
    """A function call output in the response."""

    type: str = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: str = "completed"


class ResponsesReasoningOutput(BaseModel):
    """A reasoning output in the response."""

    type: str = "reasoning"
    id: str
    summary: list[dict[str, Any]] = Field(default_factory=list)


class ResponsesUsage(BaseModel):
    """Token usage information for Responses API."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, int] | None = None
    output_tokens_details: dict[str, int] | None = None


# ============================================================================
# Response
# ============================================================================


class ResponsesResponse(BaseModel):
    """Response from the Responses API."""

    id: str
    object: str = "response"
    model: str
    status: str = "completed"
    output: list[
        ResponsesReasoningOutput
        | ResponsesMessageOutput
        | ResponsesFunctionCallOutput
        | dict[str, Any]
    ]
    usage: ResponsesUsage | None = None
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None


# ============================================================================
# Streaming Events
# ============================================================================


class ResponsesStreamEvent(BaseModel):
    """A streaming event from the Responses API."""

    type: str
    # Additional fields depend on event type


class ResponsesDeltaEvent(BaseModel):
    """A delta event in streaming response."""

    type: str = "response.output_text.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponsesDoneEvent(BaseModel):
    """A done event in streaming response."""

    type: str = "response.done"
    response: ResponsesResponse
