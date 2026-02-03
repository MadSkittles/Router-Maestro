"""OpenAI Responses API schemas for Codex models."""

from typing import Any

from pydantic import BaseModel, Field

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
    """A function call from a previous assistant response."""

    type: str = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str
    status: str = "completed"


class ResponsesFunctionCallOutput(BaseModel):
    """Output/result from a function call execution."""

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
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_output_tokens: int | None = None
    # Tool support
    tools: list[ResponsesFunctionTool | dict[str, Any]] | None = None
    tool_choice: str | ResponsesToolChoiceFunction | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


# ============================================================================
# Output Types
# ============================================================================


class ResponsesOutputText(BaseModel):
    """Text content in the response output."""

    type: str = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponsesMessageOutput(BaseModel):
    """A message output in the response."""

    type: str = "message"
    id: str
    role: str = "assistant"
    status: str = "completed"
    content: list[ResponsesOutputText]


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
