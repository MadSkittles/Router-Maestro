"""Google Gemini API-compatible schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Request types
# ============================================================================


class GeminiFunctionCall(BaseModel):
    """A function call from the model."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class GeminiFunctionResponse(BaseModel):
    """A function response from the user."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    id: str | None = None
    response: dict[str, Any] = Field(default_factory=dict)


class GeminiInlineData(BaseModel):
    """Inline binary data (e.g. images)."""

    model_config = ConfigDict(populate_by_name=True)

    mime_type: str = Field(alias="mimeType")
    data: str  # base64 encoded


class GeminiPart(BaseModel):
    """A part of a Gemini content message."""

    model_config = ConfigDict(populate_by_name=True)

    text: str | None = None
    function_call: GeminiFunctionCall | None = Field(default=None, alias="functionCall")
    function_response: GeminiFunctionResponse | None = Field(default=None, alias="functionResponse")
    inline_data: GeminiInlineData | None = Field(default=None, alias="inlineData")


class GeminiContent(BaseModel):
    """A Gemini content message."""

    model_config = ConfigDict(populate_by_name=True)

    role: str | None = None  # "user" or "model"
    parts: list[GeminiPart] = Field(default_factory=list)


class GeminiFunctionDeclaration(BaseModel):
    """A function declaration for tools."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class GeminiTool(BaseModel):
    """A Gemini tool definition."""

    model_config = ConfigDict(populate_by_name=True)

    function_declarations: list[GeminiFunctionDeclaration] | None = Field(
        default=None, alias="functionDeclarations"
    )


class GeminiFunctionCallingConfig(BaseModel):
    """Function calling configuration."""

    model_config = ConfigDict(populate_by_name=True)

    mode: str | None = None  # "AUTO", "ANY", "NONE"


class GeminiToolConfig(BaseModel):
    """Tool configuration."""

    model_config = ConfigDict(populate_by_name=True)

    function_calling_config: GeminiFunctionCallingConfig | None = Field(
        default=None, alias="functionCallingConfig"
    )


class GeminiGenerationConfig(BaseModel):
    """Generation configuration."""

    model_config = ConfigDict(populate_by_name=True)

    temperature: float | None = None
    top_p: float | None = Field(default=None, alias="topP")
    top_k: int | None = Field(default=None, alias="topK")
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")
    candidate_count: int | None = Field(default=None, alias="candidateCount")
    response_mime_type: str | None = Field(default=None, alias="responseMimeType")


class GeminiGenerateContentRequest(BaseModel):
    """Gemini generateContent request body."""

    model_config = ConfigDict(populate_by_name=True)

    contents: list[GeminiContent] | None = None
    system_instruction: GeminiContent | None = Field(default=None, alias="systemInstruction")
    generation_config: GeminiGenerationConfig | None = Field(default=None, alias="generationConfig")
    tools: list[GeminiTool] | None = None
    tool_config: GeminiToolConfig | None = Field(default=None, alias="toolConfig")


# ============================================================================
# Response types
# ============================================================================


class GeminiCandidate(BaseModel):
    """A response candidate."""

    model_config = ConfigDict(populate_by_name=True)

    content: GeminiContent | None = None
    finish_reason: str | None = Field(
        default=None, alias="finishReason"
    )  # "STOP", "MAX_TOKENS", "SAFETY", "OTHER"
    index: int = 0


class GeminiUsageMetadata(BaseModel):
    """Token usage metadata."""

    model_config = ConfigDict(populate_by_name=True)

    prompt_token_count: int = Field(default=0, alias="promptTokenCount")
    candidates_token_count: int = Field(default=0, alias="candidatesTokenCount")
    total_token_count: int = Field(default=0, alias="totalTokenCount")


class GeminiGenerateContentResponse(BaseModel):
    """Gemini generateContent response body."""

    model_config = ConfigDict(populate_by_name=True)

    candidates: list[GeminiCandidate] = Field(default_factory=list)
    usage_metadata: GeminiUsageMetadata | None = Field(default=None, alias="usageMetadata")
    model_version: str | None = Field(default=None, alias="modelVersion")


class GeminiErrorDetail(BaseModel):
    """Gemini error detail."""

    code: int
    message: str
    status: str


class GeminiErrorResponse(BaseModel):
    """Gemini error response."""

    error: GeminiErrorDetail


# ============================================================================
# Streaming state
# ============================================================================


class GeminiStreamState(BaseModel):
    """State for tracking Gemini streaming translation."""

    accumulated_text: str = ""
    estimated_input_tokens: int = 0
    accumulated_completion_tokens: int = 0
    accumulated_prompt_tokens: int = 0
    has_sent_content: bool = False
    tool_calls_buffer: list[dict[str, Any]] = Field(default_factory=list)
