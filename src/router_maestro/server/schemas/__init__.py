"""Server schemas."""

from router_maestro.server.schemas.admin import (
    AuthListResponse,
    AuthProviderInfo,
    LoginRequest,
    ModelInfo,
    ModelsResponse,
    OAuthInitResponse,
    OAuthStatusResponse,
    PrioritiesResponse,
    PrioritiesUpdateRequest,
)
from router_maestro.server.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ErrorDetail,
    ErrorResponse,
    ModelList,
    ModelObject,
)

__all__ = [
    # Admin schemas
    "AuthListResponse",
    "AuthProviderInfo",
    "LoginRequest",
    "ModelInfo",
    "ModelsResponse",
    "OAuthInitResponse",
    "OAuthStatusResponse",
    "PrioritiesResponse",
    "PrioritiesUpdateRequest",
    # OpenAI schemas
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionUsage",
    "ChatMessage",
    "ErrorDetail",
    "ErrorResponse",
    "ModelList",
    "ModelObject",
]
