"""Base provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list  # Can be str or list for multimodal content (images)
    tool_call_id: str | None = None  # Required for tool role messages
    tool_calls: list[dict] | None = None  # For assistant messages with tool calls


@dataclass
class ChatRequest:
    """Request for chat completion."""

    model: str
    messages: list[Message]
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict] | None = None  # OpenAI format tool definitions
    # "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}
    tool_choice: str | dict | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from chat completion."""

    content: str
    model: str
    finish_reason: str = "stop"
    usage: dict | None = None  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}


@dataclass
class ChatStreamChunk:
    """A chunk from streaming chat completion."""

    content: str
    finish_reason: str | None = None
    usage: dict | None = None  # Token usage info (typically in final chunk)
    tool_calls: list[dict] | None = None  # Tool call deltas for streaming


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    provider: str


class ProviderError(Exception):
    """Error from a provider."""

    def __init__(self, message: str, status_code: int = 500, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class BaseProvider(ABC):
    """Abstract base class for model providers."""

    name: str = "base"

    @abstractmethod
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion.

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        pass

    @abstractmethod
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion.

        Args:
            request: Chat completion request

        Yields:
            Chat completion chunks
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models.

        Returns:
            List of available models
        """
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if the provider is authenticated.

        Returns:
            True if authenticated
        """
        pass

    async def ensure_token(self) -> None:
        """Ensure the provider has a valid token.

        Override this for providers that need token refresh.
        """
        pass
