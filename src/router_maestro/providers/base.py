"""Base provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from logging import Logger
from typing import NoReturn

import httpx


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
    thinking_budget: int | None = None
    thinking_type: str | None = None  # "enabled", "adaptive", "disabled"
    extra: dict = field(default_factory=dict)

    def with_thinking(
        self,
        *,
        thinking_budget: int | None,
        thinking_type: str | None,
    ) -> "ChatRequest":
        """Return new ChatRequest with updated thinking parameters (immutable)."""
        return ChatRequest(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
            tools=self.tools,
            tool_choice=self.tool_choice,
            thinking_budget=thinking_budget,
            thinking_type=thinking_type,
            extra=self.extra,
        )


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
    max_prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    max_context_window_tokens: int | None = None
    supports_thinking: bool = False
    supports_vision: bool = False

    def with_overrides(
        self,
        *,
        max_prompt_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_context_window_tokens: int | None = None,
    ) -> "ModelInfo":
        """Return new ModelInfo with specified limits overridden (immutable)."""
        return ModelInfo(
            id=self.id,
            name=self.name,
            provider=self.provider,
            max_prompt_tokens=(
                max_prompt_tokens if max_prompt_tokens is not None else self.max_prompt_tokens
            ),
            max_output_tokens=(
                max_output_tokens if max_output_tokens is not None else self.max_output_tokens
            ),
            max_context_window_tokens=(
                max_context_window_tokens
                if max_context_window_tokens is not None
                else self.max_context_window_tokens
            ),
            supports_thinking=self.supports_thinking,
            supports_vision=self.supports_vision,
        )


@dataclass
class ResponsesToolCall:
    """A tool/function call from the Responses API."""

    call_id: str
    name: str
    arguments: str


@dataclass
class ResponsesRequest:
    """Request for the Responses API (used by Codex models)."""

    model: str
    input: str | list  # Can be string or list of message dicts
    stream: bool = False
    instructions: str | None = None
    temperature: float = 1.0
    max_output_tokens: int | None = None
    # Tool support
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None


@dataclass
class ResponsesResponse:
    """Response from the Responses API."""

    content: str
    model: str
    usage: dict | None = None
    tool_calls: list[ResponsesToolCall] | None = None


@dataclass
class ResponsesStreamChunk:
    """A chunk from streaming Responses API completion."""

    content: str
    finish_reason: str | None = None
    usage: dict | None = None
    # Tool call support
    tool_call: ResponsesToolCall | None = None  # A complete tool call
    tool_call_delta: dict | None = None  # Partial tool call for streaming


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

    @staticmethod
    def _raise_http_status_error(
        label: str,
        error: httpx.HTTPStatusError,
        logger: Logger,
        *,
        stream: bool = False,
        include_body: bool = False,
    ) -> NoReturn:
        """Raise a ProviderError from an HTTP status error.

        Args:
            label: Provider label for log messages
            error: The httpx status error
            logger: Logger instance for error logging
            stream: Whether this is a streaming request
            include_body: Whether to include the response body in the error message
        """
        retryable = error.response.status_code in (429, 500, 502, 503, 504)
        suffix = " stream" if stream else ""
        if include_body:
            try:
                error_body = error.response.text
            except Exception:
                error_body = ""
            logger.error(
                "%s%s API error: %d - %s",
                label,
                suffix,
                error.response.status_code,
                error_body[:200],
            )
            raise ProviderError(
                f"{label} API error: {error.response.status_code} - {error_body}",
                status_code=error.response.status_code,
                retryable=retryable,
            )
        logger.error("%s%s API error: %d", label, suffix, error.response.status_code)
        raise ProviderError(
            f"{label} API error: {error.response.status_code}",
            status_code=error.response.status_code,
            retryable=retryable,
        )

    @staticmethod
    def _raise_http_error(
        label: str,
        error: httpx.HTTPError,
        logger: Logger,
        *,
        stream: bool = False,
    ) -> NoReturn:
        """Raise a ProviderError from a generic HTTP error.

        Args:
            label: Provider label for log messages
            error: The httpx error
            logger: Logger instance for error logging
            stream: Whether this is a streaming request
        """
        suffix = " stream" if stream else ""
        logger.error("%s%s HTTP error: %s", label, suffix, error)
        raise ProviderError(f"HTTP error: {error}", retryable=True)

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        """Generate a Responses API completion (for Codex models).

        Args:
            request: Responses completion request

        Returns:
            Responses completion response

        Raises:
            NotImplementedError: If provider does not support Responses API
        """
        raise NotImplementedError("Provider does not support Responses API")

    async def responses_completion_stream(
        self, request: ResponsesRequest
    ) -> AsyncIterator[ResponsesStreamChunk]:
        """Generate a streaming Responses API completion (for Codex models).

        Args:
            request: Responses completion request

        Yields:
            Responses completion chunks

        Raises:
            NotImplementedError: If provider does not support Responses API
        """
        raise NotImplementedError("Provider does not support Responses API")
        # Make this a generator (required for type checking)
        if False:
            yield ResponsesStreamChunk(content="")
