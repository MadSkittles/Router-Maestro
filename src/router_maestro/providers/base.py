"""Base provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from logging import Logger
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self

import httpx

from router_maestro.providers.outbound_contract import (
    OutboundContract,
    PermissiveOutboundContract,
)
from router_maestro.routing.capabilities import Operation, ProviderCapabilities

if TYPE_CHECKING:
    from router_maestro.routing.attempts import AttemptRecord
    from router_maestro.routing.model_ref import ModelRef

TIMEOUT_NON_STREAMING = httpx.Timeout(connect=30.0, read=240.0, write=30.0, pool=30.0)
TIMEOUT_STREAMING = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)


class TransportTermination(StrEnum):
    """How delivery of a provider stream ended."""

    EXPLICIT_TERMINAL = "explicit_terminal"
    UNEXPECTED_EOF = "unexpected_eof"
    CLIENT_CANCELLED = "client_cancelled"
    EXCEPTION = "exception"


class ResponseStatus(StrEnum):
    """Provider response semantics, independent of transport termination."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TerminalError:
    """Safe, protocol-independent terminal error details."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """Orthogonal transport and response state for a terminal stream outcome."""

    transport: TransportTermination
    response_status: ResponseStatus
    finish_reason: str | None = None
    incomplete_details: dict[str, Any] | None = None
    error: TerminalError | None = None

    @property
    def is_success(self) -> bool:
        """Return whether this is an explicitly completed response."""
        return (
            self.transport is TransportTermination.EXPLICIT_TERMINAL
            and self.response_status is ResponseStatus.COMPLETED
        )


def resolve_terminal_outcome(
    outcome: TerminalOutcome | None,
    finish_reason: str | None,
) -> TerminalOutcome | None:
    """Validate and resolve canonical and legacy stream terminal data."""
    completed_reasons = {"stop", "tool_calls"}
    incomplete_reasons = {"length", "content_filter"}
    finish_status = {
        **dict.fromkeys(completed_reasons, ResponseStatus.COMPLETED),
        **dict.fromkeys(incomplete_reasons, ResponseStatus.INCOMPLETE),
    }

    def protocol_error(message: str) -> TerminalOutcome:
        return TerminalOutcome(
            transport=TransportTermination.EXCEPTION,
            response_status=ResponseStatus.FAILED,
            error=TerminalError(code="upstream_protocol_error", message=message),
        )

    if outcome is None:
        if finish_reason is None:
            return None
        response_status = finish_status.get(finish_reason)
        if response_status is None:
            return protocol_error(f"Unknown upstream finish reason: {finish_reason}")
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=response_status,
            finish_reason=finish_reason,
        )

    allowed_statuses = {
        TransportTermination.UNEXPECTED_EOF: {ResponseStatus.UNKNOWN},
        TransportTermination.CLIENT_CANCELLED: {ResponseStatus.CANCELLED},
        TransportTermination.EXCEPTION: {ResponseStatus.FAILED},
        TransportTermination.EXPLICIT_TERMINAL: {
            ResponseStatus.COMPLETED,
            ResponseStatus.INCOMPLETE,
            ResponseStatus.FAILED,
            ResponseStatus.CANCELLED,
        },
    }
    if outcome.response_status not in allowed_statuses[outcome.transport]:
        return protocol_error(
            "Illegal upstream terminal combination: "
            f"{outcome.transport.value} with {outcome.response_status.value}"
        )
    if outcome.response_status in {ResponseStatus.COMPLETED, ResponseStatus.INCOMPLETE} and (
        outcome.error is not None
    ):
        return protocol_error(
            f"Terminal error conflicts with response status {outcome.response_status.value}"
        )

    canonical_finish = outcome.finish_reason
    if canonical_finish is not None:
        canonical_status = finish_status.get(canonical_finish)
        if canonical_status is None:
            return protocol_error(f"Unknown upstream finish reason: {canonical_finish}")
        if outcome.transport is not TransportTermination.EXPLICIT_TERMINAL:
            return protocol_error(
                f"Finish reason {canonical_finish} is invalid for {outcome.transport.value}"
            )
        if canonical_status is not outcome.response_status:
            return protocol_error(
                f"Finish reason {canonical_finish} conflicts with "
                f"response status {outcome.response_status.value}"
            )

    if finish_reason is not None:
        legacy_status = finish_status.get(finish_reason)
        if legacy_status is None:
            return protocol_error(f"Unknown upstream finish reason: {finish_reason}")
        neutral_incomplete_projection = (
            outcome.transport is TransportTermination.EXPLICIT_TERMINAL
            and outcome.response_status is ResponseStatus.INCOMPLETE
            and canonical_finish is None
            and finish_reason == "stop"
            and (outcome.incomplete_details or {}).get("reason")
            not in {"max_output_tokens", "content_filter"}
        )
        if canonical_finish is not None and canonical_finish != finish_reason:
            return protocol_error(
                "Canonical and legacy finish reasons conflict: "
                f"{canonical_finish} != {finish_reason}"
            )
        if (
            outcome.transport is not TransportTermination.EXPLICIT_TERMINAL
            or legacy_status is not outcome.response_status
        ) and not neutral_incomplete_projection:
            return protocol_error(
                f"Finish reason {finish_reason} conflicts with "
                f"response status {outcome.response_status.value}"
            )
        if canonical_finish is None and not neutral_incomplete_projection:
            outcome = replace(outcome, finish_reason=finish_reason)
            canonical_finish = finish_reason

    if outcome.response_status not in {ResponseStatus.COMPLETED, ResponseStatus.INCOMPLETE}:
        if canonical_finish is not None:
            return protocol_error(
                f"Finish reason {canonical_finish} conflicts with "
                f"response status {outcome.response_status.value}"
            )

    incomplete_reason = (outcome.incomplete_details or {}).get("reason")
    if outcome.incomplete_details is not None and (
        outcome.response_status is not ResponseStatus.INCOMPLETE
    ):
        return protocol_error(
            f"Incomplete details conflict with response status {outcome.response_status.value}"
        )
    if canonical_finish == "length" and incomplete_reason not in {None, "max_output_tokens"}:
        return protocol_error(
            f"Finish reason length conflicts with incomplete reason {incomplete_reason}"
        )
    if canonical_finish == "content_filter" and incomplete_reason not in {None, "content_filter"}:
        return protocol_error(
            f"Finish reason content_filter conflicts with incomplete reason {incomplete_reason}"
        )

    return outcome


def finish_reason_for_outcome(outcome: TerminalOutcome) -> str | None:
    """Map a canonical terminal outcome to the legacy Chat finish reason."""
    if outcome.finish_reason is not None:
        return outcome.finish_reason
    if outcome.response_status is ResponseStatus.COMPLETED:
        return "stop"
    if outcome.response_status is ResponseStatus.INCOMPLETE:
        reason = (outcome.incomplete_details or {}).get("reason")
        if reason == "max_output_tokens":
            return "length"
        if reason == "content_filter":
            return "content_filter"
        return "stop"
    return None


def unexpected_eof_outcome() -> TerminalOutcome:
    """Build the non-success outcome for transport EOF before a terminal event."""
    return TerminalOutcome(
        transport=TransportTermination.UNEXPECTED_EOF,
        response_status=ResponseStatus.UNKNOWN,
        error=TerminalError(
            code="unexpected_eof",
            message="Upstream stream ended before an explicit terminal event",
        ),
    )


def exception_outcome(message: str, *, code: str = "stream_error") -> TerminalOutcome:
    """Build a failed outcome for an exception while consuming a stream."""
    return TerminalOutcome(
        transport=TransportTermination.EXCEPTION,
        response_status=ResponseStatus.FAILED,
        error=TerminalError(code=code, message=message),
    )


def client_cancelled_outcome() -> TerminalOutcome:
    """Build the outcome recorded when downstream cancellation stops delivery."""
    return TerminalOutcome(
        transport=TransportTermination.CLIENT_CANCELLED,
        response_status=ResponseStatus.CANCELLED,
    )


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list | None  # Can be str/list for multimodal content, or null
    tool_call_id: str | None = None  # Required for tool role messages
    tool_calls: list[dict] | None = None  # For assistant messages with tool calls
    refusal: str | None = None  # OpenAI assistant refusal history


@dataclass
class ChatRequest:
    """Request for chat completion."""

    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict] | None = None  # OpenAI format tool definitions
    # "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}
    tool_choice: str | dict | None = None
    thinking_budget: int | None = None
    thinking_type: str | None = None  # "enabled", "adaptive", "disabled"
    # Ordered effort tier; "xhigh" and "max" are Router-Maestro extensions.
    reasoning_effort: str | None = None
    # Typed generation options shared by protocol entry routes. Adapters must
    # either encode these faithfully or reject them as a CLIENT_REQUEST.
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    user: str | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    service_tier: str | None = None
    # Gemini-only options are represented explicitly so a non-Gemini adapter
    # cannot silently discard them after translation.
    candidate_count: int | None = None
    response_mime_type: str | None = None
    # Experimental: when True, eligible providers (currently Copilot+gpt-5.x)
    # should fulfil this chat request via the /responses endpoint instead of
    # /chat/completions. Set by entry routes (Anthropic, Gemini) when the
    # ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API flag is on.
    use_responses_api: bool = False
    provider_extensions: dict[str, Any] = field(default_factory=dict)
    # Deprecated construction alias retained for third-party callers. Core
    # options are always sourced from typed fields and cannot be overridden.
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Promote known legacy extras into typed fields without overwrites."""
        legacy_extensions = dict(self.extra)
        core_fields = {
            item.name: item
            for item in fields(self)
            if item.name not in {"extra", "provider_extensions"}
        }
        for name, definition in core_fields.items():
            if name not in legacy_extensions:
                continue
            value = legacy_extensions.pop(name)
            current = getattr(self, name)
            if current == value:
                continue
            if definition.default is None and current is None:
                setattr(self, name, value)
                continue
            raise ValueError(f"Conflicting typed and legacy values for ChatRequest.{name}")
        extensions = dict(legacy_extensions)
        extensions.update(self.provider_extensions)
        self.provider_extensions = extensions
        self.extra = dict(extensions)
        self._validate_options()

    def _validate_options(self) -> None:
        """Apply the same invariants to direct and legacy construction."""
        if self.reasoning_effort is not None:
            from router_maestro.utils.reasoning import VALID_EFFORTS

            effort = self.reasoning_effort.lower()
            if effort not in VALID_EFFORTS:
                raise ValueError(f"Unsupported reasoning_effort '{self.reasoning_effort}'")
            self.reasoning_effort = effort

    def with_thinking(
        self,
        *,
        thinking_budget: int | None,
        thinking_type: str | None,
    ) -> "ChatRequest":
        """Return new ChatRequest with updated thinking parameters (immutable)."""
        return replace(
            self,
            thinking_budget=thinking_budget,
            thinking_type=thinking_type,
            extra={},
        )


@dataclass
class ChatResponse:
    """Response from chat completion."""

    content: str | None
    model: str
    finish_reason: str = "stop"
    usage: dict | None = None  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    tool_calls: list[dict] | None = None  # OpenAI-format tool calls from assistant
    # Reasoning trace (Anthropic "thinking" / OpenAI "reasoning_text" / Copilot "cot_summary")
    thinking: str | None = None
    thinking_signature: str | None = None
    # Upstream reasoning item id (e.g. ``rs_…``). The encrypted ``thinking_signature``
    # is signed against this id, so the pair must travel together — Copilot's
    # /responses rejects a blob paired with a mismatched id. Carried here so the
    # Responses→Chat bridge doesn't drop it (see responses_response_to_chat_response).
    thinking_id: str | None = None
    # OpenAI refusal output. Appended to preserve the legacy positional constructor.
    refusal: str | None = None
    # Router-selected identity. Providers leave this unset; Router attaches the
    # exact candidate so protocol boundaries never infer raw/public provenance.
    selected_model: "ModelRef | None" = None
    # Canonical provider semantics. Appended to preserve positional construction.
    terminal_outcome: TerminalOutcome | None = None


@dataclass
class ChatStreamChunk:
    """A chunk from streaming chat completion."""

    content: str
    finish_reason: str | None = None
    usage: dict | None = None  # Token usage info (typically in final chunk)
    tool_calls: list[dict] | None = None  # Tool call deltas for streaming
    thinking: str | None = None  # Incremental reasoning text delta
    thinking_signature: str | None = None  # Opaque/encrypted reasoning blob, if provided
    thinking_id: str | None = None  # Upstream reasoning item id the blob is signed against
    terminal_outcome: TerminalOutcome | None = None
    # OpenAI refusal delta. Appended to preserve the legacy positional constructor.
    refusal: str | None = None
    # Internal-only payload projection for forward-compatible protocol events.
    # Routes never emit this field; RunawayGuard counts its serialized bytes.
    opaque_payload: str | None = None


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
    # Per-model reasoning_effort allowlist as advertised by the upstream
    # catalog (Copilot's ``capabilities.supports.reasoning_effort``).
    # ``None`` means "the catalog didn't say" — callers should fall back
    # to a hardcoded heuristic. ``[]`` means "explicitly no reasoning".
    reasoning_effort_values: list[str] | None = None
    # Catalog capability declarations. Missing keys intentionally mean unknown.
    # ``None`` means the upstream omitted the endpoint contract; an empty tuple
    # means it explicitly advertised no supported endpoint.
    supported_endpoints: tuple[str, ...] | None = None
    operation_capabilities: dict[str, bool] = field(default_factory=dict)
    feature_capabilities: dict[str, bool] = field(default_factory=dict)
    # Catalog adapters normally return raw upstream IDs. Compatibility sources
    # that intentionally return a public ``provider/model`` ID must opt in so
    # namespaced upstream IDs are never guessed from string shape.
    id_is_qualified: bool = False

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
            reasoning_effort_values=self.reasoning_effort_values,
            supported_endpoints=self.supported_endpoints,
            operation_capabilities=self.operation_capabilities,
            feature_capabilities=self.feature_capabilities,
            id_is_qualified=self.id_is_qualified,
        )


@dataclass
class ResponsesToolCall:
    """A tool/function call from the Responses API."""

    call_id: str
    name: str
    arguments: str
    # Discriminates how the route emits this call to the downstream client:
    #   - "function"    → standard ``function_call`` item with JSON ``arguments``
    #   - "custom"      → ``custom_tool_call`` with raw ``input`` (e.g. apply_patch)
    #   - "tool_search" → ``tool_search_call`` with ``execution: "client"`` and a
    #                     dict ``arguments`` payload. Codex's MCP tool-discovery
    #                     dispatcher only matches this exact item type — wrapping
    #                     it as a function_call(name="tool_search") makes the
    #                     dispatcher silently abort the call (v0.3.5/0.3.6 bug).
    kind: Literal["function", "custom", "tool_search"] = "function"
    # MCP namespace, when the upstream emits one (e.g. Copilot CAPI's
    # ``kusto/execute_query`` → namespace="kusto"). Must round-trip back to
    # the upstream verbatim or the next turn 400s with
    # ``Missing namespace for function_call 'X'`` (v0.3.7 → v0.3.8 bug).
    namespace: str | None = None

    @property
    def is_custom(self) -> bool:
        return self.kind == "custom"


@dataclass
class ResponsesRequest:
    """Request for the Responses API (used by Codex models)."""

    model: str
    input: str | list  # Can be string or list of message dicts
    stream: bool = False
    instructions: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    # Tool support
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    top_p: float | None = None
    metadata: dict[str, Any] | None = None
    service_tier: str | None = None
    provider_extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponsesResponse:
    """Response from the Responses API."""

    content: str
    model: str
    usage: dict | None = None
    tool_calls: list[ResponsesToolCall] | None = None
    # Reasoning summary text aggregated from /responses' "reasoning" output items.
    thinking: str | None = None
    # Upstream reasoning item id (e.g. ``rs_…``). Must round-trip back to
    # Copilot as the reasoning item's ``id`` because the encrypted blob below
    # was signed against this id; pairing the blob with a different id 400s
    # with ``Encrypted content could not be decrypted``.
    thinking_id: str | None = None
    # Upstream encrypted reasoning blob (``encrypted_content``). Round-trips
    # alongside ``thinking_id`` so Copilot can verify and continue chain-of-
    # thought across turns.
    thinking_signature: str | None = None
    # Upstream completion status mapped to chat-style finish reason
    # ("stop" | "length" | "content_filter" | "tool_calls"). None means
    # the bridge should pick a default based on tool_calls presence.
    finish_reason: str | None = None
    # Canonical Responses terminal semantics. Native Responses adapters must
    # prefer this over the lossy chat-style finish_reason above.
    terminal_outcome: TerminalOutcome | None = None
    # OpenAI Responses refusal output, distinct from ordinary output_text.
    refusal: str | None = None
    # Router-selected identity; see ChatResponse.selected_model.
    selected_model: "ModelRef | None" = None


@dataclass
class ResponsesStreamChunk:
    """A chunk from streaming Responses API completion."""

    content: str
    finish_reason: str | None = None
    usage: dict | None = None
    # Tool call support
    tool_call: ResponsesToolCall | None = None  # A complete tool call
    # Incremental reasoning summary text delta (from
    # ``response.reasoning_summary_text.delta`` events).
    thinking: str | None = None
    # Upstream reasoning item id (carried separately from the encrypted blob —
    # see ResponsesResponse.thinking_id).
    thinking_id: str | None = None
    thinking_signature: str | None = None
    terminal_outcome: TerminalOutcome | None = None
    # OpenAI Responses refusal delta, distinct from ordinary output text.
    refusal: str | None = None
    # Canonical provenance from a native Responses output item. These fields
    # let protocol boundaries preserve source ordering even when upstream
    # items are interleaved. ``None`` keeps legacy/manual chunks unindexed.
    output_index: int | None = None
    content_index: int | None = None
    reasoning_summary_index: int | None = None
    # True when a native typed event carried only source provenance. Routes
    # validate its indices without materializing an empty user-visible part.
    provenance_only: bool = False
    output_item_type: str | None = None
    output_item_done: bool = False


class ProviderFailureKind(StrEnum):
    """Stable provider failure categories used across adapter boundaries."""

    TRANSPORT = "transport"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    UPSTREAM_STATUS = "upstream_status"
    UPSTREAM_PROTOCOL = "upstream_protocol"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    CLIENT_REQUEST = "client_request"
    UNKNOWN = "unknown"


class ProviderFailureSignal(StrEnum):
    """Closed, non-sensitive classifications for narrowly handled failures."""

    COPILOT_BARE_BAD_REQUEST = "copilot_bare_bad_request"


class ProviderError(Exception):
    """Safe, typed error raised by a provider adapter.

    The original three positional/keyword arguments remain supported for
    third-party providers. New first-party raises must set ``kind``.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        retryable: bool = False,
        *,
        kind: ProviderFailureKind = ProviderFailureKind.UNKNOWN,
        upstream_status_code: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        cause: BaseException | None = None,
        parameter: str | None = None,
        signal: ProviderFailureSignal | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_message = message
        self.status_code = status_code
        self.retryable = retryable
        self.kind = kind
        self.upstream_status_code = upstream_status_code
        self.provider = provider
        self.model = model
        self.cause = cause
        self.parameter = parameter
        self.signal = signal
        self._attempts: tuple[AttemptRecord, ...] = ()

    @property
    def downstream_status_code(self) -> int:
        """HTTP status suitable for returning to the downstream caller."""
        return self.status_code

    @property
    def attempts(self) -> tuple["AttemptRecord", ...]:
        """Read-only snapshot of routing attempts that led to this failure."""
        return self._attempts

    def with_attempts(self, attempts: tuple["AttemptRecord", ...]) -> Self:
        """Copy a ``__dict__``-backed error and attach a request-local snapshot."""
        error = BaseException.__new__(type(self), *self.args)
        error.__dict__ = self.__dict__.copy()
        error._attempts = tuple(attempts)
        return error


class RequestOptionError(ProviderError):
    """A client request option cannot be represented by the selected adapter."""

    def __init__(self, message: str, *, parameter: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            status_code=400,
            retryable=False,
            kind=ProviderFailureKind.CLIENT_REQUEST,
            parameter=parameter,
            **kwargs,
        )


class BaseProvider(ABC):
    """Abstract base class for model providers."""

    name: str = "base"

    _default_outbound_contract: OutboundContract = PermissiveOutboundContract()

    @property
    def outbound_contract(self) -> OutboundContract:
        """The provider's upstream wire contract (see OutboundContract)."""
        return self._default_outbound_contract

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Transport operations implemented by the base provider contract."""
        return ProviderCapabilities(operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM}))

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

    def _validate_provider_extensions(
        self,
        request: ChatRequest | ResponsesRequest,
        *,
        allowed: frozenset[str] = frozenset(),
    ) -> None:
        """Reject request-level provider extensions outside an explicit allowlist."""
        for key in request.provider_extensions:
            if key not in allowed:
                raise RequestOptionError(
                    f"{self.name} does not support provider extension '{key}'",
                    provider=self.name,
                    model=request.model,
                    parameter=f"provider_extensions.{key}",
                )

    def validate_chat_request(self, request: ChatRequest, *, stream: bool) -> None:
        """Validate options synchronously before a downstream stream commits."""

    def validate_responses_request(self, request: ResponsesRequest) -> None:
        """Validate Responses options before a downstream stream commits."""

    @staticmethod
    def _validated_response_model(data: dict[str, Any], fallback: str) -> str:
        """Return an upstream model after validating the consumed field."""
        if "model" not in data:
            return fallback
        model = data["model"]
        if not isinstance(model, str):
            raise TypeError("response model must be a string")
        return model

    @staticmethod
    def _validated_optional_string(
        data: dict[str, Any],
        field: str,
        *,
        default: str | None = None,
    ) -> str | None:
        """Return a nullable string field without inspecting unknown metadata."""
        if field not in data:
            return default
        value = data[field]
        if value is not None and not isinstance(value, str):
            raise TypeError(f"response {field} must be a string or null")
        return value

    @staticmethod
    def _validated_token_usage(
        usage: object,
        *,
        fields: tuple[str, ...],
        label: str,
        detail_fields: dict[str, tuple[str, ...]] | None = None,
    ) -> dict[str, Any] | None:
        """Validate only token-count fields consumed by downstream adapters."""
        if usage is None:
            return None
        if not isinstance(usage, dict):
            raise TypeError(f"{label} usage must be an object or null")
        for token_field in fields:
            if token_field not in usage:
                continue
            value = usage[token_field]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{label} usage {token_field} must be an integer")
        for detail_field, nested_fields in (detail_fields or {}).items():
            if detail_field not in usage:
                continue
            details = usage[detail_field]
            if details is None:
                continue
            if not isinstance(details, dict):
                raise TypeError(f"{label} usage {detail_field} must be an object or null")
            for nested_field in nested_fields:
                if nested_field not in details:
                    continue
                value = details[nested_field]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(
                        f"{label} usage {detail_field}.{nested_field} must be an integer"
                    )
        return usage

    @staticmethod
    def _raise_protocol_error(
        provider: str,
        model: str | None,
        cause: BaseException,
        *,
        upstream_status_code: int = 200,
    ) -> NoReturn:
        """Convert an upstream parser/schema failure to a safe provider error."""
        raise ProviderError(
            f"{provider} returned a malformed upstream response",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            upstream_status_code=upstream_status_code,
            provider=provider,
            model=model,
            cause=cause,
        ) from cause

    @staticmethod
    def _raise_http_status_error(
        label: str,
        error: httpx.HTTPStatusError,
        logger: Logger,
        *,
        stream: bool = False,
        include_body: bool = False,
        provider: str | None = None,
        model: str | None = None,
        signal: ProviderFailureSignal | None = None,
    ) -> NoReturn:
        """Raise a ProviderError from an HTTP status error.

        Args:
            label: Provider label for log messages
            error: The httpx status error
            logger: Logger instance for error logging
            stream: Whether this is a streaming request
            include_body: Whether to record only the response body size in provider logs
        """
        status_code = error.response.status_code
        retryable = status_code in (429, 500, 502, 503, 504, 529)
        if status_code in (401, 403):
            kind = ProviderFailureKind.AUTHENTICATION
        elif status_code in (429, 529):
            kind = ProviderFailureKind.RATE_LIMIT
        else:
            kind = ProviderFailureKind.UPSTREAM_STATUS
        suffix = " stream" if stream else ""
        if include_body:
            try:
                error_body = error.response.text
            except httpx.ResponseNotRead:
                error_body = ""
            except Exception:
                error_body = ""
            logger.error(
                "%s%s API error: %d (body_bytes=%d)",
                label,
                suffix,
                status_code,
                len(error_body.encode("utf-8", errors="replace")),
            )
            raise ProviderError(
                f"{label} API error: {status_code}",
                status_code=status_code,
                retryable=retryable,
                kind=kind,
                upstream_status_code=status_code,
                provider=provider or label,
                model=model,
                cause=error,
                signal=signal,
            ) from error
        logger.error("%s%s API error: %d", label, suffix, error.response.status_code)
        raise ProviderError(
            f"{label} API error: {status_code}",
            status_code=status_code,
            retryable=retryable,
            kind=kind,
            upstream_status_code=status_code,
            provider=provider or label,
            model=model,
            cause=error,
            signal=signal,
        ) from error

    @staticmethod
    def _raise_timeout_error(
        label: str,
        error: httpx.TimeoutException,
        logger: Logger,
        *,
        stream: bool = False,
        provider: str | None = None,
        model: str | None = None,
    ) -> NoReturn:
        """Raise a ProviderError from an httpx timeout.

        Args:
            label: Provider label for log messages
            error: The httpx timeout exception
            logger: Logger instance for error logging
            stream: Whether this is a streaming request
        """
        timeout_type = type(error).__name__
        suffix = " stream" if stream else ""
        logger.error("%s%s timed out (%s)", label, suffix, timeout_type)
        raise ProviderError(
            f"{label} timed out ({timeout_type})",
            status_code=504,
            retryable=True,
            kind=ProviderFailureKind.TRANSPORT,
            provider=provider or label,
            model=model,
            cause=error,
        ) from error

    @staticmethod
    def _raise_http_error(
        label: str,
        error: httpx.HTTPError,
        logger: Logger,
        *,
        stream: bool = False,
        provider: str | None = None,
        model: str | None = None,
    ) -> NoReturn:
        """Raise a ProviderError from a generic HTTP error.

        Args:
            label: Provider label for log messages
            error: The httpx error
            logger: Logger instance for error logging
            stream: Whether this is a streaming request
        """
        suffix = " stream" if stream else ""
        logger.error("%s%s HTTP error (%s)", label, suffix, type(error).__name__)
        raise ProviderError(
            f"{label} transport failed ({type(error).__name__})",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.TRANSPORT,
            provider=provider or label,
            model=model,
            cause=error,
        ) from error

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        """Generate a Responses API completion (for Codex models).

        Args:
            request: Responses completion request

        Returns:
            Responses completion response

        Raises:
            ProviderError: If provider does not support Responses API
        """
        raise ProviderError(
            "Provider does not support Responses API",
            status_code=501,
            retryable=False,
            kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
            provider=self.name,
            model=request.model,
        )

    async def responses_completion_stream(
        self, request: ResponsesRequest
    ) -> AsyncIterator[ResponsesStreamChunk]:
        """Generate a streaming Responses API completion (for Codex models).

        Args:
            request: Responses completion request

        Yields:
            Responses completion chunks

        Raises:
            ProviderError: If provider does not support Responses API
        """
        raise ProviderError(
            "Provider does not support Responses API",
            status_code=501,
            retryable=False,
            kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
            provider=self.name,
            model=request.model,
        )
        # Make this a generator (required for type checking)
        if False:
            yield ResponsesStreamChunk(content="")
