"""Protocol-neutral semantic models used only when wire conversion is required."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias


class WireProtocol(StrEnum):
    """Wire formats understood by Router-Maestro protocol runtimes."""

    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    GEMINI = "gemini"

    # Compatibility shorthand. New code should use ANTHROPIC_MESSAGES.
    ANTHROPIC = ANTHROPIC_MESSAGES


class ConversionMode(StrEnum):
    """How a request is transported between its ingress and provider protocols."""

    IDENTITY = "identity"
    SEMANTIC_IR = "semantic_ir"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SemanticEventType(StrEnum):
    RESPONSE_STARTED = "response_started"
    OUTPUT_ITEM = "output_item"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_ARGUMENTS_DELTA = "tool_arguments_delta"
    USAGE = "usage"
    TERMINAL = "terminal"
    ERROR = "error"


class UsageMode(StrEnum):
    """Whether usage values replace prior totals or increment them."""

    SNAPSHOT = "snapshot"
    DELTA = "delta"


JsonScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040 - Python 3.11
FrozenJsonValue: TypeAlias = (  # noqa: UP040 - Python 3.11
    JsonScalar | tuple["FrozenJsonValue", ...] | Mapping[str, "FrozenJsonValue"]
)


def _freeze_json(value: Any) -> FrozenJsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("semantic JSON object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"unsupported semantic JSON value: {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, FrozenJsonValue]:
    frozen = _freeze_json(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by the input type
        raise TypeError("expected a semantic JSON object")
    return frozen


@dataclass(frozen=True, slots=True)
class RequestManifest:
    """Cheap request facts that a runtime can inspect without building semantic IR."""

    protocol: WireProtocol
    model: str | None = None
    stream: bool = False
    tools: bool = False
    images: bool = False
    files: bool = False
    reasoning: bool = False
    # Cheap continuation hints used to freeze routing before provider I/O.
    # Runtimes only identify RM carrier strings here; authentication remains a
    # dispatcher responsibility.
    reasoning_capsules: tuple[str, ...] = ()
    previous_response_id: str | None = None
    opaque_continuation: bool = False
    # Appended to preserve compatibility for callers using positional fields.
    parallel_tools: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasoning_capsules", tuple(self.reasoning_capsules))


@dataclass(frozen=True, slots=True)
class RepresentabilityReport:
    """Whether a semantic value can be encoded in a target wire protocol."""

    representable: bool
    lossy: bool = False
    reasons: tuple[str, ...] = ()
    parameter: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def is_exact(self) -> bool:
        return self.representable and not self.lossy


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str


@dataclass(frozen=True, slots=True)
class RefusalContent:
    """A model refusal kept distinct from ordinary assistant text."""

    refusal: str


@dataclass(frozen=True, slots=True)
class ImageContent:
    """An image URL, file reference, or immutable encoded payload."""

    source: str | bytes
    media_type: str | None = None
    detail: str | None = None
    source_kind: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source, bytearray | memoryview):
            object.__setattr__(self, "source", bytes(self.source))


@dataclass(frozen=True, slots=True)
class FileContent:
    """A file URL, file reference, or immutable encoded payload."""

    source: str | bytes
    filename: str | None = None
    media_type: str | None = None
    source_kind: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source, bytearray | memoryview):
            object.__setattr__(self, "source", bytes(self.source))


@dataclass(frozen=True, slots=True)
class OpaqueState:
    """Provider state that must never be detached from its complete provenance."""

    origin_protocol: WireProtocol
    # A protocol runtime may preserve opaque state before the dispatcher has
    # attached concrete provider provenance.  It cannot be replayed or sealed
    # into an RM capsule while this remains unset.
    origin_provider: str | None
    origin_model: str
    item_id: str
    blob: FrozenJsonValue | bytes
    origin_binding: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.blob, bytearray | memoryview):
            object.__setattr__(self, "blob", bytes(self.blob))
        elif isinstance(self.blob, Mapping | list | tuple):
            object.__setattr__(self, "blob", _freeze_json(self.blob))


@dataclass(frozen=True, slots=True)
class ReasoningSummary:
    text: str
    opaque_state: OpaqueState | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    input_schema: Mapping[str, FrozenJsonValue]
    description: str | None = None
    strict: bool | None = None
    namespace: str | None = None
    namespace_description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", _freeze_mapping(self.input_schema))


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    item_id: str | None = None
    kind: str = "function"
    namespace: str | None = None
    opaque_state: OpaqueState | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))

    @property
    def id(self) -> str:
        """Compatibility view for protocols that expose only one call identifier."""
        return self.call_id


ContentBlock: TypeAlias = (  # noqa: UP040 - Python 3.11
    TextContent | RefusalContent | ImageContent | FileContent
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    content: tuple[ContentBlock, ...] = ()
    structured_content: FrozenJsonValue = None
    is_error: bool = False
    item_id: str | None = None
    kind: str = "function"
    namespace: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", tuple(self.content))
        object.__setattr__(self, "structured_content", _freeze_json(self.structured_content))


MessageContent: TypeAlias = (  # noqa: UP040 - Python 3.11
    ContentBlock | ReasoningSummary | ToolCall | ToolResult
)


@dataclass(frozen=True, slots=True)
class SemanticMessage:
    role: MessageRole
    content: tuple[MessageContent, ...]
    name: str | None = None
    item_id: str | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", tuple(self.content))


SemanticItem: TypeAlias = SemanticMessage | MessageContent  # noqa: UP040 - Python 3.11


@dataclass(frozen=True, slots=True)
class ToolChoice:
    mode: str
    name: str | None = None
    namespace: str | None = None


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    enabled: bool | None = None
    effort: str | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.budget_tokens is not None and self.budget_tokens < 0:
            raise ValueError("reasoning budget_tokens cannot be negative")


@dataclass(frozen=True, slots=True)
class Usage:
    """Usage values with explicit missing/zero and delta/snapshot semantics."""

    mode: UsageMode = UsageMode.SNAPSHOT
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        token_counts = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.cached_input_tokens,
            self.reasoning_tokens,
        )
        if any(count is not None and count < 0 for count in token_counts):
            raise ValueError("usage token counts cannot be negative")


@dataclass(frozen=True, slots=True)
class TerminalMetadata:
    finish_reason: str | None = None
    stop_sequence: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    response_status: str | None = None
    transport_termination: str | None = None
    incomplete_details: Mapping[str, FrozenJsonValue] | None = None
    transport_status: int | None = None

    def __post_init__(self) -> None:
        if self.incomplete_details is not None:
            object.__setattr__(
                self,
                "incomplete_details",
                _freeze_mapping(self.incomplete_details),
            )


@dataclass(frozen=True, slots=True)
class SemanticRequest:
    model: str
    input: tuple[SemanticItem, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()
    stream: bool = False
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    candidate_count: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop_sequences: tuple[str, ...] = ()
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    reasoning: ReasoningConfig | None = None
    structured_output: Mapping[str, FrozenJsonValue] | None = None
    response_mime_type: str | None = None
    user: str | None = None
    service_tier: str | None = None
    metadata: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    provider_extensions: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    explicit_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", tuple(self.input))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))
        if self.structured_output is not None:
            object.__setattr__(
                self,
                "structured_output",
                _freeze_mapping(self.structured_output),
            )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(
            self,
            "provider_extensions",
            _freeze_mapping(self.provider_extensions),
        )
        object.__setattr__(self, "explicit_fields", frozenset(self.explicit_fields))


@dataclass(frozen=True, slots=True)
class SemanticResponse:
    model: str
    id: str | None = None
    output: tuple[SemanticItem, ...] = ()
    usage: Usage | None = None
    terminal: TerminalMetadata | None = None
    metadata: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", tuple(self.output))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    type: SemanticEventType
    sequence: int | None = None
    response_id: str | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None
    item: SemanticItem | None = None
    delta: str | None = None
    usage: Usage | None = None
    terminal: TerminalMetadata | None = None
    metadata: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
