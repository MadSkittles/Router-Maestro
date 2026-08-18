"""Provider endpoint bindings and generic HTTP execution contracts.

Bindings describe *where* a provider can accept a request. Protocol runtimes
remain responsible for wire decoding/encoding, while provider dialects add the
small provider-specific pieces needed to prepare an HTTP attempt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from router_maestro.protocols import WireProtocol
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef

LEGACY_OPENAI_CHAT_BINDING = "legacy-openai-chat"
LEGACY_OPENAI_RESPONSES_BINDING = "legacy-openai-responses"
LEGACY_ANTHROPIC_MESSAGES_BINDING = "legacy-anthropic-messages"

COPILOT_ANTHROPIC_MESSAGES_BINDING = "copilot-anthropic-messages"
COPILOT_OPENAI_CHAT_BINDING = "copilot-openai-chat"
COPILOT_OPENAI_RESPONSES_BINDING = "copilot-openai-responses"
OPENAI_COMPATIBLE_CHAT_BINDING = "openai-compatible-chat"
_EMPTY_STRING_MAPPING: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True, init=False)
class AttemptRequestContext:
    """Immutable ingress metadata available to a provider dialect.

    Dialects must opt in to forwarding individual values. Raw client
    authentication headers are context only and are never copied by the shared
    binding layer.
    """

    path: str
    query: Mapping[str, str]
    headers: Mapping[str, str]

    def __init__(
        self,
        path: str = "",
        query: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        _mappings_owned: bool = False,
    ) -> None:
        if not isinstance(path, str):
            raise TypeError("attempt request path must be a string")
        query_snapshot = (
            query
            if _mappings_owned and isinstance(query, dict)
            else dict(query)
            if query is not None
            else {}
        )
        header_snapshot = (
            headers
            if _mappings_owned and isinstance(headers, dict)
            else dict(headers)
            if headers is not None
            else {}
        )
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in query_snapshot.items()
        ) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in header_snapshot.items()
        ):
            raise TypeError("attempt request context must contain only string keys and values")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "query",
            MappingProxyType(query_snapshot) if query_snapshot else _EMPTY_STRING_MAPPING,
        )
        object.__setattr__(
            self,
            "headers",
            MappingProxyType(header_snapshot) if header_snapshot else _EMPTY_STRING_MAPPING,
        )

    def header(self, name: str) -> str | None:
        """Return one case-insensitive ingress header value."""
        lowered = name.lower()
        return next((value for key, value in self.headers.items() if key.lower() == lowered), None)


@dataclass(frozen=True, slots=True)
class PreparedAttempt:
    """One fully prepared outbound HTTP attempt.

    Provider-owned payloads are frozen at the top level after dialects copy
    every branch they mutate. Untrusted mapping inputs receive a defensive deep
    snapshot. Headers are always copied before they are exposed as read-only.
    """

    binding_id: str
    protocol: WireProtocol
    model: ModelRef
    url: str
    payload: Mapping[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)
    stream: bool = False
    method: str = "POST"
    _payload_owned: InitVar[bool] = False

    def __post_init__(self, _payload_owned: bool) -> None:
        binding_id = _validated_identifier(self.binding_id, label="binding ID")
        url = _validated_identifier(self.url, label="attempt URL")
        method = _validated_identifier(self.method, label="HTTP method").upper()

        headers = dict(self.headers)
        if any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
        ):
            raise TypeError("attempt headers must contain only string keys and values")

        object.__setattr__(self, "binding_id", binding_id)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "method", method)
        payload = self.payload if _payload_owned and isinstance(self.payload, dict) else None
        if payload is None:
            payload = deepcopy(dict(self.payload))
        object.__setattr__(self, "payload", MappingProxyType(payload))
        object.__setattr__(self, "headers", MappingProxyType(headers))


@runtime_checkable
class ProviderDialect(Protocol):
    """Provider-specific hook that turns encoded JSON into an HTTP attempt."""

    @property
    def id(self) -> str:
        """Stable dialect identifier used for diagnostics."""
        ...

    async def prepare_attempt(
        self,
        *,
        binding_id: str,
        protocol: WireProtocol,
        model: ModelRef,
        payload: Mapping[str, Any],
        stream: bool,
        request_context: AttemptRequestContext,
    ) -> PreparedAttempt:
        """Add provider URL/auth details to an already encoded wire payload."""
        ...


@runtime_checkable
class HttpExecutor(Protocol):
    """Transport-only executor shared by provider endpoint bindings."""

    async def execute(self, attempt: PreparedAttempt) -> Any:
        """Execute one non-streaming attempt and return the raw response."""
        ...

    def execute_stream(self, attempt: PreparedAttempt) -> AsyncIterator[Any]:
        """Execute one streaming attempt and yield raw transport frames."""
        ...


@dataclass(frozen=True, slots=True)
class EndpointBinding:
    """One provider endpoint bound to a wire protocol and transport hooks."""

    id: str
    protocol: WireProtocol
    capabilities: ProviderCapabilities
    dialect: ProviderDialect | None = None
    executor: HttpExecutor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _validated_identifier(self.id, label="binding ID"))
        if (self.dialect is None) is not (self.executor is None):
            raise ValueError(
                "binding dialect and executor must either both be set or both be omitted"
            )

    @property
    def is_legacy(self) -> bool:
        """Whether this binding delegates through the pre-binding provider API."""
        return self.dialect is None

    def supports(self, operation: Operation) -> bool:
        """Return whether this endpoint exposes one legacy operation."""
        return self.capabilities.supports(operation)

    async def prepare_attempt(
        self,
        *,
        model: ModelRef,
        payload: Mapping[str, Any],
        stream: bool,
        request_context: AttemptRequestContext | None = None,
    ) -> PreparedAttempt:
        """Prepare and validate an attempt through this binding's dialect."""
        if self.dialect is None:
            raise RuntimeError(f"legacy binding {self.id!r} has no generic HTTP dialect")

        attempt = await self.dialect.prepare_attempt(
            binding_id=self.id,
            protocol=self.protocol,
            model=model,
            payload=payload,
            stream=stream,
            request_context=request_context or AttemptRequestContext(),
        )
        if attempt.binding_id != self.id:
            raise ValueError("prepared attempt binding ID does not match its endpoint binding")
        if attempt.protocol is not self.protocol:
            raise ValueError("prepared attempt protocol does not match its endpoint binding")
        if attempt.model != model:
            raise ValueError("prepared attempt model does not match the requested model")
        if attempt.stream is not stream:
            raise ValueError("prepared attempt stream mode does not match the request")
        return attempt


def legacy_endpoint_binding(
    *,
    binding_id: str,
    protocol: WireProtocol,
    operations: frozenset[Operation],
) -> EndpointBinding:
    """Build a metadata-only binding for the existing BaseProvider methods."""
    if not operations:
        raise ValueError("legacy endpoint binding requires at least one operation")
    return EndpointBinding(
        id=binding_id,
        protocol=protocol,
        capabilities=ProviderCapabilities(operations=operations),
    )


def _validated_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain leading or trailing whitespace")
    return value
