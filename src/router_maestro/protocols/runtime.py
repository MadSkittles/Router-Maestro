"""Protocol runtime contracts and their thread-safe registry."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from router_maestro.protocols.models import (
    OpaqueState,
    RepresentabilityReport,
    RequestManifest,
    SemanticEvent,
    SemanticRequest,
    SemanticResponse,
    WireProtocol,
)


@runtime_checkable
class OpaqueStateDecodeHook(Protocol):
    """Unseal one Router-Maestro wire capsule with dispatcher-owned context."""

    def __call__(
        self,
        value: str,
        *,
        protocol: WireProtocol,
        model: str,
        item_id: str,
    ) -> OpaqueState: ...


@runtime_checkable
class OpaqueStateEncodeHook(Protocol):
    """Seal provider-owned state for exposure through a foreign wire protocol."""

    def __call__(
        self,
        state: OpaqueState,
        *,
        protocol: WireProtocol,
        model: str,
        item_id: str,
    ) -> str: ...


class UnsupportedProtocolOperationError(NotImplementedError):
    """A runtime does not yet implement one semantic conversion operation."""

    def __init__(self, protocol: WireProtocol, operation: str) -> None:
        self.protocol = protocol
        self.operation = operation
        super().__init__(f"{protocol.value} runtime does not support {operation}")


class ProtocolRepresentabilityError(ValueError):
    """Semantic IR cannot be represented faithfully by a target protocol."""

    def __init__(
        self,
        protocol: WireProtocol,
        path: str,
        reason: str,
        *,
        report: RepresentabilityReport | None = None,
    ) -> None:
        self.protocol = protocol
        self.path = path
        self.parameter = path
        self.reason = reason
        self.report = report or RepresentabilityReport(
            representable=False, reasons=(reason,), parameter=path
        )
        super().__init__(f"{protocol.value} cannot represent {path}: {reason}")


class ProtocolDecodeError(ValueError):
    """A wire value is malformed or uses an unsupported explicit extension."""

    def __init__(self, protocol: WireProtocol, path: str, reason: str) -> None:
        self.protocol = protocol
        self.path = path
        self.reason = reason
        super().__init__(f"invalid {protocol.value} value at {path}: {reason}")


@runtime_checkable
class ProtocolRuntime(Protocol):
    """One protocol's cheap inspector and semantic conversion boundary."""

    protocol: WireProtocol

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        """Inspect routing facts without materializing semantic IR."""
        ...

    async def decode_request(self, payload: Mapping[str, Any]) -> SemanticRequest:
        """Decode an ingress request when a cross-protocol path requires IR."""
        ...

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        raise UnsupportedProtocolOperationError(self.protocol, "encode_request")

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        raise UnsupportedProtocolOperationError(self.protocol, "decode_response")

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        raise UnsupportedProtocolOperationError(self.protocol, "encode_response")

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        raise UnsupportedProtocolOperationError(self.protocol, "decode_stream_event")

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        raise UnsupportedProtocolOperationError(self.protocol, "encode_stream_event")


async def check_request_representability(
    runtime: ProtocolRuntime,
    request: SemanticRequest,
) -> RepresentabilityReport:
    """Run an optional cheap preflight without expanding the structural protocol.

    Concrete runtimes may expose ``request_representability`` for checks that do
    not encode or materialize another IR.  The default deliberately reports
    exact: the one-pass ``encode_request`` call remains authoritative, and its
    typed exception carries the final report.
    """
    checker = getattr(runtime, "request_representability", None)
    if checker is None:
        return RepresentabilityReport(representable=True)
    report = checker(request)
    if inspect.isawaitable(report):
        report = await report
    if not isinstance(report, RepresentabilityReport):
        raise TypeError("runtime request_representability must return RepresentabilityReport")
    return report


class DuplicateProtocolRuntimeError(ValueError):
    def __init__(self, protocol: WireProtocol) -> None:
        self.protocol = protocol
        super().__init__(f"runtime already registered for {protocol.value}")


class ProtocolRuntimeNotFoundError(LookupError):
    def __init__(self, protocol: WireProtocol) -> None:
        self.protocol = protocol
        super().__init__(f"no runtime registered for {protocol.value}")


class ProtocolRuntimeRegistry:
    """A small, thread-safe registry keyed by wire protocol."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runtimes: dict[WireProtocol, ProtocolRuntime] = {}

    def register(self, runtime: ProtocolRuntime) -> None:
        protocol = runtime.protocol
        if not isinstance(protocol, WireProtocol):
            raise TypeError("runtime.protocol must be a WireProtocol")
        with self._lock:
            if protocol in self._runtimes:
                raise DuplicateProtocolRuntimeError(protocol)
            self._runtimes[protocol] = runtime

    def replace(self, runtime: ProtocolRuntime) -> ProtocolRuntime:
        """Explicitly replace an existing registration and return the old runtime."""
        protocol = runtime.protocol
        if not isinstance(protocol, WireProtocol):
            raise TypeError("runtime.protocol must be a WireProtocol")
        with self._lock:
            if protocol not in self._runtimes:
                raise ProtocolRuntimeNotFoundError(protocol)
            previous = self._runtimes[protocol]
            self._runtimes[protocol] = runtime
            return previous

    def get(self, protocol: WireProtocol) -> ProtocolRuntime:
        with self._lock:
            try:
                return self._runtimes[protocol]
            except KeyError:
                raise ProtocolRuntimeNotFoundError(protocol) from None

    def snapshot(self) -> Mapping[WireProtocol, ProtocolRuntime]:
        with self._lock:
            return MappingProxyType(dict(self._runtimes))

    def __contains__(self, protocol: object) -> bool:
        with self._lock:
            return protocol in self._runtimes

    def __len__(self) -> int:
        with self._lock:
            return len(self._runtimes)
