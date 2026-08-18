"""Immutable selection of one provider transport flow.

This module deliberately contains no payload and performs no fallback
execution. A dispatcher may create plans lazily for route candidates without
preparing or mutating any outbound request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from router_maestro.protocols import ConversionMode, WireProtocol
from router_maestro.providers.bindings import EndpointBinding
from router_maestro.routing.model_ref import ModelRef

if TYPE_CHECKING:
    from router_maestro.providers.base import BaseProvider


@dataclass(frozen=True, slots=True)
class FlowCandidate:
    """A candidate path from one ingress protocol to one upstream binding."""

    source_protocol: WireProtocol
    binding: EndpointBinding
    conversion_mode: ConversionMode

    def __post_init__(self) -> None:
        expected_mode = (
            ConversionMode.IDENTITY
            if self.source_protocol is self.binding.protocol
            else ConversionMode.SEMANTIC_IR
        )
        if self.conversion_mode is not expected_mode:
            raise ValueError(
                f"{self.source_protocol.value} -> {self.binding.protocol.value} "
                f"requires {expected_mode.value} conversion"
            )

    @classmethod
    def for_binding(
        cls,
        *,
        source_protocol: WireProtocol,
        binding: EndpointBinding,
    ) -> FlowCandidate:
        """Choose identity or semantic-IR conversion from protocol equality."""
        conversion_mode = (
            ConversionMode.IDENTITY
            if source_protocol is binding.protocol
            else ConversionMode.SEMANTIC_IR
        )
        return cls(
            source_protocol=source_protocol,
            binding=binding,
            conversion_mode=conversion_mode,
        )

    @property
    def target_protocol(self) -> WireProtocol:
        """The upstream wire protocol selected by this flow."""
        return self.binding.protocol


@dataclass(frozen=True, slots=True)
class TransportPlan:
    """One model/provider selection bound to one transport flow."""

    model: ModelRef
    provider: BaseProvider
    candidate: FlowCandidate

    def __post_init__(self) -> None:
        if getattr(self.provider, "name", None) != self.model.provider:
            raise ValueError("transport provider must match the selected model provider")

    @property
    def binding(self) -> EndpointBinding:
        return self.candidate.binding

    @property
    def flow(self) -> FlowCandidate:
        """Compatibility alias for callers that describe a candidate as a flow."""
        return self.candidate

    @property
    def source_protocol(self) -> WireProtocol:
        return self.candidate.source_protocol

    @property
    def target_protocol(self) -> WireProtocol:
        return self.candidate.target_protocol

    @property
    def conversion_mode(self) -> ConversionMode:
        return self.candidate.conversion_mode
