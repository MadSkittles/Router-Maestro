"""Request-scoped protocol runtimes and reasoning-capsule hooks.

The protocol codecs are intentionally provider neutral.  This factory attaches
the small amount of request-local provenance they need without putting routing
state on shared runtime singletons.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    GeminiRuntime,
    OpaqueState,
    ProtocolRuntime,
    WireProtocol,
)
from router_maestro.protocols.openai_chat import OpenAIChatRuntime
from router_maestro.protocols.openai_responses import OpenAIResponsesRuntime
from router_maestro.providers.bindings import COPILOT_OPENAI_RESPONSES_BINDING
from router_maestro.runtime.reasoning_capsule import (
    ReasoningCapsuleCodec,
    ReasoningCapsuleError,
    ReasoningCapsulePayload,
    deserialize_opaque_state,
    serialize_opaque_state,
)

if TYPE_CHECKING:
    from router_maestro.routing.router import Router
    from router_maestro.routing.transport_plan import TransportPlan


class ProtocolRuntimeFactory:
    """Create ingress and provider-bound runtimes for one Router generation."""

    def __init__(
        self,
        capsule_codec: ReasoningCapsuleCodec,
        binding_protocols: Mapping[tuple[str, str], WireProtocol],
    ) -> None:
        self._capsule_codec = capsule_codec
        self._binding_protocols = dict(binding_protocols)

    @classmethod
    def for_router(
        cls,
        router: Router,
        capsule_codec: ReasoningCapsuleCodec,
    ) -> ProtocolRuntimeFactory:
        """Snapshot binding provenance without invoking model discovery."""
        bindings: dict[tuple[str, str], WireProtocol] = {}
        for provider_name, provider in router.providers.items():
            for binding in provider.bindings():
                key = (provider_name, binding.id)
                previous = bindings.get(key)
                if previous is not None and previous is not binding.protocol:
                    raise ValueError(
                        f"provider {provider_name!r} reuses binding {binding.id!r} "
                        "for multiple protocols"
                    )
                bindings[key] = binding.protocol
        return cls(capsule_codec, bindings)

    def ingress(
        self,
        protocol: WireProtocol,
        *,
        model: str | None = None,
        stream: bool = False,
    ) -> ProtocolRuntime:
        """Build the downstream runtime before any provider is selected."""
        return self._build(
            protocol,
            model=model,
            stream=stream,
            provider=None,
            binding=None,
        )

    def for_transport(self, plan: TransportPlan) -> ProtocolRuntime:
        """Build a target runtime frozen to one provider/model binding."""
        return self._build(
            plan.target_protocol,
            model=plan.model.upstream_id,
            stream=False,
            provider=plan.model.provider,
            binding=plan.binding.id,
        )

    def resolve(
        self,
        protocol: WireProtocol,
        plan: TransportPlan | None = None,
    ) -> ProtocolRuntime:
        """Dispatcher-compatible resolver with a provider-bound target path."""
        if plan is None:
            return self.ingress(protocol)
        if plan.target_protocol is not protocol:
            raise ValueError("transport protocol does not match requested runtime")
        return self.for_transport(plan)

    def _build(
        self,
        protocol: WireProtocol,
        *,
        model: str | None,
        stream: bool,
        provider: str | None,
        binding: str | None,
    ) -> ProtocolRuntime:
        if protocol is WireProtocol.ANTHROPIC_MESSAGES:
            return AnthropicMessagesRuntime(
                origin_provider=provider,
                decode_opaque_state=self.decode_opaque_state,
                encode_opaque_state=self.encode_opaque_state,
            )
        if protocol is WireProtocol.OPENAI_CHAT:
            return OpenAIChatRuntime(
                origin_provider=provider,
                default_model=model,
            )
        if protocol is WireProtocol.OPENAI_RESPONSES:
            copilot_obfuscated_stream_ids = (
                provider == "github-copilot" and binding == COPILOT_OPENAI_RESPONSES_BINDING
            )
            return OpenAIResponsesRuntime(
                provider_name=provider,
                binding_id=binding,
                allow_per_event_response_ids=copilot_obfuscated_stream_ids,
                defer_intermediate_item_ids=copilot_obfuscated_stream_ids,
            )
        if protocol is WireProtocol.GEMINI:
            return GeminiRuntime(
                default_model=model,
                stream=stream,
                origin_provider=provider,
                decode_opaque_state=self.decode_opaque_state,
                encode_opaque_state=self.encode_opaque_state,
            )
        raise ValueError(f"unsupported wire protocol {protocol!r}")

    def decode_opaque_state(
        self,
        value: str,
        *,
        protocol: WireProtocol,
        model: str,
        item_id: str,
    ) -> OpaqueState:
        """Authenticate a foreign carrier and restore its original provenance."""
        del protocol, model, item_id
        try:
            payload = self._capsule_codec.unseal_for_routing(value)
            origin_protocol = self._binding_protocols[(payload.provider, payload.transport)]
            blob = deserialize_opaque_state(payload.opaque_state)
        except (KeyError, ReasoningCapsuleError, TypeError, ValueError):
            raise ValueError("Invalid reasoning capsule") from None
        return OpaqueState(
            origin_protocol=origin_protocol,
            origin_provider=payload.provider,
            origin_model=payload.model,
            item_id=payload.item_id,
            blob=blob,
            origin_binding=payload.transport,
        )

    def encode_opaque_state(
        self,
        state: OpaqueState,
        *,
        protocol: WireProtocol,
        model: str,
        item_id: str,
    ) -> str:
        """Seal provider-owned state for an Anthropic or Gemini carrier."""
        del protocol
        provider = state.origin_provider
        binding = state.origin_binding
        if (
            provider is None
            or binding is None
            or not item_id
            or item_id != state.item_id
            or model != state.origin_model
            or self._binding_protocols.get((provider, binding)) is not state.origin_protocol
        ):
            raise ValueError("opaque reasoning provenance is incomplete")
        return self._capsule_codec.seal(
            ReasoningCapsulePayload(
                provider=provider,
                model=state.origin_model,
                transport=binding,
                item_id=state.item_id,
                opaque_state=serialize_opaque_state(state.blob),
            )
        )


__all__ = ["ProtocolRuntimeFactory"]
