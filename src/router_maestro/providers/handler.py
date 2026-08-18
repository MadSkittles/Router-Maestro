"""Provider-owned transport selection for the generation dispatcher."""

from __future__ import annotations

from dataclasses import dataclass

from router_maestro.protocols import RequestManifest, WireProtocol
from router_maestro.providers.base import BaseProvider, ModelInfo
from router_maestro.providers.bindings import EndpointBinding
from router_maestro.routing.capabilities import model_supports_manifest
from router_maestro.routing.generation_plan import GenerationCandidate
from router_maestro.routing.transport_plan import FlowCandidate, TransportPlan

_ENDPOINT_PROTOCOLS = {
    "/chat/completions": WireProtocol.OPENAI_CHAT,
    "/responses": WireProtocol.OPENAI_RESPONSES,
}


@dataclass(frozen=True, slots=True)
class ProviderHandler:
    """Collect one provider's catalog and endpoint selection policy.

    URL/auth/header quirks remain on concrete dialects/providers; the Router
    never needs to know that Copilot uses ``/responses`` or ``/v1/messages``.
    """

    provider: BaseProvider

    def bindings_for(
        self,
        candidate: GenerationCandidate,
        ingress_protocol: WireProtocol,
        manifest: RequestManifest | None = None,
    ) -> tuple[TransportPlan, ...]:
        if candidate.provider is not self.provider:
            raise ValueError("generation candidate belongs to another provider handler")
        if manifest is not None and manifest.protocol is not ingress_protocol:
            raise ValueError("request manifest protocol must match the ingress protocol")

        declared_bindings = self.provider.bindings()
        binding_ids = [binding.id for binding in declared_bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError(f"provider {self.provider.name!r} declares duplicate binding IDs")
        bindings = dict(zip(binding_ids, declared_bindings, strict=True))
        preference = self.provider.transport_preferences(ingress_protocol)
        if len(preference) != len(set(preference)):
            raise ValueError(
                f"provider {self.provider.name!r} declares duplicate transport preferences"
            )
        unknown = set(preference) - set(bindings)
        if unknown:
            raise ValueError(
                f"provider {self.provider.name!r} prefers unknown bindings: {sorted(unknown)}"
            )
        ordered = [bindings[binding_id] for binding_id in preference]
        ordered.extend(binding for key, binding in bindings.items() if key not in preference)
        ordered = [
            *(binding for binding in ordered if binding.protocol is ingress_protocol),
            *(binding for binding in ordered if binding.protocol is not ingress_protocol),
        ]
        if manifest is not None and not model_supports_manifest(
            candidate.info.feature_capabilities,
            manifest,
        ):
            return ()

        return tuple(
            TransportPlan(
                model=candidate.model,
                provider=self.provider,
                candidate=FlowCandidate.for_binding(
                    source_protocol=ingress_protocol,
                    binding=binding,
                ),
            )
            for binding in ordered
            if self._binding_is_available(candidate.info, binding)
        )

    @staticmethod
    def _binding_is_available(info: ModelInfo, binding: EndpointBinding) -> bool:
        """Filter only explicit catalog negatives; missing metadata stays probeable."""
        transport_capabilities = getattr(info, "transport_capabilities", {})
        if binding.protocol.value in transport_capabilities:
            return transport_capabilities[binding.protocol.value] is not False

        endpoints = info.supported_endpoints
        if endpoints is not None:
            advertised = {
                WireProtocol.ANTHROPIC_MESSAGES
                if endpoint.endswith("/messages")
                else _ENDPOINT_PROTOCOLS.get(endpoint)
                for endpoint in endpoints
            }
            advertised.discard(None)
            return binding.protocol in advertised

        operation_values = info.operation_capabilities
        relevant = [
            operation_values.get(operation.value) for operation in binding.capabilities.operations
        ]
        return not relevant or any(value is not False for value in relevant)


__all__ = ["ProviderHandler"]
