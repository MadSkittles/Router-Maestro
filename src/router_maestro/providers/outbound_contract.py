"""Provider-bound knowledge of what an upstream will accept on the wire.

Round 1 governs only the raw-passthrough forward path (currently the beta
native Anthropic route): which top-level fields survive the strip before the
body is sent upstream. Round 2 will grow this contract with per-option value
reconciliation (downgrade/drop/reject) and tool filtering, migrating the rules
currently scattered across each provider's ``_build_payload``.
"""

from abc import ABC

from router_maestro.routing.capabilities import Operation


class OutboundContract(ABC):
    """What a provider's upstream accepts. Subclass per provider."""

    def forwardable_fields(self, operation: Operation) -> frozenset[str] | None:
        """Top-level fields allowed through on a raw-passthrough forward.

        ``None`` means forward everything (no strip). A provider whose upstream
        rejects unknown top-level fields returns an explicit allowlist.
        """
        return None


class PermissiveOutboundContract(OutboundContract):
    """Default contract: forward everything.

    Used by providers that build their upstream payload from typed fields
    (Path 1), where unknown options are already absent and there is nothing to
    strip.
    """
