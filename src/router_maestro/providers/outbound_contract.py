"""Provider-bound knowledge of what an upstream will accept on the wire.

The contract is the single home for *outbound* protection: Router-Maestro is
lenient with what a client sends, but normalizes or rejects anything the
upstream backend cannot accept before forwarding. ``reconcile_passthrough_body``
is the entry point the OpenAI Responses beta passthrough route calls; it
composes the per-concern hooks (``forwardable_fields`` field strip,
``filter_tools``, ``allows_temperature`` verdict, ``resolve_reasoning``
downgrade) that each provider overrides.

The beta native Anthropic route has a differently shaped body (Anthropic-format
tools, ``thinking`` budgets, ``output_config.effort``) and does not go through
this orchestrator; it applies the Copilot-specific hooks directly
(``CopilotOutboundContract.apply_native_anthropic_thinking`` and siblings) plus
its own top-level field strip. Those hooks live on the contract so it stays the
single source of the Copilot wire rules.
"""

from abc import ABC
from dataclasses import dataclass

from router_maestro.routing.capabilities import Operation


@dataclass(frozen=True, slots=True)
class ReasoningResolution:
    """Outcome of a provider's reasoning-effort resolution.

    ``effort`` is the upstream effort to send (``None`` to omit reasoning).
    ``rewrite_max_tokens_to_completion`` requests the Copilot gpt-5.4 payload-key
    rewrite (``max_tokens`` -> ``max_completion_tokens``).
    """

    effort: str | None
    rewrite_max_tokens_to_completion: bool = False


class OutboundContract(ABC):
    """What a provider's upstream accepts. Subclass per provider."""

    def forwardable_fields(self, operation: Operation) -> frozenset[str] | None:
        """Top-level fields allowed through on a raw-passthrough forward.

        ``None`` means forward everything (no strip). A provider whose upstream
        rejects unknown top-level fields returns an explicit allowlist.
        """
        return None

    def resolve_reasoning(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        thinking_budget: int | None,
        catalog_effort_values: list[str] | None,
        operation: Operation,
    ) -> ReasoningResolution:
        """Resolve the upstream reasoning effort for this provider/operation.

        Default: forward the requested effort unchanged with no rewrite.
        Providers whose upstream downgrades or rejects tiers override this.
        """
        return ReasoningResolution(effort=reasoning_effort)

    def filter_tools(
        self,
        tools: list[dict] | None,
        *,
        operation: Operation,
        model: str | None = None,
    ) -> list[dict] | None:
        """Filter tools this upstream cannot express. Default: pass through."""
        return tools

    def allows_temperature(self, operation: Operation) -> bool:
        """Whether the upstream accepts explicit ``temperature``. Default: yes."""
        return True

    def reconcile_passthrough_body(
        self,
        body: dict,
        *,
        operation: Operation,
        model: str,
        catalog_effort_values: list[str] | None,
    ) -> None:
        """Reconcile a raw passthrough body in place against the wire contract.

        Composes the per-concern hooks for the generic (Responses-shaped) body:
        top-level field strip, tool filtering, temperature verdict, and reasoning
        downgrade. Mutates ``body`` in place; raises ``RequestOptionError`` for
        options the upstream cannot represent (temperature on a rejecting
        operation, malformed tools, or unsupported reasoning). Routes whose
        passthrough body is shaped differently (e.g. the native Anthropic wire)
        call the provider's hooks directly instead of this orchestrator.
        """
        # 1. Top-level field strip (permissive default forwards everything).
        forwardable = self.forwardable_fields(operation)
        if forwardable is not None:
            for key in set(body) - forwardable:
                del body[key]

        # 2. Tools — filter_tools may drop unsupported types or raise on malformed.
        if "tools" in body:
            filtered = self.filter_tools(body.get("tools"), operation=operation, model=model)
            if filtered:
                body["tools"] = filtered
            else:
                body.pop("tools", None)

        # 3. Temperature verdict.
        if body.get("temperature") is not None and not self.allows_temperature(operation):
            from router_maestro.providers.base import RequestOptionError  # lazy: base<-contract

            raise RequestOptionError(
                "upstream does not support request option 'temperature' on this operation",
                model=model,
                parameter="temperature",
            )

        # 4. Reasoning effort — only when the client sent an explicit effort.
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
            resolution = self.resolve_reasoning(
                model=model,
                reasoning_effort=reasoning["effort"],
                thinking_budget=None,
                catalog_effort_values=catalog_effort_values,
                operation=operation,
            )
            if resolution.effort is not None:
                # Rewrite only the effort; preserve sibling keys (summary) and the
                # client's own ``include`` so encrypted reasoning still round-trips.
                body["reasoning"] = {**reasoning, "effort": resolution.effort}
            else:
                # Unreachable when the client sent a string effort (the guard
                # above): resolve_reasoning either returns a concrete tier or
                # raises. Kept only so a future None-yielding resolver cannot
                # forward an effort the upstream rejected.
                body.pop("reasoning", None)


class PermissiveOutboundContract(OutboundContract):
    """Default contract: forward everything.

    Used by providers that build their upstream payload from typed fields
    (Path 1), where unknown options are already absent and there is nothing to
    strip.
    """
