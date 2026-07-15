from router_maestro.providers.outbound_contract import (
    OutboundContract,
    PermissiveOutboundContract,
)
from router_maestro.routing.capabilities import Operation


def test_permissive_contract_forwards_everything():
    contract = PermissiveOutboundContract()
    assert contract.forwardable_fields(Operation.NATIVE_ANTHROPIC) is None


def test_permissive_is_an_outbound_contract():
    assert isinstance(PermissiveOutboundContract(), OutboundContract)


def test_base_provider_default_contract_is_permissive():
    from router_maestro.providers.anthropic import AnthropicProvider

    # A Path-1 provider inherits the permissive default (nothing to strip).
    contract = AnthropicProvider().outbound_contract
    assert isinstance(contract, OutboundContract)
    assert contract.forwardable_fields(Operation.NATIVE_ANTHROPIC) is None


def test_copilot_contract_forwards_native_anthropic_allowlist():
    from router_maestro.providers.copilot import CopilotProvider

    contract = CopilotProvider().outbound_contract
    fields = contract.forwardable_fields(Operation.NATIVE_ANTHROPIC)
    assert fields is not None
    # Forwarded (verified live: GHC applies context_management).
    assert "context_management" in fields
    assert "messages" in fields
    assert "thinking" in fields
    # Not in the allowlist -> stripped (GHC 400s these).
    assert "mcp_servers" not in fields
    assert "container" not in fields


def test_copilot_contract_is_permissive_for_other_operations():
    from router_maestro.providers.copilot import CopilotProvider

    contract = CopilotProvider().outbound_contract
    assert contract.forwardable_fields(Operation.CHAT) is None


# --- Round 2: reasoning resolution ---


def _copilot_contract():
    from router_maestro.providers.copilot import CopilotProvider

    return CopilotProvider().outbound_contract


def test_copilot_resolve_reasoning_catalog_warm_picks_closest():
    r = _copilot_contract().resolve_reasoning(
        model="github-copilot/claude-sonnet-4.6",
        reasoning_effort="max",
        thinking_budget=None,
        catalog_effort_values=["low", "medium", "high"],
        operation=Operation.CHAT,
    )
    assert r.effort == "high"  # pick_closest_effort(max, [low,medium,high])


def test_copilot_resolve_reasoning_cold_claude_downgrades_xhigh():
    r = _copilot_contract().resolve_reasoning(
        model="github-copilot/claude-opus-4.8",
        reasoning_effort="xhigh",
        thinking_budget=None,
        catalog_effort_values=None,  # cold
        operation=Operation.CHAT,
    )
    assert r.effort == "high"


def test_copilot_resolve_reasoning_gpt54_rewrites_max_tokens_flag():
    r = _copilot_contract().resolve_reasoning(
        model="github-copilot/gpt-5.4",
        reasoning_effort="high",
        thinking_budget=None,
        catalog_effort_values=["low", "medium", "high"],
        operation=Operation.CHAT,
    )
    assert r.rewrite_max_tokens_to_completion is True
    assert r.effort == "high"
