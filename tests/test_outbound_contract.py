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
