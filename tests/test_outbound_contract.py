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
