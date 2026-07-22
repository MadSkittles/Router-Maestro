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


def test_copilot_resolve_reasoning_responses_catalog_warm():
    r = _copilot_contract().resolve_reasoning(
        model="github-copilot/gpt-5.5",
        reasoning_effort="max",
        thinking_budget=None,
        catalog_effort_values=["low", "medium", "high", "xhigh"],
        operation=Operation.RESPONSES,
    )
    assert r.effort == "xhigh"  # pick_closest_effort(max, [...xhigh])
    assert r.rewrite_max_tokens_to_completion is False  # responses never rewrites


def test_copilot_resolve_reasoning_responses_cold_downgrades_via_upstream():
    r = _copilot_contract().resolve_reasoning(
        model="github-copilot/gpt-5.5",
        reasoning_effort="max",
        thinking_budget=None,
        catalog_effort_values=None,  # cold; known_reasoning_support True for gpt-5
        operation=Operation.RESPONSES,
    )
    # gpt-5 known-supported: cold path forwards verbatim (known_reasoning_support is True)
    assert r.effort == "max"


def test_copilot_resolve_reasoning_responses_unsupported_model_strips():
    r = _copilot_contract().resolve_reasoning(
        model="github-copilot/gpt-4o",  # known_reasoning_support False
        reasoning_effort="high",
        thinking_budget=None,
        catalog_effort_values=None,
        operation=Operation.RESPONSES,
    )
    assert r.effort is None


def test_copilot_resolve_reasoning_responses_clamps_up_below_floor():
    r = _copilot_contract().resolve_reasoning(
        model="github-copilot/claude-opus-4.7",
        reasoning_effort="low",
        thinking_budget=None,
        catalog_effort_values=["medium", "high"],
        operation=Operation.RESPONSES,
    )
    assert r.effort == "medium"


def test_copilot_resolve_reasoning_responses_empty_catalog_strips():
    r = _copilot_contract().resolve_reasoning(
        model="github-copilot/claude-haiku-4.5",
        reasoning_effort="high",
        thinking_budget=None,
        catalog_effort_values=[],
        operation=Operation.RESPONSES,
    )
    assert r.effort is None


# --- Round 2: tool filtering + temperature verdict ---


def test_copilot_filter_tools_keeps_function_drops_web_search():
    c = _copilot_contract()
    # A function tool and an unknown/other type are kept; a known-unsupported
    # type (web_search) is silently dropped rather than rejected, so clients
    # like Codex that inject it unconditionally still get their other tools.
    kept = c.filter_tools(
        [{"type": "function", "name": "echo"}, {"type": "custom_future"}],
        operation=Operation.RESPONSES,
        model="github-copilot/gpt-5.5",
    )
    assert kept == [{"type": "function", "name": "echo"}, {"type": "custom_future"}]

    mixed = c.filter_tools(
        [{"type": "function", "name": "echo"}, {"type": "web_search"}],
        operation=Operation.RESPONSES,
        model="github-copilot/gpt-5.5",
    )
    assert mixed == [{"type": "function", "name": "echo"}]

    # web_search alone leaves nothing → None
    only_unsupported = c.filter_tools(
        [{"type": "web_search"}],
        operation=Operation.RESPONSES,
        model="github-copilot/gpt-5.5",
    )
    assert only_unsupported is None


def test_copilot_filter_tools_rejects_empty_namespace():
    import pytest

    from router_maestro.providers import RequestOptionError

    c = _copilot_contract()
    with pytest.raises(RequestOptionError):
        c.filter_tools(
            [{"type": "namespace", "tools": []}],
            operation=Operation.RESPONSES,
            model="github-copilot/gpt-5.5",
        )


def test_copilot_allows_temperature_chat_but_not_responses():
    c = _copilot_contract()
    assert c.allows_temperature(Operation.CHAT) is True
    assert c.allows_temperature(Operation.RESPONSES) is False


def test_permissive_defaults_for_tools_and_temperature():
    from router_maestro.providers.outbound_contract import PermissiveOutboundContract

    c = PermissiveOutboundContract()
    tools = [{"type": "anything"}]
    assert c.filter_tools(tools, operation=Operation.CHAT, model="m") == tools
    assert c.allows_temperature(Operation.RESPONSES) is True


# --- reconcile_passthrough_body orchestrator (Responses shape) ---


def test_reconcile_responses_strips_filters_and_downgrades():
    c = _copilot_contract()
    body = {
        "model": "github-copilot/gpt-5.5",
        "input": "hi",
        "store": True,  # stripped (not in allowlist)
        "tools": [{"type": "function", "name": "echo"}, {"type": "web_search"}],
        "reasoning": {"effort": "xhigh", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
    }
    c.reconcile_passthrough_body(
        body,
        operation=Operation.RESPONSES,
        model="github-copilot/gpt-5.5",
        catalog_effort_values=["low", "medium", "high"],
    )
    assert "store" not in body
    assert body["tools"] == [{"type": "function", "name": "echo"}]
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["include"] == ["reasoning.encrypted_content"]  # preserved, not injected


def test_reconcile_responses_rejects_temperature():
    import pytest

    from router_maestro.providers import RequestOptionError

    c = _copilot_contract()
    with pytest.raises(RequestOptionError) as excinfo:
        c.reconcile_passthrough_body(
            {"model": "m", "input": "hi", "temperature": 0.5},
            operation=Operation.RESPONSES,
            model="github-copilot/gpt-5.5",
            catalog_effort_values=None,
        )
    assert excinfo.value.parameter == "temperature"


def test_reconcile_permissive_is_noop():
    from router_maestro.providers.outbound_contract import PermissiveOutboundContract

    c = PermissiveOutboundContract()
    body = {"model": "m", "input": "hi", "temperature": 0.5, "reasoning": {"effort": "high"}}
    before = dict(body)
    c.reconcile_passthrough_body(
        body, operation=Operation.RESPONSES, model="m", catalog_effort_values=None
    )
    assert body == before


# --- native Anthropic normalizers (folded in from the beta route) ---


def test_sanitize_output_config_keeps_only_valid_effort():
    from router_maestro.providers.copilot import CopilotOutboundContract

    body = {"output_config": {"effort": "xhigh", "format": "json"}}
    assert CopilotOutboundContract.sanitize_output_config(body) == "xhigh"
    assert body["output_config"] == {"effort": "xhigh"}

    dropped = {"output_config": {"effort": "invalid"}}
    assert CopilotOutboundContract.sanitize_output_config(dropped) is None
    assert "output_config" not in dropped


def test_resolve_native_effort_downgrades_or_clamps_up():
    from router_maestro.providers.copilot import CopilotOutboundContract

    # Unknown catalog (None) is preserved verbatim.
    assert CopilotOutboundContract.resolve_native_effort("xhigh", None) == "xhigh"
    # Downgrades to the nearest tier at or below.
    assert (
        CopilotOutboundContract.resolve_native_effort("xhigh", ("low", "medium", "high")) == "high"
    )
    # No tier at or below the request -> clamp UP to the lowest available.
    assert CopilotOutboundContract.resolve_native_effort("low", ("high",)) == "high"


def test_reject_unpreservable_native_options_flags_temp_plus_top_p():
    import pytest

    from router_maestro.providers import RequestOptionError
    from router_maestro.providers.copilot import CopilotOutboundContract

    # Either alone is fine.
    CopilotOutboundContract.reject_unpreservable_native_options({"temperature": 0.5})
    CopilotOutboundContract.reject_unpreservable_native_options({"top_p": 0.9})
    # Both together cannot be preserved on the native transport.
    with pytest.raises(RequestOptionError) as excinfo:
        CopilotOutboundContract.reject_unpreservable_native_options(
            {"temperature": 0.5, "top_p": 0.9}
        )
    assert excinfo.value.parameter == "top_p"
