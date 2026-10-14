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


def test_copilot_gpt54_allows_top_p_only_on_chat_transport():
    c = _copilot_contract()
    assert c.allows_top_p(Operation.CHAT, model="gpt-5.4") is True
    assert c.allows_top_p(Operation.RESPONSES, model="gpt-5.4") is False
    assert c.allows_top_p(Operation.RESPONSES_STREAM, model="gpt-5.4-mini") is False
    assert c.allows_top_p(Operation.NATIVE_ANTHROPIC, model="gpt-5.4") is False
    assert c.allows_top_p(Operation.RESPONSES, model="gpt-5.3-codex") is True


def test_permissive_defaults_for_tools_and_temperature():
    from router_maestro.providers.outbound_contract import PermissiveOutboundContract

    c = PermissiveOutboundContract()
    tools = [{"type": "anything"}]
    assert c.filter_tools(tools, operation=Operation.CHAT, model="m") == tools
    assert c.allows_temperature(Operation.RESPONSES) is True
    assert c.allows_top_p(Operation.RESPONSES, model="m") is True


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


def test_sanitize_output_config_normalizes_effort_and_forwards_siblings():
    from router_maestro.providers.copilot import CopilotOutboundContract

    # ``effort`` is Router-Maestro's to normalize; siblings are GHC's to judge.
    body = {"output_config": {"effort": "xhigh", "format": "json"}}
    assert CopilotOutboundContract.sanitize_output_config(body) == "xhigh"
    assert body["output_config"] == {"effort": "xhigh", "format": "json"}

    # An unusable effort is dropped without taking the rest of the object with it.
    partial = {"output_config": {"effort": "invalid", "format": "json"}}
    assert CopilotOutboundContract.sanitize_output_config(partial) is None
    assert partial["output_config"] == {"format": "json"}

    # With nothing left to send, the key is removed entirely.
    dropped = {"output_config": {"effort": "invalid"}}
    assert CopilotOutboundContract.sanitize_output_config(dropped) is None
    assert "output_config" not in dropped


def test_sanitize_output_config_strips_siblings_ghc_rejects():
    """Siblings GHC is known to reject are dropped, not forwarded into a 400.

    Verified live on claude-opus-4.6/sonnet-4.6/haiku-4.5: ``task_budget`` inside
    ``output_config`` always returns 400 ``Extra inputs are not permitted``. This
    follows the same rule as ``store`` on the Responses passthrough — an option
    the upstream is known to refuse is stripped so the request still succeeds,
    rather than surfacing an avoidable upstream error to the client.
    """
    from router_maestro.providers.copilot import CopilotOutboundContract

    body = {
        "output_config": {
            "effort": "high",
            "task_budget": {"type": "tokens", "total": 64000},
        }
    }
    assert CopilotOutboundContract.sanitize_output_config(body) == "high"
    assert body["output_config"] == {"effort": "high"}

    # Stripping the rejected sibling can empty the object; drop it entirely then.
    only_budget = {"output_config": {"task_budget": {"type": "tokens", "total": 64000}}}
    assert CopilotOutboundContract.sanitize_output_config(only_budget) is None
    assert "output_config" not in only_budget


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


def test_drop_unsigned_thinking_removes_empty_signature_blocks():
    from router_maestro.providers.copilot import CopilotOutboundContract

    signed = {"type": "thinking", "thinking": "kept", "signature": "a" * 40}
    unsigned_empty = {"type": "thinking", "thinking": "poison", "signature": ""}
    unsigned_missing = {"type": "thinking", "thinking": "poison"}
    redacted_empty = {"type": "redacted_thinking", "data": ""}
    text = {"type": "text", "text": "visible"}
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    signed,
                    unsigned_empty,
                    unsigned_missing,
                    redacted_empty,
                    text,
                    tool_use,
                ],
            }
        ]
    }

    CopilotOutboundContract.drop_unsigned_thinking(body)

    # Only the signed thinking block survives; text/tool_use are untouched.
    assert body["messages"][0]["content"] == [signed, text, tool_use]


def test_drop_unsigned_thinking_is_copy_on_write_when_all_signed():
    from router_maestro.providers.copilot import CopilotOutboundContract

    original_messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t", "signature": "s" * 40},
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    body = {"messages": original_messages}

    CopilotOutboundContract.drop_unsigned_thinking(body)

    # Nothing to drop -> the shared ingress list is left in place unchanged.
    assert body["messages"] is original_messages


def test_drop_unsigned_thinking_does_not_mutate_shared_ingress():
    from router_maestro.providers.copilot import CopilotOutboundContract

    turn_content = [
        {"type": "thinking", "thinking": "poison", "signature": ""},
        {"type": "text", "text": "hi"},
    ]
    ingress_messages = [{"role": "assistant", "content": turn_content}]
    # ``_build_native_messages_payload`` only shallow-copies the top-level body,
    # so the sanitizer must rebuild rather than mutate the caller's structures.
    body = {"messages": ingress_messages}

    CopilotOutboundContract.drop_unsigned_thinking(body)

    # The rebuilt body drops the unsigned block...
    assert body["messages"][0]["content"] == [{"type": "text", "text": "hi"}]
    # ...while the shared ingress message list and content stay intact.
    assert ingress_messages[0]["content"] is turn_content
    assert len(turn_content) == 2


def test_drop_unsigned_thinking_ignores_non_list_content():
    from router_maestro.providers.copilot import CopilotOutboundContract

    # String content (system turns) and missing messages must not raise.
    body = {"messages": [{"role": "user", "content": "plain string"}]}
    CopilotOutboundContract.drop_unsigned_thinking(body)
    assert body["messages"][0]["content"] == "plain string"

    CopilotOutboundContract.drop_unsigned_thinking({})  # no messages key


def test_is_signature_error_detects_thinking_signature_400():
    from router_maestro.providers.copilot import _is_signature_error

    real = (
        b'{"type":"error","error":{"type":"invalid_request_error","message":'
        b'"messages.3.content.3: Invalid `signature` in `thinking` block"}}'
    )
    assert _is_signature_error(real) is True
    assert _is_signature_error(real.decode()) is True
    # Needs BOTH tokens; unrelated 400s must not trigger the strip-and-retry.
    assert _is_signature_error(b'{"error":{"message":"missing signature"}}') is False
    assert _is_signature_error("thinking budget too large") is False
    assert _is_signature_error(b"") is False


def test_is_signature_error_requires_invalid_request_error_type():
    from router_maestro.providers.copilot import _is_signature_error

    # A structured error of a DIFFERENT type that merely mentions both tokens
    # must NOT misfire (would otherwise discard history and mask the real cause).
    other_type = (
        b'{"type":"error","error":{"type":"rate_limit_error","message":'
        b'"thinking signature service is overloaded"}}'
    )
    assert _is_signature_error(other_type) is False
    # A capability rejection that references both words is not a signature 400.
    unsupported = (
        b'{"error":{"type":"invalid_request_error","code":"unsupported_api_for_model",'
        b'"message":"this model does not support thinking or signatures"}}'
    )
    # Same error type + both tokens -> treated as a signature error (bounded,
    # documented tradeoff): the strip-and-retry is a safe no-op if wrong.
    assert _is_signature_error(unsupported) is True
    # Non-JSON body still recovers via the two-token fallback.
    assert _is_signature_error(b"Invalid signature in thinking block") is True


def test_strip_history_thinking_blocks_removes_all_reasoning_copy_on_write():
    from router_maestro.providers.copilot import _strip_history_thinking_blocks

    text = {"type": "text", "text": "answer"}
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
    assistant_content = [
        {"type": "thinking", "thinking": "t", "signature": "s" * 40},
        {"type": "redacted_thinking", "data": "d" * 40},
        text,
        tool_use,
    ]
    # Reasoning blocks are assistant-only in the wire schema, so the strip is
    # role-agnostic (matching drop_unsigned_thinking) — a stray thinking block on
    # any turn is removed rather than special-cased by role.
    plain_user = [{"type": "text", "text": "q"}]
    messages = [
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": plain_user},
    ]
    body = {"messages": messages}

    _strip_history_thinking_blocks(body)

    # All assistant thinking/redacted_thinking gone; text + tool_use kept.
    assert body["messages"][0]["content"] == [text, tool_use]
    # The reasoning-free user turn is left in place (copy-on-write).
    assert body["messages"][1]["content"] is plain_user
    # The shared ingress content list for the mutated turn is not mutated.
    assert assistant_content == [
        {"type": "thinking", "thinking": "t", "signature": "s" * 40},
        {"type": "redacted_thinking", "data": "d" * 40},
        text,
        tool_use,
    ]


def test_strip_history_thinking_blocks_is_noop_without_reasoning():
    from router_maestro.providers.copilot import _strip_history_thinking_blocks

    original = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
    body = {"messages": original}
    _strip_history_thinking_blocks(body)
    # Nothing removed -> shared list left in place.
    assert body["messages"] is original
    _strip_history_thinking_blocks({})  # no messages key must not raise
