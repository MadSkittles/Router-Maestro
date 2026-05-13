"""Tests for ``convert_input_to_internal``'s field preservation.

Codex CLI round-trips MCP-namespaced ``function_call`` items back to the
upstream on the next turn. If we strip the ``namespace`` field while
converting the input, Copilot CAPI rejects the request with::

    Missing namespace for function_call 'execute_query'.
    It does not exist in the default namespace.
    Round-trip the model's function_call item with its namespace field
    included.

These tests pin the input-side behavior down.
"""

from router_maestro.server.routes.responses import convert_input_to_internal


def test_function_call_namespace_preserved():
    items = convert_input_to_internal(
        [
            {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call_1",
                "name": "execute_query",
                "namespace": "kusto",
                "arguments": '{"query":"x"}',
                "status": "completed",
            },
        ]
    )
    assert isinstance(items, list)
    assert items[0]["type"] == "function_call"
    assert items[0]["namespace"] == "kusto"
    assert items[0]["call_id"] == "call_1"
    assert items[0]["name"] == "execute_query"
    assert items[0]["arguments"] == '{"query":"x"}'


def test_function_call_without_namespace_omits_field():
    # Non-MCP function_calls must NOT gain a ``namespace=None`` key —
    # Copilot may treat the presence of the key as a signal.
    items = convert_input_to_internal(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{}",
            },
        ]
    )
    assert isinstance(items, list)
    assert "namespace" not in items[0]


def test_function_call_explicit_none_namespace_omitted():
    # Defensive: even if Codex sends ``namespace: null`` we drop the field
    # rather than forwarding a literal null which Copilot may also reject.
    items = convert_input_to_internal(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{}",
                "namespace": None,
            },
        ]
    )
    assert isinstance(items, list)
    assert "namespace" not in items[0]
