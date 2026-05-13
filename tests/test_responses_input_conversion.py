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
from router_maestro.server.schemas.responses import (
    ResponsesFunctionCallInput,
    ResponsesRequest,
)


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


# ---------------------------------------------------------------------------
# Pydantic schema-level regression tests (v0.3.10 root cause)
# ---------------------------------------------------------------------------
#
# The convert_input_to_internal tests above bypass Pydantic. The actual
# v0.3.8/9/10 bug was that ``ResponsesFunctionCallInput`` silently dropped
# ``namespace`` *before* convert_input_to_internal ever saw it because
# Pydantic v2 ignores extra fields by default. These tests pin the schema
# down so a future refactor can't strip the field again.


def test_pydantic_input_model_preserves_namespace():
    m = ResponsesFunctionCallInput(
        type="function_call",
        call_id="c1",
        name="execute_query",
        arguments="{}",
        namespace="mcp__kusto_mcp__",
    )
    dumped = m.model_dump(exclude_none=True)
    assert dumped["namespace"] == "mcp__kusto_mcp__"


def test_pydantic_input_model_preserves_unknown_extras():
    # extra="allow" lets future Copilot fields (tool_metadata, etc.) survive
    # without us having to ship a release for each one.
    m = ResponsesFunctionCallInput(
        type="function_call",
        call_id="c1",
        name="x",
        arguments="{}",
        tool_metadata={"key": "v"},
    )
    dumped = m.model_dump(exclude_none=True)
    assert dumped["tool_metadata"] == {"key": "v"}


def test_responses_request_preserves_namespace_through_full_parse():
    # End-to-end mirror of how FastAPI parses the Codex request body.
    req = ResponsesRequest(
        model="gpt-5.5",
        input=[
            {
                "type": "function_call",
                "call_id": "call_kusto_1",
                "name": "execute_query",
                "namespace": "mcp__kusto_mcp__",
                "arguments": '{"query":"x"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_kusto_1",
                "output": "ok",
            },
        ],
    )
    items = convert_input_to_internal(req.input)
    fc = items[0]
    assert fc["type"] == "function_call"
    assert fc["namespace"] == "mcp__kusto_mcp__"
    assert fc["call_id"] == "call_kusto_1"
