"""Tests for Copilot tool-type filtering."""

import pytest

from router_maestro.providers import CopilotProvider
from router_maestro.providers.base import ProviderFailureKind, RequestOptionError, ResponsesRequest


class TestFilterUnsupportedTools:
    def setup_method(self):
        self.provider = CopilotProvider()

    def test_returns_none_for_empty(self):
        assert self.provider._filter_unsupported_tools(None) is None
        assert self.provider._filter_unsupported_tools([]) is None

    def test_keeps_function_tools(self):
        tools = [
            {"type": "function", "name": "foo", "parameters": {}},
            {"type": "function", "name": "bar", "parameters": {}},
        ]
        assert self.provider._filter_unsupported_tools(tools) == tools

    @pytest.mark.parametrize("inner_tools", [None, []], ids=["missing-tools", "empty-tools"])
    def test_rejects_empty_namespace_tools(self, inner_tools):
        tools = [
            {"type": "function", "name": "foo"},
            {
                "type": "namespace",
                "name": "mcp__chrome_devtools__",
                "description": "Tools in the mcp__chrome_devtools__ namespace.",
                **({"tools": inner_tools} if inner_tools is not None else {}),
            },
            {"type": "function", "name": "bar"},
        ]

        with pytest.raises(RequestOptionError) as caught:
            self.provider._build_responses_payload(
                ResponsesRequest(model="gpt-5.4", input="hi", tools=tools)
            )

        assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert caught.value.parameter == "tools"
        assert caught.value.model == "gpt-5.4"

    def test_keeps_namespace_with_inner_tools(self):
        # Codex's MCP registry shape — namespace wraps the actual function
        # tools. Dropping the wrapper means Copilot can't resolve
        # ``execute_query`` and 400s with ``Missing namespace for
        # function_call 'execute_query'`` (v0.3.8 → v0.3.9 bug).
        kusto_namespace = {
            "type": "namespace",
            "name": "mcp__kusto_mcp__",
            "description": "Tools in the mcp__kusto_mcp__ namespace.",
            "tools": [
                {
                    "type": "function",
                    "name": "execute_query",
                    "description": "Execute a KQL query",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
        tools = [
            {"type": "function", "name": "shell"},
            kusto_namespace,
        ]
        result = self.provider._filter_unsupported_tools(tools)
        assert result is not None
        assert len(result) == 2
        assert result[1] == kusto_namespace

    @pytest.mark.parametrize(
        "tool_type",
        ["web_search", "web_search_preview", "code_interpreter"],
    )
    def test_rejects_unsupported_tool_types(self, tool_type):
        with pytest.raises(RequestOptionError) as caught:
            self.provider._build_responses_payload(
                ResponsesRequest(
                    model="gpt-5.4",
                    input="hi",
                    tools=[{"type": tool_type}],
                )
            )

        assert caught.value.kind is ProviderFailureKind.CLIENT_REQUEST
        assert caught.value.parameter == "tools"
        assert caught.value.model == "gpt-5.4"

    def test_mixed_supported_and_unsupported_tools_rejects_entire_request(self):
        tools = [
            {"type": "function", "name": "lookup", "parameters": {}},
            {"type": "web_search"},
        ]

        with pytest.raises(RequestOptionError) as caught:
            self.provider._build_responses_payload(
                ResponsesRequest(model="gpt-5.4", input="hi", tools=tools)
            )

        assert caught.value.parameter == "tools"
        assert caught.value.model == "gpt-5.4"

    def test_keeps_unknown_non_function_types(self):
        # denylist semantics: anything not in UNSUPPORTED_TOOL_TYPES passes through
        tools = [{"type": "local_shell", "name": "shell"}]
        result = self.provider._filter_unsupported_tools(tools)
        assert result == tools
