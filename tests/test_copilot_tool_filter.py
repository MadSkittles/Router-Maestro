"""Tests for Copilot tool-type filtering."""

from router_maestro.providers import CopilotProvider


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

    def test_drops_empty_namespace_tools(self):
        # Bare namespace items without an inner tools[] still trigger
        # "Missing required parameter: 'tools[N].tools'" — drop those only.
        tools = [
            {"type": "function", "name": "foo"},
            {
                "type": "namespace",
                "name": "mcp__chrome_devtools__",
                "description": "Tools in the mcp__chrome_devtools__ namespace.",
            },
            {"type": "function", "name": "bar"},
        ]
        result = self.provider._filter_unsupported_tools(tools)
        assert result is not None
        assert all(t["type"] == "function" for t in result)
        assert [t["name"] for t in result] == ["foo", "bar"]

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

    def test_drops_namespace_with_empty_tools_list(self):
        # ``tools: []`` is just as bad as omitting the field — Copilot
        # 400s the same way. Drop both shapes.
        tools = [
            {"type": "function", "name": "foo"},
            {
                "type": "namespace",
                "name": "mcp__empty__",
                "description": "...",
                "tools": [],
            },
        ]
        result = self.provider._filter_unsupported_tools(tools)
        assert result is not None
        assert [t["name"] for t in result] == ["foo"]

    def test_drops_web_search_and_code_interpreter(self):
        tools = [
            {"type": "web_search"},
            {"type": "web_search_preview"},
            {"type": "code_interpreter"},
        ]
        assert self.provider._filter_unsupported_tools(tools) is None

    def test_keeps_unknown_non_function_types(self):
        # denylist semantics: anything not in UNSUPPORTED_TOOL_TYPES passes through
        tools = [{"type": "local_shell", "name": "shell"}]
        result = self.provider._filter_unsupported_tools(tools)
        assert result == tools
