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

    def test_drops_namespace_tools(self):
        # Codex CLI emits these to group MCP servers; Copilot rejects with
        # "Missing required parameter: 'tools[N].tools'".
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
