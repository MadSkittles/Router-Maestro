"""Tests for tool call recovery from XML-embedded content."""

import json

from router_maestro.providers.tool_parsing import (
    recover_invoke_tool_calls,
    recover_tool_calls_from_content,
)


class TestRecoverToolCallsFromContent:
    """Tests for recover_tool_calls_from_content."""

    def test_normal_tool_calls_present_unchanged(self):
        """When tool_calls already present, return inputs unchanged (idempotent)."""
        existing_tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "exec", "arguments": '{"command": "ls"}'},
            }
        ]
        content, tool_calls = recover_tool_calls_from_content(
            content="hi",
            tool_calls=existing_tool_calls,
            finish_reason="tool_calls",
        )
        assert content == "hi"
        assert tool_calls is existing_tool_calls

    def test_single_xml_tool_call(self):
        """Extract a single <tool_call> XML block."""
        xml_content = (
            "I will check the hostname.\n"
            "<tool_call>\n"
            '{"name": "exec", "arguments": {"command": "hostname"}}\n'
            "</tool_call>"
        )
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["function"]["name"] == "exec"
        assert json.loads(tc["function"]["arguments"]) == {"command": "hostname"}
        assert tc["id"].startswith("toolu_")
        assert tc["type"] == "function"
        # Text before the XML should be preserved
        assert content == "I will check the hostname."

    def test_multiple_xml_tool_calls_with_text(self):
        """Extract multiple <tool_call> blocks, preserve surrounding text, strip tool_result."""
        xml_content = (
            "Let me check your system.\n"
            '<tool_call>{"name": "exec", "arguments": {"command": "hostname"}}</tool_call>\n'
            '<tool_call>{"name": "exec", "arguments": {"command": "uptime"}}</tool_call>\n'
            '<tool_call>{"name": "exec", "arguments": {"command": "df -h"}}</tool_call>\n'
            "<tool_result>some result</tool_result>\n"
            "I'll report back soon."
        )
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 3
        assert tool_calls[0]["function"]["name"] == "exec"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"command": "hostname"}
        assert tool_calls[1]["function"]["name"] == "exec"
        assert json.loads(tool_calls[1]["function"]["arguments"]) == {"command": "uptime"}
        assert tool_calls[2]["function"]["name"] == "exec"
        assert json.loads(tool_calls[2]["function"]["arguments"]) == {"command": "df -h"}
        # Text preserved, tool_result stripped
        assert "Let me check your system." in content
        assert "I'll report back soon." in content
        assert "<tool_result>" not in content
        assert "<tool_call>" not in content

    def test_function_tag_variant(self):
        """Extract tool calls from <function> tags."""
        xml_content = (
            '<function>{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}</function>'
        )
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "read_file"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "/tmp/test.txt"}
        # Content should be None since it's empty after stripping
        assert content is None

    def test_finish_reason_not_tool_calls_unchanged(self):
        """When finish_reason != 'tool_calls', don't parse XML even if present."""
        xml_content = '<tool_call>{"name": "exec", "arguments": {"command": "ls"}}</tool_call>'
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="stop",
        )
        assert content == xml_content
        assert tool_calls is None

    def test_malformed_json_skipped(self):
        """Malformed JSON inside tag is skipped with a warning, no crash."""
        xml_content = (
            "<tool_call>not valid json</tool_call>\n"
            '<tool_call>{"name": "exec", "arguments": {"command": "ls"}}</tool_call>'
        )
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        # The valid block should still be recovered
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "exec"

    def test_all_blocks_malformed_returns_original(self):
        """If all blocks are malformed, return original content unchanged."""
        xml_content = "<tool_call>not json</tool_call>"
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert content == xml_content
        assert tool_calls is None

    def test_no_xml_tags_unchanged(self):
        """Plain text content without XML tags returns unchanged."""
        content, tool_calls = recover_tool_calls_from_content(
            content="Just a plain response with no tool calls.",
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert content == "Just a plain response with no tool calls."
        assert tool_calls is None

    def test_none_content_unchanged(self):
        """None content returns unchanged."""
        content, tool_calls = recover_tool_calls_from_content(
            content=None,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert content is None
        assert tool_calls is None

    def test_empty_content_unchanged(self):
        """Empty string content returns unchanged."""
        content, tool_calls = recover_tool_calls_from_content(
            content="",
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert content == ""
        assert tool_calls is None

    def test_arguments_as_string(self):
        """Arguments provided as a JSON string (not dict) are preserved."""
        xml_content = (
            '<tool_call>{"name": "exec", "arguments": "{\\"command\\": \\"ls\\"}"}</tool_call>'
        )
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == '{"command": "ls"}'

    def test_missing_name_field_skipped(self):
        """Tool call block without 'name' field is skipped."""
        xml_content = (
            '<tool_call>{"arguments": {"command": "ls"}}</tool_call>\n'
            '<tool_call>{"name": "exec", "arguments": {"command": "pwd"}}</tool_call>'
        )
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "exec"

    def test_missing_arguments_defaults_to_empty_object(self):
        """Tool call with missing arguments defaults to '{}'."""
        xml_content = '<tool_call>{"name": "get_status"}</tool_call>'
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == "{}"

    def test_tool_result_blocks_stripped(self):
        """<tool_result> blocks are removed from cleaned content."""
        xml_content = (
            "Checking...\n"
            '<tool_call>{"name": "exec", "arguments": {"command": "ls"}}</tool_call>\n'
            '<tool_result id="123">file1.txt\nfile2.txt</tool_result>\n'
            "Done."
        )
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert "<tool_result" not in content
        assert "Checking..." in content
        assert "Done." in content

    def test_unique_tool_call_ids(self):
        """Each recovered tool call gets a unique ID."""
        xml_content = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        _, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=None,
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] != tool_calls[1]["id"]

    def test_empty_tool_calls_list_triggers_recovery(self):
        """An empty list (not None) for tool_calls also triggers recovery."""
        xml_content = '<tool_call>{"name": "exec", "arguments": {"command": "ls"}}</tool_call>'
        content, tool_calls = recover_tool_calls_from_content(
            content=xml_content,
            tool_calls=[],
            finish_reason="tool_calls",
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "exec"


class TestRecoverInvokeToolCalls:
    """Tests for recover_invoke_tool_calls (antml/<invoke> dialect)."""

    def test_single_invoke_recovered(self):
        """A single <invoke> block becomes one OpenAI tool_call."""
        text = (
            "Let me check.\n\n"
            '<invoke name="Bash">\n'
            '<parameter name="command">ls -la</parameter>\n'
            "</invoke>"
        )
        calls = recover_invoke_tool_calls(text, allowed_names={"Bash"})
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "Bash"
        assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}
        assert calls[0]["id"].startswith("toolu_")
        assert calls[0]["index"] == 0

    def test_multiple_invokes_get_distinct_indexes(self):
        """Multiple leaked calls receive distinct, ascending indexes.

        Regression for the index-collision (all index=0) hazard that would
        overwrite tool_use blocks in the streaming translator.
        """
        text = (
            '<invoke name="Read">\n'
            '<parameter name="file_path">a.ts</parameter>\n'
            "</invoke>\n"
            '<invoke name="Grep">\n'
            '<parameter name="pattern">foo</parameter>\n'
            "</invoke>"
        )
        calls = recover_invoke_tool_calls(text, allowed_names={"Read", "Grep"})
        assert calls is not None
        assert [c["index"] for c in calls] == [0, 1]
        assert [c["function"]["name"] for c in calls] == ["Read", "Grep"]
        assert calls[0]["id"] != calls[1]["id"]

    def test_invoke_inside_code_fence_ignored(self):
        """A fenced code block discussing <invoke> is NOT a real call.

        This is the meta-discussion false-positive guard: a model explaining
        the tool-call syntax must not have its explanation executed.
        """
        text = (
            "The syntax looks like this:\n"
            "```\n"
            '<invoke name="Bash">\n'
            '<parameter name="command">rm -rf /</parameter>\n'
            "</invoke>\n"
            "```\n"
            "as you can see above."
        )
        assert recover_invoke_tool_calls(text, allowed_names={"Bash"}) is None

    def test_invoke_inside_inline_code_ignored(self):
        """An inline-code mention of <invoke> is ignored."""
        text = 'You can call `<invoke name="Bash"></invoke>` to run things.'
        assert recover_invoke_tool_calls(text, allowed_names={"Bash"}) is None

    def test_unclosed_invoke_ignored(self):
        """A truncated <invoke> with no closing tag is left as plain text.

        Regression for the straddling/connection-terminated case: a half
        streamed XML fragment must not be silently dropped or half-parsed.
        """
        text = '<invoke name="Bash">\n<parameter name="command">ls'
        assert recover_invoke_tool_calls(text, allowed_names={"Bash"}) is None

    def test_unknown_tool_name_dropped(self):
        """A call to a tool not offered by the request is dropped."""
        text = '<invoke name="Nonexistent"><parameter name="x">1</parameter></invoke>'
        assert recover_invoke_tool_calls(text, allowed_names={"Bash"}) is None

    def test_allowed_names_none_skips_cross_check(self):
        """When allowed_names is None the name cross-check is skipped."""
        text = '<invoke name="Anything"><parameter name="x">1</parameter></invoke>'
        calls = recover_invoke_tool_calls(text, allowed_names=None)
        assert calls is not None
        assert calls[0]["function"]["name"] == "Anything"

    def test_plain_text_returns_none(self):
        """Ordinary prose with no <invoke> returns None."""
        assert recover_invoke_tool_calls("just a normal answer", allowed_names=None) is None

    def test_empty_content_returns_none(self):
        """Empty / None content returns None."""
        assert recover_invoke_tool_calls("", allowed_names=None) is None
        assert recover_invoke_tool_calls(None, allowed_names=None) is None

    def test_invoke_without_parameters_recovered_with_empty_args(self):
        """An <invoke> with no parameters yields empty-object arguments."""
        text = '<invoke name="ListTools"></invoke>'
        calls = recover_invoke_tool_calls(text, allowed_names={"ListTools"})
        assert calls is not None
        assert json.loads(calls[0]["function"]["arguments"]) == {}

    def test_backtick_in_param_value_preserved(self):
        """A parameter value containing backticks must not be corrupted.

        Regression for PR #110 review: code-span stripping used to run over the
        whole text and erased backticked content inside real tool calls (Bash
        commands routinely contain `` `date` ``).
        """
        text = (
            "Let me check the date.\n"
            '<invoke name="Bash">\n'
            '<parameter name="command">echo `date` && ls</parameter>\n'
            "</invoke>"
        )
        calls = recover_invoke_tool_calls(text, allowed_names={"Bash"})
        assert calls is not None
        assert json.loads(calls[0]["function"]["arguments"]) == {"command": "echo `date` && ls"}

    def test_stray_prose_backtick_does_not_erase_invoke(self):
        """A lone backtick in prose must not pair across the invoke boundary.

        Regression for PR #110 review: an unpaired backtick before a well-formed
        invoke used to pair with a backtick inside the invoke body, deleting the
        whole block and returning None — reproducing the stall this code fixes.
        """
        text = (
            "check the `git log output for the "
            '<invoke name="Bash"><parameter name="command">ls `foo`</parameter></invoke>'
        )
        calls = recover_invoke_tool_calls(text, allowed_names={"Bash"})
        assert calls is not None
        assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls `foo`"}

    def test_param_value_containing_fenced_code_preserved(self):
        """A Write-style param whose value is itself a ``` fenced block is kept.

        The inner fence must not be treated as a document-level code span that
        would suppress the (real) tool call.
        """
        text = (
            '<invoke name="Write">'
            '<parameter name="content">```python\nprint(1)\n```</parameter>'
            "</invoke>"
        )
        calls = recover_invoke_tool_calls(text, allowed_names={"Write"})
        assert calls is not None
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "content": "```python\nprint(1)\n```"
        }

    def test_fenced_meta_and_real_invoke_mixed(self):
        """Only the real invoke is recovered; the fenced example is rejected."""
        text = (
            "Here is the syntax:\n"
            "```\n"
            '<invoke name="Bash"><parameter name="command">evil</parameter></invoke>\n'
            "```\n"
            'now really: <invoke name="Bash"><parameter name="command">real</parameter></invoke>'
        )
        calls = recover_invoke_tool_calls(text, allowed_names={"Bash"})
        assert calls is not None
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"command": "real"}
