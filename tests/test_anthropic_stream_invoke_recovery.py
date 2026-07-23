"""Regression tests for recovering <invoke>-dialect tool calls leaked as text
into the Anthropic streaming path.

Some upstreams (notably github-copilot/claude-* under long contexts) emit a
tool call as literal ``<invoke name="...">`` XML in the assistant *text*
instead of the structured tool_calls field. Forwarded verbatim, the client
(e.g. Claude Code) shows the XML and the agent stalls because the tool never
runs. ``stream_response`` recovers such leaks at the finish chunk, synthesizing
a real ``tool_use`` block and promoting the stop reason to ``tool_use``.
"""

import json
from collections.abc import AsyncGenerator

import pytest

from router_maestro.providers import ChatRequest, Message
from router_maestro.providers.base import ChatStreamChunk
from router_maestro.server.routes.anthropic import stream_response


def _bash_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }


async def _fake_stream(*chunks: ChatStreamChunk):
    async def gen() -> AsyncGenerator[ChatStreamChunk]:
        for c in chunks:
            yield c

    return gen(), "github-copilot"


def _parse_events(frames: list[str]) -> list[dict]:
    events = []
    for frame in frames:
        for line in frame.split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[len("data: ") :]))
                except json.JSONDecodeError:
                    pass
    return events


async def _run(chunks, *, tools=None) -> list[dict]:
    request = ChatRequest(
        model="claude-opus-4.8",
        messages=[Message(role="user", content="do it")],
        stream=True,
        tools=tools,
    )

    class _Router:
        async def chat_completion_stream(self, req):
            return await _fake_stream(*chunks)

    frames = [frame async for frame in stream_response(_Router(), request, "claude-opus-4-8")]
    return _parse_events(frames)


@pytest.mark.asyncio
async def test_leaked_invoke_becomes_tool_use_block():
    """Leaked <invoke> text + finish_reason=stop yields a real tool_use block."""
    leaked = (
        "Let me check.\n\n"
        '<invoke name="Bash">\n'
        '<parameter name="command">ls -la</parameter>\n'
        "</invoke>"
    )
    events = await _run(
        [
            ChatStreamChunk(content=leaked, finish_reason=None),
            ChatStreamChunk(content="", finish_reason="stop"),
        ],
        tools=[_bash_tool()],
    )
    types = [e.get("type") for e in events]

    # A tool_use content block was synthesized.
    tool_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "Bash"

    # Its arguments were streamed as input_json_delta.
    arg_json = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
    )
    assert json.loads(arg_json) == {"command": "ls -la"}

    # Stop reason promoted to tool_use, and the stream is properly terminated.
    msg_delta = next(e for e in events if e.get("type") == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "tool_use"
    assert "message_stop" in types


@pytest.mark.asyncio
async def test_meta_discussion_invoke_not_converted():
    """A fenced code block mentioning <invoke> stays plain text (end_turn)."""
    meta = (
        "The format that broke was:\n"
        "```\n"
        '<invoke name="Bash">\n'
        '<parameter name="command">ls</parameter>\n'
        "</invoke>\n"
        "```\n"
        "so I will retry properly."
    )
    events = await _run(
        [
            ChatStreamChunk(content=meta, finish_reason=None),
            ChatStreamChunk(content="", finish_reason="stop"),
        ],
        tools=[_bash_tool()],
    )

    tool_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert tool_starts == []
    msg_delta = next(e for e in events if e.get("type") == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_plain_text_stream_unchanged():
    """An ordinary text answer is unaffected (no tool_use, end_turn)."""
    events = await _run(
        [
            ChatStreamChunk(content="Here is the answer.", finish_reason=None),
            ChatStreamChunk(content="", finish_reason="stop"),
        ],
        tools=[_bash_tool()],
    )
    assert not any(
        e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
        for e in events
    )
    msg_delta = next(e for e in events if e.get("type") == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_real_tool_call_not_double_recovered():
    """A genuine structured tool_call is not re-recovered from text."""
    real_call = [
        {
            "id": "toolu_real",
            "type": "function",
            "index": 0,
            "function": {"name": "Bash", "arguments": '{"command": "pwd"}'},
        }
    ]
    events = await _run(
        [
            ChatStreamChunk(content="", finish_reason=None, tool_calls=real_call),
            ChatStreamChunk(content="", finish_reason="tool_calls"),
        ],
        tools=[_bash_tool()],
    )
    tool_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
    ]
    # Exactly one tool_use block — the real one, not a duplicated recovery.
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["id"] == "toolu_real"


@pytest.mark.asyncio
async def test_truncated_invoke_left_as_text():
    """A half-streamed <invoke> (no close) is forwarded as text, not dropped."""
    truncated = '<invoke name="Bash">\n<parameter name="command">ls'
    events = await _run(
        [
            ChatStreamChunk(content=truncated, finish_reason=None),
            ChatStreamChunk(content="", finish_reason="length"),
        ],
        tools=[_bash_tool()],
    )
    assert not any(
        e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
        for e in events
    )
    # Text was still delivered to the client.
    text = "".join(
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
    )
    assert "<invoke" in text
    # max_tokens stop maps to Anthropic max_tokens, stream still terminates.
    assert any(e.get("type") == "message_stop" for e in events)
