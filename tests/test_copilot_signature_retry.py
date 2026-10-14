"""Copilot native-Anthropic thinking-signature strip-and-retry recovery.

A rejected thinking-block signature otherwise poisons every later turn: the
client replays the same history and 400s forever. The recovery lives on the
PRODUCTION path — ``binding.prepare_attempt -> executor.execute/execute_stream``
(``CopilotHttpExecutor``), not the legacy provider methods the non-legacy
dispatcher never calls. These tests drive that binding/executor path.
"""

from __future__ import annotations

import json

import httpx
import pytest

from router_maestro.protocols import WireProtocol
from router_maestro.providers.copilot import CopilotProvider
from router_maestro.routing.model_ref import ModelRef


async def _noop() -> None:
    return None


_SIGNATURE_400 = json.dumps(
    {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "messages.1.content.0: Invalid `signature` in `thinking` block",
        },
    }
).encode()

_NONSTREAM_200 = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4.8",
    "content": [{"type": "text", "text": "recovered"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 2},
}

_STREAM_200 = (
    b'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-4.8",'
    b'"role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,'
    b'"usage":{"input_tokens":3,"output_tokens":0}}}\n\n'
    b'data: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n'
    b'data: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"recovered"}}\n\n'
    b'data: {"type":"content_block_stop","index":0}\n\n'
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
    b'"usage":{"output_tokens":2}}\n\n'
    b'data: {"type":"message_stop"}\n\n'
)


def _binding(provider: CopilotProvider, protocol: WireProtocol):
    return next(binding for binding in provider.bindings() if binding.protocol is protocol)


def _payload(assistant_content: list[dict]) -> dict:
    return {
        "model": "claude-opus-4.8",
        "max_tokens": 64,
        "thinking": {"type": "adaptive"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": [{"type": "text", "text": "again"}]},
        ],
    }


def _signed_thinking_payload() -> dict:
    # A validly-signed thinking block survives the proactive drop, so it reaches
    # upstream on the first attempt and exercises the reactive strip-and-retry
    # when upstream rejects a signature it does not accept.
    return _payload(
        [
            {"type": "thinking", "thinking": "reason", "signature": "s" * 64},
            {"type": "text", "text": "prior"},
        ]
    )


def _provider_with_handler(handler) -> CopilotProvider:
    provider = CopilotProvider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider.ensure_token = _noop  # type: ignore[method-assign]
    provider._get_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    return provider


def _model(provider: CopilotProvider) -> ModelRef:
    return ModelRef(provider=provider.name, upstream_id="claude-opus-4.8")


@pytest.mark.asyncio
async def test_executor_strips_thinking_and_retries_on_signature_400():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(400, content=_SIGNATURE_400)
        return httpx.Response(200, json=_NONSTREAM_200)

    provider = _provider_with_handler(handler)
    binding = _binding(provider, WireProtocol.ANTHROPIC_MESSAGES)
    attempt = await binding.prepare_attempt(
        model=_model(provider), payload=_signed_thinking_payload(), stream=False
    )
    result = await binding.executor.execute(attempt)

    # Projected native Anthropic response reaches the caller.
    assert any(b.get("text") == "recovered" for b in result["content"])
    assert len(calls) == 2, "should retry exactly once through the executor"
    first_assistant = calls[0]["messages"][1]["content"]
    retry_assistant = calls[1]["messages"][1]["content"]
    assert any(b.get("type") == "thinking" for b in first_assistant)
    assert all(b.get("type") != "thinking" for b in retry_assistant)
    assert any(b.get("type") == "text" for b in retry_assistant)


@pytest.mark.asyncio
async def test_executor_stream_strips_thinking_and_retries_on_signature_400():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(400, content=_SIGNATURE_400)
        return httpx.Response(
            200, content=_STREAM_200, headers={"content-type": "text/event-stream"}
        )

    provider = _provider_with_handler(handler)
    binding = _binding(provider, WireProtocol.ANTHROPIC_MESSAGES)
    attempt = await binding.prepare_attempt(
        model=_model(provider), payload=_signed_thinking_payload(), stream=True
    )
    frames = [frame async for frame in binding.executor.execute_stream(attempt)]

    text = "".join(
        f["delta"]["text"]
        for f in frames
        if f.get("type") == "content_block_delta" and f["delta"].get("type") == "text_delta"
    )
    assert text == "recovered"
    assert len(calls) == 2, "should retry exactly once through the executor"
    retry_assistant = calls[1]["messages"][1]["content"]
    assert all(b.get("type") != "thinking" for b in retry_assistant)


@pytest.mark.asyncio
async def test_executor_does_not_retry_on_unrelated_400():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            400,
            content=b'{"error":{"type":"invalid_request_error","message":"bad tool schema"}}',
        )

    provider = _provider_with_handler(handler)
    binding = _binding(provider, WireProtocol.ANTHROPIC_MESSAGES)
    attempt = await binding.prepare_attempt(
        model=_model(provider), payload=_signed_thinking_payload(), stream=False
    )
    with pytest.raises(Exception):
        await binding.executor.execute(attempt)

    assert len(calls) == 1, "non-signature 400 must not trigger a retry"


@pytest.mark.asyncio
async def test_proactive_drop_preserves_signed_thinking_in_tool_use_turn():
    """A signed thinking + tool_use turn (a live tool loop) is left intact.

    Anthropic requires the assistant tool-use turn to be echoed complete and
    unmodified, including its thinking blocks. ``drop_unsigned_thinking`` only
    removes UNSIGNED reasoning, so a correctly-signed tool loop is never
    filtered — the payload forwarded upstream keeps the thinking block.
    """
    forwarded: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded.append(json.loads(request.content))
        return httpx.Response(200, json=_NONSTREAM_200)

    provider = _provider_with_handler(handler)
    tool_use_turn = [
        {"type": "thinking", "thinking": "let me check", "signature": "sig" * 30},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"location": "Paris"},
        },
    ]
    payload = {
        "model": "claude-opus-4.8",
        "max_tokens": 64,
        "thinking": {"type": "adaptive"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "weather in Paris?"}]},
            {"role": "assistant", "content": tool_use_turn},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "88F"}],
            },
        ],
    }

    binding = _binding(provider, WireProtocol.ANTHROPIC_MESSAGES)
    attempt = await binding.prepare_attempt(model=_model(provider), payload=payload, stream=False)
    await binding.executor.execute(attempt)

    assert len(forwarded) == 1, "no retry: the signed tool-use turn is accepted as-is"
    sent_turn = forwarded[0]["messages"][1]["content"]
    # The signed thinking block is preserved alongside the tool_use.
    assert [b.get("type") for b in sent_turn] == ["thinking", "tool_use"]


@pytest.mark.asyncio
async def test_proactive_drop_removes_unsigned_thinking_from_tool_use_turn():
    """An UNSIGNED thinking block in a tool_use turn is the poison and is dropped.

    The unsigned block is rejected by upstream no matter what, so removing it is
    the only way the turn can succeed. Verified live that Copilot accepts a
    tool_use turn with the (unsigned) thinking removed.
    """
    forwarded: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded.append(json.loads(request.content))
        return httpx.Response(200, json=_NONSTREAM_200)

    provider = _provider_with_handler(handler)
    tool_use_turn = [
        {"type": "thinking", "thinking": "unsigned poison", "signature": ""},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"location": "Paris"},
        },
    ]
    payload = {
        "model": "claude-opus-4.8",
        "max_tokens": 64,
        "thinking": {"type": "adaptive"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "weather in Paris?"}]},
            {"role": "assistant", "content": tool_use_turn},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "88F"}],
            },
        ],
    }

    binding = _binding(provider, WireProtocol.ANTHROPIC_MESSAGES)
    attempt = await binding.prepare_attempt(model=_model(provider), payload=payload, stream=False)
    await binding.executor.execute(attempt)

    assert len(forwarded) == 1, "proactive drop fixes it without an upstream retry"
    sent_turn = forwarded[0]["messages"][1]["content"]
    # Unsigned thinking dropped; tool_use preserved so the tool_result still pairs.
    assert [b.get("type") for b in sent_turn] == ["tool_use"]


@pytest.mark.asyncio
async def test_executor_stream_retry_records_first_400_body_to_audit(tmp_path):
    """The rejected-signature 400 body is captured in the audit trace.

    The streaming transport only records ``stream opened`` (no body), so without
    this the strip-and-retry recovery would be invisible in traces.
    """
    from types import SimpleNamespace

    from router_maestro.runtime import request_context as rc
    from router_maestro.utils.audit import AuditTrace

    def handler(request: httpx.Request) -> httpx.Response:
        if not getattr(handler, "seen", False):
            handler.seen = True  # type: ignore[attr-defined]
            return httpx.Response(400, content=_SIGNATURE_400)
        return httpx.Response(
            200, content=_STREAM_200, headers={"content-type": "text/event-stream"}
        )

    provider = _provider_with_handler(handler)
    binding = _binding(provider, WireProtocol.ANTHROPIC_MESSAGES)
    attempt = await binding.prepare_attempt(
        model=_model(provider), payload=_signed_thinking_payload(), stream=True
    )

    trace = AuditTrace("req-sig-retry", tmp_path)
    token = rc._current_request_context.set(SimpleNamespace(audit=trace))
    try:
        _ = [frame async for frame in binding.executor.execute_stream(attempt)]
    finally:
        rc._current_request_context.reset(token)
    trace.flush()

    # Records are append-only (upstream_resp.json, upstream_resp_2.json, ...);
    # the signature-retry body lands in whichever record carries a body.
    trace_dir = tmp_path / "req-sig-retry"
    bodies = [
        json.loads(path.read_text())
        for path in trace_dir.glob("upstream_resp*.json")
        if "body" in json.loads(path.read_text())
    ]
    assert bodies, "the rejected-signature 400 body should be recorded"
    dumped = json.dumps(bodies[0])
    assert bodies[0]["status"] == 400
    assert "Invalid" in dumped and "signature" in dumped.lower()
