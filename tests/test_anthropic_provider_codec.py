"""Characterization tests for the native Anthropic provider codec."""

import json
from unittest.mock import patch

import httpx
import pytest

from router_maestro.providers import ChatRequest, Message, ProviderError, ProviderFailureKind
from router_maestro.providers.anthropic import AnthropicProvider
from router_maestro.providers.anthropic_codec import (
    AnthropicCodecError,
    AnthropicStopCause,
    AnthropicStreamDecoder,
    canonical_stop_reason,
    decode_message_response,
    parse_stop_cause,
    project_stop_cause,
)
from router_maestro.server.routes.anthropic import stream_response as anthropic_stream_response


@pytest.mark.parametrize(
    ("native", "canonical"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
    ],
)
def test_native_stop_reason_has_one_canonical_mapping(native: str, canonical: str) -> None:
    assert canonical_stop_reason(native) == canonical


def test_null_native_stop_reason_projects_to_nonterminal() -> None:
    assert canonical_stop_reason(None) is None


@pytest.mark.parametrize("native", ["pause_turn", "refusal", "future_reason"])
def test_noncanonical_native_stop_reason_is_not_guessed(native: str) -> None:
    with pytest.raises(
        AnthropicCodecError,
        match="cannot be projected" if native in {"pause_turn", "refusal"} else "unsupported",
    ):
        canonical_stop_reason(native)


def test_native_stop_parsing_is_lossless_before_chat_projection() -> None:
    assert parse_stop_cause("pause_turn") is AnthropicStopCause.PAUSE_TURN
    assert parse_stop_cause("refusal") is AnthropicStopCause.REFUSAL
    with pytest.raises(AnthropicCodecError, match="cannot be projected"):
        project_stop_cause(AnthropicStopCause.PAUSE_TURN)
    with pytest.raises(AnthropicCodecError, match="cannot be projected"):
        project_stop_cause(AnthropicStopCause.REFUSAL)


def test_unknown_native_stop_fails_during_typed_parsing() -> None:
    with pytest.raises(AnthropicCodecError, match="unsupported stop_reason"):
        parse_stop_cause("future_reason")


def test_nonstream_missing_stop_reason_preserves_legacy_stop_default() -> None:
    response = decode_message_response(
        {"content": [{"type": "text", "text": "answer"}]},
        fallback_model="claude-requested",
        include_reasoning=False,
    )

    assert response.finish_reason == "stop"


def test_nonstream_explicit_null_stop_reason_is_not_a_success_terminal() -> None:
    with pytest.raises(AnthropicCodecError, match="stop_reason cannot be null"):
        decode_message_response(
            {
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": None,
            },
            fallback_model="claude-requested",
            include_reasoning=False,
        )


@pytest.mark.parametrize(
    ("native", "canonical"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
    ],
)
def test_stream_and_nonstream_share_terminal_mapping(native: str, canonical: str) -> None:
    response = decode_message_response(
        {
            "model": "claude-upstream",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": native,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        },
        fallback_model="claude-requested",
        include_reasoning=False,
    )
    decoder = AnthropicStreamDecoder(include_reasoning=False)
    decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )
    decoder.decode_event(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "done"},
        }
    )
    chunks = decoder.decode_event(
        {
            "type": "message_delta",
            "delta": {"stop_reason": native},
            "usage": {"output_tokens": 3},
        }
    )

    assert response.finish_reason == canonical
    assert [chunk.finish_reason for chunk in chunks] == [canonical]


def test_stream_and_nonstream_preserve_content_reasoning_tools_and_usage() -> None:
    response = decode_message_response(
        {
            "model": "claude-upstream",
            "content": [
                {"type": "thinking", "thinking": "consider", "signature": "opaque"},
                {"type": "text", "text": "answer"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {"q": "router"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        },
        fallback_model="claude-requested",
        include_reasoning=True,
    )
    decoder = AnthropicStreamDecoder(include_reasoning=True)
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "consider"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "opa"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "que"},
        },
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "answer"},
        },
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"q": "router"}'},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 7},
        },
    ]
    chunks = [chunk for event in events for chunk in decoder.decode_event(event)]

    streamed_tool = next(chunk.tool_calls[0] for chunk in chunks if chunk.tool_calls)
    tool_fragments = [chunk.tool_calls[0] for chunk in chunks if chunk.tool_calls]
    assert response.model == "claude-upstream"
    assert response.content == "".join(chunk.content for chunk in chunks) == "answer"
    assert response.thinking == "".join(chunk.thinking or "" for chunk in chunks) == "consider"
    assert response.thinking_signature == "".join(
        chunk.thinking_signature or "" for chunk in chunks
    )
    assert response.tool_calls == [
        {
            "id": streamed_tool["id"],
            "type": "function",
            "function": {
                "name": streamed_tool["function"]["name"],
                "arguments": "".join(
                    fragment.get("function", {}).get("arguments", "") for fragment in tool_fragments
                ),
            },
        }
    ]
    assert response.finish_reason == chunks[-1].finish_reason == "tool_calls"
    assert (
        response.usage
        == chunks[-1].usage
        == {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        }
    )


def test_redacted_reasoning_signature_is_equivalent_when_requested() -> None:
    response = decode_message_response(
        {
            "content": [
                {"type": "redacted_thinking", "data": "opaque-redacted"},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
        },
        fallback_model="claude-requested",
        include_reasoning=True,
    )
    decoder = AnthropicStreamDecoder(include_reasoning=True)
    chunks = decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque-redacted"},
        }
    )

    assert response.thinking_signature == chunks[0].thinking_signature == "opaque-redacted"


def test_reasoning_is_suppressed_consistently_when_not_requested() -> None:
    response = decode_message_response(
        {
            "content": [
                {"type": "thinking", "thinking": "private", "signature": "opaque"},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
        },
        fallback_model="claude-requested",
        include_reasoning=False,
    )
    decoder = AnthropicStreamDecoder(include_reasoning=False)
    decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        }
    )
    chunks = decoder.decode_event(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "private"},
        }
    )

    assert response.thinking is None
    assert response.thinking_signature is None
    assert chunks == []


def _provider_request() -> ChatRequest:
    return ChatRequest(model="claude-requested", messages=[Message(role="user", content="hi")])


def _sse(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


async def _provider_stream(
    *events: dict,
    thinking_type: str | None = None,
) -> list:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=_sse(*events)))
    )
    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    request = ChatRequest(
        model="claude-requested",
        messages=[Message(role="user", content="hi")],
        stream=True,
        thinking_type=thinking_type,
    )
    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        return [chunk async for chunk in provider.chat_completion_stream(request)]


@pytest.mark.asyncio
async def test_provider_nonstream_uses_codec_canonical_stop_reason() -> None:
    payload = {
        "model": "claude-upstream",
        "content": [{"type": "text", "text": "answer"}],
        "stop_reason": "end_turn",
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    )
    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]

    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        response = await provider.chat_completion(_provider_request())

    assert response.model == "claude-upstream"
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_provider_wraps_unknown_stop_as_typed_upstream_protocol() -> None:
    payload = {
        "content": [{"type": "text", "text": "answer"}],
        "stop_reason": "pause_turn",
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    )
    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]

    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as caught:
            await provider.chat_completion(_provider_request())

    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert caught.value.provider == "anthropic"
    assert caught.value.model == "claude-requested"
    assert "pause_turn" not in str(caught.value)


@pytest.mark.asyncio
async def test_provider_stream_suppresses_unrequested_reasoning() -> None:
    chunks = await _provider_stream(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "private"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "public"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    )

    assert all(chunk.thinking is None for chunk in chunks)


@pytest.mark.asyncio
async def test_provider_stream_preserves_requested_redacted_signature() -> None:
    chunks = await _provider_stream(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque-redacted"},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        thinking_type="enabled",
    )

    assert [chunk.thinking_signature for chunk in chunks if chunk.thinking_signature] == [
        "opaque-redacted"
    ]


@pytest.mark.asyncio
async def test_provider_stream_null_stop_is_nonterminal_usage() -> None:
    chunks = await _provider_stream(
        {
            "type": "message_delta",
            "delta": {"stop_reason": None},
            "usage": {"output_tokens": 1},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "complete"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
    )

    assert chunks[0].finish_reason is None
    assert chunks[0].usage == {
        "prompt_tokens": 0,
        "completion_tokens": 1,
        "total_tokens": 1,
    }


@pytest.mark.asyncio
async def test_provider_stream_wraps_unknown_stop_as_typed_upstream_protocol() -> None:
    with pytest.raises(ProviderError) as caught:
        await _provider_stream(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "pause_turn"},
                "usage": {"output_tokens": 1},
            }
        )

    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert caught.value.provider == "anthropic"
    assert caught.value.model == "claude-requested"
    assert "pause_turn" not in str(caught.value)


@pytest.mark.parametrize(
    "events",
    [
        pytest.param([], id="empty"),
        pytest.param([{"type": "ping"}], id="unknown-only"),
        pytest.param(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "partial"},
                },
                {"type": "content_block_stop", "index": 0},
            ],
            id="content-without-terminal",
        ),
        pytest.param(
            [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": None},
                    "usage": {"output_tokens": 1},
                }
            ],
            id="null-stop-usage",
        ),
    ],
)
def test_stream_decoder_rejects_incomplete_normal_eof(events: list[dict]) -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)
    for event in events:
        decoder.decode_event(event)

    with pytest.raises(AnthropicCodecError):
        decoder.finalize()


def test_stream_decoder_accepts_deliverable_output_with_explicit_terminal() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)
    events = [
        {"type": "ping"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "complete"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    ]
    chunks = [chunk for event in events for chunk in decoder.decode_event(event)]

    decoder.finalize()

    assert "".join(chunk.content for chunk in chunks) == "complete"
    assert chunks[-1].finish_reason == "stop"


def test_stream_decoder_rejects_explicit_terminal_without_deliverable_output() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    with pytest.raises(AnthropicCodecError):
        decoder.decode_event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        pytest.param([], id="empty"),
        pytest.param([{"type": "ping", "opaque": "do-not-leak"}], id="unknown-only"),
        pytest.param(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "secret-partial"},
                },
                {"type": "content_block_stop", "index": 0},
            ],
            id="content-without-terminal",
        ),
        pytest.param(
            [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": None},
                    "usage": {"output_tokens": 1},
                }
            ],
            id="null-stop-usage",
        ),
    ],
)
async def test_provider_stream_wraps_incomplete_normal_eof_as_protocol_error(
    events: list[dict],
) -> None:
    with pytest.raises(ProviderError) as caught:
        await _provider_stream(*events)

    assert caught.value.status_code == 502
    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert caught.value.provider == "anthropic"
    assert caught.value.model == "claude-requested"
    assert str(caught.value) == "anthropic returned a malformed upstream response"
    assert "do-not-leak" not in str(caught.value)
    assert "secret-partial" not in str(caught.value)


@pytest.mark.asyncio
async def test_provider_stream_accepts_unknown_events_around_complete_output() -> None:
    chunks = await _provider_stream(
        {"type": "ping"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "complete"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    )

    assert "".join(chunk.content for chunk in chunks) == "complete"
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_route_rejects_native_terminal_before_any_deliverable_output() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=_sse(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 1},
                    }
                ),
            )
        )
    )
    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]
    request = ChatRequest(
        model="claude-requested",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    class ProviderBackedRouter:
        async def chat_completion_stream(self, route_request, **_kwargs):
            return provider.chat_completion_stream(route_request), provider.name

    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        frames = [
            frame
            async for frame in anthropic_stream_response(
                ProviderBackedRouter(),  # type: ignore[arg-type]
                request,
                request.model,
            )
        ]

    event_types = [
        line.removeprefix("event: ")
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("event: ")
    ]
    assert event_types.count("error") == 1
    assert "message_stop" not in event_types


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_nonstream_codec_rejects_negative_token_counts(field: str) -> None:
    with pytest.raises(AnthropicCodecError, match="non-negative integer"):
        decode_message_response(
            {
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "usage": {field: -1},
            },
            fallback_model="claude-requested",
            include_reasoning=False,
        )


def test_stream_codec_rejects_negative_input_tokens() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    with pytest.raises(AnthropicCodecError, match="non-negative integer"):
        decoder.decode_event(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": -1}},
            }
        )


def test_stream_codec_rejects_negative_output_tokens() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    with pytest.raises(AnthropicCodecError, match="non-negative integer"):
        decoder.decode_event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": -1},
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
async def test_provider_nonstream_wraps_negative_token_count_as_protocol_error(field: str) -> None:
    payload = {
        "content": [{"type": "text", "text": "answer"}],
        "stop_reason": "end_turn",
        "usage": {field: -1},
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    )
    provider = AnthropicProvider()
    provider._get_headers = lambda: {}  # type: ignore[method-assign]

    with patch("router_maestro.providers.anthropic.httpx.AsyncClient", return_value=client):
        with pytest.raises(ProviderError) as caught:
            await provider.chat_completion(_provider_request())

    assert caught.value.status_code == 502
    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert caught.value.provider == "anthropic"
    assert caught.value.model == "claude-requested"
    assert str(caught.value) == "anthropic returned a malformed upstream response"
    assert "-1" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": -1}},
            },
            id="input-tokens",
        ),
        pytest.param(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": -1},
            },
            id="output-tokens",
        ),
    ],
)
async def test_provider_stream_wraps_negative_token_count_as_protocol_error(
    event: dict,
) -> None:
    with pytest.raises(ProviderError) as caught:
        await _provider_stream(event)

    assert caught.value.status_code == 502
    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert caught.value.provider == "anthropic"
    assert caught.value.model == "claude-requested"
    assert str(caught.value) == "anthropic returned a malformed upstream response"
    assert "-1" not in str(caught.value)


def test_tool_block_start_preserves_nonempty_initial_input_as_complete_json() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    chunks = decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 3,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_seeded",
                "name": "lookup",
                "input": {"q": "router", "limit": 2},
            },
        }
    )

    assert chunks[0].tool_calls == [
        {
            "index": 0,
            "id": "toolu_seeded",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": json.dumps({"q": "router", "limit": 2}),
            },
        }
    ]


def test_tool_block_rejects_partial_json_after_nonempty_initial_input() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)
    decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 3,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_seeded",
                "name": "lookup",
                "input": {"q": "router"},
            },
        }
    )

    with pytest.raises(AnthropicCodecError, match="initial input"):
        decoder.decode_event(
            {
                "type": "content_block_delta",
                "index": 3,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"router"}'},
            }
        )


def test_text_block_start_emits_nonempty_initial_text() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    chunks = decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "seed text"},
        }
    )

    assert [chunk.content for chunk in chunks] == ["seed text"]


def test_text_block_start_rejects_nonstring_initial_text() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    with pytest.raises(AnthropicCodecError, match="text block text must be a string"):
        decoder.decode_event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": None},
            }
        )


def test_thinking_block_start_emits_seed_and_signature_when_requested() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=True)

    chunks = decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "seed thought",
                "signature": "seed signature",
            },
        }
    )

    assert "".join(chunk.thinking or "" for chunk in chunks) == "seed thought"
    assert "".join(chunk.thinking_signature or "" for chunk in chunks) == "seed signature"


def test_thinking_block_start_suppresses_seed_and_signature_when_not_requested() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    chunks = decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "private seed",
                "signature": "private signature",
            },
        }
    )

    assert chunks == []


@pytest.mark.parametrize(
    "content_block",
    [
        pytest.param({"type": "thinking", "thinking": None}, id="thinking"),
        pytest.param({"type": "thinking", "signature": None}, id="signature"),
    ],
)
def test_thinking_block_start_validates_fields_when_not_requested(
    content_block: dict,
) -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    with pytest.raises(AnthropicCodecError, match="must be a string"):
        decoder.decode_event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": content_block,
            }
        )


def test_stream_codec_rejects_duplicate_known_content_block_index() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)
    decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )

    with pytest.raises(AnthropicCodecError, match="duplicate content block index"):
        decoder.decode_event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_duplicate",
                    "name": "lookup",
                    "input": {},
                },
            }
        )


@pytest.mark.parametrize(
    ("block", "delta"),
    [
        (
            {"type": "text", "text": ""},
            {"type": "input_json_delta", "partial_json": "{}"},
        ),
        (
            {"type": "thinking", "thinking": "", "signature": ""},
            {"type": "input_json_delta", "partial_json": "{}"},
        ),
        (
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
            {"type": "text_delta", "text": "wrong block"},
        ),
    ],
    ids=["tool-delta-on-text", "tool-delta-on-thinking", "text-delta-on-tool"],
)
def test_stream_codec_rejects_delta_incompatible_with_started_block(
    block: dict,
    delta: dict,
) -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=True)
    decoder.decode_event({"type": "content_block_start", "index": 0, "content_block": block})

    with pytest.raises(AnthropicCodecError, match="incompatible"):
        decoder.decode_event({"type": "content_block_delta", "index": 0, "delta": delta})


def test_stream_codec_rejects_known_delta_after_content_block_stop() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)
    decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )
    decoder.decode_event({"type": "content_block_stop", "index": 0})

    with pytest.raises(AnthropicCodecError, match="closed content block"):
        decoder.decode_event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "late"},
            }
        )


def test_stream_codec_rejects_known_delta_without_content_block_start() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    with pytest.raises(AnthropicCodecError, match="before content block start"):
        decoder.decode_event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "orphan"},
            }
        )


def test_unknown_content_block_remains_forward_compatible() -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=False)

    assert (
        decoder.decode_event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "future_block", "opaque": {"version": 2}},
            }
        )
        == []
    )
    assert (
        decoder.decode_event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "future semantics"},
            }
        )
        == []
    )
    assert decoder.decode_event({"type": "content_block_stop", "index": 0}) == []


@pytest.mark.parametrize("second_signature", ["same", "different"])
def test_stream_and_nonstream_reject_signature_ownership_across_thinking_blocks(
    second_signature: str,
) -> None:
    with pytest.raises(AnthropicCodecError, match="signature.*multiple reasoning blocks"):
        decode_message_response(
            {
                "content": [
                    {"type": "thinking", "thinking": "first", "signature": "same"},
                    {
                        "type": "thinking",
                        "thinking": "second",
                        "signature": second_signature,
                    },
                ],
                "stop_reason": "end_turn",
            },
            fallback_model="claude-requested",
            include_reasoning=True,
        )

    decoder = AnthropicStreamDecoder(include_reasoning=True)
    decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "first",
                "signature": "same",
            },
        }
    )
    decoder.decode_event({"type": "content_block_stop", "index": 0})

    with pytest.raises(AnthropicCodecError, match="signature.*multiple reasoning blocks"):
        decoder.decode_event(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "thinking",
                    "thinking": "second",
                    "signature": second_signature,
                },
            }
        )


def test_stream_and_nonstream_allow_multiple_unsigned_thinking_blocks() -> None:
    response = decode_message_response(
        {
            "content": [
                {"type": "thinking", "thinking": "first", "signature": None},
                {"type": "thinking", "thinking": "second", "signature": None},
            ],
            "stop_reason": "end_turn",
        },
        fallback_model="claude-requested",
        include_reasoning=True,
    )
    decoder = AnthropicStreamDecoder(include_reasoning=True)
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "first"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking", "thinking": "second"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    ]
    chunks = [chunk for event in events for chunk in decoder.decode_event(event)]
    decoder.finalize()

    assert response.thinking == "".join(chunk.thinking or "" for chunk in chunks) == "firstsecond"
    assert response.thinking_signature is None
    assert all(chunk.thinking_signature is None for chunk in chunks)


def test_stream_and_nonstream_reject_redacted_signature_with_another_reasoning_block() -> None:
    with pytest.raises(AnthropicCodecError, match="signature.*multiple reasoning blocks"):
        decode_message_response(
            {
                "content": [
                    {"type": "thinking", "thinking": "first", "signature": "opaque"},
                    {"type": "redacted_thinking", "data": "redacted"},
                ],
                "stop_reason": "end_turn",
            },
            fallback_model="claude-requested",
            include_reasoning=True,
        )

    decoder = AnthropicStreamDecoder(include_reasoning=True)
    decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "first",
                "signature": "opaque",
            },
        }
    )
    decoder.decode_event({"type": "content_block_stop", "index": 0})

    with pytest.raises(AnthropicCodecError, match="signature.*multiple reasoning blocks"):
        decoder.decode_event(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "redacted_thinking", "data": "redacted"},
            }
        )


@pytest.mark.asyncio
async def test_provider_wraps_duplicate_content_block_start_as_safe_protocol_error() -> None:
    with pytest.raises(ProviderError) as caught:
        await _provider_stream(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "first"},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "private-duplicate"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
        )

    assert caught.value.status_code == 502
    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert caught.value.provider == "anthropic"
    assert caught.value.model == "claude-requested"
    assert str(caught.value) == "anthropic returned a malformed upstream response"
    assert "private-duplicate" not in str(caught.value)


@pytest.mark.parametrize(
    ("include_reasoning", "redacted"),
    [
        pytest.param(False, "opaque", id="reasoning-disabled-nonempty"),
        pytest.param(True, "", id="reasoning-enabled-empty"),
    ],
)
def test_stream_codec_suppresses_valid_redacted_thinking_without_signature(
    include_reasoning: bool,
    redacted: str,
) -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=include_reasoning)

    assert (
        decoder.decode_event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": redacted},
            }
        )
        == []
    )
    decoder.decode_event({"type": "content_block_stop", "index": 0})
    chunks = decoder.decode_event(
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": "public"},
        }
    )
    decoder.decode_event({"type": "content_block_stop", "index": 1})
    chunks.extend(
        decoder.decode_event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            }
        )
    )
    decoder.finalize()

    assert "".join(chunk.content for chunk in chunks) == "public"
    assert all(chunk.thinking_signature is None for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thinking_type", "redacted"),
    [
        pytest.param(None, "opaque", id="reasoning-disabled-nonempty"),
        pytest.param("enabled", "", id="reasoning-enabled-empty"),
    ],
)
async def test_provider_suppresses_valid_redacted_thinking_without_signature(
    thinking_type: str | None,
    redacted: str,
) -> None:
    chunks = await _provider_stream(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": redacted},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": "public"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
        thinking_type=thinking_type,
    )

    assert "".join(chunk.content for chunk in chunks) == "public"
    assert all(chunk.thinking_signature is None for chunk in chunks)


@pytest.mark.parametrize("include_reasoning", [False, True])
def test_stream_codec_rejects_nonstring_redacted_thinking_data(
    include_reasoning: bool,
) -> None:
    decoder = AnthropicStreamDecoder(include_reasoning=include_reasoning)

    with pytest.raises(AnthropicCodecError, match="redacted thinking data must be a string"):
        decoder.decode_event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": 7},
            }
        )

    assert decoder.blocks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking_type", [None, "enabled"])
async def test_provider_wraps_nonstring_redacted_thinking_as_safe_protocol_error(
    thinking_type: str | None,
) -> None:
    with pytest.raises(ProviderError) as caught:
        await _provider_stream(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": 7},
            },
            thinking_type=thinking_type,
        )

    assert caught.value.status_code == 502
    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert caught.value.provider == "anthropic"
    assert caught.value.model == "claude-requested"
    assert str(caught.value) == "anthropic returned a malformed upstream response"


def _complete_text_decoder() -> AnthropicStreamDecoder:
    decoder = AnthropicStreamDecoder(include_reasoning=False)
    for event in (
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "done"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    ):
        decoder.decode_event(event)
    return decoder


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            id="second-terminal",
        ),
        pytest.param(
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            id="message-start",
        ),
        pytest.param(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": "late"},
            },
            id="content-start",
        ),
        pytest.param(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "late"},
            },
            id="content-delta",
        ),
        pytest.param(
            {"type": "content_block_stop", "index": 0},
            id="content-stop",
        ),
    ],
)
def test_stream_codec_rejects_known_mutation_after_terminal(event: dict) -> None:
    decoder = _complete_text_decoder()

    with pytest.raises(AnthropicCodecError, match="after explicit terminal"):
        decoder.decode_event(event)


def test_stream_codec_allows_message_stop_and_unknown_event_after_terminal() -> None:
    decoder = _complete_text_decoder()

    assert decoder.decode_event({"type": "message_stop"}) == []
    assert decoder.decode_event({"type": "future_event", "opaque": {"version": 2}}) == []
    decoder.finalize()


@pytest.mark.asyncio
async def test_provider_wraps_known_event_after_terminal_as_safe_protocol_error() -> None:
    with pytest.raises(ProviderError) as caught:
        await _provider_stream(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "done"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": "private-late"},
            },
            {"type": "message_stop"},
        )

    assert caught.value.status_code == 502
    assert caught.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert caught.value.provider == "anthropic"
    assert caught.value.model == "claude-requested"
    assert str(caught.value) == "anthropic returned a malformed upstream response"
    assert "private-late" not in str(caught.value)
