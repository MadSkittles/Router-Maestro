"""Focused contracts for Copilot's protocol-native endpoint bindings."""

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import httpx
import pytest

from router_maestro.config.priorities import PrioritiesConfig
from router_maestro.protocols import (
    ConversionMode,
    OpenAIChatRuntime,
    RequestEnvelope,
    RequestManifest,
    SemanticEvent,
    SemanticEventType,
    SemanticRequest,
    SemanticResponse,
    WireProtocol,
)
from router_maestro.providers.base import (
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
)
from router_maestro.providers.bindings import (
    COPILOT_ANTHROPIC_MESSAGES_BINDING,
    COPILOT_OPENAI_CHAT_BINDING,
    COPILOT_OPENAI_RESPONSES_BINDING,
    AttemptRequestContext,
)
from router_maestro.providers.copilot import CopilotHttpExecutor, CopilotProvider
from router_maestro.routing.model_ref import ModelRef


def _binding(provider: CopilotProvider, protocol: WireProtocol):
    return next(binding for binding in provider.bindings() if binding.protocol is protocol)


@pytest.mark.asyncio
async def test_grok_responses_binding_flattens_and_restores_namespace_tools() -> None:
    provider = CopilotProvider()
    binding = _binding(provider, WireProtocol.OPENAI_RESPONSES)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="grok-4.6"),
        payload={
            "model": "github-copilot/grok-4.6",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "status",
                    "namespace": "mcp__qmd",
                    "arguments": "{}",
                }
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__qmd",
                    "description": "QMD tools",
                    "tools": [
                        {
                            "type": "function",
                            "name": "status",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        }
                    ],
                }
            ],
        },
        stream=False,
    )

    flattened_tool = attempt.payload["tools"][0]
    flattened_call = attempt.payload["input"][0]
    encoded_name = flattened_tool["name"]
    assert flattened_tool["type"] == "function"
    assert encoded_name != "status"
    assert flattened_call["name"] == encoded_name
    assert "namespace" not in flattened_call

    projected = CopilotHttpExecutor._project_response(
        WireProtocol.OPENAI_RESPONSES,
        {
            "id": "resp_1",
            "model": "grok-4.6",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": encoded_name,
                    "arguments": "{}",
                }
            ],
        },
        model="grok-4.6",
    )
    assert projected["output"][0]["name"] == "status"
    assert projected["output"][0]["namespace"] == "mcp__qmd"


@pytest.mark.asyncio
async def test_cross_chat_attempt_preserves_copilot_reasoning_opaque_for_runtime() -> None:
    provider = CopilotProvider()
    binding = _binding(provider, WireProtocol.OPENAI_CHAT)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="gemini-3.6-flash"),
        payload={"model": "gemini-3.6-flash", "messages": [{"role": "user", "content": "hi"}]},
        stream=True,
        request_context=AttemptRequestContext(conversion_mode=ConversionMode.SEMANTIC_IR),
    )
    assert attempt.capture_reasoning_state is True

    projected = CopilotHttpExecutor._project_response(
        WireProtocol.OPENAI_CHAT,
        {
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "model": "gemini-3.6-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_text": "plan",
                        "reasoning_opaque": "provider-state",
                    },
                    "finish_reason": None,
                }
            ],
        },
        model="gemini-3.6-flash",
        stream=True,
        capture_reasoning_state=attempt.capture_reasoning_state,
    )
    delta = projected["choices"][0]["delta"]
    assert delta["reasoning_opaque"] == "provider-state"


class _IdentityRuntime:
    def __init__(self, protocol: WireProtocol) -> None:
        self.protocol = protocol
        self.decode_request = AsyncMock(
            side_effect=AssertionError("identity preparation must not decode semantic IR")
        )

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        return RequestManifest(
            protocol=self.protocol,
            model=payload.get("model") if isinstance(payload.get("model"), str) else None,
            stream=payload.get("stream") is True,
        )

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        del request
        raise AssertionError("identity preparation must not encode semantic IR")

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        del payload
        raise AssertionError("identity preparation must not decode a response")

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        del response
        raise AssertionError("identity preparation must not encode a response")

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        del payload
        raise AssertionError("identity preparation must not decode stream events")

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        del event
        raise AssertionError("identity preparation must not encode stream events")


class _DeepcopyForbidden:
    def __deepcopy__(self, memo: dict) -> None:
        del memo
        raise AssertionError("identity preparation must not deep-copy nested wire values")


def _identity_envelope(
    protocol: WireProtocol,
    payload: dict,
) -> tuple[RequestEnvelope, _IdentityRuntime]:
    runtime = _IdentityRuntime(protocol)
    return RequestEnvelope(runtime, payload), runtime


def test_copilot_bindings_are_protocol_native_and_share_dialect_executor() -> None:
    provider = CopilotProvider()

    bindings = provider.bindings()

    assert [binding.id for binding in bindings] == [
        COPILOT_ANTHROPIC_MESSAGES_BINDING,
        COPILOT_OPENAI_CHAT_BINDING,
        COPILOT_OPENAI_RESPONSES_BINDING,
    ]
    assert all(not binding.is_legacy for binding in bindings)
    assert len({id(binding.dialect) for binding in bindings}) == 1
    assert len({id(binding.executor) for binding in bindings}) == 1
    assert provider.bindings() is bindings


@pytest.mark.asyncio
async def test_copilot_dialect_prepares_exact_messages_chat_and_responses_contracts() -> None:
    provider = CopilotProvider()
    provider._api_base = "https://copilot.example/api"
    model = ModelRef(provider=provider.name, upstream_id="gpt-4o")

    messages_source = {
        "model": "github-copilot/public-name",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
    }
    messages = await _binding(provider, WireProtocol.ANTHROPIC_MESSAGES).prepare_attempt(
        model=model, payload=messages_source, stream=False
    )
    assert messages.url == "https://copilot.example/api/v1/messages"
    assert dict(messages.payload) == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "stream": False,
    }
    assert dict(messages.headers) == {}

    chat_source = {
        "model": "github-copilot/public-name",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prior"},
            {"role": "user", "content": "next"},
        ],
        "max_tokens": 48,
    }
    chat = await _binding(provider, WireProtocol.OPENAI_CHAT).prepare_attempt(
        model=model,
        payload=chat_source,
        stream=True,
    )
    assert chat.url == "https://copilot.example/api/chat/completions"
    assert dict(chat.payload) == {
        **chat_source,
        "model": "gpt-4o",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert dict(chat.headers) == {"X-Initiator": "agent"}

    responses_source = {
        "model": "github-copilot/public-name",
        "input": "hello",
        "previous_response_id": "resp_previous",
        "reasoning": {"effort": "high", "summary": "detailed"},
    }
    responses_model = ModelRef(provider=provider.name, upstream_id="gpt-5.4-mini")
    responses = await _binding(provider, WireProtocol.OPENAI_RESPONSES).prepare_attempt(
        model=responses_model, payload=responses_source, stream=False
    )
    assert responses.url == "https://copilot.example/api/responses"
    assert dict(responses.payload) == {
        **responses_source,
        "model": "gpt-5.4-mini",
        "stream": False,
        "include": ["reasoning.encrypted_content"],
    }
    assert dict(responses.headers) == {"X-Initiator": "user"}

    assert messages_source["model"] == "github-copilot/public-name"
    assert "stream" not in messages_source
    assert chat_source["model"] == "github-copilot/public-name"
    assert "stream" not in chat_source
    assert responses_source["model"] == "github-copilot/public-name"
    assert "include" not in responses_source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol",
    [
        WireProtocol.ANTHROPIC_MESSAGES,
        WireProtocol.OPENAI_CHAT,
        WireProtocol.OPENAI_RESPONSES,
    ],
)
async def test_copilot_identity_preparation_uses_branch_cow_without_deepcopy(
    protocol: WireProtocol,
) -> None:
    provider = CopilotProvider()
    sentinel = _DeepcopyForbidden()

    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        model = "claude-opus-4.6"
        payload = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
            "thinking": {
                "type": "adaptive",
                "budget_tokens": 4096,
                "future": {"sentinel": sentinel},
            },
        }
    elif protocol is WireProtocol.OPENAI_CHAT:
        model = "gpt-5.4-mini"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "bad\ud800text",
                            "future": {"sentinel": sentinel},
                        }
                    ],
                }
            ],
            "stream_options": {"include_usage": False},
        }
    else:
        model = "gpt-5.4-mini"
        payload = {
            "input": "hello",
            "reasoning": {
                "effort": "high",
                "summary": "detailed",
                "future": {"sentinel": sentinel},
            },
        }

    attempt = await _binding(provider, protocol).prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id=model),
        payload=payload,
        stream=(protocol is WireProtocol.OPENAI_CHAT),
    )

    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        source_branch = payload["thinking"]
        outbound_branch = attempt.payload["thinking"]
        assert source_branch["budget_tokens"] == 4096
        assert "budget_tokens" not in outbound_branch
    elif protocol is WireProtocol.OPENAI_CHAT:
        source_branch = payload["messages"][0]["content"][0]
        outbound_branch = attempt.payload["messages"][0]["content"][0]
        assert source_branch["text"] == "bad\ud800text"
        assert outbound_branch["text"] == "bad?text"
        assert payload["stream_options"] == {"include_usage": False}
        assert attempt.payload["stream_options"] == {"include_usage": True}
    else:
        source_branch = payload["reasoning"]
        outbound_branch = attempt.payload["reasoning"]
        assert source_branch == {
            "effort": "high",
            "summary": "detailed",
            "future": {"sentinel": sentinel},
        }
        assert "include" not in payload
        assert attempt.payload["include"] == ["reasoning.encrypted_content"]

    assert outbound_branch is not source_branch
    assert source_branch["future"]["sentinel"] is sentinel
    assert outbound_branch["future"]["sentinel"] is sentinel


@pytest.mark.asyncio
async def test_copilot_messages_binding_forwards_only_opted_in_ingress_header() -> None:
    provider = CopilotProvider()
    attempt = await _binding(provider, WireProtocol.ANTHROPIC_MESSAGES).prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="claude-opus-4.6"),
        payload={"messages": [], "max_tokens": 32},
        stream=False,
        request_context=AttemptRequestContext(
            path="/api/anthropic/v1/messages",
            headers={
                "Anthropic-Beta": "context-management-2025-06-27",
                "Authorization": "Bearer client-secret",
                "X-API-Key": "client-secret",
            },
        ),
    )

    assert dict(attempt.headers) == {
        "anthropic-beta": "context-management-2025-06-27",
    }


@pytest.mark.asyncio
async def test_copilot_messages_identity_is_raw_copy_on_write_without_ir() -> None:
    provider = CopilotProvider()
    payload = {
        "model": "github-copilot/public-name",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello", "future_part": 1}],
                "future_message": {"kept": True},
            }
        ],
        "max_tokens": 64,
        "thinking": {"type": "adaptive", "budget_tokens": 4096, "future_mode": "kept"},
        "future_top_level": {"kept": True},
        "mcp_servers": [{"name": "known-rejected"}],
        "container": {"id": "known-rejected"},
    }
    original = deepcopy(payload)
    envelope, runtime = _identity_envelope(WireProtocol.ANTHROPIC_MESSAGES, payload)

    attempt = await _binding(provider, WireProtocol.ANTHROPIC_MESSAGES).prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="claude-opus-4.6"),
        payload=envelope.native_payload(),
        stream=False,
    )

    assert payload == original
    assert envelope.native_payload() == original
    expected = {
        key: value for key, value in original.items() if key not in {"mcp_servers", "container"}
    }
    expected.update(
        {
            "model": "claude-opus-4.6",
            "stream": False,
            "thinking": {"type": "adaptive", "future_mode": "kept"},
        }
    )
    assert dict(attempt.payload) == expected
    assert "mcp_servers" not in attempt.payload
    assert "container" not in attempt.payload
    assert attempt.payload["future_top_level"] == {"kept": True}
    assert attempt.payload["messages"][0]["future_message"] == {"kept": True}
    assert envelope.materialization_count == 0
    runtime.decode_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_copilot_chat_identity_is_raw_copy_on_write_without_ir() -> None:
    provider = CopilotProvider()
    payload = {
        "model": "github-copilot/public-name",
        "messages": [
            {
                "role": "user",
                "name": "alice",
                "content": [
                    {"type": "text", "text": "bad\ud800text", "future_part": {"kept": True}},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA==", "future": 1},
                    },
                ],
                "future_message": {"kept": True},
            },
            {"role": "assistant", "content": "prior"},
        ],
        "max_tokens": 48,
        "parallel_tool_calls": True,
        "reasoning_effort": "max",
        "thinking": {"type": "enabled", "budget_tokens": 4096, "future": "local-only"},
        "stream_options": {"include_usage": False, "future_stream_option": "kept"},
        "future_top_level": {"kept": True},
    }
    original = deepcopy(payload)
    envelope, runtime = _identity_envelope(WireProtocol.OPENAI_CHAT, payload)

    attempt = await _binding(provider, WireProtocol.OPENAI_CHAT).prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="gpt-5.4-mini"),
        payload=envelope.native_payload(),
        stream=True,
    )

    body = dict(attempt.payload)
    assert payload == original
    assert envelope.native_payload() == original
    assert body["model"] == "gpt-5.4-mini"
    assert body["stream"] is True
    assert body["max_completion_tokens"] == 48
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "xhigh"
    assert "thinking" not in body
    assert body["parallel_tool_calls"] is True
    assert body["future_top_level"] == {"kept": True}
    assert body["stream_options"] == {
        "include_usage": True,
        "future_stream_option": "kept",
    }
    assert body["messages"][0]["name"] == "alice"
    assert body["messages"][0]["future_message"] == {"kept": True}
    assert body["messages"][0]["content"][0] == {
        "type": "text",
        "text": "bad?text",
        "future_part": {"kept": True},
    }
    assert body["messages"][0]["content"][1]["image_url"]["future"] == 1
    assert dict(attempt.headers) == {
        "X-Initiator": "agent",
        "Copilot-Vision-Request": "true",
    }
    assert envelope.materialization_count == 0
    runtime.decode_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_copilot_responses_identity_is_raw_copy_on_write_without_ir() -> None:
    provider = CopilotProvider()
    payload = {
        "model": "github-copilot/public-name",
        "input": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AA==",
                        "future_part": {"kept": True},
                    }
                ],
                "future_item": {"kept": True},
            }
        ],
        "reasoning": {"effort": "high", "summary": "detailed", "future": "kept"},
        "previous_response_id": "resp_previous",
        "top_p": 0.7,
        "store": True,
        "future_top_level": {"kept": True},
    }
    original = deepcopy(payload)
    envelope, runtime = _identity_envelope(WireProtocol.OPENAI_RESPONSES, payload)

    attempt = await _binding(provider, WireProtocol.OPENAI_RESPONSES).prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="gpt-5.3-codex"),
        payload=envelope.native_payload(),
        stream=False,
    )

    body = dict(attempt.payload)
    assert payload == original
    assert envelope.native_payload() == original
    assert body["model"] == "gpt-5.3-codex"
    assert body["stream"] is False
    assert "store" not in body
    assert body["future_top_level"] == {"kept": True}
    assert body["input"][0]["future_item"] == {"kept": True}
    assert body["input"][0]["content"][0]["future_part"] == {"kept": True}
    assert body["top_p"] == 0.7
    assert body["reasoning"] == {
        "effort": "high",
        "summary": "detailed",
        "future": "kept",
    }
    assert body["include"] == ["reasoning.encrypted_content"]
    assert dict(attempt.headers) == {
        "X-Initiator": "agent",
        "Copilot-Vision-Request": "true",
    }
    assert envelope.materialization_count == 0
    runtime.decode_request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_copilot_responses_identity_rejects_unsupported_temperature_consistently(
    stream: bool,
) -> None:
    provider = CopilotProvider()

    with pytest.raises(RequestOptionError) as raised:
        await _binding(provider, WireProtocol.OPENAI_RESPONSES).prepare_attempt(
            model=ModelRef(provider=provider.name, upstream_id="gpt-4o"),
            payload={"input": "hello", "temperature": 0.5},
            stream=stream,
        )

    assert str(raised.value) == (
        "GitHub Copilot Responses does not support request option 'temperature'"
    )
    assert raised.value.parameter == "temperature"
    assert raised.value.provider == provider.name
    assert raised.value.model == "gpt-4o"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_copilot_gpt54_rejects_top_p_on_responses_before_io(stream: bool) -> None:
    provider = CopilotProvider()

    with pytest.raises(RequestOptionError) as raised:
        await _binding(provider, WireProtocol.OPENAI_RESPONSES).prepare_attempt(
            model=ModelRef(provider=provider.name, upstream_id="gpt-5.4"),
            payload={"input": "hello", "top_p": 0.7},
            stream=stream,
        )

    assert raised.value.parameter == "top_p"
    assert raised.value.model == "gpt-5.4"


@pytest.mark.asyncio
async def test_copilot_gpt54_rejects_top_p_on_messages_but_accepts_chat() -> None:
    provider = CopilotProvider()
    model = ModelRef(provider=provider.name, upstream_id="gpt-5.4")

    with pytest.raises(RequestOptionError) as raised:
        await _binding(provider, WireProtocol.ANTHROPIC_MESSAGES).prepare_attempt(
            model=model,
            payload={"messages": [{"role": "user", "content": "hello"}], "top_p": 0.7},
            stream=False,
        )

    assert raised.value.parameter == "top_p"
    chat = await _binding(provider, WireProtocol.OPENAI_CHAT).prepare_attempt(
        model=model,
        payload={"messages": [{"role": "user", "content": "hello"}], "top_p": 0.7},
        stream=False,
    )
    assert chat.payload["top_p"] == 0.7


@pytest.mark.asyncio
async def test_copilot_executor_reuses_transport_auth_refresh_and_raw_json() -> None:
    authorization: list[str] = []
    paths: list[str] = []
    initiators: list[str] = []
    request_bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization.append(request.headers["Authorization"])
        paths.append(request.url.path)
        initiators.append(request.headers["X-Initiator"])
        request_bodies.append(request.content)
        if len(authorization) == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(
            200,
            json={"id": "chatcmpl_1", "model": "gpt-4o", "future": {"kept": True}},
        )

    provider = CopilotProvider()
    provider._cached_token = "old-token"
    provider._token_expires = int(time.time()) + 600

    async def ensure_token(force: bool = False) -> None:
        if force:
            provider._cached_token = "new-token"

    provider.ensure_token = AsyncMock(side_effect=ensure_token)  # type: ignore[method-assign]
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    binding = _binding(provider, WireProtocol.OPENAI_CHAT)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="gpt-4o"),
        payload={"messages": [{"role": "user", "content": "hello"}]},
        stream=False,
    )

    try:
        assert binding.executor is not None
        result = await binding.executor.execute(attempt)
    finally:
        await provider.close()

    assert result == {
        "id": "chatcmpl_1",
        "model": "gpt-4o",
        "future": {"kept": True},
    }
    assert authorization == ["Bearer old-token", "Bearer new-token"]
    assert paths == ["/chat/completions", "/chat/completions"]
    assert initiators == ["user", "user"]
    assert len(request_bodies) == 2
    assert request_bodies[0] == request_bodies[1]
    assert all(b'"model":"gpt-4o"' in body for body in request_bodies)
    assert provider.ensure_token.await_args_list == [call(), call(force=True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("protocol", "payload", "expected_beta"),
    [
        (
            WireProtocol.ANTHROPIC_MESSAGES,
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
            "context-management-2025-06-27",
        ),
        (
            WireProtocol.OPENAI_CHAT,
            {"messages": [{"role": "user", "content": "hello"}]},
            None,
        ),
        (
            WireProtocol.OPENAI_RESPONSES,
            {"input": "hello"},
            None,
        ),
    ],
)
async def test_copilot_bindings_isolate_anthropic_beta_header(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    protocol: WireProtocol,
    payload: dict,
    expected_beta: str | None,
) -> None:
    observed: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.headers.get("anthropic-beta"))
        if stream:
            return httpx.Response(
                200,
                text='data: {"type":"terminal"}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"type": "terminal"})

    context = SimpleNamespace(
        audit=None,
        config=PrioritiesConfig(priorities=[]),
        request_header=lambda name: (
            "context-management-2025-06-27" if name.lower() == "anthropic-beta" else None
        ),
    )
    monkeypatch.setattr(
        "router_maestro.runtime.get_current_request_context",
        lambda: context,
    )

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider._token_expires = int(time.time()) + 600
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    binding = _binding(provider, protocol)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="test-model"),
        payload=payload,
        stream=stream,
    )

    try:
        assert binding.executor is not None
        if stream:
            assert [frame async for frame in binding.executor.execute_stream(attempt)] == [
                {"type": "terminal"}
            ]
        else:
            assert await binding.executor.execute(attempt) == {"type": "terminal"}
    finally:
        await provider.close()

    assert observed == [expected_beta]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol", "upstream", "expected"),
    [
        (
            WireProtocol.OPENAI_CHAT,
            {
                "id": "chatcmpl_1",
                "model": "gpt-5.4-2026-03-17",
                "choices": [
                    {
                        "index": 0,
                        "content_filter_results": {},
                        "message": {
                            "role": "assistant",
                            "content": "pong",
                            "reasoning_text": "private model reasoning",
                            "reasoning_opaque": "provider-only-state",
                            "padding": "private",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 7,
                    "reasoning_tokens": 4,
                },
                "prompt_filter_results": [{"prompt_index": 0, "private": True}],
                "copilot_info_messages": [{"message": "provider-only"}],
                "copilot_usage": {"private": True},
                "future": {"kept": True},
            },
            {
                "id": "chatcmpl_1",
                "model": "gpt-5.4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "pong",
                            "reasoning_text": "private model reasoning",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 7,
                    "reasoning_tokens": 4,
                },
                "future": {"kept": True},
            },
        ),
        (
            WireProtocol.OPENAI_RESPONSES,
            {
                "id": "resp_1",
                "model": "gpt-5.4-2026-03-17",
                "output": [],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 2,
                    },
                },
                "copilot_usage": {"private": True},
                "tool_usage": {"private": True},
                "future": {"kept": True},
            },
            {
                "id": "resp_1",
                "model": "gpt-5.4",
                "output": [],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0},
                },
                "future": {"kept": True},
            },
        ),
        (
            WireProtocol.ANTHROPIC_MESSAGES,
            {
                "id": "msg_1",
                "type": "message",
                "model": "claude-2026-08-01",
                "content": [],
                "copilot_usage": {"private": True},
                "stop_details": {"private": True},
                "future": {"kept": True},
            },
            {
                "id": "msg_1",
                "type": "message",
                "model": "claude",
                "content": [],
                "future": {"kept": True},
            },
        ),
    ],
)
async def test_copilot_executor_strips_only_private_nonstream_response_fields(
    protocol: WireProtocol,
    upstream: dict,
    expected: dict,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=httpx.Response(
            200,
            json=upstream,
            request=httpx.Request("POST", "https://api.githubcopilot.com"),
        )
    )
    binding = _binding(provider, protocol)
    request_payload = (
        {"input": "hello"}
        if protocol is WireProtocol.OPENAI_RESPONSES
        else {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
    )
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id=expected["model"]),
        payload=request_payload,
        stream=False,
    )

    assert binding.executor is not None
    result = await binding.executor.execute(attempt)

    assert result == expected


def test_copilot_responses_stream_strips_private_usage_context_details() -> None:
    projected = CopilotHttpExecutor._project_response(
        WireProtocol.OPENAI_RESPONSES,
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "grok-4.6",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                    "context_details": {"private": True},
                },
            },
        },
        model="grok-4.6",
        stream=True,
    )

    assert projected["response"]["usage"] == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }


@pytest.mark.parametrize("payload", [{"future": {"kept": True}}, {"model": 42}])
def test_copilot_model_projection_preserves_missing_or_invalid_model(payload: dict) -> None:
    expected = deepcopy(payload)

    result = CopilotHttpExecutor._project_response(
        WireProtocol.OPENAI_RESPONSES,
        payload,
        model="gpt-5.4",
    )

    assert result == expected


class _TrackingStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.closed = False

    async def __aiter__(self):
        yield self.body

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_copilot_executor_yields_raw_sse_data_and_closes_stream() -> None:
    upstream = _TrackingStream(
        b": keepalive\n\n"
        b'data: {"type":"ping"}\n\n'
        b"event: response.created\n"
        b'data: {"type":"response.created","response":'
        b'{"id":"resp_1","model":"gpt-4o-2026-08-01",'
        b'"future":{"kept":true}}}\n\n'
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        b"event: copilot_usage\n"
        b'data: {"type":"copilot_usage"}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=upstream)

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    binding = _binding(provider, WireProtocol.OPENAI_RESPONSES)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="gpt-4o"),
        payload={"input": "hello"},
        stream=True,
    )

    try:
        assert binding.executor is not None
        frames = [frame async for frame in binding.executor.execute_stream(attempt)]
    finally:
        await provider.close()

    assert frames == [
        {
            "type": "response.created",
            "response": {
                "id": "resp_1",
                "model": "gpt-4o",
                "future": {"kept": True},
            },
        },
        {"type": "response.output_text.delta", "delta": "hi"},
    ]
    assert upstream.closed is True
    provider.ensure_token.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_copilot_chat_stream_strips_prompt_filter_preamble() -> None:
    upstream = _TrackingStream(
        b'data: {"id":"","object":"chat.completion.chunk",'
        b'"created":10,"model":"gpt-5.4-2026-08-01","choices":[],'
        b'"prompt_filter_results":[{"prompt_index":0,"private":true}]}\n\n'
        b'data: {"id":"chat_1","object":"chat.completion.chunk",'
        b'"created":10,"model":"gpt-5.4-2026-08-01","choices":'
        b'[{"index":0,"delta":{"role":"assistant","content":"pong"}}],'
        b'"copilot_info_messages":[{"message":"provider-only"}]}\n\n'
        b'data: {"id":"chat_1","object":"chat.completion.chunk",'
        b'"created":10,"model":"gpt-5.4-2026-08-01","choices":'
        b'[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b'data: {"id":"chat_1","object":"chat.completion.chunk",'
        b'"created":10,"model":"gpt-5.4-2026-08-01","choices":[],'
        b'"usage":{"prompt_tokens":22,"completion_tokens":5,"total_tokens":27}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=upstream)

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    binding = _binding(provider, WireProtocol.OPENAI_CHAT)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="gpt-5.4"),
        payload={"messages": [{"role": "user", "content": "hello"}]},
        stream=True,
    )

    try:
        assert binding.executor is not None
        assert binding.executor._skip_sse_frame(  # type: ignore[attr-defined]
            {"id": "", "choices": [], "prompt_filter_results": []}, attempt
        )
        assert not binding.executor._skip_sse_frame(  # type: ignore[attr-defined]
            {
                "id": "",
                "choices": [],
                "prompt_filter_results": [],
                "error": {"message": "blocked"},
            },
            attempt,
        )
        frames = [frame async for frame in binding.executor.execute_stream(attempt)]
    finally:
        await provider.close()

    assert len(frames) == 3
    assert all("prompt_filter_results" not in frame for frame in frames)
    assert all("copilot_info_messages" not in frame for frame in frames)
    assert frames[0]["choices"][0]["delta"]["content"] == "pong"
    assert frames[-1]["usage"] == {
        "prompt_tokens": 22,
        "completion_tokens": 5,
        "total_tokens": 27,
    }
    decoder = OpenAIChatRuntime(
        origin_provider=provider.name,
        default_model="gpt-5.4",
    ).new_stream_decoder()
    events = [event for frame in frames for event in decoder.decode(frame)]
    assert [event.type for event in events] == [
        SemanticEventType.RESPONSE_STARTED,
        SemanticEventType.TEXT_DELTA,
        SemanticEventType.TERMINAL,
        SemanticEventType.USAGE,
    ]
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_copilot_messages_stream_strips_private_fields_and_preserves_extensions() -> None:
    upstream = _TrackingStream(
        b'data: {"type":"message_start","message":{"id":"msg_1",'
        b'"type":"message","role":"assistant","model":"claude-2026-08-01",'
        b'"content":[],'
        b'"copilot_usage":{"private":true},"stop_details":{"private":true},'
        b'"future":{"kept":true}}}\n\n'
        b'data: {"type":"message_stop","copilot_usage":{"private":true},'
        b'"amazon-bedrock-invocationMetrics":{"private":true},'
        b'"future":{"kept":true}}\n\n'
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=upstream)

    provider = CopilotProvider()
    provider._cached_token = "token"
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    binding = _binding(provider, WireProtocol.ANTHROPIC_MESSAGES)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="claude"),
        payload={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16},
        stream=True,
    )

    try:
        assert binding.executor is not None
        frames = [frame async for frame in binding.executor.execute_stream(attempt)]
    finally:
        await provider.close()

    assert frames == [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [],
                "future": {"kept": True},
            },
        },
        {"type": "message_stop", "future": {"kept": True}},
    ]
    assert upstream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "kind", "retryable"),
    [
        (429, {"error": {"message": "slow down"}}, ProviderFailureKind.RATE_LIMIT, True),
        (
            400,
            {"error": {"code": "unsupported_api_for_model"}},
            ProviderFailureKind.UNSUPPORTED_OPERATION,
            False,
        ),
    ],
)
async def test_copilot_executor_classifies_nonstream_statuses(
    status: int,
    body: dict,
    kind: ProviderFailureKind,
    retryable: bool,
) -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    response = httpx.Response(
        status,
        json=body,
        request=httpx.Request("POST", "https://api.githubcopilot.com/responses"),
    )
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=response
    )
    binding = _binding(provider, WireProtocol.OPENAI_RESPONSES)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="gpt-4o"),
        payload={"input": "hello"},
        stream=False,
    )

    assert binding.executor is not None
    with pytest.raises(ProviderError) as raised:
        await binding.executor.execute(attempt)

    assert raised.value.kind is kind
    assert raised.value.retryable is retryable
    assert raised.value.upstream_status_code == status


@pytest.mark.asyncio
async def test_copilot_executor_maps_malformed_json_to_protocol_failure() -> None:
    provider = CopilotProvider()
    provider.ensure_token = AsyncMock()  # type: ignore[method-assign]
    provider._send_with_auth_retry = AsyncMock(  # type: ignore[method-assign]
        return_value=httpx.Response(
            200,
            json=["not", "an", "object"],
            request=httpx.Request("POST", "https://api.githubcopilot.com/responses"),
        )
    )
    binding = _binding(provider, WireProtocol.OPENAI_RESPONSES)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider=provider.name, upstream_id="gpt-4o"),
        payload={"input": "hello"},
        stream=False,
    )

    assert binding.executor is not None
    with pytest.raises(ProviderError) as raised:
        await binding.executor.execute(attempt)

    assert raised.value.kind is ProviderFailureKind.UPSTREAM_PROTOCOL
    assert raised.value.retryable is True
