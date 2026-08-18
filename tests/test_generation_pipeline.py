from types import SimpleNamespace
from typing import cast

import pytest

from router_maestro.protocols import WireProtocol
from router_maestro.providers.base import ProviderError
from router_maestro.providers.bindings import legacy_endpoint_binding
from router_maestro.routing.capabilities import Operation
from router_maestro.routing.router import Router
from router_maestro.runtime.reasoning_capsule import ReasoningCapsuleCodec
from router_maestro.server.generation_pipeline import build_generation_pipeline


class _Provider:
    name = "fake"

    def bindings(self):
        return (
            legacy_endpoint_binding(
                binding_id="fake-chat",
                protocol=WireProtocol.OPENAI_CHAT,
                operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM}),
            ),
        )


def _router() -> Router:
    return cast(Router, SimpleNamespace(providers={"fake": _Provider()}))


def _codec():
    return ReasoningCapsuleCodec(bytes([41]) * 32)


def test_pipeline_construction_keeps_anthropic_request_lazy():
    payload = {
        "model": "fake/model",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
    }

    pipeline = build_generation_pipeline(
        _router(),
        _codec(),
        WireProtocol.ANTHROPIC_MESSAGES,
        payload,
        path="/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
    )

    assert pipeline.envelope.model == "fake/model"
    assert pipeline.envelope.path == "/v1/messages"
    assert pipeline.envelope.materialization_count == 0
    assert pipeline.envelope.raw_payload == payload


def test_pipeline_supplies_gemini_endpoint_model_and_stream_context():
    pipeline = build_generation_pipeline(
        _router(),
        _codec(),
        WireProtocol.GEMINI,
        {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        model="fake/gemini",
        stream=True,
    )

    assert pipeline.envelope.model == "fake/gemini"
    assert pipeline.envelope.stream is True
    assert pipeline.envelope.materialization_count == 0


@pytest.mark.parametrize("payload", [[], "hello", 1, None])
def test_pipeline_rejects_non_object_json(payload):
    with pytest.raises(ProviderError, match="Request body must be a JSON object"):
        build_generation_pipeline(
            _router(),
            _codec(),
            WireProtocol.OPENAI_CHAT,
            payload,
        )
