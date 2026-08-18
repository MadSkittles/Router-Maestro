from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.config import FallbackConfig, FallbackStrategy, PrioritiesConfig
from router_maestro.protocols import WireProtocol
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
)
from router_maestro.providers.bindings import legacy_endpoint_binding
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.router import CACHE_TTL_SECONDS, Router
from router_maestro.runtime.reasoning_capsule import (
    ReasoningCapsuleCodec,
    ReasoningCapsulePayload,
)
from router_maestro.server.routes.gemini import router as gemini_router
from router_maestro.utils.cache import TTLCache

_CAPSULE_KEY = bytes([61]) * 32
_RAW_REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_gemini_replay",
    "summary": [{"type": "summary_text", "text": "private summary"}],
    "encrypted_content": "opaque-responses-state",
    "status": "completed",
    "future_field": {"must": ["survive", 2]},
}


class _ResponsesOnlyProvider(BaseProvider):
    name = "github-copilot"

    def __init__(self, *, first_reasoning_item: dict | None = None) -> None:
        self.requests: list[ResponsesRequest] = []
        self.first_reasoning_item = first_reasoning_item

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            operations=frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM})
        )

    def bindings(self):
        return (
            legacy_endpoint_binding(
                binding_id="copilot-openai-responses",
                protocol=WireProtocol.OPENAI_RESPONSES,
                operations=frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM}),
            ),
        )

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="gpt-responses",
                name="gpt-responses",
                provider=self.name,
                supported_endpoints=("/responses",),
            )
        ]

    def is_authenticated(self) -> bool:
        return True

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError(f"Chat transport must not be selected: {request.model}")

    async def chat_completion_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatStreamChunk]:
        raise AssertionError(f"Chat transport must not be selected: {request.model}")
        if False:
            yield ChatStreamChunk(content="")

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        self.requests.append(request)
        return ResponsesResponse(
            content="hello from responses",
            model=request.model,
            usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            finish_reason="stop",
            reasoning_item=(
                deepcopy(self.first_reasoning_item)
                if len(self.requests) == 1 and self.first_reasoning_item is not None
                else None
            ),
        )

    async def responses_completion_stream(
        self,
        request: ResponsesRequest,
    ) -> AsyncIterator[ResponsesStreamChunk]:
        self.requests.append(request)
        yield ResponsesStreamChunk(content="hello")
        yield ResponsesStreamChunk(
            content="",
            usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            finish_reason="stop",
        )


def _router(provider: _ResponsesOnlyProvider) -> Router:
    router = Router.__new__(Router)
    router.providers = {provider.name: provider}
    router._models_cache = {}
    router._models_cache_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._priorities_cache = TTLCache(CACHE_TTL_SECONDS)
    router._fuzzy_cache = {}
    router._providers_ttl = TTLCache(CACHE_TTL_SECONDS)
    router._model_aliases = None
    router._managed_generation = True
    router._priorities_cache.set(
        PrioritiesConfig(
            priorities=["github-copilot/gpt-responses"],
            fallback=FallbackConfig(strategy=FallbackStrategy.NONE, maxRetries=0),
        )
    )
    router._providers_ttl.set(True)
    return router


def _client(
    provider: _ResponsesOnlyProvider | None = None,
) -> tuple[TestClient, _ResponsesOnlyProvider, Router]:
    provider = provider or _ResponsesOnlyProvider()
    model_router = _router(provider)
    app = FastAPI()
    app.state.reasoning_capsule_codec = ReasoningCapsuleCodec(_CAPSULE_KEY)
    app.include_router(gemini_router)
    return TestClient(app), provider, model_router


def _payload() -> dict:
    return {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}


def _first_reasoning_turn(
    client: TestClient,
    model_router: Router,
) -> tuple[dict, str]:
    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(
            "/api/gemini/v1beta/models/github-copilot/gpt-responses:generateContent",
            json=_payload(),
        )

    assert response.status_code == 200, response.text
    body = response.json()
    parts = body["candidates"][0]["content"]["parts"]
    thought = next(part for part in parts if part.get("thought") is True)
    signature = thought["thoughtSignature"]
    assert signature.startswith("rmr1.")
    return body["candidates"][0]["content"], signature


def _second_turn_payload(model_content: dict, signature: str) -> dict:
    replay_content = deepcopy(model_content)
    thought = next(part for part in replay_content["parts"] if part.get("thought") is True)
    thought["thoughtSignature"] = signature
    return {
        "contents": [
            *_payload()["contents"],
            replay_content,
            {"role": "user", "parts": [{"text": "continue"}]},
        ]
    }


def test_gemini_generate_content_uses_path_model_and_responses_transport() -> None:
    client, provider, model_router = _client()

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(
            "/api/gemini/v1beta/models/github-copilot/gpt-responses:generateContent",
            json=_payload(),
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["modelVersion"] == "github-copilot/gpt-responses"
    assert body["candidates"][0]["content"]["parts"] == [{"text": "hello from responses"}]
    assert provider.requests[0].model == "gpt-responses"
    assert provider.requests[0].input == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]


def test_gemini_stream_generate_content_uses_method_stream_mode() -> None:
    client, provider, model_router = _client()

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        with client.stream(
            "POST",
            "/api/gemini/v1beta/models/github-copilot/gpt-responses:streamGenerateContent",
            json=_payload(),
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"text": "hello"' in body
    assert '"finishReason": "STOP"' in body
    assert body.count('"finishReason": "STOP"') == 1
    assert len(provider.requests) == 1
    assert provider.requests[0].stream is True


def test_gemini_two_turn_capsule_replays_full_responses_reasoning_item() -> None:
    client, provider, model_router = _client(
        _ResponsesOnlyProvider(first_reasoning_item=_RAW_REASONING_ITEM)
    )
    model_content, signature = _first_reasoning_turn(client, model_router)

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(
            "/api/gemini/v1beta/models/github-copilot/gpt-responses:generateContent",
            json=_second_turn_payload(model_content, signature),
        )

    assert response.status_code == 200, response.text
    assert len(provider.requests) == 2
    replay = provider.requests[1]
    assert replay.model == "gpt-responses"
    assert replay.stream is False
    reasoning_items = [
        item for item in replay.input if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert reasoning_items == [_RAW_REASONING_ITEM]
    assert reasoning_items[0]["future_field"] == {"must": ["survive", 2]}


def test_gemini_tampered_capsule_fails_before_provider_io() -> None:
    client, provider, model_router = _client(
        _ResponsesOnlyProvider(first_reasoning_item=_RAW_REASONING_ITEM)
    )
    model_content, signature = _first_reasoning_turn(client, model_router)
    tampered = f"{signature[:-1]}{'A' if signature[-1] != 'A' else 'B'}"

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(
            "/api/gemini/v1beta/models/github-copilot/gpt-responses:generateContent",
            json=_second_turn_payload(model_content, tampered),
        )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == "Invalid reasoning capsule"
    assert len(provider.requests) == 1


def test_gemini_unknown_capsule_version_fails_before_provider_io() -> None:
    client, provider, model_router = _client(
        _ResponsesOnlyProvider(first_reasoning_item=_RAW_REASONING_ITEM)
    )
    model_content, signature = _first_reasoning_turn(client, model_router)
    unknown_version = signature.replace("rmr1.", "rmr2.", 1)

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(
            "/api/gemini/v1beta/models/github-copilot/gpt-responses:generateContent",
            json=_second_turn_payload(model_content, unknown_version),
        )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == "Invalid reasoning capsule"
    assert len(provider.requests) == 1


def test_gemini_capsule_provenance_mismatch_fails_before_provider_io() -> None:
    client, provider, model_router = _client(
        _ResponsesOnlyProvider(first_reasoning_item=_RAW_REASONING_ITEM)
    )
    model_content, signature = _first_reasoning_turn(client, model_router)
    codec = ReasoningCapsuleCodec(_CAPSULE_KEY)
    original = codec.unseal_for_routing(signature)
    mismatched = codec.seal(
        ReasoningCapsulePayload(
            provider=original.provider,
            model="other-responses-model",
            transport=original.transport,
            item_id=original.item_id,
            opaque_state=original.opaque_state,
        )
    )

    with patch("router_maestro.server.routes.gemini.get_router", return_value=model_router):
        response = client.post(
            "/api/gemini/v1beta/models/github-copilot/gpt-responses:generateContent",
            json=_second_turn_payload(model_content, mismatched),
        )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == "Invalid reasoning capsule"
    assert len(provider.requests) == 1
