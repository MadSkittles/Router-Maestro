"""Copilot's exact unsupported-operation status is preserved at its boundary."""

import json
from collections.abc import AsyncIterator
from typing import Literal

import httpx
import pytest

from router_maestro.providers import (
    ChatRequest,
    ChatStreamChunk,
    CopilotProvider,
    Message,
    ResponsesRequest,
)
from router_maestro.providers.base import ProviderError, ProviderFailureKind

Operation = Literal["chat", "chat-stream", "responses", "responses-stream"]
RAW_MARKER = "private-copilot-error-detail"


async def _noop() -> None:
    return None


def _provider_returning(*, content: bytes, status_code: int = 400) -> CopilotProvider:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=content)

    provider = CopilotProvider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider.ensure_token = _noop  # type: ignore[method-assign]
    provider._get_headers = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    return provider


async def _invoke(provider: CopilotProvider, operation: Operation) -> None:
    if operation == "chat":
        await provider.chat_completion(
            ChatRequest(model="gpt-4o", messages=[Message(role="user", content="hi")])
        )
        return
    if operation == "chat-stream":
        stream: AsyncIterator[ChatStreamChunk] = provider.chat_completion_stream(
            ChatRequest(model="gpt-4o", messages=[Message(role="user", content="hi")])
        )
        async for _ in stream:
            pass
        return
    if operation == "responses":
        await provider.responses_completion(ResponsesRequest(model="gpt-5", input="hi"))
        return
    stream = provider.responses_completion_stream(
        ResponsesRequest(model="gpt-5", input="hi", stream=True)
    )
    async for _ in stream:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "chat-stream", "responses", "responses-stream"])
async def test_copilot_exact_unsupported_api_error_is_typed_for_every_operation(
    operation: Operation,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider_returning(
        content=json.dumps(
            {
                "error": {
                    "code": "unsupported_api_for_model",
                    "message": RAW_MARKER,
                }
            }
        ).encode()
    )

    with pytest.raises(ProviderError) as exc_info:
        await _invoke(provider, operation)

    error = exc_info.value
    assert error.kind is ProviderFailureKind.UNSUPPORTED_OPERATION
    assert error.status_code == 400
    assert error.upstream_status_code == 400
    assert error.retryable is False
    assert error.provider == provider.name
    assert error.model == ("gpt-4o" if operation.startswith("chat") else "gpt-5")
    assert RAW_MARKER not in str(error)
    assert RAW_MARKER not in error.safe_message
    assert RAW_MARKER not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "chat-stream", "responses", "responses-stream"])
@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            json.dumps(
                {
                    "error": {
                        "code": "some_other_code",
                        "message": RAW_MARKER,
                    }
                }
            ).encode(),
            id="generic-json",
        ),
        pytest.param(b'{"error":' + RAW_MARKER.encode(), id="malformed-json"),
        pytest.param(b"not-json-" + RAW_MARKER.encode(), id="non-json"),
        pytest.param(
            json.dumps(
                {
                    "error": {
                        "code": "unsupported_api_for_model_lookalike",
                        "message": RAW_MARKER,
                    }
                }
            ).encode(),
            id="lookalike-code",
        ),
        pytest.param(
            json.dumps(
                {
                    "code": "unsupported_api_for_model",
                    "message": RAW_MARKER,
                }
            ).encode(),
            id="wrong-nesting",
        ),
        pytest.param(
            json.dumps(
                {
                    "error": {
                        "code": "unsupported_api_for_model",
                        "message": RAW_MARKER + ("x" * (70 * 1024)),
                    }
                }
            ).encode(),
            id="oversized-exact-error",
        ),
    ],
)
async def test_copilot_other_400_bodies_remain_generic_and_safe(
    operation: Operation,
    content: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider_returning(content=content)

    with pytest.raises(ProviderError) as exc_info:
        await _invoke(provider, operation)

    error = exc_info.value
    assert error.kind is ProviderFailureKind.UPSTREAM_STATUS
    assert error.status_code == 400
    assert error.upstream_status_code == 400
    assert error.retryable is False
    assert error.provider == provider.name
    assert error.model == ("gpt-4o" if operation.startswith("chat") else "gpt-5")
    assert RAW_MARKER not in str(error)
    assert RAW_MARKER not in error.safe_message
    assert RAW_MARKER not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "chat-stream", "responses", "responses-stream"])
async def test_copilot_exact_unsupported_code_on_non_400_remains_generic(
    operation: Operation,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider_returning(
        status_code=422,
        content=json.dumps(
            {
                "error": {
                    "code": "unsupported_api_for_model",
                    "message": RAW_MARKER,
                }
            }
        ).encode(),
    )

    with pytest.raises(ProviderError) as exc_info:
        await _invoke(provider, operation)

    error = exc_info.value
    assert error.kind is ProviderFailureKind.UPSTREAM_STATUS
    assert error.status_code == 422
    assert error.upstream_status_code == 422
    assert error.retryable is False
    assert error.provider == provider.name
    assert error.model == ("gpt-4o" if operation.startswith("chat") else "gpt-5")
    assert RAW_MARKER not in str(error)
    assert RAW_MARKER not in error.safe_message
    assert RAW_MARKER not in caplog.text
