"""Shared OpenAI-compatible chat provider logic."""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from logging import Logger
from typing import NoReturn

import httpx

from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ProviderError,
)


class OpenAIChatProvider(BaseProvider, ABC):
    """Shared OpenAI-compatible chat behavior."""

    def __init__(self, base_url: str, logger: Logger) -> None:
        self.base_url = base_url.rstrip("/")
        self._logger = logger

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Return headers for the API request."""

    def _get_payload_extra(self, request: ChatRequest) -> dict:
        """Return extra payload fields for the request."""
        return {}

    def _error_label(self) -> str:
        return self.name

    def _build_payload(self, request: ChatRequest, stream: bool) -> dict:
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        payload.update(self._get_payload_extra(request))
        return payload

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request, stream=False)
        label = self._error_label()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

                return ChatResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data.get("model", request.model),
                    finish_reason=data["choices"][0].get("finish_reason", "stop"),
                    usage=data.get("usage"),
                )
            except httpx.HTTPStatusError as e:
                self._raise_http_status_error(label, e)
            except httpx.HTTPError as e:
                self._raise_http_error(label, e)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        payload = self._build_payload(request, stream=True)
        label = self._error_label()

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        data = json.loads(data_str)

                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = data["choices"][0].get("finish_reason")
                            usage = data.get("usage")

                            if content or finish_reason or usage:
                                yield ChatStreamChunk(
                                    content=content,
                                    finish_reason=finish_reason,
                                    usage=usage,
                                )
            except httpx.HTTPStatusError as e:
                self._raise_http_status_error(label, e, stream=True)
            except httpx.HTTPError as e:
                self._raise_http_error(label, e, stream=True)

    def _raise_http_status_error(
        self,
        label: str,
        error: httpx.HTTPStatusError,
        stream: bool = False,
    ) -> NoReturn:
        retryable = error.response.status_code in (429, 500, 502, 503, 504)
        suffix = " stream" if stream else ""
        self._logger.error("%s%s API error: %d", label, suffix, error.response.status_code)
        raise ProviderError(
            f"{label} API error: {error.response.status_code}",
            status_code=error.response.status_code,
            retryable=retryable,
        )

    def _raise_http_error(
        self,
        label: str,
        error: httpx.HTTPError,
        stream: bool = False,
    ) -> NoReturn:
        suffix = " stream" if stream else ""
        self._logger.error("%s%s HTTP error: %s", label, suffix, error)
        raise ProviderError(f"HTTP error: {error}", retryable=True)
