"""OpenAI-compatible provider for custom endpoints."""

from collections.abc import AsyncIterator

import httpx

from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
)


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible provider for custom endpoints."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        models: dict[str, str] | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            name: Provider name
            base_url: Base URL for API requests
            api_key: API key for authentication
            models: Dict of model_id -> display_name
        """
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._models = models or {}

    def is_authenticated(self) -> bool:
        """Check if authenticated (always true for custom providers)."""
        return bool(self.api_key)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion."""
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        # Merge any extra parameters
        payload.update(request.extra)

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
                retryable = e.response.status_code in (429, 500, 502, 503, 504)
                raise ProviderError(
                    f"{self.name} API error: {e.response.status_code}",
                    status_code=e.response.status_code,
                    retryable=retryable,
                )
            except httpx.HTTPError as e:
                raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion."""
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},  # Request usage info in stream
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        payload.update(request.extra)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=120.0,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        import json

                        data = json.loads(data_str)

                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = data["choices"][0].get("finish_reason")
                            usage = data.get("usage")  # Capture usage info

                            if content or finish_reason:
                                yield ChatStreamChunk(
                                    content=content,
                                    finish_reason=finish_reason,
                                    usage=usage,
                                )
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code in (429, 500, 502, 503, 504)
                raise ProviderError(
                    f"{self.name} API error: {e.response.status_code}",
                    status_code=e.response.status_code,
                    retryable=retryable,
                )
            except httpx.HTTPError as e:
                raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
        if self._models:
            return [
                ModelInfo(id=model_id, name=name, provider=self.name)
                for model_id, name in self._models.items()
            ]

        # Try to fetch from API
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                return [
                    ModelInfo(id=model["id"], name=model["id"], provider=self.name)
                    for model in data.get("data", [])
                ]
            except httpx.HTTPError:
                return []
