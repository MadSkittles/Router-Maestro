"""OpenAI provider implementation."""

from collections.abc import AsyncIterator

import httpx

from router_maestro.auth import AuthManager, AuthType
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
)
from router_maestro.utils import get_logger

logger = get_logger("providers.openai")

OPENAI_API_URL = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    """OpenAI official provider."""

    name = "openai"

    def __init__(self, base_url: str = OPENAI_API_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_manager = AuthManager()

    def is_authenticated(self) -> bool:
        """Check if authenticated with OpenAI."""
        cred = self.auth_manager.get_credential("openai")
        return cred is not None and cred.type == AuthType.API_KEY

    def _get_api_key(self) -> str:
        """Get the API key."""
        cred = self.auth_manager.get_credential("openai")
        if not cred or cred.type != AuthType.API_KEY:
            logger.error("Not authenticated with OpenAI")
            raise ProviderError("Not authenticated with OpenAI", status_code=401)
        return cred.key

    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenAI API requests."""
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion via OpenAI."""
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        logger.debug("OpenAI chat completion: model=%s", request.model)
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

                logger.debug("OpenAI chat completion successful")
                return ChatResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data.get("model", request.model),
                    finish_reason=data["choices"][0].get("finish_reason", "stop"),
                    usage=data.get("usage"),
                )
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code in (429, 500, 502, 503, 504)
                logger.error("OpenAI API error: %d", e.response.status_code)
                raise ProviderError(
                    f"OpenAI API error: {e.response.status_code}",
                    status_code=e.response.status_code,
                    retryable=retryable,
                )
            except httpx.HTTPError as e:
                logger.error("OpenAI HTTP error: %s", e)
                raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion via OpenAI."""
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},  # Request usage info in stream
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        logger.debug("OpenAI streaming chat: model=%s", request.model)
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
                logger.error("OpenAI stream API error: %d", e.response.status_code)
                raise ProviderError(
                    f"OpenAI API error: {e.response.status_code}",
                    status_code=e.response.status_code,
                    retryable=retryable,
                )
            except httpx.HTTPError as e:
                logger.error("OpenAI stream HTTP error: %s", e)
                raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def list_models(self) -> list[ModelInfo]:
        """List available OpenAI models."""
        logger.debug("Fetching OpenAI models")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                models = []
                for model in data.get("data", []):
                    model_id = model["id"]
                    # Filter to chat models
                    if any(x in model_id for x in ["gpt-", "o1-", "o3-"]):
                        models.append(
                            ModelInfo(
                                id=model_id,
                                name=model_id,
                                provider=self.name,
                            )
                        )
                logger.info("Fetched %d OpenAI models", len(models))
                return models
            except httpx.HTTPError as e:
                logger.warning("Failed to list OpenAI models, using defaults: %s", e)
                # Return default models on error
                return [
                    ModelInfo(id="gpt-4o", name="GPT-4o", provider=self.name),
                    ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini", provider=self.name),
                    ModelInfo(id="gpt-4-turbo", name="GPT-4 Turbo", provider=self.name),
                ]
