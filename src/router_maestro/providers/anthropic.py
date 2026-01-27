"""Anthropic provider implementation."""

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

logger = get_logger("providers.anthropic")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1"


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider."""

    name = "anthropic"

    def __init__(self, base_url: str = ANTHROPIC_API_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_manager = AuthManager()

    def is_authenticated(self) -> bool:
        """Check if authenticated with Anthropic."""
        cred = self.auth_manager.get_credential("anthropic")
        return cred is not None and cred.type == AuthType.API_KEY

    def _get_api_key(self) -> str:
        """Get the API key."""
        cred = self.auth_manager.get_credential("anthropic")
        if not cred or cred.type != AuthType.API_KEY:
            logger.error("Not authenticated with Anthropic")
            raise ProviderError("Not authenticated with Anthropic", status_code=401)
        return cred.key

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Anthropic API requests."""
        return {
            "x-api-key": self._get_api_key(),
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def _convert_messages(self, messages: list) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-style messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                converted.append({"role": msg.role, "content": msg.content})

        return system_prompt, converted

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion via Anthropic."""
        system_prompt, messages = self._convert_messages(request.messages)

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature != 1.0:
            payload["temperature"] = request.temperature

        logger.debug("Anthropic chat completion: model=%s", request.model)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

                # Extract content from Anthropic response
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")

                logger.debug("Anthropic chat completion successful")
                return ChatResponse(
                    content=content,
                    model=data.get("model", request.model),
                    finish_reason=data.get("stop_reason", "stop"),
                    usage={
                        "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                        "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                        "total_tokens": (
                            data.get("usage", {}).get("input_tokens", 0)
                            + data.get("usage", {}).get("output_tokens", 0)
                        ),
                    },
                )
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code in (429, 500, 502, 503, 504, 529)
                logger.error("Anthropic API error: %d", e.response.status_code)
                raise ProviderError(
                    f"Anthropic API error: {e.response.status_code}",
                    status_code=e.response.status_code,
                    retryable=retryable,
                )
            except httpx.HTTPError as e:
                logger.error("Anthropic HTTP error: %s", e)
                raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion via Anthropic."""
        system_prompt, messages = self._convert_messages(request.messages)

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature != 1.0:
            payload["temperature"] = request.temperature

        logger.debug("Anthropic streaming chat: model=%s", request.model)
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=120.0,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if not data_str:
                            continue

                        import json

                        data = json.loads(data_str)
                        event_type = data.get("type")

                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield ChatStreamChunk(
                                    content=delta.get("text", ""),
                                    finish_reason=None,
                                )
                        elif event_type == "message_stop":
                            yield ChatStreamChunk(
                                content="",
                                finish_reason="stop",
                            )
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code in (429, 500, 502, 503, 504, 529)
                logger.error("Anthropic stream API error: %d", e.response.status_code)
                raise ProviderError(
                    f"Anthropic API error: {e.response.status_code}",
                    status_code=e.response.status_code,
                    retryable=retryable,
                )
            except httpx.HTTPError as e:
                logger.error("Anthropic stream HTTP error: %s", e)
                raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def list_models(self) -> list[ModelInfo]:
        """List available Anthropic models."""
        # Anthropic doesn't have a models endpoint, return known models
        logger.debug("Returning known Anthropic models")
        return [
            ModelInfo(id="claude-sonnet-4-20250514", name="Claude Sonnet 4", provider=self.name),
            ModelInfo(
                id="claude-3-5-sonnet-20241022", name="Claude 3.5 Sonnet", provider=self.name
            ),
            ModelInfo(id="claude-3-5-haiku-20241022", name="Claude 3.5 Haiku", provider=self.name),
            ModelInfo(id="claude-3-opus-20240229", name="Claude 3 Opus", provider=self.name),
        ]
