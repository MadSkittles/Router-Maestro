"""GitHub Copilot provider implementation."""

import time
from collections.abc import AsyncIterator

import httpx

from router_maestro.auth import AuthManager, AuthType
from router_maestro.auth.github_oauth import get_copilot_token
from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
)
from router_maestro.utils import get_logger

logger = get_logger("providers.copilot")

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_CHAT_URL = f"{COPILOT_BASE_URL}/chat/completions"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"

# Model cache TTL in seconds (5 minutes)
MODELS_CACHE_TTL = 300


class CopilotProvider(BaseProvider):
    """GitHub Copilot provider."""

    name = "github-copilot"

    def __init__(self) -> None:
        self.auth_manager = AuthManager()
        self._cached_token: str | None = None
        self._token_expires: int = 0
        # Model cache
        self._models_cache: list[ModelInfo] | None = None
        self._models_cache_expires: float = 0
        # Reusable HTTP client
        self._client: httpx.AsyncClient | None = None

    def is_authenticated(self) -> bool:
        """Check if authenticated with GitHub Copilot."""
        cred = self.auth_manager.get_credential("github-copilot")
        return cred is not None and cred.type == AuthType.OAUTH

    async def ensure_token(self) -> None:
        """Ensure we have a valid Copilot token, refreshing if needed."""
        cred = self.auth_manager.get_credential("github-copilot")
        if not cred or cred.type != AuthType.OAUTH:
            logger.error("Not authenticated with GitHub Copilot")
            raise ProviderError("Not authenticated with GitHub Copilot", status_code=401)

        current_time = int(time.time())

        # Check if we need to refresh (token expired or will expire soon)
        if self._cached_token and self._token_expires > current_time + 60:
            return  # Token still valid

        logger.debug("Refreshing Copilot token")
        # Refresh the Copilot token using the GitHub token
        client = self._get_client()
        try:
            copilot_token = await get_copilot_token(client, cred.refresh)
            self._cached_token = copilot_token.token
            self._token_expires = copilot_token.expires_at

            # Update stored credential with new access token
            cred.access = copilot_token.token
            cred.expires = copilot_token.expires_at
            self.auth_manager.save()
            logger.debug("Copilot token refreshed, expires at %d", copilot_token.expires_at)
        except httpx.HTTPError as e:
            logger.error("Failed to refresh Copilot token: %s", e)
            raise ProviderError(f"Failed to refresh Copilot token: {e}", retryable=True)

    def _get_headers(self, vision_request: bool = False) -> dict[str, str]:
        """Get headers for Copilot API requests.

        Args:
            vision_request: Whether this request contains images (vision)
        """
        if not self._cached_token:
            raise ProviderError("No valid token available", status_code=401)

        headers = {
            "Authorization": f"Bearer {self._cached_token}",
            "Content-Type": "application/json",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot/1.0.0",
            "Copilot-Integration-Id": "vscode-chat",
        }

        if vision_request:
            headers["Copilot-Vision-Request"] = "true"

        return headers

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create a reusable HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _build_messages_payload(self, request: ChatRequest) -> tuple[list[dict], bool]:
        """Build messages payload and detect if images are present.

        Args:
            request: The chat request

        Returns:
            Tuple of (messages list, has_images flag)
        """
        messages = []
        has_images = False

        for m in request.messages:
            msg: dict = {"role": m.role, "content": m.content}
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            messages.append(msg)

            # Check if this message contains images (multimodal content)
            if isinstance(m.content, list):
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_images = True
                        break

        return messages, has_images

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion via Copilot."""
        await self.ensure_token()

        messages, has_images = self._build_messages_payload(request)

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        logger.debug("Copilot chat completion: model=%s", request.model)
        client = self._get_client()
        try:
            response = await client.post(
                COPILOT_CHAT_URL,
                json=payload,
                headers=self._get_headers(vision_request=has_images),
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                import json

                logger.error("Copilot API returned empty choices: %s", json.dumps(data)[:500])
                raise ProviderError(
                    f"Copilot API returned empty choices: {json.dumps(data)[:500]}",
                    status_code=500,
                    retryable=True,
                )

            logger.debug("Copilot chat completion successful")
            return ChatResponse(
                content=choices[0]["message"]["content"],
                model=data.get("model", request.model),
                finish_reason=choices[0].get("finish_reason", "stop"),
                usage=data.get("usage"),
            )
        except httpx.HTTPStatusError as e:
            retryable = e.response.status_code in (429, 500, 502, 503, 504)
            try:
                error_body = e.response.text
            except Exception:
                error_body = ""
            logger.error("Copilot API error: %d - %s", e.response.status_code, error_body[:200])
            raise ProviderError(
                f"Copilot API error: {e.response.status_code} - {error_body}",
                status_code=e.response.status_code,
                retryable=retryable,
            )
        except httpx.HTTPError as e:
            logger.error("Copilot HTTP error: %s", e)
            raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion via Copilot."""
        await self.ensure_token()

        messages, has_images = self._build_messages_payload(request)

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        logger.debug("Copilot streaming chat: model=%s", request.model)
        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                COPILOT_CHAT_URL,
                json=payload,
                headers=self._get_headers(vision_request=has_images),
            ) as response:
                response.raise_for_status()

                stream_finished = False
                async for line in response.aiter_lines():
                    if stream_finished:
                        break

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    import json

                    data = json.loads(data_str)

                    # Extract usage if present (may come in separate chunk)
                    usage = data.get("usage")

                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        finish_reason = data["choices"][0].get("finish_reason")
                        tool_calls = delta.get("tool_calls")

                        if content or finish_reason or usage or tool_calls:
                            yield ChatStreamChunk(
                                content=content,
                                finish_reason=finish_reason,
                                usage=usage,
                                tool_calls=tool_calls,
                            )

                        # Mark stream as finished after receiving finish_reason
                        if finish_reason:
                            stream_finished = True
                    elif usage:
                        # Handle usage-only chunks (no choices)
                        yield ChatStreamChunk(
                            content="",
                            finish_reason=None,
                            usage=usage,
                        )
        except httpx.HTTPStatusError as e:
            retryable = e.response.status_code in (429, 500, 502, 503, 504)
            try:
                error_body = e.response.text
            except Exception:
                error_body = ""
            logger.error(
                "Copilot stream API error: %d - %s",
                e.response.status_code,
                error_body[:200],
            )
            raise ProviderError(
                f"Copilot API error: {e.response.status_code} - {error_body}",
                status_code=e.response.status_code,
                retryable=retryable,
            )
        except httpx.HTTPError as e:
            logger.error("Copilot stream HTTP error: %s", e)
            raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def list_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """List available Copilot models from API with caching.

        Args:
            force_refresh: Force refresh the cache

        Returns:
            List of available models
        """
        current_time = time.time()

        # Return cached models if valid
        if (
            not force_refresh
            and self._models_cache is not None
            and current_time < self._models_cache_expires
        ):
            logger.debug("Using cached Copilot models (%d models)", len(self._models_cache))
            return self._models_cache

        await self.ensure_token()

        logger.debug("Fetching Copilot models from API")
        client = self._get_client()
        try:
            response = await client.get(
                COPILOT_MODELS_URL,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for model in data.get("data", []):
                # Only include models that are enabled in model picker
                if model.get("model_picker_enabled", True):
                    models.append(
                        ModelInfo(
                            id=model["id"],
                            name=model.get("name", model["id"]),
                            provider=self.name,
                        )
                    )

            # Update cache
            self._models_cache = models
            self._models_cache_expires = current_time + MODELS_CACHE_TTL

            logger.info("Fetched %d Copilot models", len(models))
            return models
        except httpx.HTTPError as e:
            # If cache exists, return stale cache on error
            if self._models_cache is not None:
                logger.warning("Failed to refresh Copilot models, using stale cache: %s", e)
                return self._models_cache
            logger.error("Failed to list Copilot models: %s", e)
            raise ProviderError(f"Failed to list models: {e}", retryable=True)
