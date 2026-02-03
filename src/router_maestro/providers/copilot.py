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
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponsesToolCall,
)
from router_maestro.utils import get_logger

logger = get_logger("providers.copilot")

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_CHAT_URL = f"{COPILOT_BASE_URL}/chat/completions"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
COPILOT_RESPONSES_URL = f"{COPILOT_BASE_URL}/responses"

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

    # Tools that are not supported by Copilot Responses API
    UNSUPPORTED_TOOL_TYPES = {"web_search", "web_search_preview", "code_interpreter"}

    def _filter_unsupported_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Filter out tools that are not supported by Copilot API.

        Args:
            tools: List of tool definitions

        Returns:
            Filtered list of tools, or None if empty
        """
        if not tools:
            return None

        filtered = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            # Only include function tools, filter out unsupported built-in tools
            if tool_type == "function":
                filtered.append(tool)
            elif tool_type not in self.UNSUPPORTED_TOOL_TYPES:
                filtered.append(tool)
            else:
                logger.debug("Filtering out unsupported tool type: %s", tool_type)

        return filtered if filtered else None

    def _build_responses_payload(self, request: ResponsesRequest) -> dict:
        """Build payload for Responses API request.

        Args:
            request: The responses request

        Returns:
            Payload dictionary for the API
        """
        payload: dict = {
            "model": request.model,
            "input": request.input,
            "stream": request.stream,
        }
        if request.instructions:
            payload["instructions"] = request.instructions
        if request.temperature != 1.0:
            payload["temperature"] = request.temperature
        if request.max_output_tokens:
            payload["max_output_tokens"] = request.max_output_tokens
        # Tool support - filter out unsupported tools
        filtered_tools = self._filter_unsupported_tools(request.tools)
        if filtered_tools:
            payload["tools"] = filtered_tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        return payload

    def _extract_response_content(self, data: dict) -> str:
        """Extract text content from Responses API response.

        Args:
            data: The response JSON data

        Returns:
            The extracted text content
        """
        content = ""
        for output in data.get("output", []):
            if output.get("type") == "message":
                for content_item in output.get("content", []):
                    if content_item.get("type") == "output_text":
                        content += content_item.get("text", "")
        return content

    def _extract_tool_calls(self, data: dict) -> list[ResponsesToolCall]:
        """Extract tool calls from Responses API response.

        Args:
            data: The response JSON data

        Returns:
            List of tool calls
        """
        tool_calls = []
        for output in data.get("output", []):
            if output.get("type") == "function_call":
                tool_calls.append(
                    ResponsesToolCall(
                        call_id=output.get("call_id", ""),
                        name=output.get("name", ""),
                        arguments=output.get("arguments", "{}"),
                    )
                )
        return tool_calls

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        """Generate a Responses API completion via Copilot (for Codex models)."""
        await self.ensure_token()

        payload = self._build_responses_payload(request)

        logger.debug("Copilot responses completion: model=%s", request.model)
        client = self._get_client()
        try:
            response = await client.post(
                COPILOT_RESPONSES_URL,
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            content = self._extract_response_content(data)
            tool_calls = self._extract_tool_calls(data)

            usage = None
            if "usage" in data:
                usage = data["usage"]

            logger.debug("Copilot responses completion successful")
            return ResponsesResponse(
                content=content,
                model=data.get("model", request.model),
                usage=usage,
                tool_calls=tool_calls if tool_calls else None,
            )
        except httpx.HTTPStatusError as e:
            retryable = e.response.status_code in (429, 500, 502, 503, 504)
            try:
                error_body = e.response.text
            except Exception:
                error_body = ""
            logger.error(
                "Copilot responses API error: %d - %s",
                e.response.status_code,
                error_body[:200],
            )
            raise ProviderError(
                f"Copilot API error: {e.response.status_code} - {error_body}",
                status_code=e.response.status_code,
                retryable=retryable,
            )
        except httpx.HTTPError as e:
            logger.error("Copilot responses HTTP error: %s", e)
            raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def responses_completion_stream(
        self, request: ResponsesRequest
    ) -> AsyncIterator[ResponsesStreamChunk]:
        """Generate a streaming Responses API completion via Copilot (for Codex models)."""
        await self.ensure_token()

        payload = self._build_responses_payload(request)
        payload["stream"] = True

        logger.debug("Copilot streaming responses: model=%s", request.model)
        logger.debug("Copilot responses payload: %s", payload)
        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                COPILOT_RESPONSES_URL,
                json=payload,
                headers=self._get_headers(),
            ) as response:
                # Check for errors before processing stream
                if response.status_code >= 400:
                    # Read the error body before the context closes
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    logger.error(
                        "Copilot responses stream API error: %d - %s",
                        response.status_code,
                        error_text,
                    )
                    retryable = response.status_code in (429, 500, 502, 503, 504)
                    raise ProviderError(
                        f"Copilot API error: {response.status_code} - {error_text}",
                        status_code=response.status_code,
                        retryable=retryable,
                    )

                stream_finished = False
                final_usage = None
                # Track current function call being streamed
                current_fc: dict | None = None

                async for line in response.aiter_lines():
                    if stream_finished:
                        break

                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        # Stream ended, emit final chunk if we haven't already
                        if not stream_finished:
                            yield ResponsesStreamChunk(
                                content="",
                                finish_reason="stop",
                                usage=final_usage,
                            )
                            stream_finished = True
                        break

                    import json

                    data = json.loads(data_str)
                    event_type = data.get("type", "")

                    # Handle text delta events
                    if event_type == "response.output_text.delta":
                        delta_text = data.get("delta", "")
                        if delta_text:
                            yield ResponsesStreamChunk(content=delta_text)

                    # Handle function call output_item.added - start of a new function call
                    elif event_type == "response.output_item.added":
                        item = data.get("item", {})
                        if item.get("type") == "function_call":
                            current_fc = {
                                "id": item.get("id", ""),
                                "call_id": item.get("call_id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                                "output_index": data.get("output_index", 0),
                            }

                    # Handle function call arguments delta
                    elif event_type == "response.function_call_arguments.delta":
                        delta = data.get("delta", "")
                        if current_fc and delta:
                            current_fc["arguments"] += delta
                            # Emit delta event for streaming
                            yield ResponsesStreamChunk(
                                content="",
                                tool_call_delta={
                                    "type": "function_call_arguments_delta",
                                    "item_id": current_fc["id"],
                                    "call_id": current_fc["call_id"],
                                    "name": current_fc["name"],
                                    "output_index": current_fc["output_index"],
                                    "delta": delta,
                                },
                            )

                    # Handle function call arguments done
                    elif event_type == "response.function_call_arguments.done":
                        if current_fc:
                            current_fc["arguments"] = data.get("arguments", current_fc["arguments"])
                            # Emit complete tool call
                            yield ResponsesStreamChunk(
                                content="",
                                tool_call=ResponsesToolCall(
                                    call_id=current_fc["call_id"],
                                    name=current_fc["name"],
                                    arguments=current_fc["arguments"],
                                ),
                            )
                            current_fc = None

                    # Handle output_item.done for function calls
                    elif event_type == "response.output_item.done":
                        item = data.get("item", {})
                        if item.get("type") == "function_call":
                            # Emit complete tool call if not already done
                            yield ResponsesStreamChunk(
                                content="",
                                tool_call=ResponsesToolCall(
                                    call_id=item.get("call_id", ""),
                                    name=item.get("name", ""),
                                    arguments=item.get("arguments", "{}"),
                                ),
                            )
                            current_fc = None

                    # Handle done event to get final usage
                    elif event_type == "response.done":
                        resp = data.get("response", {})
                        final_usage = resp.get("usage")
                        yield ResponsesStreamChunk(
                            content="",
                            finish_reason="stop",
                            usage=final_usage,
                        )
                        stream_finished = True

                    # Handle completed events
                    elif event_type == "response.completed":
                        # Final response received - emit finish chunk
                        resp = data.get("response", {})
                        if not final_usage:
                            final_usage = resp.get("usage")
                        yield ResponsesStreamChunk(
                            content="",
                            finish_reason="stop",
                            usage=final_usage,
                        )
                        stream_finished = True

                # If stream ended without explicit completion event, emit final chunk
                if not stream_finished:
                    logger.debug("Stream ended without completion event, emitting final chunk")
                    yield ResponsesStreamChunk(
                        content="",
                        finish_reason="stop",
                        usage=final_usage,
                    )

        except httpx.HTTPError as e:
            logger.error("Copilot responses stream HTTP error: %s", e)
            raise ProviderError(f"HTTP error: {e}", retryable=True)
