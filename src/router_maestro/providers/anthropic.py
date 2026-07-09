"""Anthropic provider implementation."""

import json
from collections.abc import AsyncIterator

import httpx

from router_maestro.auth import AuthManager, AuthType
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    TIMEOUT_STREAMING,
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
            elif msg.role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content,
                            }
                        ],
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tool_call in msg.tool_calls:
                    function = tool_call.get("function", {})
                    arguments = function.get("arguments", "{}")
                    try:
                        tool_input = json.loads(arguments) if arguments else {}
                    except json.JSONDecodeError:
                        tool_input = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id", ""),
                            "name": function.get("name", ""),
                            "input": tool_input,
                        }
                    )
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": msg.role, "content": msg.content})

        return system_prompt, converted

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style function tools to Anthropic tool definitions."""
        converted = []
        for tool in tools:
            function = tool.get("function") if tool.get("type") == "function" else None
            if isinstance(function, dict):
                anthropic_tool = {
                    "name": function.get("name", ""),
                    "input_schema": function.get("parameters") or {"type": "object"},
                }
                if function.get("description"):
                    anthropic_tool["description"] = function["description"]
                converted.append(anthropic_tool)
            else:
                converted.append(tool)
        return converted

    def _convert_tool_choice(self, tool_choice: str | dict) -> dict | str:
        """Convert OpenAI-style tool_choice to Anthropic tool_choice."""
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "required":
            return {"type": "any"}
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            name = (tool_choice.get("function") or {}).get("name")
            if name:
                return {"type": "tool", "name": name}
        return tool_choice

    def _build_payload(self, request: ChatRequest, *, stream: bool = False) -> dict:
        """Build an Anthropic messages payload."""
        system_prompt, messages = self._convert_messages(request.messages)

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }
        if stream:
            payload["stream"] = True
        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature != 1.0:
            payload["temperature"] = request.temperature
        if request.thinking_type and request.thinking_type != "disabled":
            thinking_config: dict = {"type": request.thinking_type}
            if request.thinking_budget:
                thinking_config["budget_tokens"] = request.thinking_budget
            payload["thinking"] = thinking_config
        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)
        if request.tool_choice:
            payload["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        return payload

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion via Anthropic."""
        payload = self._build_payload(request)

        logger.debug("Anthropic chat completion: model=%s", request.model)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=TIMEOUT_NON_STREAMING,
                )
                response.raise_for_status()
                data = response.json()

                # Extract content from Anthropic response
                content = ""
                tool_calls = []
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )

                logger.debug("Anthropic chat completion successful")
                return ChatResponse(
                    content=content or None,
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
                    tool_calls=tool_calls if tool_calls else None,
                )
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code in (429, 500, 502, 503, 504, 529)
                logger.error("Anthropic API error: %d", e.response.status_code)
                raise ProviderError(
                    f"Anthropic API error: {e.response.status_code}",
                    status_code=e.response.status_code,
                    retryable=retryable,
                )
            except httpx.TimeoutException as e:
                self._raise_timeout_error("Anthropic", e, logger)
            except httpx.HTTPError as e:
                logger.error("Anthropic HTTP error: %s", e)
                raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion via Anthropic."""
        payload = self._build_payload(request, stream=True)

        logger.debug("Anthropic streaming chat: model=%s", request.model)
        # Anthropic native stop_reason -> internal OpenAI-style finish_reason.
        stop_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        # Map an Anthropic content-block index to a sequential tool-call index so
        # downstream consumers receive OpenAI-style tool_call deltas.
        block_to_tool_index: dict[int, int] = {}
        next_tool_index = 0
        prompt_tokens = 0
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=TIMEOUT_STREAMING,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if not data_str:
                            continue

                        data = json.loads(data_str)
                        event_type = data.get("type")

                        if event_type == "message_start":
                            usage = data.get("message", {}).get("usage", {})
                            prompt_tokens = usage.get("input_tokens", 0)
                        elif event_type == "content_block_start":
                            index = data.get("index", 0)
                            block = data.get("content_block", {})
                            if block.get("type") == "tool_use":
                                tool_index = next_tool_index
                                next_tool_index += 1
                                block_to_tool_index[index] = tool_index
                                yield ChatStreamChunk(
                                    content="",
                                    finish_reason=None,
                                    tool_calls=[
                                        {
                                            "index": tool_index,
                                            "id": block.get("id", ""),
                                            "type": "function",
                                            "function": {
                                                "name": block.get("name", ""),
                                                "arguments": "",
                                            },
                                        }
                                    ],
                                )
                        elif event_type == "content_block_delta":
                            index = data.get("index", 0)
                            delta = data.get("delta", {})
                            delta_type = delta.get("type")
                            if delta_type == "text_delta":
                                yield ChatStreamChunk(
                                    content=delta.get("text", ""),
                                    finish_reason=None,
                                )
                            elif delta_type == "thinking_delta":
                                yield ChatStreamChunk(
                                    content="",
                                    finish_reason=None,
                                    thinking=delta.get("thinking", "") or None,
                                )
                            elif delta_type == "signature_delta":
                                yield ChatStreamChunk(
                                    content="",
                                    finish_reason=None,
                                    thinking_signature=delta.get("signature") or None,
                                )
                            elif delta_type == "input_json_delta":
                                tool_index = block_to_tool_index.get(index)
                                if tool_index is not None:
                                    yield ChatStreamChunk(
                                        content="",
                                        finish_reason=None,
                                        tool_calls=[
                                            {
                                                "index": tool_index,
                                                "function": {
                                                    "arguments": delta.get("partial_json", ""),
                                                },
                                            }
                                        ],
                                    )
                        elif event_type == "message_delta":
                            delta = data.get("delta", {})
                            stop_reason = delta.get("stop_reason")
                            finish_reason = (
                                stop_reason_map.get(stop_reason, "stop") if stop_reason else None
                            )
                            output_tokens = data.get("usage", {}).get("output_tokens", 0)
                            yield ChatStreamChunk(
                                content="",
                                finish_reason=finish_reason,
                                usage={
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": output_tokens,
                                    "total_tokens": prompt_tokens + output_tokens,
                                },
                            )
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code in (429, 500, 502, 503, 504, 529)
                logger.error("Anthropic stream API error: %d", e.response.status_code)
                raise ProviderError(
                    f"Anthropic API error: {e.response.status_code}",
                    status_code=e.response.status_code,
                    retryable=retryable,
                )
            except httpx.TimeoutException as e:
                self._raise_timeout_error("Anthropic", e, logger, stream=True)
            except httpx.HTTPError as e:
                logger.error("Anthropic stream HTTP error: %s", e)
                raise ProviderError(f"HTTP error: {e}", retryable=True)

    async def list_models(self) -> list[ModelInfo]:
        """List available Anthropic models."""
        # Anthropic doesn't have a models endpoint, return known models
        logger.debug("Returning known Anthropic models")
        return [
            ModelInfo(
                id="claude-sonnet-4-20250514",
                name="Claude Sonnet 4",
                provider=self.name,
                max_context_window_tokens=200000,
                max_output_tokens=16384,
                supports_thinking=True,
            ),
            ModelInfo(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                provider=self.name,
                max_context_window_tokens=200000,
                max_output_tokens=8192,
            ),
            ModelInfo(
                id="claude-3-5-haiku-20241022",
                name="Claude 3.5 Haiku",
                provider=self.name,
                max_context_window_tokens=200000,
                max_output_tokens=8192,
            ),
            ModelInfo(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                provider=self.name,
                max_context_window_tokens=200000,
                max_output_tokens=4096,
            ),
        ]
