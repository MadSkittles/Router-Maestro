"""Anthropic provider implementation."""

import json
from collections.abc import AsyncIterator

import httpx

from router_maestro.auth import AuthManager, AuthType
from router_maestro.providers.anthropic_codec import (
    AnthropicCodecError,
    AnthropicStreamDecoder,
    decode_message_response,
)
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    TIMEOUT_STREAMING,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
)
from router_maestro.utils import get_logger
from router_maestro.utils.context_window import normalize_thinking_budget

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
            raise ProviderError(
                "Not authenticated with Anthropic",
                status_code=401,
                kind=ProviderFailureKind.AUTHENTICATION,
                provider=self.name,
            )
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
                if msg.refusal:
                    content.append({"type": "text", "text": msg.refusal})
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
                converted.append(
                    {
                        "role": msg.role,
                        "content": msg.content if msg.content is not None else msg.refusal,
                    }
                )

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
        self._validate_provider_extensions(request)
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
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.thinking_type == "adaptive":
            payload["thinking"] = {"type": "adaptive"}
        elif request.thinking_type == "enabled" and request.thinking_budget is not None:
            budget = normalize_thinking_budget(request.thinking_budget, payload["max_tokens"])
            if budget is not None:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }

        if request.reasoning_effort is not None:
            payload["output_config"] = {"effort": request.reasoning_effort}

        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)
        if request.tool_choice:
            payload["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        for parameter in (
            "frequency_penalty",
            "presence_penalty",
            "candidate_count",
            "response_mime_type",
        ):
            if getattr(request, parameter) is not None:
                raise RequestOptionError(
                    f"Anthropic does not support request option '{parameter}'",
                    provider=self.name,
                    model=request.model,
                    parameter=parameter,
                )
        if request.stop is not None and request.stop_sequences is not None:
            raise RequestOptionError(
                "Anthropic request contains both 'stop' and 'stop_sequences'",
                provider=self.name,
                model=request.model,
                parameter="stop",
            )
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        stop_sequences = (
            request.stop_sequences if request.stop_sequences is not None else request.stop
        )
        if stop_sequences is not None:
            payload["stop_sequences"] = (
                [stop_sequences] if isinstance(stop_sequences, str) else stop_sequences
            )
        if request.metadata is not None or request.user is not None:
            metadata = dict(request.metadata or {})
            if request.user is not None:
                if "user_id" in metadata and metadata["user_id"] != request.user:
                    raise RequestOptionError(
                        "Anthropic metadata.user_id conflicts with user",
                        provider=self.name,
                        model=request.model,
                        parameter="user",
                    )
                metadata["user_id"] = request.user
            payload["metadata"] = metadata
        if request.service_tier is not None:
            payload["service_tier"] = request.service_tier
        return payload

    def validate_chat_request(self, request: ChatRequest, *, stream: bool) -> None:
        """Exercise the payload policy without performing upstream I/O."""
        self._build_payload(request, stream=stream)

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
                try:
                    data = response.json()
                    result = decode_message_response(
                        data,
                        fallback_model=request.model,
                        include_reasoning=request.thinking_type in {"enabled", "adaptive"},
                    )
                except (json.JSONDecodeError, AnthropicCodecError) as e:
                    self._raise_protocol_error(self.name, request.model, e)

                logger.debug("Anthropic chat completion successful")
                return result
            except httpx.HTTPStatusError as e:
                self._raise_http_status_error(
                    "Anthropic", e, logger, provider=self.name, model=request.model
                )
            except httpx.TimeoutException as e:
                self._raise_timeout_error(
                    "Anthropic", e, logger, provider=self.name, model=request.model
                )
            except httpx.HTTPError as e:
                self._raise_http_error(
                    "Anthropic", e, logger, provider=self.name, model=request.model
                )

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion via Anthropic."""
        payload = self._build_payload(request, stream=True)

        logger.debug("Anthropic streaming chat: model=%s", request.model)
        decoder = AnthropicStreamDecoder(
            include_reasoning=request.thinking_type in {"enabled", "adaptive"}
        )
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

                        try:
                            data = json.loads(data_str)
                            chunks = decoder.decode_event(data)
                        except (json.JSONDecodeError, AnthropicCodecError) as e:
                            self._raise_protocol_error(self.name, request.model, e)

                        for chunk in chunks:
                            yield chunk

                    try:
                        decoder.finalize()
                    except AnthropicCodecError as e:
                        self._raise_protocol_error(self.name, request.model, e)
            except httpx.HTTPStatusError as e:
                self._raise_http_status_error(
                    "Anthropic",
                    e,
                    logger,
                    stream=True,
                    provider=self.name,
                    model=request.model,
                )
            except httpx.TimeoutException as e:
                self._raise_timeout_error(
                    "Anthropic",
                    e,
                    logger,
                    stream=True,
                    provider=self.name,
                    model=request.model,
                )
            except httpx.HTTPError as e:
                self._raise_http_error(
                    "Anthropic",
                    e,
                    logger,
                    stream=True,
                    provider=self.name,
                    model=request.model,
                )

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
