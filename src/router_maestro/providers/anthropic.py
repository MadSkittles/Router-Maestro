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
                    if not isinstance(data, dict):
                        raise TypeError("messages response must be an object")
                    blocks = data["content"]
                    if not isinstance(blocks, list) or not blocks:
                        raise ValueError("messages content must be a non-empty list")
                    model = self._validated_response_model(data, request.model)
                    stop_reason = self._validated_optional_string(
                        data, "stop_reason", default="stop"
                    )
                    usage = self._validated_token_usage(
                        data.get("usage"),
                        fields=("input_tokens", "output_tokens"),
                        label="messages response",
                    )
                    usage_values = usage or {}
                    input_tokens = usage_values.get("input_tokens", 0)
                    output_tokens = usage_values.get("output_tokens", 0)
                    for block in blocks:
                        if not isinstance(block, dict):
                            raise TypeError("content block must be an object")
                        block_type = block.get("type")
                        if not isinstance(block_type, str):
                            raise TypeError("content block type must be a string")
                        if block_type == "text" and not isinstance(block.get("text"), str):
                            raise TypeError("text content block must contain text")
                        if block_type == "tool_use" and not isinstance(
                            block.get("input", {}), dict
                        ):
                            raise TypeError("tool use input must be an object")
                        if block_type == "tool_use" and (
                            not isinstance(block.get("id"), str)
                            or not isinstance(block.get("name"), str)
                        ):
                            raise TypeError("tool use block requires string id and name")
                        if block_type == "thinking":
                            if not isinstance(block.get("thinking"), str):
                                raise TypeError("thinking block must contain thinking text")
                            signature = block.get("signature")
                            if signature is not None and not isinstance(signature, str):
                                raise TypeError("thinking signature must be a string or null")
                        if block_type == "redacted_thinking" and not isinstance(
                            block.get("data"), str
                        ):
                            raise TypeError("redacted thinking data must be a string")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    self._raise_protocol_error(self.name, request.model, e)

                # Extract content from Anthropic response
                content = ""
                tool_calls = []
                thinking = ""
                thinking_signature: str | None = None
                for block in blocks:
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
                    elif block.get("type") == "thinking" and request.thinking_type in {
                        "enabled",
                        "adaptive",
                    }:
                        block_thinking = block.get("thinking")
                        if not isinstance(block_thinking, str):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("thinking block must contain thinking text"),
                            )
                        thinking += block_thinking
                        signature = block.get("signature")
                        if signature is not None and not isinstance(signature, str):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("thinking signature must be a string or null"),
                            )
                        if signature and thinking_signature is None:
                            thinking_signature = signature
                    elif block.get("type") == "redacted_thinking" and request.thinking_type in {
                        "enabled",
                        "adaptive",
                    }:
                        redacted_data = block.get("data")
                        if not isinstance(redacted_data, str):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError("redacted thinking data must be a string"),
                            )
                        if redacted_data and thinking_signature is None:
                            thinking_signature = redacted_data

                if not content and not tool_calls and not thinking and not thinking_signature:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        ValueError("messages response contains no deliverable output"),
                    )

                logger.debug("Anthropic chat completion successful")
                return ChatResponse(
                    content=content or None,
                    model=model,
                    finish_reason=stop_reason,
                    usage={
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    tool_calls=tool_calls if tool_calls else None,
                    thinking=thinking or None,
                    thinking_signature=thinking_signature,
                )
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

                        try:
                            data = json.loads(data_str)
                            if not isinstance(data, dict):
                                raise TypeError("stream event must be an object")
                            event_type = data.get("type")
                            if not isinstance(event_type, str):
                                raise TypeError("stream event type must be a string")
                            if event_type == "content_block_delta" and not isinstance(
                                data.get("delta"), dict
                            ):
                                raise TypeError("content block delta must be an object")
                            if event_type == "content_block_start" and not isinstance(
                                data.get("content_block"), dict
                            ):
                                raise TypeError("content block must be an object")
                            if event_type == "message_delta" and not isinstance(
                                data.get("delta"), dict
                            ):
                                raise TypeError("message delta must be an object")
                            index = data.get("index")
                            if event_type in {"content_block_start", "content_block_delta"} and (
                                not isinstance(index, int) or isinstance(index, bool)
                            ):
                                raise TypeError("content block index must be an integer")
                            if event_type == "message_start":
                                message = data.get("message")
                                if not isinstance(message, dict):
                                    raise TypeError("message_start message must be an object")
                                usage = message.get("usage", {})
                                if not isinstance(usage, dict):
                                    raise TypeError("message_start usage must be an object")
                                input_tokens = usage.get("input_tokens", 0)
                                if not isinstance(input_tokens, int) or isinstance(
                                    input_tokens, bool
                                ):
                                    raise TypeError("message_start input_tokens must be integer")
                            if event_type == "content_block_start":
                                block = data["content_block"]
                                block_type = block.get("type")
                                if not isinstance(block_type, str):
                                    raise TypeError("content block type must be a string")
                                if block_type == "tool_use":
                                    if (
                                        not isinstance(block.get("id"), str)
                                        or not isinstance(block.get("name"), str)
                                        or not isinstance(block.get("input", {}), dict)
                                    ):
                                        raise TypeError("tool use block is malformed")
                            if event_type == "content_block_delta":
                                delta = data["delta"]
                                delta_type = delta.get("type")
                                if not isinstance(delta_type, str):
                                    raise TypeError("content delta type must be a string")
                                field_by_type = {
                                    "text_delta": "text",
                                    "thinking_delta": "thinking",
                                    "signature_delta": "signature",
                                    "input_json_delta": "partial_json",
                                }
                                field = field_by_type.get(delta_type)
                                if field is not None and not isinstance(delta.get(field), str):
                                    raise TypeError(f"{delta_type} {field} must be a string")
                            if event_type == "message_delta":
                                delta = data["delta"]
                                stop_reason = delta.get("stop_reason")
                                if stop_reason is not None and not isinstance(stop_reason, str):
                                    raise TypeError("message stop_reason must be a string or null")
                                if not isinstance(data.get("usage", {}), dict):
                                    raise TypeError("message_delta usage must be an object")
                                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                                if not isinstance(output_tokens, int) or isinstance(
                                    output_tokens, bool
                                ):
                                    raise TypeError("message_delta output_tokens must be integer")
                        except (json.JSONDecodeError, TypeError) as e:
                            self._raise_protocol_error(self.name, request.model, e)

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
