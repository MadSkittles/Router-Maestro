"""Shared OpenAI-compatible chat provider logic."""

import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from logging import Logger

import httpx

from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    TIMEOUT_STREAMING,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    RequestOptionError,
)
from router_maestro.providers.tool_parsing import recover_tool_calls_from_content
from router_maestro.routing.model_ref import validate_upstream_model_id
from router_maestro.utils.reasoning import budget_to_effort, downgrade_for_upstream


class OpenAIChatProvider(BaseProvider, ABC):
    """Shared OpenAI-compatible chat behavior."""

    def __init__(self, base_url: str, logger: Logger) -> None:
        self.base_url = base_url.rstrip("/")
        self._logger = logger

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Return headers for the API request."""

    def _reject_option(self, request: ChatRequest, parameter: str) -> None:
        raise RequestOptionError(
            f"{self.name} does not support request option '{parameter}'",
            provider=self.name,
            model=request.model,
            parameter=parameter,
        )

    def _error_label(self) -> str:
        return self.name

    def _parse_model_catalog(self, response: httpx.Response) -> list[str]:
        """Parse the shared OpenAI model-catalog envelope."""
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("model catalog must be an object")
            models = data["data"]
            if not isinstance(models, list):
                raise TypeError("model catalog data must be a list")
            model_ids: list[str] = []
            for model in models:
                if not isinstance(model, dict):
                    raise TypeError("model catalog entry must be an object")
                model_id = model["id"]
                validate_upstream_model_id(model_id)
                model_ids.append(model_id)
            return model_ids
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self._raise_protocol_error(self.name, None, e)

    def _build_payload(self, request: ChatRequest, stream: bool) -> dict:
        self._validate_provider_extensions(request)
        messages = []
        for m in request.messages:
            msg: dict = {"role": m.role, "content": m.content}
            if m.tool_call_id is not None:
                msg["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                msg["tool_calls"] = m.tool_calls
            if m.refusal is not None:
                msg["refusal"] = m.refusal
            messages.append(msg)

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        for parameter in ("top_k", "candidate_count", "response_mime_type"):
            if getattr(request, parameter) is not None:
                self._reject_option(request, parameter)
        if request.stop is not None and request.stop_sequences is not None:
            self._reject_option(request, "stop")
        option_values = {
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop if request.stop is not None else request.stop_sequences,
            "user": request.user,
            "metadata": request.metadata,
            "service_tier": request.service_tier,
        }
        for key, value in option_values.items():
            if value is not None:
                payload[key] = value

        # Forward OpenAI-style reasoning_effort. Fall back to deriving it from
        # thinking_budget when only the Anthropic-style budget is set. Minimal
        # is native but intentionally has no implicit token-budget equivalent;
        # xhigh/max are extensions and get downgraded to "high".
        effort = request.reasoning_effort or budget_to_effort(request.thinking_budget)
        if (
            request.reasoning_effort is None
            and request.thinking_budget is not None
            and request.thinking_budget > 0
            and effort is None
        ):
            raise RequestOptionError(
                f"{self.name} has no reasoning tier at or below the requested budget",
                provider=self.name,
                model=request.model,
                parameter="thinking_budget",
            )
        upstream_effort = downgrade_for_upstream(effort)
        if upstream_effort is not None:
            if effort in ("xhigh", "max"):
                self._logger.warning(
                    "%s does not accept reasoning_effort=%s; downgrading to high",
                    self._error_label(),
                    effort,
                )
            payload["reasoning_effort"] = upstream_effort

        return payload

    def validate_chat_request(self, request: ChatRequest, *, stream: bool) -> None:
        """Exercise the payload policy without performing upstream I/O."""
        self._build_payload(request, stream)

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request, stream=False)
        label = self._error_label()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=TIMEOUT_NON_STREAMING,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise TypeError("chat response must be an object")
                    choices = data["choices"]
                    if not isinstance(choices, list) or not choices:
                        raise ValueError("chat response choices must be a non-empty list")
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        raise TypeError("chat response choice must be an object")
                    message = choice["message"]
                    if not isinstance(message, dict):
                        raise TypeError("chat response message must be an object")
                    content = message.get("content")
                    refusal = message.get("refusal")
                    tool_calls = message.get("tool_calls")
                    if content is not None and not isinstance(content, str):
                        raise TypeError("chat response content must be a string or null")
                    if "refusal" in message and (not isinstance(refusal, str) or not refusal):
                        raise TypeError("chat response refusal must be a non-empty string")
                    if tool_calls is not None and not isinstance(tool_calls, list):
                        raise TypeError("chat response tool_calls must be a list or null")
                    if tool_calls:
                        for tool_call in tool_calls:
                            if not isinstance(tool_call, dict):
                                raise TypeError("chat response tool call must be an object")
                            function = tool_call.get("function")
                            if (
                                not isinstance(tool_call.get("id"), str)
                                or not isinstance(function, dict)
                                or not isinstance(function.get("name"), str)
                                or not isinstance(function.get("arguments"), str)
                            ):
                                raise TypeError("chat response tool call is malformed")
                    if not content and not refusal and not tool_calls:
                        raise ValueError("chat response must contain text, refusal, or tool calls")
                    model = self._validated_response_model(data, request.model)
                    finish_reason = self._validated_optional_string(
                        choice, "finish_reason", default="stop"
                    )
                    usage = self._validated_token_usage(
                        data.get("usage"),
                        fields=("prompt_tokens", "completion_tokens", "total_tokens"),
                        label="chat response",
                        detail_fields={
                            "prompt_tokens_details": ("cached_tokens",),
                            "completion_tokens_details": ("reasoning_tokens",),
                        },
                    )
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
                    self._raise_protocol_error(self.name, request.model, e)

                content, tool_calls = recover_tool_calls_from_content(
                    content, tool_calls, finish_reason
                )

                return ChatResponse(
                    content=content,
                    model=model,
                    refusal=refusal,
                    finish_reason=finish_reason,
                    usage=usage,
                    tool_calls=tool_calls,
                )
            except httpx.HTTPStatusError as e:
                self._raise_http_status_error(
                    label,
                    e,
                    self._logger,
                    provider=self.name,
                    model=request.model,
                )
            except httpx.TimeoutException as e:
                self._raise_timeout_error(
                    label,
                    e,
                    self._logger,
                    provider=self.name,
                    model=request.model,
                )
            except httpx.HTTPError as e:
                self._raise_http_error(
                    label,
                    e,
                    self._logger,
                    provider=self.name,
                    model=request.model,
                )

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
                    timeout=TIMEOUT_STREAMING,
                ) as response:
                    # Streamed responses defer body reads; if the upstream
                    # returns an error status, pull the body *inside* the
                    # stream context so the connection is still open. After
                    # the `async with` exits the response is closed and
                    # `aread()` would raise StreamClosed, leaving the log as
                    # "API error: 4xx -" with no upstream detail.
                    if response.status_code >= 400:
                        with contextlib.suppress(Exception):
                            await response.aread()
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if not isinstance(data, dict):
                                raise TypeError("stream event must be an object")
                            usage = self._validated_token_usage(
                                data.get("usage"),
                                fields=("prompt_tokens", "completion_tokens", "total_tokens"),
                                label="stream",
                                detail_fields={
                                    "prompt_tokens_details": ("cached_tokens",),
                                    "completion_tokens_details": ("reasoning_tokens",),
                                },
                            )
                            choices = data.get("choices")
                            if choices is not None and not isinstance(choices, list):
                                raise TypeError("stream choices must be a list")
                            if choices:
                                choice = choices[0]
                                if not isinstance(choice, dict):
                                    raise TypeError("stream choice must be an object")
                                delta = choice.get("delta", {})
                                if not isinstance(delta, dict):
                                    raise TypeError("stream delta must be an object")
                                content = delta.get("content")
                                refusal = delta.get("refusal")
                                tool_calls = delta.get("tool_calls")
                                finish_reason = choice.get("finish_reason")
                                if content is not None and not isinstance(content, str):
                                    raise TypeError("stream content must be a string or null")
                                if refusal is not None and not isinstance(refusal, str):
                                    raise TypeError("stream refusal must be a string or null")
                                if tool_calls is not None and not isinstance(tool_calls, list):
                                    raise TypeError("stream tool_calls must be a list or null")
                                if tool_calls:
                                    for tool_call in tool_calls:
                                        if not isinstance(tool_call, dict):
                                            raise TypeError("stream tool call delta must be object")
                                        index = tool_call.get("index")
                                        if index is not None and (
                                            not isinstance(index, int) or isinstance(index, bool)
                                        ):
                                            raise TypeError(
                                                "stream tool call index must be integer"
                                            )
                                        for field in ("id", "type"):
                                            value = tool_call.get(field)
                                            if value is not None and not isinstance(value, str):
                                                raise TypeError(
                                                    f"stream tool call {field} must be string"
                                                )
                                        function = tool_call.get("function")
                                        if function is not None:
                                            if not isinstance(function, dict):
                                                raise TypeError(
                                                    "stream tool call function must be object"
                                                )
                                            for field in ("name", "arguments"):
                                                value = function.get(field)
                                                if value is not None and not isinstance(value, str):
                                                    raise TypeError(
                                                        "stream tool function "
                                                        f"{field} must be string"
                                                    )
                                if finish_reason is not None and not isinstance(finish_reason, str):
                                    raise TypeError("stream finish_reason must be a string or null")
                        except (json.JSONDecodeError, TypeError) as e:
                            self._raise_protocol_error(self.name, request.model, e)

                        if choices:
                            content = content or ""

                            if content or refusal or finish_reason or usage or tool_calls:
                                yield ChatStreamChunk(
                                    content=content,
                                    refusal=refusal or None,
                                    finish_reason=finish_reason,
                                    usage=usage,
                                    tool_calls=tool_calls,
                                )
                        elif usage:
                            yield ChatStreamChunk(
                                content="",
                                finish_reason=None,
                                usage=usage,
                            )
            except httpx.HTTPStatusError as e:
                self._raise_http_status_error(
                    label,
                    e,
                    self._logger,
                    stream=True,
                    include_body=True,
                    provider=self.name,
                    model=request.model,
                )
            except httpx.TimeoutException as e:
                self._raise_timeout_error(
                    label,
                    e,
                    self._logger,
                    stream=True,
                    provider=self.name,
                    model=request.model,
                )
            except httpx.HTTPError as e:
                self._raise_http_error(
                    label,
                    e,
                    self._logger,
                    stream=True,
                    provider=self.name,
                    model=request.model,
                )
