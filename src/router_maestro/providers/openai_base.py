"""Shared OpenAI-compatible chat provider logic."""

import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from logging import Logger
from typing import Any, NoReturn, cast

import httpx

from router_maestro.protocols import WireProtocol
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    TIMEOUT_STREAMING,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    RequestOptionError,
)
from router_maestro.providers.bindings import (
    OPENAI_COMPATIBLE_CHAT_BINDING,
    AttemptRequestContext,
    EndpointBinding,
    PreparedAttempt,
)
from router_maestro.providers.http_executor import ProviderHttpClientPool, SharedHttpExecutor
from router_maestro.providers.tool_parsing import recover_tool_calls_from_content
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef, validate_upstream_model_id
from router_maestro.utils.reasoning import budget_to_effort, downgrade_for_upstream
from router_maestro.utils.structured_output import output_format_to_response_format


def _request_audit():
    from router_maestro.runtime import get_current_request_context

    context = get_current_request_context()
    return context.audit if context is not None else None


class OpenAIChatProvider(BaseProvider, ABC):
    """Shared OpenAI-compatible chat behavior."""

    def __init__(self, base_url: str, logger: Logger) -> None:
        self.base_url = base_url.rstrip("/")
        self._logger = logger
        self._http_client_pool = ProviderHttpClientPool(lambda: httpx.AsyncClient())

    async def close(self) -> None:
        """Close the provider-owned reusable HTTP client."""
        await self._http_client_pool.close()

    def bindings(self) -> tuple[EndpointBinding, ...]:
        """Expose the provider's Chat endpoint as a raw wire binding."""
        bindings = getattr(self, "_generation_bindings", None)
        if bindings is not None:
            return bindings

        binding = EndpointBinding(
            id=OPENAI_COMPATIBLE_CHAT_BINDING,
            protocol=WireProtocol.OPENAI_CHAT,
            capabilities=ProviderCapabilities(
                operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM})
            ),
            dialect=OpenAICompatibleProviderDialect(self),
            executor=OpenAICompatibleHttpExecutor(self),
        )
        bindings = (binding,)
        self._generation_bindings = bindings
        return bindings

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
        response_format = output_format_to_response_format(request.output_format)
        if response_format is not None:
            payload["response_format"] = response_format
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
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        audit = _request_audit()
        if audit is not None:
            audit.record_upstream("POST", url, headers, payload)

        async with self._http_client_pool.lease() as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=TIMEOUT_NON_STREAMING,
                )
                if audit is not None:
                    audit.record_upstream_response(
                        response.status_code,
                        dict(response.headers),
                        response.content,
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
                    finish_reason=cast(str, finish_reason),
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
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        audit = _request_audit()
        if audit is not None:
            audit.record_upstream("POST", url, headers, payload)

        async with self._http_client_pool.lease() as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=TIMEOUT_STREAMING,
                ) as response:
                    if audit is not None:
                        audit.record_upstream_response(
                            response.status_code,
                            dict(response.headers),
                            stream_summary="stream opened",
                        )
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


class OpenAICompatibleProviderDialect:
    """Copy-on-write preparation for an OpenAI-compatible Chat endpoint."""

    id = "openai-compatible"

    def __init__(self, provider: OpenAIChatProvider) -> None:
        self.provider = provider

    async def prepare_attempt(
        self,
        *,
        binding_id: str,
        protocol: WireProtocol,
        model: ModelRef,
        payload: Mapping[str, Any],
        stream: bool,
        request_context: AttemptRequestContext,
    ) -> PreparedAttempt:
        del request_context
        if binding_id != OPENAI_COMPATIBLE_CHAT_BINDING:
            raise ValueError(f"Unknown OpenAI-compatible binding {binding_id!r}")
        if protocol is not WireProtocol.OPENAI_CHAT:
            raise ValueError("OpenAI-compatible Chat binding requires the Chat wire protocol")
        if model.provider != self.provider.name:
            raise ValueError("OpenAI-compatible attempt model belongs to another provider")

        body = deepcopy(dict(payload))
        body["model"] = model.upstream_id
        body["stream"] = stream
        if stream:
            stream_options = body.get("stream_options")
            if stream_options is None:
                normalized_stream_options: dict[str, Any] = {}
            elif isinstance(stream_options, Mapping):
                normalized_stream_options = deepcopy(dict(stream_options))
            else:
                raise RequestOptionError(
                    "OpenAI-compatible Chat requires stream_options to be an object",
                    provider=self.provider.name,
                    model=model.upstream_id,
                    parameter="stream_options",
                )
            normalized_stream_options["include_usage"] = True
            body["stream_options"] = normalized_stream_options

        return PreparedAttempt(
            binding_id=binding_id,
            protocol=protocol,
            model=model,
            url=f"{self.provider.base_url}/chat/completions",
            payload=body,
            headers=self.provider._get_headers(),
            stream=stream,
            _payload_owned=True,
        )


class OpenAICompatibleHttpExecutor(SharedHttpExecutor):
    """Raw JSON/SSE executor shared by official and custom OpenAI providers."""

    def __init__(self, provider: OpenAIChatProvider) -> None:
        self.provider = provider
        super().__init__(client_pool=provider._http_client_pool)

    def _skip_raw_sse_data(self, data: str, attempt: PreparedAttempt) -> bool:
        del attempt
        return data == "[DONE]"

    def _validate_attempt(self, attempt: PreparedAttempt, *, stream: bool) -> None:
        if attempt.binding_id != OPENAI_COMPATIBLE_CHAT_BINDING:
            raise ValueError("OpenAI-compatible executor received an unknown binding")
        if attempt.protocol is not WireProtocol.OPENAI_CHAT:
            raise ValueError("OpenAI-compatible executor requires the Chat wire protocol")
        if attempt.model.provider != self.provider.name:
            raise ValueError("OpenAI-compatible executor received another provider's model")
        if attempt.method != "POST":
            raise ValueError("OpenAI-compatible Chat binding requires POST")
        if attempt.stream is not stream:
            raise ValueError("OpenAI-compatible attempt stream mode does not match execution")

    def _raise_status(
        self,
        error: httpx.HTTPStatusError,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        self.provider._raise_http_status_error(
            self.provider._error_label(),
            error,
            self.provider._logger,
            stream=stream,
            include_body=stream,
            provider=self.provider.name,
            model=attempt.model.upstream_id,
        )

    def _raise_timeout(
        self,
        error: httpx.TimeoutException,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        self.provider._raise_timeout_error(
            self.provider._error_label(),
            error,
            self.provider._logger,
            stream=stream,
            provider=self.provider.name,
            model=attempt.model.upstream_id,
        )

    def _raise_http_error(
        self,
        error: httpx.HTTPError,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        self.provider._raise_http_error(
            self.provider._error_label(),
            error,
            self.provider._logger,
            stream=stream,
            provider=self.provider.name,
            model=attempt.model.upstream_id,
        )

    def _raise_protocol_error(self, error: Exception, attempt: PreparedAttempt) -> NoReturn:
        self.provider._raise_protocol_error(
            self.provider.name,
            attempt.model.upstream_id,
            error,
        )
