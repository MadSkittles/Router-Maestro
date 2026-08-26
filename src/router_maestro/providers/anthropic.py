"""Anthropic provider implementation."""

import json
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any, NoReturn, cast

import httpx

from router_maestro.auth import ApiKeyCredential, AuthManager, AuthType
from router_maestro.pipeline.beta_strip import strip_beta_tokens
from router_maestro.protocols import WireProtocol
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
from router_maestro.providers.bindings import (
    AttemptRequestContext,
    EndpointBinding,
    PreparedAttempt,
)
from router_maestro.providers.http_executor import ProviderHttpClientPool, SharedHttpExecutor
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef
from router_maestro.utils import get_logger
from router_maestro.utils.context_window import normalize_thinking_budget

logger = get_logger("providers.anthropic")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1"
ANTHROPIC_MESSAGES_BINDING = "anthropic-messages"
_ANTHROPIC_STREAM_KEEPALIVE_TYPES = frozenset({"ping"})


def _request_audit():
    from router_maestro.runtime import get_current_request_context

    context = get_current_request_context()
    return context.audit if context is not None else None


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider."""

    name = "anthropic"

    def bindings(self) -> tuple[EndpointBinding, ...]:
        """Expose Anthropic Messages as a protocol-native raw binding."""
        bindings = getattr(self, "_generation_bindings", None)
        if bindings is not None:
            return bindings

        binding = EndpointBinding(
            id=ANTHROPIC_MESSAGES_BINDING,
            protocol=WireProtocol.ANTHROPIC_MESSAGES,
            capabilities=ProviderCapabilities(operations=frozenset({Operation.NATIVE_ANTHROPIC})),
            dialect=AnthropicProviderDialect(self),
            executor=AnthropicHttpExecutor(self),
        )
        bindings = (binding,)
        self._generation_bindings = bindings
        return bindings

    def __init__(self, base_url: str = ANTHROPIC_API_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_manager = AuthManager()
        self._http_client_pool = ProviderHttpClientPool(lambda: httpx.AsyncClient())

    async def close(self) -> None:
        """Close the provider-owned reusable HTTP client."""
        await self._http_client_pool.close()

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
        return cast(ApiKeyCredential, cred).key

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
        if request.output_format is not None:
            # ``format`` and ``effort`` are independent siblings on the Anthropic
            # wire; carrying both keeps a structured-output schema from being
            # dropped when an effort tier is also requested.
            payload.setdefault("output_config", {})["format"] = request.output_format

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
        url = f"{self.base_url}/messages"
        headers = self._get_headers()
        audit = _request_audit()
        if audit is not None:
            audit.record_upstream("POST", url, headers, payload)

        logger.debug("Anthropic chat completion: model=%s", request.model)
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
        url = f"{self.base_url}/messages"
        headers = self._get_headers()
        audit = _request_audit()
        if audit is not None:
            audit.record_upstream("POST", url, headers, payload)

        logger.debug("Anthropic streaming chat: model=%s", request.model)
        decoder = AnthropicStreamDecoder(
            include_reasoning=request.thinking_type in {"enabled", "adaptive"}
        )
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

    async def messages_completion(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
    ) -> ChatResponse:
        """Execute an already-encoded Anthropic Messages body without request IR."""
        body = deepcopy(dict(payload))
        body["model"] = model
        body["stream"] = False
        include_reasoning = isinstance(body.get("thinking"), dict) and body["thinking"].get(
            "type"
        ) in {"enabled", "adaptive"}
        url = f"{self.base_url}/messages"
        headers = self._get_headers()
        audit = _request_audit()
        if audit is not None:
            audit.record_upstream("POST", url, headers, body)

        async with self._http_client_pool.lease() as client:
            try:
                response = await client.post(
                    url,
                    json=body,
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
                    return decode_message_response(
                        response.json(),
                        fallback_model=model,
                        include_reasoning=include_reasoning,
                    )
                except (json.JSONDecodeError, AnthropicCodecError) as error:
                    self._raise_protocol_error(self.name, model, error)
            except httpx.HTTPStatusError as error:
                self._raise_http_status_error(
                    "Anthropic",
                    error,
                    logger,
                    include_body=True,
                    provider=self.name,
                    model=model,
                )
            except httpx.TimeoutException as error:
                self._raise_timeout_error(
                    "Anthropic",
                    error,
                    logger,
                    provider=self.name,
                    model=model,
                )
            except httpx.HTTPError as error:
                self._raise_http_error(
                    "Anthropic",
                    error,
                    logger,
                    provider=self.name,
                    model=model,
                )

    async def messages_completion_stream(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Stream an already-encoded Anthropic Messages body without request IR."""
        body = deepcopy(dict(payload))
        body["model"] = model
        body["stream"] = True
        include_reasoning = isinstance(body.get("thinking"), dict) and body["thinking"].get(
            "type"
        ) in {"enabled", "adaptive"}
        decoder = AnthropicStreamDecoder(include_reasoning=include_reasoning)
        url = f"{self.base_url}/messages"
        headers = self._get_headers()
        audit = _request_audit()
        if audit is not None:
            audit.record_upstream("POST", url, headers, body)

        async with self._http_client_pool.lease() as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    json=body,
                    headers=headers,
                    timeout=TIMEOUT_STREAMING,
                ) as response:
                    if audit is not None:
                        audit.record_upstream_response(
                            response.status_code,
                            dict(response.headers),
                            stream_summary="stream opened",
                        )
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw_data = line[6:]
                        if not raw_data:
                            continue
                        try:
                            chunks = decoder.decode_event(json.loads(raw_data))
                        except (json.JSONDecodeError, AnthropicCodecError) as error:
                            self._raise_protocol_error(self.name, model, error)
                        for chunk in chunks:
                            yield chunk
                    try:
                        decoder.finalize()
                    except AnthropicCodecError as error:
                        self._raise_protocol_error(self.name, model, error)
            except httpx.HTTPStatusError as error:
                self._raise_http_status_error(
                    "Anthropic",
                    error,
                    logger,
                    stream=True,
                    include_body=True,
                    provider=self.name,
                    model=model,
                )
            except httpx.TimeoutException as error:
                self._raise_timeout_error(
                    "Anthropic",
                    error,
                    logger,
                    stream=True,
                    provider=self.name,
                    model=model,
                )
            except httpx.HTTPError as error:
                self._raise_http_error(
                    "Anthropic",
                    error,
                    logger,
                    stream=True,
                    provider=self.name,
                    model=model,
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


class AnthropicProviderDialect:
    """Copy-on-write preparation for Anthropic's native Messages endpoint."""

    id = "anthropic"

    def __init__(self, provider: AnthropicProvider) -> None:
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
        if binding_id != ANTHROPIC_MESSAGES_BINDING:
            raise ValueError(f"Unknown Anthropic binding {binding_id!r}")
        if protocol is not WireProtocol.ANTHROPIC_MESSAGES:
            raise ValueError("Anthropic binding requires the Messages wire protocol")
        if model.provider != self.provider.name:
            raise ValueError("Anthropic attempt model belongs to another provider")

        body = deepcopy(dict(payload))
        body["model"] = model.upstream_id
        body["stream"] = stream
        headers = self.provider._get_headers()
        anthropic_beta = request_context.header("anthropic-beta")
        if anthropic_beta is not None:
            from router_maestro.runtime import get_current_request_context

            runtime_context = get_current_request_context()
            stripped = strip_beta_tokens(
                anthropic_beta,
                runtime_context.config.beta_strip if runtime_context is not None else [],
            )
            if stripped is not None:
                headers["anthropic-beta"] = stripped
        return PreparedAttempt(
            binding_id=binding_id,
            protocol=protocol,
            model=model,
            url=f"{self.provider.base_url}/messages",
            payload=body,
            headers=headers,
            stream=stream,
            _payload_owned=True,
        )


class AnthropicHttpExecutor(SharedHttpExecutor):
    """Raw JSON/SSE executor for Anthropic's native Messages binding."""

    def __init__(self, provider: AnthropicProvider) -> None:
        self.provider = provider
        super().__init__(client_pool=provider._http_client_pool)

    def _skip_sse_frame(
        self,
        frame: Mapping[str, Any],
        attempt: PreparedAttempt,
    ) -> bool:
        del attempt
        return frame.get("type") in _ANTHROPIC_STREAM_KEEPALIVE_TYPES

    def _validate_attempt(self, attempt: PreparedAttempt, *, stream: bool) -> None:
        if attempt.binding_id != ANTHROPIC_MESSAGES_BINDING:
            raise ValueError("Anthropic executor received an unknown binding")
        if attempt.protocol is not WireProtocol.ANTHROPIC_MESSAGES:
            raise ValueError("Anthropic executor requires the Messages wire protocol")
        if attempt.model.provider != self.provider.name:
            raise ValueError("Anthropic attempt model belongs to another provider")
        if attempt.method != "POST":
            raise ValueError("Anthropic generation bindings require POST")
        if attempt.stream is not stream:
            mode = "streaming" if stream else "non-streaming"
            raise ValueError(f"Anthropic executor received the wrong {mode} attempt mode")

    def _raise_status(
        self,
        error: httpx.HTTPStatusError,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        self.provider._raise_http_status_error(
            "Anthropic",
            error,
            logger,
            stream=stream,
            include_body=True,
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
            "Anthropic",
            error,
            logger,
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
            "Anthropic",
            error,
            logger,
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
