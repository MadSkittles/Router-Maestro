"""Pure payload and response codecs for Copilot Chat Completions."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from router_maestro.providers.base import (
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
)
from router_maestro.providers.tool_parsing import recover_tool_calls_from_content
from router_maestro.utils import get_logger

logger = get_logger("providers.copilot.chat_codec")


class CopilotChatCodec:
    """Encode Chat requests and decode Chat responses without transport state."""

    @staticmethod
    def sanitize_surrogates(text: str) -> str:
        return text.encode("utf-8", errors="replace").decode("utf-8")

    def sanitize_content(self, content: str | list) -> str | list:
        if isinstance(content, str):
            return self.sanitize_surrogates(content)
        if isinstance(content, list):
            result = []
            for part in content:
                is_text = (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                )
                if is_text:
                    result.append({**part, "text": self.sanitize_surrogates(part["text"])})
                else:
                    result.append(part)
            return result
        return content

    def build_messages_payload(self, request: ChatRequest) -> tuple[list[dict], bool]:
        messages = []
        has_images = False
        for message in request.messages:
            item: dict = {
                "role": message.role,
                "content": self.sanitize_content(message.content),
            }
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = message.tool_calls
            if message.refusal is not None:
                item["refusal"] = message.refusal
            messages.append(item)
            if isinstance(message.content, list) and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in message.content
            ):
                has_images = True
        return messages, has_images

    def build_payload(
        self,
        request: ChatRequest,
        *,
        stream: bool,
        validate_extensions: Callable[[ChatRequest], None],
        apply_reasoning: Callable[..., None],
        catalog_effort_values: list[str] | None,
        provider_name: str,
    ) -> dict:
        validate_extensions(request)
        messages, _has_images = self.build_messages_payload(request)
        payload: dict = {"model": request.model, "messages": messages, "stream": stream}
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
                self.reject_option(request, parameter, provider_name=provider_name)
        if request.stop is not None and request.stop_sequences is not None:
            self.reject_option(request, "stop", provider_name=provider_name)
        options = {
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop if request.stop is not None else request.stop_sequences,
            "user": request.user,
            "metadata": request.metadata,
            "service_tier": request.service_tier,
        }
        for key, value in options.items():
            if value is not None:
                payload[key] = value
        apply_reasoning(
            payload,
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            catalog_effort_values=catalog_effort_values,
        )
        return payload

    @staticmethod
    def reject_option(
        request: ChatRequest,
        parameter: str,
        *,
        provider_name: str,
    ) -> None:
        raise RequestOptionError(
            f"GitHub Copilot Chat does not support request option '{parameter}'",
            provider=provider_name,
            model=request.model,
            parameter=parameter,
        )

    @staticmethod
    def thinking_requested(request: ChatRequest) -> bool:
        return request.thinking_type in ("enabled", "adaptive")

    @staticmethod
    def extract_reasoning(part: dict | None) -> tuple[str, str | None]:
        if not part:
            return "", None
        text = ""
        for key in ("reasoning_text", "cot_summary", "thinking"):
            value = part.get(key)
            if isinstance(value, str) and value:
                text = value
                break
            if isinstance(value, dict):
                inner = value.get("text") or value.get("content")
                if isinstance(inner, str) and inner:
                    text = inner
                    break
        signature = None
        for key in ("reasoning_opaque", "cot_id", "signature"):
            value = part.get(key)
            if isinstance(value, str) and value:
                signature = value
                break
        return text, signature

    @staticmethod
    def _is_provisional_zero_usage(usage: dict[str, Any] | None, choices: list[Any] | None) -> bool:
        """Recognize Copilot's all-zero usage attached to non-terminal reasoning deltas."""
        if not usage or not choices:
            return False
        if any(choice.get("finish_reason") is not None for choice in choices):
            return False
        return all(
            usage.get(field) == 0
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        )

    def decode_response(
        self,
        data: dict[str, Any],
        request: ChatRequest,
        *,
        model: str,
        usage: dict[str, Any] | None,
        choices: list[Any],
        reasoning_capable: bool,
        validated_optional_string: Callable[[dict, str], str | None],
    ) -> ChatResponse:
        completion_tokens = (usage or {}).get("completion_tokens", 0)
        if not choices:
            max_tokens = request.max_tokens
            has_positive_output_cap = (
                isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0
            )
            thinking_budget = request.thinking_budget
            exhausted_explicit_thinking_budget = (
                request.reasoning_effort is None
                and request.thinking_type == "enabled"
                and isinstance(thinking_budget, int)
                and not isinstance(thinking_budget, bool)
                and thinking_budget > 0
                and completion_tokens == thinking_budget
            )
            if (
                reasoning_capable
                and self.thinking_requested(request)
                and has_positive_output_cap
                and (completion_tokens >= max_tokens or exhausted_explicit_thinking_budget)
            ):
                logger.warning(
                    "Copilot returned empty choices after exhausting a requested cap: "
                    "model=%s completion_tokens=%d max_tokens=%d thinking_budget=%s",
                    request.model,
                    completion_tokens,
                    max_tokens,
                    thinking_budget,
                )
                return ChatResponse(
                    content="",
                    model=model,
                    finish_reason="length",
                    usage=usage or None,
                    tool_calls=None,
                )
            raise ValueError("chat response choices must be a non-empty list")

        content = None
        refusal = None
        tool_calls = []
        finish_reason = "stop"
        thinking_text = ""
        thinking_signature: str | None = None
        parsed_choices: list[tuple[dict, dict, str | None]] = []
        for choice in choices:
            if not isinstance(choice, dict):
                raise TypeError("chat response choice must be an object")
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("chat response message must be an object")
            message_content = message.get("content")
            message_refusal = message.get("refusal")
            message_tool_calls = message.get("tool_calls")
            if message_content is not None and not isinstance(message_content, str):
                raise TypeError("chat response content must be a string or null")
            if "refusal" in message and (
                not isinstance(message_refusal, str) or not message_refusal
            ):
                raise TypeError("chat response refusal must be a non-empty string")
            if message_tool_calls is not None and not isinstance(message_tool_calls, list):
                raise TypeError("chat response tool_calls must be a list or null")
            for tool_call in message_tool_calls or []:
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
            choice_finish_reason = validated_optional_string(choice, "finish_reason")
            parsed_choices.append((choice, message, choice_finish_reason))

        for _choice, message, choice_finish_reason in parsed_choices:
            if content is None and message.get("content"):
                content = message["content"]
            if refusal is None and message.get("refusal"):
                refusal = message["refusal"]
            if message.get("tool_calls"):
                tool_calls.extend(message["tool_calls"])
            if choice_finish_reason:
                finish_reason = choice_finish_reason
            if self.thinking_requested(request):
                reasoning_text, signature = self.extract_reasoning(message)
                thinking_text += reasoning_text
                if signature and thinking_signature is None:
                    thinking_signature = signature

        if not any((content, refusal, tool_calls, thinking_text, thinking_signature)):
            raise ValueError("chat response contains no deliverable output")
        if len(choices) > 1:
            logger.info(
                "Copilot returned %d choices: content=%s, tool_calls=%d, finish_reason=%s",
                len(choices),
                len(content) if content else 0,
                len(tool_calls),
                finish_reason,
            )
        normalized_tool_calls = tool_calls or None
        content, normalized_tool_calls = recover_tool_calls_from_content(
            content,
            normalized_tool_calls,
            finish_reason,
        )
        if normalized_tool_calls and finish_reason in (None, "stop"):
            finish_reason = "tool_calls"
        return ChatResponse(
            content=content,
            model=model,
            refusal=refusal,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=normalized_tool_calls,
            thinking=thinking_text or None,
            thinking_signature=thinking_signature,
        )

    async def decode_stream(
        self,
        lines: AsyncIterator[str],
        request: ChatRequest,
        *,
        provider_name: str,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Reduce Copilot Chat SSE data lines into stable stream chunks."""
        stream_finished = False
        emitted_tool_call = False
        observed_usage: dict | None = None
        async for line in lines:
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                if not isinstance(data, dict):
                    raise TypeError("stream event must be an object")
                usage = BaseProvider._validated_token_usage(
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
                for choice in choices or []:
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
                    for tool_call in tool_calls or []:
                        if not isinstance(tool_call, dict):
                            raise TypeError("stream tool call delta must be object")
                        index = tool_call.get("index")
                        if index is not None and (
                            not isinstance(index, int) or isinstance(index, bool)
                        ):
                            raise TypeError("stream tool call index must be integer")
                        for field in ("id", "type"):
                            value = tool_call.get(field)
                            if value is not None and not isinstance(value, str):
                                raise TypeError(f"stream tool call {field} must be string")
                        function = tool_call.get("function")
                        if function is not None:
                            if not isinstance(function, dict):
                                raise TypeError("stream tool call function must be object")
                            for field in ("name", "arguments"):
                                value = function.get(field)
                                if value is not None and not isinstance(value, str):
                                    raise TypeError(f"stream tool function {field} must be string")
                    if finish_reason is not None and not isinstance(finish_reason, str):
                        raise TypeError("stream finish_reason must be a string or null")
            except (json.JSONDecodeError, TypeError) as error:
                raise ProviderError(
                    f"{provider_name} returned a malformed upstream response",
                    status_code=502,
                    retryable=True,
                    kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                    upstream_status_code=200,
                    provider=provider_name,
                    model=request.model,
                    cause=error,
                ) from error

            if observed_usage is None and self._is_provisional_zero_usage(usage, choices):
                usage = None

            usage_to_emit = None
            if usage:
                if observed_usage is not None and usage != observed_usage:
                    error = TypeError("conflicting usage in one chat stream")
                    raise ProviderError(
                        f"{provider_name} returned a malformed upstream response",
                        status_code=502,
                        retryable=True,
                        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                        upstream_status_code=200,
                        provider=provider_name,
                        model=request.model,
                        cause=error,
                    ) from error
                if observed_usage is None:
                    observed_usage = usage
                    usage_to_emit = usage

            if stream_finished:
                if choices != [] or not usage:
                    error = TypeError("non-usage event followed a terminal chat choice")
                    raise ProviderError(
                        f"{provider_name} returned a malformed upstream response",
                        status_code=502,
                        retryable=True,
                        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                        upstream_status_code=200,
                        provider=provider_name,
                        model=request.model,
                        cause=error,
                    ) from error
                if usage_to_emit is not None:
                    yield ChatStreamChunk(content="", finish_reason=None, usage=usage_to_emit)
                continue

            if choices:
                terminal_chunks = []
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    refusal = delta.get("refusal")
                    finish_reason = choice.get("finish_reason")
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        emitted_tool_call = True
                    if finish_reason == "stop" and emitted_tool_call:
                        finish_reason = "tool_calls"
                    if self.thinking_requested(request):
                        thinking_text, thinking_signature = self.extract_reasoning(delta)
                    else:
                        thinking_text, thinking_signature = "", None
                    if any(
                        (
                            content,
                            refusal,
                            finish_reason,
                            usage_to_emit,
                            tool_calls,
                            thinking_text,
                            thinking_signature,
                        )
                    ):
                        chunk = ChatStreamChunk(
                            content=content,
                            refusal=refusal or None,
                            finish_reason=finish_reason,
                            usage=usage_to_emit,
                            tool_calls=tool_calls,
                            thinking=thinking_text or None,
                            thinking_signature=thinking_signature,
                        )
                        usage_to_emit = None
                        if finish_reason:
                            terminal_chunks.append(chunk)
                        else:
                            yield chunk
                    if finish_reason:
                        stream_finished = True
                for chunk in terminal_chunks:
                    yield chunk
            elif usage_to_emit:
                yield ChatStreamChunk(content="", finish_reason=None, usage=usage_to_emit)
