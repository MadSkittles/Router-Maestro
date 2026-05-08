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
)
from router_maestro.providers.tool_parsing import recover_tool_calls_from_content
from router_maestro.utils.reasoning import budget_to_effort, downgrade_for_upstream


class OpenAIChatProvider(BaseProvider, ABC):
    """Shared OpenAI-compatible chat behavior."""

    def __init__(self, base_url: str, logger: Logger) -> None:
        self.base_url = base_url.rstrip("/")
        self._logger = logger

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Return headers for the API request."""

    def _get_payload_extra(self, request: ChatRequest) -> dict:
        """Return extra payload fields for the request."""
        return {}

    def _error_label(self) -> str:
        return self.name

    def _build_payload(self, request: ChatRequest, stream: bool) -> dict:
        messages = []
        for m in request.messages:
            msg: dict = {"role": m.role, "content": m.content}
            if m.tool_call_id is not None:
                msg["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                msg["tool_calls"] = m.tool_calls
            messages.append(msg)

        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        # Forward OpenAI-style reasoning_effort. Fall back to deriving it from
        # thinking_budget when only the Anthropic-style budget is set. xhigh is
        # downgraded to "high" since vanilla OpenAI/Copilot reject it.
        effort = request.reasoning_effort or budget_to_effort(request.thinking_budget)
        upstream_effort = downgrade_for_upstream(effort)
        if upstream_effort is not None:
            if effort == "xhigh":
                self._logger.warning(
                    "%s does not accept reasoning_effort=xhigh; downgrading to high",
                    self._error_label(),
                )
            payload["reasoning_effort"] = upstream_effort

        payload.update(self._get_payload_extra(request))
        return payload

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
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                finish_reason = data["choices"][0].get("finish_reason", "stop")

                content, tool_calls = recover_tool_calls_from_content(
                    content, tool_calls, finish_reason
                )

                return ChatResponse(
                    content=content,
                    model=data.get("model", request.model),
                    finish_reason=finish_reason,
                    usage=data.get("usage"),
                    tool_calls=tool_calls,
                )
            except httpx.HTTPStatusError as e:
                self._raise_http_status_error(label, e, self._logger)
            except httpx.TimeoutException as e:
                self._raise_timeout_error(label, e, self._logger)
            except httpx.HTTPError as e:
                self._raise_http_error(label, e, self._logger)

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

                        data = json.loads(data_str)

                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = data["choices"][0].get("finish_reason")
                            usage = data.get("usage")
                            tool_calls = delta.get("tool_calls")

                            if content or finish_reason or usage or tool_calls:
                                yield ChatStreamChunk(
                                    content=content,
                                    finish_reason=finish_reason,
                                    usage=usage,
                                    tool_calls=tool_calls,
                                )
            except httpx.HTTPStatusError as e:
                self._raise_http_status_error(
                    label, e, self._logger, stream=True, include_body=True
                )
            except httpx.TimeoutException as e:
                self._raise_timeout_error(label, e, self._logger, stream=True)
            except httpx.HTTPError as e:
                self._raise_http_error(label, e, self._logger, stream=True)
