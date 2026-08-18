"""Wire-to-legacy provider request bridges used during incremental migration.

These helpers do not construct semantic IR.  They exist only until every
first-party provider binding executes raw protocol payloads directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from router_maestro.providers.base import (
    ChatRequest,
    Message,
    RequestOptionError,
    ResponsesRequest,
)
from router_maestro.server.dispatcher import LegacyProviderExecutionAdapter


def chat_request_from_wire(
    payload: Mapping[str, Any],
    *,
    model: str,
    stream: bool,
) -> ChatRequest:
    """Project a validated Chat wire body into the existing provider DTO."""
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise RequestOptionError("messages must be an array", parameter="messages")
    messages = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            raise RequestOptionError(
                "message must be an object",
                parameter=f"messages[{index}]",
            )
        role = raw.get("role")
        if not isinstance(role, str):
            raise RequestOptionError(
                "message role must be a string",
                parameter=f"messages[{index}].role",
            )
        messages.append(
            Message(
                role=role,
                content=raw.get("content"),
                tool_call_id=raw.get("tool_call_id"),
                tool_calls=raw.get("tool_calls"),
                refusal=raw.get("refusal"),
            )
        )

    parallel = payload.get("parallel_tool_calls")
    if parallel is not None:
        raise RequestOptionError(
            "legacy Chat provider binding cannot preserve parallel_tool_calls",
            parameter="parallel_tool_calls",
        )
    thinking = payload.get("thinking")
    thinking_type = None
    thinking_budget = None
    if thinking is not None:
        if not isinstance(thinking, Mapping):
            raise RequestOptionError("thinking must be an object", parameter="thinking")
        thinking_type = thinking.get("type")
        thinking_budget = thinking.get("budget_tokens")

    known = {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "stream",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "thinking",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "user",
        "metadata",
        "service_tier",
        "response_format",
    }
    extensions = _extension_fields(payload, known)
    if isinstance(thinking, Mapping) and set(thinking) - {"type", "budget_tokens"}:
        # The legacy DTO models only these two members. Retain the full wire
        # object as an extension so an identity-capable provider can replay all
        # explicitly supplied fields instead of silently discarding siblings.
        extensions["thinking"] = deepcopy(dict(thinking))
    return ChatRequest(
        model=model,
        messages=messages,
        temperature=payload.get("temperature"),
        max_tokens=payload.get("max_tokens"),
        stream=stream,
        tools=payload.get("tools"),
        tool_choice=payload.get("tool_choice"),
        thinking_budget=thinking_budget,
        thinking_type=thinking_type,
        reasoning_effort=payload.get("reasoning_effort"),
        top_p=payload.get("top_p"),
        frequency_penalty=payload.get("frequency_penalty"),
        presence_penalty=payload.get("presence_penalty"),
        stop=payload.get("stop"),
        user=payload.get("user"),
        metadata=payload.get("metadata"),
        service_tier=payload.get("service_tier"),
        output_format=payload.get("response_format"),
        provider_extensions=extensions,
    )


def responses_request_from_wire(
    payload: Mapping[str, Any],
    *,
    model: str,
    stream: bool,
) -> ResponsesRequest:
    """Project a validated Responses body without normalizing its input items."""
    if "input" not in payload:
        raise RequestOptionError("input is required", parameter="input")
    reasoning = payload.get("reasoning")
    reasoning_effort = None
    if reasoning is not None:
        if not isinstance(reasoning, Mapping):
            raise RequestOptionError("reasoning must be an object", parameter="reasoning")
        effort = reasoning.get("effort")
        if effort is not None and not isinstance(effort, str):
            raise RequestOptionError(
                "reasoning.effort must be a string",
                parameter="reasoning.effort",
            )
        reasoning_effort = effort

    parallel_tool_calls = payload.get("parallel_tool_calls")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise RequestOptionError(
            "parallel_tool_calls must be a boolean",
            parameter="parallel_tool_calls",
        )

    known = {
        "model",
        "input",
        "stream",
        "instructions",
        "temperature",
        "max_output_tokens",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "top_p",
        "metadata",
        "service_tier",
    }
    extensions = _extension_fields(payload, known)
    if isinstance(reasoning, Mapping) and set(reasoning) - {"effort"}:
        # ResponsesRequest currently exposes only reasoning_effort. Preserve
        # summary and future reasoning options for same-protocol execution.
        extensions["reasoning"] = deepcopy(dict(reasoning))
    return ResponsesRequest(
        model=model,
        input=payload["input"],
        stream=stream,
        instructions=payload.get("instructions"),
        temperature=payload.get("temperature"),
        max_output_tokens=payload.get("max_output_tokens"),
        tools=payload.get("tools"),
        tool_choice=payload.get("tool_choice"),
        parallel_tool_calls=parallel_tool_calls,
        reasoning_effort=reasoning_effort,
        top_p=payload.get("top_p"),
        metadata=payload.get("metadata"),
        service_tier=payload.get("service_tier"),
        provider_extensions=extensions,
    )


def _extension_fields(
    payload: Mapping[str, Any],
    known: set[str],
) -> dict[str, Any]:
    """Snapshot unmodeled top-level wire fields for provider-specific replay."""
    return {key: deepcopy(value) for key, value in payload.items() if key not in known}


def legacy_execution_adapter() -> LegacyProviderExecutionAdapter:
    """Return the compatibility executor used by the shared dispatcher."""
    return LegacyProviderExecutionAdapter(
        chat_request_factory=chat_request_from_wire,
        responses_request_factory=responses_request_from_wire,
    )


__all__ = [
    "chat_request_from_wire",
    "legacy_execution_adapter",
    "responses_request_from_wire",
]
