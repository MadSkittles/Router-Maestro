"""Experimental: route ChatRequest through Copilot's /responses endpoint.

Background: Copilot exposes two completion endpoints — ``/chat/completions``
(OpenAI Chat) and ``/responses`` (OpenAI Responses). Probe results show that
only the GPT-5 family supports ``/responses`` on Copilot; Claude and Gemini
models reject it. This module gates an opt-in path where the Anthropic and
Gemini entry routes ask the Copilot provider to fulfil their request via
``/responses`` when (a) the env flag is on and (b) the resolved model is
eligible.

The experiment is controlled by ``ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API``
(values: ``1``/``true``/``yes``/``on``, case-insensitive). Default off.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy

from router_maestro.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponseStatus,
    finish_reason_for_outcome,
)

ENV_FLAG = "ROUTER_MAESTRO_EXPERIMENTAL_RESPONSES_API"

# Models confirmed by direct probing of api.githubcopilot.com/responses to
# accept the Responses API. Anything else returns 400 unsupported_api_for_model.
# Match by suffix after stripping optional ``provider/`` prefix.
RESPONSES_ELIGIBLE_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5-mini",
        "mai-code-1-flash-picker",
    }
)


def is_experimental_responses_enabled() -> bool:
    """Return True if the experimental env flag is set to a truthy value."""
    raw = os.environ.get(ENV_FLAG, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _bare_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def is_model_responses_eligible(model: str) -> bool:
    """Whether the upstream serves this model via /responses."""
    return _bare_model(model) in RESPONSES_ELIGIBLE_MODELS


def should_use_responses_for_chat(
    request: ChatRequest,
    provider_name: str,
    *,
    responses_supported: bool | None = None,
) -> bool:
    """Decide whether a ChatRequest should be fulfilled via /responses.

    Requires:
    - the experimental env flag (kill-switch enforced here, not just at the
      entry routes, so any caller setting ``use_responses_api=True`` is still
      gated by ops);
    - the per-request opt-in flag;
    - the Copilot provider (others have no /responses endpoint we target);
    - an eligible model.
    """
    if not is_experimental_responses_enabled():
        return False
    if not request.use_responses_api:
        return False
    if provider_name != "github-copilot":
        return False
    if responses_supported is not None:
        return responses_supported
    return is_model_responses_eligible(request.model)


# ---------------------------------------------------------------------------
# ChatRequest -> ResponsesRequest
# ---------------------------------------------------------------------------


def _content_to_text(content: str | list) -> str:
    """Flatten a Message.content to plain text (for system/tool messages)."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif "text" in block and isinstance(block["text"], str):
                parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "\n\n".join(p for p in parts if p)


def _content_to_responses_blocks(content: str | list, *, role: str = "user") -> list[dict]:
    """Convert OpenAI Chat content to Responses API content blocks.

    Handles:
    - str → [{"type": "input_text"/"output_text", "text": ...}]
    - {"type": "text", ...} → {"type": "input_text"/"output_text", "text": ...}
    - {"type": "image_url", "image_url": {"url": ...}} → {"type": "input_image", "image_url": ...}
    - {"type": "image_url", "image_url": "..."} (flat) → {"type": "input_image", "image_url": ...}
    - Unknown blocks → passed through as-is

    Assistant messages use "output_text"; user/other messages use "input_text".
    """
    text_type = "output_text" if role == "assistant" else "input_text"

    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content else []

    blocks: list[dict] = []
    for item in content:
        if isinstance(item, str):
            if item:
                blocks.append({"type": text_type, "text": item})
            continue
        if not isinstance(item, dict):
            blocks.append(item)
            continue

        btype = item.get("type")
        if btype == "text":
            text = item.get("text", "")
            if text:
                blocks.append({"type": text_type, "text": text})
        elif btype == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                url = image_url or ""
            blocks.append({"type": "input_image", "image_url": url})
        elif btype is None and isinstance(item.get("text"), str):
            text = item["text"]
            if text:
                blocks.append({"type": text_type, "text": text})
        else:
            blocks.append(item)
    return blocks


def _chat_tools_to_responses_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI Chat tool definitions to Responses tool definitions.

    Chat shape:    ``{"type": "function", "function": {"name", "description", "parameters"}}``
    Responses shape: ``{"type": "function", "name", "description", "parameters"}``
    """
    if not tools:
        return None
    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            entry: dict = {"type": "function", "name": fn.get("name", "")}
            if fn.get("description") is not None:
                entry["description"] = fn["description"]
            if fn.get("parameters") is not None:
                entry["parameters"] = fn["parameters"]
            if fn.get("strict") is not None:
                entry["strict"] = fn["strict"]
            out.append(entry)
        else:
            # Already in Responses shape or unknown — pass through.
            out.append(tool)
    return out or None


def _chat_tool_choice_to_responses(tool_choice: str | dict | None) -> str | dict | None:
    """Translate Chat tool_choice to Responses tool_choice."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
            return {"type": "function", "name": tool_choice["function"].get("name", "")}
        return tool_choice
    return None


def _messages_to_responses_input(
    messages: list[Message],
) -> tuple[str | None, list[dict]]:
    """Convert a list of OpenAI Chat Messages into (instructions, input_items).

    System messages collapse into the ``instructions`` field. Assistant tool_calls
    become ``function_call`` items; tool messages become ``function_call_output``
    items. User/assistant content (including multimodal) is converted to Responses
    API content blocks via ``_content_to_responses_blocks``.
    """
    instructions_parts: list[str] = []
    items: list[dict] = []

    for msg in messages:
        role = msg.role
        if role == "system":
            text = _content_to_text(msg.content)
            if text:
                instructions_parts.append(text)
            continue

        if role == "tool":
            output = msg.content if isinstance(msg.content, str) else _content_to_text(msg.content)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": output if isinstance(output, str) else json.dumps(output),
                }
            )
            continue

        if role == "assistant":
            content_blocks = (
                _content_to_responses_blocks(msg.content, role="assistant") if msg.content else []
            )
            if msg.refusal:
                content_blocks.append({"type": "refusal", "refusal": msg.refusal})
            if content_blocks:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )
            for tc in msg.tool_calls or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": args,
                    }
                )
            continue

        # user (or unknown role treated as user)
        content_blocks = _content_to_responses_blocks(msg.content)
        items.append(
            {
                "type": "message",
                "role": "user",
                "content": content_blocks,
            }
        )

    instructions = "\n\n".join(p for p in instructions_parts if p) or None
    return instructions, items


def chat_request_to_responses_request(request: ChatRequest) -> ResponsesRequest:
    """Convert a ChatRequest into a ResponsesRequest preserving reasoning effort.

    ``thinking_budget`` is left to the provider's reasoning resolver — we only
    forward the already-resolved ``reasoning_effort`` (or pass ``None`` and let
    the Copilot provider derive it from the budget).
    """
    instructions, input_items = _messages_to_responses_input(request.messages)
    tools = _chat_tools_to_responses_tools(request.tools)
    tool_choice = _chat_tool_choice_to_responses(request.tool_choice)

    # Derive reasoning_effort from thinking_budget if the entry route only
    # provided a budget (Anthropic Messages API uses budget_tokens).
    effort = request.reasoning_effort
    if effort is None and request.thinking_budget is not None:
        from router_maestro.utils.reasoning import budget_to_effort

        effort = budget_to_effort(request.thinking_budget)
        if request.thinking_budget > 0 and effort is None:
            raise RequestOptionError(
                "Responses API has no reasoning tier at or below the requested budget",
                model=request.model,
                parameter="thinking_budget",
            )

    unsupported = {
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "stop": request.stop,
        "user": request.user,
        "top_k": request.top_k,
        "stop_sequences": request.stop_sequences,
        "candidate_count": request.candidate_count,
        "response_mime_type": request.response_mime_type,
    }
    for parameter, value in unsupported.items():
        if value is not None:
            raise RequestOptionError(
                f"Responses API does not support chat option '{parameter}'",
                model=request.model,
                parameter=parameter,
            )

    return ResponsesRequest(
        model=request.model,
        input=deepcopy(input_items),
        stream=request.stream,
        instructions=instructions,
        temperature=request.temperature,
        max_output_tokens=request.max_tokens,
        tools=deepcopy(tools),
        tool_choice=deepcopy(tool_choice),
        parallel_tool_calls=None,
        reasoning_effort=effort,
        top_p=request.top_p,
        metadata=deepcopy(request.metadata),
        service_tier=request.service_tier,
        provider_extensions=deepcopy(request.provider_extensions),
    )


# ---------------------------------------------------------------------------
# ResponsesResponse -> ChatResponse
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Finish-reason mapping
# ---------------------------------------------------------------------------


def map_responses_status_to_chat(
    status: str | None,
    incomplete_reason: str | None = None,
) -> str | None:
    """Map a Responses API response status to an OpenAI Chat finish_reason.

    - ``completed`` -> ``stop``
    - ``incomplete`` + ``max_output_tokens`` -> ``length``
    - ``incomplete`` + ``content_filter`` -> ``content_filter``
    - ``incomplete`` (other reasons) -> ``stop`` (closest neutral mapping)
    - ``failed`` / ``cancelled`` -> ``None`` (callers must surface as an error,
      never as a normal finish)
    Returns None if status is unrecognised so callers can apply their own
    default (typically "stop" or "tool_calls").
    """
    if status is None:
        return None
    if status == "completed":
        return "stop"
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "length"
        if incomplete_reason == "content_filter":
            return "content_filter"
        return "stop"
    if status in ("failed", "cancelled"):
        return None
    return None


def responses_response_to_chat_response(
    resp: ResponsesResponse,
    requested_model: str,
    *,
    provider: str = "github-copilot",
) -> ChatResponse:
    """Convert a non-streaming ResponsesResponse back into a ChatResponse.

    Tool calls are reshaped into OpenAI Chat ``tool_calls`` shape so that the
    downstream Anthropic/Gemini translators don't need to know about Responses.
    Upstream ``finish_reason`` (already mapped from Responses ``status``) is
    preserved when present; otherwise it defaults to ``tool_calls`` when tool
    calls are emitted, else ``stop``.
    """
    tool_calls: list[dict] | None = None
    if resp.tool_calls:
        tool_calls = [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in resp.tool_calls
        ]

    # Map Responses usage (input/output) to Chat usage (prompt/completion).
    usage: dict | None = None
    if resp.usage:
        prompt = resp.usage.get("input_tokens", 0)
        completion = resp.usage.get("output_tokens", 0)
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": resp.usage.get("total_tokens", prompt + completion),
        }

    outcome = resp.terminal_outcome
    if outcome is not None and outcome.response_status not in {
        ResponseStatus.COMPLETED,
        ResponseStatus.INCOMPLETE,
    }:
        raise ProviderError(
            f"{provider} /responses ended with status {outcome.response_status.value}",
            status_code=502,
            retryable=False,
            kind=ProviderFailureKind.UPSTREAM_STATUS,
            provider=provider,
            model=requested_model,
        )

    finish_reason = (
        finish_reason_for_outcome(outcome) if outcome is not None else resp.finish_reason
    )
    if tool_calls and finish_reason in (None, "stop"):
        # A "completed" status with tool calls is a tool-use turn, not a normal
        # stop — Anthropic/Gemini translators key tool execution off this.
        finish_reason = "tool_calls"
    elif finish_reason is None:
        finish_reason = "stop"

    return ChatResponse(
        content=resp.content or None,
        model=resp.model or requested_model,
        finish_reason=finish_reason,
        usage=usage,
        tool_calls=tool_calls,
        thinking=getattr(resp, "thinking", None),
        thinking_signature=getattr(resp, "thinking_signature", None),
        thinking_id=getattr(resp, "thinking_id", None),
        refusal=resp.refusal,
        terminal_outcome=outcome,
    )


def responses_chunk_to_chat_chunk(chunk: ResponsesStreamChunk) -> ChatStreamChunk:
    """Convert a streaming ResponsesStreamChunk into a ChatStreamChunk."""
    tool_calls: list[dict] | None = None
    if chunk.tool_call:
        tool_calls = [
            {
                "id": chunk.tool_call.call_id,
                "type": "function",
                "function": {
                    "name": chunk.tool_call.name,
                    "arguments": chunk.tool_call.arguments,
                },
            }
        ]

    usage: dict | None = None
    if chunk.usage:
        prompt = chunk.usage.get("input_tokens", 0)
        completion = chunk.usage.get("output_tokens", 0)
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": chunk.usage.get("total_tokens", prompt + completion),
        }

    finish_reason = chunk.finish_reason
    if chunk.terminal_outcome is not None:
        finish_reason = finish_reason_for_outcome(chunk.terminal_outcome)

    return ChatStreamChunk(
        content=chunk.content or "",
        finish_reason=finish_reason,
        usage=usage,
        tool_calls=tool_calls,
        thinking=getattr(chunk, "thinking", None),
        thinking_signature=getattr(chunk, "thinking_signature", None),
        thinking_id=getattr(chunk, "thinking_id", None),
        terminal_outcome=chunk.terminal_outcome,
        refusal=chunk.refusal,
    )
