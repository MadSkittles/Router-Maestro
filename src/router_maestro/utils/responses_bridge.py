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

from router_maestro.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
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
        "gpt-5-mini",
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


def _message_has_non_text_content(content) -> bool:
    """True if a Message.content list contains any non-plain-text block.

    The Responses bridge only forwards plain text (string content or
    ``{"type": "text", ...}`` blocks). Anything else — image_url, image,
    input_image, audio, file, **document** (Anthropic), and any future
    structured block — must force fallback to /chat/completions so we
    don't silently drop modalities the user actually sent.
    """
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, str):
            continue
        if not isinstance(block, dict):
            # Unknown shape — treat as non-text to be safe.
            return True
        btype = block.get("type")
        if btype == "text":
            continue
        # Anything other than a text block (or an undecorated dict that just
        # carries text) is structured and unsafe to drop silently.
        if btype is None and isinstance(block.get("text"), str):
            continue
        return True
    return False


def request_has_non_text_content(request: ChatRequest) -> bool:
    """True if any message in the request carries non-text content blocks."""
    return any(_message_has_non_text_content(m.content) for m in request.messages)


def should_use_responses_for_chat(request: ChatRequest, provider_name: str) -> bool:
    """Decide whether a ChatRequest should be fulfilled via /responses.

    Requires:
    - the experimental env flag (kill-switch enforced here, not just at the
      entry routes, so any caller setting ``use_responses_api=True`` is still
      gated by ops);
    - the per-request opt-in flag;
    - the Copilot provider (others have no /responses endpoint we target);
    - an eligible model;
    - text-only content (multimodal requests fall back to /chat/completions).
    """
    if not is_experimental_responses_enabled():
        return False
    if not request.use_responses_api:
        return False
    if provider_name != "github-copilot":
        return False
    if not is_model_responses_eligible(request.model):
        return False
    if request_has_non_text_content(request):
        return False
    return True


# ---------------------------------------------------------------------------
# ChatRequest -> ResponsesRequest
# ---------------------------------------------------------------------------


def _content_to_text(content: str | list) -> str:
    """Flatten a Message.content (str or list of OpenAI-style blocks) to text.

    Multi-block text content is joined with a blank line separator to match
    other translators in this codebase and avoid silently merging block
    boundaries (e.g., ``"foo"`` + ``"bar"`` becoming ``"foobar"``).
    Multimodal blocks are ignored here — callers should have already
    short-circuited via ``request_has_non_text_content`` before reaching
    this point.
    """
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
    items. Plain user/assistant text become ``message`` items.
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
            text = _content_to_text(msg.content) if msg.content else ""
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        # Replayed assistant turns are *input* to the next call,
                        # so use input_text (matches the project schema and what
                        # Copilot's /responses accepts as request-history).
                        "content": [{"type": "input_text", "text": text}],
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
        text = _content_to_text(msg.content)
        items.append(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
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

    return ResponsesRequest(
        model=request.model,
        input=input_items,
        stream=request.stream,
        instructions=instructions,
        temperature=request.temperature,
        max_output_tokens=request.max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=None,
        reasoning_effort=effort,
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
    resp: ResponsesResponse, requested_model: str
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

    finish_reason = resp.finish_reason
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

    return ChatStreamChunk(
        content=chunk.content or "",
        finish_reason=chunk.finish_reason,
        usage=usage,
        tool_calls=tool_calls,
        thinking=getattr(chunk, "thinking", None),
        thinking_signature=getattr(chunk, "thinking_signature", None),
    )
