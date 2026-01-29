"""Translation between Anthropic and OpenAI API formats."""

from router_maestro.providers import ChatRequest, Message
from router_maestro.server.schemas.anthropic import (
    AnthropicAssistantContentBlock,
    AnthropicAssistantMessage,
    AnthropicImageBlock,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamState,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
    AnthropicUserMessage,
)
from router_maestro.utils import get_logger, map_openai_stop_reason_to_anthropic

logger = get_logger("server.translation")


def translate_anthropic_to_openai(request: AnthropicMessagesRequest) -> ChatRequest:
    """Translate Anthropic Messages request to OpenAI ChatCompletion request."""
    messages = _translate_messages(request.messages, request.system)
    tools = _translate_tools(request.tools) if request.tools else None
    tool_choice = _translate_tool_choice(request.tool_choice) if request.tool_choice else None

    logger.debug(
        "Translating Anthropic request: model=%s -> %s, messages=%d",
        request.model,
        _translate_model_name(request.model),
        len(messages),
    )

    return ChatRequest(
        model=_translate_model_name(request.model),
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream,
        tools=tools,
        tool_choice=tool_choice,
    )


def _translate_model_name(model: str) -> str:
    """Translate model name for compatibility.

    Claude Code uses model names like 'claude-sonnet-4-20250514' or 'claude-sonnet-4.5'.
    The Copilot API uses names like 'claude-sonnet-4' or may accept the full version.
    """
    # Handle Claude model version suffixes
    # e.g., claude-sonnet-4-20250514 -> claude-sonnet-4
    # e.g., claude-opus-4.5 -> claude-opus-4.5 (keep as-is, it's a valid model)
    # e.g., claude-haiku-4-5-20251001 -> claude-haiku-4.5 (hyphenated version to dot)
    import re

    # Pattern: claude-{tier}-{major}[-{date_suffix}]
    # We want to strip date suffixes like -20250514 but keep version numbers like .5
    match = re.match(r"^(claude-(?:sonnet|opus|haiku)-\d+(?:\.\d+)?)-\d{8}$", model)
    if match:
        return match.group(1)

    # Handle hyphenated version numbers (e.g., claude-haiku-4-5-20251001 -> claude-haiku-4.5)
    # Claude Code may send versions like "4-5" instead of "4.5"
    match = re.match(r"^(claude-(?:sonnet|opus|haiku))-(\d+)-(\d+)-(\d{8})$", model)
    if match:
        tier = match.group(1)
        major = match.group(2)
        minor = match.group(3)
        return f"{tier}-{major}.{minor}"

    return model


def _translate_tools(tools: list) -> list[dict]:
    """Translate Anthropic tools to OpenAI format.

    Anthropic format:
    {
        "name": "tool_name",
        "description": "description",
        "input_schema": {...}  # JSON Schema
    }

    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "description",
            "parameters": {...}  # JSON Schema
        }
    }
    """
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("input_schema", {})
        else:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
            input_schema = getattr(tool, "input_schema", {})

        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": input_schema,
                },
            }
        )
    return result


def _translate_tool_choice(tool_choice) -> str | dict | None:
    """Translate Anthropic tool_choice to OpenAI format.

    Anthropic format:
    - {"type": "auto"} -> "auto"
    - {"type": "any"} -> "required"
    - {"type": "tool", "name": "tool_name"} ->
      {"type": "function", "function": {"name": "tool_name"}}

    OpenAI format:
    - "auto" - model decides
    - "none" - no tools
    - "required" - must use a tool
    - {"type": "function", "function": {"name": "..."}} - specific tool
    """
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            return "required"
        elif choice_type == "tool":
            tool_name = tool_choice.get("name", "")
            return {"type": "function", "function": {"name": tool_name}}
    return None


def _sanitize_system_prompt(system: str) -> str:
    """Remove reserved keywords from system prompt that Copilot rejects."""
    import re

    # Remove x-anthropic-billing-header line (Claude Code adds this)
    # Pattern matches the header line and any following newlines
    system = re.sub(r"x-anthropic-billing-header:[^\n]*\n*", "", system)
    return system.strip()


def _translate_messages(
    messages: list, system: str | list[AnthropicTextBlock] | None
) -> list[Message]:
    """Translate Anthropic messages to OpenAI format."""
    result: list[Message] = []

    # Handle system prompt
    if system:
        if isinstance(system, str):
            system_text = _sanitize_system_prompt(system)
            result.append(Message(role="system", content=system_text))
        else:
            system_text = "\n\n".join(block.text for block in system)
            system_text = _sanitize_system_prompt(system_text)
            result.append(Message(role="system", content=system_text))

    # Handle conversation messages
    for msg in messages:
        is_user = isinstance(msg, AnthropicUserMessage) or (
            isinstance(msg, dict) and msg.get("role") == "user"
        )
        is_assistant = isinstance(msg, AnthropicAssistantMessage) or (
            isinstance(msg, dict) and msg.get("role") == "assistant"
        )
        if is_user:
            result.extend(_handle_user_message(msg))
        elif is_assistant:
            result.extend(_handle_assistant_message(msg))

    return result


def _handle_user_message(message: AnthropicUserMessage | dict) -> list[Message]:
    """Handle user message translation."""
    if isinstance(message, AnthropicUserMessage):
        content = message.content
    else:
        content = message.get("content", "")

    if isinstance(content, str):
        return [Message(role="user", content=content)]

    # Handle content blocks
    tool_results = []
    other_blocks = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
        else:
            block_type = getattr(block, "type", None)

        if block_type == "tool_result":
            tool_results.append(block)
        else:
            other_blocks.append(block)

    result: list[Message] = []

    # Tool results become tool role messages in OpenAI format
    for block in tool_results:
        if isinstance(block, dict):
            tool_content = block.get("content", "")
            tool_use_id = block.get("tool_use_id", "")
        else:
            tool_content = block.content
            tool_use_id = block.tool_use_id

        # Handle content as array of content blocks
        if isinstance(tool_content, list):
            text_parts = []
            for item in tool_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif hasattr(item, "type") and item.type == "text":
                    text_parts.append(getattr(item, "text", ""))
            tool_content = "\n".join(text_parts)

        result.append(
            Message(
                role="tool",
                content=str(tool_content),
                tool_call_id=tool_use_id,
            )
        )

    # Other content becomes user message - handle both text and images
    if other_blocks:
        multimodal_content = _extract_multimodal_content(other_blocks)
        if multimodal_content:
            result.append(Message(role="user", content=multimodal_content))

    return result if result else [Message(role="user", content="")]


def _handle_assistant_message(message: AnthropicAssistantMessage | dict) -> list[Message]:
    """Handle assistant message translation."""
    if isinstance(message, AnthropicAssistantMessage):
        content = message.content
    else:
        content = message.get("content", "")

    if isinstance(content, str):
        return [Message(role="assistant", content=content)]

    # Extract text content and tool_use blocks
    text_content = _extract_text_content(content)
    tool_calls = _extract_tool_calls(content)

    return [Message(role="assistant", content=text_content or "", tool_calls=tool_calls)]


def _extract_tool_calls(blocks: list) -> list[dict] | None:
    """Extract tool_use blocks and convert to OpenAI tool_calls format."""
    tool_calls = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "tool_use":
                tool_call = {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    },
                }
                # Convert input to JSON string if it's a dict
                if isinstance(tool_call["function"]["arguments"], dict):
                    import json

                    tool_call["function"]["arguments"] = json.dumps(
                        tool_call["function"]["arguments"]
                    )
                tool_calls.append(tool_call)
        elif isinstance(block, AnthropicToolUseBlock):
            import json

            tool_call = {
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input)
                    if isinstance(block.input, dict)
                    else str(block.input),
                },
            }
            tool_calls.append(tool_call)
    return tool_calls if tool_calls else None


def _extract_text_content(blocks: list) -> str:
    """Extract text content from content blocks."""
    texts = []
    for block in blocks:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                texts.append(block.get("text", ""))
            elif block_type == "thinking":
                texts.append(block.get("thinking", ""))
        elif isinstance(block, AnthropicTextBlock):
            texts.append(block.text)
        elif isinstance(block, AnthropicThinkingBlock):
            texts.append(block.thinking)
    return "\n\n".join(texts)


def _extract_multimodal_content(blocks: list) -> str | list:
    """Extract content from blocks, handling both text and images.

    Returns a string if only text is present, or a list of content parts
    for multimodal content (OpenAI format).
    """
    text_parts = []
    image_parts = []

    for block in blocks:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                text_parts.append(block.get("thinking", ""))
            elif block_type == "image":
                # Convert Anthropic image format to OpenAI format
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
        elif isinstance(block, AnthropicTextBlock):
            text_parts.append(block.text)
        elif isinstance(block, AnthropicThinkingBlock):
            text_parts.append(block.thinking)
        elif isinstance(block, AnthropicImageBlock):
            # Convert Anthropic image to OpenAI format
            media_type = block.source.media_type
            data = block.source.data
            image_parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
            )

    # If no images, return simple text string
    if not image_parts:
        return "\n\n".join(text_parts)

    # Build multimodal content list (OpenAI format)
    content_parts = []

    # Add text parts first
    if text_parts:
        content_parts.append({"type": "text", "text": "\n\n".join(text_parts)})

    # Add image parts
    content_parts.extend(image_parts)

    return content_parts


def translate_openai_to_anthropic(
    openai_response: dict, model: str, request_id: str
) -> AnthropicMessagesResponse:
    """Translate OpenAI ChatCompletion response to Anthropic Messages response."""
    content: list[AnthropicAssistantContentBlock] = []

    # Extract content from choices
    if "choices" in openai_response:
        for choice in openai_response["choices"]:
            message = choice.get("message", {})
            msg_content = message.get("content")

            if msg_content:
                content.append(AnthropicTextBlock(type="text", text=msg_content))

            # Handle tool calls if present
            tool_calls = message.get("tool_calls", [])
            for tool_call in tool_calls:
                content.append(
                    AnthropicToolUseBlock(
                        type="tool_use",
                        id=tool_call.get("id", ""),
                        name=tool_call.get("function", {}).get("name", ""),
                        input=tool_call.get("function", {}).get("arguments", {}),
                    )
                )

    # Map finish reason
    finish_reason = None
    if openai_response.get("choices"):
        openai_reason = openai_response["choices"][0].get("finish_reason")
        finish_reason = _map_stop_reason(openai_reason)

    # Extract usage
    openai_usage = openai_response.get("usage", {})
    usage = AnthropicUsage(
        input_tokens=openai_usage.get("prompt_tokens", 0),
        output_tokens=openai_usage.get("completion_tokens", 0),
    )

    return AnthropicMessagesResponse(
        id=request_id,
        type="message",
        role="assistant",
        content=content,
        model=model,
        stop_reason=finish_reason,
        stop_sequence=None,
        usage=usage,
    )


def _map_stop_reason(
    openai_reason: str | None,
) -> str | None:
    """Map OpenAI finish reason to Anthropic stop reason."""
    return map_openai_stop_reason_to_anthropic(openai_reason)


def translate_openai_chunk_to_anthropic_events(
    chunk: dict, state: AnthropicStreamState, model: str
) -> list[dict]:
    """Translate OpenAI streaming chunk to Anthropic SSE events."""
    events: list[dict] = []

    # Don't process any more chunks after message is complete
    if state.message_complete:
        return events

    # Track latest usage info from any chunk that contains it
    if chunk.get("usage"):
        state.last_usage = chunk["usage"]

    if not chunk.get("choices"):
        return events

    choice = chunk["choices"][0]
    delta = choice.get("delta", {})

    # Send message_start if not sent yet
    if not state.message_start_sent:
        # Determine input tokens: prefer actual usage, fall back to estimate
        input_tokens = 0
        if state.last_usage:
            input_tokens = state.last_usage.get("prompt_tokens", 0)
        elif state.estimated_input_tokens:
            input_tokens = state.estimated_input_tokens

        events.append(
            {
                "type": "message_start",
                "message": {
                    "id": chunk.get("id", ""),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 1,
                        "cache_creation_input_tokens": None,
                        "cache_read_input_tokens": None,
                        "server_tool_use": None,
                        "service_tier": "standard",
                    },
                },
            }
        )
        state.message_start_sent = True

    # Handle text content
    if delta.get("content"):
        # Close tool block if open
        if _is_tool_block_open(state):
            events.append(
                {
                    "type": "content_block_stop",
                    "index": state.content_block_index,
                }
            )
            state.content_block_index += 1
            state.content_block_open = False

        # Start text block if not open
        if not state.content_block_open:
            events.append(
                {
                    "type": "content_block_start",
                    "index": state.content_block_index,
                    "content_block": {
                        "type": "text",
                        "text": "",
                    },
                }
            )
            state.content_block_open = True

        # Send text delta
        events.append(
            {
                "type": "content_block_delta",
                "index": state.content_block_index,
                "delta": {
                    "type": "text_delta",
                    "text": delta["content"],
                },
            }
        )

    # Handle tool calls
    if delta.get("tool_calls"):
        for tool_call in delta["tool_calls"]:
            tool_index = tool_call.get("index", 0)

            if tool_call.get("id") and tool_call.get("function", {}).get("name"):
                # New tool call starting
                if state.content_block_open:
                    events.append(
                        {
                            "type": "content_block_stop",
                            "index": state.content_block_index,
                        }
                    )
                    state.content_block_index += 1
                    state.content_block_open = False

                anthropic_block_index = state.content_block_index
                state.tool_calls[tool_index] = {
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "anthropic_block_index": anthropic_block_index,
                }

                events.append(
                    {
                        "type": "content_block_start",
                        "index": anthropic_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": {},
                        },
                    }
                )
                state.content_block_open = True

            if tool_call.get("function", {}).get("arguments"):
                tool_info = state.tool_calls.get(tool_index)
                if tool_info:
                    events.append(
                        {
                            "type": "content_block_delta",
                            "index": tool_info["anthropic_block_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tool_call["function"]["arguments"],
                            },
                        }
                    )

    # Handle finish
    finish_reason = choice.get("finish_reason")
    if finish_reason:
        if state.content_block_open:
            events.append(
                {
                    "type": "content_block_stop",
                    "index": state.content_block_index,
                }
            )
            state.content_block_open = False

        # Get usage from chunk or from tracked last_usage
        usage = chunk.get("usage") or state.last_usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Use estimated_input_tokens for context display since Copilot may truncate input
        # This gives Claude Code accurate context percentage based on actual conversation size
        input_tokens_for_display = (
            state.estimated_input_tokens if state.estimated_input_tokens > 0 else prompt_tokens
        )

        events.append(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _map_stop_reason(finish_reason),
                    "stop_sequence": None,
                },
                "usage": {
                    "input_tokens": input_tokens_for_display,
                    "output_tokens": completion_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "server_tool_use": None,
                },
            }
        )
        events.append({"type": "message_stop"})
        state.message_complete = True

    return events


def _is_tool_block_open(state: AnthropicStreamState) -> bool:
    """Check if a tool block is currently open."""
    if not state.content_block_open:
        return False
    return any(
        tc["anthropic_block_index"] == state.content_block_index for tc in state.tool_calls.values()
    )
