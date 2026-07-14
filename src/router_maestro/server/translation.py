"""Translation between Anthropic and OpenAI API formats."""

import json
import re

from router_maestro.providers import ChatRequest, ChatStreamChunk, Message
from router_maestro.server.protocols.anthropic_reducer import AnthropicReducer
from router_maestro.server.schemas.anthropic import (
    AnthropicAssistantMessage,
    AnthropicMessagesRequest,
    AnthropicStreamState,
    AnthropicTextBlock,
    AnthropicUserMessage,
)
from router_maestro.utils import get_logger

logger = get_logger("server.translation")


def _get_block_field(block, field: str, default=None):
    """Get a field from a content block (dict or Pydantic object)."""
    if isinstance(block, dict):
        return block.get(field, default)
    return getattr(block, field, default)


def _get_block_type(block) -> str | None:
    """Get the 'type' field from a content block."""
    return _get_block_field(block, "type")


def _document_block_to_dict(block) -> dict | None:
    """Convert an Anthropic document block to a plain dict for passthrough.

    Returned in Anthropic-native shape so AnthropicProvider forwards it verbatim.
    """
    source = _get_block_field(block, "source")
    if source is None:
        return None
    src_type = _get_block_field(source, "type")
    if src_type is None:
        return None
    src_dict: dict = {"type": src_type}
    for f in ("media_type", "data", "url", "content"):
        v = _get_block_field(source, f)
        if v is not None:
            src_dict[f] = v
    doc: dict = {"type": "document", "source": src_dict}
    for f in ("title", "context", "citations"):
        v = _get_block_field(block, f)
        if v is not None:
            doc[f] = v
    return doc


def translate_anthropic_to_openai(request: AnthropicMessagesRequest) -> ChatRequest:
    """Translate Anthropic Messages request to OpenAI ChatCompletion request."""
    messages = _translate_messages(request.messages, request.system)
    tools = _translate_tools(request.tools) if request.tools else None
    tool_choice = _translate_tool_choice(request.tool_choice) if request.tool_choice else None

    # Extract thinking configuration
    thinking_budget = None
    thinking_type = None
    if request.thinking:
        thinking_type = request.thinking.type
        thinking_budget = request.thinking.budget_tokens

    reasoning_effort = request.output_config.effort if request.output_config else None

    logger.debug(
        "Translating Anthropic request: model=%s -> %s, messages=%d, reasoning_effort=%s",
        request.model,
        _translate_model_name(request.model),
        len(messages),
        reasoning_effort,
    )

    return ChatRequest(
        model=_translate_model_name(request.model),
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream,
        tools=tools,
        tool_choice=tool_choice,
        thinking_budget=thinking_budget,
        thinking_type=thinking_type,
        reasoning_effort=reasoning_effort,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=request.stop_sequences,
        metadata=request.metadata,
        service_tier=request.service_tier,
    )


def _translate_model_name(model: str) -> str:
    """Preserve the client's model identity for Router resolution.

    Undated names already act as family aliases. Dated/versioned names are concrete
    catalog identities and must not be collapsed before capability-aware routing.
    """
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
        name = _get_block_field(tool, "name", "")
        description = _get_block_field(tool, "description", "")
        input_schema = _get_block_field(tool, "input_schema", {})

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
    # tool_choice may arrive as a dict or as an AnthropicToolChoice Pydantic
    # object (the typed request path); _get_block_field handles both.
    choice_type = _get_block_field(tool_choice, "type")
    if choice_type == "auto":
        return "auto"
    elif choice_type == "none":
        return "none"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        tool_name = _get_block_field(tool_choice, "name", "")
        return {"type": "function", "function": {"name": tool_name}}
    return None


def _sanitize_system_prompt(system: str) -> str:
    """Remove reserved keywords from system prompt that Copilot rejects."""
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
        block_type = _get_block_type(block)

        if block_type == "tool_result":
            tool_results.append(block)
        else:
            other_blocks.append(block)

    result: list[Message] = []
    all_image_blocks: list = []

    # Tool results become tool role messages in OpenAI format
    for block in tool_results:
        tool_content = _get_block_field(block, "content", "")
        tool_use_id = _get_block_field(block, "tool_use_id", "")

        # Handle content as array of content blocks
        if isinstance(tool_content, list):
            text_parts = []
            for item in tool_content:
                item_type = _get_block_type(item)
                if item_type == "text":
                    text_parts.append(_get_block_field(item, "text", ""))
                elif item_type == "image":
                    all_image_blocks.append(item)
                elif item_type == "document":
                    all_image_blocks.append(item)
                elif item_type == "tool_reference":
                    logger.debug(
                        "Skipping tool_reference block: %s",
                        _get_block_field(item, "tool_name"),
                    )
                elif item_type is not None:
                    logger.warning("Unknown content block type in tool_result: %s", item_type)
            tool_content = "\n".join(text_parts)

        result.append(
            Message(
                role="tool",
                content=str(tool_content),
                tool_call_id=tool_use_id,
            )
        )

    # OpenAI tool messages only support text content, so inject images
    # from tool results as a follow-up user message after all tool messages
    if all_image_blocks:
        multimodal = _extract_multimodal_content(all_image_blocks)
        if multimodal:
            result.append(Message(role="user", content=multimodal))

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
        if _get_block_type(block) != "tool_use":
            continue

        block_id = _get_block_field(block, "id", "")
        block_name = _get_block_field(block, "name", "")
        block_input = _get_block_field(block, "input", {})

        arguments = json.dumps(block_input) if isinstance(block_input, dict) else str(block_input)

        tool_calls.append(
            {
                "id": block_id,
                "type": "function",
                "function": {
                    "name": block_name,
                    "arguments": arguments,
                },
            }
        )
    return tool_calls if tool_calls else None


def _extract_text_content(blocks: list) -> str:
    """Extract text content from content blocks."""
    texts = []
    for block in blocks:
        block_type = _get_block_type(block)
        if block_type == "text":
            texts.append(_get_block_field(block, "text", ""))
        # thinking blocks are intentionally dropped: they have no place in the
        # OpenAI assistant `content` field and replaying them poisons history.
    return "\n\n".join(texts)


def _extract_multimodal_content(blocks: list) -> str | list:
    """Extract content from blocks, handling both text and images.

    Returns a string if only text is present, or a list of content parts
    for multimodal content (OpenAI format).
    """
    text_parts = []
    image_parts = []

    for block in blocks:
        block_type = _get_block_type(block)
        if block_type == "text":
            text_parts.append(_get_block_field(block, "text", ""))
        elif block_type == "image":
            source = _get_block_field(block, "source", {})
            # Handle both dict and object sources
            source_type = _get_block_field(source, "type") if source else None
            if source_type == "base64":
                media_type = _get_block_field(source, "media_type", "image/png")
                data = _get_block_field(source, "data", "")
                image_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )
        elif block_type == "document":
            doc_part = _document_block_to_dict(block)
            if doc_part is not None:
                image_parts.append(doc_part)

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


def translate_openai_chunk_to_anthropic_events(
    chunk: dict, state: AnthropicStreamState, model: str
) -> list[dict]:
    """Adapt a legacy OpenAI chunk dict to the canonical Anthropic reducer."""
    if state.message_complete:
        return []

    reducer = AnthropicReducer(
        response_id=chunk.get("id", ""),
        model=model,
        state=state,
    )
    choices = chunk.get("choices") or []
    if not choices:
        reducer.observe_usage(chunk.get("usage"))
        return []

    choice = choices[0]
    delta = choice.get("delta") or {}
    thinking = None
    for key in ("reasoning_text", "cot_summary", "thinking"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            thinking = value
            break
        if isinstance(value, dict):
            inner = value.get("text") or value.get("content")
            if isinstance(inner, str) and inner:
                thinking = inner
                break
    thinking_signature = None
    for key in ("reasoning_opaque", "cot_id", "signature"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            thinking_signature = value
            break
    return reducer.reduce(
        ChatStreamChunk(
            content=delta.get("content") or "",
            refusal=delta.get("refusal") or None,
            finish_reason=choice.get("finish_reason"),
            usage=chunk.get("usage"),
            tool_calls=delta.get("tool_calls"),
            thinking=thinking,
            thinking_signature=thinking_signature,
        )
    )
