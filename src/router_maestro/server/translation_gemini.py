"""Translation between Gemini and OpenAI API formats."""

from __future__ import annotations

import json
from typing import Any

from router_maestro.providers import ChatRequest, Message
from router_maestro.server.schemas.gemini import (
    GeminiCandidate,
    GeminiContent,
    GeminiFunctionCall,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiPart,
    GeminiStreamState,
    GeminiUsageMetadata,
)
from router_maestro.utils import get_logger

logger = get_logger("server.translation_gemini")

# ============================================================================
# Schema type normalization
# ============================================================================

_TYPE_NORMALIZATION_MAP: dict[str, str] = {
    "STRING": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "OBJECT": "object",
    "NULL": "null",
    "String": "string",
    "Number": "number",
    "Integer": "integer",
    "Boolean": "boolean",
    "Array": "array",
    "Object": "object",
    "Null": "null",
}

_NON_SCHEMA_FIELDS = frozenset({"default", "example", "const", "enum"})

_MAX_SCHEMA_DEPTH = 100


def normalize_schema_types(schema: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    """Normalize JSON Schema type values from uppercase to lowercase.

    Gemini uses Protocol Buffer-style uppercase types (e.g. ``"STRING"``,
    ``"OBJECT"``).  Standard JSON Schema uses lowercase.  This function
    recursively normalizes all ``type`` fields.
    """
    if schema is None or not isinstance(schema, (dict, list)):
        return schema

    if _depth >= _MAX_SCHEMA_DEPTH:
        return schema

    if _seen is None:
        _seen = set()

    obj_id = id(schema)
    if obj_id in _seen:
        return schema
    _seen.add(obj_id)

    if isinstance(schema, list):
        return [normalize_schema_types(item, _depth=_depth + 1, _seen=_seen) for item in schema]

    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            upper = value.upper()
            if upper == "TYPE_UNSPECIFIED":
                continue
            normalized[key] = _TYPE_NORMALIZATION_MAP.get(value, value.lower())
        elif key in _NON_SCHEMA_FIELDS:
            normalized[key] = value
        else:
            normalized[key] = normalize_schema_types(value, _depth=_depth + 1, _seen=_seen)
    return normalized


# ============================================================================
# Finish reason mappings
# ============================================================================

_GEMINI_TO_OPENAI_FINISH: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "stop",
    "RECITATION": "stop",
    "OTHER": "stop",
}

_OPENAI_TO_GEMINI_FINISH: dict[str, str] = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",
    "content_filter": "SAFETY",
}


def _map_finish_reason_to_openai(reason: str | None) -> str | None:
    if reason is None:
        return None
    return _GEMINI_TO_OPENAI_FINISH.get(reason, "stop")


def _map_finish_reason_to_gemini(reason: str | None) -> str | None:
    if reason is None:
        return None
    return _OPENAI_TO_GEMINI_FINISH.get(reason, "STOP")


# ============================================================================
# Gemini -> OpenAI translation
# ============================================================================


def translate_gemini_to_openai(request: GeminiGenerateContentRequest, model: str) -> ChatRequest:
    """Translate a Gemini generateContent request to an internal ChatRequest."""
    messages = _translate_contents_to_messages(
        contents=request.contents or [],
        system_instruction=request.system_instruction,
    )
    tools = _translate_gemini_tools(request.tools) if request.tools else None
    tool_choice = (
        _translate_gemini_tool_config(request.tool_config) if request.tool_config else None
    )

    temperature = 1.0
    max_tokens: int | None = None
    if request.generation_config:
        if request.generation_config.temperature is not None:
            temperature = request.generation_config.temperature
        max_tokens = request.generation_config.max_output_tokens

    return ChatRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        tools=tools,
        tool_choice=tool_choice,
    )


def _translate_contents_to_messages(
    contents: list[GeminiContent],
    system_instruction: GeminiContent | None = None,
) -> list[Message]:
    """Convert Gemini contents + systemInstruction to OpenAI messages."""
    messages: list[Message] = []

    # System instruction -> system message
    if system_instruction and system_instruction.parts:
        system_texts = [p.text for p in system_instruction.parts if p.text is not None]
        if system_texts:
            messages.append(Message(role="system", content="\n\n".join(system_texts)))

    for content in contents:
        role = content.role or "user"

        if role == "model":
            messages.extend(_translate_model_content(content))
        else:
            # "user" role — may contain text, function responses, inline data
            messages.extend(_translate_user_content(content))

    return messages


def _translate_user_content(content: GeminiContent) -> list[Message]:
    """Translate a Gemini user content to OpenAI message(s)."""
    messages: list[Message] = []
    text_parts: list[str] = []
    image_parts: list[dict[str, Any]] = []

    for part in content.parts:
        if part.function_response:
            # Function responses -> tool role messages
            fr = part.function_response
            response_text = json.dumps(fr.response) if fr.response else ""
            tool_call_id = fr.id or f"call_{fr.name}"
            messages.append(
                Message(
                    role="tool",
                    content=response_text,
                    tool_call_id=tool_call_id,
                )
            )
        elif part.inline_data:
            media_type = part.inline_data.mime_type or "image/png"
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{part.inline_data.data}"},
                }
            )
        elif part.text is not None:
            text_parts.append(part.text)

    # Build user message from text + images
    if text_parts or image_parts:
        if image_parts:
            content_parts: list[dict[str, Any]] = []
            if text_parts:
                content_parts.append({"type": "text", "text": "\n\n".join(text_parts)})
            content_parts.extend(image_parts)
            messages.append(Message(role="user", content=content_parts))
        else:
            messages.append(Message(role="user", content="\n\n".join(text_parts)))

    if not messages:
        messages.append(Message(role="user", content=""))

    return messages


def _translate_model_content(content: GeminiContent) -> list[Message]:
    """Translate a Gemini model content to OpenAI assistant message."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in content.parts:
        if part.function_call:
            fc = part.function_call
            call_id = fc.id or f"call_{fc.name}"
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(fc.args),
                    },
                }
            )
        elif part.text is not None:
            text_parts.append(part.text)

    return [
        Message(
            role="assistant",
            content="\n\n".join(text_parts) if text_parts else "",
            tool_calls=tool_calls if tool_calls else None,
        )
    ]


def _translate_gemini_tools(tools: list) -> list[dict[str, Any]] | None:
    """Translate Gemini tools to OpenAI function tools."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        declarations = (
            tool.function_declarations
            if hasattr(tool, "function_declarations")
            else tool.get("functionDeclarations", [])
        )
        if not declarations:
            continue
        for decl in declarations:
            name = decl.name if hasattr(decl, "name") else decl.get("name", "")
            if not name:
                continue
            description = (
                decl.description if hasattr(decl, "description") else decl.get("description", "")
            ) or ""
            raw_params = (
                decl.parameters if hasattr(decl, "parameters") else decl.get("parameters", {})
            ) or {}
            parameters = normalize_schema_types(raw_params)
            # Ensure parameters has at least {"type": "object"} for OpenAI compat
            if not parameters or "type" not in parameters:
                parameters = {"type": "object", "properties": {}, "required": []}

            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
    return result if result else None


def _translate_gemini_tool_config(tool_config) -> str | None:
    """Translate Gemini toolConfig to OpenAI tool_choice."""
    if tool_config is None:
        return None
    fcc = (
        tool_config.function_calling_config
        if hasattr(tool_config, "function_calling_config")
        else tool_config.get("functionCallingConfig")
    )
    if fcc is None:
        return None
    mode = fcc.mode if hasattr(fcc, "mode") else fcc.get("mode")
    if mode is None:
        return None

    mode_upper = mode.upper()
    if mode_upper in ("AUTO", "VALIDATED"):
        return "auto"
    elif mode_upper == "ANY":
        return "required"
    elif mode_upper == "NONE":
        return "none"
    return None


# ============================================================================
# OpenAI -> Gemini translation (non-streaming)
# ============================================================================


def translate_openai_to_gemini(
    response: Any,
    model: str,
    input_tokens: int = 0,
) -> GeminiGenerateContentResponse:
    """Translate an OpenAI ChatResponse to Gemini generateContent response."""
    parts: list[GeminiPart] = []

    # Text content
    if response.content:
        parts.append(GeminiPart(text=response.content))

    # Tool calls
    if response.tool_calls:
        for tc in response.tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            call_id = tc.get("id") or f"call_{name}"
            parts.append(
                GeminiPart(
                    function_call=GeminiFunctionCall(
                        name=name,
                        args=args,
                        id=call_id,
                    )
                )
            )

    # Determine finish reason
    finish_reason = _map_finish_reason_to_gemini(response.finish_reason)

    # Usage
    prompt_tokens = 0
    completion_tokens = 0
    if response.usage:
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)

    # Use input_tokens estimate if upstream didn't provide
    if prompt_tokens == 0 and input_tokens > 0:
        prompt_tokens = input_tokens

    return GeminiGenerateContentResponse(
        candidates=[
            GeminiCandidate(
                content=GeminiContent(parts=parts, role="model"),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        usage_metadata=GeminiUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=prompt_tokens + completion_tokens,
        ),
        model_version=model,
    )


# ============================================================================
# OpenAI -> Gemini translation (streaming)
# ============================================================================


def translate_openai_chunk_to_gemini(
    chunk: dict[str, Any],
    state: GeminiStreamState,
    model: str,
) -> GeminiGenerateContentResponse | None:
    """Translate a single OpenAI streaming chunk to a Gemini SSE response.

    Returns ``None`` if the chunk produces no Gemini event.
    """
    # Track usage
    if chunk.get("usage"):
        usage = chunk["usage"]
        ct = usage.get("completion_tokens", 0)
        if ct > 0:
            state.accumulated_completion_tokens = max(state.accumulated_completion_tokens, ct)
        pt = usage.get("prompt_tokens", 0)
        if pt > 0:
            state.accumulated_prompt_tokens = max(state.accumulated_prompt_tokens, pt)

    if not chunk.get("choices"):
        return None

    choice = chunk["choices"][0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    # Text delta
    if delta.get("content"):
        text = delta["content"]
        state.accumulated_text += text
        state.has_sent_content = True
        return GeminiGenerateContentResponse(
            candidates=[
                GeminiCandidate(
                    content=GeminiContent(
                        parts=[GeminiPart(text=text)],
                        role="model",
                    ),
                    index=0,
                )
            ],
        )

    # Tool call deltas — only buffer, never emit mid-stream (args arrive across chunks)
    if delta.get("tool_calls"):
        for tc in delta["tool_calls"]:
            if tc.get("id") and tc.get("function", {}).get("name"):
                # New tool call start
                state.tool_calls_buffer.append(
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", ""),
                    }
                )
            elif tc.get("function", {}).get("arguments"):
                # Continuation of existing tool call
                idx = tc.get("index", 0)
                if idx < len(state.tool_calls_buffer):
                    state.tool_calls_buffer[idx]["arguments"] += tc["function"]["arguments"]
        # Do NOT emit here — wait for finish chunk where args are complete

    # Finish
    if finish_reason:
        gemini_finish = _map_finish_reason_to_gemini(finish_reason)

        # Emit final tool calls with accumulated arguments if any
        final_parts: list[GeminiPart] = []
        for tc_buf in state.tool_calls_buffer:
            try:
                args = json.loads(tc_buf["arguments"]) if tc_buf["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            final_parts.append(
                GeminiPart(
                    function_call=GeminiFunctionCall(
                        name=tc_buf["name"],
                        args=args,
                        id=tc_buf["id"],
                    )
                )
            )
        state.tool_calls_buffer.clear()

        prompt_tokens = state.accumulated_prompt_tokens or state.estimated_input_tokens
        completion_tokens = state.accumulated_completion_tokens

        candidate = GeminiCandidate(
            finish_reason=gemini_finish,
            index=0,
        )
        if final_parts:
            candidate.content = GeminiContent(parts=final_parts, role="model")

        return GeminiGenerateContentResponse(
            candidates=[candidate],
            usage_metadata=GeminiUsageMetadata(
                prompt_token_count=prompt_tokens,
                candidates_token_count=completion_tokens,
                total_token_count=prompt_tokens + completion_tokens,
            ),
            model_version=model,
        )

    return None
