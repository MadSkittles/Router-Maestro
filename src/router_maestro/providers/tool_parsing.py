"""Recovery utility for tool calls embedded as XML in content text.

Some providers (notably GitHub Copilot) sometimes return finish_reason="tool_calls"
but place the tool calls as XML text in message.content instead of the proper
message.tool_calls field. This module detects and recovers those cases.

NOTE: Streaming recovery is not yet supported. XML tool call detection in streamed
text deltas would require buffering to detect XML boundaries, which is significantly
more complex. Non-streaming recovery covers the NanoBot use case.
"""

import json
import re
from uuid import uuid4

from router_maestro.utils import get_logger

logger = get_logger("providers.tool_parsing")

# Matches <tool_call>{ ... }</tool_call> or <function>{ ... }</function>
_TOOL_CALL_RE = re.compile(
    r"<(tool_call|function)\s*>\s*(\{.*?\})\s*</\1>",
    re.DOTALL,
)

# Matches <tool_result>...</tool_result> blocks (Copilot sometimes self-inserts results)
_TOOL_RESULT_RE = re.compile(
    r"<tool_result\b[^>]*>.*?</tool_result>",
    re.DOTALL,
)


def recover_tool_calls_from_content(
    content: str | None,
    tool_calls: list[dict] | None,
    finish_reason: str | None,
) -> tuple[str | None, list[dict] | None]:
    """Recover tool calls from content text when provider embeds them as XML.

    Returns (cleaned_content, tool_calls).
    Only activates when ALL conditions are met:
    1. finish_reason == "tool_calls"
    2. tool_calls is None or empty
    3. content contains recognized tool call XML tags

    If tool_calls already present, returns inputs unchanged (idempotent).
    On parse errors, skips malformed blocks (logs warning), never fails the request.
    """
    # Guard: only activate when provider signals tool calls but field is empty
    if finish_reason != "tool_calls":
        return content, tool_calls
    if tool_calls:
        return content, tool_calls
    if not content:
        return content, tool_calls

    matches = _TOOL_CALL_RE.findall(content)
    if not matches:
        return content, tool_calls

    recovered: list[dict] = []
    for _tag, json_str in matches:
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed tool call JSON: %.200s", json_str)
            continue

        name = parsed.get("name")
        arguments = parsed.get("arguments")
        if not name:
            logger.warning("Skipping tool call block without 'name': %.200s", json_str)
            continue

        # arguments may be a dict or a string; normalize to string
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif arguments is None:
            arguments = "{}"

        tool_call_id = f"toolu_{uuid4().hex[:24]}"
        recovered.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )

    if not recovered:
        # All blocks were malformed; return original unchanged
        return content, tool_calls

    logger.info("Recovered %d tool call(s) from XML in content", len(recovered))

    # Clean content: remove matched XML blocks and tool_result blocks
    cleaned = _TOOL_CALL_RE.sub("", content)
    cleaned = _TOOL_RESULT_RE.sub("", cleaned)
    cleaned = cleaned.strip()

    return cleaned or None, recovered
