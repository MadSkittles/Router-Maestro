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

# Matches the antml/"invoke" tool-call dialect that Claude models emit as TEXT
# when they simulate a tool call instead of using the structured tool_calls
# field (observed on github-copilot/claude-* under long contexts):
#
#   <invoke name="Bash">
#     <parameter name="command">ls</parameter>
#   </invoke>
#
# Requires a closing </invoke> so half-streamed / truncated fragments never
# match (they pass through as plain text rather than being silently dropped).
_INVOKE_RE = re.compile(
    r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>',
    re.DOTALL,
)
_INVOKE_PARAM_RE = re.compile(
    r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>',
    re.DOTALL,
)
# Fenced (``` ... ```) and inline (`...`) code spans. Stripped before scanning
# for <invoke> so that prose *discussing* tool-call syntax inside code spans is
# not misread as a real tool call.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`", re.DOTALL)


def recover_invoke_tool_calls(
    content: str | None,
    allowed_names: set[str] | None,
) -> list[dict] | None:
    """Recover antml/<invoke> tool calls leaked as text into assistant content.

    Returns a list of OpenAI-format tool_call dicts (each with an explicit
    ``index``), or ``None`` when nothing safely recoverable is found.

    Three false-positive guards must ALL pass for a block to be recovered, so
    that prose merely *mentioning* ``<invoke>`` (e.g. a model explaining the
    bug, or a fenced code sample) is never converted into a real tool call:

    1. Structural close: only ``<invoke ...>...</invoke>`` pairs match; a bare
       or truncated ``<invoke ...>`` is ignored.
    2. Code-span exclusion: fenced ```` ``` ```` and inline ``` ` ``` spans are
       stripped before scanning, so tool-call syntax shown as code is skipped.
    3. Tool-name cross-check: when ``allowed_names`` is provided, a recovered
       call whose ``name`` is not in the request's tool list is dropped.

    This is intentionally conservative: when in doubt it recovers nothing and
    the caller forwards the original text unchanged.
    """
    if not content:
        return None

    # Guard 2: remove code spans before scanning so documented syntax is safe.
    scan_text = _CODE_FENCE_RE.sub("", content)
    scan_text = _INLINE_CODE_RE.sub("", scan_text)

    matches = _INVOKE_RE.findall(scan_text)
    if not matches:
        return None

    recovered: list[dict] = []
    for name, body in matches:
        name = name.strip()
        if not name:
            continue
        # Guard 3: only honour calls to tools the client actually offered.
        if allowed_names is not None and name not in allowed_names:
            logger.warning(
                "Skipping leaked <invoke> for unknown tool %r (not in request tools)",
                name,
            )
            continue

        params: dict[str, str] = {}
        for param_name, param_value in _INVOKE_PARAM_RE.findall(body):
            params[param_name.strip()] = param_value

        recovered.append(
            {
                "id": f"toolu_{uuid4().hex[:24]}",
                "type": "function",
                "index": len(recovered),
                "function": {
                    "name": name,
                    "arguments": json.dumps(params),
                },
            }
        )

    if not recovered:
        return None

    logger.info("Recovered %d leaked <invoke> tool call(s) from streamed content", len(recovered))
    return recovered


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
