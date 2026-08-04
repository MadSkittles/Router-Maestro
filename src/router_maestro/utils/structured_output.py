"""Translation of Anthropic ``output_config.format`` to OpenAI ``response_format``.

Anthropic and OpenAI both express structured outputs as a JSON Schema, but nest
it differently. Router-Maestro carries the schema on the typed
``ChatRequest.output_format`` field so no adapter can silently discard it; this
module is the single place that reshapes it for OpenAI-style chat wires.

Forwarding is best-effort by design. Verified live against GitHub Copilot's
``/chat/completions``: the translated payload is accepted (200) for Claude and
Gemini models alike, though only the native Anthropic transport enforces the
schema. Rejecting instead would fail requests that succeed today, so the
translated path forwards and lets upstream decide.
"""

from __future__ import annotations

from typing import Any

# Anthropic's ``format`` object carries the schema inline and needs no name;
# OpenAI's ``json_schema`` wrapper requires one, so supply a stable default.
_DEFAULT_SCHEMA_NAME = "response"


def output_format_to_response_format(output_format: Any) -> dict | None:
    """Reshape an Anthropic output format into an OpenAI ``response_format``.

    Returns ``None`` when there is nothing faithful to send, so callers omit the
    key rather than forwarding a malformed one.
    """
    if not isinstance(output_format, dict):
        return None

    if output_format.get("type") != "json_schema":
        # Unknown format types are not guessed at; upstream owns that judgment
        # on the native path, and the translated path simply omits them.
        return None

    schema = output_format.get("schema")
    if not isinstance(schema, dict):
        return None

    json_schema: dict[str, Any] = {
        "name": output_format.get("name") or _DEFAULT_SCHEMA_NAME,
        "schema": schema,
    }
    if "strict" in output_format:
        json_schema["strict"] = output_format["strict"]

    return {"type": "json_schema", "json_schema": json_schema}
