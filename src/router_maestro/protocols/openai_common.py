"""Shared strict helpers for OpenAI Chat and Responses protocol runtimes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Never

from router_maestro.protocols.models import (
    FrozenJsonValue,
    SemanticEvent,
    TerminalMetadata,
    ToolChoice,
    Usage,
    UsageMode,
    WireProtocol,
)
from router_maestro.protocols.runtime import (
    ProtocolDecodeError,
    ProtocolRepresentabilityError,
)
from router_maestro.providers.base import (
    ResponseStatus,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
)


def reject(protocol: WireProtocol, parameter: str, reason: str) -> Never:
    raise ProtocolRepresentabilityError(protocol, parameter, reason)


def decode_reject(protocol: WireProtocol, parameter: str, reason: str) -> Never:
    raise ProtocolDecodeError(protocol, parameter, reason)


def require_mapping(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        decode_reject(protocol, parameter, "must be an object")
    for key in value:
        if not isinstance(key, str):
            decode_reject(protocol, parameter, "object keys must be strings")
    return value


def require_list(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        decode_reject(protocol, parameter, "must be an array")
    return value


def require_string(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
    allow_empty: bool = True,
) -> str:
    if not isinstance(value, str):
        decode_reject(protocol, parameter, "must be a string")
    if not allow_empty and not value:
        decode_reject(protocol, parameter, "must be a non-empty string")
    return value


def optional_string(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> str | None:
    if value is None:
        return None
    return require_string(value, protocol=protocol, parameter=parameter)


def optional_bool(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        decode_reject(protocol, parameter, "must be a boolean")
    return value


def optional_int(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        decode_reject(protocol, parameter, "must be an integer")
    return value


def optional_number(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> int | float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        decode_reject(protocol, parameter, "must be a number")
    return value


def reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    protocol: WireProtocol,
    parameter: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        suffix = f".{unknown[0]}" if parameter else unknown[0]
        decode_reject(protocol, f"{parameter}{suffix}", "field is not representable")


def thaw_json(value: FrozenJsonValue | object) -> Any:
    """Return ordinary JSON containers from immutable semantic JSON values."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [thaw_json(item) for item in value]
    return value


def parse_arguments(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str,
) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            decode_reject(protocol, parameter, f"must contain valid JSON ({error.msg})")
    return require_mapping(value, protocol=protocol, parameter=parameter)


def encode_arguments(arguments: Mapping[str, FrozenJsonValue]) -> str:
    return json.dumps(thaw_json(arguments), separators=(",", ":"), ensure_ascii=False)


def decode_tool_choice(
    value: object,
    *,
    protocol: WireProtocol,
    parameter: str = "tool_choice",
    nested_function: bool,
) -> ToolChoice | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            decode_reject(protocol, parameter, f"unsupported mode {value!r}")
        return ToolChoice(mode=value)

    choice = require_mapping(value, protocol=protocol, parameter=parameter)
    reject_unknown_keys(
        choice,
        frozenset({"type", "function"} if nested_function else {"type", "name", "namespace"}),
        protocol=protocol,
        parameter=parameter,
    )
    choice_type = require_string(
        choice.get("type"),
        protocol=protocol,
        parameter=f"{parameter}.type",
    )
    if choice_type != "function":
        decode_reject(
            protocol,
            f"{parameter}.type",
            "only function tool choice is supported",
        )
    if nested_function:
        function = require_mapping(
            choice.get("function"),
            protocol=protocol,
            parameter=f"{parameter}.function",
        )
        reject_unknown_keys(
            function,
            frozenset({"name"}),
            protocol=protocol,
            parameter=f"{parameter}.function",
        )
        name = require_string(
            function.get("name"),
            protocol=protocol,
            parameter=f"{parameter}.function.name",
            allow_empty=False,
        )
    else:
        name = require_string(
            choice.get("name"),
            protocol=protocol,
            parameter=f"{parameter}.name",
            allow_empty=False,
        )
    namespace = (
        optional_string(
            choice.get("namespace"),
            protocol=protocol,
            parameter=f"{parameter}.namespace",
        )
        if not nested_function
        else None
    )
    return ToolChoice(mode="function", name=name, namespace=namespace)


def encode_tool_choice(
    choice: ToolChoice | None,
    *,
    protocol: WireProtocol,
    nested_function: bool,
) -> str | dict[str, Any] | None:
    if choice is None:
        return None
    if choice.mode in {"auto", "none", "required"}:
        if choice.name is not None or choice.namespace is not None:
            reject(protocol, "tool_choice.name", f"mode {choice.mode!r} cannot select a tool")
        return choice.mode
    if choice.mode != "function" or not choice.name:
        reject(protocol, "tool_choice.mode", f"unsupported mode {choice.mode!r}")
    if nested_function:
        if choice.namespace is not None:
            reject(protocol, "tool_choice.namespace", "nested function choice lacks namespace")
        return {"type": "function", "function": {"name": choice.name}}
    payload = {"type": "function", "name": choice.name}
    if choice.namespace is not None:
        payload["namespace"] = choice.namespace
    return payload


def decode_usage(
    value: object,
    *,
    protocol: WireProtocol,
    input_field: str,
    output_field: str,
    input_details_field: str,
    output_details_field: str,
    top_level_reasoning_field: str | None = None,
) -> Usage | None:
    if value is None:
        return None
    usage = require_mapping(value, protocol=protocol, parameter="usage")
    allowed = {
        input_field,
        output_field,
        "total_tokens",
        input_details_field,
        output_details_field,
    }
    if top_level_reasoning_field is not None:
        allowed.add(top_level_reasoning_field)
    reject_unknown_keys(
        usage,
        frozenset(allowed),
        protocol=protocol,
        parameter="usage",
    )
    input_details = usage.get(input_details_field)
    output_details = usage.get(output_details_field)
    cached_tokens = None
    reasoning_tokens = (
        optional_int(
            usage.get(top_level_reasoning_field),
            protocol=protocol,
            parameter=f"usage.{top_level_reasoning_field}",
        )
        if top_level_reasoning_field is not None
        else None
    )
    if input_details is not None:
        details = require_mapping(
            input_details,
            protocol=protocol,
            parameter=f"usage.{input_details_field}",
        )
        reject_unknown_keys(
            details,
            frozenset({"cached_tokens", "audio_tokens"}),
            protocol=protocol,
            parameter=f"usage.{input_details_field}",
        )
        cached_tokens = optional_int(
            details.get("cached_tokens"),
            protocol=protocol,
            parameter=f"usage.{input_details_field}.cached_tokens",
        )
    if output_details is not None:
        details = require_mapping(
            output_details,
            protocol=protocol,
            parameter=f"usage.{output_details_field}",
        )
        reject_unknown_keys(
            details,
            frozenset(
                {
                    "reasoning_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                    "audio_tokens",
                }
            ),
            protocol=protocol,
            parameter=f"usage.{output_details_field}",
        )
        nested_reasoning_tokens = optional_int(
            details.get("reasoning_tokens"),
            protocol=protocol,
            parameter=f"usage.{output_details_field}.reasoning_tokens",
        )
        if (
            reasoning_tokens is not None
            and nested_reasoning_tokens is not None
            and reasoning_tokens != nested_reasoning_tokens
        ):
            decode_reject(
                protocol,
                f"usage.{output_details_field}.reasoning_tokens",
                f"conflicts with usage.{top_level_reasoning_field}",
            )
        if nested_reasoning_tokens is not None:
            reasoning_tokens = nested_reasoning_tokens
    return Usage(
        input_tokens=optional_int(
            usage.get(input_field),
            protocol=protocol,
            parameter=f"usage.{input_field}",
        ),
        output_tokens=optional_int(
            usage.get(output_field),
            protocol=protocol,
            parameter=f"usage.{output_field}",
        ),
        total_tokens=optional_int(
            usage.get("total_tokens"),
            protocol=protocol,
            parameter="usage.total_tokens",
        ),
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def encode_usage(
    usage: Usage | None,
    *,
    protocol: WireProtocol,
    input_field: str,
    output_field: str,
    input_details_field: str,
    output_details_field: str,
) -> dict[str, Any] | None:
    if usage is None:
        return None
    if usage.mode is not UsageMode.SNAPSHOT:
        reject(protocol, "usage.mode", "non-stream responses require snapshot usage")
    required = {
        input_field: usage.input_tokens,
        output_field: usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    missing = next((field for field, value in required.items() if value is None), None)
    if missing is not None:
        reject(protocol, f"usage.{missing}", "is required for non-stream usage")
    payload: dict[str, Any] = dict(required)
    if usage.cached_input_tokens is not None:
        payload[input_details_field] = {"cached_tokens": usage.cached_input_tokens}
    if usage.reasoning_tokens is not None:
        payload[output_details_field] = {"reasoning_tokens": usage.reasoning_tokens}
    return payload


def terminal_event_values(
    outcome: TerminalOutcome,
) -> tuple[TerminalMetadata, dict[str, Any]]:
    """Preserve a provider terminal outcome in semantic event fields."""
    error = outcome.error
    terminal = TerminalMetadata(
        finish_reason=outcome.finish_reason,
        error_code=error.code if error is not None else None,
        error_message=error.message if error is not None else None,
        response_status=outcome.response_status.value,
        transport_termination=outcome.transport.value,
        incomplete_details=outcome.incomplete_details,
    )
    metadata: dict[str, Any] = {"transport_termination": outcome.transport.value}
    if outcome.incomplete_details is not None:
        metadata["incomplete_details"] = outcome.incomplete_details
    return terminal, metadata


def terminal_outcome_from_event(
    event: SemanticEvent,
    *,
    protocol: WireProtocol,
) -> TerminalOutcome:
    """Rebuild the provider terminal type without conflating transport/status."""
    terminal = event.terminal
    if terminal is None:
        reject(protocol, "event.terminal", "terminal event requires metadata")
    transport_value = terminal.transport_termination
    if transport_value is None:
        transport_value = event.metadata.get(
            "transport_termination", TransportTermination.EXPLICIT_TERMINAL.value
        )
    status_value = terminal.response_status
    if status_value is None:
        status_value = (
            ResponseStatus.INCOMPLETE.value
            if terminal.finish_reason in {"length", "content_filter"}
            else ResponseStatus.COMPLETED.value
        )
    try:
        transport = TransportTermination(transport_value)
    except (TypeError, ValueError):
        reject(protocol, "event.metadata.transport_termination", "unsupported value")
    try:
        status = ResponseStatus(status_value)
    except ValueError:
        reject(protocol, "event.terminal.response_status", "unsupported value")
    error = None
    if terminal.error_code is not None or terminal.error_message is not None:
        error = TerminalError(
            code=terminal.error_code or "stream_error",
            message=terminal.error_message or "stream failed",
        )
    incomplete_details = terminal.incomplete_details
    if incomplete_details is None:
        incomplete_details = event.metadata.get("incomplete_details")
    if incomplete_details is not None and not isinstance(incomplete_details, Mapping):
        reject(protocol, "event.metadata.incomplete_details", "must be an object")
    return TerminalOutcome(
        transport=transport,
        response_status=status,
        finish_reason=terminal.finish_reason,
        incomplete_details=(
            thaw_json(incomplete_details) if incomplete_details is not None else None
        ),
        error=error,
    )


def encode_stream_usage(
    usage: Usage,
    *,
    protocol: WireProtocol,
    input_field: str,
    output_field: str,
    input_details_field: str,
    output_details_field: str,
) -> dict[str, Any]:
    """Encode snapshot usage without inventing missing stream counters."""
    if usage.mode is not UsageMode.SNAPSHOT:
        reject(protocol, "event.usage.mode", "OpenAI stream usage is a snapshot")
    payload: dict[str, Any] = {
        name: value
        for name, value in (
            (input_field, usage.input_tokens),
            (output_field, usage.output_tokens),
            ("total_tokens", usage.total_tokens),
        )
        if value is not None
    }
    if usage.cached_input_tokens is not None:
        payload[input_details_field] = {"cached_tokens": usage.cached_input_tokens}
    if usage.reasoning_tokens is not None:
        payload[output_details_field] = {"reasoning_tokens": usage.reasoning_tokens}
    return payload
