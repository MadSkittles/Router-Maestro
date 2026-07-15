"""Pure payload and response codecs for the Copilot Responses API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from router_maestro.providers.base import (
    BaseProvider,
    RequestOptionError,
    ResponsesRequest,
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
    resolve_terminal_outcome,
)
from router_maestro.utils import get_logger

logger = get_logger("providers.copilot.responses_codec")

_BENIGN_UPSTREAM_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
    }
)
_BENIGN_DONE_ITEM_TYPES = frozenset({"message"})


class CopilotResponsesCodec:
    """Encode Responses requests and decode validated Responses output."""

    name = "github-copilot"

    unsupported_tool_types = frozenset({"web_search", "web_search_preview", "code_interpreter"})

    @staticmethod
    def input_has_vision(value: Any, depth: int = 0) -> bool:
        if depth > 32 or value is None:
            return False
        if isinstance(value, list):
            return any(CopilotResponsesCodec.input_has_vision(item, depth + 1) for item in value)
        if not isinstance(value, dict):
            return False
        item_type = value.get("type")
        if isinstance(item_type, str) and item_type.lower() in ("input_image", "image_url"):
            return True
        if "image_url" in value:
            return True
        content = value.get("content")
        if isinstance(content, list):
            return any(CopilotResponsesCodec.input_has_vision(item, depth + 1) for item in content)
        return False

    def filter_unsupported_tools(
        self,
        tools: list[dict] | None,
        *,
        provider_name: str,
        model: str | None = None,
    ) -> list[dict] | None:
        if not tools:
            return None
        validated = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                validated.append(tool)
            elif tool_type == "namespace":
                inner = tool.get("tools")
                if isinstance(inner, list) and inner:
                    validated.append(tool)
                else:
                    raise RequestOptionError(
                        "GitHub Copilot requires namespace tools to contain a non-empty tools list",
                        provider=provider_name,
                        model=model,
                        parameter="tools",
                    )
            elif tool_type not in self.unsupported_tool_types:
                validated.append(tool)
            else:
                raise RequestOptionError(
                    f"GitHub Copilot does not support Responses tool type '{tool_type}'",
                    provider=provider_name,
                    model=model,
                    parameter="tools",
                )
        return validated or None

    def build_payload(
        self,
        request: ResponsesRequest,
        *,
        provider_name: str,
        validate_extensions: Callable[[ResponsesRequest], None],
        catalog_effort_values: list[str] | None,
        resolve_reasoning: Callable[..., Any],
    ) -> dict:
        validate_extensions(request)
        if request.temperature is not None:
            raise RequestOptionError(
                "GitHub Copilot Responses does not support request option 'temperature'",
                provider=provider_name,
                model=request.model,
                parameter="temperature",
            )
        payload: dict = {
            "model": request.model,
            "input": request.input,
            "stream": request.stream,
        }
        if request.instructions:
            payload["instructions"] = request.instructions
        if request.max_output_tokens:
            payload["max_output_tokens"] = request.max_output_tokens
        filtered_tools = self.filter_unsupported_tools(
            request.tools,
            provider_name=provider_name,
            model=request.model,
        )
        if filtered_tools:
            payload["tools"] = filtered_tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        for key, value in {
            "top_p": request.top_p,
            "metadata": request.metadata,
            "service_tier": request.service_tier,
        }.items():
            if value is not None:
                payload[key] = value

        from router_maestro.routing.capabilities import Operation

        resolution = resolve_reasoning(
            model=request.model,
            reasoning_effort=request.reasoning_effort,
            thinking_budget=None,
            catalog_effort_values=catalog_effort_values,
            operation=Operation.RESPONSES,
        )
        if resolution.effort is not None:
            payload["reasoning"] = {"effort": resolution.effort, "summary": "auto"}
            payload["include"] = ["reasoning.encrypted_content"]
        return payload

    @staticmethod
    def extract_response_content(data: dict) -> tuple[str, str | None]:
        content = ""
        refusal_parts: list[str] = []
        for output in data.get("output", []):
            if output.get("type") == "message":
                for content_item in output.get("content", []):
                    if content_item.get("type") == "output_text":
                        content += content_item.get("text", "")
                    elif content_item.get("type") == "refusal":
                        refusal_parts.append(content_item.get("refusal", ""))
        return content, "".join(refusal_parts) or None

    @staticmethod
    def extract_reasoning(data: dict) -> tuple[str | None, str | None, str | None]:
        output = next(
            (item for item in data.get("output", []) if item.get("type") == "reasoning"),
            None,
        )
        if output is None:
            return None, None, None
        text = "".join(summary.get("text") or "" for summary in output.get("summary", []))
        return text or None, output.get("id"), output.get("encrypted_content")

    @staticmethod
    def extract_tool_calls(data: dict) -> list[ResponsesToolCall]:
        tool_calls = []
        for output in data.get("output", []):
            if output.get("type") == "function_call":
                tool_calls.append(
                    ResponsesToolCall(
                        call_id=output.get("call_id", ""),
                        name=output.get("name", ""),
                        arguments=output.get("arguments", "{}"),
                        namespace=output.get("namespace"),
                    )
                )
            elif output.get("type") == "custom_tool_call":
                tool_calls.append(
                    ResponsesToolCall(
                        call_id=output.get("call_id", ""),
                        name=output.get("name", ""),
                        arguments=output.get("input", ""),
                        kind="custom",
                    )
                )
            elif output.get("type") == "tool_search_call":
                arguments = output.get("arguments")
                arguments_string = (
                    arguments
                    if isinstance(arguments, str)
                    else "{}"
                    if arguments is None
                    else json.dumps(arguments)
                )
                tool_calls.append(
                    ResponsesToolCall(
                        call_id=output.get("call_id", ""),
                        name="tool_search",
                        arguments=arguments_string,
                        kind="tool_search",
                    )
                )
        return tool_calls

    @staticmethod
    def validate_output(data: dict) -> bool:
        output = data.get("output")
        if not isinstance(output, list):
            raise TypeError("Responses output must be a list")
        has_deliverable = False
        reasoning_item_count = 0
        for item in output:
            if not isinstance(item, dict):
                raise TypeError("Responses output item must be an object")
            item_type = item.get("type")
            if not isinstance(item_type, str):
                raise TypeError("Responses output item type must be a string")
            if item_type == "message":
                content = item.get("content")
                if not isinstance(content, list):
                    raise TypeError("Responses message content must be a list")
                for part in content:
                    if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                        raise TypeError("Responses message content part must be typed object")
                    if part.get("type") == "output_text" and not isinstance(part.get("text"), str):
                        raise TypeError("Responses output_text part must contain text")
                    if part.get("type") == "output_text" and part.get("text"):
                        has_deliverable = True
                    if part.get("type") == "refusal":
                        refusal = part.get("refusal")
                        if not isinstance(refusal, str) or not refusal:
                            raise TypeError(
                                "Responses refusal part must contain non-empty refusal text"
                            )
                        has_deliverable = True
            elif item_type in {"function_call", "custom_tool_call", "tool_search_call"}:
                required_fields = {
                    "function_call": ("call_id", "name", "arguments"),
                    "custom_tool_call": ("call_id", "name", "input"),
                    "tool_search_call": ("call_id",),
                }[item_type]
                if any(not isinstance(item.get(field), str) for field in required_fields):
                    raise TypeError(f"Responses {item_type} item is malformed")
                has_deliverable = True
            elif item_type == "reasoning":
                reasoning_item_count += 1
                if reasoning_item_count > 1:
                    raise TypeError("Responses response contains multiple atomic reasoning items")
                summary = item.get("summary", [])
                if not isinstance(summary, list):
                    raise TypeError("Responses reasoning summary must be a list")
                for entry in summary:
                    if not isinstance(entry, dict):
                        raise TypeError("Responses reasoning summary entry must be an object")
                    if "type" in entry and not isinstance(entry["type"], str):
                        raise TypeError("Responses reasoning summary type must be a string")
                    if "text" in entry and not isinstance(entry["text"], str):
                        raise TypeError("Responses reasoning summary text must be a string")
                for field in ("id", "encrypted_content"):
                    value = item.get(field)
                    if value is not None and not isinstance(value, str):
                        raise TypeError(f"Responses reasoning {field} must be a string or null")
                if item.get("encrypted_content") and not item.get("id"):
                    raise TypeError("Responses encrypted reasoning is missing its upstream id")
                if any(
                    isinstance(entry.get("text"), str) and bool(entry["text"]) for entry in summary
                ) or bool(item.get("encrypted_content")):
                    has_deliverable = True
        return has_deliverable

    @staticmethod
    def validate_usage(usage: object) -> None:
        BaseProvider._validated_token_usage(
            usage,
            fields=("input_tokens", "output_tokens", "total_tokens"),
            label="Responses",
            detail_fields={
                "input_tokens_details": ("cached_tokens",),
                "output_tokens_details": ("reasoning_tokens",),
            },
        )

    @staticmethod
    def terminal_outcome(response: Any) -> TerminalOutcome:
        def protocol_error(message: str) -> TerminalOutcome:
            return TerminalOutcome(
                transport=TransportTermination.EXCEPTION,
                response_status=ResponseStatus.FAILED,
                error=TerminalError(code="upstream_protocol_error", message=message),
            )

        if not isinstance(response, dict):
            return protocol_error("Copilot Responses terminal response must be an object")
        raw_status = response.get("status")
        if not isinstance(raw_status, str):
            return protocol_error("Copilot Responses status must be a string terminal value")
        try:
            status = ResponseStatus(raw_status)
        except ValueError:
            return protocol_error("Copilot Responses status is not a recognized terminal value")
        raw_error = response.get("error")
        if raw_error is not None and not isinstance(raw_error, dict):
            return protocol_error("Copilot Responses error must be an object or null")
        error = None
        if isinstance(raw_error, dict):
            code = raw_error.get("code")
            message = raw_error.get("message")
            if code is not None and not isinstance(code, str):
                return protocol_error("Copilot Responses error code must be a string or null")
            if message is not None and not isinstance(message, str):
                return protocol_error("Copilot Responses error message must be a string or null")
            error = TerminalError(
                code=code or "upstream_error",
                message=message or "Upstream response failed",
            )
        details = response.get("incomplete_details")
        if details is not None and not isinstance(details, dict):
            return protocol_error("Copilot Responses incomplete_details must be an object or null")
        if isinstance(details, dict):
            reason = details.get("reason")
            if reason is not None and not isinstance(reason, str):
                return protocol_error(
                    "Copilot Responses incomplete reason must be a string or null"
                )
        outcome = TerminalOutcome(
            transport=TransportTermination.EXPLICIT_TERMINAL,
            response_status=status,
            incomplete_details=details,
            error=error,
        )
        return resolve_terminal_outcome(outcome, None) or protocol_error(
            "Copilot Responses terminal payload has no outcome"
        )

    @staticmethod
    def _raise_protocol_error(
        provider: str,
        model: str | None,
        cause: BaseException,
    ) -> None:
        BaseProvider._raise_protocol_error(provider, model, cause)

    async def decode_stream(
        self,
        lines: AsyncIterator[str],
        request: ResponsesRequest,
    ) -> AsyncIterator[ResponsesStreamChunk]:
        # Reduce Copilot Responses SSE events into stable stream chunks.
        stream_finished = False
        final_usage = None
        emitted_tool_call = False
        text_parts: dict[tuple[int, int], str] = {}
        completed_text_parts: dict[tuple[int, int], str] = {}
        refusal_parts: dict[tuple[int, int], str] = {}
        completed_refusal_parts: dict[tuple[int, int], str] = {}
        reasoning_parts: dict[tuple[int, int], str] = {}
        completed_reasoning_parts: dict[tuple[int, int], str] = {}
        reasoning_item_ids: dict[int, str] = {}
        completed_reasoning_items: dict[int, tuple[tuple[str, ...], str | None]] = {}
        declared_output_item_types: dict[int, str] = {}
        completed_output_items: dict[int, tuple[str, str]] = {}
        emitted_reasoning_signatures: set[int] = set()
        # Track pending function calls being streamed, keyed by output_index
        # (Copilot obfuscates item IDs differently across events, so we can't match by ID)
        pending_fcs: dict[int, dict] = {}
        # Diagnostic: count any event types we don't explicitly handle
        # so we can spot custom_tool_call_input.* or other channels we
        # might be dropping.
        unknown_event_counts: dict[str, int] = {}

        def bind_declared_type(output_index: int, item_type: str) -> None:
            declared_type = declared_output_item_types.get(output_index)
            if declared_type is not None and declared_type != item_type:
                self._raise_protocol_error(
                    self.name,
                    request.model,
                    TypeError("Responses output index changed item type"),
                )
            declared_output_item_types[output_index] = item_type

        async for line in lines:
            if stream_finished:
                break

            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                # ``[DONE]`` terminates the SSE transport but carries no
                # Responses semantic status. The route detects an
                # unexpected EOF unless response.done/completed already
                # produced an explicit terminal chunk.
                break

            try:
                data = json.loads(data_str)
                if not isinstance(data, dict):
                    raise TypeError("Responses stream event must be an object")
                event_type = data.get("type", "")
                if not isinstance(event_type, str):
                    raise TypeError("Responses stream event type must be a string")
                if event_type in {
                    "response.output_item.added",
                    "response.output_item.done",
                }:
                    item = data.get("item")
                    if not isinstance(item, dict):
                        raise TypeError("Responses stream output item must be an object")
                    item_type = item.get("type")
                    if not isinstance(item_type, str):
                        raise TypeError("Responses output item type must be a string")
                    output_index = data.get("output_index", 0)
                    if not isinstance(output_index, int) or isinstance(output_index, bool):
                        raise TypeError("Responses output_index must be an integer")
                    if item_type in {"function_call", "custom_tool_call"}:
                        for field in ("call_id", "name"):
                            if not isinstance(item.get(field), str):
                                raise TypeError(f"Responses {item_type} {field} must be a string")
                    if item_type == "function_call":
                        namespace = item.get("namespace")
                        if namespace is not None and not isinstance(namespace, str):
                            raise TypeError(
                                "Responses function call namespace must be a string or null"
                            )
                    if item_type == "tool_search_call" and not isinstance(item.get("call_id"), str):
                        raise TypeError("Responses tool_search_call call_id must be string")
                    if item_type == "reasoning":
                        summary = item.get("summary", []) or []
                        if not isinstance(summary, list):
                            raise TypeError("Responses reasoning summary must be a list")
                        for entry in summary:
                            if not isinstance(entry, dict):
                                raise TypeError("Responses reasoning summary entry malformed")
                            text = entry.get("text")
                            if text is not None and not isinstance(text, str):
                                raise TypeError("Responses reasoning text must be string")
                        for field in ("id", "encrypted_content"):
                            value = item.get(field)
                            if value is not None and not isinstance(value, str):
                                raise TypeError(f"Responses reasoning {field} must be a string")
                    if item_type == "message":
                        content = item.get("content", [])
                        if not isinstance(content, list):
                            raise TypeError("Responses message content must be a list")
                        for part in content:
                            if not isinstance(part, dict):
                                raise TypeError("Responses message content part must be an object")
                            part_type = part.get("type")
                            if part_type not in {"output_text", "refusal"}:
                                raise TypeError(
                                    "Responses message content part type is unsupported"
                                )
                            if part_type == "output_text" and not isinstance(part.get("text"), str):
                                raise TypeError("Responses message output text must be a string")
                            if part_type == "refusal" and not isinstance(part.get("refusal"), str):
                                raise TypeError("Responses message refusal must be a string")
                if event_type in {
                    "response.done",
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                    "response.cancelled",
                }:
                    terminal_response = data.get("response")
                    if not isinstance(terminal_response, dict):
                        raise TypeError("Responses terminal response must be an object")
                    self.validate_usage(terminal_response.get("usage"))
                if event_type in {
                    "response.output_text.delta",
                    "response.reasoning_summary_text.delta",
                    "response.refusal.delta",
                } and not isinstance(data.get("delta"), str):
                    raise TypeError("Responses text delta must be a string")
                if event_type == "response.output_text.done" and not isinstance(
                    data.get("text"), str
                ):
                    raise TypeError("Responses output text must be a string")
                if event_type in {
                    "response.output_text.delta",
                    "response.output_text.done",
                }:
                    for field in ("output_index", "content_index"):
                        value = data.get(field, 0)
                        if not isinstance(value, int) or isinstance(value, bool):
                            raise TypeError(f"Responses output text {field} must be an integer")
                if event_type == "response.refusal.done" and not isinstance(
                    data.get("refusal"), str
                ):
                    raise TypeError("Responses refusal must be a string")
                if event_type in {
                    "response.refusal.delta",
                    "response.refusal.done",
                }:
                    for field in ("output_index", "content_index"):
                        value = data.get(field, 0)
                        if not isinstance(value, int) or isinstance(value, bool):
                            raise TypeError(f"Responses refusal {field} must be an integer")
                if event_type == "response.reasoning_summary_text.done" and not isinstance(
                    data.get("text"), str
                ):
                    raise TypeError("Responses reasoning done text must be a string")
                if event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_summary_text.done",
                }:
                    item_id = data.get("item_id")
                    if item_id is not None and not isinstance(item_id, str):
                        raise TypeError("Responses reasoning item_id must be a string or null")
                    for field in ("output_index", "summary_index"):
                        value = data.get(field, 0)
                        if not isinstance(value, int) or isinstance(value, bool):
                            raise TypeError(f"Responses reasoning {field} must be an integer")
                if event_type in {
                    "response.reasoning_summary_part.added",
                    "response.reasoning_summary_part.done",
                }:
                    item_id = data.get("item_id")
                    if item_id is not None and not isinstance(item_id, str):
                        raise TypeError("Responses reasoning item_id must be a string or null")
                    for field in ("output_index", "summary_index"):
                        value = data.get(field, 0)
                        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                            raise TypeError(
                                f"Responses reasoning {field} must be a non-negative integer"
                            )
                    part = data.get("part")
                    if not isinstance(part, dict):
                        raise TypeError("Responses reasoning summary part must be an object")
                    if part.get("type") != "summary_text":
                        raise TypeError(
                            "Responses reasoning summary part type must be summary_text"
                        )
                    if not isinstance(part.get("text"), str):
                        raise TypeError("Responses reasoning summary part text must be a string")
                if event_type in {
                    "response.function_call_arguments.delta",
                    "response.custom_tool_call_input.delta",
                } and not isinstance(data.get("delta"), str):
                    raise TypeError("Responses tool delta must be a string")
                if event_type == "response.function_call_arguments.done" and not isinstance(
                    data.get("arguments"), str
                ):
                    raise TypeError("Responses function arguments must be a string")
                if event_type == "response.custom_tool_call_input.done" and not isinstance(
                    data.get("input"), str
                ):
                    raise TypeError("Responses custom tool input must be a string")
                if event_type in {
                    "response.function_call_arguments.delta",
                    "response.function_call_arguments.done",
                    "response.custom_tool_call_input.delta",
                    "response.custom_tool_call_input.done",
                }:
                    output_index = data.get("output_index", 0)
                    if not isinstance(output_index, int) or isinstance(output_index, bool):
                        raise TypeError("Responses output_index must be an integer")
                if event_type == "response.output_item.done":
                    item = data["item"]
                    item_type = item.get("type")
                    if item_type == "function_call":
                        for field in ("call_id", "name", "arguments"):
                            if not isinstance(item.get(field), str):
                                raise TypeError(f"Responses function call {field} must be a string")
                    elif item_type == "custom_tool_call" and not isinstance(item.get("input"), str):
                        raise TypeError("Responses custom tool input must be a string")
            except (json.JSONDecodeError, TypeError) as e:
                self._raise_protocol_error(self.name, request.model, e)

            typed_item_type = {
                "response.output_text.delta": "message",
                "response.output_text.done": "message",
                "response.refusal.delta": "message",
                "response.refusal.done": "message",
                "response.reasoning_summary_text.delta": "reasoning",
                "response.reasoning_summary_text.done": "reasoning",
                "response.reasoning_summary_part.added": "reasoning",
                "response.reasoning_summary_part.done": "reasoning",
                "response.function_call_arguments.delta": "function_call",
                "response.function_call_arguments.done": "function_call",
                "response.custom_tool_call_input.delta": "custom_tool_call",
                "response.custom_tool_call_input.done": "custom_tool_call",
            }.get(event_type)
            if typed_item_type is not None:
                bind_declared_type(data.get("output_index", 0), typed_item_type)

            # Handle text delta events
            if event_type == "response.output_text.delta":
                delta_text = data.get("delta", "")
                text_part = (
                    data.get("output_index", 0),
                    data.get("content_index", 0),
                )
                if text_part in completed_text_parts:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses output text delta followed a completed part"),
                    )
                if delta_text:
                    text_parts[text_part] = text_parts.get(text_part, "") + delta_text
                    yield ResponsesStreamChunk(
                        content=delta_text,
                        output_index=text_part[0],
                        content_index=text_part[1],
                        output_item_type="message",
                    )
                else:
                    yield ResponsesStreamChunk(
                        content="",
                        output_index=text_part[0],
                        content_index=text_part[1],
                        provenance_only=True,
                        output_item_type="message",
                    )

            elif event_type == "response.output_text.done":
                snapshot = data.get("text", "")
                text_part = (
                    data.get("output_index", 0),
                    data.get("content_index", 0),
                )
                completed = completed_text_parts.get(text_part)
                if completed is not None:
                    if completed != snapshot:
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError(
                                "Responses output text part completed with conflicting snapshots"
                            ),
                        )
                    continue
                accumulated = text_parts.get(text_part, "")
                if not snapshot.startswith(accumulated):
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses output text done snapshot conflicts with its deltas"),
                    )
                completed_text_parts[text_part] = snapshot
                suffix = snapshot[len(accumulated) :]
                if suffix:
                    text_parts[text_part] = snapshot
                    yield ResponsesStreamChunk(
                        content=suffix,
                        output_index=text_part[0],
                        content_index=text_part[1],
                        output_item_type="message",
                    )
                elif not accumulated:
                    yield ResponsesStreamChunk(
                        content="",
                        output_index=text_part[0],
                        content_index=text_part[1],
                        provenance_only=True,
                        output_item_type="message",
                    )

            elif event_type == "response.refusal.delta":
                refusal_delta = data.get("delta", "")
                refusal_part = (
                    data.get("output_index", 0),
                    data.get("content_index", 0),
                )
                if refusal_part in completed_refusal_parts:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses refusal delta followed a completed part"),
                    )
                if refusal_delta:
                    refusal_parts[refusal_part] = (
                        refusal_parts.get(refusal_part, "") + refusal_delta
                    )
                    yield ResponsesStreamChunk(
                        content="",
                        refusal=refusal_delta,
                        output_index=refusal_part[0],
                        content_index=refusal_part[1],
                        output_item_type="message",
                    )
                else:
                    yield ResponsesStreamChunk(
                        content="",
                        output_index=refusal_part[0],
                        content_index=refusal_part[1],
                        provenance_only=True,
                        output_item_type="message",
                    )

            elif event_type == "response.refusal.done":
                snapshot = data.get("refusal", "")
                refusal_part = (
                    data.get("output_index", 0),
                    data.get("content_index", 0),
                )
                if refusal_part in completed_refusal_parts:
                    if completed_refusal_parts[refusal_part] != snapshot:
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError(
                                "Responses refusal part completed with conflicting snapshots"
                            ),
                        )
                    continue
                accumulated = refusal_parts.get(refusal_part, "")
                if not snapshot.startswith(accumulated):
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses refusal done snapshot conflicts with its deltas"),
                    )
                completed_refusal_parts[refusal_part] = snapshot
                suffix = snapshot[len(accumulated) :]
                if suffix:
                    refusal_parts[refusal_part] = snapshot
                    yield ResponsesStreamChunk(
                        content="",
                        refusal=suffix,
                        output_index=refusal_part[0],
                        content_index=refusal_part[1],
                        output_item_type="message",
                    )
                elif not accumulated:
                    yield ResponsesStreamChunk(
                        content="",
                        output_index=refusal_part[0],
                        content_index=refusal_part[1],
                        provenance_only=True,
                        output_item_type="message",
                    )

            # Reasoning summary (chain-of-thought) deltas — surfaced so
            # entry routes (Anthropic, Gemini) can forward them as
            # thinking blocks. Copilot may obfuscate ``item_id``
            # independently on every summary event, so only
            # ``output_index`` correlates these deltas. The canonical
            # identity arrives with ``output_item.done`` and its
            # encrypted blob.
            elif event_type == "response.reasoning_summary_text.delta":
                delta = data.get("delta", "")
                output_index = data.get("output_index", 0)
                reasoning_part = (output_index, data.get("summary_index", 0))
                if output_index in completed_reasoning_items:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses reasoning delta followed a completed item"),
                    )
                if reasoning_part in completed_reasoning_parts:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses reasoning delta followed a completed summary part"),
                    )
                if delta:
                    reasoning_parts[reasoning_part] = (
                        reasoning_parts.get(reasoning_part, "") + delta
                    )
                    yield ResponsesStreamChunk(
                        content="",
                        thinking=delta,
                        output_index=output_index,
                        reasoning_summary_index=reasoning_part[1],
                        output_item_type="reasoning",
                    )
                else:
                    yield ResponsesStreamChunk(
                        content="",
                        output_index=output_index,
                        reasoning_summary_index=reasoning_part[1],
                        provenance_only=True,
                        output_item_type="reasoning",
                    )

            elif event_type == "response.reasoning_summary_text.done":
                # Don't yield ``thinking_signature=item_id`` here. The
                # Codex path treats every signature as ``encrypted_content``
                # and round-trips it to Copilot, which then 400s with
                # ``Encrypted content could not be decrypted`` because
                # ``item_id`` is just a local identifier, not the real
                # encrypted blob. The real blob arrives later on
                # ``output_item.done.item.encrypted_content``; emit the
                # signature there so both Codex and Anthropic round-trips
                # use the verifiable value.
                output_index = data.get("output_index", 0)
                known_item_id = reasoning_item_ids.get(output_index)
                summary_index = data.get("summary_index", 0)
                reasoning_part = (output_index, summary_index)
                snapshot = data.get("text", "")
                completed_item = completed_reasoning_items.get(output_index)
                if completed_item is not None:
                    summary_snapshots = completed_item[0]
                    if (
                        summary_index >= len(summary_snapshots)
                        or summary_snapshots[summary_index] != snapshot
                    ):
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError("Responses reasoning done conflicts with a completed item"),
                        )
                    encrypted_blob = completed_item[1]
                    if (
                        encrypted_blob
                        and known_item_id
                        and output_index not in emitted_reasoning_signatures
                    ):
                        emitted_reasoning_signatures.add(output_index)
                        yield ResponsesStreamChunk(
                            content="",
                            output_index=output_index,
                            reasoning_summary_index=summary_index,
                            output_item_type="reasoning",
                            output_item_done=True,
                            thinking_id=known_item_id,
                            thinking_signature=encrypted_blob,
                        )
                    continue
                if reasoning_part in completed_reasoning_parts:
                    if completed_reasoning_parts[reasoning_part] != snapshot:
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError(
                                "Responses reasoning summary completed with conflicting snapshots"
                            ),
                        )
                    continue
                accumulated = reasoning_parts.get(reasoning_part, "")
                if not snapshot.startswith(accumulated):
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses reasoning done snapshot conflicts with its deltas"),
                    )
                completed_reasoning_parts[reasoning_part] = snapshot
                suffix = snapshot[len(accumulated) :]
                if suffix:
                    reasoning_parts[reasoning_part] = snapshot
                    yield ResponsesStreamChunk(
                        content="",
                        thinking=suffix,
                        output_index=output_index,
                        reasoning_summary_index=summary_index,
                        output_item_type="reasoning",
                    )
                elif not accumulated:
                    yield ResponsesStreamChunk(
                        content="",
                        output_index=output_index,
                        reasoning_summary_index=summary_index,
                        provenance_only=True,
                        output_item_type="reasoning",
                    )

            # Handle function call output_item.added - start of a new function call
            elif event_type == "response.output_item.added":
                item = data.get("item", {})
                output_idx = data.get("output_index", 0)
                item_type = item.get("type")
                bind_declared_type(output_idx, item_type)
                if item.get("type") == "function_call":
                    pending_fcs[output_idx] = {
                        "call_id": item.get("call_id", ""),
                        "name": item.get("name", ""),
                        "arguments": "",
                        "kind": "function",
                        # MCP namespace (e.g. "kusto"). Required on
                        # round-trip or Copilot 400s the next turn.
                        "namespace": item.get("namespace"),
                    }
                elif item.get("type") == "custom_tool_call":
                    # Custom tools (e.g. Codex's apply_patch) stream
                    # raw text via custom_tool_call_input.delta. Same
                    # bookkeeping as function_call but flagged so the
                    # route emits the right event shape downstream.
                    pending_fcs[output_idx] = {
                        "call_id": item.get("call_id", ""),
                        "name": item.get("name", ""),
                        "arguments": "",
                        "kind": "custom",
                    }
                elif item.get("type") == "tool_search_call":
                    # Codex CLI registers a `tool_search` tool
                    # (execution=client) so the model can dynamically
                    # discover MCP tools. Codex's dispatcher matches on
                    # ResponseItem::ToolSearchCall — wrapping this as a
                    # function_call(name="tool_search") makes the call
                    # silently abort (registry has no function tool of
                    # that name). Tag with kind="tool_search" so the
                    # route emits a real tool_search_call item.
                    # NOTE: arguments arrive whole on output_item.done;
                    # if Copilot ever streams them via a dedicated
                    # delta event we'll spot it via unknown_event_counts.
                    pending_fcs[output_idx] = {
                        "call_id": item.get("call_id", ""),
                        "name": "tool_search",
                        "arguments": "",
                        "kind": "tool_search",
                    }

            # Handle function call arguments delta - accumulate silently
            elif event_type == "response.function_call_arguments.delta":
                delta = data.get("delta", "")
                output_idx = data.get("output_index", 0)
                fc = pending_fcs.get(output_idx)
                if fc is None:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses function call payload is missing its item"),
                    )
                if delta:
                    fc["arguments"] += delta

            elif event_type == "response.custom_tool_call_input.delta":
                delta = data.get("delta", "")
                output_idx = data.get("output_index", 0)
                fc = pending_fcs.get(output_idx)
                if fc is None:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses custom tool payload is missing its item"),
                    )
                if delta:
                    fc["arguments"] += delta

            # Handle function call arguments done — finalize arguments
            # but DON'T emit yet. Copilot CAPI sends the ``namespace``
            # field (required for MCP-namespaced tools like
            # ``kusto/execute_query``) on the *later* ``output_item.done``
            # event, not on this one. Emitting here loses namespace and
            # the next turn 400s with ``Missing namespace for
            # function_call 'X'``. Defer to output_item.done.
            elif event_type == "response.function_call_arguments.done":
                output_idx = data.get("output_index", 0)
                fc = pending_fcs.get(output_idx)
                if fc is None:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses function call payload is missing its item"),
                    )
                fc["arguments"] = data.get("arguments", fc["arguments"])

            elif event_type == "response.custom_tool_call_input.done":
                output_idx = data.get("output_index", 0)
                fc = pending_fcs.get(output_idx)
                if fc is None:
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses custom tool payload is missing its item"),
                    )
                fc["arguments"] = data.get("input", fc["arguments"])

            # Handle output_item.done for function calls. Copilot
            # delivers ``namespace`` (for MCP tools) on this event only.
            elif event_type == "response.output_item.done":
                item = data.get("item", {})
                item_type = item.get("type")
                output_idx = data.get("output_index", 0)
                bind_declared_type(output_idx, item_type)
                completed_type = "reasoning" if output_idx in completed_reasoning_items else None
                if item_type in {
                    "message",
                    "function_call",
                    "custom_tool_call",
                    "tool_search_call",
                }:
                    if completed_type is not None:
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError(
                                "Responses output index completed with conflicting item types"
                            ),
                        )
                    item_snapshot = json.dumps(
                        item,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    completed_snapshot = completed_output_items.get(output_idx)
                    if completed_snapshot is not None:
                        if completed_snapshot != (item_type, item_snapshot):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses output item completed with conflicting snapshots"
                                ),
                            )
                        continue
                    completed_output_items[output_idx] = (item_type, item_snapshot)
                if item.get("type") == "function_call":
                    fc = pending_fcs.pop(output_idx, None) or {}
                    # Prefer the final item payload (carries namespace
                    # and the canonical arguments). Fall back to the
                    # bookkeeping dict if the item is sparse.
                    emitted_tool_call = True
                    yield ResponsesStreamChunk(
                        content="",
                        tool_call=ResponsesToolCall(
                            call_id=item.get("call_id") or fc.get("call_id", ""),
                            name=item.get("name") or fc.get("name", ""),
                            arguments=item.get("arguments") or fc.get("arguments", "") or "{}",
                            kind=fc.get("kind", "function"),
                            namespace=item.get("namespace") or fc.get("namespace"),
                        ),
                        output_index=output_idx,
                        output_item_type="function_call",
                        output_item_done=True,
                    )
                elif item.get("type") == "custom_tool_call":
                    # Fallback: if custom_tool_call_input.done didn't
                    # fire (or pending_fcs was already drained), emit
                    # from the final item payload.
                    fc = pending_fcs.pop(output_idx, None)
                    emitted_tool_call = True
                    yield ResponsesStreamChunk(
                        content="",
                        tool_call=ResponsesToolCall(
                            call_id=item.get("call_id") or (fc or {}).get("call_id", ""),
                            name=item.get("name") or (fc or {}).get("name", ""),
                            arguments=item.get("input") or (fc or {}).get("arguments", ""),
                            kind="custom",
                        ),
                        output_index=output_idx,
                        output_item_type="custom_tool_call",
                        output_item_done=True,
                    )
                elif item.get("type") == "reasoning":
                    # Copilot CAPI delivers the reasoning summary here
                    # rather than via reasoning_summary_text.delta.
                    summary_list = item.get("summary", []) or []
                    upstream_id = item.get("id")
                    output_index = data.get("output_index", 0)
                    known_item_id = reasoning_item_ids.get(output_index)
                    if (
                        upstream_id is not None
                        and known_item_id is not None
                        and upstream_id != known_item_id
                    ):
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError("Responses reasoning item changed id for one output index"),
                        )
                    if upstream_id is not None:
                        reasoning_item_ids[output_index] = upstream_id
                    effective_item_id = upstream_id or known_item_id
                    summary_snapshots = tuple(
                        (summary.get("text") or "")
                        for summary in summary_list
                        if isinstance(summary, dict)
                    )
                    encrypted_blob = item.get("encrypted_content")
                    item_snapshot = (summary_snapshots, encrypted_blob)
                    completed_non_reasoning = completed_output_items.get(output_index)
                    if completed_non_reasoning is not None:
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError(
                                "Responses output index completed with conflicting item types"
                            ),
                        )
                    if output_index in completed_reasoning_items:
                        if completed_reasoning_items[output_index] != item_snapshot:
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses reasoning item completed with conflicting snapshots"
                                ),
                            )
                        if (
                            encrypted_blob
                            and effective_item_id
                            and output_index not in emitted_reasoning_signatures
                        ):
                            emitted_reasoning_signatures.add(output_index)
                            yield ResponsesStreamChunk(
                                content="",
                                output_index=output_index,
                                reasoning_summary_index=(
                                    len(completed_reasoning_items[output_index][0]) - 1
                                    if completed_reasoning_items[output_index][0]
                                    else None
                                ),
                                output_item_type="reasoning",
                                output_item_done=True,
                                thinking_id=effective_item_id,
                                thinking_signature=encrypted_blob,
                            )
                        continue
                    tracked_summary_indices = {
                        summary_index
                        for part_output_index, summary_index in (
                            set(reasoning_parts) | set(completed_reasoning_parts)
                        )
                        if part_output_index == output_index
                    }
                    final_summary_indices = set(range(len(summary_snapshots)))
                    if not tracked_summary_indices.issubset(final_summary_indices):
                        self._raise_protocol_error(
                            self.name,
                            request.model,
                            TypeError("Responses reasoning item omitted a streamed summary part"),
                        )
                    logger.info(
                        "Copilot /responses reasoning item: summary_segments=%d encrypted=%s id=%s",
                        len(summary_list),
                        bool(item.get("encrypted_content")),
                        upstream_id,
                    )
                    for summary_index, snapshot in enumerate(summary_snapshots):
                        reasoning_part = (output_index, summary_index)
                        if reasoning_part in completed_reasoning_parts:
                            if completed_reasoning_parts[reasoning_part] != snapshot:
                                self._raise_protocol_error(
                                    self.name,
                                    request.model,
                                    TypeError(
                                        "Responses reasoning item summary conflicts with "
                                        "its completed snapshot"
                                    ),
                                )
                            continue
                        accumulated = reasoning_parts.get(reasoning_part, "")
                        if not snapshot.startswith(accumulated):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses reasoning item summary conflicts with its deltas"
                                ),
                            )
                        completed_reasoning_parts[reasoning_part] = snapshot
                        suffix = snapshot[len(accumulated) :]
                        if suffix:
                            reasoning_parts[reasoning_part] = snapshot
                            yield ResponsesStreamChunk(
                                content="",
                                thinking=suffix,
                                output_index=output_index,
                                reasoning_summary_index=summary_index,
                                output_item_type="reasoning",
                                thinking_id=effective_item_id,
                            )
                    completed_reasoning_items[output_index] = item_snapshot
                    # Emit the upstream id and the encrypted blob
                    # separately so the route can pair them on the
                    # reasoning output item it forwards to Codex. The
                    # blob is only valid against its own id; using a
                    # locally-generated id (or worse, treating the id
                    # as the signature) 400s the next turn with
                    # ``Encrypted content could not be decrypted``.
                    if encrypted_blob and effective_item_id:
                        emitted_reasoning_signatures.add(output_index)
                        yield ResponsesStreamChunk(
                            content="",
                            output_index=output_index,
                            reasoning_summary_index=(
                                len(summary_snapshots) - 1 if summary_snapshots else None
                            ),
                            output_item_type="reasoning",
                            output_item_done=True,
                            thinking_id=effective_item_id,
                            thinking_signature=encrypted_blob,
                        )
                    elif effective_item_id:
                        yield ResponsesStreamChunk(
                            content="",
                            output_index=output_index,
                            reasoning_summary_index=(
                                len(summary_snapshots) - 1 if summary_snapshots else None
                            ),
                            output_item_type="reasoning",
                            output_item_done=True,
                            thinking_id=effective_item_id,
                        )
                    elif not encrypted_blob:
                        yield ResponsesStreamChunk(
                            content="",
                            output_index=output_index,
                            reasoning_summary_index=(
                                len(summary_snapshots) - 1 if summary_snapshots else None
                            ),
                            output_item_type="reasoning",
                            output_item_done=True,
                        )
                elif item.get("type") == "tool_search_call":
                    # Forward as kind="tool_search" so the route emits
                    # an actual tool_search_call wire item — codex's
                    # dispatcher refuses anything else (see v0.3.7
                    # changelog). Arguments arrive as a dict here;
                    # serialize to a JSON string for transport on the
                    # ResponsesToolCall dataclass; the route deserializes
                    # before re-emitting.
                    fc = pending_fcs.pop(output_idx, None)
                    args = item.get("arguments")
                    if isinstance(args, str):
                        args_str = args
                    elif args is None:
                        args_str = "{}"
                    else:
                        args_str = json.dumps(args)
                    call_id = item.get("call_id") or (fc and fc.get("call_id")) or ""
                    emitted_tool_call = True
                    yield ResponsesStreamChunk(
                        content="",
                        tool_call=ResponsesToolCall(
                            call_id=call_id,
                            name="tool_search",
                            arguments=args_str,
                            kind="tool_search",
                        ),
                        output_index=output_idx,
                        output_item_type="tool_search_call",
                        output_item_done=True,
                    )
                else:
                    if item_type == "message":
                        streamed_content_indices = {
                            content_index
                            for part_output_index, content_index in (
                                set(text_parts)
                                | set(refusal_parts)
                                | set(completed_text_parts)
                                | set(completed_refusal_parts)
                            )
                            if part_output_index == output_idx
                        }
                        snapshot_content_indices = set(range(len(item.get("content", []))))
                        if not streamed_content_indices.issubset(snapshot_content_indices):
                            self._raise_protocol_error(
                                self.name,
                                request.model,
                                TypeError(
                                    "Responses message snapshot conflicts with streamed "
                                    "content indices"
                                ),
                            )
                        for content_index, part in enumerate(item.get("content", [])):
                            part_type = part.get("type")
                            if part_type == "output_text":
                                snapshot = part.get("text", "")
                                key = (output_idx, content_index)
                                accumulated = text_parts.get(key, "")
                                if not snapshot.startswith(accumulated):
                                    self._raise_protocol_error(
                                        self.name,
                                        request.model,
                                        TypeError(
                                            "Responses message text snapshot conflicts "
                                            "with streamed deltas"
                                        ),
                                    )
                                suffix = snapshot[len(accumulated) :]
                                if suffix:
                                    text_parts[key] = snapshot
                                    yield ResponsesStreamChunk(
                                        content=suffix,
                                        output_index=output_idx,
                                        content_index=content_index,
                                        output_item_type="message",
                                    )
                            elif part_type == "refusal":
                                snapshot = part.get("refusal", "")
                                key = (output_idx, content_index)
                                accumulated = refusal_parts.get(key, "")
                                if not snapshot.startswith(accumulated):
                                    self._raise_protocol_error(
                                        self.name,
                                        request.model,
                                        TypeError(
                                            "Responses message refusal snapshot conflicts "
                                            "with streamed deltas"
                                        ),
                                    )
                                suffix = snapshot[len(accumulated) :]
                                if suffix:
                                    refusal_parts[key] = snapshot
                                    yield ResponsesStreamChunk(
                                        content="",
                                        refusal=suffix,
                                        output_index=output_idx,
                                        content_index=content_index,
                                        output_item_type="message",
                                    )
                        yield ResponsesStreamChunk(
                            content="",
                            output_index=output_idx,
                            output_item_type="message",
                            output_item_done=True,
                        )
                    elif item_type not in _BENIGN_DONE_ITEM_TYPES:
                        key = f"output_item.done:{item_type}"
                        unknown_event_counts[key] = unknown_event_counts.get(key, 0) + 1

            # Preserve the inner response status even when an upstream
            # gateway uses a mismatched outer event name (for example,
            # response.completed carrying status=incomplete).
            elif event_type in {
                "response.done",
                "response.completed",
                "response.incomplete",
                "response.failed",
                "response.cancelled",
            }:
                resp = data.get("response", {})
                if isinstance(resp, dict):
                    final_usage = resp.get("usage") or final_usage
                terminal_outcome = self.terminal_outcome(resp)
                if any(
                    encrypted_blob and output_index not in reasoning_item_ids
                    for output_index, (_, encrypted_blob) in (completed_reasoning_items.items())
                ):
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        TypeError("Responses encrypted reasoning is missing its upstream id"),
                    )
                if (
                    terminal_outcome.transport is TransportTermination.EXCEPTION
                    and terminal_outcome.error is not None
                    and terminal_outcome.error.code == "upstream_protocol_error"
                ):
                    self._raise_protocol_error(
                        self.name,
                        request.model,
                        ValueError(terminal_outcome.error.message),
                    )
                logger.info(
                    "Copilot /responses stream terminal: model=%s outer_type=%s "
                    "status=%s emitted_tool_call=%s usage=%s",
                    request.model,
                    event_type,
                    terminal_outcome.response_status.value,
                    emitted_tool_call,
                    final_usage,
                )
                yield ResponsesStreamChunk(
                    content="",
                    usage=final_usage,
                    terminal_outcome=terminal_outcome,
                )
                stream_finished = True

            # Catch-all: count any other event types so we can spot
            # things we silently drop (e.g. custom_tool_call_input.*).
            # ``_BENIGN_UPSTREAM_EVENTS`` are intentionally skipped —
            # the route synthesizes equivalents from the deltas we
            # already consume.
            else:
                if event_type not in _BENIGN_UPSTREAM_EVENTS:
                    unknown_event_counts[event_type] = unknown_event_counts.get(event_type, 0) + 1

        if not stream_finished:
            logger.warning(
                "Copilot /responses transport ended without an explicit terminal event: model=%s",
                request.model,
            )

        if unknown_event_counts:
            logger.warning(
                "Copilot /responses unhandled event types: model=%s counts=%s",
                request.model,
                unknown_event_counts,
            )
