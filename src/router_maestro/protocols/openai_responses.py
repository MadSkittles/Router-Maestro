"""Concrete OpenAI Responses protocol runtime."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Literal, cast

from router_maestro.protocols._tool_result_projection import (
    ToolResultProjectionError,
    project_tool_result_output,
    unproject_tool_result_output,
)
from router_maestro.protocols._wire import is_reasoning_capsule_carrier
from router_maestro.protocols.models import (
    ContentBlock,
    FileContent,
    FrozenJsonValue,
    ImageContent,
    MessageContent,
    MessageRole,
    OpaqueState,
    ReasoningConfig,
    ReasoningSummary,
    RefusalContent,
    RequestManifest,
    SemanticEvent,
    SemanticEventType,
    SemanticItem,
    SemanticMessage,
    SemanticRequest,
    SemanticResponse,
    TerminalMetadata,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResult,
    WireProtocol,
)
from router_maestro.protocols.openai_common import (
    decode_reject,
    decode_tool_choice,
    decode_usage,
    encode_arguments,
    encode_stream_usage,
    encode_tool_choice,
    encode_usage,
    has_typed_block,
    optional_bool,
    optional_int,
    optional_number,
    optional_string,
    parse_arguments,
    reject,
    reject_unknown_keys,
    require_list,
    require_mapping,
    require_string,
    terminal_event_values,
    terminal_outcome_from_event,
    thaw_json,
)
from router_maestro.providers.base import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponsesToolCall,
    resolve_terminal_outcome,
)
from router_maestro.utils.reasoning import budget_to_effort

_PROTOCOL = WireProtocol.OPENAI_RESPONSES
_REQUEST_FIELDS = frozenset(
    {
        "model",
        "input",
        "stream",
        "instructions",
        "temperature",
        "max_output_tokens",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "top_p",
        "metadata",
        "service_tier",
        "text",
        "previous_response_id",
    }
)
_RESPONSE_CORE_FIELDS = frozenset(
    {
        "id",
        "object",
        "created_at",
        "model",
        "status",
        "output",
        "usage",
        "error",
        "incomplete_details",
    }
)
# Responses objects repeat these request/configuration values.  They do not
# contribute new generated semantics, so cross-protocol decoding deliberately
# accepts and ignores this closed set while still rejecting unknown siblings.
_RESPONSE_REQUEST_ECHO_FIELDS = frozenset(
    {
        "instructions",
        "max_output_tokens",
        "output_text",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_p",
        "truncation",
    }
)
_RESPONSE_FIELDS = _RESPONSE_CORE_FIELDS | _RESPONSE_REQUEST_ECHO_FIELDS
_RESPONSE_MESSAGE_IGNORED_FIELDS = frozenset({"phase"})
_TOOL_CALL_OUTPUT_TYPES = frozenset({"function_call", "custom_tool_call", "tool_search_call"})


class OpenAIResponsesRuntime:
    """Strict Responses wire codec used only when semantic conversion is needed."""

    protocol = _PROTOCOL

    def __init__(
        self,
        *,
        provider_name: str | None = "openai",
        binding_id: str | None = None,
        allow_per_event_response_ids: bool = False,
        defer_intermediate_item_ids: bool = False,
    ) -> None:
        if provider_name == "":
            raise ValueError("provider_name must be non-empty when provided")
        if binding_id == "":
            raise ValueError("binding_id must be non-empty when provided")
        self.provider_name = provider_name
        self.binding_id = binding_id
        self.allow_per_event_response_ids = allow_per_event_response_ids
        self.defer_intermediate_item_ids = defer_intermediate_item_ids
        self._stream_decoder: ContextVar[OpenAIResponsesStreamDecoder | None] = ContextVar(
            f"openai_responses_stream_decoder_{id(self)}",
            default=None,
        )
        self._stream_encoder: ContextVar[OpenAIResponsesStreamEncoder | None] = ContextVar(
            f"openai_responses_stream_encoder_{id(self)}",
            default=None,
        )

    def inspect_request(self, payload: Mapping[str, Any]) -> RequestManifest:
        input_value = payload.get("input")
        reasoning_carriers, opaque_continuation = _inspect_reasoning_continuation(input_value)
        return RequestManifest(
            protocol=self.protocol,
            model=payload.get("model") if isinstance(payload.get("model"), str) else None,
            stream=payload.get("stream") is True,
            tools=bool(payload.get("tools")),
            images=has_typed_block(input_value, {"input_image", "image_url"}),
            files=has_typed_block(input_value, {"input_file"}),
            reasoning=bool(payload.get("reasoning")) or opaque_continuation,
            parallel_tools=payload.get("parallel_tool_calls") is True,
            reasoning_capsules=tuple(
                carrier for carrier in reasoning_carriers if is_reasoning_capsule_carrier(carrier)
            ),
            previous_response_id=(
                payload.get("previous_response_id")
                if isinstance(payload.get("previous_response_id"), str)
                else None
            ),
            opaque_continuation=opaque_continuation,
        )

    async def decode_request(self, payload: Mapping[str, Any]) -> SemanticRequest:
        return _decode_request(
            payload,
            provider_name=self.provider_name,
            binding_id=self.binding_id,
        )

    async def encode_request(self, request: SemanticRequest) -> Mapping[str, Any]:
        return _encode_request(
            request,
            target_provider=self.provider_name,
            target_binding=self.binding_id,
        )

    async def decode_response(self, payload: Mapping[str, Any]) -> SemanticResponse:
        return _decode_response(
            payload,
            provider_name=self.provider_name,
            binding_id=self.binding_id,
        )

    async def encode_response(self, response: SemanticResponse) -> Mapping[str, Any]:
        return _encode_response(response)

    def new_stream_decoder(
        self,
        *,
        sequence_start: int = 0,
    ) -> OpenAIResponsesStreamDecoder:
        return OpenAIResponsesStreamDecoder(
            provider_name=self.provider_name,
            binding_id=self.binding_id,
            sequence_start=sequence_start,
            allow_per_event_response_ids=self.allow_per_event_response_ids,
            defer_intermediate_item_ids=self.defer_intermediate_item_ids,
        )

    def new_stream_encoder(
        self,
        *,
        model: str | None = None,
        response_id: str | None = None,
    ) -> OpenAIResponsesStreamEncoder:
        return OpenAIResponsesStreamEncoder(
            model=model,
            response_id=response_id,
            provider_name=self.provider_name,
            binding_id=self.binding_id,
        )

    async def decode_stream_event(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        decoder = self._stream_decoder.get()
        if decoder is None or (decoder.terminal and payload.get("type") == "response.created"):
            decoder = self.new_stream_decoder()
            self._stream_decoder.set(decoder)
        return decoder.decode(payload)

    async def encode_stream_event(
        self,
        event: SemanticEvent,
    ) -> tuple[Mapping[str, Any], ...]:
        encoder = self._stream_encoder.get()
        if encoder is None or (
            encoder.terminal and event.type is SemanticEventType.RESPONSE_STARTED
        ):
            model = event.metadata.get("model")
            encoder = self.new_stream_encoder(
                model=model if isinstance(model, str) else None,
                response_id=event.response_id,
            )
            self._stream_encoder.set(encoder)
        return encoder.encode(event)


class OpenAIResponsesStreamDecoder:
    """Stateful decoder for one OpenAI Responses typed-event stream."""

    _TERMINAL_TYPES = frozenset(
        {
            "response.done",
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
        }
    )

    def __init__(
        self,
        *,
        provider_name: str | None = "openai",
        binding_id: str | None = None,
        sequence_start: int = 0,
        allow_per_event_response_ids: bool = False,
        defer_intermediate_item_ids: bool = False,
    ) -> None:
        self.provider_name = provider_name
        self.binding_id = binding_id
        self.allow_per_event_response_ids = allow_per_event_response_ids
        self.defer_intermediate_item_ids = defer_intermediate_item_ids
        self._sequence = sequence_start
        self._started = False
        self._terminal = False
        self._response_id: str | None = None
        self._model: str | None = None
        self._items: dict[int, Mapping[str, Any]] = {}
        self._done_items: set[int] = set()
        self._content_parts: dict[tuple[int, int], str] = {}
        self._done_content_parts: set[tuple[int, int]] = set()
        self._text_parts: dict[tuple[int, int], str] = {}
        self._completed_text_parts: dict[tuple[int, int], str] = {}
        self._refusal_parts: dict[tuple[int, int], str] = {}
        self._completed_refusal_parts: dict[tuple[int, int], str] = {}
        self._reasoning_parts: dict[tuple[int, int], str] = {}
        self._completed_reasoning_parts: dict[tuple[int, int], str] = {}

    @property
    def terminal(self) -> bool:
        return self._terminal

    def decode(self, payload: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        frame = require_mapping(payload, protocol=_PROTOCOL, parameter="stream")
        frame_type = require_string(
            frame.get("type"),
            protocol=_PROTOCOL,
            parameter="stream.type",
            allow_empty=False,
        )
        if self._terminal:
            decode_reject(_PROTOCOL, "stream.type", "frame arrived after terminal event")
        if frame_type == "error":
            return self._decode_error(frame)
        if frame_type in {"response.created", "response.in_progress"}:
            self._capture_response(frame.get("response"), path="stream.response")
            return self._ensure_started()
        if frame_type in self._TERMINAL_TYPES:
            return self._decode_terminal(frame)

        specs = list(self._ensure_started_specs())
        if frame_type in {"response.output_item.added", "response.output_item.done"}:
            specs.extend(self._decode_output_item(frame, done=frame_type.endswith(".done")))
        elif frame_type in {"response.content_part.added", "response.content_part.done"}:
            specs.extend(
                self._decode_content_part_lifecycle(
                    frame,
                    done=frame_type.endswith(".done"),
                )
            )
        elif frame_type == "response.output_text.delta":
            specs.append(
                self._decode_part_delta(
                    frame,
                    refusal=False,
                    delta=require_string(
                        frame.get("delta"),
                        protocol=_PROTOCOL,
                        parameter="stream.delta",
                    ),
                )
            )
        elif frame_type == "response.output_text.done":
            specs.extend(
                self._decode_part_snapshot(
                    frame,
                    refusal=False,
                    snapshot=require_string(
                        frame.get("text"),
                        protocol=_PROTOCOL,
                        parameter="stream.text",
                    ),
                    parameter="stream.text",
                )
            )
            specs.append(self._part_done_spec(frame, output_item_type="message"))
        elif frame_type == "response.refusal.delta":
            specs.append(
                self._decode_part_delta(
                    frame,
                    refusal=True,
                    delta=require_string(
                        frame.get("delta"),
                        protocol=_PROTOCOL,
                        parameter="stream.delta",
                    ),
                )
            )
        elif frame_type == "response.refusal.done":
            specs.extend(
                self._decode_part_snapshot(
                    frame,
                    refusal=True,
                    snapshot=require_string(
                        frame.get("refusal"),
                        protocol=_PROTOCOL,
                        parameter="stream.refusal",
                    ),
                    parameter="stream.refusal",
                )
            )
            specs.append(self._part_done_spec(frame, output_item_type="message"))
        elif frame_type == "response.reasoning_summary_text.delta":
            specs.append(
                self._decode_reasoning_delta(
                    frame,
                    delta=require_string(
                        frame.get("delta"),
                        protocol=_PROTOCOL,
                        parameter="stream.delta",
                    ),
                )
            )
        elif frame_type == "response.reasoning_summary_text.done":
            specs.extend(
                self._decode_reasoning_snapshot(
                    frame,
                    snapshot=require_string(
                        frame.get("text"),
                        protocol=_PROTOCOL,
                        parameter="stream.text",
                    ),
                    parameter="stream.text",
                )
            )
            specs.append(self._reasoning_part_done_spec(frame))
        elif frame_type in {
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            part = require_mapping(frame.get("part"), protocol=_PROTOCOL, parameter="stream.part")
            if part.get("type") != "summary_text":
                decode_reject(_PROTOCOL, "stream.part.type", "must be summary_text")
            done = frame_type.endswith(".done")
            snapshot_specs = self._decode_reasoning_snapshot(
                frame,
                snapshot=require_string(
                    part.get("text"),
                    protocol=_PROTOCOL,
                    parameter="stream.part.text",
                ),
                parameter="stream.part.text",
                complete=done,
            )
            lifecycle = self._reasoning_part_lifecycle_spec(frame, part=part, done=done)
            specs.extend((*snapshot_specs, lifecycle) if done else (lifecycle, *snapshot_specs))
        elif frame_type in {
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
        }:
            output_index = self._frame_index(frame, "output_index")
            stored = self._items.get(output_index, {})
            metadata: dict[str, Any] = {
                "output_item_type": (
                    "custom_tool_call" if "custom_tool" in frame_type else "function_call"
                )
            }
            for key in ("call_id", "name"):
                value = stored.get(key)
                if isinstance(value, str):
                    metadata[key] = value
            specs.append(
                (
                    SemanticEventType.TOOL_ARGUMENTS_DELTA,
                    {
                        "item_id": self._intermediate_item_id(frame),
                        "output_index": output_index,
                        "delta": require_string(
                            frame.get("delta"),
                            protocol=_PROTOCOL,
                            parameter="stream.delta",
                        ),
                        "metadata": metadata,
                    },
                )
            )
        elif frame_type in {
            "response.function_call_arguments.done",
            "response.custom_tool_call_input.done",
        }:
            output_index = self._frame_index(frame, "output_index")
            specs.append(
                (
                    SemanticEventType.OUTPUT_ITEM,
                    {
                        "item_id": self._intermediate_item_id(frame),
                        "output_index": output_index,
                        "metadata": {
                            "output_item_type": (
                                "custom_tool_call"
                                if "custom_tool" in frame_type
                                else "function_call"
                            ),
                            "arguments_done": True,
                        },
                    },
                )
            )
        else:
            decode_reject(_PROTOCOL, "stream.type", f"unsupported event {frame_type!r}")
        return self._events(specs)

    def finish_eof(self) -> tuple[SemanticEvent, ...]:
        if self._terminal:
            return ()
        terminal = TerminalMetadata(
            error_code="unexpected_eof",
            error_message="Upstream stream ended before a terminal response event",
            response_status="unknown",
            transport_termination="unexpected_eof",
        )
        self._terminal = True
        return self._events(
            [
                (SemanticEventType.ERROR, {"terminal": terminal}),
                (SemanticEventType.TERMINAL, {"terminal": terminal}),
            ]
        )

    def _capture_response(self, value: object, *, path: str) -> Mapping[str, Any]:
        response = require_mapping(value, protocol=_PROTOCOL, parameter=path)
        response_id = require_string(
            response.get("id"), protocol=_PROTOCOL, parameter=f"{path}.id", allow_empty=False
        )
        model = require_string(
            response.get("model"),
            protocol=_PROTOCOL,
            parameter=f"{path}.model",
            allow_empty=False,
        )
        if (
            self._response_id is not None
            and response_id != self._response_id
            and not self.allow_per_event_response_ids
        ):
            decode_reject(_PROTOCOL, f"{path}.id", "response ID changed within stream")
        if self._model is not None and model != self._model:
            decode_reject(_PROTOCOL, f"{path}.model", "model changed within stream")
        # Copilot encrypts the Responses envelope ID independently on each SSE
        # event.  Its raw identity binding remains untouched; only the
        # provider-bound semantic decoder treats the first ID as the canonical
        # stream identity and correlates later events by their typed indices.
        if self._response_id is None:
            self._response_id = response_id
        self._model = model
        return response

    def _ensure_started(self) -> tuple[SemanticEvent, ...]:
        return self._events(list(self._ensure_started_specs()))

    def _ensure_started_specs(
        self,
    ) -> tuple[tuple[SemanticEventType, dict[str, Any]], ...]:
        if self._started:
            return ()
        self._started = True
        metadata = {"model": self._model} if self._model is not None else {}
        return ((SemanticEventType.RESPONSE_STARTED, {"metadata": metadata}),)

    def _decode_output_item(
        self,
        frame: Mapping[str, Any],
        *,
        done: bool,
    ) -> list[tuple[SemanticEventType, dict[str, Any]]]:
        output_index = self._frame_index(frame, "output_index")
        item = require_mapping(frame.get("item"), protocol=_PROTOCOL, parameter="stream.item")
        item_type = require_string(
            item.get("type"), protocol=_PROTOCOL, parameter="stream.item.type"
        )
        previous = self._items.get(output_index)
        if previous is not None and previous.get("type") != item_type:
            decode_reject(_PROTOCOL, "stream.item.type", "output item type changed")
        self._items[output_index] = item
        specs: list[tuple[SemanticEventType, dict[str, Any]]] = []
        decoded_item: object | None = None
        if done and item_type == "reasoning":
            if not self._model:
                decode_reject(
                    _PROTOCOL,
                    "stream.item",
                    "reasoning output requires response model context",
                )
            decoded_reasoning = _decode_reasoning_item(
                item,
                parameter="stream.item",
                model=self._model,
                provider_name=self.provider_name,
                binding_id=self.binding_id,
            )
            decoded_item = ReasoningSummary(
                "",
                opaque_state=decoded_reasoning.opaque_state,
            )
            summary = require_list(
                item.get("summary", []),
                protocol=_PROTOCOL,
                parameter="stream.item.summary",
            )
            tracked_indices = {
                summary_index
                for part_output_index, summary_index in (
                    set(self._reasoning_parts) | set(self._completed_reasoning_parts)
                )
                if part_output_index == output_index
            }
            final_indices = set(range(len(summary)))
            if not tracked_indices.issubset(final_indices):
                decode_reject(
                    _PROTOCOL,
                    "stream.item.summary",
                    "reasoning item omitted a streamed summary part",
                )
            item_id = self._optional_mapping_string(item, "id", path="stream.item")
            for summary_index, raw_part in enumerate(summary):
                path = f"stream.item.summary[{summary_index}]"
                part = require_mapping(raw_part, protocol=_PROTOCOL, parameter=path)
                part_type = require_string(
                    part.get("type"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.type",
                )
                if part_type != "summary_text":
                    decode_reject(_PROTOCOL, f"{path}.type", "must be summary_text")
                specs.extend(
                    self._decode_reasoning_snapshot(
                        {
                            "item_id": item_id,
                            "output_index": output_index,
                            "summary_index": summary_index,
                        },
                        snapshot=require_string(
                            part.get("text"),
                            protocol=_PROTOCOL,
                            parameter=f"{path}.text",
                        ),
                        parameter=f"{path}.text",
                    )
                )
        elif done and item_type in {
            "function_call",
            "custom_tool_call",
            "tool_search_call",
        }:
            decoded_item = _decode_tool_call(item, parameter="stream.item")
        elif done and item_type == "message":
            message = _decode_message(item, parameter="stream.item")
            if message.role is not MessageRole.ASSISTANT:
                decode_reject(
                    _PROTOCOL,
                    "stream.item.role",
                    "Responses output message must be assistant",
                )
            item_id = self._optional_mapping_string(item, "id", path="stream.item")
            specs = self._decode_message_snapshot(
                message,
                item_id=item_id,
                output_index=output_index,
            )
            self._done_items.add(output_index)
            return specs
        elif item_type not in {
            "message",
            "reasoning",
            "function_call",
            "custom_tool_call",
            "tool_search_call",
        }:
            decode_reject(_PROTOCOL, "stream.item.type", f"unsupported item {item_type!r}")
        if done:
            self._done_items.add(output_index)
        item_id = self._optional_mapping_string(item, "id", path="stream.item")
        if not done and self.defer_intermediate_item_ids:
            item_id = None
        specs.append(
            (
                SemanticEventType.OUTPUT_ITEM,
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "item": decoded_item,
                    "metadata": {
                        "output_item_type": item_type,
                        "output_item_done": done,
                        "provenance_only": decoded_item is None,
                    },
                },
            )
        )
        return specs

    def _decode_content_part_lifecycle(
        self,
        frame: Mapping[str, Any],
        *,
        done: bool,
    ) -> list[tuple[SemanticEventType, dict[str, Any]]]:
        reject_unknown_keys(
            frame,
            frozenset(
                {
                    "type",
                    "item_id",
                    "output_index",
                    "content_index",
                    "part",
                    "sequence_number",
                }
            ),
            protocol=_PROTOCOL,
            parameter="stream",
        )
        if "sequence_number" in frame:
            sequence_number = optional_int(
                frame.get("sequence_number"),
                protocol=_PROTOCOL,
                parameter="stream.sequence_number",
            )
            if sequence_number is None or sequence_number < 0:
                decode_reject(
                    _PROTOCOL,
                    "stream.sequence_number",
                    "must be a non-negative integer",
                )
        output_index = self._frame_index(frame, "output_index")
        content_index = self._frame_index(frame, "content_index")
        key = (output_index, content_index)
        part = require_mapping(
            frame.get("part"),
            protocol=_PROTOCOL,
            parameter="stream.part",
        )
        part_type, snapshot = self._validate_output_content_part(part)
        previous_type = self._content_parts.get(key)
        if previous_type is not None and previous_type != part_type:
            decode_reject(_PROTOCOL, "stream.part.type", "content part type changed")
        if key in self._done_content_parts:
            decode_reject(_PROTOCOL, "stream.type", "content part event followed done")
        self._content_parts[key] = part_type
        if done:
            self._done_content_parts.add(key)
        lifecycle = (
            SemanticEventType.OUTPUT_ITEM,
            {
                "item_id": self._intermediate_item_id(frame),
                "output_index": output_index,
                "content_index": content_index,
                "metadata": {
                    "output_item_type": "message",
                    "content_part": thaw_json(part),
                    "content_part_done" if done else "content_part_added": True,
                    "provenance_only": True,
                },
            },
        )
        snapshot_specs = self._decode_part_snapshot(
            frame,
            refusal=part_type == "refusal",
            snapshot=snapshot,
            parameter=("stream.part.refusal" if part_type == "refusal" else "stream.part.text"),
            complete=done,
        )
        if done:
            return [*snapshot_specs, lifecycle]
        return [lifecycle, *snapshot_specs]

    @staticmethod
    def _validate_output_content_part(part: Mapping[str, Any]) -> tuple[str, str]:
        part_type = require_string(
            part.get("type"),
            protocol=_PROTOCOL,
            parameter="stream.part.type",
            allow_empty=False,
        )
        if part_type == "output_text":
            reject_unknown_keys(
                part,
                frozenset({"type", "text", "annotations", "logprobs"}),
                protocol=_PROTOCOL,
                parameter="stream.part",
            )
            text = require_string(
                part.get("text"),
                protocol=_PROTOCOL,
                parameter="stream.part.text",
            )
            for field in ("annotations", "logprobs"):
                values = require_list(
                    part.get(field, []),
                    protocol=_PROTOCOL,
                    parameter=f"stream.part.{field}",
                )
                if values:
                    decode_reject(
                        _PROTOCOL,
                        f"stream.part.{field}",
                        f"{field.replace('_', ' ')} are not modeled",
                    )
            return part_type, text
        if part_type == "refusal":
            reject_unknown_keys(
                part,
                frozenset({"type", "refusal"}),
                protocol=_PROTOCOL,
                parameter="stream.part",
            )
            refusal = require_string(
                part.get("refusal"),
                protocol=_PROTOCOL,
                parameter="stream.part.refusal",
            )
            return part_type, refusal
        decode_reject(
            _PROTOCOL,
            "stream.part.type",
            f"unsupported output content part {part_type!r}",
        )

    def _decode_part_delta(
        self,
        frame: Mapping[str, Any],
        *,
        refusal: bool,
        delta: str,
    ) -> tuple[SemanticEventType, dict[str, Any]]:
        output_index = self._frame_index(frame, "output_index")
        content_index = self._frame_index(frame, "content_index")
        key = (output_index, content_index)
        parts = self._refusal_parts if refusal else self._text_parts
        completed = self._completed_refusal_parts if refusal else self._completed_text_parts
        if key in completed:
            decode_reject(_PROTOCOL, "stream.delta", "delta followed a completed content part")
        parts[key] = parts.get(key, "") + delta
        return self._part_delta_spec(
            item_id=self._intermediate_item_id(frame),
            output_index=output_index,
            content_index=content_index,
            refusal=refusal,
            delta=delta,
        )

    def _decode_part_snapshot(
        self,
        frame: Mapping[str, Any],
        *,
        refusal: bool,
        snapshot: str,
        parameter: str,
        complete: bool = True,
    ) -> list[tuple[SemanticEventType, dict[str, Any]]]:
        output_index = self._frame_index(frame, "output_index")
        content_index = self._frame_index(frame, "content_index")
        key = (output_index, content_index)
        parts = self._refusal_parts if refusal else self._text_parts
        completed = self._completed_refusal_parts if refusal else self._completed_text_parts
        previous_snapshot = completed.get(key)
        if previous_snapshot is not None:
            if previous_snapshot != snapshot:
                decode_reject(_PROTOCOL, parameter, "conflicts with completed content part")
            return []
        accumulated = parts.get(key, "")
        if not snapshot.startswith(accumulated):
            decode_reject(_PROTOCOL, parameter, "conflicts with streamed deltas")
        suffix = snapshot[len(accumulated) :]
        parts[key] = snapshot
        if complete:
            completed[key] = snapshot
        if not suffix:
            return []
        return [
            self._part_delta_spec(
                item_id=self._intermediate_item_id(frame),
                output_index=output_index,
                content_index=content_index,
                refusal=refusal,
                delta=suffix,
            )
        ]

    def _decode_reasoning_delta(
        self,
        frame: Mapping[str, Any],
        *,
        delta: str,
    ) -> tuple[SemanticEventType, dict[str, Any]]:
        output_index = self._frame_index(frame, "output_index")
        summary_index = self._frame_index(frame, "summary_index")
        key = (output_index, summary_index)
        if output_index in self._done_items:
            decode_reject(_PROTOCOL, "stream.delta", "delta followed a completed reasoning item")
        if key in self._completed_reasoning_parts:
            decode_reject(_PROTOCOL, "stream.delta", "delta followed a completed reasoning part")
        self._reasoning_parts[key] = self._reasoning_parts.get(key, "") + delta
        return self._reasoning_delta_spec(
            item_id=self._intermediate_item_id(frame),
            output_index=output_index,
            summary_index=summary_index,
            delta=delta,
        )

    def _decode_reasoning_snapshot(
        self,
        frame: Mapping[str, Any],
        *,
        snapshot: str,
        parameter: str,
        complete: bool = True,
    ) -> list[tuple[SemanticEventType, dict[str, Any]]]:
        output_index = self._frame_index(frame, "output_index")
        summary_index = self._frame_index(frame, "summary_index")
        key = (output_index, summary_index)
        previous_snapshot = self._completed_reasoning_parts.get(key)
        if previous_snapshot is not None:
            if previous_snapshot != snapshot:
                decode_reject(_PROTOCOL, parameter, "conflicts with completed reasoning part")
            return []
        if output_index in self._done_items:
            decode_reject(_PROTOCOL, parameter, "reasoning part followed a completed item")
        accumulated = self._reasoning_parts.get(key, "")
        if not snapshot.startswith(accumulated):
            decode_reject(_PROTOCOL, parameter, "conflicts with streamed reasoning deltas")
        suffix = snapshot[len(accumulated) :]
        self._reasoning_parts[key] = snapshot
        if complete:
            self._completed_reasoning_parts[key] = snapshot
        if not suffix:
            return []
        return [
            self._reasoning_delta_spec(
                item_id=self._intermediate_item_id(frame),
                output_index=output_index,
                summary_index=summary_index,
                delta=suffix,
            )
        ]

    @staticmethod
    def _reasoning_delta_spec(
        *,
        item_id: str | None,
        output_index: int,
        summary_index: int,
        delta: str,
    ) -> tuple[SemanticEventType, dict[str, Any]]:
        return (
            SemanticEventType.REASONING_DELTA,
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": summary_index,
                "delta": delta,
                "metadata": {
                    "output_item_type": "reasoning",
                    "reasoning_summary_index": summary_index,
                },
            },
        )

    def _decode_message_snapshot(
        self,
        message: SemanticMessage,
        *,
        item_id: str | None,
        output_index: int,
    ) -> list[tuple[SemanticEventType, dict[str, Any]]]:
        specs: list[tuple[SemanticEventType, dict[str, Any]]] = []
        for content_index, part in enumerate(message.content):
            if isinstance(part, TextContent):
                refusal = False
                snapshot = part.text
                field = "text"
            elif isinstance(part, RefusalContent):
                refusal = True
                snapshot = part.refusal
                field = "refusal"
            else:
                decode_reject(
                    _PROTOCOL,
                    f"stream.item.content[{content_index}]",
                    f"unsupported output content {type(part).__name__}",
                )
            specs.extend(
                self._decode_part_snapshot(
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                    },
                    refusal=refusal,
                    snapshot=snapshot,
                    parameter=f"stream.item.content[{content_index}].{field}",
                )
            )
        specs.append(
            (
                SemanticEventType.OUTPUT_ITEM,
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "metadata": {
                        "output_item_type": "message",
                        "output_item_done": True,
                        "provenance_only": True,
                    },
                },
            )
        )
        return specs

    @staticmethod
    def _part_delta_spec(
        *,
        item_id: str | None,
        output_index: int,
        content_index: int,
        refusal: bool,
        delta: str,
    ) -> tuple[SemanticEventType, dict[str, Any]]:
        if refusal:
            return (
                SemanticEventType.OUTPUT_ITEM,
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "item": RefusalContent(delta),
                    "metadata": {"output_item_type": "message", "delta": True},
                },
            )
        return (
            SemanticEventType.TEXT_DELTA,
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": delta,
            },
        )

    def _decode_terminal(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEvent, ...]:
        response = self._capture_response(frame.get("response"), path="stream.response")
        specs = list(self._ensure_started_specs())
        outputs = response.get("output", [])
        if outputs is not None:
            output_items = require_list(
                outputs, protocol=_PROTOCOL, parameter="stream.response.output"
            )
            for index, raw_item in enumerate(output_items):
                if index in self._done_items:
                    continue
                item_frame = {"output_index": index, "item": raw_item}
                specs.extend(self._decode_output_item(item_frame, done=True))
        usage = response.get("usage")
        if usage is not None:
            specs.append(
                (
                    SemanticEventType.USAGE,
                    {
                        "usage": decode_usage(
                            usage,
                            protocol=_PROTOCOL,
                            input_field="input_tokens",
                            output_field="output_tokens",
                            input_details_field="input_tokens_details",
                            output_details_field="output_tokens_details",
                        )
                    },
                )
            )
        status = response.get("status")
        if not isinstance(status, str):
            status = {
                "response.completed": "completed",
                "response.incomplete": "incomplete",
                "response.failed": "failed",
                "response.cancelled": "cancelled",
                "response.done": "completed",
            }[frame["type"]]
        incomplete = response.get("incomplete_details")
        finish_reason = _finish_reason(
            status,
            incomplete,
            has_tool_calls=any(
                item.get("type") in _TOOL_CALL_OUTPUT_TYPES for item in self._items.values()
            ),
        )
        error_code = error_message = None
        if response.get("error") is not None:
            error = require_mapping(
                response.get("error"),
                protocol=_PROTOCOL,
                parameter="stream.response.error",
            )
            code = error.get("code", error.get("type"))
            if code is not None:
                error_code = require_string(
                    code, protocol=_PROTOCOL, parameter="stream.response.error.code"
                )
            if error.get("message") is not None:
                error_message = require_string(
                    error.get("message"),
                    protocol=_PROTOCOL,
                    parameter="stream.response.error.message",
                )
        terminal = TerminalMetadata(
            finish_reason=finish_reason,
            error_code=error_code,
            error_message=error_message,
            response_status=status,
            transport_termination="explicit_terminal",
            incomplete_details=(
                require_mapping(
                    incomplete,
                    protocol=_PROTOCOL,
                    parameter="stream.response.incomplete_details",
                )
                if incomplete is not None
                else None
            ),
        )
        if error_code is not None or error_message is not None or status == "failed":
            specs.append((SemanticEventType.ERROR, {"terminal": terminal}))
        specs.append((SemanticEventType.TERMINAL, {"terminal": terminal}))
        self._terminal = True
        return self._events(specs)

    def _decode_error(self, frame: Mapping[str, Any]) -> tuple[SemanticEvent, ...]:
        error = frame.get("error", frame)
        error = require_mapping(error, protocol=_PROTOCOL, parameter="stream.error")
        code = error.get("code", error.get("type", "upstream_error"))
        terminal = TerminalMetadata(
            error_code=require_string(code, protocol=_PROTOCOL, parameter="stream.error.code"),
            error_message=require_string(
                error.get("message"),
                protocol=_PROTOCOL,
                parameter="stream.error.message",
            ),
            response_status="failed",
            transport_termination="explicit_terminal",
        )
        self._terminal = True
        return self._events(
            [
                (SemanticEventType.ERROR, {"terminal": terminal}),
                (SemanticEventType.TERMINAL, {"terminal": terminal}),
            ]
        )

    def _part_done_spec(
        self,
        frame: Mapping[str, Any],
        *,
        output_item_type: str,
    ) -> tuple[SemanticEventType, dict[str, Any]]:
        return (
            SemanticEventType.OUTPUT_ITEM,
            {
                "item_id": self._intermediate_item_id(frame),
                "output_index": self._frame_index(frame, "output_index"),
                "content_index": self._frame_index(frame, "content_index"),
                "metadata": {
                    "output_item_type": output_item_type,
                    "content_part_done": True,
                    "provenance_only": True,
                },
            },
        )

    def _reasoning_part_done_spec(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[SemanticEventType, dict[str, Any]]:
        summary_index = self._frame_index(frame, "summary_index")
        return (
            SemanticEventType.OUTPUT_ITEM,
            {
                "item_id": self._intermediate_item_id(frame),
                "output_index": self._frame_index(frame, "output_index"),
                "content_index": summary_index,
                "metadata": {
                    "output_item_type": "reasoning",
                    "reasoning_summary_index": summary_index,
                    "content_part_done": True,
                    "provenance_only": True,
                },
            },
        )

    def _reasoning_part_lifecycle_spec(
        self,
        frame: Mapping[str, Any],
        *,
        part: Mapping[str, Any],
        done: bool,
    ) -> tuple[SemanticEventType, dict[str, Any]]:
        output_index = self._frame_index(frame, "output_index")
        summary_index = self._frame_index(frame, "summary_index")
        if output_index in self._done_items and not done:
            decode_reject(
                _PROTOCOL,
                "stream.type",
                "reasoning part event followed a completed item",
            )
        return (
            SemanticEventType.OUTPUT_ITEM,
            {
                "item_id": self._intermediate_item_id(frame),
                "output_index": output_index,
                "content_index": summary_index,
                "metadata": {
                    "output_item_type": "reasoning",
                    "reasoning_summary_index": summary_index,
                    "summary_part": thaw_json(part),
                    "content_part_done" if done else "content_part_added": True,
                    "provenance_only": True,
                },
            },
        )

    def _frame_index(self, frame: Mapping[str, Any], field: str) -> int:
        value = optional_int(frame.get(field, 0), protocol=_PROTOCOL, parameter=f"stream.{field}")
        if value is None or value < 0:
            decode_reject(_PROTOCOL, f"stream.{field}", "must be a non-negative integer")
        return value

    def _optional_frame_string(
        self,
        frame: Mapping[str, Any],
        field: str,
    ) -> str | None:
        return optional_string(frame.get(field), protocol=_PROTOCOL, parameter=f"stream.{field}")

    def _intermediate_item_id(self, frame: Mapping[str, Any]) -> str | None:
        item_id = self._optional_frame_string(frame, "item_id")
        # Copilot re-obfuscates item IDs across SSE events. Correlate transient
        # events by their typed indices and retain only output_item.done's ID.
        if self.defer_intermediate_item_ids:
            return None
        return item_id

    def _optional_mapping_string(
        self,
        value: Mapping[str, Any],
        field: str,
        *,
        path: str,
    ) -> str | None:
        return optional_string(value.get(field), protocol=_PROTOCOL, parameter=f"{path}.{field}")

    def _events(
        self,
        specs: list[tuple[SemanticEventType, dict[str, Any]]],
    ) -> tuple[SemanticEvent, ...]:
        events = []
        for event_type, values in specs:
            events.append(
                SemanticEvent(
                    type=event_type,
                    sequence=self._sequence,
                    response_id=self._response_id,
                    **values,
                )
            )
            self._sequence += 1
        return tuple(events)


class OpenAIResponsesStreamEncoder:
    """Stateful encoder for one OpenAI Responses typed-event stream."""

    def __init__(
        self,
        *,
        model: str | None = None,
        response_id: str | None = None,
        provider_name: str | None = "openai",
        binding_id: str | None = None,
    ) -> None:
        self.model = model
        self.response_id = response_id
        self.provider_name = provider_name
        self.binding_id = binding_id
        self._created_at = int(time.time())
        self._started = False
        self._terminal = False
        self._pending_error: TerminalMetadata | None = None
        self._pending_usage = None
        self._output: dict[int, dict[str, Any]] = {}
        self._implicit_output_indices: dict[tuple[str, str], int] = {}
        self._claimed_output_indices: set[int] = set()
        self._next_output_index = 0

    @property
    def terminal(self) -> bool:
        return self._terminal

    def encode(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        if self._terminal:
            reject(_PROTOCOL, "event.type", "event arrived after terminal event")
        self._capture_identity(event)
        if event.type is SemanticEventType.RESPONSE_STARTED:
            return tuple(self._ensure_started())
        if event.type is SemanticEventType.ERROR:
            self._pending_error = event.terminal or TerminalMetadata(
                error_code="upstream_error",
                error_message="Upstream stream failed",
                response_status="failed",
            )
            return ()
        if event.type is SemanticEventType.USAGE:
            if event.usage is None:
                reject(_PROTOCOL, "event.usage", "usage event requires Usage")
            self._pending_usage = event.usage
            return ()
        if event.type is SemanticEventType.TERMINAL:
            return self._encode_terminal(event)
        frames = self._ensure_started()
        frames.extend(self._encode_payload(event))
        return tuple(frames)

    def finish_eof(self) -> tuple[Mapping[str, Any], ...]:
        if self._terminal:
            return ()
        self._terminal = True
        return (
            {
                "type": "error",
                "code": "unexpected_eof",
                "message": "Semantic event stream ended before terminal event",
            },
        )

    def _capture_identity(self, event: SemanticEvent) -> None:
        metadata_model = event.metadata.get("model")
        if isinstance(metadata_model, str):
            if self.model is not None and self.model != metadata_model:
                reject(_PROTOCOL, "event.metadata.model", "model changed within stream")
            self.model = metadata_model
        if event.response_id is not None:
            if self.response_id is not None and self.response_id != event.response_id:
                reject(_PROTOCOL, "event.response_id", "response ID changed within stream")
            self.response_id = event.response_id

    def _ensure_started(self) -> list[Mapping[str, Any]]:
        if self._started:
            return []
        self._started = True
        return [{"type": "response.created", "response": self._response("in_progress")}]

    def _encode_payload(self, event: SemanticEvent) -> list[Mapping[str, Any]]:
        output_index = self._event_output_index(event)
        item_id = event.item_id
        if event.type is SemanticEventType.TEXT_DELTA:
            message = self._append_message_delta(
                output_index,
                item_id=item_id,
                content_index=event.content_index or 0,
                part_type="output_text",
                delta=event.delta or "",
            )
            return [
                {
                    "type": "response.output_text.delta",
                    "item_id": item_id or message["id"],
                    "output_index": output_index,
                    "content_index": event.content_index or 0,
                    "delta": event.delta or "",
                }
            ]
        if event.type is SemanticEventType.REASONING_DELTA:
            reasoning = self._reasoning_snapshot(output_index, item_id=item_id)
            summary_index = self._reasoning_index(event)
            summary = reasoning["summary"]
            while len(summary) <= summary_index:
                summary.append({"type": "summary_text", "text": ""})
            summary[summary_index]["text"] += event.delta or ""
            return [
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": item_id or reasoning["id"],
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "delta": event.delta or "",
                }
            ]
        if event.type is SemanticEventType.TOOL_ARGUMENTS_DELTA:
            item_type = event.metadata.get("output_item_type", "function_call")
            if item_type == "custom_tool_call":
                frame_type = "response.custom_tool_call_input.delta"
            elif item_type == "function_call":
                frame_type = "response.function_call_arguments.delta"
            else:
                reject(
                    _PROTOCOL,
                    "event.metadata.output_item_type",
                    "tool arguments require function_call or custom_tool_call",
                )
            raw_item = self._output.get(output_index)
            frames: list[Mapping[str, Any]] = []
            if raw_item is None and item_type == "function_call":
                call_id = event.metadata.get("call_id") or event.item_id
                name = event.metadata.get("name")
                if not isinstance(call_id, str) or not call_id:
                    reject(_PROTOCOL, "event.item_id", "tool delta requires a call ID")
                if not isinstance(name, str) or not name:
                    reject(_PROTOCOL, "event.metadata.name", "tool delta requires a name")
                raw_item = {
                    "type": "function_call",
                    "id": event.item_id or f"fc_{output_index}",
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                    "status": "completed",
                }
                self._output[output_index] = raw_item
                frames.append(
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": dict(raw_item),
                    }
                )
            if raw_item is None or raw_item.get("type") != item_type:
                reject(
                    _PROTOCOL,
                    "event.output_index",
                    "tool argument delta has no matching output item",
                )
            field = "input" if item_type == "custom_tool_call" else "arguments"
            previous = raw_item.get(field, "")
            if not isinstance(previous, str):
                reject(_PROTOCOL, f"event.item.{field}", "tool arguments must be text")
            raw_item[field] = previous + (event.delta or "")
            frames.append(
                {
                    "type": frame_type,
                    "item_id": item_id or raw_item.get("id"),
                    "output_index": output_index,
                    "delta": event.delta or "",
                }
            )
            return frames
        if event.type is not SemanticEventType.OUTPUT_ITEM:
            reject(_PROTOCOL, "event.type", f"unsupported event {event.type.value!r}")
        item = event.item
        if item is None:
            return []
        if isinstance(item, TextContent):
            message = self._append_message_delta(
                output_index,
                item_id=item_id,
                content_index=event.content_index or 0,
                part_type="output_text",
                delta=item.text,
            )
            return [
                {
                    "type": "response.output_text.delta",
                    "item_id": item_id or message["id"],
                    "output_index": output_index,
                    "content_index": event.content_index or 0,
                    "delta": item.text,
                }
            ]
        if isinstance(item, RefusalContent):
            message = self._append_message_delta(
                output_index,
                item_id=item_id,
                content_index=event.content_index or 0,
                part_type="refusal",
                delta=item.refusal,
            )
            return [
                {
                    "type": "response.refusal.delta",
                    "item_id": item_id or message["id"],
                    "output_index": output_index,
                    "content_index": event.content_index or 0,
                    "delta": item.refusal,
                }
            ]
        raw_item = self._encode_item(item, event=event)
        if (
            isinstance(item, ToolCall)
            and item.kind == "function"
            and not item.arguments
            and "tool_index" in event.metadata
        ):
            # Legacy Chat emits a declaration followed by argument deltas.
            raw_item["arguments"] = ""
        self._output[output_index] = raw_item
        done = event.metadata.get("output_item_done", True) is True
        return [
            {
                "type": ("response.output_item.done" if done else "response.output_item.added"),
                "output_index": output_index,
                "item": raw_item,
            }
        ]

    def _event_output_index(self, event: SemanticEvent) -> int:
        """Resolve absent legacy indices without colliding unlike output items."""
        if event.output_index is not None:
            index = event.output_index
            self._claimed_output_indices.add(index)
            self._next_output_index = max(self._next_output_index, index + 1)
            return index

        item = event.item
        if event.type is SemanticEventType.REASONING_DELTA or isinstance(item, ReasoningSummary):
            kind = "reasoning"
            identifier = event.item_id or "default"
        elif event.type is SemanticEventType.TOOL_ARGUMENTS_DELTA or isinstance(item, ToolCall):
            kind = "tool"
            tool_id = item.call_id if isinstance(item, ToolCall) else None
            metadata_index = event.metadata.get("tool_index")
            identifier = event.item_id or tool_id or str(metadata_index)
        else:
            kind = "message"
            identifier = event.item_id or "assistant"

        key = (kind, identifier)
        existing = self._implicit_output_indices.get(key)
        if existing is not None:
            return existing
        while self._next_output_index in self._claimed_output_indices:
            self._next_output_index += 1
        index = self._next_output_index
        self._next_output_index += 1
        self._claimed_output_indices.add(index)
        self._implicit_output_indices[key] = index
        return index

    def _append_message_delta(
        self,
        output_index: int,
        *,
        item_id: str | None,
        content_index: int,
        part_type: str,
        delta: str,
    ) -> dict[str, Any]:
        raw_item = self._output.get(output_index)
        if raw_item is None:
            raw_item = {
                "type": "message",
                "id": item_id or f"msg_{output_index}",
                "role": "assistant",
                "status": "completed",
                "content": [],
            }
            self._output[output_index] = raw_item
        if raw_item.get("type") != "message":
            reject(_PROTOCOL, "event.output_index", "message output index is already in use")
        content = raw_item.get("content")
        if not isinstance(content, list):  # pragma: no cover - built locally or validated upstream
            reject(_PROTOCOL, "event.output_index", "message content must be an array")
        while len(content) <= content_index:
            next_type = part_type if len(content) == content_index else "output_text"
            field = "refusal" if next_type == "refusal" else "text"
            content.append({"type": next_type, field: ""})
        part = content[content_index]
        if part.get("type") != part_type:
            reject(_PROTOCOL, "event.content_index", "message content type changed")
        field = "refusal" if part_type == "refusal" else "text"
        previous = part.get(field)
        if not isinstance(previous, str):
            reject(_PROTOCOL, "event.content_index", "message content must be text")
        part[field] = previous + delta
        return raw_item

    def _reasoning_snapshot(
        self,
        output_index: int,
        *,
        item_id: str | None,
    ) -> dict[str, Any]:
        raw_item = self._output.get(output_index)
        if raw_item is None:
            raw_item = {
                "type": "reasoning",
                "id": item_id or f"rs_{output_index}",
                "summary": [],
            }
            self._output[output_index] = raw_item
        if raw_item.get("type") != "reasoning":
            reject(_PROTOCOL, "event.output_index", "reasoning output index is already in use")
        return raw_item

    def _encode_item(self, item: object, *, event: SemanticEvent) -> dict[str, Any]:
        if isinstance(item, ReasoningSummary):
            return _encode_reasoning_item(
                item,
                parameter="event.item",
                model=self._required_model(),
                target_provider=self.provider_name,
                target_binding=self.binding_id,
            )
        if isinstance(item, ToolCall):
            return _encode_tool_call(item, parameter="event.item")
        if isinstance(item, SemanticMessage):
            response = SemanticResponse(
                id=self._required_response_id(),
                model=self._required_model(),
                output=(item,),
            )
            encoded = _encode_response_output(response)
            if len(encoded) != 1:
                reject(_PROTOCOL, "event.item", "message split into multiple output items")
            return encoded[0]
        reject(_PROTOCOL, "event.item", f"unsupported stream item {type(item).__name__}")

    def _encode_terminal(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]:
        terminal = (
            self._pending_error
            or event.terminal
            or TerminalMetadata(
                finish_reason="stop",
                response_status="completed",
            )
        )
        status = terminal.response_status or (
            "incomplete" if terminal.finish_reason in {"length", "content_filter"} else "completed"
        )
        if terminal.error_code is not None or terminal.error_message is not None:
            status = "failed"
        response = self._response(status)
        if self._pending_usage is not None:
            response["usage"] = encode_stream_usage(
                self._pending_usage,
                protocol=_PROTOCOL,
                input_field="input_tokens",
                output_field="output_tokens",
                input_details_field="input_tokens_details",
                output_details_field="output_tokens_details",
            )
        if status == "incomplete":
            if terminal.incomplete_details is not None:
                response["incomplete_details"] = thaw_json(terminal.incomplete_details)
            else:
                response["incomplete_details"] = {
                    "reason": _encode_incomplete_reason(terminal.finish_reason)
                }
        if status == "failed":
            response["error"] = {
                "code": terminal.error_code or "upstream_error",
                "message": terminal.error_message or "Upstream stream failed",
            }
        self._terminal = True
        event_type = {
            "completed": "response.completed",
            "incomplete": "response.incomplete",
            "failed": "response.failed",
            "cancelled": "response.cancelled",
        }.get(status)
        if event_type is None:
            reject(_PROTOCOL, "event.terminal.response_status", "unsupported value")
        return ({"type": event_type, "response": response},)

    def _response(self, status: str) -> dict[str, Any]:
        return {
            "id": self._required_response_id(),
            "object": "response",
            "created_at": self._created_at,
            "model": self._required_model(),
            "status": status,
            "output": [self._output[index] for index in sorted(self._output)],
            "usage": None,
            "error": None,
            "incomplete_details": None,
        }

    def _reasoning_index(self, event: SemanticEvent) -> int:
        value = event.metadata.get("reasoning_summary_index", event.content_index or 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            reject(_PROTOCOL, "event.metadata.reasoning_summary_index", "must be non-negative")
        return value

    def _required_model(self) -> str:
        if not self.model:
            reject(_PROTOCOL, "event.metadata.model", "Responses stream requires a model")
        return self.model

    def _required_response_id(self) -> str:
        if not self.response_id:
            reject(_PROTOCOL, "event.response_id", "Responses stream requires a response ID")
        return self.response_id


def _decode_request(
    payload: Mapping[str, Any],
    *,
    provider_name: str | None,
    binding_id: str | None,
) -> SemanticRequest:
    body = require_mapping(payload, protocol=_PROTOCOL, parameter="request")
    reject_unknown_keys(body, _REQUEST_FIELDS, protocol=_PROTOCOL, parameter="")
    if "previous_response_id" in body:
        decode_reject(
            _PROTOCOL,
            "previous_response_id",
            "server-side continuation is available only on an identity Responses path",
        )
    model = require_string(
        body.get("model"), protocol=_PROTOCOL, parameter="model", allow_empty=False
    )
    input_items = list(
        _decode_input(
            body.get("input"),
            model=model,
            provider_name=provider_name,
            binding_id=binding_id,
        )
    )
    instructions = body.get("instructions")
    if instructions is not None:
        input_items.insert(
            0,
            SemanticMessage(
                role=MessageRole.SYSTEM,
                content=(
                    TextContent(
                        require_string(
                            instructions,
                            protocol=_PROTOCOL,
                            parameter="instructions",
                        )
                    ),
                ),
            ),
        )
    reasoning = _decode_reasoning(body.get("reasoning"))
    metadata = body.get("metadata") or {}
    metadata = require_mapping(metadata, protocol=_PROTOCOL, parameter="metadata")
    structured_output = body.get("text")
    if structured_output is not None:
        structured_output = require_mapping(structured_output, protocol=_PROTOCOL, parameter="text")
    return SemanticRequest(
        model=model,
        input=tuple(input_items),
        tools=_decode_tools(body.get("tools")),
        stream=optional_bool(body.get("stream", False), protocol=_PROTOCOL, parameter="stream")
        or False,
        max_output_tokens=optional_int(
            body.get("max_output_tokens"),
            protocol=_PROTOCOL,
            parameter="max_output_tokens",
        ),
        temperature=optional_number(
            body.get("temperature"), protocol=_PROTOCOL, parameter="temperature"
        ),
        top_p=optional_number(body.get("top_p"), protocol=_PROTOCOL, parameter="top_p"),
        tool_choice=decode_tool_choice(
            body.get("tool_choice"), protocol=_PROTOCOL, nested_function=False
        ),
        parallel_tool_calls=optional_bool(
            body.get("parallel_tool_calls"),
            protocol=_PROTOCOL,
            parameter="parallel_tool_calls",
        ),
        reasoning=reasoning,
        structured_output=structured_output,
        service_tier=optional_string(
            body.get("service_tier"), protocol=_PROTOCOL, parameter="service_tier"
        ),
        metadata=metadata,
        explicit_fields=frozenset(body),
    )


def _inspect_reasoning_continuation(value: object) -> tuple[tuple[str, ...], bool]:
    """Inspect top-level Responses input items without materializing semantic IR."""
    if not isinstance(value, list | tuple):
        return (), False
    carriers: list[str] = []
    opaque = False
    for item in value:
        if not isinstance(item, Mapping) or item.get("type") != "reasoning":
            continue
        encrypted = item.get("encrypted_content")
        if isinstance(encrypted, str):
            carriers.append(encrypted)
            opaque = True
        elif encrypted is not None:
            opaque = True
    return tuple(carriers), opaque


def _decode_reasoning(value: object) -> ReasoningConfig | None:
    if value is None:
        return None
    reasoning = require_mapping(value, protocol=_PROTOCOL, parameter="reasoning")
    reject_unknown_keys(
        reasoning,
        frozenset({"effort"}),
        protocol=_PROTOCOL,
        parameter="reasoning",
    )
    return ReasoningConfig(
        enabled=True,
        effort=optional_string(
            reasoning.get("effort"), protocol=_PROTOCOL, parameter="reasoning.effort"
        ),
    )


def _decode_input(
    value: object,
    *,
    model: str,
    provider_name: str | None,
    binding_id: str | None,
) -> tuple[SemanticItem, ...]:
    if isinstance(value, str):
        return (SemanticMessage(role=MessageRole.USER, content=(TextContent(value),)),)
    items = require_list(value, protocol=_PROTOCOL, parameter="input")
    decoded: list[SemanticItem] = []
    calls_by_id: dict[str, ToolCall] = {}
    for index, raw_item in enumerate(items):
        path = f"input[{index}]"
        item = require_mapping(raw_item, protocol=_PROTOCOL, parameter=path)
        item_type = item.get("type", "message" if "role" in item else None)
        item_type = require_string(item_type, protocol=_PROTOCOL, parameter=f"{path}.type")
        if item_type == "message":
            decoded.append(_decode_message(item, parameter=path))
        elif item_type in {"function_call", "custom_tool_call", "tool_search_call"}:
            call = _decode_tool_call(item, parameter=path)
            calls_by_id[call.call_id] = call
            decoded.append(call)
        elif item_type in {
            "function_call_output",
            "custom_tool_call_output",
            "tool_search_output",
        }:
            result = _decode_tool_result(item, parameter=path)
            matching_call = calls_by_id.get(result.call_id)
            if matching_call is not None:
                if matching_call.kind != result.kind:
                    decode_reject(
                        _PROTOCOL,
                        f"{path}.type",
                        f"does not match preceding {matching_call.kind!r} call",
                    )
                if result.namespace is not None and result.namespace != matching_call.namespace:
                    decode_reject(
                        _PROTOCOL,
                        f"{path}.namespace",
                        "does not match preceding tool call namespace",
                    )
                if result.namespace is None:
                    result = replace(result, namespace=matching_call.namespace)
            decoded.append(result)
        elif item_type == "reasoning":
            decoded.append(
                _decode_reasoning_item(
                    item,
                    parameter=path,
                    model=model,
                    provider_name=provider_name,
                    binding_id=binding_id,
                )
            )
        else:
            decode_reject(_PROTOCOL, f"{path}.type", f"unsupported input item {item_type!r}")
    return tuple(decoded)


def _decode_message(value: Mapping[str, Any], *, parameter: str) -> SemanticMessage:
    reject_unknown_keys(
        value,
        frozenset({"type", "role", "content", "id", "status", "name"})
        | _RESPONSE_MESSAGE_IGNORED_FIELDS,
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    role_value = require_string(
        value.get("role"), protocol=_PROTOCOL, parameter=f"{parameter}.role"
    )
    try:
        role = MessageRole(role_value)
    except ValueError:
        decode_reject(_PROTOCOL, f"{parameter}.role", f"unsupported role {role_value!r}")
    content = _decode_message_content(value.get("content"), parameter=f"{parameter}.content")
    return SemanticMessage(
        role=role,
        content=content,
        name=optional_string(value.get("name"), protocol=_PROTOCOL, parameter=f"{parameter}.name"),
        item_id=optional_string(value.get("id"), protocol=_PROTOCOL, parameter=f"{parameter}.id"),
        status=optional_string(
            value.get("status"), protocol=_PROTOCOL, parameter=f"{parameter}.status"
        ),
    )


def _decode_message_content(value: object, *, parameter: str) -> tuple[MessageContent, ...]:
    if isinstance(value, str):
        return (TextContent(value),)
    blocks = require_list(value, protocol=_PROTOCOL, parameter=parameter)
    decoded: list[MessageContent] = []
    for index, raw_block in enumerate(blocks):
        path = f"{parameter}[{index}]"
        block = require_mapping(raw_block, protocol=_PROTOCOL, parameter=path)
        block_type = require_string(block.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type")
        if block_type in {"input_text", "output_text"}:
            allowed = {"type", "text"}
            if block_type == "output_text":
                allowed.update({"annotations", "logprobs"})
                annotations = block.get("annotations", [])
                if annotations:
                    decode_reject(_PROTOCOL, f"{path}.annotations", "annotations are not modeled")
                logprobs = block.get("logprobs", [])
                if not isinstance(logprobs, list):
                    decode_reject(_PROTOCOL, f"{path}.logprobs", "must be an array")
                if logprobs:
                    decode_reject(
                        _PROTOCOL,
                        f"{path}.logprobs",
                        "log probabilities are not modeled",
                    )
            reject_unknown_keys(block, frozenset(allowed), protocol=_PROTOCOL, parameter=path)
            decoded.append(
                TextContent(
                    require_string(block.get("text"), protocol=_PROTOCOL, parameter=f"{path}.text")
                )
            )
        elif block_type == "refusal":
            reject_unknown_keys(
                block,
                frozenset({"type", "refusal"}),
                protocol=_PROTOCOL,
                parameter=path,
            )
            decoded.append(
                RefusalContent(
                    require_string(
                        block.get("refusal"),
                        protocol=_PROTOCOL,
                        parameter=f"{path}.refusal",
                    )
                )
            )
        elif block_type in {"input_image", "image_url"}:
            decoded.append(_decode_image(block, parameter=path))
        elif block_type == "input_file":
            decoded.append(_decode_file(block, parameter=path))
        else:
            decode_reject(_PROTOCOL, f"{path}.type", f"unsupported content type {block_type!r}")
    return tuple(decoded)


def _decode_image(value: Mapping[str, Any], *, parameter: str) -> ImageContent:
    reject_unknown_keys(
        value,
        frozenset({"type", "image_url", "file_id", "detail"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    image_url = value.get("image_url")
    file_id = value.get("file_id")
    if (image_url is None) == (file_id is None):
        decode_reject(_PROTOCOL, parameter, "exactly one of image_url or file_id is required")
    source = image_url if image_url is not None else file_id
    return ImageContent(
        source=require_string(source, protocol=_PROTOCOL, parameter=f"{parameter}.source"),
        detail=optional_string(
            value.get("detail"), protocol=_PROTOCOL, parameter=f"{parameter}.detail"
        ),
        source_kind="url" if image_url is not None else "file_id",
    )


def _decode_file(value: Mapping[str, Any], *, parameter: str) -> FileContent:
    reject_unknown_keys(
        value,
        frozenset({"type", "file_id", "file_url", "file_data", "filename"}),
        protocol=_PROTOCOL,
        parameter=parameter,
    )
    present = [name for name in ("file_id", "file_url", "file_data") if value.get(name) is not None]
    if len(present) != 1:
        decode_reject(
            _PROTOCOL,
            parameter,
            "exactly one of file_id, file_url, or file_data is required",
        )
    source_field = present[0]
    return FileContent(
        source=require_string(
            value[source_field], protocol=_PROTOCOL, parameter=f"{parameter}.{source_field}"
        ),
        filename=optional_string(
            value.get("filename"), protocol=_PROTOCOL, parameter=f"{parameter}.filename"
        ),
        source_kind={"file_url": "url", "file_data": "base64"}.get(source_field, source_field),
    )


def _decode_tool_call(value: Mapping[str, Any], *, parameter: str) -> ToolCall:
    item_type = require_string(value.get("type"), protocol=_PROTOCOL, parameter=f"{parameter}.type")
    common = {"type", "id", "call_id", "status"}
    status = value.get("status")
    if status not in {None, "completed", "in_progress"}:
        decode_reject(_PROTOCOL, f"{parameter}.status", f"unsupported status {status!r}")
    call_id = require_string(
        value.get("call_id"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.call_id",
        allow_empty=False,
    )
    item_id = optional_string(value.get("id"), protocol=_PROTOCOL, parameter=f"{parameter}.id")
    if item_type == "function_call":
        reject_unknown_keys(
            value,
            frozenset(common | {"name", "arguments", "namespace"}),
            protocol=_PROTOCOL,
            parameter=parameter,
        )
        return ToolCall(
            call_id=call_id,
            item_id=item_id,
            name=require_string(
                value.get("name"),
                protocol=_PROTOCOL,
                parameter=f"{parameter}.name",
                allow_empty=False,
            ),
            arguments=parse_arguments(
                value.get("arguments", "{}"),
                protocol=_PROTOCOL,
                parameter=f"{parameter}.arguments",
            ),
            namespace=optional_string(
                value.get("namespace"),
                protocol=_PROTOCOL,
                parameter=f"{parameter}.namespace",
            ),
        )
    if item_type == "custom_tool_call":
        reject_unknown_keys(
            value,
            frozenset(common | {"name", "input", "namespace"}),
            protocol=_PROTOCOL,
            parameter=parameter,
        )
        return ToolCall(
            call_id=call_id,
            item_id=item_id,
            name=require_string(
                value.get("name"),
                protocol=_PROTOCOL,
                parameter=f"{parameter}.name",
                allow_empty=False,
            ),
            arguments={
                "input": require_string(
                    value.get("input"),
                    protocol=_PROTOCOL,
                    parameter=f"{parameter}.input",
                )
            },
            kind="custom",
            namespace=optional_string(
                value.get("namespace"),
                protocol=_PROTOCOL,
                parameter=f"{parameter}.namespace",
            ),
        )
    if item_type == "tool_search_call":
        reject_unknown_keys(
            value,
            frozenset(common | {"arguments", "name", "execution"}),
            protocol=_PROTOCOL,
            parameter=parameter,
        )
        arguments = value.get("arguments", {})
        arguments = parse_arguments(
            arguments, protocol=_PROTOCOL, parameter=f"{parameter}.arguments"
        )
        execution = value.get("execution")
        if execution not in {None, "client"}:
            decode_reject(
                _PROTOCOL,
                f"{parameter}.execution",
                "tool-search execution must be client",
            )
        return ToolCall(
            call_id=call_id,
            item_id=item_id,
            name=optional_string(
                value.get("name"), protocol=_PROTOCOL, parameter=f"{parameter}.name"
            )
            or "tool_search",
            arguments=arguments,
            kind="tool_search",
        )
    decode_reject(_PROTOCOL, f"{parameter}.type", f"unsupported call type {item_type!r}")


def _decode_tool_result(value: Mapping[str, Any], *, parameter: str) -> ToolResult:
    item_type = require_string(value.get("type"), protocol=_PROTOCOL, parameter=f"{parameter}.type")
    call_id = require_string(
        value.get("call_id"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.call_id",
        allow_empty=False,
    )
    item_id = optional_string(value.get("id"), protocol=_PROTOCOL, parameter=f"{parameter}.id")
    status = value.get("status")
    if status not in {None, "completed"}:
        decode_reject(_PROTOCOL, f"{parameter}.status", "must be completed")
    if item_type in {"function_call_output", "custom_tool_call_output"}:
        allowed = {"type", "id", "call_id", "output", "status"}
        if item_type == "custom_tool_call_output":
            allowed.add("namespace")
        reject_unknown_keys(
            value,
            frozenset(allowed),
            protocol=_PROTOCOL,
            parameter=parameter,
        )
        try:
            output, is_error = unproject_tool_result_output(value.get("output"))
        except ToolResultProjectionError as exc:
            decode_reject(_PROTOCOL, f"{parameter}.output", str(exc))
        return ToolResult(
            call_id=call_id,
            content=_decode_tool_output(output, parameter=f"{parameter}.output"),
            is_error=is_error,
            item_id=item_id,
            kind=("custom" if item_type == "custom_tool_call_output" else "function"),
            namespace=(
                optional_string(
                    value.get("namespace"),
                    protocol=_PROTOCOL,
                    parameter=f"{parameter}.namespace",
                )
                if item_type == "custom_tool_call_output"
                else None
            ),
        )
    if item_type == "tool_search_output":
        reject_unknown_keys(
            value,
            frozenset({"type", "id", "call_id", "execution", "status", "tools"}),
            protocol=_PROTOCOL,
            parameter=parameter,
        )
        execution = require_string(
            value.get("execution"),
            protocol=_PROTOCOL,
            parameter=f"{parameter}.execution",
        )
        if execution != "client":
            decode_reject(
                _PROTOCOL,
                f"{parameter}.execution",
                "tool-search output execution must be client",
            )
        tools = require_list(value.get("tools"), protocol=_PROTOCOL, parameter=f"{parameter}.tools")
        return ToolResult(
            call_id=call_id,
            structured_content=cast(
                FrozenJsonValue,
                {
                    "execution": execution,
                    "status": status or "completed",
                    "tools": tools,
                },
            ),
            item_id=item_id,
            kind="tool_search",
        )
    decode_reject(
        _PROTOCOL,
        f"{parameter}.type",
        f"unsupported tool result type {item_type!r}",
    )


def _decode_tool_output(value: object, *, parameter: str) -> tuple[ContentBlock, ...]:
    if isinstance(value, str):
        return (TextContent(value),)
    blocks = require_list(value, protocol=_PROTOCOL, parameter=parameter)
    decoded: list[ContentBlock] = []
    for index, raw_block in enumerate(blocks):
        path = f"{parameter}[{index}]"
        block = require_mapping(raw_block, protocol=_PROTOCOL, parameter=path)
        block_type = require_string(block.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type")
        if block_type == "input_text":
            reject_unknown_keys(
                block,
                frozenset({"type", "text"}),
                protocol=_PROTOCOL,
                parameter=path,
            )
            decoded.append(
                TextContent(
                    require_string(block.get("text"), protocol=_PROTOCOL, parameter=f"{path}.text")
                )
            )
        elif block_type == "input_image":
            decoded.append(_decode_image(block, parameter=path))
        elif block_type == "input_file":
            decoded.append(_decode_file(block, parameter=path))
        else:
            decode_reject(
                _PROTOCOL,
                f"{path}.type",
                f"unsupported tool output content {block_type!r}",
            )
    return tuple(decoded)


def _decode_reasoning_item(
    value: Mapping[str, Any],
    *,
    parameter: str,
    model: str,
    provider_name: str | None,
    binding_id: str | None,
) -> ReasoningSummary:
    item_type = require_string(value.get("type"), protocol=_PROTOCOL, parameter=f"{parameter}.type")
    if item_type != "reasoning":
        decode_reject(_PROTOCOL, f"{parameter}.type", "must be reasoning")
    item_id = require_string(
        value.get("id"),
        protocol=_PROTOCOL,
        parameter=f"{parameter}.id",
        allow_empty=False,
    )
    summary = require_list(
        value.get("summary", []), protocol=_PROTOCOL, parameter=f"{parameter}.summary"
    )
    texts = []
    for index, raw_part in enumerate(summary):
        path = f"{parameter}.summary[{index}]"
        part = require_mapping(raw_part, protocol=_PROTOCOL, parameter=path)
        summary_type = require_string(
            part.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type"
        )
        if summary_type == "summary_text":
            texts.append(
                require_string(part.get("text"), protocol=_PROTOCOL, parameter=f"{path}.text")
            )
    return ReasoningSummary(
        text="".join(texts),
        opaque_state=OpaqueState(
            origin_protocol=_PROTOCOL,
            origin_provider=provider_name,
            origin_model=model,
            item_id=item_id,
            blob=value,
            origin_binding=binding_id,
        ),
    )


def _decode_tools(value: object) -> tuple[ToolDefinition, ...]:
    if value is None:
        return ()
    tools = require_list(value, protocol=_PROTOCOL, parameter="tools")
    decoded = []
    for index, raw_tool in enumerate(tools):
        path = f"tools[{index}]"
        tool = require_mapping(raw_tool, protocol=_PROTOCOL, parameter=path)
        reject_unknown_keys(
            tool,
            frozenset({"type", "name", "description", "parameters", "strict"}),
            protocol=_PROTOCOL,
            parameter=path,
        )
        tool_type = require_string(tool.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type")
        if tool_type != "function":
            decode_reject(_PROTOCOL, f"{path}.type", "only function tools are modeled")
        parameters = require_mapping(
            tool.get("parameters", {}),
            protocol=_PROTOCOL,
            parameter=f"{path}.parameters",
        )
        decoded.append(
            ToolDefinition(
                name=require_string(
                    tool.get("name"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.name",
                    allow_empty=False,
                ),
                description=optional_string(
                    tool.get("description"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.description",
                ),
                input_schema=parameters,
                strict=optional_bool(
                    tool.get("strict"),
                    protocol=_PROTOCOL,
                    parameter=f"{path}.strict",
                ),
            )
        )
    return tuple(decoded)


def _encode_request(
    request: SemanticRequest,
    *,
    target_provider: str | None = None,
    target_binding: str | None = None,
) -> dict[str, Any]:
    _reject_request_fields(request)
    instructions, input_items = _encode_input(
        request.input,
        model=request.model,
        target_provider=target_provider,
        target_binding=target_binding,
    )
    payload: dict[str, Any] = {"model": request.model, "input": input_items}
    if instructions is not None:
        payload["instructions"] = instructions
    _put(payload, "stream", request.stream, request, default=False)
    _put(payload, "temperature", request.temperature, request)
    _put(payload, "max_output_tokens", request.max_output_tokens, request)
    _put(payload, "top_p", request.top_p, request)
    _put(payload, "service_tier", request.service_tier, request)
    if request.metadata:
        payload["metadata"] = thaw_json(request.metadata)
    if request.tools:
        payload["tools"] = [_encode_tool(tool) for tool in request.tools]
    choice = encode_tool_choice(request.tool_choice, protocol=_PROTOCOL, nested_function=False)
    if choice is not None:
        payload["tool_choice"] = choice
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.reasoning is not None:
        if request.reasoning.enabled is False:
            reject(_PROTOCOL, "reasoning.enabled", "Responses has no disabled reasoning value")
        reasoning = {}
        effort = request.reasoning.effort
        if effort is None and request.reasoning.budget_tokens is not None:
            effort = budget_to_effort(request.reasoning.budget_tokens)
            if effort is None:
                reject(
                    _PROTOCOL,
                    "reasoning.budget_tokens",
                    "does not map to a Responses effort tier",
                )
        if effort is not None:
            reasoning["effort"] = effort
        payload["reasoning"] = reasoning
    if request.structured_output is not None:
        payload["text"] = thaw_json(request.structured_output)
    return payload


def _reject_request_fields(request: SemanticRequest) -> None:
    if request.candidate_count not in {None, 1}:
        reject(
            _PROTOCOL,
            "candidate_count",
            "Responses supports exactly one candidate",
        )
    for name, value in (
        ("frequency_penalty", request.frequency_penalty),
        ("presence_penalty", request.presence_penalty),
        ("top_k", request.top_k),
        ("response_mime_type", request.response_mime_type),
        ("user", request.user),
    ):
        if value is not None:
            reject(_PROTOCOL, name, "field is not supported by Responses")
    if request.stop_sequences:
        reject(_PROTOCOL, "stop_sequences", "Responses does not support stop sequences")
    if request.provider_extensions:
        key = sorted(request.provider_extensions)[0]
        reject(_PROTOCOL, key, "provider extension is not portable")


def _put(
    payload: dict[str, Any],
    name: str,
    value: object,
    request: SemanticRequest,
    *,
    default: object = None,
) -> None:
    if value != default or name in request.explicit_fields:
        payload[name] = value


def _encode_input(
    items: tuple[object, ...],
    *,
    model: str,
    target_provider: str | None,
    target_binding: str | None,
) -> tuple[str | None, list[dict[str, Any]]]:
    _validate_input_tool_links(items)
    instruction_parts = []
    encoded: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        path = f"input[{index}]"
        if isinstance(item, SemanticMessage) and item.role is MessageRole.SYSTEM:
            if item.item_id is not None or item.status is not None or item.name is not None:
                reject(_PROTOCOL, path, "system message metadata cannot become instructions")
            for part in item.content:
                if not isinstance(part, TextContent):
                    reject(_PROTOCOL, f"{path}.content", "instructions support text only")
                instruction_parts.append(part.text)
            continue
        encoded.extend(
            _encode_input_item(
                item,
                parameter=path,
                model=model,
                target_provider=target_provider,
                target_binding=target_binding,
            )
        )
    return ("\n".join(instruction_parts) if instruction_parts else None), encoded


def _validate_input_tool_links(items: tuple[object, ...]) -> None:
    calls_by_id: dict[str, ToolCall] = {}
    for index, item in enumerate(items):
        path = f"input[{index}]"
        parts = item.content if isinstance(item, SemanticMessage) else (item,)
        for part_index, part in enumerate(parts):
            part_path = (
                f"{path}.content[{part_index}]" if isinstance(item, SemanticMessage) else path
            )
            if isinstance(part, ToolCall):
                calls_by_id[part.call_id] = part
                continue
            if not isinstance(part, ToolResult):
                continue
            matching_call = calls_by_id.get(part.call_id)
            if matching_call is not None and matching_call.kind != part.kind:
                reject(
                    _PROTOCOL,
                    f"{part_path}.kind",
                    f"does not match preceding {matching_call.kind!r} call",
                )
            if part.namespace is not None:
                if part.kind == "custom":
                    if matching_call is not None and matching_call.namespace != part.namespace:
                        reject(
                            _PROTOCOL,
                            f"{part_path}.namespace",
                            "does not match preceding tool call namespace",
                        )
                elif matching_call is None or matching_call.namespace != part.namespace:
                    reject(
                        _PROTOCOL,
                        f"{part_path}.namespace",
                        "tool result namespace requires a matching preceding call",
                    )


def _encode_input_item(
    item: object,
    *,
    parameter: str,
    model: str,
    target_provider: str | None,
    target_binding: str | None,
) -> list[dict[str, Any]]:
    if isinstance(item, ToolCall):
        return [_encode_tool_call(item, parameter=parameter)]
    if isinstance(item, ToolResult):
        return [_encode_tool_result(item, parameter=parameter)]
    if isinstance(item, ReasoningSummary):
        return [
            _encode_reasoning_item(
                item,
                parameter=parameter,
                model=model,
                target_provider=target_provider,
                target_binding=target_binding,
            )
        ]
    if isinstance(item, TextContent | ImageContent | FileContent | RefusalContent):
        item = SemanticMessage(role=MessageRole.USER, content=(item,))
    if not isinstance(item, SemanticMessage):
        reject(_PROTOCOL, parameter, f"unsupported input item {type(item).__name__}")
    encoded_parts: list[dict[str, Any] | tuple[str, dict[str, Any]]] = []
    message_parts: list[dict[str, Any]] = []

    def flush_message_parts() -> None:
        nonlocal message_parts
        if message_parts:
            encoded_parts.append(("message", {"content": message_parts}))
            message_parts = []

    for index, part in enumerate(item.content):
        part_path = f"{parameter}.content[{index}]"
        if isinstance(part, ToolCall):
            if item.role is not MessageRole.ASSISTANT:
                reject(_PROTOCOL, part_path, "tool calls require assistant role")
            flush_message_parts()
            encoded_parts.append(_encode_tool_call(part, parameter=part_path))
        elif isinstance(part, ToolResult):
            flush_message_parts()
            encoded_parts.append(_encode_tool_result(part, parameter=part_path))
        elif isinstance(part, ReasoningSummary):
            if item.role is not MessageRole.ASSISTANT:
                reject(_PROTOCOL, part_path, "reasoning requires assistant role")
            flush_message_parts()
            encoded_parts.append(
                _encode_reasoning_item(
                    part,
                    parameter=part_path,
                    model=model,
                    target_provider=target_provider,
                    target_binding=target_binding,
                )
            )
        else:
            message_parts.append(_encode_content_part(part, role=item.role, parameter=part_path))
    flush_message_parts()
    if not encoded_parts:
        encoded_parts.append(("message", {"content": []}))

    message_count = sum(isinstance(part, tuple) for part in encoded_parts)
    if (item.name is not None or item.item_id is not None or item.status is not None) and (
        message_count != 1
    ):
        reject(
            _PROTOCOL,
            parameter,
            "message metadata cannot survive splitting around sibling output items",
        )

    result: list[dict[str, Any]] = []
    for encoded_part in encoded_parts:
        if not isinstance(encoded_part, tuple):
            result.append(encoded_part)
            continue
        message = {
            "type": "message",
            "role": item.role.value,
            "content": encoded_part[1]["content"],
        }
        if item.name is not None:
            message["name"] = item.name
        if item.item_id is not None:
            message["id"] = item.item_id
        if item.status is not None:
            message["status"] = item.status
        result.append(message)
    return result


def _encode_content_part(part: object, *, role: MessageRole, parameter: str) -> dict[str, Any]:
    if isinstance(part, TextContent):
        block_type = "output_text" if role is MessageRole.ASSISTANT else "input_text"
        result: dict[str, Any] = {"type": block_type, "text": part.text}
        if block_type == "output_text":
            result["annotations"] = []
        return result
    if isinstance(part, RefusalContent):
        if role is not MessageRole.ASSISTANT:
            reject(_PROTOCOL, parameter, "refusal requires assistant role")
        return {"type": "refusal", "refusal": part.refusal}
    if isinstance(part, ImageContent):
        if isinstance(part.source, bytes):
            reject(_PROTOCOL, parameter, "binary image must be encoded as a data URL")
        source_kind = part.source_kind or "url"
        if source_kind in {"base64", "inline_data"}:
            if part.media_type is None:
                reject(
                    _PROTOCOL,
                    f"{parameter}.media_type",
                    "base64 images require a media type",
                )
            field = "image_url"
            source = f"data:{part.media_type};base64,{part.source}"
        elif source_kind in {"url", "file_id"}:
            field = "file_id" if source_kind == "file_id" else "image_url"
            source = part.source
        else:
            reject(_PROTOCOL, f"{parameter}.source_kind", f"unsupported kind {source_kind!r}")
        result = {"type": "input_image", field: source}
        if part.detail is not None:
            result["detail"] = part.detail
        return result
    if isinstance(part, FileContent):
        if isinstance(part.source, bytes):
            reject(_PROTOCOL, parameter, "binary file must be base64 encoded")
        source_kind = part.source_kind or "file_id"
        field = {"file_id": "file_id", "url": "file_url", "base64": "file_data"}.get(source_kind)
        if field is None:
            reject(_PROTOCOL, f"{parameter}.source_kind", f"unsupported kind {source_kind!r}")
        result = {"type": "input_file", field: part.source}
        if part.filename is not None:
            result["filename"] = part.filename
        return result
    reject(_PROTOCOL, parameter, f"unsupported content {type(part).__name__}")


def _encode_tool_call(call: ToolCall, *, parameter: str) -> dict[str, Any]:
    if call.opaque_state is not None:
        reject(
            _PROTOCOL,
            f"{parameter}.opaque_state",
            "Responses tool calls cannot carry opaque tool state",
        )
    common: dict[str, Any] = {"call_id": call.call_id, "status": "completed"}
    if call.item_id is not None:
        common["id"] = call.item_id
    if call.kind == "function":
        result = {
            **common,
            "type": "function_call",
            "name": call.name,
            "arguments": encode_arguments(call.arguments),
        }
        if call.namespace is not None:
            result["namespace"] = call.namespace
        return result
    if call.kind == "custom":
        raw_arguments = thaw_json(call.arguments)
        if set(raw_arguments) != {"input"} or not isinstance(raw_arguments["input"], str):
            reject(
                _PROTOCOL,
                f"{parameter}.arguments",
                "custom calls require exactly one string input field",
            )
        result = {
            **common,
            "type": "custom_tool_call",
            "name": call.name,
            "input": raw_arguments["input"],
        }
        if call.namespace is not None:
            result["namespace"] = call.namespace
        return result
    if call.kind == "tool_search":
        if call.namespace is not None:
            reject(_PROTOCOL, f"{parameter}.namespace", "tool-search calls lack namespaces")
        if call.name != "tool_search":
            reject(
                _PROTOCOL,
                f"{parameter}.name",
                "tool-search calls require the canonical tool_search name",
            )
        return {
            **common,
            "type": "tool_search_call",
            "execution": "client",
            "arguments": thaw_json(call.arguments),
        }
    reject(_PROTOCOL, f"{parameter}.kind", f"unsupported call kind {call.kind!r}")


def _encode_tool_result(result: ToolResult, *, parameter: str) -> dict[str, Any]:
    if result.kind == "tool_search":
        if result.is_error:
            reject(
                _PROTOCOL,
                f"{parameter}.is_error",
                "tool-search output cannot mark errors",
            )
        if result.namespace is not None:
            reject(
                _PROTOCOL,
                f"{parameter}.namespace",
                "tool-search outputs lack namespaces",
            )
        if result.content:
            reject(
                _PROTOCOL,
                f"{parameter}.content",
                "tool-search output uses structured_content.tools",
            )
        raw = thaw_json(result.structured_content)
        if not isinstance(raw, Mapping):
            reject(
                _PROTOCOL,
                f"{parameter}.structured_content",
                "tool-search output requires an object",
            )
        unknown = sorted(set(raw) - {"execution", "status", "tools"})
        if unknown:
            reject(
                _PROTOCOL,
                f"{parameter}.structured_content.{unknown[0]}",
                "unsupported tool-search output field",
            )
        if raw.get("execution") != "client":
            reject(
                _PROTOCOL,
                f"{parameter}.structured_content.execution",
                "tool-search output execution must be client",
            )
        if raw.get("status") != "completed":
            reject(
                _PROTOCOL,
                f"{parameter}.structured_content.status",
                "tool-search output status must be completed",
            )
        tools = raw.get("tools")
        if not isinstance(tools, list):
            reject(
                _PROTOCOL,
                f"{parameter}.structured_content.tools",
                "tool-search output tools must be an array",
            )
        search_payload: dict[str, Any] = {
            "type": "tool_search_output",
            "call_id": result.call_id,
            "execution": "client",
            "status": "completed",
            "tools": tools,
        }
        if result.item_id is not None:
            search_payload["id"] = result.item_id
        return search_payload
    result_type = {
        "function": "function_call_output",
        "custom": "custom_tool_call_output",
    }.get(result.kind)
    if result_type is None:
        reject(
            _PROTOCOL,
            f"{parameter}.kind",
            f"unsupported tool result kind {result.kind!r}",
        )
    output = _encode_tool_output(result, parameter=parameter)
    if result.is_error and not isinstance(output, str):
        reject(
            _PROTOCOL,
            f"{parameter}.content",
            "Responses cannot preserve error semantics for multi-block tool output",
        )
    payload: dict[str, Any] = {
        "type": result_type,
        "call_id": result.call_id,
        "output": project_tool_result_output(
            output,
            is_error=result.is_error,
        ),
    }
    if result.item_id is not None:
        payload["id"] = result.item_id
    if result.kind == "custom" and result.namespace is not None:
        payload["namespace"] = result.namespace
    return payload


def _encode_tool_output(result: ToolResult, *, parameter: str) -> object:
    if result.structured_content is not None:
        if result.content:
            reject(
                _PROTOCOL,
                f"{parameter}.content",
                "cannot combine content and structured tool output",
            )
        return json.dumps(thaw_json(result.structured_content), ensure_ascii=False)
    if not result.content:
        return ""
    if len(result.content) == 1 and isinstance(result.content[0], TextContent):
        return result.content[0].text
    return [
        _encode_content_part(
            part,
            role=MessageRole.USER,
            parameter=f"{parameter}.content[{index}]",
        )
        for index, part in enumerate(result.content)
    ]


def _encode_reasoning_item(
    reasoning: ReasoningSummary,
    *,
    parameter: str,
    model: str,
    target_provider: str | None = None,
    target_binding: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "reasoning",
        "summary": ([{"type": "summary_text", "text": reasoning.text}] if reasoning.text else []),
    }
    if reasoning.opaque_state is not None:
        state = reasoning.opaque_state
        if state.origin_protocol is not _PROTOCOL:
            reject(_PROTOCOL, parameter, "opaque reasoning originated from another protocol")
        if state.origin_model != model:
            reject(
                _PROTOCOL,
                f"{parameter}.opaque_state.origin_model",
                "model provenance does not match the target",
            )
        if target_provider is not None:
            if state.origin_provider is None:
                reject(
                    _PROTOCOL,
                    f"{parameter}.opaque_state.origin_provider",
                    "provider provenance is required for replay",
                )
            if state.origin_provider != target_provider:
                reject(
                    _PROTOCOL,
                    f"{parameter}.opaque_state.origin_provider",
                    "provider provenance does not match the target",
                )
        if target_binding is not None:
            if state.origin_binding is None:
                reject(
                    _PROTOCOL,
                    f"{parameter}.opaque_state.origin_binding",
                    "binding provenance is required for replay",
                )
            if state.origin_binding != target_binding:
                reject(
                    _PROTOCOL,
                    f"{parameter}.opaque_state.origin_binding",
                    "binding provenance does not match the target",
                )
        if isinstance(state.blob, Mapping):
            raw_item = thaw_json(state.blob)
            raw_type = raw_item.get("type")
            if raw_type != "reasoning":
                reject(
                    _PROTOCOL,
                    f"{parameter}.opaque_state.blob.type",
                    "raw state must be a reasoning item",
                )
            if raw_item.get("id") != state.item_id:
                reject(
                    _PROTOCOL,
                    f"{parameter}.opaque_state.blob.id",
                    "raw item ID does not match provenance",
                )
            raw_text = _summary_text_for_raw_reasoning(raw_item, parameter=parameter)
            if raw_text != reasoning.text:
                reject(
                    _PROTOCOL,
                    f"{parameter}.text",
                    "reasoning summary differs from the preserved raw item",
                )
            return raw_item
        if not isinstance(state.blob, str):
            reject(
                _PROTOCOL,
                f"{parameter}.opaque_state.blob",
                "legacy Responses state must be text or a full reasoning item",
            )
        payload["id"] = state.item_id
        payload["encrypted_content"] = state.blob
    return payload


def _summary_text_for_raw_reasoning(
    raw_item: Mapping[str, Any],
    *,
    parameter: str,
) -> str:
    summary = raw_item.get("summary", [])
    if not isinstance(summary, list):
        reject(
            _PROTOCOL,
            f"{parameter}.opaque_state.blob.summary",
            "must be an array",
        )
    texts = []
    for index, raw_part in enumerate(summary):
        if not isinstance(raw_part, Mapping):
            reject(
                _PROTOCOL,
                f"{parameter}.opaque_state.blob.summary[{index}]",
                "must be an object",
            )
        if raw_part.get("type") == "summary_text":
            text = raw_part.get("text")
            if not isinstance(text, str):
                reject(
                    _PROTOCOL,
                    f"{parameter}.opaque_state.blob.summary[{index}].text",
                    "must be a string",
                )
            texts.append(text)
    return "".join(texts)


def _encode_tool(tool: ToolDefinition) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "parameters": thaw_json(tool.input_schema),
    }
    if tool.description is not None:
        payload["description"] = tool.description
    if tool.strict is not None:
        payload["strict"] = tool.strict
    return payload


def _decode_response(
    payload: Mapping[str, Any],
    *,
    provider_name: str | None,
    binding_id: str | None,
) -> SemanticResponse:
    body = require_mapping(payload, protocol=_PROTOCOL, parameter="response")
    reject_unknown_keys(
        body,
        _RESPONSE_FIELDS,
        protocol=_PROTOCOL,
        parameter="response",
    )
    model = require_string(
        body.get("model"),
        protocol=_PROTOCOL,
        parameter="response.model",
        allow_empty=False,
    )
    outputs = require_list(body.get("output"), protocol=_PROTOCOL, parameter="response.output")
    decoded = []
    for index, raw_item in enumerate(outputs):
        path = f"response.output[{index}]"
        item = require_mapping(raw_item, protocol=_PROTOCOL, parameter=path)
        item_type = require_string(item.get("type"), protocol=_PROTOCOL, parameter=f"{path}.type")
        if item_type == "message":
            decoded.append(_decode_message(item, parameter=path))
        elif item_type in _TOOL_CALL_OUTPUT_TYPES:
            decoded.append(_decode_tool_call(item, parameter=path))
        elif item_type == "reasoning":
            decoded.append(
                _decode_reasoning_item(
                    item,
                    parameter=path,
                    model=model,
                    provider_name=provider_name,
                    binding_id=binding_id,
                )
            )
        else:
            decode_reject(_PROTOCOL, f"{path}.type", f"unsupported output item {item_type!r}")
    status = require_string(body.get("status"), protocol=_PROTOCOL, parameter="response.status")
    error = body.get("error")
    error_code = error_message = None
    if error is not None:
        error_object = require_mapping(error, protocol=_PROTOCOL, parameter="response.error")
        reject_unknown_keys(
            error_object,
            frozenset({"code", "message", "type", "param"}),
            protocol=_PROTOCOL,
            parameter="response.error",
        )
        error_code = optional_string(
            error_object.get("code"), protocol=_PROTOCOL, parameter="response.error.code"
        )
        error_message = optional_string(
            error_object.get("message"),
            protocol=_PROTOCOL,
            parameter="response.error.message",
        )
    incomplete_details = body.get("incomplete_details")
    finish_reason = _finish_reason(
        status,
        incomplete_details,
        has_tool_calls=any(isinstance(item, ToolCall) for item in decoded),
    )
    metadata: dict[str, Any] = {}
    if "created_at" in body:
        metadata["created_at"] = optional_number(
            body.get("created_at"), protocol=_PROTOCOL, parameter="response.created_at"
        )
    if "object" in body:
        metadata["object"] = require_string(
            body.get("object"), protocol=_PROTOCOL, parameter="response.object"
        )
    return SemanticResponse(
        id=require_string(
            body.get("id"), protocol=_PROTOCOL, parameter="response.id", allow_empty=False
        ),
        model=model,
        output=tuple(decoded),
        usage=decode_usage(
            body.get("usage"),
            protocol=_PROTOCOL,
            input_field="input_tokens",
            output_field="output_tokens",
            input_details_field="input_tokens_details",
            output_details_field="output_tokens_details",
        ),
        terminal=TerminalMetadata(
            finish_reason=finish_reason,
            error_code=error_code,
            error_message=error_message,
            response_status=status,
            transport_termination="explicit_terminal",
            incomplete_details=(
                require_mapping(
                    incomplete_details,
                    protocol=_PROTOCOL,
                    parameter="response.incomplete_details",
                )
                if incomplete_details is not None
                else None
            ),
        ),
        metadata=metadata,
    )


def _finish_reason(
    status: str,
    incomplete_details: object,
    *,
    has_tool_calls: bool = False,
) -> str | None:
    if status == "completed":
        return "tool_calls" if has_tool_calls else "stop"
    if status == "incomplete":
        details = require_mapping(
            incomplete_details,
            protocol=_PROTOCOL,
            parameter="response.incomplete_details",
        )
        reject_unknown_keys(
            details,
            frozenset({"reason"}),
            protocol=_PROTOCOL,
            parameter="response.incomplete_details",
        )
        reason = optional_string(
            details.get("reason"),
            protocol=_PROTOCOL,
            parameter="response.incomplete_details.reason",
        )
        return _decode_incomplete_reason(reason)
    if status in {"failed", "cancelled"}:
        return None
    decode_reject(_PROTOCOL, "response.status", f"unsupported terminal status {status!r}")


def _encode_incomplete_reason(finish_reason: str | None) -> str | None:
    if finish_reason == "length":
        return "max_output_tokens"
    return finish_reason


def _decode_incomplete_reason(reason: str | None) -> str | None:
    if reason == "max_output_tokens":
        return "length"
    return reason


def _encode_response(response: SemanticResponse) -> dict[str, Any]:
    if response.id is None:
        reject(_PROTOCOL, "response.id", "Responses requires an ID")
    output = _encode_response_output(response)
    terminal = response.terminal
    status = terminal.response_status if terminal and terminal.response_status else "completed"
    unknown = sorted(set(response.metadata) - {"object", "created", "created_at"})
    if unknown:
        reject(_PROTOCOL, f"response.metadata.{unknown[0]}", "metadata is not portable")
    created_at = response.metadata.get(
        "created_at",
        response.metadata.get("created", int(time.time())),
    )
    if (
        "created" in response.metadata
        and "created_at" in response.metadata
        and response.metadata["created"] != response.metadata["created_at"]
    ):
        reject(_PROTOCOL, "response.metadata.created", "conflicts with created_at")
    if not isinstance(created_at, int | float) or isinstance(created_at, bool):
        source = "created_at" if "created_at" in response.metadata else "created"
        reject(_PROTOCOL, f"response.metadata.{source}", "must be a number")
    source_object = response.metadata.get("object")
    if source_object is not None and source_object not in {"response", "chat.completion"}:
        reject(_PROTOCOL, "response.metadata.object", "metadata is not portable")
    payload: dict[str, Any] = {
        "id": response.id,
        "object": "response",
        "created_at": created_at,
        "model": response.model,
        "status": status,
        "output": output,
        "usage": encode_usage(
            response.usage,
            protocol=_PROTOCOL,
            input_field="input_tokens",
            output_field="output_tokens",
            input_details_field="input_tokens_details",
            output_details_field="output_tokens_details",
        ),
        "error": None,
        "incomplete_details": None,
    }
    if terminal is not None:
        if terminal.transport_status is not None:
            reject(
                _PROTOCOL,
                "response.terminal.transport_status",
                "transport status is not a Responses payload field",
            )
        if terminal.transport_termination not in {None, "explicit_terminal"}:
            reject(
                _PROTOCOL,
                "response.terminal.transport_termination",
                "a Responses payload can carry only an explicit terminal outcome",
            )
        if terminal.error_code is not None or terminal.error_message is not None:
            payload["error"] = {
                "code": terminal.error_code,
                "message": terminal.error_message,
            }
        if status == "incomplete":
            if terminal.incomplete_details is not None:
                payload["incomplete_details"] = thaw_json(terminal.incomplete_details)
            else:
                reason = _encode_incomplete_reason(terminal.finish_reason)
                payload["incomplete_details"] = {"reason": reason}
    return payload


def _encode_response_output(response: SemanticResponse) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    loose_parts: list[TextContent | RefusalContent] = []
    generated_message_index = 0

    def append_message(
        parts: list[TextContent | RefusalContent],
        *,
        parameter: str,
        item_id: str | None = None,
        status: str | None = None,
    ) -> None:
        nonlocal generated_message_index
        output.append(
            {
                "type": "message",
                "id": item_id or f"msg_{generated_message_index}",
                "role": "assistant",
                "status": status or "completed",
                "content": [
                    _encode_content_part(
                        part,
                        role=MessageRole.ASSISTANT,
                        parameter=f"{parameter}[{part_index}]",
                    )
                    for part_index, part in enumerate(parts)
                ],
            }
        )
        generated_message_index += 1

    def flush_loose() -> None:
        nonlocal loose_parts
        if loose_parts:
            append_message(loose_parts, parameter="response.output")
            loose_parts = []

    for index, item in enumerate(response.output):
        path = f"response.output[{index}]"
        if isinstance(item, SemanticMessage):
            flush_loose()
            if item.role is not MessageRole.ASSISTANT:
                reject(_PROTOCOL, f"{path}.role", "response messages must be assistant")
            if item.name is not None:
                reject(_PROTOCOL, f"{path}.name", "Responses output messages lack names")
            encoded_parts: list[list[TextContent | RefusalContent] | dict[str, Any]] = []
            message_parts: list[TextContent | RefusalContent] = []

            def flush_message_parts() -> None:
                nonlocal message_parts
                if message_parts:
                    encoded_parts.append(message_parts)
                    message_parts = []

            for part_index, part in enumerate(item.content):
                part_path = f"{path}.content[{part_index}]"
                if isinstance(part, TextContent | RefusalContent):
                    message_parts.append(part)
                elif isinstance(part, ToolCall):
                    flush_message_parts()
                    encoded_parts.append(_encode_tool_call(part, parameter=part_path))
                elif isinstance(part, ReasoningSummary):
                    flush_message_parts()
                    encoded_parts.append(
                        _encode_reasoning_item(part, parameter=part_path, model=response.model)
                    )
                else:
                    reject(_PROTOCOL, part_path, "unsupported response message content")
            flush_message_parts()
            if not encoded_parts:
                encoded_parts.append([])
            message_segments = [part for part in encoded_parts if isinstance(part, list)]
            if (item.item_id is not None or item.status is not None) and len(message_segments) != 1:
                reject(
                    _PROTOCOL,
                    path,
                    "message metadata cannot survive splitting around sibling output items",
                )
            for encoded_part in encoded_parts:
                if isinstance(encoded_part, list):
                    append_message(
                        encoded_part,
                        parameter=f"{path}.content",
                        item_id=item.item_id,
                        status=item.status,
                    )
                else:
                    output.append(encoded_part)
        elif isinstance(item, ToolCall):
            flush_loose()
            output.append(_encode_tool_call(item, parameter=path))
        elif isinstance(item, ReasoningSummary):
            flush_loose()
            output.append(_encode_reasoning_item(item, parameter=path, model=response.model))
        elif isinstance(item, TextContent | RefusalContent):
            loose_parts.append(item)
        else:
            reject(_PROTOCOL, path, f"unsupported response item {type(item).__name__}")
    flush_loose()
    return output


def responses_request_to_semantic(
    request: ResponsesRequest,
    *,
    provider_name: str | None = "openai",
    binding_id: str | None = None,
) -> SemanticRequest:
    """Convert the legacy provider-facing Responses request into semantic IR."""
    payload: dict[str, Any] = {
        "model": request.model,
        "input": request.input,
        "stream": request.stream,
    }
    for key, value in {
        "instructions": request.instructions,
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
        "tools": request.tools,
        "tool_choice": request.tool_choice,
        "parallel_tool_calls": request.parallel_tool_calls,
        "top_p": request.top_p,
        "metadata": request.metadata,
        "service_tier": request.service_tier,
    }.items():
        if value is not None:
            payload[key] = value
    if request.reasoning_effort is not None:
        payload["reasoning"] = {"effort": request.reasoning_effort}
    semantic = _decode_request(
        payload,
        provider_name=provider_name,
        binding_id=binding_id,
    )
    return replace(semantic, provider_extensions=request.provider_extensions)


def semantic_to_responses_request(request: SemanticRequest) -> ResponsesRequest:
    """Convert semantic IR into the legacy provider-facing Responses request."""
    # Provider extensions are an internal DTO escape hatch, not portable
    # Responses wire fields. Preserve them alongside a strict core projection.
    payload = _encode_request(replace(request, provider_extensions={}))
    return ResponsesRequest(
        model=payload["model"],
        input=payload["input"],
        stream=payload.get("stream", False),
        instructions=payload.get("instructions"),
        temperature=payload.get("temperature"),
        max_output_tokens=payload.get("max_output_tokens"),
        tools=payload.get("tools"),
        tool_choice=payload.get("tool_choice"),
        parallel_tool_calls=payload.get("parallel_tool_calls"),
        reasoning_effort=(payload.get("reasoning") or {}).get("effort"),
        top_p=payload.get("top_p"),
        metadata=payload.get("metadata"),
        service_tier=payload.get("service_tier"),
        provider_extensions=thaw_json(request.provider_extensions),
    )


def responses_response_to_semantic(
    response: ResponsesResponse,
    *,
    response_id: str,
    origin_provider: str | None,
    origin_binding: str | None = None,
) -> SemanticResponse:
    """Convert a legacy Responses response into semantic IR."""
    output: list[SemanticItem] = []
    if response.reasoning_item is not None:
        raw_item = require_mapping(
            response.reasoning_item,
            protocol=_PROTOCOL,
            parameter="response.reasoning_item",
        )
        reasoning = _decode_reasoning_item(
            raw_item,
            parameter="response.reasoning_item",
            model=response.model,
            provider_name=origin_provider,
            binding_id=origin_binding,
        )
        _validate_legacy_reasoning_views(response, raw_item, reasoning)
        output.append(reasoning)
    elif response.thinking is not None or response.thinking_signature is not None:
        opaque = None
        if response.thinking_id is not None:
            synthesized_item: dict[str, Any] = {
                "type": "reasoning",
                "id": response.thinking_id,
                "summary": (
                    [{"type": "summary_text", "text": response.thinking}]
                    if response.thinking
                    else []
                ),
            }
            if response.thinking_signature is not None:
                synthesized_item["encrypted_content"] = response.thinking_signature
            opaque = OpaqueState(
                origin_protocol=_PROTOCOL,
                origin_provider=origin_provider,
                origin_model=response.model,
                item_id=response.thinking_id,
                blob=synthesized_item,
                origin_binding=origin_binding,
            )
        elif response.thinking_signature is not None:
            if response.thinking_id is None:
                reject(
                    _PROTOCOL,
                    "response.thinking_signature",
                    "opaque reasoning requires thinking_id",
                )
        output.append(ReasoningSummary(response.thinking or "", opaque_state=opaque))
    message_parts = []
    if response.content:
        message_parts.append(TextContent(response.content))
    if response.refusal is not None:
        message_parts.append(RefusalContent(response.refusal))
    if message_parts:
        output.append(SemanticMessage(role=MessageRole.ASSISTANT, content=tuple(message_parts)))
    for raw_call in response.tool_calls or []:
        output.append(
            ToolCall(
                call_id=raw_call.call_id,
                name=raw_call.name,
                arguments=(
                    {"input": raw_call.arguments}
                    if raw_call.kind == "custom"
                    else parse_arguments(
                        raw_call.arguments,
                        protocol=_PROTOCOL,
                        parameter="response.tool_calls.arguments",
                    )
                ),
                kind=raw_call.kind,
                namespace=raw_call.namespace,
            )
        )
    outcome = resolve_terminal_outcome(
        response.terminal_outcome,
        response.finish_reason,
    )
    if outcome is not None:
        terminal, _ = terminal_event_values(outcome)
    else:
        terminal = TerminalMetadata(
            response_status="completed",
            transport_termination="explicit_terminal",
        )
    usage = decode_usage(
        response.usage,
        protocol=_PROTOCOL,
        input_field="input_tokens",
        output_field="output_tokens",
        input_details_field="input_tokens_details",
        output_details_field="output_tokens_details",
    )
    return SemanticResponse(
        id=response_id,
        model=response.model,
        output=tuple(output),
        usage=usage,
        terminal=terminal,
    )


def _validate_legacy_reasoning_views(
    response: ResponsesResponse,
    raw_item: Mapping[str, Any],
    reasoning: ReasoningSummary,
) -> None:
    """Reject only identity/blob contradictions; the full item owns all other fields."""
    state = reasoning.opaque_state
    if state is None:  # pragma: no cover - _decode_reasoning_item always attaches state
        return
    if response.thinking_id is not None and response.thinking_id != state.item_id:
        reject(
            _PROTOCOL,
            "response.thinking_id",
            "does not match response.reasoning_item.id",
        )
    raw_signature = raw_item.get("encrypted_content")
    if response.thinking_signature is not None and response.thinking_signature != raw_signature:
        reject(
            _PROTOCOL,
            "response.thinking_signature",
            "does not match response.reasoning_item.encrypted_content",
        )


def semantic_to_responses_response(response: SemanticResponse) -> ResponsesResponse:
    """Convert semantic IR into the legacy provider-facing Responses response DTO."""
    for index, item in enumerate(response.output):
        parts = item.content if isinstance(item, SemanticMessage) else (item,)
        for part_index, part in enumerate(parts):
            if not isinstance(part, ToolCall):
                continue
            path = (
                f"response.output[{index}].content[{part_index}]"
                if isinstance(item, SemanticMessage)
                else f"response.output[{index}]"
            )
            if part.item_id is not None:
                reject(
                    _PROTOCOL,
                    f"{path}.item_id",
                    "legacy ResponsesToolCall cannot preserve item IDs",
                )
            if part.opaque_state is not None:
                reject(
                    _PROTOCOL,
                    f"{path}.opaque_state",
                    "legacy ResponsesToolCall cannot preserve opaque tool state",
                )
    terminal_outcome = None
    if response.terminal is not None:
        terminal_outcome = terminal_outcome_from_event(
            SemanticEvent(
                type=SemanticEventType.TERMINAL,
                terminal=response.terminal,
            ),
            protocol=_PROTOCOL,
        )
    wire_terminal = response.terminal
    if wire_terminal is not None and wire_terminal.transport_termination not in {
        None,
        "explicit_terminal",
    }:
        wire_terminal = TerminalMetadata(
            finish_reason=wire_terminal.finish_reason,
            error_code=wire_terminal.error_code,
            error_message=wire_terminal.error_message,
            response_status=wire_terminal.response_status,
            incomplete_details=wire_terminal.incomplete_details,
        )
    encoded = _encode_response(replace(response, terminal=wire_terminal))
    content = []
    refusal = []
    reasoning = []
    opaque_states = []
    calls = []
    for item in response.output:
        parts = item.content if isinstance(item, SemanticMessage) else (item,)
        for part in parts:
            if isinstance(part, TextContent):
                content.append(part.text)
            elif isinstance(part, RefusalContent):
                refusal.append(part.refusal)
            elif isinstance(part, ReasoningSummary):
                reasoning.append(part.text)
                if part.opaque_state is not None:
                    opaque_states.append(part.opaque_state)
            elif isinstance(part, ToolCall):
                arguments = (
                    thaw_json(part.arguments).get("input", "")
                    if part.kind == "custom"
                    else encode_arguments(part.arguments)
                )
                calls.append(
                    ResponsesToolCall(
                        call_id=part.call_id,
                        name=part.name,
                        arguments=arguments,
                        kind=cast(Literal["function", "custom", "tool_search"], part.kind),
                        namespace=part.namespace,
                    )
                )
    if len(opaque_states) > 1:
        reject(
            _PROTOCOL,
            "response.output",
            "legacy ResponsesResponse can preserve only one raw reasoning item",
        )
    opaque = opaque_states[0] if opaque_states else None
    encrypted_content = None
    reasoning_item = None
    if opaque is not None:
        if isinstance(opaque.blob, Mapping):
            raw_item = thaw_json(opaque.blob)
            reasoning_item = raw_item
            encrypted = raw_item.get("encrypted_content")
            if encrypted is not None and not isinstance(encrypted, str):
                reject(
                    _PROTOCOL,
                    "response.reasoning.encrypted_content",
                    "must be a string",
                )
            encrypted_content = encrypted
        elif isinstance(opaque.blob, str):
            encrypted_content = opaque.blob
        else:
            reject(
                _PROTOCOL,
                "response.reasoning",
                "legacy ResponsesResponse requires text encrypted content",
            )
    finish_reason = response.terminal.finish_reason if response.terminal else None
    return ResponsesResponse(
        content="".join(content),
        model=response.model,
        usage=encoded.get("usage"),
        tool_calls=calls or None,
        thinking="".join(reasoning) or None,
        thinking_id=opaque.item_id if opaque is not None else None,
        thinking_signature=encrypted_content,
        finish_reason=finish_reason,
        terminal_outcome=terminal_outcome,
        refusal="".join(refusal) or None,
        reasoning_item=reasoning_item,
    )


def responses_chunk_to_semantic_events(
    chunk: ResponsesStreamChunk,
    *,
    sequence_start: int = 0,
    response_id: str | None = None,
    model: str | None = None,
    origin_provider: str | None = None,
    origin_binding: str | None = None,
) -> tuple[SemanticEvent, ...]:
    """Project one typed Responses chunk to ordered semantic events.

    The legacy chunk carries complete tool calls and source indices. New
    producers attach the full raw reasoning item; id/signature-only chunks are
    still accepted for compatibility with older providers.
    """
    if chunk.provenance_only and any(
        (
            chunk.content,
            chunk.refusal,
            chunk.thinking,
            chunk.thinking_id,
            chunk.thinking_signature,
            chunk.reasoning_item is not None,
            chunk.tool_call,
        )
    ):
        decode_reject(
            _PROTOCOL,
            "chunk.provenance_only",
            "cannot accompany a user-visible payload",
        )

    events: list[SemanticEvent] = []
    provenance = _chunk_provenance(chunk)

    def emit(event_type: SemanticEventType, **values: Any) -> None:
        metadata = dict(provenance)
        metadata.update(values.pop("metadata", {}))
        events.append(
            SemanticEvent(
                type=event_type,
                sequence=sequence_start + len(events),
                response_id=response_id,
                output_index=chunk.output_index,
                content_index=(
                    chunk.content_index
                    if chunk.content_index is not None
                    else chunk.reasoning_summary_index
                ),
                metadata=metadata,
                **values,
            )
        )

    if chunk.thinking:
        _require_chunk_item_type(chunk, "reasoning", "chunk.thinking")
        emit(SemanticEventType.REASONING_DELTA, delta=chunk.thinking)
    if chunk.reasoning_item is not None:
        _require_chunk_item_type(chunk, "reasoning", "chunk.reasoning_item")
        if model is None:
            reject(
                _PROTOCOL,
                "chunk.reasoning_item",
                "opaque reasoning requires model provenance",
            )
        raw_item = require_mapping(
            chunk.reasoning_item,
            protocol=_PROTOCOL,
            parameter="chunk.reasoning_item",
        )
        decoded = _decode_reasoning_item(
            raw_item,
            parameter="chunk.reasoning_item",
            model=model,
            provider_name=origin_provider,
            binding_id=origin_binding,
        )
        state = decoded.opaque_state
        if state is None:  # pragma: no cover - decoder always attaches state
            raise RuntimeError("reasoning decoder did not preserve opaque state")
        if chunk.thinking_id is not None and chunk.thinking_id != state.item_id:
            reject(
                _PROTOCOL,
                "chunk.thinking_id",
                "does not match chunk.reasoning_item.id",
            )
        raw_signature = raw_item.get("encrypted_content")
        if chunk.thinking_signature is not None and chunk.thinking_signature != raw_signature:
            reject(
                _PROTOCOL,
                "chunk.thinking_signature",
                "does not match chunk.reasoning_item.encrypted_content",
            )
        emit(
            SemanticEventType.OUTPUT_ITEM,
            item_id=state.item_id,
            item=ReasoningSummary("", opaque_state=state),
        )
    elif chunk.thinking_signature is not None:
        _require_chunk_item_type(chunk, "reasoning", "chunk.thinking_signature")
        if chunk.thinking_id is None or model is None:
            reject(
                _PROTOCOL,
                "chunk.thinking_signature",
                "opaque reasoning requires thinking_id and model",
            )
        raw_item = {
            "type": "reasoning",
            "id": chunk.thinking_id,
            "summary": [],
            "encrypted_content": chunk.thinking_signature,
        }
        emit(
            SemanticEventType.OUTPUT_ITEM,
            item_id=chunk.thinking_id,
            item=ReasoningSummary(
                "",
                opaque_state=OpaqueState(
                    origin_protocol=_PROTOCOL,
                    origin_provider=origin_provider,
                    origin_model=model,
                    item_id=chunk.thinking_id,
                    blob=raw_item,
                    origin_binding=origin_binding,
                ),
            ),
        )
    elif chunk.thinking_id is not None:
        _require_chunk_item_type(chunk, "reasoning", "chunk.thinking_id")
        emit(SemanticEventType.OUTPUT_ITEM, item_id=chunk.thinking_id)
    if chunk.content:
        _require_chunk_item_type(chunk, "message", "chunk.content")
        emit(SemanticEventType.TEXT_DELTA, delta=chunk.content)
    if chunk.refusal:
        _require_chunk_item_type(chunk, "message", "chunk.refusal")
        emit(SemanticEventType.OUTPUT_ITEM, item=RefusalContent(chunk.refusal))
    if chunk.tool_call is not None:
        expected_type = {
            "function": "function_call",
            "custom": "custom_tool_call",
            "tool_search": "tool_search_call",
        }.get(chunk.tool_call.kind)
        if expected_type is None:
            decode_reject(
                _PROTOCOL,
                "chunk.tool_call.kind",
                f"unsupported kind {chunk.tool_call.kind!r}",
            )
        _require_chunk_item_type(chunk, expected_type, "chunk.tool_call")
        emit(
            SemanticEventType.OUTPUT_ITEM,
            item=_semantic_tool_call_from_responses(chunk.tool_call),
        )

    has_payload_event = bool(events)
    if not has_payload_event and (
        chunk.provenance_only
        or chunk.output_item_done
        or chunk.output_item_type is not None
        or chunk.output_index is not None
        or chunk.content_index is not None
        or chunk.reasoning_summary_index is not None
    ):
        emit(SemanticEventType.OUTPUT_ITEM)
    if chunk.usage is not None:
        emit(
            SemanticEventType.USAGE,
            usage=decode_usage(
                chunk.usage,
                protocol=_PROTOCOL,
                input_field="input_tokens",
                output_field="output_tokens",
                input_details_field="input_tokens_details",
                output_details_field="output_tokens_details",
            ),
        )
    outcome = resolve_terminal_outcome(chunk.terminal_outcome, chunk.finish_reason)
    if outcome is not None:
        terminal, metadata = terminal_event_values(outcome)
        if terminal.error_code is not None or terminal.error_message is not None:
            emit(SemanticEventType.ERROR, terminal=terminal, metadata=metadata)
        emit(SemanticEventType.TERMINAL, terminal=terminal, metadata=metadata)
    return tuple(events)


def _chunk_provenance(chunk: ResponsesStreamChunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if chunk.output_item_type is not None:
        metadata["output_item_type"] = chunk.output_item_type
    if chunk.output_item_done:
        metadata["output_item_done"] = True
    if chunk.provenance_only:
        metadata["provenance_only"] = True
    if chunk.reasoning_summary_index is not None:
        metadata["reasoning_summary_index"] = chunk.reasoning_summary_index
    return metadata


def _require_chunk_item_type(
    chunk: ResponsesStreamChunk,
    expected: str,
    parameter: str,
) -> None:
    if chunk.output_item_type not in {None, expected}:
        decode_reject(
            _PROTOCOL,
            "chunk.output_item_type",
            f"{parameter} requires {expected!r}",
        )


def _semantic_tool_call_from_responses(call: ResponsesToolCall) -> ToolCall:
    if not call.call_id:
        decode_reject(_PROTOCOL, "chunk.tool_call.call_id", "must be non-empty")
    if not call.name:
        decode_reject(_PROTOCOL, "chunk.tool_call.name", "must be non-empty")
    if call.kind == "custom":
        if call.namespace is not None:
            decode_reject(
                _PROTOCOL,
                "chunk.tool_call.namespace",
                "custom calls cannot carry namespaces",
            )
        arguments: Mapping[str, Any] = {"input": call.arguments}
    elif call.kind in {"function", "tool_search"}:
        if call.kind == "tool_search" and call.namespace is not None:
            decode_reject(
                _PROTOCOL,
                "chunk.tool_call.namespace",
                "tool-search calls cannot carry namespaces",
            )
        arguments = parse_arguments(
            call.arguments,
            protocol=_PROTOCOL,
            parameter="chunk.tool_call.arguments",
        )
    else:
        decode_reject(
            _PROTOCOL,
            "chunk.tool_call.kind",
            f"unsupported kind {call.kind!r}",
        )
    return ToolCall(
        call_id=call.call_id,
        name=call.name,
        arguments=arguments,
        kind=call.kind,
        namespace=call.namespace,
    )


def semantic_events_to_responses_chunks(
    events: tuple[SemanticEvent, ...] | list[SemanticEvent],
) -> tuple[ResponsesStreamChunk, ...]:
    """Project semantic events to legacy Responses chunks in source order."""
    event_list = tuple(events)
    chunks: list[ResponsesStreamChunk] = []
    terminal_indices = {
        index for index, event in enumerate(event_list) if event.type is SemanticEventType.TERMINAL
    }
    for index, event in enumerate(event_list):
        if event.type is SemanticEventType.RESPONSE_STARTED:
            continue
        if event.type is SemanticEventType.TEXT_DELTA:
            chunks.append(
                ResponsesStreamChunk(
                    content=event.delta or "",
                    output_index=event.output_index,
                    content_index=event.content_index,
                    output_item_type=_event_item_type(event, "message"),
                )
            )
        elif event.type is SemanticEventType.REASONING_DELTA:
            chunks.append(
                ResponsesStreamChunk(
                    content="",
                    thinking=event.delta or "",
                    output_index=event.output_index,
                    reasoning_summary_index=_reasoning_summary_index(event),
                    output_item_type=_event_item_type(event, "reasoning"),
                )
            )
        elif event.type is SemanticEventType.TOOL_ARGUMENTS_DELTA:
            reject(
                _PROTOCOL,
                "event.type",
                "legacy ResponsesStreamChunk carries only complete tool calls",
            )
        elif event.type is SemanticEventType.OUTPUT_ITEM:
            chunks.extend(_responses_chunks_for_item(event))
        elif event.type is SemanticEventType.USAGE:
            if event.usage is None:
                reject(_PROTOCOL, "event.usage", "usage event requires Usage")
            chunks.append(
                ResponsesStreamChunk(
                    content="",
                    usage=encode_stream_usage(
                        event.usage,
                        protocol=_PROTOCOL,
                        input_field="input_tokens",
                        output_field="output_tokens",
                        input_details_field="input_tokens_details",
                        output_details_field="output_tokens_details",
                    ),
                )
            )
        elif event.type is SemanticEventType.ERROR:
            if not any(terminal_index > index for terminal_index in terminal_indices):
                chunks.append(
                    ResponsesStreamChunk(
                        content="",
                        terminal_outcome=terminal_outcome_from_event(event, protocol=_PROTOCOL),
                    )
                )
        elif event.type is SemanticEventType.TERMINAL:
            outcome = terminal_outcome_from_event(event, protocol=_PROTOCOL)
            chunks.append(
                ResponsesStreamChunk(
                    content="",
                    finish_reason=outcome.finish_reason,
                    terminal_outcome=outcome,
                )
            )
    return tuple(chunks)


def _responses_chunks_for_item(event: SemanticEvent) -> list[ResponsesStreamChunk]:
    item = event.item
    output_item_type = event.metadata.get("output_item_type")
    output_item_done = event.metadata.get("output_item_done") is True
    provenance_only = event.metadata.get("provenance_only") is True
    common = {
        "output_index": event.output_index,
        "content_index": event.content_index,
        "output_item_done": output_item_done,
        "provenance_only": provenance_only,
    }
    if item is None:
        return [
            ResponsesStreamChunk(
                content="",
                output_item_type=(output_item_type if isinstance(output_item_type, str) else None),
                reasoning_summary_index=_reasoning_summary_index(event),
                **common,
            )
        ]
    if isinstance(item, TextContent):
        return [
            ResponsesStreamChunk(
                content=item.text,
                output_item_type=_event_item_type(event, "message"),
                **common,
            )
        ]
    if isinstance(item, RefusalContent):
        return [
            ResponsesStreamChunk(
                content="",
                refusal=item.refusal,
                output_item_type=_event_item_type(event, "message"),
                **common,
            )
        ]
    if isinstance(item, ReasoningSummary):
        state = item.opaque_state
        thinking_id = event.item_id
        encrypted_content = None
        reasoning_item = None
        if state is not None:
            if state.origin_protocol is not _PROTOCOL:
                reject(
                    _PROTOCOL,
                    "event.item.opaque_state",
                    "opaque reasoning originated from another protocol",
                )
            if thinking_id is not None and thinking_id != state.item_id:
                reject(
                    _PROTOCOL,
                    "event.item_id",
                    "does not match opaque reasoning provenance",
                )
            thinking_id = state.item_id
            if isinstance(state.blob, Mapping):
                raw_item = thaw_json(state.blob)
                reasoning_item = raw_item
                if raw_item.get("id") != state.item_id:
                    reject(
                        _PROTOCOL,
                        "event.item.opaque_state.blob.id",
                        "does not match opaque reasoning provenance",
                    )
                status = raw_item.get("status")
                if status not in {None, "completed"}:
                    reject(
                        _PROTOCOL,
                        "event.item.opaque_state.blob.status",
                        "legacy done chunks require completed status",
                    )
                encrypted_content = raw_item.get("encrypted_content")
                if encrypted_content is not None and not isinstance(encrypted_content, str):
                    reject(
                        _PROTOCOL,
                        "event.item.opaque_state.blob.encrypted_content",
                        "must be a string",
                    )
            elif isinstance(state.blob, str):
                encrypted_content = state.blob
            else:
                reject(
                    _PROTOCOL,
                    "event.item.opaque_state.blob",
                    "legacy ResponsesStreamChunk requires text encrypted content",
                )
        return [
            ResponsesStreamChunk(
                content="",
                thinking=item.text or None,
                thinking_id=thinking_id,
                thinking_signature=encrypted_content,
                reasoning_item=reasoning_item,
                output_item_type=_event_item_type(event, "reasoning"),
                reasoning_summary_index=_reasoning_summary_index(event),
                **common,
            )
        ]
    if isinstance(item, ToolCall):
        if item.item_id is not None:
            reject(
                _PROTOCOL,
                "event.item.item_id",
                "legacy ResponsesToolCall cannot preserve output item IDs",
            )
        if item.opaque_state is not None:
            reject(
                _PROTOCOL,
                "event.item.opaque_state",
                "legacy ResponsesToolCall cannot preserve opaque tool state",
            )
        if item.kind == "custom":
            raw_arguments = thaw_json(item.arguments)
            if set(raw_arguments) != {"input"} or not isinstance(raw_arguments["input"], str):
                reject(
                    _PROTOCOL,
                    "event.item.arguments",
                    "custom calls require exactly one string input field",
                )
            arguments = raw_arguments["input"]
        else:
            arguments = encode_arguments(item.arguments)
        expected_type = {
            "function": "function_call",
            "custom": "custom_tool_call",
            "tool_search": "tool_search_call",
        }.get(item.kind)
        if expected_type is None:
            reject(_PROTOCOL, "event.item.kind", f"unsupported kind {item.kind!r}")
        call = ResponsesToolCall(
            call_id=item.call_id,
            name=item.name,
            arguments=arguments,
            kind=cast(Literal["function", "custom", "tool_search"], item.kind),
            namespace=item.namespace,
        )
        return [
            ResponsesStreamChunk(
                content="",
                tool_call=call,
                output_item_type=_event_item_type(event, expected_type),
                **common,
            )
        ]
    if isinstance(item, SemanticMessage):
        if item.item_id is not None or item.status is not None or item.name is not None:
            reject(
                _PROTOCOL,
                "event.item",
                "legacy ResponsesStreamChunk cannot preserve message metadata",
            )
        chunks = []
        for part in item.content:
            chunks.extend(_responses_chunks_for_item(replace(event, item=part)))
        return chunks
    reject(_PROTOCOL, "event.item", f"unsupported stream item {type(item).__name__}")


def _event_item_type(event: SemanticEvent, default: str) -> str:
    item_type = event.metadata.get("output_item_type", default)
    if not isinstance(item_type, str):
        reject(_PROTOCOL, "event.metadata.output_item_type", "must be a string")
    if item_type != default:
        reject(
            _PROTOCOL,
            "event.metadata.output_item_type",
            f"event payload requires {default!r}",
        )
    return item_type


def _reasoning_summary_index(event: SemanticEvent) -> int | None:
    index = event.metadata.get("reasoning_summary_index", event.content_index)
    if index is not None and (not isinstance(index, int) or isinstance(index, bool)):
        reject(_PROTOCOL, "event.metadata.reasoning_summary_index", "must be an integer")
    return index


__all__ = [
    "OpenAIResponsesRuntime",
    "OpenAIResponsesStreamDecoder",
    "OpenAIResponsesStreamEncoder",
    "responses_chunk_to_semantic_events",
    "responses_request_to_semantic",
    "responses_response_to_semantic",
    "semantic_events_to_responses_chunks",
    "semantic_to_responses_request",
    "semantic_to_responses_response",
]
