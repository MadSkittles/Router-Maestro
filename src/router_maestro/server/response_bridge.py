"""Shared downstream response bridge for protocol-aware generation dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from time import time
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from router_maestro.protocols import (
    ConversionMode,
    ProtocolRuntime,
    ProtocolRuntimeNotFoundError,
    ProtocolRuntimeRegistry,
    SemanticEvent,
    SemanticEventType,
    SemanticResponse,
    TerminalMetadata,
    UnsupportedProtocolOperationError,
    Usage,
    UsageMode,
    WireProtocol,
)
from router_maestro.protocols.legacy import (
    semantic_events_from_legacy_chat_chunk,
    semantic_response_from_legacy_chat,
)
from router_maestro.protocols.openai_responses import (
    responses_chunk_to_semantic_events,
    responses_response_to_semantic,
)
from router_maestro.providers.base import (
    ChatResponse,
    ChatStreamChunk,
    ProviderError,
    ProviderFailureKind,
    ResponsesResponse,
    ResponsesStreamChunk,
)
from router_maestro.routing.transport_plan import TransportPlan
from router_maestro.server.dispatcher import (
    DispatchResult,
    OpenedDispatchStream,
    ProtocolRuntimeResolver,
)
from router_maestro.utils.async_iterators import close_async_iterator


@runtime_checkable
class SemanticStreamDecoder(Protocol):
    """Request-local decoder state for one upstream wire stream."""

    def decode(self, payload: Mapping[str, Any]) -> tuple[SemanticEvent, ...]: ...


@runtime_checkable
class SemanticStreamEncoder(Protocol):
    """Request-local encoder state for one downstream wire stream."""

    def encode(self, event: SemanticEvent) -> tuple[Mapping[str, Any], ...]: ...


@runtime_checkable
class StreamDecoderFactory(Protocol):
    def __call__(
        self,
        runtime: ProtocolRuntime,
        plan: TransportPlan,
    ) -> SemanticStreamDecoder: ...


@runtime_checkable
class StreamEncoderFactory(Protocol):
    def __call__(
        self,
        runtime: ProtocolRuntime,
        plan: TransportPlan,
        response_id: str,
        public_model: str,
    ) -> SemanticStreamEncoder: ...


@runtime_checkable
class IdentityTerminalClassifier(Protocol):
    """Identify terminal raw frames without materializing semantic events."""

    def __call__(self, plan: TransportPlan, frame: Mapping[str, Any]) -> bool: ...


@runtime_checkable
class ResponseIdFactory(Protocol):
    def __call__(self, protocol: WireProtocol) -> str: ...


class GenerationResponseBridge:
    """Encode one selected dispatch back to its ingress wire protocol.

    Raw identity responses stay on a copy-on-write fast path. Cross-protocol
    responses, and legacy provider DTOs, pass through semantic response/event
    IR only after the dispatcher has committed a transport selection.
    """

    def __init__(
        self,
        runtimes: Mapping[WireProtocol, ProtocolRuntime] | ProtocolRuntimeRegistry,
        *,
        runtime_resolver: ProtocolRuntimeResolver | None = None,
        stream_decoder_factory: StreamDecoderFactory | None = None,
        stream_encoder_factory: StreamEncoderFactory | None = None,
        identity_terminal_classifier: IdentityTerminalClassifier | None = None,
        response_id_factory: ResponseIdFactory | None = None,
    ) -> None:
        self._runtimes = runtimes
        self._runtime_resolver = runtime_resolver
        self._stream_decoder_factory = stream_decoder_factory or _new_stream_decoder
        self._stream_encoder_factory = stream_encoder_factory or _new_stream_encoder
        self._identity_terminal_classifier = identity_terminal_classifier or _is_identity_terminal
        self._response_id_factory = response_id_factory or _new_response_id

    async def encode_result(
        self,
        result: DispatchResult,
        ingress_runtime: ProtocolRuntime,
    ) -> Mapping[str, Any]:
        """Encode a non-streaming dispatch result for the original client."""
        plan = result.selection.plan
        self._validate_ingress(plan, ingress_runtime)
        value = result.value
        public_model = plan.model.qualified_id

        if plan.conversion_mode is ConversionMode.IDENTITY and isinstance(value, Mapping):
            return _project_raw_identity(
                value,
                plan.source_protocol,
                public_model,
                stream=False,
            )

        semantic = await self._semantic_response(
            value,
            plan,
            response_id=self._response_id_factory(plan.source_protocol),
        )
        semantic = replace(semantic, model=public_model)
        encoded = await ingress_runtime.encode_response(semantic)
        if not isinstance(encoded, Mapping):
            raise TypeError("protocol runtime encode_response must return a mapping")
        return encoded

    async def encode_stream(
        self,
        opened: OpenedDispatchStream,
        ingress_runtime: ProtocolRuntime,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Encode one selected stream and own its upstream iterator lifecycle."""
        plan = opened.selection.plan
        self._validate_ingress(plan, ingress_runtime)
        source = opened.frames
        public_model = plan.model.qualified_id
        response_id = self._response_id_factory(plan.source_protocol)

        mode: str | None = None
        semantic_source: str | None = None
        decoder = opened.semantic_decoder
        first_events = opened.first_events
        if (decoder is None) != (first_events is None):
            raise ValueError("predecoded stream state must include decoder and first events")
        if decoder is not None and not isinstance(decoder, SemanticStreamDecoder):
            raise TypeError("predecoded stream decoder does not implement decode")
        if plan.conversion_mode is ConversionMode.IDENTITY and decoder is not None:
            raise ValueError("identity streams cannot carry semantic decoder state")
        encoder: SemanticStreamEncoder | None = None
        sequence = 0
        terminal_count = 0
        pending_terminal: SemanticEvent | None = None
        post_terminal_usage_allowed = False
        post_terminal_usage_seen = False
        usage_snapshot: Usage | None = None
        identity_response_snapshot: Mapping[str, Any] | None = None
        identity_sequence_number = -1

        try:
            async for frame in source:
                if mode is None:
                    mode = (
                        "raw_identity"
                        if plan.conversion_mode is ConversionMode.IDENTITY
                        and isinstance(frame, Mapping)
                        else "semantic"
                    )
                    if mode == "semantic":
                        encoder = self._stream_encoder_factory(
                            ingress_runtime,
                            plan,
                            response_id,
                            public_model,
                        )

                if mode == "raw_identity":
                    if not isinstance(frame, Mapping):
                        raise TypeError("raw identity streams cannot change frame representation")
                    _validate_identity_stream_frame(plan, frame)
                    is_terminal = self._identity_terminal_classifier(plan, frame)
                    if terminal_count:
                        if is_terminal:
                            raise _stream_protocol_error(
                                plan,
                                "Upstream stream emitted multiple terminal events",
                            )
                        if (
                            not post_terminal_usage_allowed
                            or post_terminal_usage_seen
                            or not _is_allowed_identity_post_terminal_usage(plan, frame)
                        ):
                            raise _stream_protocol_error(
                                plan,
                                "Upstream stream emitted an event after its terminal event",
                            )
                        post_terminal_usage_seen = True
                    elif is_terminal:
                        terminal_count += 1
                        post_terminal_usage_allowed = _identity_terminal_allows_usage_tail(
                            plan, frame
                        )
                    nested_response = frame.get("response")
                    if isinstance(nested_response, Mapping):
                        identity_response_snapshot = nested_response
                    sequence_number = frame.get("sequence_number")
                    if isinstance(sequence_number, int) and not isinstance(sequence_number, bool):
                        identity_sequence_number = max(identity_sequence_number, sequence_number)
                    yield _project_raw_identity(
                        frame,
                        plan.source_protocol,
                        public_model,
                        stream=True,
                    )
                    continue

                frame_source = _semantic_source_kind(frame)
                if semantic_source is None:
                    semantic_source = frame_source
                elif semantic_source != frame_source:
                    raise TypeError("upstream stream cannot mix wire mappings and legacy DTOs")

                if isinstance(frame, Mapping):
                    if first_events is not None:
                        events = first_events
                        first_events = None
                    else:
                        if decoder is None:
                            target_runtime = self._runtime(plan.target_protocol, plan)
                            decoder = self._stream_decoder_factory(target_runtime, plan)
                        events = decoder.decode(frame)
                elif isinstance(frame, ChatStreamChunk):
                    events = semantic_events_from_legacy_chat_chunk(
                        frame,
                        response_id=response_id,
                        model=plan.model.upstream_id,
                        origin_protocol=plan.target_protocol,
                        origin_provider=plan.model.provider,
                        origin_binding=plan.binding.id,
                        sequence_start=sequence,
                    )
                elif isinstance(frame, ResponsesStreamChunk):
                    events = responses_chunk_to_semantic_events(
                        frame,
                        response_id=response_id,
                        model=plan.model.upstream_id,
                        origin_provider=plan.model.provider,
                        origin_binding=plan.binding.id,
                        sequence_start=sequence,
                    )
                else:  # pragma: no cover - _semantic_source_kind guards this branch
                    raise TypeError(f"unsupported upstream stream frame {type(frame).__name__}")

                if not isinstance(events, tuple) or not all(
                    isinstance(event, SemanticEvent) for event in events
                ):
                    raise TypeError("stream decoder must return a tuple of SemanticEvent")
                terminal_events = tuple(
                    event for event in events if event.type is SemanticEventType.TERMINAL
                )
                if terminal_count:
                    if terminal_events:
                        raise _stream_protocol_error(
                            plan,
                            "Upstream stream emitted multiple terminal events",
                        )
                    if (
                        not post_terminal_usage_allowed
                        or post_terminal_usage_seen
                        or len(events) != 1
                        or events[0].type is not SemanticEventType.USAGE
                        or events[0].usage is None
                    ):
                        raise _stream_protocol_error(
                            plan,
                            "Upstream stream emitted an event after its terminal event",
                        )
                    post_terminal_usage_seen = True
                elif terminal_events:
                    if len(terminal_events) != 1:
                        raise _stream_protocol_error(
                            plan,
                            "Upstream stream emitted multiple terminal events",
                        )
                    terminal_event = terminal_events[0]
                    if events[-1] is not terminal_event:
                        raise _stream_protocol_error(
                            plan,
                            "Upstream stream emitted an event after its terminal event",
                        )
                    post_terminal_usage_allowed = _semantic_terminal_allows_usage_tail(
                        plan,
                        events,
                        terminal_event,
                    )
                sequence += len(events)
                if encoder is None:  # pragma: no cover - initialized with semantic mode
                    raise RuntimeError("semantic stream encoder was not initialized")

                for event in events:
                    if event.type is SemanticEventType.USAGE and event.usage is not None:
                        usage_snapshot = _merge_stream_usage(usage_snapshot, event.usage)
                        event = replace(event, usage=usage_snapshot)
                    event = _prepare_event(event, response_id=response_id, model=public_model)
                    if event.type is SemanticEventType.TERMINAL:
                        if terminal_count:
                            raise _stream_protocol_error(
                                plan,
                                "Upstream stream emitted multiple terminal events",
                            )
                        terminal_count = 1
                        pending_terminal = event
                        continue
                    if pending_terminal is not None and event.type is not SemanticEventType.USAGE:
                        raise _stream_protocol_error(
                            plan,
                            "Upstream stream emitted an event after its terminal event",
                        )
                    encoded_frames = encoder.encode(event)
                    if not isinstance(encoded_frames, tuple) or not all(
                        isinstance(encoded, Mapping) for encoded in encoded_frames
                    ):
                        raise TypeError("stream encoder must return a tuple of mappings")
                    for encoded in encoded_frames:
                        yield encoded

            if pending_terminal is not None:
                if encoder is None:  # pragma: no cover - semantic mode creates it
                    raise RuntimeError("semantic stream encoder was not initialized")
                encoded_frames = encoder.encode(pending_terminal)
                if not isinstance(encoded_frames, tuple) or not all(
                    isinstance(encoded, Mapping) for encoded in encoded_frames
                ):
                    raise TypeError("stream encoder must return a tuple of mappings")
                for encoded in encoded_frames:
                    yield encoded

            if terminal_count == 0:
                if mode == "raw_identity" or (
                    mode is None and plan.conversion_mode is ConversionMode.IDENTITY
                ):
                    yield _identity_unexpected_eof(
                        plan.source_protocol,
                        response_id=response_id,
                        public_model=public_model,
                        response_snapshot=identity_response_snapshot,
                        sequence_number=identity_sequence_number + 1,
                    )
                else:
                    if encoder is None:
                        encoder = self._stream_encoder_factory(
                            ingress_runtime,
                            plan,
                            response_id,
                            public_model,
                        )
                    eof_events = _finish_semantic_source(decoder)
                    for event in eof_events:
                        event = _prepare_event(
                            event,
                            response_id=response_id,
                            model=public_model,
                        )
                        encoded_frames = encoder.encode(event)
                        for encoded in encoded_frames:
                            yield encoded
                terminal_count = 1
            if terminal_count != 1:  # pragma: no cover - duplicate terminals fail above
                raise _stream_protocol_error(plan, "Invalid upstream terminal count")
        finally:
            await close_async_iterator(source)

    async def _semantic_response(
        self,
        value: Any,
        plan: TransportPlan,
        *,
        response_id: str,
    ) -> SemanticResponse:
        if isinstance(value, SemanticResponse):
            semantic = value
        elif isinstance(value, Mapping):
            runtime = self._runtime(plan.target_protocol, plan)
            semantic = await runtime.decode_response(value)
        elif isinstance(value, ChatResponse):
            semantic = semantic_response_from_legacy_chat(
                replace(value, model=plan.model.upstream_id),
                response_id=response_id,
                origin_protocol=plan.target_protocol,
                origin_provider=plan.model.provider,
                origin_binding=plan.binding.id,
            )
        elif isinstance(value, ResponsesResponse):
            semantic = responses_response_to_semantic(
                replace(value, model=plan.model.upstream_id),
                response_id=response_id,
                origin_provider=plan.model.provider,
                origin_binding=plan.binding.id,
            )
        else:
            raise TypeError(f"unsupported upstream response {type(value).__name__}")
        if not isinstance(semantic, SemanticResponse):
            raise TypeError("response decoder must return SemanticResponse")
        return semantic

    def _runtime(
        self,
        protocol: WireProtocol,
        plan: TransportPlan | None = None,
    ) -> ProtocolRuntime:
        if self._runtime_resolver is not None:
            runtime = self._runtime_resolver(protocol, plan)
            if runtime.protocol is not protocol:
                raise ValueError("resolved runtime protocol does not match the request")
            return runtime
        if isinstance(self._runtimes, ProtocolRuntimeRegistry):
            return self._runtimes.get(protocol)
        try:
            return self._runtimes[protocol]
        except KeyError:
            raise ProtocolRuntimeNotFoundError(protocol) from None

    @staticmethod
    def _validate_ingress(plan: TransportPlan, runtime: ProtocolRuntime) -> None:
        if runtime.protocol is not plan.source_protocol:
            raise ValueError("ingress runtime protocol does not match the transport plan")


def _new_stream_decoder(
    runtime: ProtocolRuntime,
    _plan: TransportPlan,
) -> SemanticStreamDecoder:
    factory = getattr(runtime, "new_stream_decoder", None)
    if not callable(factory):
        raise UnsupportedProtocolOperationError(runtime.protocol, "new_stream_decoder")
    decoder = factory()
    if not isinstance(decoder, SemanticStreamDecoder):
        raise TypeError("protocol runtime stream decoder does not implement decode")
    return decoder


def _new_stream_encoder(
    runtime: ProtocolRuntime,
    _plan: TransportPlan,
    response_id: str,
    public_model: str,
) -> SemanticStreamEncoder:
    factory = getattr(runtime, "new_stream_encoder", None)
    if not callable(factory):
        raise UnsupportedProtocolOperationError(runtime.protocol, "new_stream_encoder")
    if runtime.protocol is WireProtocol.GEMINI:
        encoder = factory(model=public_model)
    else:
        encoder = factory(model=public_model, response_id=response_id)
    if not isinstance(encoder, SemanticStreamEncoder):
        raise TypeError("protocol runtime stream encoder does not implement encode")
    return encoder


def _new_response_id(protocol: WireProtocol) -> str:
    prefix = {
        WireProtocol.ANTHROPIC_MESSAGES: "msg_",
        WireProtocol.OPENAI_CHAT: "chatcmpl-",
        WireProtocol.OPENAI_RESPONSES: "resp_",
        WireProtocol.GEMINI: "rm-",
    }[protocol]
    return f"{prefix}{uuid4().hex}"


def _semantic_source_kind(frame: object) -> str:
    if isinstance(frame, Mapping):
        return "mapping"
    if isinstance(frame, ChatStreamChunk):
        return "legacy_chat"
    if isinstance(frame, ResponsesStreamChunk):
        return "legacy_responses"
    raise TypeError(f"unsupported upstream stream frame {type(frame).__name__}")


def _prepare_event(event: SemanticEvent, *, response_id: str, model: str) -> SemanticEvent:
    metadata = dict(event.metadata)
    metadata["model"] = model
    return replace(
        event,
        response_id=response_id,
        metadata=metadata,
    )


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)


def _merge_stream_usage(snapshot: Usage | None, update: Usage) -> Usage:
    """Fold a partial stream usage update into a downstream snapshot.

    Protocol decoders preserve whether each event is a delta or snapshot.  The
    downstream stream encoders consume snapshots, so normalize here without
    inventing counters that the upstream omitted.
    """
    values: dict[str, int | None] = {}
    for field in _USAGE_FIELDS:
        previous = getattr(snapshot, field) if snapshot is not None else None
        current = getattr(update, field)
        if current is None:
            values[field] = previous
        elif update.mode is UsageMode.DELTA:
            values[field] = (previous if previous is not None else 0) + current
        else:
            values[field] = current
    return Usage(mode=UsageMode.SNAPSHOT, **values)


def _finish_semantic_source(
    decoder: SemanticStreamDecoder | None,
) -> tuple[SemanticEvent, ...]:
    if decoder is not None:
        finish = getattr(decoder, "finish_eof", None)
        if callable(finish):
            events = finish()
            if not isinstance(events, tuple) or not all(
                isinstance(event, SemanticEvent) for event in events
            ):
                raise TypeError("stream decoder finish_eof must return SemanticEvent values")
            if events:
                return events
    terminal = TerminalMetadata(
        error_code="unexpected_eof",
        error_message="Upstream stream ended before an explicit terminal event",
        response_status="failed",
        transport_termination="unexpected_eof",
    )
    return (
        SemanticEvent(type=SemanticEventType.ERROR, terminal=terminal),
        SemanticEvent(type=SemanticEventType.TERMINAL, terminal=terminal),
    )


def _identity_unexpected_eof(
    protocol: WireProtocol,
    *,
    response_id: str,
    public_model: str,
    response_snapshot: Mapping[str, Any] | None,
    sequence_number: int,
) -> Mapping[str, Any]:
    message = "Upstream stream ended before an explicit terminal event"
    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        return {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": message,
                "code": "unexpected_eof",
            },
        }
    if protocol is WireProtocol.OPENAI_RESPONSES:
        response = dict(response_snapshot or {})
        response.setdefault("id", response_id)
        response.setdefault("object", "response")
        response.setdefault("created_at", int(time()))
        response["model"] = public_model
        response["status"] = "incomplete"
        response.setdefault("output", [])
        response.setdefault("usage", None)
        response["incomplete_details"] = {"reason": "unexpected_eof"}
        response["error"] = None
        return {
            "type": "response.incomplete",
            "sequence_number": sequence_number,
            "response": response,
        }
    if protocol is WireProtocol.OPENAI_CHAT:
        return {
            "error": {
                "type": "unexpected_eof",
                "code": "unexpected_eof",
                "message": message,
            }
        }
    return {
        "error": {
            "code": 502,
            "message": message,
            "status": "INTERNAL",
            "details": [{"reason": "unexpected_eof"}],
        }
    }


def _project_raw_identity(
    payload: Mapping[str, Any],
    protocol: WireProtocol,
    public_model: str,
    *,
    stream: bool,
) -> Mapping[str, Any]:
    """Apply shallow wire guards without rebuilding an identity response/frame."""
    result: Mapping[str, Any] = payload

    def set_top_level(key: str) -> None:
        nonlocal result
        if key not in result or result.get(key) == public_model:
            return
        result = dict(result)
        result[key] = public_model

    if protocol is WireProtocol.OPENAI_CHAT:
        expected_object = "chat.completion.chunk" if stream else "chat.completion"
        if result.get("object") != expected_object:
            result = dict(result)
            result["object"] = expected_object

    set_top_level("model")
    if protocol is WireProtocol.GEMINI:
        set_top_level("modelVersion")

    nested_fields = []
    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        nested_fields.append("message")
    if protocol is WireProtocol.OPENAI_RESPONSES:
        nested_fields.append("response")
    for field in nested_fields:
        nested = result.get(field)
        if not isinstance(nested, Mapping) or "model" not in nested:
            continue
        if nested.get("model") == public_model:
            continue
        outer = dict(result)
        rewritten = dict(nested)
        rewritten["model"] = public_model
        outer[field] = rewritten
        result = outer
    return result


def _is_identity_terminal(plan: TransportPlan, frame: Mapping[str, Any]) -> bool:
    protocol = plan.target_protocol
    frame_type = frame.get("type")
    if frame_type == "error" or "error" in frame:
        return True
    if protocol is WireProtocol.ANTHROPIC_MESSAGES:
        return frame_type == "message_stop"
    if protocol is WireProtocol.OPENAI_RESPONSES:
        return frame_type in {
            "response.done",
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
        }
    if protocol is WireProtocol.OPENAI_CHAT:
        choices = frame.get("choices")
        return isinstance(choices, list) and any(
            isinstance(choice, Mapping) and choice.get("finish_reason") is not None
            for choice in choices
        )
    if protocol is WireProtocol.GEMINI:
        candidates = frame.get("candidates")
        candidate_terminal = isinstance(candidates, list) and any(
            isinstance(candidate, Mapping) and candidate.get("finishReason") is not None
            for candidate in candidates
        )
        prompt_feedback = frame.get("promptFeedback")
        prompt_blocked = isinstance(prompt_feedback, Mapping) and isinstance(
            prompt_feedback.get("blockReason"), str
        )
        return candidate_terminal or prompt_blocked
    return False  # pragma: no cover - WireProtocol is currently closed


def _validate_identity_stream_frame(
    plan: TransportPlan,
    frame: Mapping[str, Any],
) -> None:
    """Shallow-check raw frames without decoding them into semantic events."""
    protocol = plan.target_protocol
    frame_type = frame.get("type")
    error = frame.get("error")
    if frame_type == "error" or error is not None:
        if error is None or isinstance(error, Mapping):
            return
        raise _malformed_identity_stream(plan)

    valid = False
    if protocol is WireProtocol.OPENAI_CHAT:
        choices = frame.get("choices")
        valid = isinstance(choices, list) and all(
            isinstance(choice, Mapping)
            and ("delta" not in choice or isinstance(choice.get("delta"), Mapping))
            and (
                "finish_reason" not in choice
                or choice.get("finish_reason") is None
                or isinstance(choice.get("finish_reason"), str)
            )
            for choice in choices
        )
        usage = frame.get("usage")
        valid = valid and (usage is None or isinstance(usage, Mapping))
    elif protocol is WireProtocol.OPENAI_RESPONSES:
        valid = isinstance(frame_type, str) and bool(frame_type)
        if "response" in frame:
            valid = valid and isinstance(frame.get("response"), Mapping)
    elif protocol is WireProtocol.ANTHROPIC_MESSAGES:
        valid = isinstance(frame_type, str) and bool(frame_type)
        required_mapping = (
            {
                "message_start": "message",
                "content_block_start": "content_block",
                "content_block_delta": "delta",
                "message_delta": "delta",
            }.get(frame_type)
            if isinstance(frame_type, str)
            else None
        )
        if required_mapping is not None:
            valid = valid and isinstance(frame.get(required_mapping), Mapping)
    elif protocol is WireProtocol.GEMINI:
        candidates = frame.get("candidates")
        prompt_feedback = frame.get("promptFeedback")
        valid = (
            isinstance(candidates, list)
            and all(isinstance(candidate, Mapping) for candidate in candidates)
        ) or isinstance(prompt_feedback, Mapping)

    if not valid:
        raise _malformed_identity_stream(plan)


def _malformed_identity_stream(plan: TransportPlan) -> ProviderError:
    return _stream_protocol_error(
        plan,
        f"{plan.model.provider} returned a malformed upstream response",
    )


def _is_allowed_identity_post_terminal_usage(
    plan: TransportPlan,
    frame: Mapping[str, Any],
) -> bool:
    """Allow only OpenAI Chat's standard trailing usage-only chunk."""
    return (
        plan.target_protocol is WireProtocol.OPENAI_CHAT
        and frame.get("choices") == []
        and isinstance(frame.get("usage"), Mapping)
    )


def _identity_terminal_allows_usage_tail(
    plan: TransportPlan,
    frame: Mapping[str, Any],
) -> bool:
    """Open a Chat usage-tail window only when the terminal has no usage or error."""
    return (
        plan.target_protocol is WireProtocol.OPENAI_CHAT
        and frame.get("usage") is None
        and "error" not in frame
    )


def _semantic_terminal_allows_usage_tail(
    plan: TransportPlan,
    events: tuple[SemanticEvent, ...],
    terminal_event: SemanticEvent,
) -> bool:
    """Apply the same Chat usage-tail rule to decoded and legacy event batches."""
    terminal = terminal_event.terminal
    return (
        plan.target_protocol is WireProtocol.OPENAI_CHAT
        and terminal is not None
        and terminal.finish_reason is not None
        and terminal.error_code is None
        and terminal.error_message is None
        and terminal.response_status not in {"failed", "cancelled", "unknown"}
        and terminal_event.usage is None
        and not any(
            event.type in {SemanticEventType.USAGE, SemanticEventType.ERROR} for event in events
        )
    )


def _stream_protocol_error(plan: TransportPlan, message: str) -> ProviderError:
    return ProviderError(
        message,
        status_code=502,
        retryable=True,
        kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
        provider=plan.model.provider,
        model=plan.model.upstream_id,
    )


__all__ = [
    "GenerationResponseBridge",
    "IdentityTerminalClassifier",
    "ResponseIdFactory",
    "SemanticStreamDecoder",
    "SemanticStreamEncoder",
    "StreamDecoderFactory",
    "StreamEncoderFactory",
]
