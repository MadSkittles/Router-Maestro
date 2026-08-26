"""Protocol-aware generation dispatch with transport-before-model fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from router_maestro.protocols import (
    ConversionMode,
    ProtocolDecodeError,
    ProtocolRepresentabilityError,
    ProtocolRuntime,
    ProtocolRuntimeNotFoundError,
    ProtocolRuntimeRegistry,
    RequestEnvelope,
    SemanticEvent,
    SemanticResponse,
    UnsupportedProtocolOperationError,
    WireProtocol,
    check_request_representability,
)
from router_maestro.providers.base import (
    ChatRequest,
    ProviderError,
    ProviderFailureKind,
    RequestOptionError,
    ResponsesRequest,
)
from router_maestro.providers.bindings import AttemptRequestContext
from router_maestro.providers.handler import ProviderHandler
from router_maestro.routing.capabilities import Operation
from router_maestro.routing.generation_plan import GenerationRoutePlan, plan_generation_route
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.transport_plan import TransportPlan
from router_maestro.runtime.reasoning_capsule import (
    ReasoningCapsuleCodec,
    ReasoningCapsuleError,
    ReasoningCapsulePayload,
    deserialize_opaque_state,
)
from router_maestro.runtime.request_context import get_current_request_context
from router_maestro.utils.async_iterators import close_async_iterator
from router_maestro.utils.logging import get_logger

if TYPE_CHECKING:
    from router_maestro.routing.router import Router

logger = get_logger("server.dispatcher")

_SOURCE_PARAMETER_ALIASES = {
    WireProtocol.ANTHROPIC_MESSAGES: {
        "max_output_tokens": "max_tokens",
        "stop": "stop_sequences",
        "stop_sequences": "stop_sequences",
    },
    WireProtocol.OPENAI_CHAT: {
        "max_output_tokens": "max_tokens",
        "reasoning.effort": "reasoning_effort",
        "structured_output": "response_format",
        "stop": "stop",
        "stop_sequences": "stop",
    },
    WireProtocol.OPENAI_RESPONSES: {
        "reasoning.effort": "reasoning.effort",
        "structured_output": "text",
        "stop": "stop",
        "stop_sequences": "stop",
    },
    WireProtocol.GEMINI: {
        "candidate_count": "generationConfig.candidateCount",
        "frequency_penalty": "generationConfig.frequencyPenalty",
        "max_output_tokens": "generationConfig.maxOutputTokens",
        "presence_penalty": "generationConfig.presencePenalty",
        "response_mime_type": "generationConfig.responseMimeType",
        "stop": "generationConfig.stopSequences",
        "stop_sequences": "generationConfig.stopSequences",
        "temperature": "generationConfig.temperature",
        "top_k": "generationConfig.topK",
        "top_p": "generationConfig.topP",
    },
}

_SOURCE_PARAMETER_PREFIXES = {
    WireProtocol.ANTHROPIC_MESSAGES: {
        "parallel_tool_calls": "tool_choice.disable_parallel_tool_use",
        "reasoning.budget_tokens": "thinking.budget_tokens",
        "reasoning.effort": "output_config.effort",
        "reasoning.enabled": "thinking.type",
        "structured_output": "output_config.format",
    },
    WireProtocol.OPENAI_CHAT: {
        "reasoning.effort": "reasoning_effort",
        "structured_output": "response_format",
    },
    WireProtocol.OPENAI_RESPONSES: {
        "reasoning": "reasoning",
        "structured_output": "text",
    },
    WireProtocol.GEMINI: {
        "reasoning.budget_tokens": "generationConfig.thinkingConfig.thinkingBudget",
        "reasoning.effort": "generationConfig.thinkingConfig.thinkingBudget",
        "tool_choice": "toolConfig.functionCallingConfig",
    },
}


def _source_parameter(envelope: RequestEnvelope, parameter: str) -> str:
    protocol = envelope.protocol
    direct = _SOURCE_PARAMETER_ALIASES.get(protocol, {}).get(parameter)
    if direct is not None:
        return direct

    raw_payload = envelope.raw_payload
    if protocol is WireProtocol.GEMINI:
        generation = raw_payload.get("generationConfig")
        if isinstance(generation, Mapping):
            if parameter == "reasoning.enabled":
                thinking = generation.get("thinkingConfig")
                if isinstance(thinking, Mapping):
                    if "includeThoughts" in thinking:
                        return "generationConfig.thinkingConfig.includeThoughts"
                    if "thinkingBudget" in thinking:
                        return "generationConfig.thinkingConfig.thinkingBudget"
            if parameter == "structured_output" or parameter.startswith("structured_output."):
                for field in ("responseJsonSchema", "responseSchema"):
                    if field in generation:
                        suffix = parameter.removeprefix("structured_output")
                        return f"generationConfig.{field}{suffix}"

    for semantic, source in _SOURCE_PARAMETER_PREFIXES.get(protocol, {}).items():
        if parameter == semantic:
            return source
        if parameter.startswith(f"{semantic}."):
            return f"{source}{parameter[len(semantic) :]}"
    return parameter


def _source_provider_rejection(
    error: ProviderError,
    envelope: RequestEnvelope,
) -> ProviderError:
    if not isinstance(error, RequestOptionError) or error.parameter is None:
        return error
    parameter = _source_parameter(envelope, error.parameter)
    if parameter == error.parameter:
        return error
    translated = RequestOptionError(
        f"Request field '{parameter}' is not supported by any available provider transport",
        parameter=parameter,
        upstream_status_code=error.upstream_status_code,
        provider=error.provider,
        model=error.model,
        cause=error,
        signal=error.signal,
    )
    return translated.with_attempts(error.attempts) if error.attempts else translated


class LegacyChatRequestFactory(Protocol):
    """Build the legacy Chat DTO without routing through semantic IR."""

    def __call__(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool,
    ) -> ChatRequest: ...


class LegacyResponsesRequestFactory(Protocol):
    """Build the legacy Responses DTO without routing through semantic IR."""

    def __call__(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool,
    ) -> ResponsesRequest: ...


class GenerationExecutionAdapter(Protocol):
    """Execute one already selected transport and return validated provider values.

    Stream adapters must suppress protocol keepalives and other non-frame input.
    The first item they yield is therefore the first frame that commits delivery.
    """

    async def execute(
        self,
        plan: TransportPlan,
        payload: Mapping[str, Any],
        *,
        request_context: AttemptRequestContext | None = None,
    ) -> Any: ...

    async def open_stream(
        self,
        plan: TransportPlan,
        payload: Mapping[str, Any],
        *,
        request_context: AttemptRequestContext | None = None,
    ) -> AsyncIterator[Any]: ...


class ProtocolRuntimeResolver(Protocol):
    """Resolve a target codec with the selected provider binding attached."""

    def __call__(
        self,
        protocol: WireProtocol,
        plan: TransportPlan | None = None,
    ) -> ProtocolRuntime: ...


class DispatchAttemptOutcome(StrEnum):
    """Closed attempt outcomes suitable for low-cardinality metrics labels."""

    SELECTED = "selected"
    RETRYABLE_FAILURE = "retryable_failure"
    UNSUPPORTED = "unsupported"
    UNREPRESENTABLE = "unrepresentable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DispatchAttemptObservation:
    """Protocol-only attempt facts safe to expose to a metrics collector.

    Provider, model, and binding identity remain in request audit only. Keeping
    them out of this hook prevents configured model IDs from becoming an
    unbounded Prometheus label.
    """

    entry_protocol: WireProtocol
    upstream_transport: WireProtocol
    conversion_mode: ConversionMode
    outcome: DispatchAttemptOutcome
    ir_materialized: bool


class DispatchAttemptObserver(Protocol):
    """Receive one completed or statically rejected transport attempt."""

    def __call__(self, observation: DispatchAttemptObservation, /) -> None: ...


class LegacyProviderExecutionAdapter:
    """Bridge endpoint bindings to the existing provider execution surface.

    Native Messages and generic dialect/executor bindings need no DTO bridge.
    Chat and Responses factories stay injectable until routes adopt the shared
    wire-to-legacy helpers; importantly, an identity attempt never decodes IR.
    """

    def __init__(
        self,
        *,
        chat_request_factory: LegacyChatRequestFactory | None = None,
        responses_request_factory: LegacyResponsesRequestFactory | None = None,
    ) -> None:
        self._chat_request_factory = chat_request_factory
        self._responses_request_factory = responses_request_factory

    async def execute(
        self,
        plan: TransportPlan,
        payload: Mapping[str, Any],
        *,
        request_context: AttemptRequestContext | None = None,
    ) -> Any:
        if not plan.binding.is_legacy:
            attempt = await plan.binding.prepare_attempt(
                model=plan.model,
                payload=payload,
                stream=False,
                request_context=request_context,
            )
            executor = plan.binding.executor
            if executor is None:  # pragma: no cover - EndpointBinding enforces the pair
                raise RuntimeError("non-legacy binding is missing its executor")
            return await executor.execute(attempt)

        provider = plan.provider
        model = plan.model.upstream_id
        if plan.target_protocol is WireProtocol.ANTHROPIC_MESSAGES:
            return await provider.messages_completion(payload, model=model)
        if plan.target_protocol is WireProtocol.OPENAI_CHAT:
            request = self._chat_request(payload, model=model, stream=False)
            return await provider.chat_completion(request)
        if plan.target_protocol is WireProtocol.OPENAI_RESPONSES:
            request = self._responses_request(payload, model=model, stream=False)
            return await provider.responses_completion(request)
        raise UnsupportedProtocolOperationError(plan.target_protocol, "legacy execution")

    async def open_stream(
        self,
        plan: TransportPlan,
        payload: Mapping[str, Any],
        *,
        request_context: AttemptRequestContext | None = None,
    ) -> AsyncIterator[Any]:
        if not plan.binding.is_legacy:
            attempt = await plan.binding.prepare_attempt(
                model=plan.model,
                payload=payload,
                stream=True,
                request_context=request_context,
            )
            executor = plan.binding.executor
            if executor is None:  # pragma: no cover - EndpointBinding enforces the pair
                raise RuntimeError("non-legacy binding is missing its executor")
            return executor.execute_stream(attempt)

        provider = plan.provider
        model = plan.model.upstream_id
        if plan.target_protocol is WireProtocol.ANTHROPIC_MESSAGES:
            return provider.messages_completion_stream(payload, model=model)
        if plan.target_protocol is WireProtocol.OPENAI_CHAT:
            request = self._chat_request(payload, model=model, stream=True)
            return provider.chat_completion_stream(request)
        if plan.target_protocol is WireProtocol.OPENAI_RESPONSES:
            request = self._responses_request(payload, model=model, stream=True)
            return provider.responses_completion_stream(request)
        raise UnsupportedProtocolOperationError(plan.target_protocol, "legacy stream execution")

    def _chat_request(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool,
    ) -> ChatRequest:
        if self._chat_request_factory is None:
            raise UnsupportedProtocolOperationError(
                WireProtocol.OPENAI_CHAT,
                "legacy request bridge",
            )
        request = self._chat_request_factory(payload, model=model, stream=stream)
        if not isinstance(request, ChatRequest):
            raise TypeError("chat request factory must return ChatRequest")
        return request

    def _responses_request(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool,
    ) -> ResponsesRequest:
        if self._responses_request_factory is None:
            raise UnsupportedProtocolOperationError(
                WireProtocol.OPENAI_RESPONSES,
                "legacy request bridge",
            )
        request = self._responses_request_factory(payload, model=model, stream=stream)
        if not isinstance(request, ResponsesRequest):
            raise TypeError("responses request factory must return ResponsesRequest")
        return request


@dataclass(frozen=True, slots=True)
class DispatchSelection:
    """The model and transport that committed one dispatch."""

    plan: TransportPlan


@dataclass(frozen=True, slots=True)
class DispatchResult:
    value: Any
    selection: DispatchSelection


@dataclass(frozen=True, slots=True)
class OpenedDispatchStream:
    frames: AsyncIterator[Any]
    selection: DispatchSelection
    semantic_decoder: Any | None = None
    first_events: tuple[SemanticEvent, ...] | None = None


@dataclass(frozen=True, slots=True)
class _CapsuleAffinity:
    provider: str
    model: str
    transport: str

    @classmethod
    def from_payload(cls, payload: ReasoningCapsulePayload) -> _CapsuleAffinity:
        return cls(
            provider=payload.provider,
            model=payload.model,
            transport=payload.transport,
        )


@dataclass(slots=True)
class _FailureState:
    first_retryable: ProviderError | None = None
    first_provider_rejection: ProviderError | None = None
    first_representability: ProtocolRepresentabilityError | None = None
    first_static_unsupported: BaseException | None = None

    def record_provider(self, error: ProviderError) -> bool:
        """Record a fallback-safe provider failure; return whether to continue."""
        if error.retryable:
            if self.first_retryable is None:
                self.first_retryable = error
            return True
        # RequestOptionError is adapter-scoped: another binding for the same
        # model may preserve the option. Model switching remains gated below on
        # an actual retryable upstream failure.
        if isinstance(error, RequestOptionError) or (
            error.kind is ProviderFailureKind.UNSUPPORTED_OPERATION
        ):
            if self.first_provider_rejection is None:
                self.first_provider_rejection = error
            return True
        return False

    def record_representability(self, error: ProtocolRepresentabilityError) -> None:
        if self.first_representability is None:
            self.first_representability = error

    def record_static(self, error: BaseException) -> None:
        if self.first_static_unsupported is None:
            self.first_static_unsupported = error

    def final_error(self, envelope: RequestEnvelope) -> BaseException:
        if self.first_retryable is not None:
            return self.first_retryable
        if self.first_provider_rejection is not None:
            return _source_provider_rejection(self.first_provider_rejection, envelope)
        if self.first_representability is not None:
            error = self.first_representability
            parameter = _source_parameter(
                envelope,
                error.report.parameter or error.parameter,
            )
            return ProviderError(
                f"No provider transport can represent request field '{parameter}'",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
                cause=error,
                parameter=parameter,
            )
        if self.first_static_unsupported is not None:
            return ProviderError(
                f"No implemented transport can execute {envelope.protocol.value}",
                status_code=501,
                retryable=False,
                kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
                cause=self.first_static_unsupported,
            )
        return ProviderError(
            f"No provider transport is available for {envelope.protocol.value}",
            status_code=400,
            retryable=False,
            kind=ProviderFailureKind.CLIENT_REQUEST,
        )


class _PrimedStream:
    """Replay one primed frame and own the selected iterator's lifecycle."""

    def __init__(self, first: Any, source: AsyncIterator[Any]) -> None:
        self._first = first
        self._source = source
        self._first_pending = True
        self._closed = False

    def __aiter__(self) -> _PrimedStream:
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        if self._first_pending:
            self._first_pending = False
            return self._first
        try:
            return await anext(self._source)
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await close_async_iterator(self._source)


class GenerationDispatcher:
    """Select a model, exhaust its transports, then move to model fallback."""

    def __init__(
        self,
        runtimes: Mapping[WireProtocol, ProtocolRuntime] | ProtocolRuntimeRegistry,
        *,
        execution: GenerationExecutionAdapter | None = None,
        reasoning_capsule_codec: ReasoningCapsuleCodec | None = None,
        attempt_observer: DispatchAttemptObserver | None = None,
        runtime_resolver: ProtocolRuntimeResolver | None = None,
    ) -> None:
        self._runtimes = runtimes
        self._execution = execution or LegacyProviderExecutionAdapter()
        self._reasoning_capsule_codec = reasoning_capsule_codec
        self._attempt_observer = attempt_observer
        self._runtime_resolver = runtime_resolver

    async def dispatch(self, router: Router, envelope: RequestEnvelope) -> DispatchResult:
        capsule_affinity = self._capsule_affinity(envelope)
        route = await self._plan_generation_route(router, envelope, capsule_affinity)
        previous_response_affinity = self._previous_response_affinity(envelope, route)
        if (
            capsule_affinity is not None
            and previous_response_affinity is not None
            and capsule_affinity != previous_response_affinity
        ):
            raise self._invalid_reasoning_capsule()
        continuation_affinity = capsule_affinity or previous_response_affinity
        failures = _FailureState()
        eligible_transport_seen = False

        for candidate_index, candidate in enumerate(route.candidates):
            transports = ProviderHandler(candidate.provider).bindings_for(
                candidate,
                envelope.protocol,
                envelope.manifest,
            )
            for transport in transports:
                if not self._transport_is_eligible(
                    envelope,
                    transport,
                    continuation_affinity,
                    primary_model=candidate_index == 0,
                ):
                    continue
                eligible_transport_seen = True
                materialization_count_before = envelope.materialization_count
                if not self._supports_mode(transport, stream=False):
                    failures.record_static(
                        UnsupportedProtocolOperationError(
                            transport.target_protocol,
                            "non-stream execution",
                        )
                    )
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.UNSUPPORTED,
                        materialization_count_before=materialization_count_before,
                    )
                    continue
                try:
                    payload = await self._payload(envelope, transport)
                    self._verify_capsules_for_execution(envelope, transport)
                    value = await self._execution.execute(
                        transport,
                        payload,
                        request_context=self._attempt_request_context(envelope, transport),
                    )
                    value = await self._validated_response(transport, value)
                except ProtocolRepresentabilityError as error:
                    failures.record_representability(error)
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.UNREPRESENTABLE,
                        materialization_count_before=materialization_count_before,
                    )
                    continue
                except ProtocolDecodeError as error:
                    raise self._invalid_ingress(error) from error
                except (UnsupportedProtocolOperationError, ProtocolRuntimeNotFoundError) as error:
                    failures.record_static(error)
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.UNSUPPORTED,
                        materialization_count_before=materialization_count_before,
                    )
                    continue
                except ProviderError as error:
                    self._record_attempt(
                        envelope,
                        transport,
                        self._provider_failure_outcome(error),
                        materialization_count_before=materialization_count_before,
                    )
                    if failures.record_provider(error):
                        continue
                    raise
                except BaseException:
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.FAILED,
                        materialization_count_before=materialization_count_before,
                    )
                    raise
                self._record_attempt(
                    envelope,
                    transport,
                    DispatchAttemptOutcome.SELECTED,
                    materialization_count_before=materialization_count_before,
                )
                return DispatchResult(value, DispatchSelection(transport))

            if (
                candidate_index + 1 < len(route.candidates)
                and failures.first_retryable is None
                and (continuation_affinity is None or eligible_transport_seen)
            ):
                # RoutePlan model switches are recovery from an upstream failure.
                # A request rejected during static transport preparation must stay
                # a client/implementation error instead of silently changing model.
                # Capsule affinity is the one exception: candidates before the
                # pinned provider/model are intentionally ineligible and must be
                # skipped without requiring a failed upstream attempt.
                raise failures.final_error(envelope)

        if continuation_affinity is not None and not eligible_transport_seen:
            raise self._invalid_reasoning_capsule()
        raise failures.final_error(envelope)

    async def dispatch_stream(
        self,
        router: Router,
        envelope: RequestEnvelope,
    ) -> OpenedDispatchStream:
        capsule_affinity = self._capsule_affinity(envelope)
        route = await self._plan_generation_route(router, envelope, capsule_affinity)
        previous_response_affinity = self._previous_response_affinity(envelope, route)
        if (
            capsule_affinity is not None
            and previous_response_affinity is not None
            and capsule_affinity != previous_response_affinity
        ):
            raise self._invalid_reasoning_capsule()
        continuation_affinity = capsule_affinity or previous_response_affinity
        failures = _FailureState()
        eligible_transport_seen = False

        for candidate_index, candidate in enumerate(route.candidates):
            transports = ProviderHandler(candidate.provider).bindings_for(
                candidate,
                envelope.protocol,
                envelope.manifest,
            )
            for transport in transports:
                iterator: AsyncIterator[Any] | None = None
                semantic_decoder: Any | None = None
                first_events: tuple[SemanticEvent, ...] | None = None
                if not self._transport_is_eligible(
                    envelope,
                    transport,
                    continuation_affinity,
                    primary_model=candidate_index == 0,
                ):
                    continue
                eligible_transport_seen = True
                materialization_count_before = envelope.materialization_count
                if not self._supports_mode(transport, stream=True):
                    failures.record_static(
                        UnsupportedProtocolOperationError(
                            transport.target_protocol,
                            "stream execution",
                        )
                    )
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.UNSUPPORTED,
                        materialization_count_before=materialization_count_before,
                    )
                    continue
                try:
                    payload = await self._payload(envelope, transport)
                    self._verify_capsules_for_execution(envelope, transport)
                    iterator = await self._execution.open_stream(
                        transport,
                        payload,
                        request_context=self._attempt_request_context(envelope, transport),
                    )
                    first = await anext(iterator)
                    if isinstance(first, Mapping):
                        self._validate_first_stream_frame(
                            transport.target_protocol,
                            transport,
                            first,
                        )
                        if transport.conversion_mode is ConversionMode.SEMANTIC_IR:
                            semantic_decoder, first_events = self._decode_first_cross_frame(
                                transport,
                                first,
                            )
                except StopAsyncIteration:
                    await close_async_iterator(iterator)
                    failures.record_provider(self._empty_stream_error(transport))
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.RETRYABLE_FAILURE,
                        materialization_count_before=materialization_count_before,
                    )
                    continue
                except ProtocolRepresentabilityError as error:
                    await close_async_iterator(iterator)
                    failures.record_representability(error)
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.UNREPRESENTABLE,
                        materialization_count_before=materialization_count_before,
                    )
                    continue
                except ProtocolDecodeError as error:
                    await close_async_iterator(iterator)
                    raise self._invalid_ingress(error) from error
                except (UnsupportedProtocolOperationError, ProtocolRuntimeNotFoundError) as error:
                    await close_async_iterator(iterator)
                    failures.record_static(error)
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.UNSUPPORTED,
                        materialization_count_before=materialization_count_before,
                    )
                    continue
                except ProviderError as error:
                    await close_async_iterator(iterator)
                    self._record_attempt(
                        envelope,
                        transport,
                        self._provider_failure_outcome(error),
                        materialization_count_before=materialization_count_before,
                    )
                    if failures.record_provider(error):
                        continue
                    raise
                except BaseException:
                    await close_async_iterator(iterator)
                    self._record_attempt(
                        envelope,
                        transport,
                        DispatchAttemptOutcome.FAILED,
                        materialization_count_before=materialization_count_before,
                    )
                    raise

                self._record_attempt(
                    envelope,
                    transport,
                    DispatchAttemptOutcome.SELECTED,
                    materialization_count_before=materialization_count_before,
                )
                return OpenedDispatchStream(
                    frames=_PrimedStream(first, iterator),
                    selection=DispatchSelection(transport),
                    semantic_decoder=semantic_decoder,
                    first_events=first_events,
                )

            if (
                candidate_index + 1 < len(route.candidates)
                and failures.first_retryable is None
                and (continuation_affinity is None or eligible_transport_seen)
            ):
                # Keep the streaming and non-streaming model-switch rules exact:
                # only a pre-commit retryable upstream failure authorizes RoutePlan
                # to select another model.
                raise failures.final_error(envelope)

        if continuation_affinity is not None and not eligible_transport_seen:
            raise self._invalid_reasoning_capsule()
        raise failures.final_error(envelope)

    def _record_attempt(
        self,
        envelope: RequestEnvelope,
        transport: TransportPlan,
        outcome: DispatchAttemptOutcome,
        *,
        materialization_count_before: int,
    ) -> None:
        # Report whether this attempt caused lazy IR materialization. Cached IR
        # reuse and a later identity fallback both remain false, while an actual
        # identity regression stays observable instead of being masked by mode.
        ir_materialized = envelope.materialization_count > materialization_count_before
        context = get_current_request_context()
        audit = context.audit if context is not None else None
        if audit is not None:
            audit.record_dispatch_attempt(
                provider=transport.model.provider,
                model=transport.model.upstream_id,
                binding=transport.binding.id,
                entry_protocol=transport.source_protocol.value,
                upstream_transport=transport.target_protocol.value,
                conversion_mode=transport.conversion_mode.value,
                outcome=outcome.value,
                ir_materialized=ir_materialized,
            )

        if self._attempt_observer is not None:
            try:
                self._attempt_observer(
                    DispatchAttemptObservation(
                        entry_protocol=transport.source_protocol,
                        upstream_transport=transport.target_protocol,
                        conversion_mode=transport.conversion_mode,
                        outcome=outcome,
                        ir_materialized=ir_materialized,
                    )
                )
            except Exception:
                logger.warning("Dispatch attempt observer failed", exc_info=True)

    @staticmethod
    def _provider_failure_outcome(error: ProviderError) -> DispatchAttemptOutcome:
        if error.retryable:
            return DispatchAttemptOutcome.RETRYABLE_FAILURE
        if isinstance(error, RequestOptionError):
            return DispatchAttemptOutcome.UNREPRESENTABLE
        if error.kind is ProviderFailureKind.UNSUPPORTED_OPERATION:
            return DispatchAttemptOutcome.UNSUPPORTED
        return DispatchAttemptOutcome.FAILED

    def _capsule_affinity(self, envelope: RequestEnvelope) -> _CapsuleAffinity | None:
        capsules = envelope.manifest.reasoning_capsules
        if not capsules:
            return None
        codec = self._reasoning_capsule_codec
        if codec is None:
            raise self._invalid_reasoning_capsule()

        try:
            affinities = tuple(
                _CapsuleAffinity.from_payload(codec.unseal_for_routing(capsule))
                for capsule in capsules
            )
        except ReasoningCapsuleError:
            raise self._invalid_reasoning_capsule() from None

        affinity = affinities[0]
        if any(candidate != affinity for candidate in affinities[1:]):
            raise self._invalid_reasoning_capsule()
        return affinity

    async def _plan_generation_route(
        self,
        router: Router,
        envelope: RequestEnvelope,
        capsule_affinity: _CapsuleAffinity | None,
    ) -> GenerationRoutePlan:
        """Resolve continuation affinity independently of the fallback window."""
        requested_model = self._request_model(envelope)
        from router_maestro.routing.router import AUTO_ROUTE_MODEL

        pinned_auto_route = False
        if capsule_affinity is not None and requested_model == AUTO_ROUTE_MODEL:
            pinned_auto_route = True
            requested_model = ModelRef(
                capsule_affinity.provider,
                capsule_affinity.model,
            ).qualified_id
        try:
            return await plan_generation_route(router, requested_model, envelope.manifest)
        except ProviderError:
            if pinned_auto_route:
                raise self._invalid_reasoning_capsule() from None
            raise

    def _previous_response_affinity(
        self,
        envelope: RequestEnvelope,
        route: GenerationRoutePlan,
    ) -> _CapsuleAffinity | None:
        """Pin opaque Responses continuation IDs before any provider I/O.

        ``previous_response_id`` contains no Router-Maestro provenance.  The
        only safe stateless routing rule is therefore an exact provider/model
        selection with one unambiguous native Responses binding.  Auto-route,
        fuzzy aliases, and providers with multiple eligible Responses bindings
        must fail closed instead of guessing where the prior response lived.
        """
        previous_response_id = envelope.manifest.previous_response_id
        if previous_response_id is None:
            return None
        if (
            not previous_response_id
            or previous_response_id != previous_response_id.strip()
            or not route.explicit
            or envelope.manifest.model != route.primary.model.qualified_id
        ):
            raise self._invalid_previous_response_id()

        transports = ProviderHandler(route.primary.provider).bindings_for(
            route.primary,
            envelope.protocol,
            envelope.manifest,
        )
        native_responses = tuple(
            transport
            for transport in transports
            if transport.target_protocol is WireProtocol.OPENAI_RESPONSES
            and transport.conversion_mode is ConversionMode.IDENTITY
        )
        if len(native_responses) != 1:
            raise self._invalid_previous_response_id()
        transport = native_responses[0]
        return _CapsuleAffinity(
            provider=transport.model.provider,
            model=transport.model.upstream_id,
            transport=transport.binding.id,
        )

    def _verify_capsules_for_execution(
        self,
        envelope: RequestEnvelope,
        transport: TransportPlan,
    ) -> None:
        capsules = envelope.manifest.reasoning_capsules
        if not capsules:
            return
        codec = self._reasoning_capsule_codec
        if codec is None:  # pragma: no cover - _capsule_affinity fails first
            raise self._invalid_reasoning_capsule()

        try:
            for capsule in capsules:
                codec.unseal(
                    capsule,
                    expected_provider=transport.model.provider,
                    expected_model=transport.model.upstream_id,
                    expected_transport=transport.binding.id,
                )
        except ReasoningCapsuleError:
            raise self._invalid_reasoning_capsule() from None

    @staticmethod
    def _transport_is_eligible(
        envelope: RequestEnvelope,
        transport: TransportPlan,
        continuation_affinity: _CapsuleAffinity | None,
        *,
        primary_model: bool,
    ) -> bool:
        manifest = envelope.manifest
        if manifest.previous_response_id is not None:
            return primary_model and continuation_affinity == _CapsuleAffinity(
                provider=transport.model.provider,
                model=transport.model.upstream_id,
                transport=transport.binding.id,
            )
        if (
            manifest.opaque_continuation
            and transport.target_protocol is WireProtocol.OPENAI_CHAT
            and continuation_affinity is None
        ):
            return False
        if continuation_affinity is None:
            return True
        return continuation_affinity == _CapsuleAffinity(
            provider=transport.model.provider,
            model=transport.model.upstream_id,
            transport=transport.binding.id,
        )

    @staticmethod
    def _invalid_reasoning_capsule() -> ProviderError:
        return ProviderError(
            "Invalid reasoning capsule",
            status_code=400,
            retryable=False,
            kind=ProviderFailureKind.CLIENT_REQUEST,
        )

    @staticmethod
    def _invalid_previous_response_id() -> ProviderError:
        return ProviderError(
            "previous_response_id requires an explicit provider/model with one "
            "native Responses transport",
            status_code=400,
            retryable=False,
            kind=ProviderFailureKind.CLIENT_REQUEST,
            parameter="previous_response_id",
        )

    @staticmethod
    def _invalid_ingress(error: ProtocolDecodeError) -> ProviderError:
        message = f"{error.path}: {error.reason}" if error.path else error.reason
        return ProviderError(
            message,
            status_code=400,
            retryable=False,
            kind=ProviderFailureKind.CLIENT_REQUEST,
            parameter=error.path,
            cause=error,
        )

    async def _payload(
        self,
        envelope: RequestEnvelope,
        transport: TransportPlan,
    ) -> Mapping[str, Any]:
        if transport.conversion_mode is ConversionMode.IDENTITY:
            # First-party raw dialects obey the copy-on-write contract. Legacy
            # third-party adapters predate it, so isolate their DTO bridge with
            # a full copy until those providers migrate to endpoint bindings.
            payload = (
                envelope.raw_payload if transport.binding.is_legacy else envelope.native_payload()
            )
            if (
                transport.target_protocol is WireProtocol.OPENAI_RESPONSES
                and envelope.manifest.reasoning_capsules
            ):
                return self._restore_responses_identity_capsules(
                    payload,
                    envelope,
                    transport,
                )
            return payload
        runtime = self._runtime(transport.target_protocol, transport)
        semantic_request = await envelope.semantic_ir()
        attempt_request = replace(
            semantic_request,
            model=transport.model.upstream_id,
            stream=envelope.stream,
        )
        report = await check_request_representability(runtime, attempt_request)
        if not report.is_exact:
            parameter = report.parameter or "request"
            reason = "; ".join(report.reasons)
            if not reason:
                reason = "conversion would be lossy" if report.lossy else "field is unsupported"
            raise ProtocolRepresentabilityError(
                transport.target_protocol,
                parameter,
                reason,
                report=report,
            )
        encoded = await runtime.encode_request(attempt_request)
        if not isinstance(encoded, Mapping):
            raise TypeError("protocol runtime encode_request must return a mapping")
        return encoded

    def _restore_responses_identity_capsules(
        self,
        payload: dict[str, Any],
        envelope: RequestEnvelope,
        transport: TransportPlan,
    ) -> Mapping[str, Any]:
        """Unwrap RM carriers on a raw Responses fast path without building IR."""
        codec = self._reasoning_capsule_codec
        if codec is None:
            raise self._invalid_reasoning_capsule()
        raw_input = payload.get("input")
        if not isinstance(raw_input, list | tuple):
            raise self._invalid_reasoning_capsule()

        capsules = envelope.manifest.reasoning_capsules
        restored_count = 0
        restored_input: list[Any] = []
        try:
            for raw_item in raw_input:
                if not isinstance(raw_item, Mapping) or raw_item.get("type") != "reasoning":
                    restored_input.append(raw_item)
                    continue
                carrier = raw_item.get("encrypted_content")
                if not isinstance(carrier, str) or carrier not in capsules:
                    restored_input.append(raw_item)
                    continue
                item_id = raw_item.get("id")
                if not isinstance(item_id, str) or not item_id:
                    raise ValueError
                capsule = codec.unseal(
                    carrier,
                    expected_provider=transport.model.provider,
                    expected_model=transport.model.upstream_id,
                    expected_transport=transport.binding.id,
                    expected_item_id=item_id,
                )
                state = deserialize_opaque_state(capsule.opaque_state)
                if isinstance(state, Mapping):
                    restored = dict(state)
                    if restored.get("type") != "reasoning" or restored.get("id") != item_id:
                        raise ValueError
                    restored_input.append(restored)
                elif isinstance(state, str):
                    restored_input.append({**dict(raw_item), "encrypted_content": state})
                else:
                    raise ValueError
                restored_count += 1
        except (ReasoningCapsuleError, TypeError, ValueError):
            raise self._invalid_reasoning_capsule() from None

        if restored_count != len(capsules):
            raise self._invalid_reasoning_capsule()
        payload["input"] = restored_input
        return payload

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
    def _request_model(envelope: RequestEnvelope) -> str:
        model = envelope.manifest.model
        if model is None:
            raise ProviderError(
                "Request model must be a string",
                status_code=400,
                retryable=False,
                kind=ProviderFailureKind.CLIENT_REQUEST,
                parameter="model",
            )
        return model

    @staticmethod
    def _attempt_request_context(
        envelope: RequestEnvelope,
        transport: TransportPlan,
    ) -> AttemptRequestContext:
        """Expose immutable ingress metadata without coupling dialects to FastAPI."""
        return AttemptRequestContext(
            path=envelope.path,
            query=envelope.query,
            headers=envelope.headers,
            conversion_mode=transport.conversion_mode,
            _mappings_owned=True,
        )

    @staticmethod
    def _supports_mode(plan: TransportPlan, *, stream: bool) -> bool:
        operations = {
            WireProtocol.ANTHROPIC_MESSAGES: (
                Operation.NATIVE_ANTHROPIC,
                Operation.CHAT_STREAM if stream else Operation.CHAT,
            ),
            WireProtocol.OPENAI_CHAT: (Operation.CHAT_STREAM if stream else Operation.CHAT,),
            WireProtocol.OPENAI_RESPONSES: (
                Operation.RESPONSES_STREAM if stream else Operation.RESPONSES,
            ),
            # There is no production Gemini-native provider yet. Reuse the
            # generic generation operation pair so a future non-legacy Gemini
            # binding can exercise identity dispatch without extending the
            # legacy public Operation metadata ahead of that provider.
            WireProtocol.GEMINI: (Operation.CHAT_STREAM if stream else Operation.CHAT,),
        }
        eligible = operations.get(plan.target_protocol, ())
        return any(plan.binding.supports(operation) for operation in eligible)

    async def _validated_response(self, plan: TransportPlan, value: Any) -> Any:
        """Validate a selected response before its transport commits.

        Identity mappings stay on the shallow, IR-free guard.  A cross-protocol
        mapping must be fully decoded while fallback is still possible; the
        resulting immutable semantic response is cached in ``DispatchResult``
        so the response bridge does not decode it a second time.
        """
        self._validate_raw_response(plan, value)
        if plan.conversion_mode is ConversionMode.IDENTITY or not isinstance(value, Mapping):
            return value

        runtime = self._runtime(plan.target_protocol, plan)
        try:
            semantic = await runtime.decode_response(value)
        except ProtocolDecodeError as error:
            raise self._malformed_cross_response(plan, cause=error) from error
        if not isinstance(semantic, SemanticResponse):
            raise self._malformed_cross_response(plan)
        return semantic

    @staticmethod
    def _malformed_cross_response(
        plan: TransportPlan,
        *,
        cause: BaseException | None = None,
    ) -> ProviderError:
        return ProviderError(
            f"Upstream {plan.target_protocol.value} response cannot be decoded",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            provider=plan.model.provider,
            model=plan.model.upstream_id,
            cause=cause,
        )

    def _decode_first_cross_frame(
        self,
        plan: TransportPlan,
        frame: Mapping[str, Any],
    ) -> tuple[Any, tuple[SemanticEvent, ...]]:
        """Fully validate and cache the first cross-protocol stream frame."""
        runtime = self._runtime(plan.target_protocol, plan)
        factory = getattr(runtime, "new_stream_decoder", None)
        if not callable(factory):
            raise UnsupportedProtocolOperationError(
                plan.target_protocol,
                "new_stream_decoder",
            )
        decoder = factory()
        decode = getattr(decoder, "decode", None)
        if not callable(decode):
            raise self._malformed_cross_stream(plan)
        try:
            events = decode(frame)
        except ProtocolDecodeError as error:
            raise self._malformed_cross_stream(plan, cause=error) from error
        if not isinstance(events, tuple) or not all(
            isinstance(event, SemanticEvent) for event in events
        ):
            raise self._malformed_cross_stream(plan)
        return decoder, events

    @staticmethod
    def _malformed_cross_stream(
        plan: TransportPlan,
        *,
        cause: BaseException | None = None,
    ) -> ProviderError:
        return ProviderError(
            f"Upstream {plan.target_protocol.value} stream cannot be decoded",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            provider=plan.model.provider,
            model=plan.model.upstream_id,
            cause=cause,
        )

    @staticmethod
    def _validate_raw_response(plan: TransportPlan, value: Any) -> None:
        """Reject an obviously malformed raw response before committing its model.

        This is intentionally a shallow identity-safe guard. Cross-protocol
        mappings receive full decoding in ``_validated_response``; typed legacy
        DTOs have already been validated by their provider.
        """
        if not isinstance(value, Mapping):
            return

        protocol = plan.target_protocol
        valid = False
        if protocol is WireProtocol.ANTHROPIC_MESSAGES:
            valid = value.get("type") == "message" and isinstance(value.get("content"), list)
        elif protocol is WireProtocol.OPENAI_CHAT:
            valid = isinstance(value.get("choices"), list)
        elif protocol is WireProtocol.OPENAI_RESPONSES:
            valid = value.get("status") in {
                "completed",
                "incomplete",
                "failed",
                "cancelled",
            } and isinstance(value.get("output"), list)
        elif protocol is WireProtocol.GEMINI:
            valid = isinstance(value.get("candidates"), list) or isinstance(
                value.get("promptFeedback"), Mapping
            )

        if valid:
            return
        raise ProviderError(
            f"Upstream {protocol.value} response is malformed",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            provider=plan.model.provider,
            model=plan.model.upstream_id,
        )

    @staticmethod
    def _validate_first_stream_frame(
        protocol: WireProtocol,
        plan: TransportPlan,
        frame: Mapping[str, Any],
    ) -> None:
        """Shallow-check one primed frame before committing its transport.

        This deliberately does not invoke a semantic stream decoder: identity
        attempts must remain IR-free, and the original frame must still be
        replayed unchanged by ``_PrimedStream`` after selection.
        """
        if protocol is not plan.target_protocol:
            raise ValueError("target protocol does not match the transport plan")

        frame_type = frame.get("type")
        error = frame.get("error")
        if protocol is WireProtocol.OPENAI_CHAT:
            if error is not None:
                valid = isinstance(error, Mapping)
            else:
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
            valid = GenerationDispatcher._valid_responses_first_frame(frame_type, frame)
        elif protocol is WireProtocol.ANTHROPIC_MESSAGES:
            valid = GenerationDispatcher._valid_anthropic_first_frame(frame_type, frame)
        elif protocol is WireProtocol.GEMINI:
            if error is not None:
                valid = isinstance(error, Mapping)
            else:
                candidates = frame.get("candidates")
                prompt_feedback = frame.get("promptFeedback")
                valid = (
                    isinstance(candidates, list)
                    and all(isinstance(candidate, Mapping) for candidate in candidates)
                ) or isinstance(prompt_feedback, Mapping)
        else:  # pragma: no cover - WireProtocol is closed
            valid = False

        if valid:
            return
        raise ProviderError(
            f"Upstream {protocol.value} stream emitted a malformed first frame",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            provider=plan.model.provider,
            model=plan.model.upstream_id,
        )

    @staticmethod
    def _valid_anthropic_first_frame(
        frame_type: object,
        frame: Mapping[str, Any],
    ) -> bool:
        if frame_type == "error":
            error = frame.get("error")
            if not isinstance(error, Mapping):
                return False
            error_type = error.get("type", "upstream_error")
            return GenerationDispatcher._non_empty_string(error_type) and isinstance(
                error.get("message"), str
            )
        if frame_type != "message_start":
            return False

        message = frame.get("message")
        if not isinstance(message, Mapping):
            return False
        content = message.get("content", [])
        usage = message.get("usage")
        return (
            GenerationDispatcher._non_empty_string(message.get("id"))
            and GenerationDispatcher._non_empty_string(message.get("model"))
            and message.get("type", "message") == "message"
            and message.get("role", "assistant") == "assistant"
            and content in (None, [])
            and (usage is None or isinstance(usage, Mapping))
        )

    @staticmethod
    def _valid_responses_first_frame(
        frame_type: object,
        frame: Mapping[str, Any],
    ) -> bool:
        if not isinstance(frame_type, str):
            return False
        if frame_type == "error":
            error = frame.get("error")
            if not isinstance(error, Mapping):
                return False
            code = error.get("code", error.get("type"))
            return GenerationDispatcher._non_empty_string(code) and isinstance(
                error.get("message"), str
            )

        response_events = {
            "response.created",
            "response.in_progress",
            "response.done",
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
        }
        if frame_type in response_events:
            response = frame.get("response")
            return (
                isinstance(response, Mapping)
                and GenerationDispatcher._non_empty_string(response.get("id"))
                and GenerationDispatcher._non_empty_string(response.get("model"))
            )

        if not GenerationDispatcher._valid_optional_string(frame.get("item_id")):
            return False
        if not GenerationDispatcher._valid_stream_index(frame, "output_index"):
            return False

        if frame_type in {"response.output_item.added", "response.output_item.done"}:
            item = frame.get("item")
            if not isinstance(item, Mapping):
                return False
            return item.get("type") in {
                "message",
                "function_call",
                "custom_tool_call",
                "tool_search_call",
            }

        if frame_type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            return GenerationDispatcher._valid_stream_index(frame, "content_index") and isinstance(
                frame.get("delta"), str
            )
        if frame_type in {
            "response.output_text.done",
            "response.refusal.done",
        }:
            field = "text" if frame_type == "response.output_text.done" else "refusal"
            return GenerationDispatcher._valid_stream_index(frame, "content_index") and isinstance(
                frame.get(field), str
            )
        if frame_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
        }:
            field = "delta" if frame_type.endswith(".delta") else "text"
            return GenerationDispatcher._valid_stream_index(frame, "summary_index") and isinstance(
                frame.get(field), str
            )
        if frame_type in {
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            part = frame.get("part")
            return (
                GenerationDispatcher._valid_stream_index(frame, "summary_index")
                and isinstance(part, Mapping)
                and part.get("type") == "summary_text"
            )
        if frame_type in {
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
        }:
            return isinstance(frame.get("delta"), str)
        if frame_type in {
            "response.function_call_arguments.done",
            "response.custom_tool_call_input.done",
        }:
            return True
        return False

    @staticmethod
    def _valid_stream_index(frame: Mapping[str, Any], field: str) -> bool:
        value = frame.get(field, 0)
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _valid_optional_string(value: object) -> bool:
        return value is None or isinstance(value, str)

    @staticmethod
    def _non_empty_string(value: object) -> bool:
        return isinstance(value, str) and bool(value)

    @staticmethod
    def _empty_stream_error(plan: TransportPlan) -> ProviderError:
        return ProviderError(
            "Upstream stream ended before its first valid frame",
            status_code=502,
            retryable=True,
            kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
            provider=plan.provider.name,
            model=plan.model.upstream_id,
        )


__all__ = [
    "DispatchAttemptObservation",
    "DispatchAttemptObserver",
    "DispatchAttemptOutcome",
    "DispatchResult",
    "DispatchSelection",
    "GenerationDispatcher",
    "GenerationExecutionAdapter",
    "LegacyChatRequestFactory",
    "LegacyProviderExecutionAdapter",
    "LegacyResponsesRequestFactory",
    "OpenedDispatchStream",
    "ProtocolRuntimeResolver",
]
