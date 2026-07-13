"""Provider and per-model routing capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

from router_maestro.routing.model_ref import ModelRef

if TYPE_CHECKING:
    from router_maestro.providers.base import ChatRequest, ResponsesRequest


class Operation(StrEnum):
    CHAT = "chat"
    CHAT_STREAM = "chat_stream"
    RESPONSES = "responses"
    RESPONSES_STREAM = "responses_stream"
    NATIVE_ANTHROPIC = "native_anthropic"


class Feature(StrEnum):
    TOOLS = "tools"
    VISION = "vision"
    REASONING = "reasoning"
    PARALLEL_TOOLS = "parallel_tools"


class CapabilitySupport(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"

    @property
    def is_supported(self) -> bool:
        return self is CapabilitySupport.SUPPORTED

    @property
    def is_unsupported(self) -> bool:
        return self is CapabilitySupport.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class RequestFeatures:
    tools: bool = False
    vision: bool = False
    reasoning: bool = False
    parallel_tools: bool = False
    responses_bridge: bool = False
    reasoning_parameter: str | None = field(default=None, compare=False, repr=False)

    @classmethod
    def for_chat(cls, request: ChatRequest) -> RequestFeatures:
        from router_maestro.utils.responses_bridge import is_experimental_responses_enabled

        return cls(
            tools=bool(request.tools),
            vision=_contains_image(request.messages),
            reasoning=bool(
                request.reasoning_effort
                or request.thinking_budget
                or request.thinking_type in {"enabled", "adaptive"}
            ),
            responses_bridge=request.use_responses_api and is_experimental_responses_enabled(),
            reasoning_parameter=(
                "reasoning_effort"
                if request.reasoning_effort is not None
                else (
                    "thinking_budget"
                    if request.thinking_budget is not None
                    or request.thinking_type in {"enabled", "adaptive"}
                    else None
                )
            ),
        )

    @classmethod
    def for_responses(cls, request: ResponsesRequest) -> RequestFeatures:
        return cls(
            tools=bool(request.tools),
            vision=_contains_image(request.input),
            reasoning=bool(request.reasoning_effort),
            parallel_tools=request.parallel_tool_calls is True,
            reasoning_parameter=(
                "reasoning_effort" if request.reasoning_effort is not None else None
            ),
        )

    @classmethod
    def for_anthropic_native(cls, body: Mapping[str, object]) -> RequestFeatures:
        """Derive native Messages requirements from one raw wire request."""
        from router_maestro.utils.reasoning import VALID_EFFORTS

        thinking = body.get("thinking")
        thinking_type = thinking.get("type") if isinstance(thinking, dict) else None
        output_config = body.get("output_config")
        effort = output_config.get("effort") if isinstance(output_config, dict) else None
        has_effort = isinstance(effort, str) and effort in VALID_EFFORTS
        has_thinking = isinstance(thinking_type, str) and thinking_type in {
            "enabled",
            "adaptive",
        }
        return cls(
            tools=bool(body.get("tools")),
            vision=_contains_image(body.get("messages")),
            reasoning=has_effort or has_thinking,
            reasoning_parameter=(
                "output_config.effort" if has_effort else ("thinking" if has_thinking else None)
            ),
        )

    def required(self) -> tuple[Feature, ...]:
        return tuple(
            feature
            for feature, enabled in (
                (Feature.TOOLS, self.tools),
                (Feature.VISION, self.vision),
                (Feature.REASONING, self.reasoning),
                (Feature.PARALLEL_TOOLS, self.parallel_tools),
            )
            if enabled
        )


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    operations: frozenset[Operation] = frozenset({Operation.CHAT, Operation.CHAT_STREAM})
    operation_bridges: Mapping[Operation, Operation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", frozenset(self.operations))
        bridges = dict(self.operation_bridges)
        if any(
            source not in self.operations or target not in self.operations
            for source, target in bridges.items()
        ):
            raise ValueError("operation bridges must reference provider-supported operations")
        object.__setattr__(self, "operation_bridges", MappingProxyType(bridges))

    def supports(self, operation: Operation) -> bool:
        return operation in self.operations

    def bridge_for(self, operation: Operation) -> Operation | None:
        """Return the provider-declared alternate transport for one operation."""
        return self.operation_bridges.get(operation)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    model: ModelRef
    operations: Mapping[Operation, CapabilitySupport] = field(default_factory=dict)
    features: Mapping[Feature, CapabilitySupport] = field(default_factory=dict)
    reasoning_effort_values: tuple[str, ...] | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", MappingProxyType(dict(self.operations)))
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        if self.reasoning_effort_values is not None:
            object.__setattr__(
                self,
                "reasoning_effort_values",
                tuple(self.reasoning_effort_values),
            )

    def operation(self, operation: Operation) -> CapabilitySupport:
        return self.operations.get(operation, CapabilitySupport.UNKNOWN)

    def feature(self, feature: Feature) -> CapabilitySupport:
        return self.features.get(feature, CapabilitySupport.UNKNOWN)

    def support_for(
        self,
        operation: Operation,
        features: RequestFeatures,
    ) -> CapabilitySupport:
        required = [self.operation(operation)]
        required.extend(self.feature(feature) for feature in features.required())
        if CapabilitySupport.UNSUPPORTED in required:
            return CapabilitySupport.UNSUPPORTED
        if all(support is CapabilitySupport.SUPPORTED for support in required):
            return CapabilitySupport.SUPPORTED
        return CapabilitySupport.UNKNOWN


def _contains_image(value: object) -> bool:
    if isinstance(value, list):
        return any(_contains_image(item) for item in value)
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type in {"image", "image_url", "input_image"}:
            return True
        return any(_contains_image(item) for item in value.values())
    content = getattr(value, "content", None)
    return _contains_image(content) if content is not None else False
