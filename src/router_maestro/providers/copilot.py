"""GitHub Copilot provider implementation."""

import contextlib
import json
import re
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, NoReturn

import httpx

from router_maestro.auth.github_oauth import get_copilot_token
from router_maestro.auth.storage import OAuthCredential
from router_maestro.pipeline.beta_strip import strip_beta_tokens
from router_maestro.protocols import ConversionMode, WireProtocol
from router_maestro.protocols._tool_namespace import (
    decode_namespaced_tool_name,
    encode_namespaced_tool_name,
)
from router_maestro.providers.anthropic_codec import (
    AnthropicCodecError,
    AnthropicStreamDecoder,
    decode_message_response,
)
from router_maestro.providers.base import (
    TIMEOUT_NON_STREAMING,
    BaseProvider,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Message,
    ModelInfo,
    ProviderError,
    ProviderFailureKind,
    ProviderFailureSignal,
    RequestOptionError,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamChunk,
    ResponseStatus,
    ResponsesToolCall,
    TerminalOutcome,
    TransportTermination,
)
from router_maestro.providers.bindings import (
    COPILOT_ANTHROPIC_MESSAGES_BINDING,
    COPILOT_OPENAI_CHAT_BINDING,
    COPILOT_OPENAI_RESPONSES_BINDING,
    AttemptRequestContext,
    EndpointBinding,
    PreparedAttempt,
)
from router_maestro.providers.copilot_support.auth_session import CopilotAuthSession
from router_maestro.providers.copilot_support.catalog import CopilotCatalog
from router_maestro.providers.copilot_support.catalog import (
    normalize_supported_endpoints as _catalog_normalize_supported_endpoints,
)
from router_maestro.providers.copilot_support.catalog import (
    operation_capabilities as _catalog_operation_capabilities,
)
from router_maestro.providers.copilot_support.chat_codec import CopilotChatCodec
from router_maestro.providers.copilot_support.responses_codec import CopilotResponsesCodec
from router_maestro.providers.copilot_support.transport import CopilotTransport
from router_maestro.providers.http_executor import SharedHttpExecutor
from router_maestro.providers.outbound_contract import OutboundContract, ReasoningResolution
from router_maestro.routing.capabilities import Operation, ProviderCapabilities
from router_maestro.routing.model_ref import ModelRef
from router_maestro.utils import get_logger
from router_maestro.utils.context_window import resolve_thinking_budget
from router_maestro.utils.reasoning import (
    VALID_EFFORTS,
    budget_to_effort,
    downgrade_for_upstream,
    resolve_effort_within_catalog,
)

logger = get_logger("providers.copilot")

COPILOT_CHAT_PATH = "/chat/completions"
COPILOT_RESPONSES_PATH = "/responses"
COPILOT_MESSAGES_PATH = "/v1/messages"
COPILOT_COUNT_TOKENS_PATH = "/v1/messages/count_tokens"

# Internal model aliases bound to the Copilot provider. Maps a synthetic client
# model id (with no GHC catalog entry) to a real upstream Copilot model. Codex's
# Auto-review (guardian) mode issues Responses requests with model
# ``codex-auto-review``, which only exists on the ChatGPT/Codex subscription
# backend; route it to GHC's low-latency ``gpt-5.4-mini`` subagent model.
COPILOT_MODEL_ALIASES: dict[str, str] = {"codex-auto-review": "gpt-5.4-mini"}

_COPILOT_UNSUPPORTED_OPERATION_CODE = "unsupported_api_for_model"
_MAX_COPILOT_ERROR_BODY_BYTES = 64 * 1024
_EXACT_COPILOT_BARE_BAD_REQUEST = b"Bad Request\n"


def _responses_terminal_outcome(response: Any) -> TerminalOutcome:
    """Preserve one native Responses terminal payload without chat mapping."""
    return CopilotResponsesCodec.terminal_outcome(response)


_CLAUDE_VERSION_PATTERN = re.compile(
    r"^claude-(?P<family>opus|sonnet|haiku)-(?P<major>\d+)"
    r"(?:(?:\.(?P<minor_dot>\d+))|(?:-(?P<minor_dash>\d{1,2})(?=-|\[|$)))?"
    r"(?:-|\[|$)"
)


def _claude_family_version(bare_lower: str) -> tuple[str, int, int] | None:
    """Parse the family and numeric generation without consuming dated suffixes."""
    match = _CLAUDE_VERSION_PATTERN.match(bare_lower)
    if match is None:
        return None
    return (
        match.group("family"),
        int(match.group("major")),
        int(match.group("minor_dot") or match.group("minor_dash") or 0),
    )


def _known_claude_reasoning_support(bare_lower: str) -> bool | None:
    """Return static support only for Claude versions verified on Copilot.

    Per the Copilot model picker (vscode-copilot-chat), the 4.6+ generations
    accept ``reasoning_effort``. Older models (sonnet-4, sonnet-4.5,
    opus-4.5, haiku-4.5) have no reasoning control surface.
    """
    parsed = _claude_family_version(bare_lower)
    if parsed is None:
        return None
    family, major, minor = parsed
    if major != 4:
        return None
    if family == "haiku" and minor <= 5:
        return False
    if minor < 6:
        return False
    if family == "opus" and minor in {6, 7, 8}:
        return True
    if family == "sonnet" and minor == 6:
        return True
    return None


def _claude_supports_reasoning(bare_lower: str) -> bool:
    """Whether static compatibility data confirms Claude reasoning support."""
    return _known_claude_reasoning_support(bare_lower) is True


def _known_reasoning_support(model: str) -> bool | None:
    """Return static reasoning support for known Copilot model families."""
    bare = model.split("/", 1)[1] if "/" in model else model
    bare_lower = bare.lower()
    claude_support = _known_claude_reasoning_support(bare_lower)
    if claude_support is not None:
        return claude_support
    if bare_lower.startswith(("gpt-5", "o1", "o3", "o4")):
        return True
    if bare_lower.startswith(("gpt-4", "gemini-2.5")):
        return False
    return None


def normalize_supported_endpoints(model: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Preserve missing versus explicit Copilot endpoint catalog state."""
    return _catalog_normalize_supported_endpoints(model)


def copilot_operation_capabilities(model: Mapping[str, Any]) -> dict[str, bool]:
    """Derive operation support, preferring the live catalog endpoint contract."""
    return _catalog_operation_capabilities(
        model,
        normalize_endpoints=normalize_supported_endpoints,
    )


def apply_copilot_chat_reasoning(
    payload: dict,
    model: str,
    thinking_budget: int | None,
    reasoning_effort: str | None,
    catalog_effort_values: list[str] | None = None,
) -> None:
    """Inject reasoning fields into a Copilot ``/chat/completions`` payload.

    Thin adapter over ``CopilotOutboundContract.resolve_reasoning`` (the single
    source of Copilot's effort rule). Applies the resolved effort and the gpt-5.4
    ``max_tokens`` -> ``max_completion_tokens`` rewrite to ``payload`` in place.
    """
    resolution = CopilotOutboundContract().resolve_reasoning(
        model=model,
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
        catalog_effort_values=catalog_effort_values,
        operation=Operation.CHAT,
    )
    if resolution.effort is not None:
        payload["reasoning_effort"] = resolution.effort
    if resolution.rewrite_max_tokens_to_completion and "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")


def _thinking_requested(request: ChatRequest) -> bool:
    """Whether the client opted into reasoning passthrough.

    We only surface upstream chain-of-thought to clients that explicitly asked
    for it. Reasoning traces can leak prompt fragments, hidden instructions,
    and tool-planning state, so emitting them by default would be a
    sensitive-data exposure surface.
    """
    return CopilotChatCodec.thinking_requested(request)


def _extract_reasoning_from_chunk(part: dict | None) -> tuple[str, str | None]:
    """Pull reasoning text/signature out of a Copilot message or delta.

    Mirrors vscode-copilot-chat's ``extractThinkingDeltaFromChoice``: Copilot
    streams reasoning under several legacy field names depending on the
    upstream model family.

    Returns ``(text, signature)`` where either may be empty/None.
    """
    return CopilotChatCodec.extract_reasoning(part)


_COPILOT_NATIVE_ANTHROPIC_FORWARD_FIELDS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "stream",
        "system",
        "thinking",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "metadata",
        "output_config",
        # Forwarded verbatim: GHC accepts and applies context_management
        # (verified live, echoes applied_edits). The paired anthropic-beta header
        # is forwarded by the Copilot transport, so the option only needs to
        # survive body stripping.
        "context_management",
    }
)


# Top-level fields the native OpenAI Responses passthrough forwards to GHC's
# ``/responses`` endpoint. Fields outside this set are stripped before the raw
# body is sent, because GHC rejects unknown top-level fields. The set covers
# what OpenAI's Responses API defines and what Codex actually sends, so
# fidelity-relevant options the translated route drops (``include``,
# ``previous_response_id``, ``prompt_cache_key``, ``text``, encrypted-reasoning
# round-trip) survive the passthrough.
#
# ``store`` is intentionally excluded: GHC 400s on ``store: true``
# (unsupported_value), so it must be stripped rather than forwarded.
_COPILOT_RESPONSES_FORWARD_FIELDS = frozenset(
    {
        "model",
        "input",
        "stream",
        "instructions",
        "max_output_tokens",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "reasoning",
        "include",
        "metadata",
        "service_tier",
        "previous_response_id",
        "prompt_cache_key",
        "text",
        "truncation",
    }
)


# Keys inside ``output_config`` that GHC's native Anthropic endpoint refuses.
# Verified live on claude-opus-4.6, claude-sonnet-4.6 and claude-haiku-4.5:
# ``task_budget`` always returns 400 ``Extra inputs are not permitted``. Claude
# Code already carries this option behind the ``task-budgets-2026-03-13`` beta,
# so it can arrive unprompted.
#
# This is the ``store`` rule applied one level down: an option the upstream is
# known to reject is stripped so the request still succeeds. Siblings that are
# merely unrecognized here are NOT listed — those are forwarded for GHC to
# judge, which is what keeps a newly-supported field from being pre-emptively
# broken by a stale local allowlist.
_OUTPUT_CONFIG_REJECTED_BY_UPSTREAM = frozenset({"task_budget"})


def _normalize_copilot_tool(tool: Any) -> Any:
    """Copy one tool definition, filling Copilot-required namespace descriptions."""
    if not isinstance(tool, dict):
        return tool
    normalized = dict(tool)
    description = normalized.get("description")
    if normalized.get("type") == "namespace" and (
        description is None or isinstance(description, str) and not description.strip()
    ):
        name = normalized.get("name")
        if isinstance(name, str) and name:
            normalized["description"] = f"Tools in the {name} namespace."
        else:
            normalized["description"] = "Tools in this namespace."
    inner_tools = normalized.get("tools")
    if isinstance(inner_tools, list):
        normalized["tools"] = [_normalize_copilot_tool(inner) for inner in inner_tools]
    return normalized


def _normalize_gemini_schema_types(value: Any) -> Any:
    """Project nullable JSON Schema unions to Gemini's OpenAPI-style shape."""
    if isinstance(value, list):
        return [_normalize_gemini_schema_types(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    normalized = {key: _normalize_gemini_schema_types(item) for key, item in value.items()}
    schema_type = normalized.get("type")
    if isinstance(schema_type, list) and all(isinstance(item, str) for item in schema_type):
        non_null_types = [item for item in schema_type if item != "null"]
        if "null" in schema_type and len(non_null_types) == 1:
            normalized["type"] = non_null_types[0]
            normalized["nullable"] = True
        elif "null" in schema_type and len(non_null_types) > 1:
            normalized["type"] = non_null_types
    enum = normalized.get("enum")
    if (
        normalized.get("type") == "array"
        and isinstance(enum, list)
        and enum
        and not any(isinstance(item, list) for item in enum)
    ):
        items = normalized.get("items")
        normalized_items = dict(items) if isinstance(items, Mapping) else {}
        normalized_items.setdefault("enum", enum)
        normalized["items"] = normalized_items
        normalized.pop("enum", None)
    return normalized


def _normalize_gemini_chat_tools(tools: Any, *, model: str) -> Any:
    if not model.split("/", 1)[-1].lower().startswith("gemini-") or not isinstance(tools, list):
        return tools

    normalized_tools = []
    for raw_tool in tools:
        if not isinstance(raw_tool, Mapping):
            normalized_tools.append(raw_tool)
            continue
        tool = dict(raw_tool)
        function = tool.get("function")
        if isinstance(function, Mapping):
            normalized_function = dict(function)
            if "parameters" in normalized_function:
                normalized_function["parameters"] = _normalize_gemini_schema_types(
                    normalized_function["parameters"]
                )
            tool["function"] = normalized_function
        normalized_tools.append(tool)
    return normalized_tools


def _requires_namespace_flattening(model: str) -> bool:
    return model.split("/", 1)[-1].lower().startswith("grok-")


def _flatten_namespace_tools(tools: list[Any], *, model: str) -> list[Any]:
    flattened: list[Any] = []
    for index, raw_tool in enumerate(tools):
        if not isinstance(raw_tool, Mapping) or raw_tool.get("type") != "namespace":
            flattened.append(raw_tool)
            continue
        namespace = raw_tool.get("name")
        inner_tools = raw_tool.get("tools")
        if not isinstance(namespace, str) or not namespace or not isinstance(inner_tools, list):
            raise RequestOptionError(
                "GitHub Copilot requires valid namespace tools",
                provider="github-copilot",
                model=model,
                parameter=f"tools[{index}]",
            )
        for inner_index, raw_inner in enumerate(inner_tools):
            if not isinstance(raw_inner, Mapping) or raw_inner.get("type") != "function":
                raise RequestOptionError(
                    "GitHub Copilot Grok namespace members must be functions",
                    provider="github-copilot",
                    model=model,
                    parameter=f"tools[{index}].tools[{inner_index}]",
                )
            name = raw_inner.get("name")
            if not isinstance(name, str) or not name:
                raise RequestOptionError(
                    "GitHub Copilot Grok namespace function name is required",
                    provider="github-copilot",
                    model=model,
                    parameter=f"tools[{index}].tools[{inner_index}].name",
                )
            try:
                encoded_name = encode_namespaced_tool_name(namespace, name)
            except ValueError as error:
                raise RequestOptionError(
                    str(error),
                    provider="github-copilot",
                    model=model,
                    parameter=f"tools[{index}].tools[{inner_index}].name",
                    cause=error,
                ) from error
            inner = dict(raw_inner)
            inner["name"] = encoded_name
            inner.pop("defer_loading", None)
            flattened.append(inner)
    return flattened


def _flatten_grok_namespace_body(body: dict[str, Any], *, model: str) -> None:
    tools = body.get("tools")
    if isinstance(tools, list):
        body["tools"] = _flatten_namespace_tools(tools, model=model)

    response_input = body.get("input")
    if isinstance(response_input, list):
        normalized_input = []
        for raw_item in response_input:
            if not isinstance(raw_item, Mapping):
                normalized_input.append(raw_item)
                continue
            item = dict(raw_item)
            if item.get("type") == "additional_tools" and isinstance(item.get("tools"), list):
                item["tools"] = _flatten_namespace_tools(item["tools"], model=model)
            if item.get("type") == "function_call":
                namespace = item.get("namespace")
                name = item.get("name")
                if isinstance(namespace, str) and namespace and isinstance(name, str) and name:
                    try:
                        item["name"] = encode_namespaced_tool_name(namespace, name)
                    except ValueError as error:
                        raise RequestOptionError(
                            str(error),
                            provider="github-copilot",
                            model=model,
                            parameter="input.function_call.name",
                            cause=error,
                        ) from error
                    item.pop("namespace", None)
            normalized_input.append(item)
        body["input"] = normalized_input

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, Mapping) and tool_choice.get("type") == "function":
        namespace = tool_choice.get("namespace")
        name = tool_choice.get("name")
        if isinstance(namespace, str) and namespace and isinstance(name, str) and name:
            try:
                encoded_name = encode_namespaced_tool_name(namespace, name)
            except ValueError as error:
                raise RequestOptionError(
                    str(error),
                    provider="github-copilot",
                    model=model,
                    parameter="tool_choice.name",
                    cause=error,
                ) from error
            body["tool_choice"] = {**tool_choice, "name": encoded_name}
            body["tool_choice"].pop("namespace", None)


def _restore_namespaced_function_calls(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_namespaced_function_calls(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    restored = {key: _restore_namespaced_function_calls(item) for key, item in value.items()}
    if restored.get("type") == "function_call" and isinstance(restored.get("name"), str):
        namespaced = decode_namespaced_tool_name(restored["name"])
        if namespaced is not None:
            restored["namespace"], restored["name"] = namespaced
    return restored


class CopilotOutboundContract(OutboundContract):
    """Copilot upstream wire contract.

    Round 1: the native Anthropic passthrough forward allowlist. Fields outside
    this set are stripped before the raw body is sent to GitHub Copilot, which
    rejects unknown top-level fields (e.g. mcp_servers, container).
    """

    def forwardable_fields(self, operation: Operation) -> frozenset[str] | None:
        if operation is Operation.NATIVE_ANTHROPIC:
            return _COPILOT_NATIVE_ANTHROPIC_FORWARD_FIELDS
        if operation in (Operation.RESPONSES, Operation.RESPONSES_STREAM):
            return _COPILOT_RESPONSES_FORWARD_FIELDS
        return None

    def resolve_reasoning(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        thinking_budget: int | None,
        catalog_effort_values: list[str] | None,
        operation: Operation,
    ) -> ReasoningResolution:
        """The single Copilot effort rule: catalog-first, family-fallback.

        Reproduces the previously-scattered logic in ``apply_copilot_chat_reasoning``
        (chat) and ``responses_codec.build_payload`` (responses). The two operations
        differ only in the cold-catalog path and the gpt-5.4 max_tokens rewrite, both
        preserved here.
        """
        if operation in (Operation.RESPONSES, Operation.RESPONSES_STREAM):
            return self._resolve_reasoning_responses(
                model=model,
                reasoning_effort=reasoning_effort,
                catalog_effort_values=catalog_effort_values,
            )
        return self._resolve_reasoning_chat(
            model=model,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            catalog_effort_values=catalog_effort_values,
        )

    @staticmethod
    def _resolve_reasoning_chat(
        *,
        model: str,
        reasoning_effort: str | None,
        thinking_budget: int | None,
        catalog_effort_values: list[str] | None,
    ) -> ReasoningResolution:
        bare_lower = (model.split("/", 1)[1] if "/" in model else model).lower()
        rewrite = bare_lower.startswith("gpt-5.4")
        known_reasoning_support = _known_reasoning_support(model)

        # Catalog-driven path: trust whatever Copilot advertises.
        if catalog_effort_values is not None:
            if not catalog_effort_values:
                # Category A: model advertises no reasoning surface -> strip.
                if reasoning_effort is not None or thinking_budget is not None:
                    logger.debug(
                        "Copilot model %s advertises no reasoning tiers; "
                        "stripping requested reasoning",
                        model,
                    )
                return ReasoningResolution(effort=None, rewrite_max_tokens_to_completion=rewrite)
            desired = reasoning_effort or budget_to_effort(thinking_budget)
            if desired is None:
                if thinking_budget is not None:
                    # Category B: budget too small to derive a tier -> clamp up.
                    picked = resolve_effort_within_catalog("minimal", catalog_effort_values)
                    return ReasoningResolution(
                        effort=picked, rewrite_max_tokens_to_completion=rewrite
                    )
                return ReasoningResolution(effort=None, rewrite_max_tokens_to_completion=rewrite)
            picked = resolve_effort_within_catalog(desired, catalog_effort_values)
            return ReasoningResolution(effort=picked, rewrite_max_tokens_to_completion=rewrite)

        # Hardcoded fallback when the catalog hasn't been fetched yet.
        if known_reasoning_support is False:
            # Category A: statically known to lack reasoning -> strip.
            if reasoning_effort is not None or thinking_budget is not None:
                logger.debug(
                    "Copilot model %s has no known reasoning support; "
                    "stripping requested reasoning",
                    model,
                )
            return ReasoningResolution(effort=None, rewrite_max_tokens_to_completion=rewrite)

        effort = reasoning_effort or budget_to_effort(thinking_budget)
        if bare_lower.startswith("claude-") and effort in ("xhigh", "max"):
            effort = "high"
        elif effort == "max":
            effort = "xhigh"
        elif known_reasoning_support is None and effort == "xhigh":
            effort = "high"
        if effort is None and thinking_budget is not None:
            # Category B (cold path): supported family, budget below the lowest
            # mapped tier -> clamp up to the lowest upstream-native tier.
            effort = "low"
        if effort in ("minimal", "low", "medium", "high", "xhigh"):
            return ReasoningResolution(effort=effort, rewrite_max_tokens_to_completion=rewrite)
        return ReasoningResolution(effort=None, rewrite_max_tokens_to_completion=rewrite)

    @staticmethod
    def _resolve_reasoning_responses(
        *,
        model: str,
        reasoning_effort: str | None,
        catalog_effort_values: list[str] | None,
    ) -> ReasoningResolution:
        known_reasoning_support = _known_reasoning_support(model)
        if catalog_effort_values is not None:
            if not catalog_effort_values:
                # Category A: no reasoning surface -> strip.
                if reasoning_effort is not None:
                    logger.debug(
                        "Copilot Responses model %s advertises no reasoning "
                        "tiers; stripping reasoning",
                        model,
                    )
                return ReasoningResolution(effort=None)
            if reasoning_effort is None:
                return ReasoningResolution(effort=None)
            # Category B: clamp up when below the catalog floor.
            upstream_effort = resolve_effort_within_catalog(reasoning_effort, catalog_effort_values)
            return ReasoningResolution(effort=upstream_effort)

        if reasoning_effort is not None and known_reasoning_support is False:
            # Category A (cold path): statically known to lack reasoning -> strip.
            logger.debug(
                "Copilot Responses model %s has no known reasoning support; stripping reasoning",
                model,
            )
            return ReasoningResolution(effort=None)
        upstream_effort = (
            reasoning_effort
            if known_reasoning_support is True
            else downgrade_for_upstream(reasoning_effort)
        )
        if (
            known_reasoning_support is None
            and reasoning_effort in ("xhigh", "max")
            and upstream_effort == "high"
        ):
            logger.warning(
                "Copilot Responses catalog cold for %s; "
                "downgrading reasoning_effort=%s to high as a precaution",
                model,
                reasoning_effort,
            )
        return ReasoningResolution(effort=upstream_effort)

    _RESPONSES_UNSUPPORTED_TOOL_TYPES = frozenset(
        {"web_search", "web_search_preview", "code_interpreter"}
    )

    def filter_tools(
        self,
        tools: list[dict] | None,
        *,
        operation: Operation,
        model: str | None = None,
    ) -> list[dict] | None:
        """Drop/reject tool types Copilot Responses cannot express."""
        if not tools:
            return None
        validated = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                validated.append(_normalize_copilot_tool(tool))
            elif tool_type == "namespace":
                inner = tool.get("tools")
                if isinstance(inner, list) and inner:
                    validated.append(_normalize_copilot_tool(tool))
                else:
                    raise RequestOptionError(
                        "GitHub Copilot requires namespace tools to contain a non-empty tools list",
                        provider="github-copilot",
                        model=model,
                        parameter="tools",
                    )
            elif tool_type not in self._RESPONSES_UNSUPPORTED_TOOL_TYPES:
                validated.append(_normalize_copilot_tool(tool))
            # else: silently drop tools Copilot Responses cannot express
            # (web_search, web_search_preview, code_interpreter). Clients like
            # Codex inject these unconditionally; 400-ing the whole request over
            # a tool the backend can't run is worse than dropping it.
        return validated or None

    def normalize_responses_input(
        self,
        value: Any,
        *,
        operation: Operation,
        model: str | None = None,
    ) -> Any:
        """Sanitize Codex history and additional-tools registries for Responses."""
        if not isinstance(value, list):
            return value
        normalized_input = []
        for item in value:
            if not isinstance(item, dict):
                normalized_input.append(item)
                continue

            if (
                item.get("type") == "reasoning"
                and item.get("encrypted_content")
                and not item.get("id")
            ):
                # Codex stores the upstream reasoning id but intentionally omits
                # it when serializing the next request. Copilot signs the opaque
                # blob against that id, so forwarding the orphaned blob always
                # fails with "Could not decrypt the provided encrypted_content".
                normalized_item = dict(item)
                normalized_item.pop("encrypted_content", None)
                normalized_input.append(normalized_item)
                continue

            if item.get("type") != "additional_tools":
                normalized_input.append(item)
                continue
            normalized_item = dict(item)
            tools = normalized_item.get("tools")
            if isinstance(tools, list):
                normalized_item["tools"] = [_normalize_copilot_tool(tool) for tool in tools]
            normalized_input.append(normalized_item)
        return normalized_input

    def allows_temperature(self, operation: Operation) -> bool:
        """Copilot Responses rejects explicit temperature; Chat forwards it."""
        return operation not in (Operation.RESPONSES, Operation.RESPONSES_STREAM)

    def allows_top_p(
        self,
        operation: Operation,
        *,
        model: str | None = None,
    ) -> bool:
        """Copilot GPT-5.4 rejects top_p on Responses and Messages, not Chat."""
        if model is None or operation in (Operation.CHAT, Operation.CHAT_STREAM):
            return True
        bare_model = model.split("/", 1)[-1].lower()
        if not bare_model.startswith("gpt-5.4"):
            return True
        return operation not in (
            Operation.NATIVE_ANTHROPIC,
            Operation.RESPONSES,
            Operation.RESPONSES_STREAM,
        )

    def allows_stop(
        self,
        operation: Operation,
        *,
        model: str | None = None,
    ) -> bool:
        """Copilot's GPT-5 Chat surface rejects the ``stop`` parameter."""
        if operation not in (Operation.CHAT, Operation.CHAT_STREAM) or model is None:
            return True
        bare_model = model.split("/", 1)[-1].lower()
        return not bare_model.startswith("gpt-5")

    # ------------------------------------------------------------------
    # Native Anthropic passthrough normalization (folded in from the beta
    # Anthropic route). These own the outbound wire knowledge; the route keeps
    # thin seams that delegate here. Kept as staticmethods because they carry no
    # instance state and are Copilot-native-Anthropic specific.
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_output_config_effort(body: dict) -> None:
        """Remove only the reasoning effort from a non-reasoning model's body.

        Such models reject ``output_config.effort`` ("does not support reasoning
        effort") but still honor siblings like ``format`` — verified live on
        ``claude-haiku-4.5``, which returns 200 with a schema-conforming reply.
        Dropping the whole object would silently disable structured output.
        """
        output_config = body.get("output_config")
        if not isinstance(output_config, dict):
            body.pop("output_config", None)
            return

        remaining = {key: value for key, value in output_config.items() if key != "effort"}
        if remaining:
            body["output_config"] = remaining
        else:
            body.pop("output_config", None)

    @staticmethod
    def sanitize_output_config(body: dict) -> str | None:
        """Normalize the effort Router-Maestro consumes, forwarding siblings intact.

        Only ``effort`` is Router-Maestro's to own: it feeds effort resolution and
        the thinking-budget precedence rules. Every other key belongs to GHC, which
        accepts ``format`` (structured outputs, verified live: 200 with
        schema-conforming output on both streaming and non-streaming) and rejects
        genuinely unknown siblings itself (``Extra inputs are not permitted``).
        Stripping them here would turn a supported feature into a silent no-op.

        The exception is ``_OUTPUT_CONFIG_REJECTED_BY_UPSTREAM``: siblings GHC is
        known to refuse are dropped so the request still succeeds, mirroring how
        ``store`` is handled on the Responses passthrough.

        An unusable ``effort`` is dropped rather than rejected, but the rest of the
        object still goes upstream; ``output_config`` is removed only when nothing
        remains to send.
        """
        output_config = body.get("output_config")
        if not isinstance(output_config, dict):
            body.pop("output_config", None)
            return None

        effort = output_config.get("effort")
        if effort not in VALID_EFFORTS:
            effort = None

        forwarded = {
            key: value
            for key, value in output_config.items()
            if key != "effort" and key not in _OUTPUT_CONFIG_REJECTED_BY_UPSTREAM
        }
        if effort is not None:
            forwarded["effort"] = effort

        if not forwarded:
            body.pop("output_config", None)
            return None

        body["output_config"] = forwarded
        return effort

    @staticmethod
    def resolve_native_effort(
        effort: str,
        allowed: tuple[str, ...] | list[str] | None,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """Resolve a native effort within the catalog, clamping up below the floor.

        The native passthrough resolves ``output_config.effort`` with its own
        catalog-or-passthrough rule (distinct error surface and no family fallback),
        so it is intentionally NOT routed through ``resolve_reasoning`` (the
        chat/responses orchestration). Both share the ``pick_closest_effort``
        primitive as the single source of the downgrade math.
        """
        if allowed is None:
            return effort
        mapped_effort = resolve_effort_within_catalog(effort, list(allowed))
        if mapped_effort is None:
            # allowed held no valid tier: treat as no reasoning surface. The
            # caller (apply_native_anthropic_thinking) strips output_config in
            # that case, so this path is defensive only.
            raise RequestOptionError(
                "output_config.effort has no supported tier",
                parameter="output_config.effort",
                provider=provider,
                model=model,
            )
        return mapped_effort

    @staticmethod
    def reject_unpreservable_native_options(body: dict) -> None:
        """Reject native Anthropic options the transport cannot preserve together."""
        if "temperature" in body and "top_p" in body:
            raise RequestOptionError(
                "top_p cannot be combined with temperature on the native Anthropic transport",
                parameter="top_p",
            )

    @staticmethod
    def apply_native_anthropic_thinking(
        body: dict,
        actual_model: str,
        reasoning_effort_values: tuple[str, ...] | list[str] | None = None,
    ) -> dict:
        """Apply effort precedence or the server budget fallback to a raw body.

        Explicit ``output_config.effort`` removes an adaptive thinking budget.
        Manual enabled thinking retains a normalized budget, using the server
        fallback when the client omits it. Without effort, budget resolution is
        unchanged.
        """
        # Category A: models with no reasoning surface cannot accept thinking or
        # output_config.effort. Strip both before any resolution so the native
        # passthrough forwards a valid body. Static heuristic first (survives a
        # cold catalog), then the warm-cache capability flag.
        if _known_reasoning_support(actual_model) is False:
            body.pop("thinking", None)
            CopilotOutboundContract._strip_output_config_effort(body)
            return body
        from router_maestro.routing import get_router

        _cache_router = get_router()
        if hasattr(_cache_router, "_models_cache"):
            _entry = _cache_router._models_cache.get(actual_model)
            if _entry is not None and _entry[1].supports_thinking is False:
                body.pop("thinking", None)
                CopilotOutboundContract._strip_output_config_effort(body)
                return body

        effort = CopilotOutboundContract.sanitize_output_config(body)
        client_thinking = body.get("thinking")

        model_router = None
        model_info = None

        if effort is not None:
            effort = CopilotOutboundContract.resolve_native_effort(
                effort,
                reasoning_effort_values,
                provider="github-copilot",
                model=actual_model,
            )
            # Update effort in place: siblings the sanitizer preserved (e.g.
            # ``format``) belong to GHC and must survive effort resolution.
            body["output_config"] = {**body["output_config"], "effort": effort}

        if isinstance(client_thinking, dict) and client_thinking.get("type") == "adaptive":
            thinking = dict(client_thinking)
            thinking.pop("budget_tokens", None)
            body["thinking"] = thinking
            return body

        if effort is not None:
            if not (
                isinstance(client_thinking, dict)
                and client_thinking.get("type") in ("enabled", "disabled")
            ):
                return body

        from router_maestro.runtime import get_current_request_context

        context = get_current_request_context()
        if context is not None:
            priorities = context.config
        else:
            from router_maestro.config import load_priorities_config

            priorities = load_priorities_config()
        thinking_config = priorities.thinking

        client_budget = None
        client_type = None
        if isinstance(client_thinking, dict):
            client_budget = client_thinking.get("budget_tokens")
            client_type = client_thinking.get("type")

        if model_router is None:
            from router_maestro.routing import get_router

            model_router = get_router()
        if model_info is None and hasattr(model_router, "_models_cache"):
            cache_entry = model_router._models_cache.get(actual_model)
            if cache_entry:
                _, model_info = cache_entry

        supports_thinking = model_info.supports_thinking if model_info else True
        max_output = (model_info.max_output_tokens or 16384) if model_info else 16384
        request_max_output = body.get("max_tokens")
        if isinstance(request_max_output, int):
            max_output = min(max_output, request_max_output)

        budget, thinking_type = resolve_thinking_budget(
            client_budget=client_budget,
            client_thinking_type=client_type,
            model_id=actual_model,
            max_output_tokens=max_output,
            thinking_config=thinking_config,
            supports_thinking=supports_thinking,
        )

        if thinking_type == "adaptive":
            adaptive = dict(client_thinking) if isinstance(client_thinking, dict) else {}
            adaptive["type"] = "adaptive"
            adaptive.pop("budget_tokens", None)
            body["thinking"] = adaptive
            if budget != client_budget or thinking_type != client_type:
                logger.debug(
                    "Thinking budget adjusted: %s/%s -> %s/%s for model=%s",
                    client_type,
                    client_budget,
                    thinking_type,
                    budget,
                    actual_model,
                )
        elif thinking_type == "enabled" and budget is not None:
            enabled = dict(client_thinking) if isinstance(client_thinking, dict) else {}
            enabled["type"] = "enabled"
            enabled["budget_tokens"] = budget
            body["thinking"] = enabled
            if budget != client_budget or thinking_type != client_type:
                logger.debug(
                    "Thinking budget adjusted: %s/%s -> %s/%s for model=%s",
                    client_type,
                    client_budget,
                    thinking_type,
                    budget,
                    actual_model,
                )
        else:
            body.pop("thinking", None)
            if client_type is not None:
                logger.debug(
                    "Thinking removed: client had %s/%s, resolved to %s for model=%s",
                    client_type,
                    client_budget,
                    thinking_type,
                    actual_model,
                )

        return body


class CopilotProvider(BaseProvider):
    """GitHub Copilot provider."""

    name = "github-copilot"

    _copilot_outbound_contract = CopilotOutboundContract()

    @property
    def outbound_contract(self) -> CopilotOutboundContract:
        return self._copilot_outbound_contract

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(operations=frozenset(Operation))

    def bindings(self) -> tuple[EndpointBinding, ...]:
        """Declare every Copilot generation transport in one provider-owned place."""
        bindings = getattr(self, "_generation_bindings", None)
        if bindings is not None:
            return bindings

        dialect = CopilotProviderDialect(self)
        executor = CopilotHttpExecutor(self)
        bindings = (
            EndpointBinding(
                id=COPILOT_ANTHROPIC_MESSAGES_BINDING,
                protocol=WireProtocol.ANTHROPIC_MESSAGES,
                capabilities=ProviderCapabilities(
                    operations=frozenset({Operation.NATIVE_ANTHROPIC})
                ),
                dialect=dialect,
                executor=executor,
            ),
            EndpointBinding(
                id=COPILOT_OPENAI_CHAT_BINDING,
                protocol=WireProtocol.OPENAI_CHAT,
                capabilities=ProviderCapabilities(
                    operations=frozenset({Operation.CHAT, Operation.CHAT_STREAM})
                ),
                dialect=dialect,
                executor=executor,
            ),
            EndpointBinding(
                id=COPILOT_OPENAI_RESPONSES_BINDING,
                protocol=WireProtocol.OPENAI_RESPONSES,
                capabilities=ProviderCapabilities(
                    operations=frozenset({Operation.RESPONSES, Operation.RESPONSES_STREAM})
                ),
                dialect=dialect,
                executor=executor,
            ),
        )
        self._generation_bindings = bindings
        return bindings

    def transport_preferences(
        self,
        ingress_protocol: WireProtocol | None = None,
    ) -> tuple[str, ...]:
        """Return Copilot's ingress-specific transport preference contract."""
        preferences = {
            WireProtocol.ANTHROPIC_MESSAGES: (
                COPILOT_ANTHROPIC_MESSAGES_BINDING,
                COPILOT_OPENAI_RESPONSES_BINDING,
                COPILOT_OPENAI_CHAT_BINDING,
            ),
            WireProtocol.OPENAI_CHAT: (
                COPILOT_OPENAI_CHAT_BINDING,
                COPILOT_OPENAI_RESPONSES_BINDING,
                COPILOT_ANTHROPIC_MESSAGES_BINDING,
            ),
            WireProtocol.OPENAI_RESPONSES: (
                COPILOT_OPENAI_RESPONSES_BINDING,
                COPILOT_OPENAI_CHAT_BINDING,
                COPILOT_ANTHROPIC_MESSAGES_BINDING,
            ),
            WireProtocol.GEMINI: (
                COPILOT_OPENAI_RESPONSES_BINDING,
                COPILOT_OPENAI_CHAT_BINDING,
                COPILOT_ANTHROPIC_MESSAGES_BINDING,
            ),
        }
        if ingress_protocol is None:
            return tuple(binding.id for binding in self.bindings())
        return preferences.get(ingress_protocol, tuple(binding.id for binding in self.bindings()))

    # Recycle the HTTP/2 client after this many seconds to avoid GOAWAY races
    _CLIENT_MAX_AGE = 300  # 5 minutes

    def __init__(self) -> None:
        self._auth_session = CopilotAuthSession()
        self._transport = CopilotTransport(self._auth_session)
        self._catalog = CopilotCatalog()
        self._chat_codec = CopilotChatCodec()
        self._responses_codec = CopilotResponsesCodec()

    @property
    def _models_ttl_cache(self):
        return self._catalog.models_ttl_cache

    @_models_ttl_cache.setter
    def _models_ttl_cache(self, value) -> None:
        self._catalog.models_ttl_cache = value

    @property
    def auth_manager(self):
        return self._auth_session.auth_manager

    @auth_manager.setter
    def auth_manager(self, value) -> None:
        self._auth_session.auth_manager = value

    @property
    def _cached_token(self) -> str | None:
        return self._auth_session.cached_token

    @_cached_token.setter
    def _cached_token(self, value: str | None) -> None:
        self._auth_session.cached_token = value

    @property
    def _token_expires(self) -> int:
        return self._auth_session.token_expires

    @_token_expires.setter
    def _token_expires(self, value: int) -> None:
        self._auth_session.token_expires = value

    @property
    def _api_base(self) -> str:
        return self._auth_session.api_base

    @_api_base.setter
    def _api_base(self, value: str) -> None:
        self._auth_session.api_base = value

    @property
    def _token_refresh_lock(self):
        return self._auth_session.token_refresh_lock

    @_token_refresh_lock.setter
    def _token_refresh_lock(self, value) -> None:
        self._auth_session.token_refresh_lock = value

    @property
    def _client(self) -> httpx.AsyncClient | None:
        return self._transport.client

    @_client.setter
    def _client(self, value: httpx.AsyncClient | None) -> None:
        self._transport.client = value

    @property
    def _client_created_at(self) -> float:
        return self._transport.client_created_at

    @_client_created_at.setter
    def _client_created_at(self, value: float) -> None:
        self._transport.client_created_at = value

    def is_authenticated(self) -> bool:
        """Check if authenticated with GitHub Copilot."""
        return self._auth_session.is_authenticated()

    def model_aliases(self) -> Mapping[str, str]:
        """Synthetic model ids routed to real GHC models (see COPILOT_MODEL_ALIASES)."""
        return COPILOT_MODEL_ALIASES

    async def ensure_token(self, force: bool = False) -> None:
        """Ensure we have a valid Copilot token, refreshing if needed.

        Args:
            force: Re-mint the Copilot token even if the locally-cached one
                looks unexpired. Used by the 401/403 retry path when the
                upstream rejects a token our clock still considers valid.
        """
        await self._auth_session.ensure_token(
            force,
            persist=self._persist_credential,
            mint=get_copilot_token,
        )

    async def _persist_credential(self, cred: OAuthCredential) -> None:
        """Store and flush a credential to auth.json off the event loop."""
        await self._auth_session.persist_credential(cred)

    async def _refresh_for_auth_status(self, path: str, status_code: int) -> bool:
        """Decide whether an auth-status response warrants a forced re-mint.

        Shared by every Copilot call site (chat, responses, models — streaming
        and not) so the retry policy lives in exactly one place. Only 401/403
        trigger a refresh; 429/5xx pass through to the caller. A 403 we can't
        heal (policy/entitlement/quota) just costs one wasted re-mint+retry,
        which is cheaper than a token-age heuristic that would blind us to a
        genuinely revoked token in its first seconds. Returns True iff a forced
        refresh was performed and the caller should retry once.
        """
        return await self._auth_session.refresh_for_auth_status(
            path,
            status_code,
            ensure_token=self.ensure_token,
        )

    def _raise_auth_failure(
        self, path: str, status_code: int, *, model: str | None = None
    ) -> NoReturn:
        """Raise a retryable auth error for a 401/403 that survived the retry.

        Centralizes the policy so every call site (chat, responses, models —
        streaming and not) treats a post-retry auth rejection identically:
        retryable so the router can fall back to other configured providers,
        with a message that tells the user how to recover if Copilot is their
        only provider.
        """
        self._auth_session.raise_auth_failure(path, status_code, model=model)

    def _raise_unsupported_operation_if_applicable(
        self,
        status_code: int,
        body: bytes,
        *,
        model: str,
        cause: BaseException | None = None,
    ) -> None:
        """Classify Copilot's exact, bounded unsupported-API error payload."""
        if status_code != 400 or len(body) > _MAX_COPILOT_ERROR_BODY_BYTES:
            return
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        error = payload.get("error")
        if not isinstance(error, dict):
            return
        if error.get("code") != _COPILOT_UNSUPPORTED_OPERATION_CODE:
            return
        raise ProviderError(
            "Copilot does not support this API operation for the requested model",
            status_code=400,
            retryable=False,
            kind=ProviderFailureKind.UNSUPPORTED_OPERATION,
            upstream_status_code=400,
            provider=self.name,
            model=model,
            cause=cause,
        ) from cause

    @staticmethod
    def _failure_signal(status_code: int, body: bytes) -> ProviderFailureSignal | None:
        """Classify one exact Copilot failure without retaining its body."""
        if status_code == 400 and body == _EXACT_COPILOT_BARE_BAD_REQUEST:
            return ProviderFailureSignal.COPILOT_BARE_BAD_REQUEST
        return None

    async def _send_with_auth_retry(
        self,
        method: str,
        path: str,
        *,
        client: httpx.AsyncClient | None = None,
        json: dict | None = None,
        headers_kwargs: dict | None = None,
        timeout: Any = TIMEOUT_NON_STREAMING,
        model: str | None = None,
        prepared_headers: Mapping[str, str] | None = None,
        strip_anthropic_beta: bool = False,
    ) -> httpx.Response:
        """Send a non-streaming Copilot request, force-refreshing once on 401/403.

        Shared by chat/responses (POST, shared pool) and models (GET, fresh
        short-lived ``client``). Auth rejections on a token our clock still
        considers valid self-heal by re-minting and retrying once; a 401/403
        that survives the retry raises a retryable auth error so the router can
        fall back. 429/5xx pass through unchanged for the caller to handle.

        Also handles connection-level errors (HTTP/2 GOAWAY, pool timeouts) by
        recycling the HTTP client and retrying once.
        """
        get_headers: Callable[..., dict[str, str]] = self._get_headers
        if prepared_headers or strip_anthropic_beta:
            header_hints = dict(prepared_headers or {})

            def get_headers_with_hints(**kwargs: Any) -> dict[str, str]:
                # Authentication and request IDs are regenerated for every retry;
                # prepared hints contain only stable protocol-derived headers.
                headers = {**header_hints, **self._get_headers(**kwargs)}
                if strip_anthropic_beta:
                    headers = {
                        name: value
                        for name, value in headers.items()
                        if name.lower() != "anthropic-beta"
                    }
                return headers

            get_headers = get_headers_with_hints

        return await self._transport.send_with_auth_retry(
            method,
            path,
            client=client,
            json=json,
            headers_kwargs=headers_kwargs,
            timeout=timeout,
            model=model,
            get_client=self._get_client,
            get_headers=get_headers,
            recycle_client=self._recycle_client,
            refresh_for_auth_status=self._refresh_for_auth_status,
            raise_auth_failure=self._raise_auth_failure,
        )

    async def _recycle_client(self) -> None:
        """Close and discard the current HTTP client so the next call creates a fresh one."""
        await self._transport.recycle_client()

    @contextlib.asynccontextmanager
    async def _stream_with_auth_retry(
        self,
        path: str,
        *,
        json: dict,
        headers_kwargs: dict,
        model: str | None = None,
        prepared_headers: Mapping[str, str] | None = None,
        strip_anthropic_beta: bool = False,
    ) -> AsyncIterator[httpx.Response]:
        """Open a Copilot stream, force-refreshing and retrying once on 401/403.

        Mirrors ``_send_with_auth_retry`` for streaming. The 401/403 is detected
        from the response status *before* any chunk is yielded, so the retry
        never replays partial output. A 401/403 that survives the retry raises a
        retryable auth error (consistent with the non-streaming path); other
        statuses are yielded for the caller to pre-read and raise on.

        Also handles connection-level errors by recycling the HTTP client.
        """
        get_headers: Callable[..., dict[str, str]] = self._get_headers
        if prepared_headers or strip_anthropic_beta:
            header_hints = dict(prepared_headers or {})

            def get_headers_with_hints(**kwargs: Any) -> dict[str, str]:
                headers = {**header_hints, **self._get_headers(**kwargs)}
                if strip_anthropic_beta:
                    headers = {
                        name: value
                        for name, value in headers.items()
                        if name.lower() != "anthropic-beta"
                    }
                return headers

            get_headers = get_headers_with_hints

        async with self._transport.stream_with_auth_retry(
            path,
            json=json,
            headers_kwargs=headers_kwargs,
            model=model,
            get_client=self._get_client,
            get_headers=get_headers,
            recycle_client=self._recycle_client,
            refresh_for_auth_status=self._refresh_for_auth_status,
            raise_auth_failure=self._raise_auth_failure,
        ) as response:
            yield response

    def _url(self, path: str) -> str:
        """Build a Copilot API URL from the token-advertised API base."""
        return self._transport.url(path)

    @staticmethod
    def _chat_initiator(messages: list[Message] | None) -> str:
        """Infer Copilot X-Initiator for chat-completions payloads."""
        return CopilotTransport.chat_initiator(messages)

    @staticmethod
    def _responses_initiator(response_input: str | list[dict[str, Any]] | None) -> str:
        """Infer Copilot X-Initiator for Responses API payloads."""
        return CopilotTransport.responses_initiator(response_input)

    def _get_headers(
        self,
        vision_request: bool = False,
        *,
        messages: list[Message] | None = None,
        response_input: str | list[dict[str, Any]] | None = None,
        intent: str = "conversation-panel",
    ) -> dict[str, str]:
        """Get headers for Copilot API requests.

        Args:
            vision_request: Whether this request contains images (vision)
        """
        return self._transport.headers(
            vision_request,
            messages=messages,
            response_input=response_input,
            intent=intent,
        )

    def _catalog_effort_values(self, model: str) -> list[str] | None:
        """Look up the catalog-advertised reasoning_effort allowlist for ``model``.

        Pulls from the in-memory model cache only — never blocks the request
        on a network fetch. Returns ``None`` if the cache is cold or the model
        isn't in it, in which case ``apply_copilot_chat_reasoning`` falls back
        to the hardcoded heuristic.
        """
        return self._catalog.effort_values(model)

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create a reusable HTTP client.

        Recycles the client after _CLIENT_MAX_AGE seconds to avoid HTTP/2
        GOAWAY races from server-side connection lifetime enforcement.
        """
        self._transport.client_max_age = self._CLIENT_MAX_AGE
        return self._transport.get_client()

    async def close(self) -> None:
        """Close the HTTP client. Called during app shutdown."""
        try:
            await self._catalog.aclose()
        finally:
            await self._transport.close()

    async def count_native_anthropic_tokens(
        self,
        payload: dict[str, Any],
        *,
        model: str,
    ) -> int:
        """Return an exact native Anthropic input-token count from Copilot."""
        await self.ensure_token()
        try:
            response = await self._send_with_auth_retry(
                "POST",
                COPILOT_COUNT_TOKENS_PATH,
                json=payload,
                model=model,
            )
            response.raise_for_status()
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("Copilot native token-count response must be an object")
                input_tokens = data.get("input_tokens")
                if (
                    not isinstance(input_tokens, int)
                    or isinstance(input_tokens, bool)
                    or input_tokens < 0
                ):
                    raise TypeError(
                        "Copilot native token-count input_tokens must be a non-negative integer"
                    )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self._raise_protocol_error(self.name, model, error)
            return input_tokens
        except httpx.HTTPStatusError as error:
            self._raise_http_status_error(
                "Copilot",
                error,
                logger,
                include_body=True,
                provider=self.name,
                model=model,
            )
        except httpx.TimeoutException as error:
            self._raise_timeout_error(
                "Copilot",
                error,
                logger,
                provider=self.name,
                model=model,
            )
        except httpx.HTTPError as error:
            self._raise_http_error(
                "Copilot",
                error,
                logger,
                provider=self.name,
                model=model,
            )

    def _build_native_messages_payload(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool,
    ) -> dict[str, Any]:
        """Prepare one copy-on-write native Messages attempt.

        The ingress body remains untouched so another transport can reuse it.
        Copilot-specific option reconciliation and the forward allowlist live
        here instead of in an API route.
        """
        body = dict(payload)
        body["model"] = model
        body["stream"] = stream
        if body.get("top_p") is not None and not self.outbound_contract.allows_top_p(
            Operation.NATIVE_ANTHROPIC,
            model=model,
        ):
            raise RequestOptionError(
                "GitHub Copilot Messages does not support request option 'top_p' for this model",
                parameter="top_p",
                provider=self.name,
                model=model,
            )
        self.outbound_contract.reject_unpreservable_native_options(body)
        self.outbound_contract.apply_native_anthropic_thinking(
            body,
            model,
            self._catalog_effort_values(model),
        )
        # Identity forwarding is future-compatible: preserve fields RM does not
        # know yet and remove only options Copilot is known to reject.
        for field in _COPILOT_NATIVE_ANTHROPIC_LOCALLY_STRIPPED_FIELDS:
            body.pop(field, None)
        return body

    async def messages_completion(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
    ) -> ChatResponse:
        """Execute Copilot's native Anthropic Messages transport."""
        await self.ensure_token()
        body = self._build_native_messages_payload(payload, model=model, stream=False)
        include_reasoning = isinstance(body.get("thinking"), dict) and body["thinking"].get(
            "type"
        ) in {"enabled", "adaptive"}
        try:
            response = await self._send_with_auth_retry(
                "POST",
                COPILOT_MESSAGES_PATH,
                json=body,
                model=model,
            )
            response.raise_for_status()
            try:
                data = response.json()
                result = decode_message_response(
                    data,
                    fallback_model=model,
                    include_reasoning=include_reasoning,
                )
            except (json.JSONDecodeError, AnthropicCodecError, TypeError, ValueError) as error:
                self._raise_protocol_error(self.name, model, error)
            return result
        except httpx.HTTPStatusError as error:
            self._raise_unsupported_operation_if_applicable(
                error.response.status_code,
                error.response.content,
                model=model,
                cause=error,
            )
            self._raise_http_status_error(
                "Copilot",
                error,
                logger,
                include_body=True,
                provider=self.name,
                model=model,
                signal=self._failure_signal(
                    error.response.status_code,
                    error.response.content,
                ),
            )
        except httpx.TimeoutException as error:
            self._raise_timeout_error(
                "Copilot",
                error,
                logger,
                provider=self.name,
                model=model,
            )
        except httpx.HTTPError as error:
            self._raise_http_error(
                "Copilot",
                error,
                logger,
                provider=self.name,
                model=model,
            )

    async def messages_completion_stream(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Stream Copilot's native Anthropic Messages transport.

        Authentication retries happen before any response frame is consumed.
        Thinking signatures are never stripped or replayed on failure.
        """
        await self.ensure_token()
        body = self._build_native_messages_payload(payload, model=model, stream=True)
        include_reasoning = isinstance(body.get("thinking"), dict) and body["thinking"].get(
            "type"
        ) in {"enabled", "adaptive"}
        decoder = AnthropicStreamDecoder(include_reasoning=include_reasoning)
        try:
            async with self._stream_with_auth_retry(
                COPILOT_MESSAGES_PATH,
                json=body,
                headers_kwargs={},
                model=model,
            ) as response:
                if response.status_code >= 400:
                    error_body = await response.aread()
                    self._raise_unsupported_operation_if_applicable(
                        response.status_code,
                        error_body,
                        model=model,
                    )
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw_data = line[6:]
                    if not raw_data:
                        continue
                    try:
                        data = json.loads(raw_data)
                        if isinstance(data, dict) and data.get("type") == "error":
                            error_data = data.get("error")
                            message = (
                                error_data.get("message")
                                if isinstance(error_data, dict)
                                else "Copilot Messages stream failed"
                            )
                            raise ProviderError(
                                str(message),
                                status_code=502,
                                retryable=False,
                                kind=ProviderFailureKind.UPSTREAM_PROTOCOL,
                                upstream_status_code=200,
                                provider=self.name,
                                model=model,
                            )
                        chunks = decoder.decode_event(data)
                    except (json.JSONDecodeError, AnthropicCodecError) as error:
                        self._raise_protocol_error(self.name, model, error)
                    for chunk in chunks:
                        yield chunk
                try:
                    decoder.finalize()
                except AnthropicCodecError as error:
                    self._raise_protocol_error(self.name, model, error)
        except httpx.HTTPStatusError as error:
            try:
                error_body = error.response.content
            except httpx.ResponseNotRead:
                error_body = b""
            self._raise_unsupported_operation_if_applicable(
                error.response.status_code,
                error_body,
                model=model,
                cause=error,
            )
            self._raise_http_status_error(
                "Copilot",
                error,
                logger,
                stream=True,
                include_body=True,
                provider=self.name,
                model=model,
                signal=self._failure_signal(error.response.status_code, error_body),
            )
        except httpx.TimeoutException as error:
            self._raise_timeout_error(
                "Copilot",
                error,
                logger,
                stream=True,
                provider=self.name,
                model=model,
            )
        except httpx.HTTPError as error:
            self._raise_http_error(
                "Copilot",
                error,
                logger,
                stream=True,
                provider=self.name,
                model=model,
            )

    @staticmethod
    def _sanitize_surrogates(text: str) -> str:
        """Remove lone surrogate characters that cannot be encoded as UTF-8."""
        return CopilotChatCodec.sanitize_surrogates(text)

    def _sanitize_content(self, content: str | list | None) -> str | list | None:
        """Sanitize message content to remove lone surrogate characters."""
        return self._chat_codec.sanitize_content(content)

    def _build_messages_payload(self, request: ChatRequest) -> tuple[list[dict], bool]:
        """Build messages payload and detect if images are present.

        Args:
            request: The chat request

        Returns:
            Tuple of (messages list, has_images flag)
        """
        return self._chat_codec.build_messages_payload(request)

    def _responses_input_has_vision(self, value: Any, depth: int = 0) -> bool:
        """Whether a Responses API input contains an image block."""
        return self._responses_codec.input_has_vision(value, depth)

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion via Copilot."""
        await self.ensure_token()

        messages, has_images = self._build_messages_payload(request)

        payload = self._build_chat_payload(request, stream=False)

        logger.debug(
            "Copilot chat completion: model=%s thinking_budget=%s reasoning_effort=%s "
            "payload_thinking_budget=%s payload_reasoning_effort=%s",
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            payload.get("thinking_budget"),
            payload.get("reasoning_effort"),
        )
        try:
            response = await self._send_with_auth_retry(
                "POST",
                COPILOT_CHAT_PATH,
                json=payload,
                headers_kwargs={"vision_request": has_images, "messages": request.messages},
                model=request.model,
            )
            response.raise_for_status()
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("chat response must be an object")
                choices = data.get("choices")
                if not isinstance(choices, list):
                    raise TypeError("chat response choices must be a list")
                model = self._validated_response_model(data, request.model)
                usage = self._validated_token_usage(
                    data.get("usage"),
                    fields=("prompt_tokens", "completion_tokens", "total_tokens"),
                    label="chat response",
                    detail_fields={
                        "prompt_tokens_details": ("cached_tokens",),
                        "completion_tokens_details": ("reasoning_tokens",),
                    },
                )
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                self._raise_protocol_error(self.name, request.model, e)

            bare_lower = (
                request.model.split("/", 1)[1] if "/" in request.model else request.model
            ).lower()
            reasoning_capable = bare_lower.startswith("claude-") and _claude_supports_reasoning(
                bare_lower
            )
            try:
                result = self._chat_codec.decode_response(
                    data,
                    request,
                    model=model,
                    usage=usage,
                    choices=choices,
                    reasoning_capable=reasoning_capable,
                    validated_optional_string=self._validated_optional_string,
                )
            except (KeyError, TypeError, ValueError) as error:
                if not choices:
                    logger.error(
                        "Copilot API returned empty choices: model=%s has_usage=%s",
                        request.model,
                        "usage" in data,
                    )
                self._raise_protocol_error(self.name, request.model, error)
            logger.debug("Copilot chat completion successful")
            return result
        except httpx.HTTPStatusError as e:
            self._raise_unsupported_operation_if_applicable(
                e.response.status_code,
                e.response.content,
                model=request.model,
                cause=e,
            )
            self._raise_http_status_error(
                "Copilot",
                e,
                logger,
                include_body=True,
                provider=self.name,
                model=request.model,
                signal=self._failure_signal(e.response.status_code, e.response.content),
            )
        except httpx.TimeoutException as e:
            self._raise_timeout_error("Copilot", e, logger, provider=self.name, model=request.model)
        except httpx.HTTPError as e:
            self._raise_http_error("Copilot", e, logger, provider=self.name, model=request.model)

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Generate a streaming chat completion via Copilot."""
        await self.ensure_token()

        messages, has_images = self._build_messages_payload(request)

        payload = self._build_chat_payload(request, stream=True)

        logger.debug(
            "Copilot streaming chat: model=%s thinking_budget=%s reasoning_effort=%s "
            "payload_thinking_budget=%s payload_reasoning_effort=%s",
            request.model,
            request.thinking_budget,
            request.reasoning_effort,
            payload.get("thinking_budget"),
            payload.get("reasoning_effort"),
        )
        try:
            async with self._stream_with_auth_retry(
                COPILOT_CHAT_PATH,
                json=payload,
                headers_kwargs={"vision_request": has_images, "messages": request.messages},
                model=request.model,
            ) as response:
                # Streamed responses defer body reads; if the upstream returns
                # an error status, pull the body *inside* the stream context
                # so the connection is still open. Reading after the
                # `async with` exits raises StreamClosed and the helper would
                # log "Copilot stream API error: 4xx -" with no upstream
                # detail (the original symptom this fix addresses).
                if response.status_code >= 400:
                    with contextlib.suppress(Exception):
                        await response.aread()
                response.raise_for_status()

                async for chunk in self._chat_codec.decode_stream(
                    response.aiter_lines(),
                    request,
                    provider_name=self.name,
                ):
                    yield chunk
        except httpx.HTTPStatusError as e:
            try:
                error_body = e.response.content
            except httpx.ResponseNotRead:
                error_body = b""
            self._raise_unsupported_operation_if_applicable(
                e.response.status_code,
                error_body,
                model=request.model,
                cause=e,
            )
            self._raise_http_status_error(
                "Copilot",
                e,
                logger,
                stream=True,
                include_body=True,
                provider=self.name,
                model=request.model,
                signal=self._failure_signal(e.response.status_code, error_body),
            )
        except httpx.TimeoutException as e:
            self._raise_timeout_error(
                "Copilot",
                e,
                logger,
                stream=True,
                provider=self.name,
                model=request.model,
            )
        except httpx.HTTPError as e:
            logger.error(
                "Copilot stream HTTP error: type=%s model=%s vision=%s url=%s",
                type(e).__name__,
                request.model,
                has_images,
                self._url(COPILOT_CHAT_PATH),
            )
            self._raise_http_error(
                "Copilot",
                e,
                logger,
                stream=True,
                provider=self.name,
                model=request.model,
            )

    async def list_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """List available Copilot models from API with caching.

        Args:
            force_refresh: Force refresh the cache

        Returns:
            List of available models
        """
        return await self._catalog.list_models(
            force_refresh,
            provider_name=self.name,
            ensure_token=self.ensure_token,
            send=self._send_with_auth_retry,
            normalize_endpoints=normalize_supported_endpoints,
            derive_operations=copilot_operation_capabilities,
            raise_protocol_error=self._raise_protocol_error,
        )

    # Tools that are not supported by Copilot Responses API.
    # ``namespace`` items are conditionally allowed: if they carry an inner
    # ``tools`` array (Codex's MCP registry shape) they MUST pass through
    # so Copilot can resolve namespaced function_calls like
    # ``kusto/execute_query``. Explicitly unsupported or malformed tools are
    # client errors; silently dropping them changes the requested semantics.
    UNSUPPORTED_TOOL_TYPES = {
        "web_search",
        "web_search_preview",
        "code_interpreter",
    }

    def _filter_unsupported_tools(
        self,
        tools: list[dict] | None,
        *,
        model: str | None = None,
    ) -> list[dict] | None:
        """Validate and preserve tools supported by the Copilot Responses API.

        Delegates to the provider's outbound contract (the single source of the
        Copilot tool-filter rule).
        """
        return self.outbound_contract.filter_tools(
            tools,
            operation=Operation.RESPONSES,
            model=model,
        )

    def _build_responses_payload(self, request: ResponsesRequest) -> dict:
        """Build payload for Responses API request.

        Args:
            request: The responses request

        Returns:
            Payload dictionary for the API
        """
        return self._responses_codec.build_payload(
            request,
            provider_name=self.name,
            validate_extensions=self._validate_provider_extensions,
            catalog_effort_values=self._catalog_effort_values(request.model),
            resolve_reasoning=self.outbound_contract.resolve_reasoning,
            allows_temperature=self.outbound_contract.allows_temperature,
            allows_top_p=self.outbound_contract.allows_top_p,
            filter_tools=self.outbound_contract.filter_tools,
            normalize_input=self.outbound_contract.normalize_responses_input,
        )

    def validate_responses_request(self, request: ResponsesRequest) -> None:
        """Exercise the Responses payload policy without upstream I/O."""
        self._build_responses_payload(request)

    def _extract_response_content(self, data: dict) -> tuple[str, str | None]:
        """Extract text and refusal content from a Responses API response.

        Args:
            data: The response JSON data

        Returns:
            The extracted text content
        """
        return self._responses_codec.extract_response_content(data)

    def _extract_reasoning(self, data: dict) -> tuple[str | None, str | None, str | None]:
        """Extract one atomic reasoning item's summary, upstream id, and blob.

        Returns ``(thinking_text, thinking_id, thinking_signature)``. All
        three are ``None`` when the response has no reasoning output. The
        ``id`` and the ``encrypted_content`` blob must be round-tripped
        together — the blob is signed against its id, so pairing it with a
        locally-generated id 400s the next turn.
        """
        return self._responses_codec.extract_reasoning(data)

    def _extract_reasoning_item(self, data: dict) -> dict[str, Any] | None:
        """Extract the complete upstream reasoning item for lossless replay."""
        return self._responses_codec.extract_reasoning_item(data)

    def _extract_tool_calls(self, data: dict) -> list[ResponsesToolCall]:
        """Extract tool calls from Responses API response.

        Args:
            data: The response JSON data

        Returns:
            List of tool calls
        """
        return self._responses_codec.extract_tool_calls(data)

    @staticmethod
    def _validate_responses_output(data: dict) -> bool:
        """Validate the structural fields consumed by the Responses adapter."""
        return CopilotResponsesCodec.validate_output(data)

    @staticmethod
    def _validate_responses_usage(usage: object) -> None:
        """Validate the Responses usage fields consumed by downstream adapters."""
        CopilotResponsesCodec.validate_usage(usage)

    async def responses_completion(self, request: ResponsesRequest) -> ResponsesResponse:
        """Generate a Responses API completion via Copilot (for Codex models)."""
        await self.ensure_token()

        payload = self._build_responses_payload(request)

        logger.debug("Copilot responses completion: model=%s", request.model)
        try:
            response = await self._send_with_auth_retry(
                "POST",
                COPILOT_RESPONSES_PATH,
                json=payload,
                headers_kwargs={
                    "vision_request": self._responses_input_has_vision(request.input),
                    "response_input": (
                        request.input if isinstance(request.input, (str, list)) else None
                    ),
                },
                model=request.model,
            )
            response.raise_for_status()
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("Responses response must be an object")
                model = self._validated_response_model(data, request.model)
                self._validate_responses_usage(data.get("usage"))
                has_deliverable = self._validate_responses_output(data)
                terminal_outcome = _responses_terminal_outcome(data)
                if (
                    terminal_outcome.transport is TransportTermination.EXCEPTION
                    and terminal_outcome.error is not None
                    and terminal_outcome.error.code == "upstream_protocol_error"
                ):
                    raise ValueError(terminal_outcome.error.message)
                if (
                    terminal_outcome.response_status is ResponseStatus.COMPLETED
                    and not has_deliverable
                ):
                    raise ValueError("completed Responses response contains no deliverable output")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                self._raise_protocol_error(self.name, request.model, e)

            content, refusal = self._extract_response_content(data)
            tool_calls = self._extract_tool_calls(data)
            thinking, thinking_id, thinking_sig = self._extract_reasoning(data)
            reasoning_item = self._extract_reasoning_item(data)

            usage = None
            if "usage" in data:
                usage = data["usage"]

            logger.debug("Copilot responses completion successful")
            return ResponsesResponse(
                content=content,
                model=model,
                usage=usage,
                tool_calls=tool_calls if tool_calls else None,
                thinking=thinking,
                thinking_id=thinking_id,
                thinking_signature=thinking_sig,
                terminal_outcome=terminal_outcome,
                refusal=refusal,
                reasoning_item=reasoning_item,
            )
        except httpx.HTTPStatusError as e:
            self._raise_unsupported_operation_if_applicable(
                e.response.status_code,
                e.response.content,
                model=request.model,
                cause=e,
            )
            self._raise_http_status_error(
                "Copilot",
                e,
                logger,
                include_body=True,
                provider=self.name,
                model=request.model,
                signal=self._failure_signal(e.response.status_code, e.response.content),
            )
        except httpx.TimeoutException as e:
            self._raise_timeout_error("Copilot", e, logger, provider=self.name, model=request.model)
        except httpx.HTTPError as e:
            self._raise_http_error("Copilot", e, logger, provider=self.name, model=request.model)

    async def responses_completion_stream(
        self, request: ResponsesRequest
    ) -> AsyncIterator[ResponsesStreamChunk]:
        """Generate a streaming Responses API completion via Copilot (for Codex models)."""
        await self.ensure_token()

        payload = self._build_responses_payload(request)
        payload["stream"] = True

        logger.debug("Copilot streaming responses: model=%s", request.model)
        logger.debug(
            "Copilot responses payload metadata: model=%s stream=%s tools=%d input_type=%s",
            request.model,
            payload.get("stream"),
            len(payload.get("tools") or []),
            type(request.input).__name__,
        )
        try:
            async with self._stream_with_auth_retry(
                COPILOT_RESPONSES_PATH,
                json=payload,
                headers_kwargs={
                    "vision_request": self._responses_input_has_vision(request.input),
                    "response_input": (
                        request.input if isinstance(request.input, (str, list)) else None
                    ),
                },
                model=request.model,
            ) as response:
                # Check for errors before processing stream
                if response.status_code >= 400:
                    # Read the error body before the context closes
                    error_body = await response.aread()
                    logger.error(
                        "Copilot responses stream API error: %d (body_bytes=%d)",
                        response.status_code,
                        len(error_body),
                    )
                    self._raise_unsupported_operation_if_applicable(
                        response.status_code,
                        error_body,
                        model=request.model,
                    )
                    # 401/403 are handled by _stream_with_auth_retry (refresh +
                    # retry, then a retryable auth error), so they never reach
                    # here; only transient 429/5xx are retryable at this point.
                    status_code = response.status_code
                    retryable = status_code in (429, 500, 502, 503, 504, 529)
                    kind = (
                        ProviderFailureKind.RATE_LIMIT
                        if status_code in (429, 529)
                        else ProviderFailureKind.UPSTREAM_STATUS
                    )
                    raise ProviderError(
                        f"Copilot API error: {status_code}",
                        status_code=status_code,
                        retryable=retryable,
                        kind=kind,
                        upstream_status_code=status_code,
                        provider=self.name,
                        model=request.model,
                        signal=self._failure_signal(status_code, error_body),
                    )

                async for chunk in self._responses_codec.decode_stream(
                    response.aiter_lines(),
                    request,
                ):
                    yield chunk

        except httpx.TimeoutException as e:
            self._raise_timeout_error(
                "Copilot",
                e,
                logger,
                stream=True,
                provider=self.name,
                model=request.model,
            )
        except httpx.HTTPError as e:
            self._raise_http_error(
                "Copilot",
                e,
                logger,
                stream=True,
                provider=self.name,
                model=request.model,
            )

    def _reject_option(self, request: ChatRequest, parameter: str) -> None:
        self._chat_codec.reject_option(request, parameter, provider_name=self.name)

    def _build_chat_payload(self, request: ChatRequest, *, stream: bool) -> dict:
        """Build a Copilot Chat payload with an explicit option policy."""
        payload = self._chat_codec.build_payload(
            request,
            stream=stream,
            validate_extensions=self._validate_provider_extensions,
            apply_reasoning=apply_copilot_chat_reasoning,
            catalog_effort_values=self._catalog_effort_values(request.model),
            provider_name=self.name,
            allows_stop=self.outbound_contract.allows_stop,
        )
        tools = _normalize_gemini_chat_tools(payload.get("tools"), model=request.model)
        if tools is None:
            payload.pop("tools", None)
        else:
            payload["tools"] = tools
        return payload

    def validate_chat_request(self, request: ChatRequest, *, stream: bool) -> None:
        """Exercise the same Chat payload policy as actual execution."""
        self._build_chat_payload(request, stream=stream)


_COPILOT_BINDING_SPECS = {
    COPILOT_ANTHROPIC_MESSAGES_BINDING: (
        WireProtocol.ANTHROPIC_MESSAGES,
        COPILOT_MESSAGES_PATH,
    ),
    COPILOT_OPENAI_CHAT_BINDING: (WireProtocol.OPENAI_CHAT, COPILOT_CHAT_PATH),
    COPILOT_OPENAI_RESPONSES_BINDING: (
        WireProtocol.OPENAI_RESPONSES,
        COPILOT_RESPONSES_PATH,
    ),
}
_COPILOT_RESPONSES_LOCALLY_STRIPPED_FIELDS = frozenset({"store"})
_COPILOT_STREAM_KEEPALIVE_TYPES = frozenset({"copilot_usage", "ping"})
_COPILOT_RESPONSES_INTERNAL_RESPONSE_FIELDS = frozenset({"copilot_usage", "tool_usage"})
_COPILOT_RESPONSES_INTERNAL_USAGE_FIELDS = frozenset(
    {
        "context_details",
        "cost_in_usd_ticks",
        "num_server_side_tools_used",
        "num_sources_used",
    }
)
_COPILOT_RESPONSES_INTERNAL_INPUT_DETAIL_FIELDS = frozenset({"cache_write_tokens"})
# Copilot can expose Azure-style prompt filtering metadata before the first
# Chat choice. It has no public Chat/semantic equivalent and must not make a
# valid cross-protocol stream fail before content arrives.
_COPILOT_CHAT_INTERNAL_RESPONSE_FIELDS = frozenset(
    {"copilot_info_messages", "copilot_usage", "prompt_filter_results"}
)
_COPILOT_CHAT_CHOICE_INTERNAL_FIELDS = frozenset({"content_filter_results"})
# ``reasoning_opaque`` is Copilot-private state. Identity Chat responses must
# not expose it, while a cross-protocol attempt keeps it long enough for the
# bound Chat runtime to assign a stable internal item ID and seal a capsule.
_COPILOT_CHAT_MESSAGE_INTERNAL_FIELDS = frozenset({"padding", "reasoning_opaque"})
_COPILOT_ANTHROPIC_INTERNAL_RESPONSE_FIELDS = frozenset({"copilot_usage", "stop_details"})
_COPILOT_ANTHROPIC_MESSAGE_STOP_INTERNAL_FIELDS = frozenset(
    {"copilot_usage", "amazon-bedrock-invocationMetrics"}
)
_COPILOT_NATIVE_ANTHROPIC_LOCALLY_STRIPPED_FIELDS = frozenset({"mcp_servers", "container"})


class CopilotProviderDialect:
    """Copy-on-write wire preparation for Copilot's three generation APIs."""

    id = "github-copilot"

    def __init__(self, provider: CopilotProvider) -> None:
        self.provider = provider

    async def prepare_attempt(
        self,
        *,
        binding_id: str,
        protocol: WireProtocol,
        model: ModelRef,
        payload: Mapping[str, Any],
        stream: bool,
        request_context: AttemptRequestContext,
    ) -> PreparedAttempt:
        expected_protocol, path = self._binding_spec(binding_id)
        if protocol is not expected_protocol:
            raise ValueError("Copilot binding protocol does not match its binding ID")
        if model.provider != self.provider.name:
            raise ValueError("Copilot attempt model belongs to another provider")

        if protocol is WireProtocol.ANTHROPIC_MESSAGES:
            body = self.provider._build_native_messages_payload(
                payload,
                model=model.upstream_id,
                stream=stream,
            )
            headers = self._anthropic_headers(request_context)
        elif protocol is WireProtocol.OPENAI_CHAT:
            body, headers = self._prepare_chat(
                payload,
                model=model.upstream_id,
                stream=stream,
            )
        elif protocol is WireProtocol.OPENAI_RESPONSES:
            body, headers = self._prepare_responses(
                payload,
                model=model.upstream_id,
                stream=stream,
            )
        else:  # pragma: no cover - the binding table is closed over three protocols
            raise ValueError(f"Unsupported Copilot wire protocol {protocol.value!r}")

        return PreparedAttempt(
            binding_id=binding_id,
            protocol=protocol,
            model=model,
            url=self.provider._url(path),
            payload=body,
            headers=headers,
            stream=stream,
            capture_reasoning_state=(
                protocol is WireProtocol.OPENAI_CHAT
                and request_context.conversion_mode is ConversionMode.SEMANTIC_IR
            ),
            _payload_owned=True,
        )

    @staticmethod
    def _anthropic_headers(request_context: AttemptRequestContext) -> dict[str, str]:
        if not request_context.headers:
            return {}
        value = request_context.header("anthropic-beta")
        if value is None:
            return {}
        from router_maestro.runtime import get_current_request_context

        runtime_context = get_current_request_context()
        stripped = strip_beta_tokens(
            value,
            runtime_context.config.beta_strip if runtime_context is not None else [],
        )
        return {"anthropic-beta": stripped} if stripped is not None else {}

    def _prepare_chat(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body = dict(payload)
        raw_messages = body.get("messages")
        messages, has_images = self._reconcile_chat_messages(raw_messages, model=model)
        body["messages"] = messages
        tools = _normalize_gemini_chat_tools(body.get("tools"), model=model)
        if tools is None:
            body.pop("tools", None)
        else:
            body["tools"] = tools
        body["model"] = model
        body["stream"] = stream

        if stream:
            stream_options = body.get("stream_options")
            if stream_options is None:
                normalized_stream_options: dict[str, Any] = {}
            elif isinstance(stream_options, Mapping):
                normalized_stream_options = dict(stream_options)
            else:
                self._reject_chat_option("stream_options", model=model)
            normalized_stream_options["include_usage"] = True
            body["stream_options"] = normalized_stream_options

        operation = Operation.CHAT_STREAM if stream else Operation.CHAT
        if body.get("stop") is not None and not self.provider.outbound_contract.allows_stop(
            operation,
            model=model,
        ):
            self._reject_chat_option("stop", model=model)

        thinking = body.pop("thinking", None)
        thinking_budget = thinking.get("budget_tokens") if isinstance(thinking, Mapping) else None
        reasoning_effort = body.pop("reasoning_effort", None)
        apply_copilot_chat_reasoning(
            body,
            model,
            thinking_budget,
            reasoning_effort if isinstance(reasoning_effort, str) else None,
            catalog_effort_values=self.provider._catalog_effort_values(model),
        )

        headers = {"X-Initiator": self._chat_initiator_from_wire(messages)}
        if has_images:
            headers["Copilot-Vision-Request"] = "true"
        return body, headers

    def _reconcile_chat_messages(
        self,
        value: Any,
        *,
        model: str,
    ) -> tuple[list[Any], bool]:
        if not isinstance(value, list):
            raise RequestOptionError(
                "messages must be an array",
                parameter="messages",
                provider=self.provider.name,
                model=model,
            )

        messages: list[Any] = []
        has_images = False
        for index, raw_message in enumerate(value):
            if not isinstance(raw_message, Mapping):
                raise RequestOptionError(
                    "message must be an object",
                    parameter=f"messages[{index}]",
                    provider=self.provider.name,
                    model=model,
                )
            role = raw_message.get("role")
            if not isinstance(role, str):
                raise RequestOptionError(
                    "message role must be a string",
                    parameter=f"messages[{index}].role",
                    provider=self.provider.name,
                    model=model,
                )
            message = dict(raw_message)
            if "content" in message:
                message["content"] = self.provider._sanitize_content(message["content"])
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(part, Mapping) and part.get("type") == "image_url" for part in content
            ):
                has_images = True
            messages.append(message)
        return messages, has_images

    @staticmethod
    def _chat_initiator_from_wire(messages: list[Any]) -> str:
        return (
            "agent"
            if any(
                isinstance(message, Mapping) and message.get("role") in {"assistant", "tool"}
                for message in messages
            )
            else "user"
        )

    def _reject_chat_option(self, parameter: str, *, model: str) -> NoReturn:
        raise RequestOptionError(
            f"GitHub Copilot Chat does not support request option '{parameter}'",
            parameter=parameter,
            provider=self.provider.name,
            model=model,
        )

    def _prepare_responses(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        stream: bool,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if "input" not in payload:
            raise RequestOptionError(
                "input is required",
                parameter="input",
                provider=self.provider.name,
                model=model,
            )
        operation = Operation.RESPONSES_STREAM if stream else Operation.RESPONSES
        body = dict(payload)
        for field in _COPILOT_RESPONSES_LOCALLY_STRIPPED_FIELDS:
            body.pop(field, None)
        if body.get("temperature") is not None:
            raise RequestOptionError(
                "GitHub Copilot Responses does not support request option 'temperature'",
                parameter="temperature",
                provider=self.provider.name,
                model=model,
            )
        self.provider.outbound_contract.reconcile_passthrough_body(
            body,
            operation=operation,
            model=model,
            catalog_effort_values=self.provider._catalog_effort_values(model),
            preserve_unknown_fields=True,
        )
        if _requires_namespace_flattening(model):
            _flatten_grok_namespace_body(body, model=model)
        body["model"] = model
        body["stream"] = stream
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
            body["reasoning"] = {"summary": "auto", **reasoning}
            body.setdefault("include", ["reasoning.encrypted_content"])

        response_input = body.get("input")
        headers = {
            "X-Initiator": self.provider._responses_initiator(
                response_input if isinstance(response_input, (str, list)) else None
            )
        }
        if self.provider._responses_input_has_vision(response_input):
            headers["Copilot-Vision-Request"] = "true"
        return body, headers

    @staticmethod
    def _binding_spec(binding_id: str) -> tuple[WireProtocol, str]:
        try:
            return _COPILOT_BINDING_SPECS[binding_id]
        except KeyError:
            raise ValueError(f"Unknown Copilot binding {binding_id!r}") from None


class CopilotHttpExecutor(SharedHttpExecutor):
    """Raw JSON/SSE executor shared by all Copilot endpoint bindings."""

    def __init__(self, provider: CopilotProvider) -> None:
        self.provider = provider
        super().__init__(transport_records_audit=True)

    async def _send(self, attempt: PreparedAttempt, *, timeout: Any) -> httpx.Response:
        path = self._path(attempt)
        await self.provider.ensure_token()
        return await self.provider._send_with_auth_retry(
            attempt.method,
            path,
            json=dict(attempt.payload),
            timeout=timeout,
            model=attempt.model.upstream_id,
            prepared_headers=attempt.headers,
            strip_anthropic_beta=(attempt.protocol is not WireProtocol.ANTHROPIC_MESSAGES),
        )

    @contextlib.asynccontextmanager
    async def _open_stream(
        self,
        attempt: PreparedAttempt,
        *,
        timeout: Any,
    ) -> AsyncIterator[httpx.Response]:
        del timeout
        path = self._path(attempt)
        await self.provider.ensure_token()
        async with self.provider._stream_with_auth_retry(
            path,
            json=dict(attempt.payload),
            headers_kwargs={},
            model=attempt.model.upstream_id,
            prepared_headers=attempt.headers,
            strip_anthropic_beta=(attempt.protocol is not WireProtocol.ANTHROPIC_MESSAGES),
        ) as response:
            yield response

    def _skip_raw_sse_data(self, data: str, attempt: PreparedAttempt) -> bool:
        del attempt
        return data in {"[DONE]", "keepalive"}

    def _skip_sse_frame(
        self,
        frame: Mapping[str, Any],
        attempt: PreparedAttempt,
    ) -> bool:
        if (
            attempt.protocol is WireProtocol.OPENAI_CHAT
            and frame.get("choices") == []
            and frame.get("id") in {None, ""}
            and "prompt_filter_results" in frame
            and frame.get("usage") is None
            and frame.get("error") is None
        ):
            return True
        return frame.get("type") in _COPILOT_STREAM_KEEPALIVE_TYPES

    def _project_payload(
        self,
        payload: dict[str, Any],
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> Mapping[str, Any]:
        return self._project_response(
            attempt.protocol,
            payload,
            model=attempt.model.upstream_id,
            stream=stream,
            capture_reasoning_state=attempt.capture_reasoning_state,
        )

    def _validate_attempt(self, attempt: PreparedAttempt, *, stream: bool) -> None:
        self._path(attempt)
        if attempt.model.provider != self.provider.name:
            raise ValueError("Copilot executor received another provider's model")
        if attempt.stream is not stream:
            mode = "streaming" if stream else "non-streaming"
            raise ValueError(f"Copilot executor received the wrong {mode} attempt mode")

    @staticmethod
    def _project_response(
        protocol: WireProtocol,
        payload: dict[str, Any],
        *,
        model: str,
        stream: bool = False,
        capture_reasoning_state: bool = False,
    ) -> dict[str, Any]:
        """Remove only documented Copilot-private response fields.

        The raw binding deliberately preserves unknown wire extensions.  These
        keys are the small provider-owned exception: forwarding them leaks
        Copilot accounting/transport metadata and makes cross-protocol codecs
        reject otherwise valid responses.
        """
        CopilotHttpExecutor._project_model(payload, model)

        if protocol is WireProtocol.OPENAI_RESPONSES:
            CopilotHttpExecutor._strip_responses_internal_fields(payload)
            nested = payload.get("response")
            if isinstance(nested, dict):
                CopilotHttpExecutor._strip_responses_internal_fields(nested)
                if stream:
                    CopilotHttpExecutor._project_model(nested, model)
            if _requires_namespace_flattening(model):
                payload = _restore_namespaced_function_calls(payload)
            return payload

        if protocol is WireProtocol.OPENAI_CHAT:
            for field in _COPILOT_CHAT_INTERNAL_RESPONSE_FIELDS:
                payload.pop(field, None)
            choices = payload.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    for field in _COPILOT_CHAT_CHOICE_INTERNAL_FIELDS:
                        choice.pop(field, None)
                    for container_name in ("message", "delta"):
                        container = choice.get(container_name)
                        if not isinstance(container, dict):
                            continue
                        internal_fields = _COPILOT_CHAT_MESSAGE_INTERNAL_FIELDS
                        if capture_reasoning_state:
                            internal_fields = internal_fields - {"reasoning_opaque"}
                        for field in internal_fields:
                            container.pop(field, None)
                        if container_name == "message":
                            tool_calls = container.get("tool_calls")
                            if isinstance(tool_calls, list):
                                for tool_call in tool_calls:
                                    if isinstance(tool_call, dict):
                                        tool_call.pop("index", None)
            return payload

        if protocol is not WireProtocol.ANTHROPIC_MESSAGES:
            return payload

        for field in _COPILOT_ANTHROPIC_INTERNAL_RESPONSE_FIELDS:
            payload.pop(field, None)
        nested = payload.get("message")
        if isinstance(nested, dict):
            for field in _COPILOT_ANTHROPIC_INTERNAL_RESPONSE_FIELDS:
                nested.pop(field, None)
            if stream:
                CopilotHttpExecutor._project_model(nested, model)
        if stream and payload.get("type") == "message_stop":
            for field in _COPILOT_ANTHROPIC_MESSAGE_STOP_INTERNAL_FIELDS:
                payload.pop(field, None)
        return payload

    @staticmethod
    def _project_model(payload: dict[str, Any], model: str) -> None:
        """Normalize a valid Copilot response alias to the attempted catalog model."""
        if isinstance(payload.get("model"), str):
            payload["model"] = model

    @staticmethod
    def _strip_responses_internal_fields(payload: dict[str, Any]) -> None:
        CopilotHttpExecutor._repair_responses_usage_from_copilot(payload)
        for field in _COPILOT_RESPONSES_INTERNAL_RESPONSE_FIELDS:
            payload.pop(field, None)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        for field in _COPILOT_RESPONSES_INTERNAL_USAGE_FIELDS:
            usage.pop(field, None)
        input_details = usage.get("input_tokens_details")
        if not isinstance(input_details, dict):
            return
        for field in _COPILOT_RESPONSES_INTERNAL_INPUT_DETAIL_FIELDS:
            input_details.pop(field, None)

    @staticmethod
    def _repair_responses_usage_from_copilot(payload: dict[str, Any]) -> None:
        """Recover zero public usage from Copilot's private billing counters.

        Under sustained traffic, Copilot can return a valid Responses body whose
        standard usage snapshot is all zero while ``copilot_usage.token_details``
        still contains the actual counts. Repair only absent or zero standard
        fields; malformed or already-positive public values retain their normal
        strict-decoder behavior.
        """
        private_usage = payload.get("copilot_usage")
        if not isinstance(private_usage, dict):
            return
        token_details = private_usage.get("token_details")
        if not isinstance(token_details, list):
            return

        counts = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
        saw_count = False
        for detail in token_details:
            if not isinstance(detail, dict):
                continue
            token_type = detail.get("token_type")
            token_count = detail.get("token_count")
            if (
                not isinstance(token_type, str)
                or token_type not in counts
                or not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count < 0
            ):
                continue
            counts[token_type] += token_count
            saw_count = True
        if not saw_count:
            return

        usage = payload.get("usage")
        if usage is None:
            usage = {}
            payload["usage"] = usage
        if not isinstance(usage, dict):
            return

        input_tokens = counts["input"] + counts["cache_read"] + counts["cache_write"]
        output_tokens = counts["output"]
        CopilotHttpExecutor._replace_empty_token_count(usage, "input_tokens", input_tokens)
        CopilotHttpExecutor._replace_empty_token_count(usage, "output_tokens", output_tokens)

        public_input = usage.get("input_tokens")
        public_output = usage.get("output_tokens")
        if (
            isinstance(public_input, int)
            and not isinstance(public_input, bool)
            and isinstance(public_output, int)
            and not isinstance(public_output, bool)
        ):
            CopilotHttpExecutor._replace_empty_token_count(
                usage,
                "total_tokens",
                public_input + public_output,
            )

        if counts["cache_read"] <= 0:
            return
        input_details = usage.get("input_tokens_details")
        if input_details is None:
            input_details = {}
            usage["input_tokens_details"] = input_details
        if isinstance(input_details, dict):
            CopilotHttpExecutor._replace_empty_token_count(
                input_details,
                "cached_tokens",
                counts["cache_read"],
            )

    @staticmethod
    def _replace_empty_token_count(target: dict[str, Any], field: str, value: int) -> None:
        current = target.get(field)
        if value > 0 and (
            current is None
            or (isinstance(current, int) and not isinstance(current, bool) and current == 0)
        ):
            target[field] = value

    def _path(self, attempt: PreparedAttempt) -> str:
        protocol, path = CopilotProviderDialect._binding_spec(attempt.binding_id)
        if attempt.protocol is not protocol:
            raise ValueError("Copilot attempt protocol does not match its binding ID")
        if attempt.method != "POST":
            raise ValueError("Copilot generation bindings require POST")
        return path

    def _raise_status(
        self,
        error: httpx.HTTPStatusError,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        try:
            body = error.response.content
        except httpx.ResponseNotRead:
            body = b""
        self.provider._raise_unsupported_operation_if_applicable(
            error.response.status_code,
            body,
            model=attempt.model.upstream_id,
            cause=error,
        )
        self.provider._raise_http_status_error(
            "Copilot",
            error,
            logger,
            stream=stream,
            include_body=True,
            provider=self.provider.name,
            model=attempt.model.upstream_id,
            signal=self.provider._failure_signal(error.response.status_code, body),
        )

    def _raise_timeout(
        self,
        error: httpx.TimeoutException,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        self.provider._raise_timeout_error(
            "Copilot",
            error,
            logger,
            stream=stream,
            provider=self.provider.name,
            model=attempt.model.upstream_id,
        )

    def _raise_http_error(
        self,
        error: httpx.HTTPError,
        attempt: PreparedAttempt,
        *,
        stream: bool,
    ) -> NoReturn:
        self.provider._raise_http_error(
            "Copilot",
            error,
            logger,
            stream=stream,
            provider=self.provider.name,
            model=attempt.model.upstream_id,
        )

    def _raise_protocol_error(self, error: Exception, attempt: PreparedAttempt) -> NoReturn:
        self.provider._raise_protocol_error(
            self.provider.name,
            attempt.model.upstream_id,
            error,
        )
