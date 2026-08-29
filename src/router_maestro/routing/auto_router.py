"""Bounded task classification for the virtual Router-Maestro Auto model."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from router_maestro.config import AutoCapabilityPolicy, AutoTaskType
from router_maestro.protocols import RequestEnvelope, SemanticMessage, SemanticResponse, TextContent
from router_maestro.routing.generation_plan import (
    GenerationCandidate,
    auto_candidate_for_id,
    select_auto_context_candidates,
)

_MAX_ROUTING_TEXT_CHARS = 12_000
_MAX_ROUTING_TEXT_ITEMS = 6

_TASK_DESCRIPTIONS = {
    AutoTaskType.FAST: "short answers, extraction, formatting, classification, or a tiny edit",
    AutoTaskType.GENERAL: "ordinary conversation or work that does not fit another category",
    AutoTaskType.CODING: "implementation, debugging, code review, repositories, or tool-heavy work",
    AutoTaskType.DEEP_REASONING: (
        "architecture, complex planning, research synthesis, or hard analysis"
    ),
}


@dataclass(frozen=True, slots=True)
class AutoTaskCandidate:
    """One task label and its capability-compatible configured execution model."""

    task: AutoTaskType
    candidate: GenerationCandidate


def eligible_auto_tasks(router: Any, envelope: RequestEnvelope) -> tuple[AutoTaskCandidate, ...]:
    """Resolve task mappings that can satisfy the request's hard requirements."""
    config = router._get_priorities_config().auto
    strict_unknown = config.capability_policy is AutoCapabilityPolicy.STRICT
    strict_manifest = envelope.manifest
    if strict_unknown and (strict_manifest.files or strict_manifest.structured_output):
        feature_keys = {
            key.value if hasattr(key, "value") else str(key)
            for model_id in config.task_router.task_models.values()
            if (entry := router._models_cache.get(model_id)) is not None
            for key in entry[1].feature_capabilities
        }
        # Current provider catalogs do not yet advertise these two portable
        # capabilities. Keep adapter representability authoritative until at
        # least one configured model publishes an explicit verdict; once the
        # field exists, strict unknown filtering applies to every target.
        from dataclasses import replace

        strict_manifest = replace(
            strict_manifest,
            files=strict_manifest.files and "files" in feature_keys,
            structured_output=(
                strict_manifest.structured_output and "structured_output" in feature_keys
            ),
        )
    hard_tasks: list[AutoTaskCandidate] = []
    for task, model_id in config.task_router.task_models.items():
        candidate = auto_candidate_for_id(
            router,
            model_id,
            manifest=strict_manifest,
            strict_unknown=strict_unknown,
        )
        if candidate is not None:
            hard_tasks.append(AutoTaskCandidate(task=task, candidate=candidate))

    selected_models = {
        candidate.model
        for candidate in select_auto_context_candidates(
            tuple(task.candidate for task in hard_tasks),
            envelope.manifest.estimated_input_tokens,
        )
    }
    return tuple(task for task in hard_tasks if task.candidate.model in selected_models)


def build_classifier_payload(
    envelope: RequestEnvelope,
    *,
    router_model: str,
    tasks: Sequence[AutoTaskCandidate],
    disable_reasoning: bool = False,
) -> dict[str, Any]:
    """Build a bounded, schema-constrained, tool-free classifier request."""
    allowed = tuple(dict.fromkeys(task.task for task in tasks))
    categories = "\n".join(f"- {task.value}: {_TASK_DESCRIPTIONS[task]}" for task in allowed)
    manifest = envelope.manifest
    projection = {
        "protocol": envelope.protocol.value,
        "requirements": {
            "tools": manifest.tools,
            "parallel_tools": manifest.parallel_tools,
            "images": manifest.images,
            "files": manifest.files,
            "reasoning": manifest.reasoning,
            "structured_output": manifest.structured_output,
            "stream": manifest.stream,
        },
        "recent_user_text": _recent_user_text(envelope),
    }
    system = (
        "You classify requests for Router-Maestro. Select the best task type only from the "
        "allowed list below. Capabilities have already been filtered. Return exactly one compact "
        'JSON object such as {"task_type":"coding"}; do not add markdown or explanation.\n\n'
        f"Allowed task types:\n{categories}"
    )
    payload = {
        "model": router_model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "max_tokens": 32,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "router_maestro_task_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": [task.value for task in allowed],
                        }
                    },
                    "required": ["task_type"],
                    "additionalProperties": False,
                },
            },
        },
    }
    if disable_reasoning:
        # ``none`` is an upstream catalog sentinel, not a public RM effort.
        # It is emitted only for an internal request when the selected model
        # explicitly advertises it, keeping the 32-token classifier bounded.
        payload["reasoning_effort"] = "none"
    return payload


def parse_classifier_result(value: Any, allowed: Sequence[AutoTaskType]) -> AutoTaskType:
    """Parse a provider result while keeping the router model on a fixed enum."""
    text = _classifier_text(value).strip()
    candidate: object = None
    try:
        decoded = json.loads(text)
        if isinstance(decoded, Mapping):
            candidate = decoded.get("task_type")
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"task_type"\s*:\s*"([a-z_\-]+)"[^{}]*\}', text)
        candidate = match.group(1) if match is not None else text.strip('` \n\t"')
    try:
        task = AutoTaskType(str(candidate))
    except ValueError as error:
        raise ValueError("router model returned an invalid task type") from error
    if task not in allowed:
        raise ValueError("router model selected a capability-ineligible task type")
    return task


def _classifier_text(value: Any) -> str:
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(value, Mapping):
        choices = value.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content
    if isinstance(value, SemanticResponse):
        fragments: list[str] = []
        for item in value.output:
            content = item.content if isinstance(item, SemanticMessage) else (item,)
            fragments.extend(part.text for part in content if isinstance(part, TextContent))
        if fragments:
            return "".join(fragments)
    raise ValueError("router model response did not contain text")


def _recent_user_text(envelope: RequestEnvelope) -> str:
    payload = envelope.native_payload()
    protocol = envelope.protocol.value
    values: list[str] = []
    if protocol == "openai_chat":
        _collect_role_text(payload.get("messages"), values, user_roles={"user"})
    elif protocol == "openai_responses":
        raw = payload.get("input")
        if isinstance(raw, str):
            values.append(raw)
        else:
            _collect_role_text(raw, values, user_roles={"user"})
    elif protocol == "anthropic_messages":
        _collect_role_text(payload.get("messages"), values, user_roles={"user"})
    else:
        _collect_role_text(payload.get("contents"), values, user_roles={"user"})
    selected = values[-_MAX_ROUTING_TEXT_ITEMS:]
    text = "\n\n".join(selected)
    return text[-_MAX_ROUTING_TEXT_CHARS:]


def _collect_role_text(value: object, output: list[str], *, user_roles: set[str]) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        if role is None and "type" in item:
            role = "user"
        if role not in user_roles:
            continue
        _collect_text(item.get("content", item.get("parts")), output)


def _collect_text(value: object, output: list[str]) -> None:
    if isinstance(value, str):
        output.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text(item, output)
        return
    if not isinstance(value, Mapping):
        return
    block_type = value.get("type")
    text = value.get("text")
    if isinstance(text, str) and block_type not in {"thinking", "reasoning"}:
        output.append(text)
    elif isinstance(value.get("input_text"), str):
        output.append(value["input_text"])


__all__ = [
    "AutoTaskCandidate",
    "build_classifier_payload",
    "eligible_auto_tasks",
    "parse_classifier_result",
]
