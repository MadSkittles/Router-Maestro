"""Guard: inference request schemas must never reject unknown client options.

Clients evolve faster than Router-Maestro models them. Any Pydantic model
reachable from an inference request body must tolerate unknown fields
(extra != "forbid"), so an unmodeled option is ignored, not 400'd. Admin/config
request roots are intentionally excluded — rejecting a typo'd config key there
is correct.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel

from router_maestro.server.schemas.anthropic import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from router_maestro.server.schemas.gemini import GeminiGenerateContentRequest
from router_maestro.server.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionStreamOptions,
    OpenAIThinkingConfig,
)
from router_maestro.server.schemas.responses import (
    ResponsesReasoningConfig,
    ResponsesRequest,
)

# Roots whose full nested field graph is validated by FastAPI on the request body.
INFERENCE_REQUEST_ROOTS = [
    ChatCompletionRequest,
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
    GeminiGenerateContentRequest,
    ResponsesRequest,
]

# Request sub-schemas the routes re-validate manually via ``model_validate`` on a
# field typed ``Any`` (so they are NOT reachable by walking annotations). These
# are exactly the surface that produced the F1 (thinking) and v0.6.2 (reasoning)
# regressions, so the guard must cover them explicitly.
ROUTE_REVALIDATED_REQUEST_SCHEMAS = [
    ChatCompletionStreamOptions,
    OpenAIThinkingConfig,
    ResponsesReasoningConfig,
]


def _iter_model_types(annotation: object) -> typing.Iterator[type[BaseModel]]:
    """Yield BaseModel subclasses nested anywhere in a type annotation."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    for arg in typing.get_args(annotation):
        yield from _iter_model_types(arg)


def _reachable_models(root: type[BaseModel]) -> set[type[BaseModel]]:
    """Collect every BaseModel reachable through root's field annotations."""
    seen: set[type[BaseModel]] = set()
    stack: list[type[BaseModel]] = [root]
    while stack:
        model = stack.pop()
        if model in seen:
            continue
        seen.add(model)
        for field in model.model_fields.values():
            for annotated in _iter_model_types(field.annotation):
                if annotated not in seen:
                    stack.append(annotated)
    return seen


def test_no_inference_request_schema_forbids_extra():
    roots = INFERENCE_REQUEST_ROOTS + ROUTE_REVALIDATED_REQUEST_SCHEMAS
    offenders = []
    for root in roots:
        for model in _reachable_models(root):
            if model.model_config.get("extra") == "forbid":
                offenders.append(f"{model.__module__}.{model.__qualname__}")
    assert not offenders, (
        "Inference request schemas must not use extra='forbid' "
        f"(clients send unmodeled options): {sorted(offenders)}"
    )


def test_guard_covers_the_route_revalidated_schemas():
    """Regression guard for the guard itself: the F1/v0.6.2 schemas are typed
    ``Any`` on their request roots, so annotation-walking alone misses them.
    This asserts they are explicitly covered so a future extra='forbid' on them
    fails CI."""
    assert OpenAIThinkingConfig in ROUTE_REVALIDATED_REQUEST_SCHEMAS
    assert ResponsesReasoningConfig in ROUTE_REVALIDATED_REQUEST_SCHEMAS
    # And prove they are NOT otherwise reachable (why the explicit list is needed).
    reachable = set()
    for root in INFERENCE_REQUEST_ROOTS:
        reachable |= _reachable_models(root)
    assert OpenAIThinkingConfig not in reachable
    assert ResponsesReasoningConfig not in reachable

