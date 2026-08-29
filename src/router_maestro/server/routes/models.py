"""Models route."""

import time

from fastapi import APIRouter, Depends

from router_maestro.routing.generation_plan import list_models_with_auto
from router_maestro.routing.model_ref import catalog_model_public_id
from router_maestro.routing.router import Router
from router_maestro.server.dependencies import get_app_router
from router_maestro.server.schemas import ContextWindowOption, ModelList, ModelObject

router = APIRouter()


@router.get("/api/openai/v1/models")
async def list_models(model_router: Router = Depends(get_app_router)) -> ModelList:
    """List available models."""
    models = await list_models_with_auto(model_router)

    return ModelList(
        data=[
            ModelObject(
                id=catalog_model_public_id(
                    model.provider,
                    model.id,
                    id_is_qualified=model.id_is_qualified,
                    is_virtual=model.virtual,
                ),
                created=int(time.time()),
                owned_by=model.provider,
                max_prompt_tokens=model.max_prompt_tokens,
                max_output_tokens=model.max_output_tokens,
                max_context_window_tokens=model.max_context_window_tokens,
                context_window_options=[
                    ContextWindowOption(
                        tier=option.tier,
                        max_prompt_tokens=option.max_prompt_tokens,
                        is_default=option.is_default,
                    )
                    for option in model.effective_context_window_options()
                ],
                supports_thinking=model.supports_thinking or None,
                supports_vision=model.supports_vision or None,
                operation_capabilities=dict(model.operation_capabilities),
                feature_capabilities=dict(model.feature_capabilities),
                transport_capabilities=dict(model.transport_capabilities),
                reasoning_effort_values=model.reasoning_effort_values,
                virtual=model.virtual,
            )
            for model in models
        ]
    )
