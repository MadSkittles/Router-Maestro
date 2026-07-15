"""Models route."""

import time

from fastapi import APIRouter, Depends

from router_maestro.routing.model_ref import catalog_model_public_id
from router_maestro.routing.router import Router
from router_maestro.server.dependencies import get_app_router
from router_maestro.server.schemas import ModelList, ModelObject

router = APIRouter()


@router.get("/api/openai/v1/models")
async def list_models(model_router: Router = Depends(get_app_router)) -> ModelList:
    """List available models."""
    models = await model_router.list_models()

    return ModelList(
        data=[
            ModelObject(
                id=catalog_model_public_id(
                    model.provider,
                    model.id,
                    id_is_qualified=model.id_is_qualified,
                ),
                created=int(time.time()),
                owned_by=model.provider,
                max_prompt_tokens=model.max_prompt_tokens,
                max_output_tokens=model.max_output_tokens,
                max_context_window_tokens=model.max_context_window_tokens,
                supports_thinking=model.supports_thinking or None,
                supports_vision=model.supports_vision or None,
            )
            for model in models
        ]
    )
