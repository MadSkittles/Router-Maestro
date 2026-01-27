"""Models route."""

import time

from fastapi import APIRouter

from router_maestro.routing import Router
from router_maestro.server.schemas import ModelList, ModelObject

router = APIRouter()


def get_router() -> Router:
    """Get the router instance."""
    return Router()


@router.get("/models")
@router.get("/v1/models")
async def list_models() -> ModelList:
    """List available models."""
    model_router = get_router()
    models = await model_router.list_models()

    return ModelList(
        data=[
            ModelObject(
                id=model.id,
                created=int(time.time()),
                owned_by=model.provider,
            )
            for model in models
        ]
    )
