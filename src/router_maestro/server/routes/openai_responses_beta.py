"""Retired OpenAI Responses beta URL kept as a stable-route alias."""

from fastapi import APIRouter
from fastapi import Request as FastAPIRequest

from router_maestro.server.routes.responses import (
    responses_endpoint as standard_responses_endpoint,
)
from router_maestro.server.schemas import ResponsesRequest

router = APIRouter()


@router.post("/api/openai/beta/v1/responses")
async def beta_responses(request: ResponsesRequest, raw_request: FastAPIRequest):
    """Delegate the retired beta URL to the stable Responses handler."""
    return await standard_responses_endpoint(request=request, raw_request=raw_request)
