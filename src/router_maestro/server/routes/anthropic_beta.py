"""Retired Anthropic beta URLs kept as stable-route aliases."""

from fastapi import APIRouter
from fastapi import Request as FastAPIRequest

from router_maestro.server.routes.anthropic import (
    count_tokens as standard_count_tokens,
)
from router_maestro.server.routes.anthropic import (
    messages as standard_messages,
)
from router_maestro.server.schemas.anthropic import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)

router = APIRouter()


@router.post("/api/anthropic/beta/v1/messages")
async def beta_messages(request: AnthropicMessagesRequest, raw_request: FastAPIRequest):
    """Delegate the retired beta URL to the stable Messages handler."""
    return await standard_messages(request=request, raw_request=raw_request)


@router.post("/api/anthropic/beta/v1/messages/count_tokens")
async def beta_count_tokens(request: AnthropicCountTokensRequest):
    """Delegate the retired beta URL to the stable token-count handler."""
    return await standard_count_tokens(request)
