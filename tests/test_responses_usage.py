"""Tests for Responses API usage propagation."""

import pytest

from router_maestro.providers.base import ResponsesResponse as InternalResponsesResponse
from router_maestro.server.routes import responses as responses_route
from router_maestro.server.schemas import ResponsesUsage
from router_maestro.server.schemas.responses import ResponsesRequest


def test_responses_usage_schema_preserves_detail_fields():
    """ResponsesUsage should serialize upstream detail fields."""
    usage = ResponsesUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details={"cached_tokens": 60},
        output_tokens_details={"reasoning_tokens": 9},
    )

    assert usage.model_dump(exclude_none=True) == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "input_tokens_details": {"cached_tokens": 60},
        "output_tokens_details": {"reasoning_tokens": 9},
    }


class _StubRouter:
    async def rewrite_to_reasoning_variant(self, request):
        return request

    async def responses_completion(self, request):
        return (
            InternalResponsesResponse(
                content="hello",
                model=request.model,
                usage={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 60},
                    "output_tokens_details": {"reasoning_tokens": 9},
                },
            ),
            "github-copilot",
        )


@pytest.mark.asyncio
async def test_create_response_preserves_usage_detail_fields(monkeypatch):
    """Non-streaming Responses route should forward upstream detail fields."""
    monkeypatch.setattr(responses_route, "get_router", lambda: _StubRouter())

    response = await responses_route.create_response(
        ResponsesRequest(model="github-copilot/gpt-5.5", input="hi")
    )

    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20
    assert response.usage.total_tokens == 120
    assert response.usage.input_tokens_details == {"cached_tokens": 60}
    assert response.usage.output_tokens_details == {"reasoning_tokens": 9}
