"""Regression: Responses streaming must not crash on an early ProviderError.

If responses_completion_stream() raises before returning (auth failure, model
not found, expired token), the except handler previously referenced
response_id/created_at which were only assigned inside the try — causing an
UnboundLocalError that swallowed the stream and produced a silent empty
response. These hoisted variables must now be available.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from router_maestro.providers import ResponsesRequest as InternalResponsesRequest
from router_maestro.providers.base import ProviderError
from router_maestro.server.routes.responses import stream_response


class _FailingRouter:
    """Router stub whose stream open raises before yielding anything."""

    async def responses_completion_stream(
        self, request: InternalResponsesRequest, fallback: bool = True
    ) -> tuple[AsyncIterator, str]:
        raise ProviderError("model not found", status_code=404)


def _parse_sse(events: list[str]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for evt in events:
        for line in evt.splitlines():
            if line.startswith("data: "):
                parsed.append(json.loads(line[len("data: ") :]))
    return parsed


@pytest.mark.asyncio
async def test_early_provider_error_emits_failed_event():
    router = _FailingRouter()
    req = InternalResponsesRequest(model="gpt-5.5", input="hi", stream=True)

    raw_events: list[str] = []
    async for evt in stream_response(router, req, request_id="req-x", start_time=0.0):  # type: ignore[arg-type]
        raw_events.append(evt)

    events = _parse_sse(raw_events)
    # Must produce a clean response.failed event, not a silent empty stream.
    failed = [e for e in events if e.get("type") == "response.failed"]
    assert len(failed) == 1
    resp = failed[0]["response"]
    assert resp["status"] == "failed"
    assert resp["id"].startswith("resp")
    assert isinstance(resp["created_at"], int)
    assert "model not found" in resp["error"]["message"]
