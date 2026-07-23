"""Tests for the OpenAI Responses beta passthrough route."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.providers.copilot import (
    _COPILOT_RESPONSES_FORWARD_FIELDS,
    CopilotOutboundContract,
    CopilotProvider,
)
from router_maestro.routing.capabilities import (
    CapabilitySupport,
    ModelCapabilities,
    Operation,
    RequestFeatures,
)
from router_maestro.routing.model_ref import ModelRef
from router_maestro.routing.route_plan import RouteCandidate, RoutePlan
from router_maestro.server.routes.openai_responses_beta import (
    _candidate_effort_values,
    _frame_terminal_outcome,
    _guard_chunk,
    _iter_sse_frames,
    _operation_for,
    _ResponsesResolution,
    router,
)

# ---------------------------------------------------------------------------
# Unit tests: forward allowlist / feature classifier / helpers
# ---------------------------------------------------------------------------


def test_forwardable_fields_returns_responses_allowlist() -> None:
    contract = CopilotOutboundContract()
    assert contract.forwardable_fields(Operation.RESPONSES) is _COPILOT_RESPONSES_FORWARD_FIELDS
    assert (
        contract.forwardable_fields(Operation.RESPONSES_STREAM) is _COPILOT_RESPONSES_FORWARD_FIELDS
    )
    # Chat still forwards everything; only passthrough ops have an allowlist.
    assert contract.forwardable_fields(Operation.CHAT) is None


def test_responses_allowlist_excludes_store_includes_fidelity_fields() -> None:
    assert "store" not in _COPILOT_RESPONSES_FORWARD_FIELDS
    for field in ("include", "previous_response_id", "prompt_cache_key", "text", "reasoning"):
        assert field in _COPILOT_RESPONSES_FORWARD_FIELDS


def test_for_responses_native_classifies_features() -> None:
    features = RequestFeatures.for_responses_native(
        {
            "tools": [{"type": "function", "name": "x"}],
            "reasoning": {"effort": "high"},
            "input": "hello",
        }
    )
    assert features.tools is True
    assert features.reasoning is True
    assert features.reasoning_parameter == "reasoning.effort"


def test_for_responses_native_no_reasoning_when_effort_absent() -> None:
    features = RequestFeatures.for_responses_native({"input": "hi"})
    assert features.reasoning is False
    assert features.reasoning_parameter is None
    assert features.tools is False


def test_for_responses_native_detects_vision_in_input() -> None:
    features = RequestFeatures.for_responses_native(
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "http://x/y.png"}],
                }
            ]
        }
    )
    assert features.vision is True


def test_operation_for_stream_flag() -> None:
    assert _operation_for(True) is Operation.RESPONSES_STREAM
    assert _operation_for(False) is Operation.RESPONSES


def test_candidate_effort_values_reads_capabilities() -> None:
    provider = MagicMock(spec=CopilotProvider)
    provider.name = "github-copilot"
    plan = _responses_plan_with_efforts(provider, efforts=("low", "high"))
    assert _candidate_effort_values(plan.primary) == ["low", "high"]
    # None catalog (cold) yields None, not an empty list.
    bare = _responses_plan_with_efforts(provider, efforts=None)
    assert _candidate_effort_values(bare.primary) is None


def test_reconcile_passthrough_body_strips_and_reconciles() -> None:
    contract = CopilotOutboundContract()
    body = {
        "model": "gpt-5.2",
        "input": "hi",
        "store": True,  # rejected by GHC -> stripped
        "mcp_servers": [],  # not in the allowlist -> stripped
        "reasoning": {"effort": "high"},
    }
    contract.reconcile_passthrough_body(
        body,
        operation=Operation.RESPONSES,
        model="gpt-5.2",
        catalog_effort_values=["low", "medium", "high"],
    )
    assert "store" not in body
    assert "mcp_servers" not in body
    assert body["model"] == "gpt-5.2"
    assert body["reasoning"] == {"effort": "high"}


def test_reconcile_passthrough_body_drops_unsupported_tool() -> None:
    contract = CopilotOutboundContract()
    body = {
        "model": "gpt-5.5",
        "input": "hi",
        "tools": [{"type": "function", "name": "echo"}, {"type": "web_search"}],
    }
    contract.reconcile_passthrough_body(
        body,
        operation=Operation.RESPONSES,
        model="gpt-5.5",
        catalog_effort_values=None,
    )
    assert body["tools"] == [{"type": "function", "name": "echo"}]


def test_reconcile_passthrough_body_rejects_temperature() -> None:
    from router_maestro.providers import RequestOptionError

    contract = CopilotOutboundContract()
    body = {"model": "gpt-5.5", "input": "hi", "temperature": 0.5}
    with pytest.raises(RequestOptionError) as excinfo:
        contract.reconcile_passthrough_body(
            body,
            operation=Operation.RESPONSES,
            model="gpt-5.5",
            catalog_effort_values=None,
        )
    assert excinfo.value.parameter == "temperature"


def test_reconcile_passthrough_body_downgrades_reasoning_effort() -> None:
    contract = CopilotOutboundContract()
    body = {
        "model": "gpt-5.5",
        "input": "hi",
        "reasoning": {"effort": "xhigh", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
    }
    contract.reconcile_passthrough_body(
        body,
        operation=Operation.RESPONSES,
        model="gpt-5.5",
        catalog_effort_values=["low", "medium", "high"],
    )
    # Effort downgraded to the nearest supported tier; sibling keys + client
    # include preserved (no summary/include injection on passthrough).
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["include"] == ["reasoning.encrypted_content"]


# ---------------------------------------------------------------------------
# Unit tests: guard projection + terminal-outcome mapping
# ---------------------------------------------------------------------------


def test_guard_chunk_projects_output_text_delta() -> None:
    chunk = _guard_chunk(
        "response.output_text.delta",
        {"delta": "hello"},
        "wire-frame",
    )
    assert chunk.content == "hello"
    assert chunk.opaque_payload == "wire-frame"


def test_guard_chunk_projects_reasoning_delta_as_thinking() -> None:
    chunk = _guard_chunk(
        "response.reasoning_summary_text.delta",
        {"delta": "thinking..."},
        "wire",
    )
    assert chunk.content == ""
    assert chunk.thinking == "thinking..."


def test_guard_chunk_projects_function_call_arguments() -> None:
    chunk = _guard_chunk(
        "response.function_call_arguments.delta",
        {"delta": '{"a":1}'},
        "wire",
    )
    assert chunk.tool_calls == [{"function": {"arguments": '{"a":1}'}}]


def test_guard_chunk_unknown_event_carries_only_wire_payload() -> None:
    chunk = _guard_chunk("response.created", {"type": "response.created"}, "wire")
    assert chunk.content == ""
    assert chunk.thinking is None
    assert chunk.tool_calls is None
    assert chunk.opaque_payload == "wire"


def test_frame_terminal_outcome_maps_completed() -> None:
    outcome = _frame_terminal_outcome(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        },
    )
    assert outcome is not None
    assert outcome.response_status.value == "completed"


def test_frame_terminal_outcome_none_for_non_terminal() -> None:
    assert _frame_terminal_outcome("response.output_text.delta", {"delta": "x"}) is None


def test_frame_terminal_outcome_maps_response_done() -> None:
    # Some upstream gateways emit ``response.done`` instead of
    # ``response.completed`` as the terminal event.
    outcome = _frame_terminal_outcome(
        "response.done",
        {
            "type": "response.done",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        },
    )
    assert outcome is not None
    assert outcome.response_status.value == "completed"


@pytest.mark.asyncio
async def test_iter_sse_frames_treats_response_done_as_terminal() -> None:
    class _Response:
        async def aiter_lines(self):
            yield "event: response.output_text.delta"
            yield 'data: {"delta":"hi"}'
            yield ""
            yield "event: response.done"
            yield 'data: {"type":"response.done","response":{"status":"completed"}}'
            yield ""

    # A ``response.done`` terminal must not raise unexpected-EOF.
    frames = [frame async for frame in _iter_sse_frames(_Response())]
    events = [
        line.removeprefix("event: ")
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("event: ")
    ]
    assert events == ["response.output_text.delta", "response.done"]


@pytest.mark.asyncio
async def test_iter_sse_frames_filters_copilot_internal_events() -> None:
    class _Response:
        async def aiter_lines(self):
            yield "event: copilot_usage"
            yield 'data: {"internal":true}'
            yield ""
            yield "event: response.output_text.delta"
            yield 'data: {"delta":"hi"}'
            yield ""
            yield "event: response.completed"
            yield 'data: {"type":"response.completed","response":{"status":"completed"}}'
            yield ""

    frames = [frame async for frame in _iter_sse_frames(_Response())]
    events = [
        line.removeprefix("event: ")
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("event: ")
    ]
    assert "copilot_usage" not in events
    assert events == ["response.output_text.delta", "response.completed"]


@pytest.mark.asyncio
async def test_iter_sse_frames_raises_on_missing_terminal() -> None:
    from router_maestro.server.routes.openai_responses_beta import _ResponsesUnexpectedEOFError

    class _Response:
        async def aiter_lines(self):
            yield "event: response.output_text.delta"
            yield 'data: {"delta":"hi"}'
            yield ""

    with pytest.raises(_ResponsesUnexpectedEOFError):
        _ = [frame async for frame in _iter_sse_frames(_Response())]


# ---------------------------------------------------------------------------
# Integration tests with TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def non_raising_client(app):
    return TestClient(app, raise_server_exceptions=False)


def _native_provider() -> MagicMock:
    provider = MagicMock(spec=CopilotProvider)
    provider.name = "github-copilot"
    provider.ensure_token = AsyncMock()
    provider.outbound_contract = CopilotOutboundContract()
    provider._responses_input_has_vision = MagicMock(return_value=False)

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "resp_123",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.2",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    }
    provider._send_with_auth_retry = AsyncMock(return_value=response)
    return provider


def _responses_plan(provider, support, *, model="gpt-5.2") -> RoutePlan:
    ref = ModelRef("github-copilot", model)
    operation = Operation.RESPONSES
    features = RequestFeatures()
    capabilities = ModelCapabilities(model=ref, operations={operation: support})
    return RoutePlan(
        operation=operation,
        features=features,
        primary=RouteCandidate(
            model=ref,
            provider=provider,
            capabilities=capabilities,
            evaluated_operation=operation,
            evaluated_features=features,
            support=support,
        ),
        fallbacks=(),
        explicit=True,
    )


def test_nonstream_passthrough_forwards_raw_and_rewrites_model(client) -> None:
    provider = _native_provider()
    plan = _responses_plan(provider, CapabilitySupport.SUPPORTED)
    resolution = _ResponsesResolution(
        provider_name="github-copilot",
        actual_model="gpt-5.2",
        copilot_provider=provider,
        support=CapabilitySupport.SUPPORTED,
        plan=plan,
    )
    with patch(
        "router_maestro.server.routes.openai_responses_beta._resolve_responses_model",
        new_callable=AsyncMock,
        return_value=resolution,
    ):
        downstream = client.post(
            "/api/openai/beta/v1/responses",
            json={
                "model": "github-copilot/gpt-5.2",
                "input": "hi",
                "store": True,
                "previous_response_id": "resp_prev",
            },
        )
    assert downstream.status_code == 200
    body = downstream.json()
    # Public model id is the qualified selection.
    assert body["model"] == "github-copilot/gpt-5.2"

    # The raw body was forwarded to GHC with store stripped and the fidelity
    # field preserved, model rewritten to the upstream id.
    sent_payload = provider._send_with_auth_retry.await_args.kwargs["json"]
    assert sent_payload["model"] == "gpt-5.2"
    assert "store" not in sent_payload
    assert sent_payload["previous_response_id"] == "resp_prev"
    assert sent_payload["input"] == "hi"


@pytest.mark.parametrize("bad_effort", ["bogus", "none"])
def test_beta_responses_rejects_invalid_reasoning_effort(non_raising_client, bad_effort) -> None:
    """Beta native must reject an out-of-spec effort with a native 400 (matching
    the standard route) before any planning or upstream dispatch, instead of
    clamp-and-forwarding it."""
    send_mock = AsyncMock()
    with patch(
        "router_maestro.server.routes.openai_responses_beta._resolve_responses_model",
        new_callable=AsyncMock,
    ) as resolve_mock:
        resolve_mock.side_effect = AssertionError("must 400 before resolving/dispatch")
        downstream = non_raising_client.post(
            "/api/openai/beta/v1/responses",
            json={
                "model": "github-copilot/gpt-5.5",
                "input": "hi",
                "reasoning": {"effort": bad_effort},
            },
        )
    assert downstream.status_code == 400
    body = downstream.json()
    assert body["error"]["param"] == "reasoning_effort"
    send_mock.assert_not_awaited()


def _responses_plan_with_efforts(provider, *, model="gpt-5.5", efforts=None) -> RoutePlan:
    ref = ModelRef("github-copilot", model)
    operation = Operation.RESPONSES
    features = RequestFeatures()
    capabilities = ModelCapabilities(
        model=ref,
        operations={operation: CapabilitySupport.SUPPORTED},
        reasoning_effort_values=tuple(efforts) if efforts is not None else None,
    )
    return RoutePlan(
        operation=operation,
        features=features,
        primary=RouteCandidate(
            model=ref,
            provider=provider,
            capabilities=capabilities,
            evaluated_operation=operation,
            evaluated_features=features,
            support=CapabilitySupport.SUPPORTED,
        ),
        fallbacks=(),
        explicit=True,
    )


def _native_resolution(provider, plan, *, model="gpt-5.5") -> _ResponsesResolution:
    return _ResponsesResolution(
        provider_name="github-copilot",
        actual_model=model,
        copilot_provider=provider,
        support=CapabilitySupport.SUPPORTED,
        plan=plan,
    )


def test_nonstream_passthrough_drops_unsupported_tool(client) -> None:
    provider = _native_provider()
    plan = _responses_plan_with_efforts(provider)
    resolution = _native_resolution(provider, plan)
    with patch(
        "router_maestro.server.routes.openai_responses_beta._resolve_responses_model",
        new_callable=AsyncMock,
        return_value=resolution,
    ):
        downstream = client.post(
            "/api/openai/beta/v1/responses",
            json={
                "model": "github-copilot/gpt-5.5",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "echo"},
                    {"type": "web_search"},
                ],
            },
        )
    assert downstream.status_code == 200
    sent_payload = provider._send_with_auth_retry.await_args.kwargs["json"]
    assert sent_payload["tools"] == [{"type": "function", "name": "echo"}]


def test_nonstream_passthrough_rejects_temperature(client) -> None:
    provider = _native_provider()
    plan = _responses_plan_with_efforts(provider)
    resolution = _native_resolution(provider, plan)
    with patch(
        "router_maestro.server.routes.openai_responses_beta._resolve_responses_model",
        new_callable=AsyncMock,
        return_value=resolution,
    ):
        downstream = client.post(
            "/api/openai/beta/v1/responses",
            json={
                "model": "github-copilot/gpt-5.5",
                "input": "hi",
                "temperature": 0.5,
            },
        )
    assert downstream.status_code == 400
    assert downstream.json()["error"]["type"] == "invalid_request_error"
    # Upstream was never called — rejected locally before transport.
    provider._send_with_auth_retry.assert_not_awaited()


def test_nonstream_passthrough_downgrades_reasoning_effort(client) -> None:
    provider = _native_provider()
    plan = _responses_plan_with_efforts(provider, efforts=("low", "medium", "high"))
    resolution = _native_resolution(provider, plan)
    with patch(
        "router_maestro.server.routes.openai_responses_beta._resolve_responses_model",
        new_callable=AsyncMock,
        return_value=resolution,
    ):
        downstream = client.post(
            "/api/openai/beta/v1/responses",
            json={
                "model": "github-copilot/gpt-5.5",
                "input": "hi",
                "reasoning": {"effort": "xhigh", "summary": "auto"},
            },
        )
    assert downstream.status_code == 200
    sent_payload = provider._send_with_auth_retry.await_args.kwargs["json"]
    assert sent_payload["reasoning"]["effort"] == "high"
    # Sibling reasoning keys preserved (no injection on passthrough).
    assert sent_payload["reasoning"]["summary"] == "auto"


def test_nonstream_falls_back_to_standard_for_unsupported(client) -> None:
    resolution = _ResponsesResolution(
        provider_name="github-copilot",
        actual_model="claude-opus-4.6",
        copilot_provider=MagicMock(spec=CopilotProvider),
        support=CapabilitySupport.UNSUPPORTED,
        plan=None,
    )
    sentinel = MagicMock()
    with (
        patch(
            "router_maestro.server.routes.openai_responses_beta._resolve_responses_model",
            new_callable=AsyncMock,
            return_value=resolution,
        ),
        patch(
            "router_maestro.server.routes.openai_responses_beta.create_response",
            new_callable=AsyncMock,
            return_value=sentinel,
        ) as create_mock,
    ):
        client.post(
            "/api/openai/beta/v1/responses",
            json={"model": "github-copilot/claude-opus-4.6", "input": "hi"},
        )
    # The standard translated handler was invoked with a parsed ResponsesRequest.
    create_mock.assert_awaited_once()
    request_arg = create_mock.await_args.args[0]
    assert request_arg.model == "github-copilot/claude-opus-4.6"


def test_non_copilot_provider_falls_back(client) -> None:
    resolution = _ResponsesResolution(
        provider_name="openai",
        actual_model="gpt-5.2",
        copilot_provider=None,
        support=CapabilitySupport.SUPPORTED,
        plan=None,
    )
    with (
        patch(
            "router_maestro.server.routes.openai_responses_beta._resolve_responses_model",
            new_callable=AsyncMock,
            return_value=resolution,
        ),
        patch(
            "router_maestro.server.routes.openai_responses_beta.create_response",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as create_mock,
    ):
        client.post(
            "/api/openai/beta/v1/responses",
            json={"model": "openai/gpt-5.2", "input": "hi"},
        )
    create_mock.assert_awaited_once()


def test_invalid_json_returns_openai_error(non_raising_client) -> None:
    downstream = non_raising_client.post(
        "/api/openai/beta/v1/responses",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert downstream.status_code == 400
    assert downstream.json()["error"]["type"] == "invalid_request_error"


def test_missing_model_returns_openai_error(non_raising_client) -> None:
    downstream = non_raising_client.post(
        "/api/openai/beta/v1/responses",
        json={"input": "hi"},
    )
    assert downstream.status_code == 400
    assert downstream.json()["error"]["type"] == "invalid_request_error"


def test_stream_passthrough_emits_raw_frames(client) -> None:
    provider = _native_provider()
    plan = _responses_plan(provider, CapabilitySupport.SUPPORTED)
    resolution = _ResponsesResolution(
        provider_name="github-copilot",
        actual_model="gpt-5.2",
        copilot_provider=provider,
        support=CapabilitySupport.SUPPORTED,
        plan=plan,
    )

    async def selected_stream():
        yield 'event: response.output_text.delta\ndata: {"delta":"hi"}\n\n'
        yield (
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        )

    with (
        patch(
            "router_maestro.server.routes.openai_responses_beta._resolve_responses_model",
            new_callable=AsyncMock,
            return_value=resolution,
        ),
        patch(
            "router_maestro.server.routes.openai_responses_beta.Router.execute_plan_stream",
            new_callable=AsyncMock,
            return_value=(selected_stream(), "github-copilot"),
        ),
    ):
        with client.stream(
            "POST",
            "/api/openai/beta/v1/responses",
            json={"model": "github-copilot/gpt-5.2", "input": "hi", "stream": True},
        ) as downstream:
            text = "".join(chunk for chunk in downstream.iter_text())

    assert "response.output_text.delta" in text
    assert "response.completed" in text
