"""Copilot off/none compatibility at translated and native wire boundaries."""

from copy import deepcopy

import pytest

from router_maestro.protocols import WireProtocol
from router_maestro.providers.base import ModelInfo, ResponsesRequest
from router_maestro.providers.copilot import (
    COPILOT_MODEL_ALIASES,
    CopilotProvider,
    apply_copilot_chat_reasoning,
    copilot_operation_capabilities,
    normalize_supported_endpoints,
)
from router_maestro.providers.copilot_support.catalog import (
    CopilotCatalog,
    effective_reasoning_effort_values,
)
from router_maestro.routing.model_ref import ModelRef

_RAW_EFFORTS = ["none", "low", "medium", "high", "xhigh", "max"]


def _provider(model: str, allowed: list[str] | None) -> CopilotProvider:
    provider = CopilotProvider()
    if allowed is not None:
        provider._models_ttl_cache.set(
            [
                ModelInfo(
                    id=model,
                    name=model,
                    provider=provider.name,
                    reasoning_effort_values=deepcopy(allowed),
                )
            ]
        )
    return provider


def test_catalog_corrects_verified_none_discrepancy_without_mutating_raw_data():
    raw = {
        "data": [
            {
                "id": model,
                "supported_endpoints": ["/responses", "/chat/completions"],
                "capabilities": {"supports": {"reasoning_effort": _RAW_EFFORTS}},
            }
            for model in ["gpt-6.1-sol", "gpt-5.6-luna", "future-model"]
        ]
    }
    original = deepcopy(raw)
    parsed = CopilotCatalog.parse_models(
        raw,
        provider_name="github-copilot",
        normalize_endpoints=normalize_supported_endpoints,
        derive_operations=copilot_operation_capabilities,
    )

    assert parsed[0].reasoning_effort_values == _RAW_EFFORTS[1:]
    assert parsed[1].reasoning_effort_values == _RAW_EFFORTS
    assert parsed[2].reasoning_effort_values == _RAW_EFFORTS
    assert raw == original


@pytest.mark.parametrize("disabled", ["off", "none"])
@pytest.mark.parametrize("allowed", [None, _RAW_EFFORTS])
@pytest.mark.parametrize("qualified", [False, True])
def test_translated_responses_known_required_model_uses_actual_floor(disabled, allowed, qualified):
    provider = _provider("gpt-6.1-sol", allowed)
    model = "github-copilot/gpt-6.1-sol" if qualified else "gpt-6.1-sol"
    payload = provider._build_responses_payload(
        ResponsesRequest(model=model, input="hi", reasoning_effort=disabled)
    )

    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}


@pytest.mark.parametrize("protocol", [WireProtocol.OPENAI_CHAT, WireProtocol.OPENAI_RESPONSES])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("disabled", ["off", "none"])
@pytest.mark.parametrize("allowed", [None, _RAW_EFFORTS])
@pytest.mark.asyncio
async def test_native_binding_clamps_once_and_keeps_original_payload(
    protocol, stream, disabled, allowed
):
    provider = _provider("gpt-6.1-sol", allowed)
    binding = next(binding for binding in provider.bindings() if binding.protocol is protocol)
    if protocol is WireProtocol.OPENAI_RESPONSES:
        payload = {
            "model": "gpt-6.1-sol",
            "input": "hi",
            "reasoning": {"effort": disabled, "summary": "detailed"},
            "include": ["reasoning.encrypted_content"],
        }
    else:
        payload = {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": disabled,
        }
    original = deepcopy(payload)
    attempt = await binding.prepare_attempt(
        model=ModelRef(provider.name, "gpt-6.1-sol"), payload=payload, stream=stream
    )

    if protocol is WireProtocol.OPENAI_RESPONSES:
        assert attempt.payload["reasoning"] == {"effort": "low", "summary": "detailed"}
        assert attempt.payload["include"] == payload["include"]
    else:
        assert attempt.payload["reasoning_effort"] == "low"
    assert payload == original


@pytest.mark.parametrize("disabled", ["off", "none"])
@pytest.mark.parametrize(
    "model,allowed,expected",
    [
        ("gpt-6.1-sol", None, "low"),
        ("gpt-6.1-sol", _RAW_EFFORTS, "low"),
        ("gpt-6.1-sol", ["none", "high", "medium"], "medium"),
        ("gpt-5.6-luna", ["none", "low", "medium"], "none"),
        ("future-model", ["none"], "none"),
        ("future-model", ["high", "minimal", "low"], "minimal"),
        ("claude-opus-4.7", ["high", "medium"], "medium"),
        ("gpt-4o", [], None),
        ("gpt-4o", None, None),
        ("future-model", None, None),
    ],
)
def test_chat_and_responses_disabled_semantics_match(disabled, model, allowed, expected):
    provider = _provider(model, allowed)
    responses = provider._build_responses_payload(
        ResponsesRequest(model=model, input="hi", reasoning_effort=disabled)
    )
    chat = {}
    apply_copilot_chat_reasoning(chat, model, None, disabled, catalog_effort_values=allowed)

    assert responses.get("reasoning", {}).get("effort") == expected
    assert chat.get("reasoning_effort") == expected


@pytest.mark.parametrize("disabled", ["off", "none"])
@pytest.mark.parametrize("qualified", [False, True])
def test_model_alias_uses_target_effort_catalog(disabled, qualified):
    provider = _provider("gpt-5.4-mini", ["medium", "high"])
    alias = "github-copilot/codex-auto-review" if qualified else "codex-auto-review"
    payload = provider._build_responses_payload(
        ResponsesRequest(model=alias, input="hi", reasoning_effort=disabled)
    )

    assert payload["reasoning"]["effort"] == "medium"


@pytest.mark.parametrize("allowed", [None, _RAW_EFFORTS])
def test_internal_alias_to_required_model_uses_target_compatibility_policy(monkeypatch, allowed):
    monkeypatch.setitem(COPILOT_MODEL_ALIASES, "test-required-alias", "gpt-6.1-sol")
    provider = _provider("gpt-6.1-sol", allowed)
    payload = provider._build_responses_payload(
        ResponsesRequest(model="test-required-alias", input="hi", reasoning_effort="off")
    )
    assert payload["reasoning"]["effort"] == "low"


def test_expired_catalog_uses_known_floor_without_fetching(monkeypatch):
    monkeypatch.setattr("router_maestro.utils.cache.time.monotonic", lambda: 1000.0)
    provider = _provider("gpt-6.1-sol", _RAW_EFFORTS)
    provider._models_ttl_cache._timestamp = 0.0
    assert not provider._models_ttl_cache.is_valid
    payload = provider._build_responses_payload(
        ResponsesRequest(model="gpt-6.1-sol", input="hi", reasoning_effort="off")
    )
    assert payload["reasoning"]["effort"] == "low"


@pytest.mark.parametrize("disabled", ["off", "none"])
def test_disabled_request_does_not_enable_unknown_cold_model_from_budget(disabled):
    chat = {}
    apply_copilot_chat_reasoning(chat, "future-model", 1024, disabled)
    assert "reasoning_effort" not in chat


def test_compatibility_correction_does_not_mutate_or_leak_catalog_lists():
    values = deepcopy(_RAW_EFFORTS)
    effective = effective_reasoning_effort_values("github-copilot/gpt-6.1-sol", values)
    assert effective == _RAW_EFFORTS[1:]
    assert values == _RAW_EFFORTS
    effective.append("future")
    assert effective_reasoning_effort_values("gpt-6.1-sol", None) == _RAW_EFFORTS[1:]


@pytest.mark.parametrize("values", [[], ["none"], ["none", "off"]])
def test_warm_explicit_no_reasoning_stays_empty_after_sentinel_correction(values):
    assert effective_reasoning_effort_values("gpt-6.1-sol", values) == []
    provider = _provider("gpt-6.1-sol", values)
    payload = provider._build_responses_payload(
        ResponsesRequest(model="gpt-6.1-sol", input="hi", reasoning_effort="none")
    )
    assert "reasoning" not in payload


def test_warm_catalog_preserves_non_disabled_future_tiers_and_order():
    assert effective_reasoning_effort_values(
        "gpt-6.1-sol", ["none", "future", "high", "minimal"]
    ) == ["future", "high", "minimal"]


@pytest.mark.parametrize("model", ["gpt-6.1-sol-1m", "gpt-6.1-sol-fast", "gpt-6-astra"])
def test_unverified_variant_is_not_automatically_in_the_discrepancy_table(model):
    assert effective_reasoning_effort_values(model, _RAW_EFFORTS) == _RAW_EFFORTS
    assert effective_reasoning_effort_values(model, None) is None


def test_unset_effort_is_still_omitted_with_known_required_floor():
    payload = CopilotProvider()._build_responses_payload(
        ResponsesRequest(model="gpt-6.1-sol", input="hi")
    )
    assert "reasoning" not in payload
