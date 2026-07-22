"""Tests for per-model reasoning dispatch on Copilot's chat endpoint."""

import pytest

from router_maestro.providers.copilot import apply_copilot_chat_reasoning


def _base_payload(extra: dict | None = None) -> dict:
    payload = {"model": "x", "messages": [], "max_tokens": 100}
    if extra:
        payload.update(extra)
    return payload


def test_claude_47_cold_start_uses_requested_effort():
    """opus-4.7 cold-start (no catalog yet): pass through low/medium/high as-is.

    Earlier the gateway clamped to ``medium``, but Copilot now advertises
    low/medium/high/xhigh/max for opus-4.7 — we no longer artificially clamp.
    """
    for input_effort, expected in (("low", "low"), ("medium", "medium"), ("high", "high")):
        for budget in (1024, 16000, None):
            p = _base_payload()
            apply_copilot_chat_reasoning(p, "claude-opus-4.7", budget, input_effort)
            assert p.get("reasoning_effort") == expected, (input_effort, budget)
            assert "thinking_budget" not in p


def test_claude_47_cold_start_clamps_xhigh_and_max_to_high():
    """Without the catalog we don't know xhigh/max are accepted — downgrade."""
    for input_effort in ("xhigh", "max"):
        p = _base_payload()
        apply_copilot_chat_reasoning(p, "claude-opus-4.7", None, input_effort)
        assert p.get("reasoning_effort") == "high", input_effort


def test_claude_48_cold_start_effort_wins_over_budget():
    """Opus 4.8 must retain explicit effort before the catalog is warm."""
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-opus-4.8", 1024, "xhigh")
    assert p.get("reasoning_effort") == "high"
    assert "thinking_budget" not in p


def test_claude_46_uses_reasoning_effort_not_thinking_budget():
    """opus-4.6 / sonnet-4.6 on Copilot expose effort, not budget."""
    for model in ("claude-opus-4.6", "claude-sonnet-4.6"):
        p = _base_payload()
        apply_copilot_chat_reasoning(p, model, 16000, None)
        assert "thinking_budget" not in p, model
        assert p.get("reasoning_effort") == "high", model


def test_claude_46_explicit_effort_wins_and_xhigh_clamped():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-sonnet-4.6", 1024, "xhigh")
    assert p.get("reasoning_effort") == "high"
    assert "thinking_budget" not in p


def test_claude_46_tiny_budget_clamps_up_to_lowest_tier():
    """A tiny budget below every tier clamps UP to the model floor, not a 400."""
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-opus-4.6", 100, None)
    # cold catalog, opus-4.6 supported: budget too small to derive an effort,
    # but a client that opted into thinking gets the lowest usable tier.
    assert p.get("reasoning_effort") == "low"
    assert "thinking_budget" not in p


def test_claude_old_models_strip_explicit_reasoning():
    """Models with no reasoning surface silently drop reasoning, no 400."""
    for model in (
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-haiku-4.5",
    ):
        p = _base_payload()
        apply_copilot_chat_reasoning(p, model, 16000, "high")
        assert "reasoning_effort" not in p, model
        assert "thinking_budget" not in p, model


def test_claude_47_dated_alias_still_supports_reasoning():
    """A future dated alias like claude-opus-4.7-20260101 must keep reasoning
    support — it is *not* one of the tier-encoded variants."""
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-opus-4.7-20260101", 16000, "high")
    assert p.get("reasoning_effort") == "high"


def test_claude_with_provider_prefix():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "github-copilot/claude-sonnet-4.6", 8192, "high")
    assert p.get("reasoning_effort") == "high"
    assert "thinking_budget" not in p


def test_gpt5_uses_reasoning_effort_not_thinking_budget():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-5.2", 16384, "high")
    assert p.get("reasoning_effort") == "high"
    assert "thinking_budget" not in p


def test_gpt5_derives_effort_from_budget_when_effort_missing():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-5.2", 16384, None)
    assert p.get("reasoning_effort") == "xhigh"
    assert "thinking_budget" not in p


def test_gpt5_preserves_xhigh():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-5.4", 24000, "xhigh")
    assert p.get("reasoning_effort") == "xhigh"


def test_gpt5_4_rewrites_max_tokens():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-5.4", None, "low")
    assert "max_tokens" not in p
    assert p.get("max_completion_tokens") == 100


def test_gpt5_2_keeps_max_tokens():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-5.2", None, "low")
    assert p.get("max_tokens") == 100
    assert "max_completion_tokens" not in p


def test_gpt4o_strips_explicit_reasoning():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-4o", 8192, "medium")
    assert "reasoning_effort" not in p


def test_gemini_strips_explicit_reasoning():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gemini-2.5-pro", 8192, "medium")
    assert "reasoning_effort" not in p


@pytest.mark.parametrize(
    "model",
    ["gemini-3.2-pro", "claude-sonnet-5", "mai-code-1-flash-picker"],
)
def test_new_reasoning_families_remain_observable_when_catalog_is_cold(model):
    payload = {}

    apply_copilot_chat_reasoning(payload, model, None, "low")

    assert payload["reasoning_effort"] == "low"


def test_no_reasoning_inputs_emits_nothing():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-5.2", None, None)
    assert "thinking_budget" not in p
    assert "reasoning_effort" not in p


def test_o_series_treated_as_reasoning():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "o3-mini", None, "medium")
    assert p.get("reasoning_effort") == "medium"
    assert "thinking_budget" not in p


# --- Catalog-driven path: trust whatever the model's
# capabilities.supports.reasoning_effort advertises ---


def test_catalog_exact_match_wins():
    """If desired effort is in the catalog, send it as-is."""
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p, "claude-opus-4.7", None, "high", catalog_effort_values=["low", "medium", "high"]
    )
    assert p.get("reasoning_effort") == "high"


def test_catalog_overrides_hardcoded_clamp():
    """If Copilot one day opens 'high' on opus-4.7, we should use it
    instead of the hardcoded medium clamp."""
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p, "claude-opus-4.7", 16000, None, catalog_effort_values=["medium", "high"]
    )
    # budget=16000 → desired "xhigh" → no exact match → step down to nearest
    # available, which here is "high" (highest in the allowlist).
    assert p.get("reasoning_effort") == "high"


def test_catalog_clamps_up_when_only_higher_tiers_are_available():
    """Desired low, catalog floors at medium → clamp UP to medium, not a 400."""
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p, "claude-opus-4.7", None, "low", catalog_effort_values=["medium", "high"]
    )
    assert p.get("reasoning_effort") == "medium"


def test_catalog_falls_back_lower_when_no_higher_available():
    """Desired 'xhigh' but catalog tops out at 'medium' → step down."""
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p, "claude-opus-4.7", None, "xhigh", catalog_effort_values=["low", "medium"]
    )
    assert p.get("reasoning_effort") == "medium"


def test_catalog_empty_list_strips_explicit_reasoning():
    """Catalog explicitly says no reasoning_effort → strip, no 400."""
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-haiku-4.5", 16000, "high", catalog_effort_values=[])
    assert "reasoning_effort" not in p


def test_catalog_empty_list_strips_budget_only_request() -> None:
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-haiku-4.5", 16000, None, catalog_effort_values=[])
    assert "reasoning_effort" not in p


def test_known_unsupported_model_strips_budget_only_request() -> None:
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-4o", 8192, None)
    assert "reasoning_effort" not in p


def test_catalog_preserves_xhigh_for_gpt5():
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p,
        "gpt-5.4",
        None,
        "xhigh",
        catalog_effort_values=["low", "medium", "high", "xhigh"],
    )
    assert p.get("reasoning_effort") == "xhigh"


def test_catalog_path_still_rewrites_max_tokens_for_gpt54():
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p, "gpt-5.4", None, "medium", catalog_effort_values=["low", "medium", "high"]
    )
    assert "max_tokens" not in p
    assert p.get("max_completion_tokens") == 100


def test_catalog_thinking_budget_maps_through_normal_table():
    """In the catalog path, budget→effort uses the normal mapping table."""
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p, "claude-sonnet-4.6", 8192, None, catalog_effort_values=["low", "medium", "high"]
    )
    # 8192 → "high" per EFFORT_TO_BUDGET threshold
    assert p.get("reasoning_effort") == "high"


def test_catalog_passes_max_through_when_advertised():
    """If the catalog advertises 'max', desired='max' should be sent as-is."""
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p,
        "claude-opus-4.7",
        None,
        "max",
        catalog_effort_values=["low", "medium", "high", "xhigh", "max"],
    )
    assert p.get("reasoning_effort") == "max"


def test_catalog_substitutes_xhigh_down_to_high_not_up_to_max():
    """A missing xhigh tier must map down to high, never upward to max."""
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p,
        "claude-opus-4.6",
        None,
        "xhigh",
        catalog_effort_values=["low", "medium", "high", "max"],
    )
    assert p.get("reasoning_effort") == "high"


def test_catalog_thinking_only_aims_at_catalog_top_tier():
    """When the client sends thinking_budget without an explicit effort, the
    catalog-driven path should aim at the catalog's top tier (not hardcoded 'high').
    """
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p,
        "claude-opus-4.6",
        4096,  # below "high" threshold but client opted into thinking
        None,
        catalog_effort_values=["low", "medium", "high", "max"],
    )
    # budget=4096 → desired derives to "medium", picked exactly: medium
    assert p.get("reasoning_effort") == "medium"

    p = _base_payload()
    apply_copilot_chat_reasoning(
        p,
        "claude-opus-4.6",
        None,
        None,
        catalog_effort_values=["low", "medium", "high", "max"],
    )
    # no budget, no effort → emit nothing
    assert "reasoning_effort" not in p


def test_catalog_tiny_thinking_budget_clamps_up_to_lowest_tier():
    """budget=1 has no downward tier → clamp UP to the catalog floor."""
    p = _base_payload()
    apply_copilot_chat_reasoning(
        p,
        "claude-opus-4.8",
        1,
        None,
        catalog_effort_values=["low", "medium", "high", "xhigh", "max"],
    )
    assert p.get("reasoning_effort") == "low"


def test_haiku_chat_payload_has_no_reasoning_real_layer():
    """Repro: Claude Code Haiku subagent sends thinking; payload must be valid
    and carry no reasoning field, instead of raising a 400."""
    from router_maestro.providers.base import ChatRequest, Message
    from router_maestro.providers.copilot import CopilotProvider

    provider = CopilotProvider()
    request = ChatRequest(
        model="github-copilot/claude-haiku-4.5",
        messages=[Message(role="user", content="hi")],
        max_tokens=1024,
        reasoning_effort="high",
        thinking_budget=16000,
        thinking_type="enabled",
    )

    payload = provider._build_chat_payload(request, stream=False)

    assert "reasoning_effort" not in payload
    assert "thinking_budget" not in payload
    assert payload["model"] == "github-copilot/claude-haiku-4.5"
