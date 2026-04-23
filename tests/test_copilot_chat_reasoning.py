"""Tests for per-model reasoning dispatch on Copilot's chat endpoint."""

from router_maestro.providers.copilot import apply_copilot_chat_reasoning


def _base_payload(extra: dict | None = None) -> dict:
    payload = {"model": "x", "messages": [], "max_tokens": 100}
    if extra:
        payload.update(extra)
    return payload


def test_claude_47_clamped_to_medium():
    """opus-4.7 gateway only accepts 'medium' — anything else gets clamped."""
    for input_effort in ("low", "high", None):
        for budget in (1024, 16000, None):
            if input_effort is None and budget is None:
                continue
            p = _base_payload()
            apply_copilot_chat_reasoning(p, "claude-opus-4.7", budget, input_effort)
            assert p.get("reasoning_effort") == "medium", (input_effort, budget)
            assert "thinking_budget" not in p


def test_claude_46_uses_reasoning_effort_not_thinking_budget():
    """opus-4.6 / sonnet-4.6 / opus-4.6-1m on Copilot expose effort, not budget."""
    for model in ("claude-opus-4.6", "claude-opus-4.6-1m", "claude-sonnet-4.6"):
        p = _base_payload()
        apply_copilot_chat_reasoning(p, model, 16000, None)
        assert "thinking_budget" not in p, model
        assert p.get("reasoning_effort") == "high", model


def test_claude_46_explicit_effort_wins_and_xhigh_clamped():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-sonnet-4.6", 1024, "xhigh")
    assert p.get("reasoning_effort") == "high"
    assert "thinking_budget" not in p


def test_claude_46_thinking_requested_with_tiny_budget_defaults_to_high():
    """Client asked for thinking but budget too small to map — be aggressive."""
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-opus-4.6", 100, None)
    assert p.get("reasoning_effort") == "high"


def test_claude_old_models_send_no_reasoning_field():
    for model in (
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-haiku-4.5",
    ):
        p = _base_payload()
        apply_copilot_chat_reasoning(p, model, 16000, "high")
        assert "thinking_budget" not in p, model
        assert "reasoning_effort" not in p, model


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


def test_gpt4o_omits_both_fields():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-4o", 8192, "medium")
    assert "thinking_budget" not in p
    assert "reasoning_effort" not in p


def test_gemini_omits_both_fields():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gemini-2.5-pro", 8192, "medium")
    assert "thinking_budget" not in p
    assert "reasoning_effort" not in p


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
