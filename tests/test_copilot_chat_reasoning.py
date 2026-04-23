"""Tests for per-model reasoning dispatch on Copilot's chat endpoint."""

from router_maestro.providers.copilot import apply_copilot_chat_reasoning


def _base_payload(extra: dict | None = None) -> dict:
    payload = {"model": "x", "messages": [], "max_tokens": 100}
    if extra:
        payload.update(extra)
    return payload


def test_claude_uses_thinking_budget_not_reasoning_effort():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "claude-opus-4.7", 32000, None)
    assert p.get("thinking_budget") == 32000
    assert "reasoning_effort" not in p


def test_claude_with_provider_prefix():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "github-copilot/claude-sonnet-4.6", 8192, "high")
    assert p.get("thinking_budget") == 8192
    assert "reasoning_effort" not in p


def test_gpt5_uses_reasoning_effort_not_thinking_budget():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-5.2", 16384, "high")
    assert p.get("reasoning_effort") == "high"
    assert "thinking_budget" not in p


def test_gpt5_derives_effort_from_budget_when_effort_missing():
    p = _base_payload()
    apply_copilot_chat_reasoning(p, "gpt-5.2", 16384, None)
    assert p.get("reasoning_effort") == "high"
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
