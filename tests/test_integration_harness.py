"""Tests for the local-only integration test harness."""

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_integration_tests_are_outside_default_pytest_tree():
    """Integration tests should not be discovered by the default tests/ run."""
    integration_dir = ROOT / "integration_tests"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert integration_dir.is_dir()
    assert 'testpaths = ["tests"]' in pyproject


def test_makefile_exposes_explicit_integration_test_target():
    """Local live-backend tests should have an explicit Makefile entrypoint."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "integration-test:" in makefile
    assert "uv run pytest integration_tests/ -v" in makefile


def test_integration_model_matrix_defaults_to_full():
    """The default integration suite should cover the full Copilot model matrix."""
    conftest = (ROOT / "integration_tests" / "conftest.py").read_text(encoding="utf-8")

    assert "DEFAULT_MAX_MODEL_MATRIX = 0" in conftest


def test_model_matrix_payload_has_reasoning_safe_output_budget():
    """The full matrix should give reasoning-heavy models enough output budget."""
    conftest = importlib.import_module("integration_tests.conftest")

    payload = conftest.model_matrix_chat_payload("github-copilot/gemini-2.5-pro")

    assert payload["max_tokens"] >= 512
    assert payload["reasoning_effort"] == "low"


def test_reasoning_matrix_payloads_cover_budget_and_effort_controls():
    """Live matrix helpers should exercise thinking budgets and reasoning effort."""
    conftest = importlib.import_module("integration_tests.conftest")

    anthropic = conftest.anthropic_reasoning_payload(
        "github-copilot/claude-sonnet-4.6",
        budget=4096,
        stream=True,
    )
    openai = conftest.openai_reasoning_payload(
        "github-copilot/gpt-5.4",
        effort="high",
        stream=True,
    )

    assert anthropic["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert anthropic["max_tokens"] > 4096
    assert openai["reasoning_effort"] == "high"
    assert openai["stream_options"] == {"include_usage": True}


def test_integration_tests_include_reasoning_and_gemini_family_matrices():
    """Local live tests should cover the e2e reasoning and Gemini family gaps."""
    integration_dir = ROOT / "integration_tests"

    reasoning = (integration_dir / "test_live_reasoning_matrix.py").read_text(encoding="utf-8")
    gemini = (integration_dir / "test_live_gemini_matrix.py").read_text(encoding="utf-8")

    assert "test_anthropic_claude_thinking_budget_matrix" in reasoning
    assert "test_anthropic_gpt5_responses_bridge_thinking_budget_matrix" in reasoning
    assert "test_openai_chat_reasoning_effort_matrix" in reasoning
    assert "test_gemini_family_generate_content_matrix" in gemini


def test_integration_harness_documents_existing_config_usage():
    """The harness should say it reuses the user's existing RM configuration."""
    conftest = (ROOT / "integration_tests" / "conftest.py").read_text(encoding="utf-8")

    assert "get_current_context_api_key" in conftest
    assert "ROUTER_MAESTRO_API_KEY" in conftest
    assert "router_maestro.server:app" in conftest


def test_integration_tests_do_not_cover_admin_endpoints():
    """Live integration tests should cover model calls, not admin endpoints."""
    integration_dir = ROOT / "integration_tests"

    for path in integration_dir.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        assert "/api/admin/" not in content, path
