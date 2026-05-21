"""Tests for the local-only integration test harness."""

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
