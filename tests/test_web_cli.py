"""CLI tests for ``router-maestro web``."""

from unittest.mock import MagicMock

from typer.testing import CliRunner

from router_maestro.cli import web
from router_maestro.cli.main import app

runner = CliRunner()


def test_web_command_runs_loopback_portal(monkeypatch) -> None:
    portal_app = object()
    run = MagicMock()
    monkeypatch.setattr(web, "create_portal_app", lambda: portal_app)
    monkeypatch.setattr(web.uvicorn, "run", run)

    result = runner.invoke(app, ["web", "--no-open", "--port", "9876"])

    assert result.exit_code == 0
    run.assert_called_once_with(
        portal_app,
        host="127.0.0.1",
        port=9876,
        log_level="warning",
    )


def test_web_command_rejects_non_loopback_bind(monkeypatch) -> None:
    run = MagicMock()
    monkeypatch.setattr(web.uvicorn, "run", run)

    result = runner.invoke(app, ["web", "--no-open", "--host", "0.0.0.0"])

    assert result.exit_code == 2
    assert "only bind to a loopback" in result.stdout
    run.assert_not_called()
