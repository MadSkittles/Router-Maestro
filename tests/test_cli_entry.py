"""Tests for CLI entry point and KeyboardInterrupt handling."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from router_maestro.cli.entry import cli
from router_maestro.cli.main import app

runner = CliRunner()


class TestCliEntry:
    """Tests for the cli() entry point wrapper."""

    def test_cli_invokes_app_successfully(self) -> None:
        """The wrapper should invoke the Typer app and show version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "router-maestro version" in result.stdout

    def test_cli_help(self) -> None:
        """The wrapper should pass through --help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Multi-model routing" in result.stdout

    def test_cli_keyboard_interrupt_during_app_import(self) -> None:
        """KeyboardInterrupt during heavy import should exit with code 130."""
        with patch(
            "router_maestro.cli.entry.app",
            side_effect=KeyboardInterrupt,
            create=True,
        ):
            # Patch the import inside cli() to raise KeyboardInterrupt
            with patch(
                "builtins.__import__",
                side_effect=KeyboardInterrupt,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli()
                assert exc_info.value.code == 130

    def test_cli_subcommands_registered(self) -> None:
        """All expected subcommands should be registered."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ["server", "auth", "model", "context", "config"]:
            assert cmd in result.stdout
