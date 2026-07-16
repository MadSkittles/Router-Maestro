"""Gemini CLI (`~/.gemini/.env`) config generation."""

from __future__ import annotations

from pathlib import Path

from rich.panel import Panel

from router_maestro.cli.client_configs.base import (
    ClientConfig,
    GenerateContext,
    console,
)


def get_gemini_cli_paths() -> dict[str, Path]:
    """Get Gemini CLI config paths."""
    return {
        "user": Path.home() / ".gemini" / ".env",
        "project": Path.cwd() / ".gemini" / ".env",
    }


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, preserving order."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    except OSError:
        pass
    return env


def _write_env_file(path: Path, env: dict[str, str]) -> None:
    """Write a dict as a .env file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in env.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class GeminiConfig(ClientConfig):
    """Generate Gemini CLI .env for router-maestro."""

    key = "gemini"
    display_name = "Gemini CLI"
    description = "Generate .env for Gemini CLI"

    def paths(self) -> dict[str, Path]:
        return get_gemini_cli_paths()

    def level_menu(self) -> tuple[str, str]:
        return (
            "User-level (~/.gemini/.env)",
            "Project-level (./.gemini/.env)",
        )

    def write(self, *, level: str, path: Path, models: list[str], ctx: GenerateContext) -> None:
        model_name = models[0]
        gemini_url = f"{self._base_url()}/api/gemini"

        # Load existing .env to preserve other variables
        existing_env = _parse_env_file(path)

        # Router-Maestro's Gemini path converter accepts the provider-qualified public
        # ID, preserving an unambiguous selection when providers share an upstream ID.
        existing_env["GOOGLE_GEMINI_BASE_URL"] = gemini_url
        existing_env["GEMINI_API_KEY"] = self._auth_token()
        existing_env["GEMINI_MODEL"] = model_name
        existing_env["GEMINI_TELEMETRY_ENABLED"] = "false"

        _write_env_file(path, existing_env)

    def render_success(
        self, *, level: str, path: Path, models: list[str], ctx: GenerateContext
    ) -> None:
        model_name = models[0]
        gemini_url = f"{self._base_url()}/api/gemini"
        console.print(
            Panel(
                f"[green]Created {path}[/green]\n\n"
                f"Model: {model_name}\n"
                f"Backend URL: {gemini_url}\n"
                f"Telemetry: disabled\n\n"
                "[dim]Start router-maestro server before using Gemini CLI:[/dim]\n"
                "  router-maestro server start\n\n"
                "[dim]Then run Gemini CLI normally:[/dim]\n"
                "  gemini",
                title="Success",
                border_style="green",
            )
        )
