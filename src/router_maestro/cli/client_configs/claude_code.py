"""Claude Code (`~/.claude/settings.json`) config generation.

Claude Code is the only client that selects two models (main + small/fast),
injects synthetic 1M-context variants, and offers the auto-compact-window and
beta-endpoint prompts. All of that lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt

from router_maestro.cli.client_configs.base import (
    ClientConfig,
    GenerateContext,
    _bare_upstream_model_id,
    _model_key,
    _select_model_dict,
    _upstream_context_window,
    console,
)
from router_maestro.cli.client_configs.model_id import (
    ModelFamily,
    detect_family,
    to_anthropic_official,
)
from router_maestro.routing.model_ref import qualify_model_id

# Claude Code native model IDs for 1M context variants.
# Copilot no longer ships dedicated `-1m` / `-1m-internal` Opus variants — the
# base catalog entries for Opus 4.6/4.7/4.8, Sonnet 4.6, and Sonnet 5 already advertise
# max_context_window_tokens=1000000, so every native key maps straight to the
# base id. The `[1m]` suffix here only exists so Claude Code raises its
# auto-compact threshold to ~1M instead of clamping at the default 200K.
_OPUS_1M_NATIVE_KEY = "claude-opus-4-6[1m]"
_OPUS_1M_SOURCE_MODEL = "github-copilot/claude-opus-4.6"
_OPUS_47_1M_NATIVE_KEY = "claude-opus-4-7[1m]"
_OPUS_47_1M_SOURCE_MODEL = "github-copilot/claude-opus-4.7"
_OPUS_48_1M_NATIVE_KEY = "claude-opus-4-8[1m]"
_OPUS_48_1M_SOURCE_MODEL = "github-copilot/claude-opus-4.8"
_SONNET_46_1M_NATIVE_KEY = "claude-sonnet-4-6[1m]"
_SONNET_46_1M_SOURCE_MODEL = "github-copilot/claude-sonnet-4.6"
_SONNET_5_1M_NATIVE_KEY = "claude-sonnet-5[1m]"
_SONNET_5_1M_SOURCE_MODEL = "github-copilot/claude-sonnet-5"

_INJECTABLE_1M_VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    # (source_model, native_key, bare_id, display_name)
    (
        _OPUS_1M_SOURCE_MODEL,
        _OPUS_1M_NATIVE_KEY,
        "claude-opus-4.6",
        "Opus 4.6 1M (Auto-activated)",
    ),
    (
        _OPUS_47_1M_SOURCE_MODEL,
        _OPUS_47_1M_NATIVE_KEY,
        "claude-opus-4.7",
        "Opus 4.7 1M (Auto-activated)",
    ),
    (
        _OPUS_48_1M_SOURCE_MODEL,
        _OPUS_48_1M_NATIVE_KEY,
        "claude-opus-4.8",
        "Opus 4.8 1M (Auto-activated)",
    ),
    (
        _SONNET_46_1M_SOURCE_MODEL,
        _SONNET_46_1M_NATIVE_KEY,
        "claude-sonnet-4.6",
        "Sonnet 4.6 1M (Auto-activated)",
    ),
    (
        _SONNET_5_1M_SOURCE_MODEL,
        _SONNET_5_1M_NATIVE_KEY,
        "claude-sonnet-5",
        "Sonnet 5 1M (Auto-activated)",
    ),
)

# Claude Code recognizes 1M context windows natively for these model keys (the
# ones we inject via `_maybe_inject_opus_1m`) — the prompt offers a 1M default
# for them instead of the upstream-value option.
_CLAUDE_CODE_NATIVE_1M_KEYS: frozenset[str] = frozenset(
    {
        _OPUS_1M_NATIVE_KEY,
        _OPUS_47_1M_NATIVE_KEY,
        _OPUS_48_1M_NATIVE_KEY,
        _SONNET_46_1M_NATIVE_KEY,
        _SONNET_5_1M_NATIVE_KEY,
    }
)

# Default CLAUDE_CODE_AUTO_COMPACT_WINDOW for non-Claude models. Matches
# Claude Code's built-in window for Claude Opus / Sonnet (200K).
_CLAUDE_CODE_DEFAULT_AUTO_COMPACT_WINDOW = 200_000


def get_claude_code_paths() -> dict[str, Path]:
    """Get Claude Code settings paths."""
    return {
        "user": Path.home() / ".claude" / "settings.json",
        "project": Path.cwd() / ".claude" / "settings.json",
    }


def _maybe_inject_opus_1m(models: list[dict]) -> list[dict]:
    """Prepend Claude Code-native 1M context options for any source models present.

    Returns a new list (never mutates the input).
    """
    models_by_key = {_model_key(model): model for model in models}
    injected: list[dict] = []
    for source_model, native_key, bare_id, display_name in _INJECTABLE_1M_VARIANTS:
        source = models_by_key.get(source_model)
        if source is not None and (_upstream_context_window(source) or 0) >= 1_000_000:
            injected.append(
                {
                    "provider": "github-copilot",
                    "id": bare_id,
                    "name": display_name,
                    "display_key": native_key,
                    "wire_key": qualify_model_id("github-copilot", native_key),
                }
            )
    if not injected:
        return models
    return [*injected, *models]


def _prompt_endpoint_mode(model: dict | None) -> bool:
    """Prompt whether to use the beta native passthrough endpoint.

    Only offered when the selected model is a Claude model on GitHub Copilot.
    Returns True to use the beta endpoint, False for standard.
    """
    if model is None:
        return False
    provider = model.get("provider", "")
    model_id = _bare_upstream_model_id(model)
    if provider != "github-copilot" or not model_id.lower().startswith("claude-"):
        return False

    console.print("\n[bold]Endpoint mode[/bold]")
    console.print("  1. Standard (translation-based, battle-tested)")
    console.print("  2. Beta (native Copilot Anthropic passthrough — full thinking/cache fidelity)")
    choice = Prompt.ask("Select", choices=["1", "2"], default="2")
    return choice == "2"


def _prompt_auto_compact_window(model: dict | None) -> int | None:
    """Prompt the user whether to set CLAUDE_CODE_AUTO_COMPACT_WINDOW.

    Returns the chosen token count to write, or ``None`` to skip the env var.

    In Claude Code 2.1.162+, auto-compact's threshold check is short-circuited
    in interactive mode when the window source is "auto" — only ``env`` or
    ``settings`` source actually arms the trigger. So setting this env var is
    what turns the feature on at all; the exact value is secondary.

    For Claude Code-native 1M model keys (e.g. ``claude-opus-4-7[1m]``), the
    default offered is 1M and the upstream-value option is dropped — Copilot's
    real prompt cap on the 1M variant is below 1M but matching Claude Code's
    own view of the window is the more useful default here.

    For everything else, the prompt offers:
      * ``y`` — use the upstream context window (``max_prompt_tokens`` +
        ``max_output_tokens``, matching what Copilot's own model picker shows)
      * ``n`` — skip; do not set the env var
      * ``d`` — set the default 200K (matches Claude Opus/Sonnet's window)
    """
    if model is None:
        return None
    model_key = _model_key(model)
    native_key = model.get("display_key", model_key)
    is_native_1m = native_key in _CLAUDE_CODE_NATIVE_1M_KEYS

    upstream = _upstream_context_window(model)
    default_value = 1_000_000 if is_native_1m else _CLAUDE_CODE_DEFAULT_AUTO_COMPACT_WINDOW

    console.print()
    if is_native_1m:
        console.print(
            "[bold]Set CLAUDE_CODE_AUTO_COMPACT_WINDOW?[/bold]\n"
            f"  Selected: {model_key}\n"
            f"  [dim]Claude Code's interactive auto-compact only arms when this env var\n"
            f"  (or settings.autoCompactWindow) is set. Without it, the trigger is\n"
            f"  short-circuited regardless of model. Default ({default_value}) matches\n"
            f"  Claude Code's native 1M window for this model.[/dim]"
        )
        choices = ["n", "d"]
        prompt_text = f"n = skip / d = default: {default_value}"
    else:
        upstream_line = (
            f"  Upstream context window: {upstream}"
            if upstream is not None
            else "  Upstream context window: (unknown)"
        )
        console.print(
            "[bold]Set CLAUDE_CODE_AUTO_COMPACT_WINDOW?[/bold]\n"
            f"  Selected: {model_key}\n"
            f"{upstream_line}\n"
            f"  [dim]Claude Code's interactive auto-compact only arms when this env var\n"
            f"  (or settings.autoCompactWindow) is set. Without it, the trigger is\n"
            f"  short-circuited regardless of model. Default ({default_value}) matches\n"
            f"  Claude Opus/Sonnet's 200K window.[/dim]"
        )
        can_use_upstream = upstream is not None
        if can_use_upstream:
            choices = ["y", "n", "d"]
            prompt_text = f"y = upstream: {upstream} / n = skip / d = default: {default_value}"
        else:
            choices = ["n", "d"]
            prompt_text = f"n = skip / d = default: {default_value}"

    choice = Prompt.ask(prompt_text, choices=choices, default="d").lower()

    if choice == "n":
        return None
    if choice == "y" and not is_native_1m and upstream is not None:
        return int(upstream)
    return default_value


class ClaudeCodeConfig(ClientConfig):
    """Generate Claude Code CLI settings.json for router-maestro."""

    key = "claude-code"
    display_name = "Claude Code"
    description = "Generate settings.json for Claude Code CLI"

    def paths(self) -> dict[str, Path]:
        return get_claude_code_paths()

    def level_menu(self) -> tuple[str, str]:
        return (
            "User-level (~/.claude/settings.json)",
            "Project-level (./.claude/settings.json)",
        )

    def is_native_family(self, bare_id: str) -> bool:
        return detect_family(bare_id) is ModelFamily.ANTHROPIC

    def to_official_id(self, bare_id: str) -> str:
        return to_anthropic_official(bare_id)

    def load_models(self) -> list[dict]:
        # Import from base's module namespace so tests that patch
        # ``base._fetch_models`` are observed here.
        from router_maestro.cli.client_configs import base

        models = base._fetch_models()
        # If the 1M variant is available, offer the Claude Code-native model key
        # as an extra option. Claude Code sends the extended-context beta header
        # when this key is used, and the router resolves it automatically.
        models = _maybe_inject_opus_1m(models)
        base._display_models(models)
        return models

    def select_models(self, models: list[dict]) -> list[dict | None]:
        console.print("\n[bold]Step 3: Select main model[/bold]")
        main_model_dict = _select_model_dict(models, "Enter number (or 0 for auto-routing)")
        console.print("\n[bold]Step 4: Select small/fast model[/bold]")
        fast_model_dict = _select_model_dict(models, "Enter number", default="1")
        return [main_model_dict, fast_model_dict]

    def prompt_extras(self, selected_dicts: list[dict | None]) -> dict:
        main_model_dict = selected_dicts[0] if selected_dicts else None
        return {
            "auto_compact_window": _prompt_auto_compact_window(main_model_dict),
            "use_beta_endpoint": _prompt_endpoint_mode(main_model_dict),
        }

    def _anthropic_url(self, ctx: GenerateContext) -> str:
        path = "/api/anthropic/beta" if ctx.extras.get("use_beta_endpoint") else "/api/anthropic"
        return f"{self._base_url()}{path}"

    def write(self, *, level: str, path: Path, models: list[str], ctx: GenerateContext) -> None:
        main_model, fast_model = models[0], models[1]
        auth_token = self._auth_token()
        anthropic_url = self._anthropic_url(ctx)
        auto_compact_window = ctx.extras.get("auto_compact_window")

        env_config = {
            "ANTHROPIC_BASE_URL": anthropic_url,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "ANTHROPIC_MODEL": main_model,
            "ANTHROPIC_SMALL_FAST_MODEL": fast_model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ENABLE_LSP": "1",
        }
        if auto_compact_window is not None:
            env_config["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(auto_compact_window)

        # Load existing settings to preserve other sections (e.g., MCP servers)
        existing_config: dict = {}
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    existing_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # If file is corrupted, start fresh

        # Merge: update env variables while preserving existing ones
        existing_env = existing_config.get("env", {})
        if not isinstance(existing_env, dict):
            existing_env = {}
        existing_config["env"] = {**existing_env, **env_config}

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_config, f, indent=2)

    def render_success(
        self, *, level: str, path: Path, models: list[str], ctx: GenerateContext
    ) -> None:
        main_model, fast_model = models[0], models[1]
        anthropic_url = self._anthropic_url(ctx)
        auto_compact_window = ctx.extras.get("auto_compact_window")
        auto_compact_line = (
            f"Auto-compact window: {auto_compact_window} tokens\n\n"
            if auto_compact_window is not None
            else ""
        )
        console.print(
            Panel(
                f"[green]Created {path}[/green]\n\n"
                f"Main model: {main_model}\n"
                f"Fast model: {fast_model}\n\n"
                f"{auto_compact_line}"
                f"Endpoint: {anthropic_url}\n\n"
                "[dim]Start router-maestro server before using Claude Code:[/dim]\n"
                "  router-maestro server start",
                title="Success",
                border_style="green",
            )
        )
