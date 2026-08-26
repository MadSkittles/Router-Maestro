"""Claude Code (`~/.claude/settings.json`) config generation.

Claude Code receives the live Router-Maestro model catalog without synthetic
variants. Each selected model is paired with a client-side context choice; a
1M choice is encoded by appending ``[1m]`` after model-id resolution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt

from router_maestro.cli.client_configs.base import (
    ClientConfig,
    ContextWindowChoice,
    GenerateContext,
    IdStyle,
    ModelSelection,
    _bare_upstream_model_id,
    _format_token_count,
    _model_key,
    _model_operation_support,
    _select_model_dict,
    _upstream_context_window,
    console,
)
from router_maestro.cli.client_configs.model_id import (
    ModelFamily,
    detect_family,
    to_anthropic_official,
)
from router_maestro.cli.client_configs.prompts import select_dropdown, supports_dropdowns
from router_maestro.routing.capabilities import Operation

_CONTEXT_1M_SUFFIX = "[1m]"

_ROLE_SLOTS: tuple[tuple[str, str, str], ...] = (
    ("fable", "ANTHROPIC_DEFAULT_FABLE_MODEL", "Fable"),
    ("opus", "ANTHROPIC_DEFAULT_OPUS_MODEL", "Opus"),
    ("sonnet", "ANTHROPIC_DEFAULT_SONNET_MODEL", "Sonnet"),
    ("haiku", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "Haiku"),
    ("subagent", "CLAUDE_CODE_SUBAGENT_MODEL", "Subagent"),
)
_ROLE_ENV_BY_SLOT = {slot: env_key for slot, env_key, _ in _ROLE_SLOTS}
_ROLE_LABEL_BY_SLOT = {slot: label for slot, _, label in _ROLE_SLOTS}

_MANAGED_ENV_ORDER: tuple[str, ...] = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "CLAUDE_CODE_ENABLE_LSP",
)
_LEGACY_ENV_KEYS = frozenset({"ANTHROPIC_SMALL_FAST_MODEL"})


def get_claude_code_paths() -> dict[str, Path]:
    """Get Claude Code settings paths."""
    return {
        "user": Path.home() / ".claude" / "settings.json",
        "project": Path.cwd() / ".claude" / "settings.json",
    }


def _prompt_endpoint_mode(model: dict | None) -> bool:
    """Legacy endpoint-mode prompt retained for compatibility tests.

    Stable config generation no longer calls this helper.
    """
    if model is None:
        return False
    provider = model.get("provider", "")
    if provider != "github-copilot":
        return False
    supported = _model_operation_support(model, Operation.NATIVE_ANTHROPIC.value)
    if supported is None:
        supported = _bare_upstream_model_id(model).lower().startswith("claude-")
    if not supported:
        return False

    console.print("\n[bold]Endpoint mode[/bold]")
    console.print("  1. Standard (translation-based, battle-tested)")
    console.print("  2. Beta (native Copilot Anthropic passthrough — full thinking/cache fidelity)")
    choice = Prompt.ask("Select", choices=["1", "2"], default="2")
    return choice == "2"


def _load_existing_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            value = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _existing_env(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get("env", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _strip_context_suffix(value: str) -> str:
    return value[: -len(_CONTEXT_1M_SUFFIX)] if value.endswith(_CONTEXT_1M_SUFFIX) else value


def _context_from_existing_value(
    value: object,
    *,
    auto_compact_window: object = None,
) -> ContextWindowChoice:
    if isinstance(value, str) and value.endswith(_CONTEXT_1M_SUFFIX):
        return ContextWindowChoice.CONTEXT_1M
    if str(auto_compact_window) == "1000000":
        return ContextWindowChoice.CONTEXT_1M
    if str(auto_compact_window) == "200000":
        return ContextWindowChoice.CONTEXT_200K
    return ContextWindowChoice.DEFAULT


def _find_existing_model(
    models: list[dict],
    value: object,
    *,
    fallback: dict | None,
) -> dict | None:
    if not isinstance(value, str) or not value:
        return fallback
    configured = _strip_context_suffix(value).casefold()
    if configured == "router-maestro":
        return None

    for model in models:
        bare_id = _bare_upstream_model_id(model)
        candidates = {
            _strip_context_suffix(_model_key(model)).casefold(),
            _strip_context_suffix(bare_id).casefold(),
            _strip_context_suffix(to_anthropic_official(bare_id)).casefold(),
        }
        display_key = model.get("display_key")
        if isinstance(display_key, str):
            candidates.add(_strip_context_suffix(display_key).casefold())
        if configured in candidates:
            return model
    return fallback


def _catalog_has_claude_model(models: list[dict]) -> bool:
    return any(
        detect_family(_bare_upstream_model_id(model)) is ModelFamily.ANTHROPIC for model in models
    )


def _one_million_label(model: dict) -> str:
    upstream = _upstream_context_window(model)
    if upstream is not None and upstream < 1_000_000:
        return f"1M ([1m]; upstream advertises {_format_token_count(upstream)})"
    return "1M ([1m])"


def _catalog_context_choices(
    model: dict,
) -> list[tuple[str, ContextWindowChoice]] | None:
    """Map server context tiers to the context hints Claude Code can encode."""
    raw_options = model.get("context_window_options")
    if not isinstance(raw_options, list):
        return None

    options: list[tuple[int, bool]] = []
    for option in raw_options:
        if not isinstance(option, dict):
            continue
        limit = option.get("max_prompt_tokens")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            continue
        options.append((limit, option.get("is_default") is True))
    if not options:
        return None

    one_million_options = [option for option in options if _format_token_count(option[0]) == "1M"]
    standard_options = [option for option in options if option not in one_million_options]
    choices: list[tuple[str, ContextWindowChoice]] = []

    if standard_options:
        standard_limit, _ = next(
            (option for option in standard_options if option[1]),
            standard_options[0],
        )
        choices.append(
            (
                f"{_format_token_count(standard_limit)} (standard; no [1m])",
                ContextWindowChoice.DEFAULT,
            )
        )
    if one_million_options:
        long_limit = max(limit for limit, _ in one_million_options)
        choices.append(
            (
                f"{_format_token_count(long_limit)} ([1m])",
                ContextWindowChoice.CONTEXT_1M,
            )
        )
    return choices or None


def _select_context_window(
    model: dict | None,
    *,
    label: str,
    main: bool,
    default: ContextWindowChoice,
) -> ContextWindowChoice:
    if model is None:
        return ContextWindowChoice.DEFAULT

    choices = _catalog_context_choices(model)
    if choices is not None and len(choices) == 1:
        return choices[0][1]
    if choices is None and main:
        choices = [
            ("Client default (do not set auto-compact)", ContextWindowChoice.DEFAULT),
            ("200K", ContextWindowChoice.CONTEXT_200K),
            (_one_million_label(model), ContextWindowChoice.CONTEXT_1M),
        ]
    elif choices is None:
        choices = [
            ("Standard context (no suffix)", ContextWindowChoice.DEFAULT),
            (_one_million_label(model), ContextWindowChoice.CONTEXT_1M),
        ]

    if supports_dropdowns():
        return select_dropdown(
            f"{label} context window",
            choices,
            default=default,
        )

    allowed = [choice.value for _, choice in choices]
    fallback = default.value if default.value in allowed else allowed[0]
    answer = Prompt.ask(
        f"{label} context window",
        choices=allowed,
        default=fallback,
    )
    return ContextWindowChoice(answer)


def _select_model_with_context(
    models: list[dict],
    *,
    slot: str,
    label: str,
    default_model: dict | None,
    default_context: ContextWindowChoice,
    allow_auto: bool,
    main: bool,
) -> ModelSelection:
    model = _select_model_dict(
        models,
        f"Select {label} model",
        default_model=default_model,
        allow_auto=allow_auto,
    )
    context = _select_context_window(
        model,
        label=label,
        main=main,
        default=default_context,
    )
    return ModelSelection(slot=slot, model=model, context_window=context)


def _with_context_suffix(model: str, context: ContextWindowChoice) -> str:
    if context is not ContextWindowChoice.CONTEXT_1M or model.endswith(_CONTEXT_1M_SUFFIX):
        return model
    return f"{model}{_CONTEXT_1M_SUFFIX}"


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

    def select_models(self, models: list[dict], *, level: str, path: Path) -> list[ModelSelection]:
        existing_env = _existing_env(_load_existing_config(path))
        main_value = existing_env.get("ANTHROPIC_MODEL")
        main_default = _find_existing_model(models, main_value, fallback=None)
        main_context = _context_from_existing_value(
            main_value,
            auto_compact_window=existing_env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW"),
        )

        console.print("\n[bold]Step 2: Select main model and context[/bold]")
        main_selection = _select_model_with_context(
            models,
            slot="main",
            label="main",
            default_model=main_default,
            default_context=main_context,
            allow_auto=True,
            main=True,
        )
        selections = [main_selection]

        if _catalog_has_claude_model(models):
            return selections
        if level != "user":
            console.print(
                "\n[yellow]No Claude models are available. Default Claude model mappings "
                "are user-level only; run this command again and choose User-level to "
                "configure Fable, Opus, Sonnet, Haiku, and subagents.[/yellow]"
            )
            return selections

        console.print(
            "\n[bold]Step 3: Map Claude default model roles[/bold]\n"
            "[dim]No Claude-family model was found in the live catalog.[/dim]"
        )
        for slot, env_key, label in _ROLE_SLOTS:
            existing_value = existing_env.get(env_key)
            role_default = _find_existing_model(
                models,
                existing_value,
                fallback=main_selection.model or models[0],
            )
            role_context = _context_from_existing_value(existing_value)
            if existing_value is None:
                role_context = main_selection.context_window
            selections.append(
                _select_model_with_context(
                    models,
                    slot=slot,
                    label=label,
                    default_model=role_default,
                    default_context=role_context,
                    allow_auto=False,
                    main=False,
                )
            )
        return selections

    def resolve_model_selection(self, selection: ModelSelection, id_style: IdStyle) -> str:
        resolved = super().resolve_model_selection(selection, id_style)
        return _with_context_suffix(resolved, selection.context_window)

    def _anthropic_url(self, ctx: GenerateContext) -> str:
        del ctx
        return f"{self._base_url()}/api/anthropic"

    def write(
        self, *, level: str, path: Path, models: dict[str, str], ctx: GenerateContext
    ) -> None:
        auth_token = self._auth_token()
        anthropic_url = self._anthropic_url(ctx)
        selection_by_slot = {selection.slot: selection for selection in ctx.selections}

        env_config: dict[str, object] = {
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "ANTHROPIC_BASE_URL": anthropic_url,
            "ANTHROPIC_MODEL": models["main"],
        }
        if level == "user":
            for slot, env_key, _ in _ROLE_SLOTS:
                if slot in models:
                    env_config[env_key] = models[slot]

        main_context = selection_by_slot["main"].context_window
        if main_context is ContextWindowChoice.CONTEXT_200K:
            env_config["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "200000"
        elif main_context is ContextWindowChoice.CONTEXT_1M:
            env_config["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "1000000"

        env_config["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env_config["CLAUDE_CODE_ENABLE_LSP"] = "1"

        existing_config = _load_existing_config(path)
        existing_env = _existing_env(existing_config)
        selected_role_slots = _ROLE_ENV_BY_SLOT.keys() & models.keys()
        preserve_existing_roles = level == "user" and not selected_role_slots

        ordered_env: dict[str, object] = {}
        role_env_keys = set(_ROLE_ENV_BY_SLOT.values())
        for key in _MANAGED_ENV_ORDER:
            if key in env_config:
                ordered_env[key] = env_config[key]
            elif preserve_existing_roles and key in role_env_keys and key in existing_env:
                ordered_env[key] = existing_env[key]

        for key, value in existing_env.items():
            if key in _MANAGED_ENV_ORDER or key in _LEGACY_ENV_KEYS:
                continue
            ordered_env[key] = value

        existing_config["env"] = ordered_env
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(existing_config, file, indent=2)

    def render_success(
        self, *, level: str, path: Path, models: dict[str, str], ctx: GenerateContext
    ) -> None:
        del level
        selection_by_slot = {selection.slot: selection for selection in ctx.selections}
        lines = [f"[green]Created {path}[/green]", ""]
        for slot, model in models.items():
            label = "Main" if slot == "main" else _ROLE_LABEL_BY_SLOT[slot]
            context = selection_by_slot[slot].context_window.value
            lines.append(f"{label}: {model} ({context})")
        lines.extend(
            [
                "",
                f"Endpoint: {self._anthropic_url(ctx)}",
                "",
                "[dim]Start router-maestro server before using Claude Code:[/dim]",
                "  router-maestro server start",
            ]
        )
        console.print(
            Panel(
                "\n".join(lines),
                title="Success",
                border_style="green",
            )
        )


__all__ = [
    "ClaudeCodeConfig",
    "get_claude_code_paths",
    "_catalog_has_claude_model",
    "_context_from_existing_value",
    "_find_existing_model",
    "_prompt_endpoint_mode",
    "_select_context_window",
    "_select_model_with_context",
    "_with_context_suffix",
]
