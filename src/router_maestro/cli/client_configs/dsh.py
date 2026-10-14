"""DeepSeek Harness (`~/.dsh/settings.yaml`) config generation."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from io import StringIO
from pathlib import Path
from typing import Any

from rich.panel import Panel
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from router_maestro.cli.client_configs.base import (
    ClientConfig,
    GenerateContext,
    IdStyle,
    ModelSelection,
    _bare_upstream_model_id,
    _model_key,
    _select_model_dict,
    console,
)

_DSH_PROVIDER_ID = "router-maestro"
_DSH_DEFAULT_CONTEXT_WINDOW = 262_144
_DSH_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
_DSH_SECRET_KEYS = {
    "apikey",
    "authtoken",
    "accesstoken",
    "refreshtoken",
    "password",
    "secret",
    "clientsecret",
    "privatekey",
    "authorization",
    "proxyauthorization",
    "xapikey",
}
_INTERNAL_ONLY_DISPLAY_SUFFIX = re.compile(
    r"\s*\(\s*internal[\s_-]+only\s*\)\s*$",
    re.IGNORECASE,
)


def get_dsh_paths() -> dict[str, Path]:
    """Return DSH's user-level settings target.

    DSH currently has no automatically loaded project settings file, so this
    client intentionally exposes only user scope.
    """
    return {"user": Path.home() / ".dsh" / "settings.yaml"}


def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 100
    return yaml


def _load_settings(path: Path) -> CommentedMap:
    if not path.exists():
        return CommentedMap()
    with open(path, encoding="utf-8") as file:
        document = _yaml().load(file)
    if document is None:
        return CommentedMap()
    if not isinstance(document, CommentedMap):
        raise TypeError("DSH settings.yaml must contain a top-level mapping")
    return document


def redact_dsh_settings(content: str) -> str:
    """Mask inline credentials in a generated DSH portal preview."""
    yaml = _yaml()
    document = yaml.load(content)
    if document is None:
        return content

    def redact(value: object) -> None:
        if isinstance(value, MutableMapping):
            for key, child in value.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                if normalized in _DSH_SECRET_KEYS:
                    value[key] = "********"
                else:
                    redact(child)
        elif isinstance(value, list):
            for child in value:
                redact(child)

    redact(document)
    output = StringIO()
    yaml.dump(document, output)
    return output.getvalue()


def _mapping(parent: MutableMapping[str, Any], key: str) -> MutableMapping[str, Any]:
    value = parent.get(key)
    if value is None:
        value = CommentedMap()
        parent[key] = value
    if not isinstance(value, MutableMapping):
        raise TypeError(f"DSH settings section '{key}' must be a mapping")
    return value


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _maximum_prompt_window(model: dict[str, Any]) -> int | None:
    candidates: list[int] = []
    scalar = _positive_int(model.get("max_prompt_tokens"))
    if scalar is not None:
        candidates.append(scalar)
    options = model.get("context_window_options")
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            prompt = _positive_int(option.get("max_prompt_tokens"))
            if prompt is not None:
                candidates.append(prompt)
    return max(candidates) if candidates else None


def _dsh_context_window(model: dict[str, Any]) -> int | None:
    """Return DSH's combined request + response capacity for one model."""
    total = _positive_int(model.get("max_context_window_tokens"))
    if total is not None:
        return total
    prompt = _maximum_prompt_window(model)
    output = _positive_int(model.get("max_output_tokens"))
    if prompt is not None and output is not None:
        return prompt + output
    return prompt


def _dsh_model_name(model: dict[str, Any]) -> str:
    raw = model.get("name")
    if not isinstance(raw, str) or not raw.strip():
        return _bare_upstream_model_id(model) or _model_key(model)
    return _INTERNAL_ONLY_DISPLAY_SUFFIX.sub("", raw).strip() or _model_key(model)


def _dsh_reasoning_efforts(model: dict[str, Any]) -> CommentedMap | bool | None:
    raw = model.get("reasoning_effort_values")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None

    available = {value for value in raw if isinstance(value, str)}
    supported = [effort for effort in _DSH_REASONING_EFFORTS if effort in available]
    # DSH requires a non-off entry; an explicit no-reasoning catalog uses
    # its boolean form instead of an effort map with only an off selector.
    if not supported:
        return False

    efforts = CommentedMap()
    # The selector describes actual support; provider-side compatibility
    # clamping does not make disabled reasoning available to this model.
    if "none" in available:
        efforts["off"] = "none"
    for effort in supported:
        efforts[effort] = effort
    return efforts


def _dsh_model_entry(model: dict[str, Any]) -> CommentedMap:
    entry = CommentedMap()
    entry["id"] = _model_key(model)
    entry["name"] = _dsh_model_name(model)
    context_window = _dsh_context_window(model)
    if context_window is not None:
        entry["contextWindow"] = context_window

    features = model.get("feature_capabilities")
    vision = isinstance(features, dict) and features.get("vision") is True
    entry["input"] = CommentedSeq(["text", "image"] if vision else ["text"])
    reasoning = _dsh_reasoning_efforts(model)
    if reasoning is not None:
        entry["reasoningEfforts"] = reasoning
    return entry


def _selected_model_from_settings(path: Path, models: list[dict]) -> dict | None:
    try:
        document = _load_settings(path)
    except (OSError, TypeError):
        return None
    selection = document.get("agent-default-model")
    if not isinstance(selection, MutableMapping):
        return None
    model_id = selection.get("model")
    if not isinstance(model_id, str):
        return None
    return next((model for model in models if _model_key(model) == model_id), None)


class DshConfig(ClientConfig):
    """Generate DeepSeek Harness settings for Router-Maestro."""

    key = "dsh"
    display_name = "DeepSeek Harness"
    description = "Generate settings.yaml for DSH"

    def paths(self) -> dict[str, Path]:
        return get_dsh_paths()

    def level_menu(self) -> tuple[str, str]:
        return (
            "User-level (~/.dsh/settings.yaml)",
            "Project-level is not supported by DSH",
        )

    def _select_level_and_path(self) -> tuple[str, Path]:
        console.print("\n[bold]Step 1: Configuration level[/bold]")
        console.print("  User-level (~/.dsh/settings.yaml) [dim](DSH has no project config)[/dim]")
        return "user", self.paths()["user"]

    def load_models(self) -> list[dict]:
        models = super().load_models()
        self._available_models = models
        return models

    def select_models(self, models: list[dict], *, level: str, path: Path) -> list[ModelSelection]:
        del level
        console.print("\n[bold]Step 2: Select default model[/bold]")
        selected = _select_model_dict(
            models,
            "Select DSH default model",
            default_model=_selected_model_from_settings(path, models),
            allow_auto=True,
        )
        return [ModelSelection(slot="main", model=selected)]

    def resolve_id_style(
        self,
        id_style: IdStyle | None,
        selected: list[dict | None],
    ) -> IdStyle:
        del selected
        if id_style not in (None, IdStyle.QUALIFIED):
            console.print(
                "[yellow]DSH model IDs remain provider-qualified to avoid ambiguous "
                "routes.[/yellow]"
            )
        return IdStyle.QUALIFIED

    def resolve_model_selection(self, selection: ModelSelection, id_style: IdStyle) -> str:
        """Keep DSH routes provider-qualified regardless of a generic caller hint."""
        del id_style
        return super().resolve_model_selection(selection, IdStyle.QUALIFIED)

    def is_native_family(self, bare_id: str) -> bool:
        del bare_id
        return False

    def to_official_id(self, bare_id: str) -> str:
        return bare_id

    def write(
        self,
        *,
        level: str,
        path: Path,
        models: dict[str, str],
        ctx: GenerateContext,
    ) -> None:
        if level != "user":
            raise ValueError("DSH supports user-level configuration only")

        source_value = ctx.extras.get("source_path")
        source_path = Path(source_value) if isinstance(source_value, str) else path
        document = _load_settings(source_path)
        llm = _mapping(document, "llm-pi-ai")
        providers = _mapping(llm, "providers")

        available_models = getattr(
            self,
            "_available_models",
            [model for model in ctx.selected_dicts if model is not None],
        )
        provider = CommentedMap()
        provider["displayName"] = "Router Maestro"
        provider["apiKeyEnv"] = "ROUTER_MAESTRO_API_KEY"
        provider["api"] = "openai-responses"
        provider["baseURL"] = f"{self._base_url_for(ctx)}/api/openai/v1"
        provider["defaultContextWindow"] = _DSH_DEFAULT_CONTEXT_WINDOW
        provider["models"] = CommentedSeq([_dsh_model_entry(model) for model in available_models])
        providers[_DSH_PROVIDER_ID] = provider

        selection = CommentedMap()
        selection["provider"] = _DSH_PROVIDER_ID
        selection["model"] = models["main"]
        document["agent-default-model"] = selection

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            _yaml().dump(document, file)

    def render_success(
        self,
        *,
        level: str,
        path: Path,
        models: dict[str, str],
        ctx: GenerateContext,
    ) -> None:
        del level
        model_count = len(getattr(self, "_available_models", ()))
        console.print(
            Panel(
                f"[green]Created {path}[/green]\n\n"
                f"Default model: {models['main']}\n"
                f"Catalog models: {model_count}\n"
                f"Backend URL: {self._base_url_for(ctx)}/api/openai/v1\n\n"
                "[dim]Make ROUTER_MAESTRO_API_KEY available to DSH, then run:[/dim]\n"
                "  dsh",
                title="Success",
                border_style="green",
            )
        )
