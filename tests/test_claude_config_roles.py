"""Claude Code model/context selection and settings migration tests."""

from __future__ import annotations

import json

import pytest

from router_maestro.cli.client_configs import claude_code as cc_claude
from router_maestro.cli.client_configs.base import (
    ClientConfig,
    ContextWindowChoice,
    GenerateContext,
    IdStyle,
    ModelSelection,
)
from router_maestro.cli.client_configs.claude_code import ClaudeCodeConfig


def _gpt_models() -> list[dict]:
    return [
        {"provider": "github-copilot", "id": "gpt-5.6-sol", "name": "Sol"},
        {"provider": "github-copilot", "id": "gpt-5.6-terra", "name": "Terra"},
        {"provider": "github-copilot", "id": "gpt-5.6-luna", "name": "Luna"},
    ]


def test_claude_uses_live_catalog_without_synthetic_injection(monkeypatch):
    models = _gpt_models()
    monkeypatch.setattr(
        "router_maestro.cli.client_configs.base._fetch_and_display_models",
        lambda: models,
    )

    assert ClaudeCodeConfig.load_models is ClientConfig.load_models
    assert ClaudeCodeConfig().load_models() is models


def test_context_prompt_uses_server_advertised_contexts(monkeypatch):
    captured = {}

    def fake_select(message, choices, **kwargs):
        captured.update(message=message, choices=choices, kwargs=kwargs)
        return ContextWindowChoice.CONTEXT_1M

    monkeypatch.setattr(cc_claude, "supports_dropdowns", lambda: True)
    monkeypatch.setattr(cc_claude, "select_dropdown", fake_select)

    selected = cc_claude._select_context_window(
        {
            "context_window_options": [
                {
                    "tier": "default",
                    "max_prompt_tokens": 272_000,
                    "is_default": True,
                },
                {
                    "tier": "long_context",
                    "max_prompt_tokens": 922_000,
                    "is_default": False,
                },
            ]
        },
        label="main",
        main=True,
        default=ContextWindowChoice.DEFAULT,
    )

    assert selected is ContextWindowChoice.CONTEXT_1M
    assert [label for label, _ in captured["choices"]] == [
        "272K (standard; no [1m])",
        "1M ([1m])",
    ]


def test_single_server_context_skips_context_prompt(monkeypatch):
    monkeypatch.setattr(
        cc_claude,
        "select_dropdown",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
    )

    selected = cc_claude._select_context_window(
        {
            "context_window_options": [
                {
                    "tier": "default",
                    "max_prompt_tokens": 128_000,
                    "is_default": True,
                }
            ]
        },
        label="main",
        main=True,
        default=ContextWindowChoice.DEFAULT,
    )

    assert selected is ContextWindowChoice.DEFAULT


@pytest.mark.parametrize("level", ["user", "project"])
def test_no_claude_catalog_prompts_roles_in_strength_order(tmp_path, monkeypatch, level):
    models = _gpt_models()
    calls: list[tuple[str, str, bool, bool]] = []

    def fake_select(catalog, *, slot, label, allow_auto, main, **kwargs):
        calls.append((slot, label, allow_auto, main))
        return ModelSelection(slot=slot, model=catalog[0])

    monkeypatch.setattr(cc_claude, "_select_model_with_context", fake_select)

    selections = ClaudeCodeConfig().select_models(
        models,
        level=level,
        path=tmp_path / "settings.json",
    )

    assert [selection.slot for selection in selections] == [
        "main",
        "fable",
        "opus",
        "sonnet",
        "haiku",
        "subagent",
    ]
    assert calls == [
        ("main", "main", True, True),
        ("fable", "Fable", False, False),
        ("opus", "Opus", False, False),
        ("sonnet", "Sonnet", False, False),
        ("haiku", "Haiku", False, False),
        ("subagent", "Subagent", False, False),
    ]


def test_any_claude_catalog_model_skips_role_mapping(tmp_path, monkeypatch):
    models = [
        *_gpt_models(),
        {"provider": "github-copilot", "id": "claude-sonnet-5", "name": "Claude"},
    ]

    def fake_select(catalog, *, slot, **kwargs):
        return ModelSelection(slot=slot, model=catalog[0])

    monkeypatch.setattr(cc_claude, "_select_model_with_context", fake_select)

    selections = ClaudeCodeConfig().select_models(
        models,
        level="user",
        path=tmp_path / "settings.json",
    )

    assert [selection.slot for selection in selections] == ["main"]


@pytest.mark.parametrize("level", ["user", "project"])
def test_write_maps_roles_cleans_legacy_and_orders_managed_keys(tmp_path, monkeypatch, level):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "env": {
                    "CUSTOM_LAST": "keep",
                    "ANTHROPIC_SMALL_FAST_MODEL": "legacy",
                    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000",
                },
                "permissions": {"allow": []},
            }
        ),
        encoding="utf-8",
    )
    selections = (
        ModelSelection("main", {}, ContextWindowChoice.CONTEXT_1M),
        ModelSelection("fable", {}, ContextWindowChoice.CONTEXT_1M),
        ModelSelection("opus", {}, ContextWindowChoice.CONTEXT_1M),
        ModelSelection("sonnet", {}, ContextWindowChoice.CONTEXT_1M),
        ModelSelection("haiku", {}, ContextWindowChoice.DEFAULT),
        ModelSelection("subagent", {}, ContextWindowChoice.CONTEXT_1M),
    )
    models = {
        "main": "github-copilot/gpt-5.6-sol[1m]",
        "fable": "github-copilot/gpt-5.6-sol[1m]",
        "opus": "github-copilot/gpt-5.6-sol[1m]",
        "sonnet": "github-copilot/gpt-5.6-terra[1m]",
        "haiku": "github-copilot/gpt-5.6-luna",
        "subagent": "github-copilot/gpt-5.6-terra[1m]",
    }
    config = ClaudeCodeConfig()
    monkeypatch.setattr(config, "_auth_token", lambda: "test-key")
    monkeypatch.setattr(config, "_base_url", lambda: "https://rm.example")

    config.write(
        level=level,
        path=path,
        models=models,
        ctx=GenerateContext(id_style=IdStyle.QUALIFIED, selections=selections),
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    env = data["env"]
    assert list(env) == [
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
        "CUSTOM_LAST",
    ]
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "1000000"
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in env
    assert data["permissions"] == {"allow": []}


@pytest.mark.parametrize("level", ["user", "project"])
def test_native_catalog_run_preserves_role_overrides_but_clears_stale_context(
    tmp_path,
    monkeypatch,
    level,
):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_DEFAULT_FABLE_MODEL": "existing-fable",
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": "existing-opus",
                    "ANTHROPIC_SMALL_FAST_MODEL": "legacy",
                    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000",
                }
            }
        ),
        encoding="utf-8",
    )
    selection = ModelSelection("main", {}, ContextWindowChoice.DEFAULT)
    config = ClaudeCodeConfig()
    monkeypatch.setattr(config, "_auth_token", lambda: "test-key")
    monkeypatch.setattr(config, "_base_url", lambda: "https://rm.example")

    config.write(
        level=level,
        path=path,
        models={"main": "github-copilot/claude-sonnet-5"},
        ctx=GenerateContext(id_style=IdStyle.QUALIFIED, selections=(selection,)),
    )

    env = json.loads(path.read_text(encoding="utf-8"))["env"]
    assert env["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "existing-fable"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "existing-opus"
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in env
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env
