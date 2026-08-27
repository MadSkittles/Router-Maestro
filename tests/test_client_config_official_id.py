"""Tests for qualified, bare, and official ID resolution on ClientConfig.

Covers ``resolve_model_string`` (per selected model) and ``resolve_id_style``
(whether to prompt), including every edge the feature must honor: the
auto-routing sentinel and explicit wire/custom keys are never converted;
``BARE`` removes only the provider prefix, while legacy ``OFFICIAL`` keeps its
native-family conversion semantics.
"""

from rich.console import Console

from router_maestro.cli.client_configs import base as cc_base
from router_maestro.cli.client_configs.base import IdStyle, ModelSelection
from router_maestro.cli.client_configs.claude_code import ClaudeCodeConfig
from router_maestro.cli.client_configs.codex import CodexConfig
from router_maestro.cli.client_configs.gemini import GeminiConfig

_GPT = {"provider": "github-copilot", "id": "gpt-5.5", "name": "GPT-5.5"}
_CLAUDE = {"provider": "github-copilot", "id": "claude-opus-4.6", "name": "Claude Opus 4.6"}
_GEMINI = {"provider": "github-copilot", "id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"}
_WIRE_1M = {
    "provider": "github-copilot",
    "id": "claude-opus-4.6",
    "name": "Opus 4.6 1M",
    "display_key": "claude-opus-4-6[1m]",
    "wire_key": "github-copilot/claude-opus-4-6[1m]",
}
_CUSTOM = {
    "provider": "github-copilot",
    "id": "claude-opus-4.6-1m",
    "name": "custom",
    "custom_key": "claude-opus-4-6[1m]",
}


class TestResolveModelString:
    def test_sentinel_never_converted(self):
        assert CodexConfig().resolve_model_string(None, IdStyle.OFFICIAL) == "router-maestro"
        assert CodexConfig().resolve_model_string(None, IdStyle.QUALIFIED) == "router-maestro"

    def test_qualified_returns_provider_prefixed(self):
        assert (
            CodexConfig().resolve_model_string(_GPT, IdStyle.QUALIFIED) == "github-copilot/gpt-5.5"
        )

    def test_official_native_openai_keeps_dots_unprefixed(self):
        assert CodexConfig().resolve_model_string(_GPT, IdStyle.OFFICIAL) == "gpt-5.5"

    def test_official_native_anthropic_dashes(self):
        assert (
            ClaudeCodeConfig().resolve_model_string(_CLAUDE, IdStyle.OFFICIAL) == "claude-opus-4-6"
        )

    def test_official_native_gemini_keeps_dots(self):
        assert GeminiConfig().resolve_model_string(_GEMINI, IdStyle.OFFICIAL) == "gemini-2.5-pro"

    def test_bare_non_native_drops_only_provider_prefix(self):
        assert CodexConfig().resolve_model_string(_CLAUDE, IdStyle.BARE) == "claude-opus-4.6"

    def test_official_non_native_stays_qualified_and_warns(self, monkeypatch):
        rec = Console(record=True, width=120)
        monkeypatch.setattr(cc_base, "console", rec)
        # Codex is OpenAI-native; a Claude model must NOT be converted.
        result = CodexConfig().resolve_model_string(_CLAUDE, IdStyle.OFFICIAL)
        assert result == "github-copilot/claude-opus-4.6"
        assert "not a native" in rec.export_text()

    def test_official_wire_key_entry_untouched(self):
        # Claude 1M injected entry: explicit wire_key must survive OFFICIAL mode.
        assert (
            ClaudeCodeConfig().resolve_model_string(_WIRE_1M, IdStyle.OFFICIAL)
            == "github-copilot/claude-opus-4-6[1m]"
        )

    def test_official_custom_key_entry_untouched(self):
        assert (
            ClaudeCodeConfig().resolve_model_string(_CUSTOM, IdStyle.OFFICIAL)
            == "claude-opus-4-6[1m]"
        )


class TestResolveIdStyle:
    def test_cli_option_wins_no_prompt(self, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("prompt must not be shown when id_style is explicit")

        monkeypatch.setattr(cc_base.Prompt, "ask", boom)
        assert CodexConfig().resolve_id_style(IdStyle.OFFICIAL, [_GPT]) is IdStyle.OFFICIAL
        assert CodexConfig().resolve_id_style(IdStyle.BARE, [_GPT]) is IdStyle.BARE
        assert CodexConfig().resolve_id_style(IdStyle.QUALIFIED, [_GPT]) is IdStyle.QUALIFIED

    def test_remove_prefix_choice_returns_bare(self, monkeypatch):
        monkeypatch.setattr(cc_base.Prompt, "ask", lambda *a, **kw: "no")
        assert CodexConfig().resolve_id_style(None, [_GPT]) is IdStyle.BARE

    def test_prompt_default_qualified(self, monkeypatch):
        # Simulate the user pressing enter → Prompt.ask returns the default it was given.
        monkeypatch.setattr(cc_base.Prompt, "ask", lambda *a, **kw: kw.get("default"))
        assert CodexConfig().resolve_id_style(None, [_GPT]) is IdStyle.QUALIFIED

    def test_no_prompt_when_nothing_convertible(self, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("prompt must not be shown when nothing is convertible")

        monkeypatch.setattr(cc_base.Prompt, "ask", boom)
        # Only the auto-routing sentinel selected → nothing to convert.
        assert CodexConfig().resolve_id_style(None, [None]) is IdStyle.QUALIFIED
        # Wire-key entry is not convertible.
        assert ClaudeCodeConfig().resolve_id_style(None, [_WIRE_1M]) is IdStyle.QUALIFIED

    def test_prompts_for_non_native_prefixed_model(self, monkeypatch):
        monkeypatch.setattr(cc_base.Prompt, "ask", lambda *a, **kw: "no")
        assert CodexConfig().resolve_id_style(None, [_CLAUDE]) is IdStyle.BARE


def test_provider_prefix_choice_is_the_last_interactive_hook(tmp_path, monkeypatch):
    config = CodexConfig()
    events: list[str] = []
    selection = ModelSelection(slot="main", model=_GPT)

    monkeypatch.setattr(
        config,
        "_select_level_and_path",
        lambda: ("user", tmp_path / "config.toml"),
    )
    monkeypatch.setattr(cc_base, "_backup_if_exists", lambda path: None)
    monkeypatch.setattr(config, "load_models", lambda: [_GPT])
    monkeypatch.setattr(
        config,
        "select_models",
        lambda models, **kwargs: events.append("models") or [selection],
    )
    monkeypatch.setattr(
        config,
        "prompt_extras",
        lambda selections: events.append("extras") or {},
    )
    monkeypatch.setattr(
        config,
        "resolve_id_style",
        lambda style, selected: events.append("prefix") or IdStyle.QUALIFIED,
    )
    monkeypatch.setattr(config, "write", lambda **kwargs: events.append("write"))
    monkeypatch.setattr(config, "render_success", lambda **kwargs: events.append("success"))

    config.generate()

    assert events == ["models", "extras", "prefix", "write", "success"]
