"""Interactive dropdown adapter tests."""

from __future__ import annotations

import pytest
import typer

from router_maestro.cli.client_configs import prompts
from router_maestro.cli.client_configs.base import _model_choice_label


class _PromptResult:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


def test_searchable_dropdown_returns_typed_value(monkeypatch):
    captured = {}

    def fake_autocomplete(message, **kwargs):
        captured.update(message=message, **kwargs)
        return _PromptResult("Second")

    monkeypatch.setattr(prompts.questionary, "autocomplete", fake_autocomplete)

    result = prompts.select_dropdown(
        "Pick model",
        [("First", {"id": "first"}), ("Second", {"id": "second"})],
        searchable=True,
    )

    assert result == {"id": "second"}
    assert captured["match_middle"] is True
    assert captured["ignore_case"] is True


def test_searchable_dropdown_accepts_unique_substring(monkeypatch):
    monkeypatch.setattr(
        prompts.questionary,
        "autocomplete",
        lambda *args, **kwargs: _PromptResult("gpt-5.6-sol"),
    )

    result = prompts.select_dropdown(
        "Pick model",
        [
            ("github-copilot/gpt-5.6-sol — GPT-5.6 Sol", "sol"),
            ("github-copilot/gpt-5.6-terra — GPT-5.6 Terra", "terra"),
        ],
        searchable=True,
    )

    assert result == "sol"


def test_cancelled_dropdown_aborts(monkeypatch):
    monkeypatch.setattr(
        prompts.questionary,
        "select",
        lambda *args, **kwargs: _PromptResult(None),
    )

    with pytest.raises(typer.Abort):
        prompts.select_dropdown("Pick", [("Only", "value")])


def test_model_choice_label_displays_all_supported_contexts():
    label = _model_choice_label(
        {
            "provider": "github-copilot",
            "id": "gpt-5.6-sol",
            "name": "GPT-5.6 Sol",
            "context_window_options": [
                {"tier": "default", "max_prompt_tokens": 272_000},
                {"tier": "long_context", "max_prompt_tokens": 922_000},
            ],
        }
    )

    assert label == "github-copilot/gpt-5.6-sol — GPT-5.6 Sol — 272K / 1M"
