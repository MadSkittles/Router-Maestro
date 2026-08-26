"""Interactive selection helpers for client config generation."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TypeVar

import questionary
import typer

T = TypeVar("T")


def supports_dropdowns() -> bool:
    """Return whether the current terminal can host interactive selectors."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def select_dropdown(
    message: str,
    choices: Sequence[tuple[str, T]],
    *,
    default: T | None = None,
    searchable: bool = False,
) -> T:
    """Select one typed value using an arrow-key dropdown.

    The caller owns non-interactive fallback behavior. Returning ``None`` from
    Questionary means the user cancelled the prompt, not that a choice whose
    value is ``None`` was selected; labels therefore remain the prompt's wire
    value and are mapped back to the typed payload here.
    """
    if not choices:
        raise ValueError("select_dropdown requires at least one choice")

    values_by_label = {label: value for label, value in choices}
    default_label = next((label for label, value in choices if value == default), None)
    labels = list(values_by_label)

    def resolve_label(value: str) -> str | None:
        if value in values_by_label:
            return value
        if not value and default_label is not None:
            return default_label
        folded = value.casefold()
        matches = [label for label in labels if folded in label.casefold()]
        return matches[0] if len(matches) == 1 else None

    if searchable:
        searchable_message = (
            f"{message} (Enter uses the current default)" if default_label is not None else message
        )
        prompt = questionary.autocomplete(
            searchable_message,
            choices=labels,
            default="",
            ignore_case=True,
            match_middle=True,
            validate=lambda value: resolve_label(value) is not None or "Select a listed model",
        )
    else:
        prompt = questionary.select(
            message,
            choices=labels,
            default=default_label,
        )

    answer = prompt.ask()
    if answer is None:
        raise typer.Abort()
    resolved = resolve_label(answer)
    if resolved is None:  # defensive: validation should make this unreachable
        raise ValueError(f"Unknown dropdown selection: {answer}")
    return values_by_label[resolved]


__all__ = ["select_dropdown", "supports_dropdowns"]
