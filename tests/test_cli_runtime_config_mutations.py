"""CLI runtime-config mutations must use one compare-and-swap snapshot."""

from __future__ import annotations

from copy import deepcopy

import pytest
import typer
from rich.console import Console

from router_maestro.cli import model as model_cli
from router_maestro.cli.client import AdminConfigConflictError


def _runtime_config() -> dict:
    return {
        "revision": "a" * 64,
        "priorities": ["first/model", "second/model"],
        "fallback": {"strategy": "priority", "maxRetries": 2},
        "model_overrides": {"sentinel/model": {"max_output_tokens": 1234}},
        "thinking": {
            "default_budget": 4321,
            "auto_enable": True,
            "model_budgets": {"sentinel/model": 2345},
        },
        "guards": {
            "leak_guard": {"enabled": False},
            "runaway_guard": {
                "enabled": False,
                "max_bytes": 7654321,
                "max_deltas": 23456,
            },
        },
        "beta_strip": ["sentinel-beta-*"],
        "audit": {"enabled": True, "trace_dir": "/sentinel/traces"},
    }


class _RuntimeConfigClient:
    def __init__(self, *, conflict: bool = False) -> None:
        self.source = _runtime_config()
        self.conflict = conflict
        self.get_calls = 0
        self.patch_calls: list[tuple[dict, str]] = []

    async def get_runtime_config(self) -> dict:
        self.get_calls += 1
        return deepcopy(self.source)

    async def get_priorities(self) -> dict:
        raise AssertionError("mutations must read the complete versioned runtime config")

    async def set_priorities(self, *args, **kwargs) -> bool:
        raise AssertionError("set_priorities performs a second GET and must not be used")

    async def patch_runtime_config(self, *, config: dict, revision: str) -> dict:
        self.patch_calls.append((deepcopy(config), revision))
        if self.conflict:
            raise AdminConfigConflictError("b" * 64)
        return {**config, "revision": "c" * 64}


def _assert_single_snapshot_patch(client: _RuntimeConfigClient) -> dict:
    assert client.get_calls == 1
    assert len(client.patch_calls) == 1
    patched, revision = client.patch_calls[0]
    assert revision == "a" * 64
    assert "revision" not in patched
    for field in ("model_overrides", "thinking", "guards", "beta_strip", "audit"):
        assert patched[field] == client.source[field]
    return patched


def test_priority_add_patches_the_same_complete_snapshot(monkeypatch) -> None:
    client = _RuntimeConfigClient()
    monkeypatch.setattr(model_cli, "get_admin_client", lambda: client)

    model_cli.priority_add("new/model", position=1)

    patched = _assert_single_snapshot_patch(client)
    assert patched["priorities"] == ["new/model", "first/model", "second/model"]
    assert patched["fallback"] == client.source["fallback"]


def test_priority_remove_patches_the_same_complete_snapshot(monkeypatch) -> None:
    client = _RuntimeConfigClient()
    monkeypatch.setattr(model_cli, "get_admin_client", lambda: client)

    model_cli.priority_remove("first/model")

    patched = _assert_single_snapshot_patch(client)
    assert patched["priorities"] == ["second/model"]
    assert patched["fallback"] == client.source["fallback"]


def test_priority_clear_patches_the_same_complete_snapshot(monkeypatch) -> None:
    client = _RuntimeConfigClient()
    monkeypatch.setattr(model_cli, "get_admin_client", lambda: client)

    model_cli.priority_clear()

    patched = _assert_single_snapshot_patch(client)
    assert patched["priorities"] == []
    assert patched["fallback"] == client.source["fallback"]


def test_fallback_set_patches_the_same_complete_snapshot(monkeypatch) -> None:
    client = _RuntimeConfigClient()
    monkeypatch.setattr(model_cli, "get_admin_client", lambda: client)

    model_cli.fallback_set(strategy="same-model", max_retries=7)

    patched = _assert_single_snapshot_patch(client)
    assert patched["priorities"] == client.source["priorities"]
    assert patched["fallback"] == {"strategy": "same-model", "maxRetries": 7}


def test_config_conflict_never_prints_mutation_success(monkeypatch) -> None:
    client = _RuntimeConfigClient(conflict=True)
    output = Console(record=True, width=120)
    monkeypatch.setattr(model_cli, "get_admin_client", lambda: client)
    monkeypatch.setattr(model_cli, "console", output)

    with pytest.raises(typer.Exit):
        model_cli.priority_add("new/model", position=None)

    rendered = output.export_text()
    assert "Runtime configuration changed; refresh and retry" in rendered
    assert "Added 'new/model'" not in rendered
    _assert_single_snapshot_patch(client)
