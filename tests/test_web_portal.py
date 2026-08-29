"""Tests for the loopback-only Router-Maestro web portal."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Literal

import httpx
import pytest
from fastapi.testclient import TestClient

from router_maestro.cli.client_configs.base import ContextWindowChoice
from router_maestro.config.contexts import ContextConfig, ContextsConfig
from router_maestro.web.app import create_portal_app
from router_maestro.web.service import (
    PortalAutoConfigRequest,
    PortalConfigRequest,
    PortalService,
    PortalServiceError,
)


def _contexts() -> ContextsConfig:
    return ContextsConfig(
        current="hk",
        contexts={
            "hk": ContextConfig(
                endpoint="https://router.example",
                api_key="sk-rm-secret",
            )
        },
    )


def _catalog() -> dict:
    return {
        "models": [
            {
                "provider": "github-copilot",
                "id": "github-copilot/gpt-5.6-sol",
                "name": "GPT-5.6 Sol",
                "context_window_options": [
                    {
                        "tier": "default",
                        "max_prompt_tokens": 272_000,
                        "is_default": True,
                    },
                    {
                        "tier": "long_context",
                        "max_prompt_tokens": 1_000_000,
                        "is_default": False,
                    },
                ],
                "operation_capabilities": {
                    "responses": True,
                    "chat": True,
                },
            },
            {
                "provider": "openai",
                "id": "openai/gpt-5.6-sol",
                "name": "GPT-5.6 Sol",
                "context_window_options": [
                    {
                        "tier": "default",
                        "max_prompt_tokens": 272_000,
                        "is_default": True,
                    }
                ],
                "operation_capabilities": {"responses": True},
            },
            {
                "provider": "github-copilot",
                "id": "github-copilot/gpt-5.6-sol-fast",
                "name": "GPT-5.6 Sol Fast (internal only)",
                "context_window_options": [
                    {
                        "tier": "default",
                        "max_prompt_tokens": 272_000,
                        "is_default": True,
                    }
                ],
                "operation_capabilities": {"responses": True},
            },
        ]
    }


def _stub_bundled_codex_catalog() -> dict:
    return {
        "models": [
            {
                "slug": "gpt-5.6-terra",
                "display_name": "GPT-5.6 Terra",
                "description": "Bundled baseline",
                "context_window": 400_000,
                "max_context_window": 1_050_000,
            }
        ]
    }


def _transport(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "healthy"})
    if request.url.path == "/api/admin/models":
        assert request.headers["Authorization"] == "Bearer sk-rm-secret"
        return httpx.Response(200, json=_catalog())
    if request.url.path == "/api/admin/priorities":
        config = {
            "revision": "a" * 64,
            "priorities": [],
            "fallback": {"strategy": "priority", "maxRetries": 2},
            "model_overrides": {},
            "thinking": {"default_budget": 16000, "auto_enable": False, "model_budgets": {}},
            "guards": {
                "leak_guard": {"enabled": True},
                "runaway_guard": {"enabled": True, "max_bytes": 10000000, "max_deltas": 50000},
            },
            "beta_strip": [],
            "audit": {"enabled": False, "trace_dir": None},
            "auto": {
                "mode": "task-router",
                "capability_policy": "strict",
                "priority_chain": [],
                "task_router": {
                    "router_model": "github-copilot/gpt-5.6-sol",
                    "task_models": {
                        task: "github-copilot/gpt-5.6-sol"
                        for task in ("fast", "general", "coding", "deep_reasoning")
                    },
                },
            },
        }
        if request.method == "PATCH":
            body = json.loads(request.content)
            config.update(body)
            config["revision"] = "b" * 64
        return httpx.Response(200, json=config)
    return httpx.Response(404)


def _service(tmp_path: Path, **overrides) -> PortalService:
    defaults = {
        "contexts_loader": _contexts,
        "home": tmp_path / "home",
        "projects_file": tmp_path / "router-maestro" / "projects.json",
        "environment": {},
        "transport": httpx.MockTransport(_transport),
    }
    defaults.update(overrides)
    return PortalService(**defaults)


def test_context_list_redacts_keys(tmp_path: Path) -> None:
    service = _service(tmp_path)

    contexts = service.list_contexts()
    assert contexts[0].model_dump() == {
        "name": "hk",
        "endpoint": "https://router.example",
        "current": True,
        "has_api_key": True,
    }


def test_model_display_names_use_consistent_product_casing() -> None:
    assert PortalService._model_name({"name": "gpt-5.6-sol", "id": "gpt-5.6-sol"}) == (
        "GPT-5.6 Sol"
    )
    internal = {
        "name": "GPT-5.6 Sol Fast (internal only)",
        "id": "gpt-5.6-sol-fast",
    }
    assert PortalService._model_name(internal) == "GPT-5.6 Sol Fast"
    assert PortalService._is_internal_model(internal) is True
    assert PortalService._is_internal_model({"name": "Internal Affairs"}) is False


def test_virtual_auto_model_uses_aggregate_transport_capability_summary() -> None:
    assert PortalService._transport_names(
        {
            "virtual": True,
            "operation_capabilities": {"responses": True},
        }
    ) == ["Responses", "Chat", "Messages"]
    assert PortalService._transport_names(
        {
            "virtual": False,
            "operation_capabilities": {"responses": True},
        }
    ) == ["Responses"]
    assert (
        PortalService._model_name({"name": "MAI-Code-1.1-Flash", "id": "mai-code-1.1-flash"})
        == "MAI Code 1.1 Flash"
    )


@pytest.mark.asyncio
async def test_health_and_models_are_context_scoped(tmp_path: Path) -> None:
    ticks = [10.0, 10.041]

    def clock() -> float:
        return ticks.pop(0) if ticks else 20.0

    service = _service(tmp_path, clock=clock)

    health = await service.health("hk")
    catalog = await service.list_models("hk")

    assert health.healthy is True
    assert health.latency_ms == 41
    assert catalog.requires_claude_role_mappings is True
    assert [model.key for model in catalog.models] == [
        "github-copilot/gpt-5.6-sol",
        "openai/gpt-5.6-sol",
        "github-copilot/gpt-5.6-sol-fast",
    ]
    assert catalog.models[0].provider_name == "GitHub Copilot"
    assert catalog.models[0].context_label == "272K / 1M"
    assert catalog.models[0].transports == ["Responses", "Chat"]
    assert catalog.models[2].name == "GPT-5.6 Sol Fast"
    assert catalog.models[2].internal is True


@pytest.mark.asyncio
async def test_auto_config_round_trips_through_selected_context(tmp_path: Path) -> None:
    service = _service(tmp_path)
    current = await service.get_auto_config("hk")

    updated = await service.update_auto_config(
        "hk",
        PortalAutoConfigRequest(
            revision=current["revision"],
            mode="priority-chain",
            capability_policy="strict",
            priority_chain=["github-copilot/gpt-5.6-sol"],
            router_model="github-copilot/gpt-5.6-sol",
            task_models=current["auto"]["task_router"]["task_models"],
        ),
    )

    assert updated["revision"] == "b" * 64
    assert updated["auto"]["mode"] == "priority-chain"
    assert updated["auto"]["priority_chain"] == ["github-copilot/gpt-5.6-sol"]


def test_project_registry_merges_client_trust_and_explicit_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    shared = tmp_path / "shared"
    codex_only = tmp_path / "codex-only"
    gemini_parent = tmp_path / "gemini-workspaces"
    gemini_project = gemini_parent / "gemini-project"
    added = tmp_path / "added"
    for path in (shared, codex_only, gemini_project, added):
        path.mkdir(parents=True)
        (path / ".git").mkdir()

    home.mkdir()
    (home / ".claude.json").write_text(
        json.dumps(
            {
                "projects": {
                    str(shared): {"hasTrustDialogAccepted": True},
                    str(tmp_path / "ignored"): {"hasTrustDialogAccepted": False},
                }
            }
        ),
        encoding="utf-8",
    )
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text(
        (
            f'[projects."{shared}"]\ntrust_level = "trusted"\n\n'
            f'[projects."{codex_only}"]\ntrust_level = "trusted"\n'
        ),
        encoding="utf-8",
    )
    (home / ".gemini").mkdir()
    (home / ".gemini" / "trustedFolders.json").write_text(
        f'{{\n  // Gemini CLI accepts JSONC comments\n  "{gemini_project}": "TRUST_PARENT"\n}}\n',
        encoding="utf-8",
    )
    projects_file = tmp_path / "config" / "projects.json"
    projects_file.parent.mkdir()
    projects_file.write_text(json.dumps({"projects": [str(added)]}), encoding="utf-8")

    service = PortalService(
        contexts_loader=_contexts,
        home=home,
        projects_file=projects_file,
        environment={},
    )
    projects = {project.path: project.sources for project in service.list_projects()}

    assert projects[str(shared.resolve())] == ["Claude", "Codex"]
    assert projects[str(codex_only.resolve())] == ["Codex"]
    assert projects[str(gemini_project.resolve())] == ["Gemini"]
    assert projects[str(added.resolve())] == ["Added"]


def test_add_project_persists_without_modifying_client_trust(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service = _service(tmp_path)

    added = service.add_project(str(project))

    assert added.path == str(project.resolve())
    assert "Added" in added.sources
    assert json.loads(service.projects_file.read_text(encoding="utf-8")) == {
        "projects": [str(project.resolve())]
    }


def _claude_request(
    level: Literal["user", "project"] = "user",
    project_path: str | None = None,
) -> PortalConfigRequest:
    roles = {
        role: "github-copilot/gpt-5.6-sol"
        for role in ("fable", "opus", "sonnet", "haiku", "subagent")
    }
    return PortalConfigRequest(
        context="hk",
        client="claude-code",
        level=level,
        project_path=project_path,
        main_model="github-copilot/gpt-5.6-sol",
        context_window=ContextWindowChoice.CONTEXT_1M,
        role_models=roles,
        keep_provider_prefix=True,
    )


@pytest.mark.asyncio
async def test_preview_uses_selected_context_and_redacts_api_key(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = await service.preview_config(_claude_request())

    assert result.target_path == str(service.home / ".claude" / "settings.json")
    assert not Path(result.target_path).exists()
    assert '"ANTHROPIC_AUTH_TOKEN": "********"' in result.content
    assert "sk-rm-secret" not in result.content
    assert '"ANTHROPIC_BASE_URL": "https://router.example/api/anthropic"' in result.content
    assert '"ANTHROPIC_MODEL": "github-copilot/gpt-5.6-sol[1m]"' in result.content


@pytest.mark.asyncio
async def test_apply_backs_up_and_preserves_unmanaged_claude_settings(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = service.home / ".claude" / "settings.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Read"]},
                "env": {"CUSTOM_VALUE": "keep", "ANTHROPIC_AUTH_TOKEN": "old"},
            }
        ),
        encoding="utf-8",
    )

    result = await service.apply_config(_claude_request())
    written = json.loads(target.read_text(encoding="utf-8"))

    assert result.backup_path is not None
    assert Path(result.backup_path).exists()
    assert written["permissions"] == {"allow": ["Read"]}
    assert written["env"]["CUSTOM_VALUE"] == "keep"
    assert written["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-rm-secret"
    assert "sk-rm-secret" not in result.content


@pytest.mark.asyncio
async def test_codex_preview_does_not_require_running_codex_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called():
        raise AssertionError("preview must not inspect the installed Codex catalog")

    monkeypatch.setattr(
        "router_maestro.cli.client_configs.codex._load_bundled_codex_catalog",
        fail_if_called,
    )
    service = _service(tmp_path)
    request = PortalConfigRequest(
        context="hk",
        client="codex",
        main_model="openai/gpt-5.6-sol",
    )

    result = await service.preview_config(request)

    assert 'model = "openai/gpt-5.6-sol"' in result.content
    assert 'base_url = "https://router.example/api/openai/v1"' in result.content
    assert str(service.home / ".codex" / "router-maestro-models.json") in result.content
    assert result.model_catalog_path == str(service.home / ".codex" / "router-maestro-models.json")
    assert result.model_catalog_updated is False
    assert not (service.home / ".codex" / "router-maestro-models.json").exists()


@pytest.mark.asyncio
async def test_codex_apply_refreshes_user_model_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "router_maestro.cli.client_configs.codex._load_bundled_codex_catalog",
        lambda: _stub_bundled_codex_catalog(),
    )
    service = _service(tmp_path)
    service._model_cache["hk"] = (
        service._clock(),
        [{"provider": "stale", "id": "stale/model", "name": "Stale Model"}],
    )
    request = PortalConfigRequest(
        context="hk",
        client="codex",
        main_model="openai/gpt-5.6-sol",
    )

    result = await service.apply_config(request)

    catalog_path = service.home / ".codex" / "router-maestro-models.json"
    assert result.model_catalog_path == str(catalog_path)
    assert result.model_catalog_updated is True
    assert catalog_path.exists()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert "openai/gpt-5.6-sol" in {model["slug"] for model in catalog["models"]}


@pytest.mark.asyncio
async def test_codex_apply_can_skip_existing_model_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called():
        raise AssertionError("skipped update must not inspect the Codex catalog")

    monkeypatch.setattr(
        "router_maestro.cli.client_configs.codex._load_bundled_codex_catalog",
        fail_if_called,
    )
    service = _service(tmp_path)
    catalog_path = service.home / ".codex" / "router-maestro-models.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text('{"sentinel": true}\n', encoding="utf-8")
    request = PortalConfigRequest(
        context="hk",
        client="codex",
        main_model="openai/gpt-5.6-sol",
        update_model_catalog=False,
    )

    result = await service.apply_config(request)

    assert result.model_catalog_path is None
    assert result.model_catalog_updated is False
    assert json.loads(catalog_path.read_text(encoding="utf-8")) == {"sentinel": True}


@pytest.mark.asyncio
async def test_codex_apply_reports_catalog_generation_failure_without_broken_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "router_maestro.cli.client_configs.codex._load_bundled_codex_catalog",
        lambda: None,
    )
    service = _service(tmp_path)
    request = PortalConfigRequest(
        context="hk",
        client="codex",
        main_model="openai/gpt-5.6-sol",
    )

    result = await service.apply_config(request)

    assert result.model_catalog_updated is False
    assert result.model_catalog_error is not None
    assert "model_catalog_json" not in result.content
    assert not (service.home / ".codex" / "router-maestro-models.json").exists()


@pytest.mark.asyncio
async def test_codex_project_apply_refreshes_and_points_to_shared_user_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "router_maestro.cli.client_configs.codex._load_bundled_codex_catalog",
        lambda: _stub_bundled_codex_catalog(),
    )
    service = _service(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    service.add_project(str(project))
    user_config = service.home / ".codex" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        'model_provider = "router-maestro-hk"\n\n'
        "[model_providers.router-maestro-hk]\n"
        'base_url = "https://router.example/api/openai/v1"\n',
        encoding="utf-8",
    )
    request = PortalConfigRequest(
        context="hk",
        client="codex",
        level="project",
        project_path=str(project),
        main_model="openai/gpt-5.6-sol",
    )

    result = await service.apply_config(request)

    catalog_path = service.home / ".codex" / "router-maestro-models.json"
    assert result.model_catalog_updated is True
    assert result.model_catalog_path == str(catalog_path)
    assert catalog_path.exists()
    with open(project / ".codex" / "config.toml", "rb") as file:
        project_config = tomllib.load(file)
    assert project_config["model_catalog_json"] == str(catalog_path)


@pytest.mark.asyncio
async def test_codex_project_requires_matching_user_level_context(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    service.add_project(str(project))
    request = PortalConfigRequest(
        context="hk",
        client="codex",
        level="project",
        project_path=str(project),
        main_model="openai/gpt-5.6-sol",
    )

    with pytest.raises(PortalServiceError, match="user-level Router-Maestro provider"):
        await service.preview_config(request)

    user_config = service.home / ".codex" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        'model_provider = "router-maestro-hk"\n\n'
        "[model_providers.router-maestro-hk]\n"
        'base_url = "https://router.example/api/openai/beta/v1"\n',
        encoding="utf-8",
    )

    result = await service.preview_config(request)
    assert result.target_path == str(project / ".codex" / "config.toml")
    assert 'model = "openai/gpt-5.6-sol"' in result.content
    assert (
        f'model_catalog_json = "{service.home / ".codex" / "router-maestro-models.json"}"'
        in result.content
    )


@pytest.mark.asyncio
async def test_codex_project_mismatch_explains_model_override_boundary(tmp_path: Path) -> None:
    contexts = ContextsConfig(
        current="hk",
        contexts={
            "hk": ContextConfig(endpoint="https://router.example", api_key="sk-rm-secret"),
            "jp": ContextConfig(endpoint="https://router-jp.example", api_key="sk-rm-jp"),
        },
    )
    service = _service(tmp_path, contexts_loader=lambda: contexts)
    project = tmp_path / "project"
    project.mkdir()
    service.add_project(str(project))
    user_config = service.home / ".codex" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        'model_provider = "router-maestro-jp"\n\n'
        "[model_providers.router-maestro-jp]\n"
        'base_url = "https://router-jp.example/api/openai/v1"\n',
        encoding="utf-8",
    )
    request = PortalConfigRequest(
        context="hk",
        client="codex",
        level="project",
        project_path=str(project),
        main_model="openai/gpt-5.6-sol",
    )

    with pytest.raises(PortalServiceError) as error:
        await service.preview_config(request)

    assert "can override model" in error.value.detail
    assert "context 'jp'" in error.value.detail


def test_portal_app_serves_ui_and_sensitive_key_only_on_explicit_route(tmp_path: Path) -> None:
    app = create_portal_app(_service(tmp_path))

    with TestClient(app) as client:
        page = client.get("/")
        favicon = client.get("/favicon.svg")
        meta = client.get("/api/meta")
        contexts = client.get("/api/contexts")
        key = client.get("/api/contexts/hk/key")

    assert page.status_code == 200
    assert "ROUTER-MAESTRO" in page.text
    assert "Codex Catalog" in page.text
    assert "SAVE AUTO PROFILE" in page.text
    assert "Task routing and fallback settings" in page.text
    assert "Exclude unknown models" in page.text
    assert "Smart Auto only" in page.text
    assert 'id="rm-auto-chain-list"' in page.text
    assert 'id="rm-auto-chain-add"' in page.text
    assert 'id="rm-auto-router-model-meta"' in page.text
    assert 'data-auto-task-meta="fast"' in page.text
    assert 'data-auto-task-meta="deep_reasoning"' in page.text
    assert 'className = "rm-auto-chain-choice"' in page.text
    assert "+ ADD FALLBACK" in page.text
    assert 'id="rm-auto-chain"' not in page.text
    assert 'option.textContent = model.name + " // " + model.provider_name' not in page.text
    assert "option.textContent = model.name;" in page.text
    assert 'model.virtual ? " rm-model-row--auto" : ""' in page.text
    assert 'model.virtual ? " rm-model-name--auto" : ""' in page.text
    assert 'internalBadge.textContent = "INTERNAL"' in page.text
    assert 'internalBadge.className = "rm-model-badge--internal"' in page.text
    assert "@keyframes rm-auto-name-flow" in page.text
    assert "@keyframes rm-auto-row-flow" in page.text
    assert "grid-template-rows 760ms" in page.text
    assert "transition-duration: 520ms" in page.text
    assert ".rm-auto-editor.is-open:not(.is-revealing)" in page.text
    assert "> .rm-auto-editor-content" in page.text
    assert "overflow: visible" in page.text
    assert "@keyframes rm-auto-editor-pulse" in page.text
    assert "@keyframes rm-auto-editor-flare" in page.text
    assert 'autoRouting.classList.add("is-open", "is-revealing")' in page.text
    assert 'autoRouting.classList.add("is-collapsing")' in page.text
    assert 'autoRouting.addEventListener("transitionend"' in page.text
    assert 'event.propertyName !== "grid-template-rows"' in page.text
    assert "function setAutoRoutingVisible(visible)" in page.text
    assert "function syncAutoRouting()" in page.text
    assert 'autoRouting.addEventListener("animationend"' in page.text
    assert 'window.matchMedia("(prefers-reduced-motion: reduce)")' in page.text
    assert "targetPath.textContent = result.target_path" not in page.text
    assert 'windowOption.max_prompt_tokens > 900000 ? " extended" : " standard"' in page.text
    assert 'windowOption.max_prompt_tokens > 900000 ? " [1m]" : " standard"' not in page.text
    assert "@media (prefers-reduced-motion: reduce)" in page.text
    assert "rm-catalog-toggle" in page.text
    assert '<link rel="icon" href="/favicon.svg" type="image/svg+xml">' in page.text
    assert favicon.status_code == 200
    assert favicon.headers["content-type"].startswith("image/svg+xml")
    assert "Router-Maestro Route-M" in favicon.text
    assert "LOOPBACK ONLY" in page.text
    assert page.headers["X-Frame-Options"] == "DENY"
    assert meta.json()["version"]
    assert contexts.json() == [
        {
            "name": "hk",
            "endpoint": "https://router.example",
            "current": True,
            "has_api_key": True,
        }
    ]
    assert "sk-rm-secret" not in contexts.text
    assert key.json() == {"api_key": "sk-rm-secret"}
    assert key.headers["Cache-Control"] == "no-store"


def test_portal_rejects_non_loopback_host_header(tmp_path: Path) -> None:
    app = create_portal_app(_service(tmp_path))
    with TestClient(app) as client:
        response = client.get("/", headers={"Host": "attacker.example"})
    assert response.status_code == 400
