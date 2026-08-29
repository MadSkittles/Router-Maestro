"""Backend services for the loopback-only Router-Maestro web portal."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import threading
import time
import tomllib
from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from router_maestro.cli.client_configs import get_client
from router_maestro.cli.client_configs.base import (
    ContextWindowChoice,
    GenerateContext,
    IdStyle,
    ModelSelection,
    _bare_upstream_model_id,
    _context_windows_label,
    _format_token_count,
    _model_key,
)
from router_maestro.cli.client_configs.claude_code import _catalog_has_claude_model
from router_maestro.config import PROJECTS_FILE, ContextConfig, ContextsConfig, load_contexts_config
from router_maestro.config.settings import write_json_owner_only

PortalClient = Literal["claude-code", "codex", "gemini"]
PortalLevel = Literal["user", "project"]

_CLAUDE_ROLE_SLOTS = ("fable", "opus", "sonnet", "haiku", "subagent")
_SOURCE_ORDER = {"Claude": 0, "Codex": 1, "Gemini": 2, "Added": 3}
_PROJECT_MARKERS = (
    ".git",
    ".claude",
    ".codex",
    ".gemini",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)


class PortalServiceError(RuntimeError):
    """A user-facing portal failure with an HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class PortalContext(BaseModel):
    """One locally configured Router-Maestro deployment context."""

    name: str
    endpoint: str
    current: bool
    has_api_key: bool


class PortalHealth(BaseModel):
    """Liveness and round-trip timing for one context."""

    healthy: bool
    latency_ms: int
    status_code: int | None = None
    error: str | None = None


class PortalContextWindow(BaseModel):
    """One context choice advertised by a model catalog."""

    tier: str
    max_prompt_tokens: int
    label: str
    is_default: bool = False


class PortalModel(BaseModel):
    """Portal-friendly projection of an admin model-catalog entry."""

    key: str
    upstream_id: str
    name: str
    provider: str
    provider_name: str
    context_label: str
    context_windows: list[PortalContextWindow] = Field(default_factory=list)
    transports: list[str] = Field(default_factory=list)
    virtual: bool = False


class PortalModels(BaseModel):
    """Context-scoped model catalog."""

    models: list[PortalModel] = Field(default_factory=list)
    requires_claude_role_mappings: bool


class PortalProject(BaseModel):
    """One project discovered from client trust stores or explicitly added."""

    path: str
    name: str
    sources: list[str]


class PortalConfigRequest(BaseModel):
    """Config selection submitted by the portal UI."""

    context: str
    client: PortalClient
    level: PortalLevel = "user"
    project_path: str | None = None
    main_model: str
    context_window: ContextWindowChoice = ContextWindowChoice.DEFAULT
    role_models: dict[str, str] = Field(default_factory=dict)
    keep_provider_prefix: bool = True
    update_model_catalog: bool = True


class PortalConfigResult(BaseModel):
    """Preview or completed config write."""

    target_path: str
    content: str
    backup_path: str | None = None
    model_catalog_path: str | None = None
    model_catalog_updated: bool = False
    model_catalog_error: str | None = None


class PortalAutoConfigRequest(BaseModel):
    """Versioned Auto-profile replacement sent through the local portal."""

    revision: str
    mode: Literal["task-router", "priority-chain"]
    capability_policy: Literal["strict", "optimistic"] = "strict"
    priority_chain: list[str] = Field(default_factory=list)
    router_model: str
    task_models: dict[str, str]


class PortalService:
    """Coordinate local context access, project discovery, and config writes."""

    def __init__(
        self,
        *,
        contexts_loader: Callable[[], ContextsConfig] = load_contexts_config,
        home: Path | None = None,
        projects_file: Path = PROJECTS_FILE,
        environment: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        model_cache_ttl: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._contexts_loader = contexts_loader
        self.home = (home or Path.home()).expanduser().resolve(strict=False)
        self.projects_file = projects_file
        self.environment = dict(os.environ if environment is None else environment)
        self._transport = transport
        self._model_cache_ttl = model_cache_ttl
        self._clock = clock
        self._model_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._model_cache_lock = asyncio.Lock()
        self._projects_lock = threading.RLock()

    def list_contexts(self) -> list[PortalContext]:
        """Return configured contexts without exposing API key material."""
        config = self._contexts_loader()
        return [
            PortalContext(
                name=name,
                endpoint=context.endpoint.rstrip("/"),
                current=name == config.current,
                has_api_key=bool(context.api_key),
            )
            for name, context in config.contexts.items()
        ]

    def get_context(self, name: str) -> ContextConfig:
        """Resolve one context or raise a stable client error."""
        context = self._contexts_loader().contexts.get(name)
        if context is None:
            raise PortalServiceError(404, f"Context '{name}' was not found")
        return context

    def get_api_key(self, name: str) -> str:
        """Return one context key only for the explicit clipboard action."""
        api_key = self.get_context(name).api_key
        if not api_key:
            raise PortalServiceError(404, f"Context '{name}' has no API key")
        return api_key

    def _http_client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            follow_redirects=False,
        )

    async def health(self, name: str) -> PortalHealth:
        """Measure unauthenticated ``/health`` liveness RTT for one context."""
        context = self.get_context(name)
        started = self._clock()
        try:
            async with self._http_client(5.0) as client:
                response = await client.get(f"{context.endpoint.rstrip('/')}/health")
            latency_ms = round((self._clock() - started) * 1000)
            healthy_payload = False
            if response.status_code == 200:
                try:
                    body = response.json()
                    healthy_payload = isinstance(body, dict) and body.get("status") == "healthy"
                except ValueError:
                    healthy_payload = False
            return PortalHealth(
                healthy=response.status_code == 200 and healthy_payload,
                latency_ms=max(latency_ms, 0),
                status_code=response.status_code,
                error=None if response.status_code == 200 else f"HTTP {response.status_code}",
            )
        except httpx.HTTPError as error:
            latency_ms = round((self._clock() - started) * 1000)
            return PortalHealth(
                healthy=False,
                latency_ms=max(latency_ms, 0),
                error=type(error).__name__,
            )

    @staticmethod
    def _headers(context: ContextConfig) -> dict[str, str]:
        if context.api_key:
            return {"Authorization": f"Bearer {context.api_key}"}
        return {}

    async def _load_raw_models(self, name: str, *, force_refresh: bool = False) -> list[dict]:
        context = self.get_context(name)
        now = self._clock()
        cached = self._model_cache.get(name)
        if not force_refresh and cached is not None and now - cached[0] < self._model_cache_ttl:
            return cached[1]

        async with self._model_cache_lock:
            now = self._clock()
            cached = self._model_cache.get(name)
            if not force_refresh and cached is not None and now - cached[0] < self._model_cache_ttl:
                return cached[1]
            try:
                async with self._http_client(20.0) as client:
                    response = await client.get(
                        f"{context.endpoint.rstrip('/')}/api/admin/models",
                        headers=self._headers(context),
                    )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPStatusError as error:
                raise PortalServiceError(
                    502,
                    f"Context '{name}' model catalog returned HTTP {error.response.status_code}",
                ) from error
            except (httpx.HTTPError, ValueError) as error:
                raise PortalServiceError(
                    502,
                    f"Context '{name}' model catalog is unavailable",
                ) from error

            raw_models = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(raw_models, list):
                raise PortalServiceError(502, f"Context '{name}' returned an invalid model catalog")
            models = [model for model in raw_models if isinstance(model, dict)]
            self._model_cache[name] = (self._clock(), models)
            return models

    async def get_auto_config(self, name: str) -> dict[str, Any]:
        """Return the selected server's revisioned Auto configuration."""
        context = self.get_context(name)
        try:
            async with self._http_client(10.0) as client:
                response = await client.get(
                    f"{context.endpoint.rstrip('/')}/api/admin/priorities",
                    headers=self._headers(context),
                )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise PortalServiceError(502, f"Context '{name}' Auto config is unavailable") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("revision"), str):
            raise PortalServiceError(502, f"Context '{name}' returned an invalid Auto config")
        return payload

    async def update_auto_config(
        self, name: str, request: PortalAutoConfigRequest
    ) -> dict[str, Any]:
        """CAS-update only the Auto profile while preserving all other runtime settings."""
        context = self.get_context(name)
        current = await self.get_auto_config(name)
        if current["revision"] != request.revision:
            raise PortalServiceError(409, "Runtime configuration changed; reload Auto settings")
        task_router = current.get("auto", {}).get("task_router", {})
        replacement = {key: value for key, value in current.items() if key != "revision"}
        replacement["auto"] = {
            "mode": request.mode,
            "capability_policy": request.capability_policy,
            "priority_chain": request.priority_chain,
            "task_router": {
                "router_model": request.router_model or task_router.get("router_model"),
                "task_models": request.task_models or task_router.get("task_models", {}),
            },
        }
        try:
            async with self._http_client(20.0) as client:
                response = await client.patch(
                    f"{context.endpoint.rstrip('/')}/api/admin/priorities",
                    headers=self._headers(context),
                    json={**replacement, "revision": request.revision},
                )
            if response.status_code == 409:
                raise PortalServiceError(409, "Runtime configuration changed; reload Auto settings")
            response.raise_for_status()
            payload = response.json()
        except PortalServiceError:
            raise
        except (httpx.HTTPError, ValueError) as error:
            raise PortalServiceError(502, f"Context '{name}' Auto config update failed") from error
        if not isinstance(payload, dict):
            raise PortalServiceError(502, f"Context '{name}' returned an invalid Auto config")
        self._model_cache.pop(name, None)
        return payload

    @staticmethod
    def _provider_name(provider: str) -> str:
        special = {
            "github-copilot": "GitHub Copilot",
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "gemini": "Gemini",
        }
        return special.get(provider, provider.replace("-", " ").title())

    @staticmethod
    def _model_name(model: dict) -> str:
        raw_name = str(model.get("name") or _bare_upstream_model_id(model))
        words = [word for word in re.split(r"[-_\s]+", raw_name) if word]
        acronyms = {"ai": "AI", "gpt": "GPT", "mai": "MAI"}
        rendered: list[str] = []
        for word in words:
            lowered = word.casefold()
            if lowered in acronyms:
                rendered.append(acronyms[lowered])
            elif re.fullmatch(r"o\d+", lowered):
                rendered.append(lowered)
            elif word[0].isdigit():
                rendered.append(word.lower())
            elif word.isupper() and len(word) <= 5:
                rendered.append(word)
            else:
                rendered.append(word[0].upper() + word[1:].lower())
        if len(rendered) >= 2 and rendered[0] == "GPT" and rendered[1][0].isdigit():
            rendered[0:2] = [f"{rendered[0]}-{rendered[1]}"]
        return " ".join(rendered)

    @staticmethod
    def _transport_names(model: dict) -> list[str]:
        if model.get("virtual") is True:
            # Auto aggregates execution-model capabilities. Keep this summary
            # stable even when the current profile happens to contain only one
            # native transport; the concrete request still selects its actual
            # binding at dispatch time.
            return ["Responses", "Chat", "Messages"]
        capabilities = model.get("operation_capabilities")
        if not isinstance(capabilities, dict):
            return []
        transports: list[str] = []
        if capabilities.get("native_anthropic") is True:
            transports.append("Messages")
        if capabilities.get("responses") is True or capabilities.get("responses_stream") is True:
            transports.append("Responses")
        if capabilities.get("chat") is True or capabilities.get("chat_stream") is True:
            transports.append("Chat")
        return transports

    @staticmethod
    def _context_windows(model: dict) -> list[PortalContextWindow]:
        options = model.get("context_window_options")
        if not isinstance(options, list):
            return []
        result: list[PortalContextWindow] = []
        for option in options:
            if not isinstance(option, dict):
                continue
            tokens = option.get("max_prompt_tokens")
            tier = option.get("tier")
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
                continue
            if not isinstance(tier, str) or not tier:
                tier = "default"
            result.append(
                PortalContextWindow(
                    tier=tier,
                    max_prompt_tokens=tokens,
                    label=_format_token_count(tokens),
                    is_default=option.get("is_default") is True,
                )
            )
        return result

    async def list_models(self, name: str, *, force_refresh: bool = False) -> PortalModels:
        """Return the selected context's provider-qualified model catalog."""
        raw_models = await self._load_raw_models(name, force_refresh=force_refresh)
        models = [
            PortalModel(
                key=_model_key(model),
                upstream_id=_bare_upstream_model_id(model),
                name=self._model_name(model),
                provider=str(model.get("provider") or "unknown"),
                provider_name=self._provider_name(str(model.get("provider") or "unknown")),
                context_label=_context_windows_label(model),
                context_windows=self._context_windows(model),
                transports=self._transport_names(model),
                virtual=model.get("virtual") is True,
            )
            for model in raw_models
            if isinstance(model.get("id"), str) and isinstance(model.get("provider"), str)
        ]
        return PortalModels(
            models=models,
            requires_claude_role_mappings=not _catalog_has_claude_model(raw_models),
        )

    @staticmethod
    def _strip_json_comments(content: str) -> str:
        """Remove JSONC comments while preserving comment markers inside strings."""
        result: list[str] = []
        index = 0
        in_string = False
        escaped = False
        while index < len(content):
            char = content[index]
            following = content[index + 1] if index + 1 < len(content) else ""
            if in_string:
                result.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                index += 1
                continue
            if char == '"':
                in_string = True
                result.append(char)
                index += 1
                continue
            if char == "/" and following == "/":
                index += 2
                while index < len(content) and content[index] not in "\r\n":
                    index += 1
                continue
            if char == "/" and following == "*":
                index += 2
                while index + 1 < len(content) and content[index : index + 2] != "*/":
                    index += 1
                index = min(index + 2, len(content))
                continue
            result.append(char)
            index += 1
        return "".join(result)

    def _read_json(self, path: Path) -> object:
        try:
            content = path.read_text(encoding="utf-8")
            return json.loads(self._strip_json_comments(content))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _trusted_roots(self) -> dict[Path, set[str]]:
        roots: dict[Path, set[str]] = defaultdict(set)

        claude = self._read_json(self.home / ".claude.json")
        claude_projects = claude.get("projects") if isinstance(claude, dict) else None
        if isinstance(claude_projects, dict):
            for raw_path, settings in claude_projects.items():
                if (
                    isinstance(raw_path, str)
                    and isinstance(settings, dict)
                    and settings.get("hasTrustDialogAccepted") is True
                ):
                    roots[Path(raw_path).expanduser()].add("Claude")

        codex_path = self.home / ".codex" / "config.toml"
        try:
            with open(codex_path, "rb") as file:
                codex = tomllib.load(file)
        except (OSError, tomllib.TOMLDecodeError):
            codex = {}
        codex_projects = codex.get("projects") if isinstance(codex, dict) else None
        if isinstance(codex_projects, dict):
            for raw_path, settings in codex_projects.items():
                if (
                    isinstance(raw_path, str)
                    and isinstance(settings, dict)
                    and settings.get("trust_level") == "trusted"
                ):
                    roots[Path(raw_path).expanduser()].add("Codex")

        gemini_override = self.environment.get("GEMINI_CLI_TRUSTED_FOLDERS_PATH")
        gemini_path = (
            Path(gemini_override).expanduser()
            if gemini_override
            else self.home / ".gemini" / "trustedFolders.json"
        )
        gemini = self._read_json(gemini_path)
        if isinstance(gemini, dict) and isinstance(gemini.get("config"), dict):
            gemini = gemini["config"]
        if isinstance(gemini, dict):
            for raw_path, trust_level in gemini.items():
                if not isinstance(raw_path, str):
                    continue
                if trust_level == "TRUST_FOLDER":
                    roots[Path(raw_path).expanduser()].add("Gemini")
                elif trust_level == "TRUST_PARENT":
                    roots[Path(raw_path).expanduser().parent].add("Gemini")

        return roots

    def _explicit_projects(self) -> list[Path]:
        data = self._read_json(self.projects_file)
        projects = data.get("projects") if isinstance(data, dict) else None
        if not isinstance(projects, list):
            return []
        return [Path(path).expanduser() for path in projects if isinstance(path, str)]

    @staticmethod
    def _canonical_directory(path: Path) -> Path | None:
        if not path.is_absolute():
            return None
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return None
        return resolved if resolved.is_dir() else None

    @staticmethod
    def _looks_like_project(path: Path) -> bool:
        return any((path / marker).exists() for marker in _PROJECT_MARKERS)

    def _expand_trusted_root(self, path: Path) -> list[Path]:
        canonical = self._canonical_directory(path)
        if canonical is None:
            return []
        if self._looks_like_project(canonical):
            return [canonical]
        children: list[Path] = []
        try:
            candidates = sorted(canonical.iterdir(), key=lambda item: item.name.casefold())[:200]
        except OSError:
            return [canonical]
        for child in candidates:
            resolved = self._canonical_directory(child)
            if resolved is not None and self._looks_like_project(resolved):
                children.append(resolved)
        return children or [canonical]

    def list_projects(self) -> list[PortalProject]:
        """Merge Claude, Codex, Gemini trust stores with explicitly added paths."""
        discovered: dict[Path, set[str]] = defaultdict(set)
        for root, sources in self._trusted_roots().items():
            for project in self._expand_trusted_root(root):
                discovered[project].update(sources)
        for path in self._explicit_projects():
            canonical = self._canonical_directory(path)
            if canonical is not None:
                discovered[canonical].add("Added")

        return [
            PortalProject(
                path=str(path),
                name=path.name or str(path),
                sources=sorted(sources, key=lambda source: _SOURCE_ORDER[source]),
            )
            for path, sources in sorted(
                discovered.items(),
                key=lambda item: (item[0].name.casefold(), str(item[0]).casefold()),
            )
        ]

    def add_project(self, raw_path: str) -> PortalProject:
        """Persist one explicit project path without modifying client trust stores."""
        path = Path(raw_path).expanduser()
        canonical = self._canonical_directory(path)
        if canonical is None:
            raise PortalServiceError(400, "Project path must be an existing absolute directory")
        with self._projects_lock:
            existing = self._explicit_projects()
            canonical_existing = {
                resolved
                for item in existing
                if (resolved := self._canonical_directory(item)) is not None
            }
            if canonical not in canonical_existing:
                write_json_owner_only(
                    self.projects_file,
                    {
                        "projects": [
                            str(path)
                            for path in sorted(
                                [*canonical_existing, canonical],
                                key=lambda item: str(item).casefold(),
                            )
                        ]
                    },
                )
        project = next(
            (item for item in self.list_projects() if item.path == str(canonical)),
            None,
        )
        if project is None:
            raise PortalServiceError(500, "Added project could not be loaded")
        return project

    def _validate_project(self, raw_path: str | None) -> Path:
        if raw_path is None:
            raise PortalServiceError(400, "A project path is required for project-level config")
        canonical = self._canonical_directory(Path(raw_path).expanduser())
        if canonical is None:
            raise PortalServiceError(400, "Project path must be an existing absolute directory")
        allowed = {Path(project.path) for project in self.list_projects()}
        if canonical not in allowed:
            raise PortalServiceError(403, "Project path is not trusted or explicitly added")
        return canonical

    def _target_path(self, request: PortalConfigRequest) -> Path:
        user_paths = {
            "claude-code": self.home / ".claude" / "settings.json",
            "codex": self.home / ".codex" / "config.toml",
            "gemini": self.home / ".gemini" / ".env",
        }
        project_paths = {
            "claude-code": Path(".claude/settings.json"),
            "codex": Path(".codex/config.toml"),
            "gemini": Path(".gemini/.env"),
        }
        if request.level == "user":
            return user_paths[request.client]
        return self._validate_project(request.project_path) / project_paths[request.client]

    def _validate_codex_project_context(self, context: ContextConfig) -> None:
        """Ensure project-local Codex model override inherits the selected context."""
        path = self.home / ".codex" / "config.toml"
        try:
            with open(path, "rb") as file:
                config = tomllib.load(file)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise PortalServiceError(
                409,
                "Codex project config requires a user-level Router-Maestro provider first",
            ) from error
        providers = config.get("model_providers")
        provider_name = config.get("model_provider")
        provider = (
            providers.get(provider_name)
            if isinstance(providers, dict) and isinstance(provider_name, str)
            else None
        )
        base_url = provider.get("base_url") if isinstance(provider, dict) else None
        if not isinstance(provider_name, str) or not isinstance(base_url, str):
            raise PortalServiceError(
                409,
                "Codex project config requires a user-level Router-Maestro provider first",
            )
        inherited_endpoint = base_url.rstrip("/")
        for suffix in ("/api/openai/beta/v1", "/api/openai/v1"):
            if inherited_endpoint.endswith(suffix):
                inherited_endpoint = inherited_endpoint[: -len(suffix)]
                break
        selected_endpoint = context.endpoint.rstrip("/")
        if inherited_endpoint != selected_endpoint:
            inherited_context = next(
                (
                    name
                    for name, candidate in self._contexts_loader().contexts.items()
                    if candidate.endpoint.rstrip("/") == inherited_endpoint
                ),
                inherited_endpoint,
            )
            raise PortalServiceError(
                409,
                "Codex project config can override model, but it inherits user-level provider "
                f"'{provider_name}' from context '{inherited_context}'. Select that context or "
                "change the user-level Codex provider first",
            )

    @staticmethod
    def _supports_context(model: dict, choice: ContextWindowChoice) -> bool:
        if choice is not ContextWindowChoice.CONTEXT_1M:
            return True
        options = model.get("context_window_options")
        if not isinstance(options, list) or not options:
            return True
        return any(
            isinstance(option, dict)
            and isinstance(option.get("max_prompt_tokens"), int)
            and option["max_prompt_tokens"] > 900_000
            for option in options
        )

    def _prepare_writer(
        self,
        request: PortalConfigRequest,
        raw_models: list[dict],
        *,
        target_path: Path,
        preview_only: bool,
    ) -> tuple[Any, dict[str, str], GenerateContext]:
        model_by_key = {_model_key(model): model for model in raw_models}
        main = model_by_key.get(request.main_model)
        if main is None:
            raise PortalServiceError(400, "Selected main model is not in this context's catalog")
        if not self._supports_context(main, request.context_window):
            raise PortalServiceError(
                400,
                "Selected main model does not support the requested context",
            )

        selections = [
            ModelSelection(
                slot="main",
                model=main,
                context_window=request.context_window,
            )
        ]
        if request.client == "claude-code" and not _catalog_has_claude_model(raw_models):
            missing = [slot for slot in _CLAUDE_ROLE_SLOTS if slot not in request.role_models]
            if missing:
                raise PortalServiceError(
                    400,
                    f"Claude role mappings are required: {', '.join(missing)}",
                )
            for slot in _CLAUDE_ROLE_SLOTS:
                model = model_by_key.get(request.role_models[slot])
                if model is None:
                    raise PortalServiceError(400, f"Claude {slot} model is not in the catalog")
                if not self._supports_context(model, request.context_window):
                    raise PortalServiceError(
                        400,
                        f"Claude {slot} model does not support the requested context",
                    )
                selections.append(
                    ModelSelection(
                        slot=slot,
                        model=model,
                        context_window=request.context_window,
                    )
                )

        id_style = IdStyle.QUALIFIED if request.keep_provider_prefix else IdStyle.BARE
        client = get_client(request.client)()
        context = self.get_context(request.context)
        if request.client == "codex" and request.level == "project":
            self._validate_codex_project_context(context)
        generation = GenerateContext(
            id_style=id_style,
            selections=tuple(selections),
            extras={
                "preview_only": preview_only,
                "target_path": str(target_path),
                "update_model_catalog": request.update_model_catalog,
                "model_catalog_path": str(self.home / ".codex" / "router-maestro-models.json"),
            },
            endpoint=context.endpoint,
            api_key=context.api_key,
        )
        model_strings = {
            selection.slot: client.resolve_model_selection(selection, id_style)
            for selection in selections
        }
        if request.client == "codex":
            setattr(client, "_available_models", raw_models)
        return client, model_strings, generation

    @staticmethod
    def _redact_preview(client: PortalClient, content: str) -> str:
        if client == "claude-code":
            try:
                document = json.loads(content)
            except json.JSONDecodeError:
                return content
            env = document.get("env") if isinstance(document, dict) else None
            if isinstance(env, dict) and "ANTHROPIC_AUTH_TOKEN" in env:
                env["ANTHROPIC_AUTH_TOKEN"] = "********"
            return json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        if client == "gemini":
            lines = []
            for line in content.splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    line = "GEMINI_API_KEY=********"
                lines.append(line)
            return "\n".join(lines) + ("\n" if content.endswith("\n") else "")
        return content

    def _preview_sync(
        self,
        request: PortalConfigRequest,
        raw_models: list[dict],
        target_path: Path,
    ) -> PortalConfigResult:
        with tempfile.TemporaryDirectory(prefix="router-maestro-portal-") as temp_dir:
            temp_target = Path(temp_dir) / target_path.name
            client, models, generation = self._prepare_writer(
                request,
                raw_models,
                target_path=target_path,
                preview_only=True,
            )
            client.write(level=request.level, path=temp_target, models=models, ctx=generation)
            content = temp_target.read_text(encoding="utf-8")
        return PortalConfigResult(
            target_path=str(target_path),
            content=self._redact_preview(request.client, content),
            model_catalog_path=(
                str(self.home / ".codex" / "router-maestro-models.json")
                if request.client == "codex" and request.update_model_catalog
                else None
            ),
        )

    @staticmethod
    def _backup(path: Path) -> Path | None:
        if not path.exists():
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = path.with_suffix(f"{path.suffix}.backup.{timestamp}")
        shutil.copy2(path, backup)
        return backup

    def _apply_sync(
        self,
        request: PortalConfigRequest,
        raw_models: list[dict],
        target_path: Path,
    ) -> PortalConfigResult:
        client, models, generation = self._prepare_writer(
            request,
            raw_models,
            target_path=target_path,
            preview_only=False,
        )
        backup = self._backup(target_path)
        client.write(level=request.level, path=target_path, models=models, ctx=generation)
        content = target_path.read_text(encoding="utf-8")
        return PortalConfigResult(
            target_path=str(target_path),
            content=self._redact_preview(request.client, content),
            backup_path=str(backup) if backup is not None else None,
            model_catalog_path=(
                str(generation.extras.get("model_catalog_path"))
                if request.client == "codex" and request.update_model_catalog
                else None
            ),
            model_catalog_updated=generation.extras.get("model_catalog_updated") is True,
            model_catalog_error=(
                str(generation.extras["model_catalog_error"])
                if "model_catalog_error" in generation.extras
                else None
            ),
        )

    async def preview_config(self, request: PortalConfigRequest) -> PortalConfigResult:
        """Render the exact config change without touching the destination."""
        target = self._target_path(request)
        raw_models = await self._load_raw_models(request.context)
        return await asyncio.to_thread(self._preview_sync, request, raw_models, target)

    async def apply_config(self, request: PortalConfigRequest) -> PortalConfigResult:
        """Back up and write one selected client configuration."""
        target = self._target_path(request)
        raw_models = await self._load_raw_models(
            request.context,
            force_refresh=request.client == "codex" and request.update_model_catalog,
        )
        return await asyncio.to_thread(self._apply_sync, request, raw_models, target)
