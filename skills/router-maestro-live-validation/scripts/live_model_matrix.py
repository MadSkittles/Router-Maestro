#!/usr/bin/env python3
"""Run repeatable Claude Code and Codex validation against one RM context.

The runner deliberately separates automated model coverage from the interactive
file/MCP rounds documented by the skill. ``smoke`` sends one request per model;
``recall`` creates a fresh persisted client session and verifies a second request
can recover an opaque token from the first turn.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from router_maestro.cli.client_configs.codex import _build_codex_model_catalog
from router_maestro.config import load_contexts_config
from router_maestro.config.settings import write_json_owner_only

CLIENTS = ("claude", "codex")
PHASES = ("smoke", "recall")
CODEX_PROVIDER_NAME = "router-maestro-live"
FALLBACK_METADATA_WARNING = "defaulting to fallback metadata"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CAPSULE_PATTERN = re.compile(r"rmr1\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
HTTP_400_PATTERN = re.compile(
    r"(?:status(?:_code)?\D{0,8}|api error:\s*|http(?:/\d(?:\.\d)?)?\s+)400\b",
    re.I,
)
HTTP_5XX_PATTERN = re.compile(
    r"(?:status(?:_code)?\D{0,8}|api error:\s*|http(?:/\d(?:\.\d)?)?\s+)"
    r"(500|502|503|504)\b",
    re.I,
)
SENSITIVE_LOG_FIELDS = frozenset(
    {
        "api_key",
        "authorization",
        "capsule",
        "encrypted_content",
        "reasoning_opaque",
        "signature",
        "thoughtsignature",
    }
)


class ValidationError(RuntimeError):
    """A safe user-facing validation setup error."""


@dataclass(frozen=True)
class TargetContext:
    name: str
    endpoint: str
    api_key: str | None

    @property
    def auth_value(self) -> str:
        return self.api_key or "router-maestro-live-validation"


@dataclass(frozen=True)
class CatalogModel:
    wire_id: str
    provider: str
    bare_id: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


@dataclass
class CaseAttempt:
    passed: bool
    expected: str
    actual: str
    processes: list[ProcessResult]
    detail: str

    @property
    def transient_5xx(self) -> bool:
        text = "\n".join(process.combined_output for process in self.processes)
        return is_explicit_transient_5xx(text)


@dataclass
class CaseResult:
    client: str
    phase: str
    model: str
    provider: str
    passed: bool
    attempts: int
    duration_seconds: float
    detail: str
    attempt_logs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Runtime:
    target: TargetContext
    project: Path
    output_dir: Path
    claude_config_dir: Path
    codex_home: Path
    codex_catalog_path: Path | None
    timeout_seconds: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dynamic Router-Maestro model smoke and two-turn recall checks through "
            "Claude Code and Codex."
        )
    )
    parser.add_argument(
        "--context",
        help="Router-Maestro context name (defaults to the current context).",
    )
    parser.add_argument(
        "--client",
        choices=(*CLIENTS, "all"),
        default="all",
        help="Client to validate (default: all).",
    )
    parser.add_argument(
        "--phase",
        choices=(*PHASES, "all"),
        default="all",
        help="Validation phase (default: all).",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Keep one provider; repeat to select several.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Keep one exact qualified or bare model ID; repeat to select several.",
    )
    parser.add_argument(
        "--model-pattern",
        action="append",
        default=[],
        help="Keep model IDs matching this shell-style glob; repeat to select several.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        help="Cap the selected model list after sorting (useful for a bounded canary).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Trusted project directory used as the client working directory.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="Per client invocation timeout in seconds (default: 240).",
    )
    parser.add_argument(
        "--transient-retries",
        type=int,
        default=1,
        help="Retries for failures containing an explicit HTTP 500/502/503/504 (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write summaries and sanitized logs here; implies retention.",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="Retain a secure temporary output directory instead of deleting it.",
    )
    return parser.parse_args(argv)


def resolve_target_context(name: str | None) -> TargetContext:
    config = load_contexts_config()
    context_name = name or config.current
    context = config.contexts.get(context_name)
    if context is None:
        available = ", ".join(sorted(config.contexts)) or "none"
        raise ValidationError(f"Unknown context {context_name!r}; available contexts: {available}")
    endpoint = context.endpoint.rstrip("/")
    if not endpoint:
        raise ValidationError(f"Context {context_name!r} has an empty endpoint")
    return TargetContext(context_name, endpoint, context.api_key)


def fetch_server_catalog(
    target: TargetContext,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[CatalogModel]]:
    headers = {}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    timeout = min(timeout_seconds, 60.0)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            health_response = client.get(f"{target.endpoint}/health")
            health_response.raise_for_status()
            root_response = client.get(f"{target.endpoint}/")
            root_response.raise_for_status()
            models_response = client.get(
                f"{target.endpoint}/api/openai/v1/models",
                headers=headers,
            )
            models_response.raise_for_status()
    except httpx.HTTPError as error:
        raise ValidationError(
            f"Could not query context {target.name!r} at {target.endpoint}: {error}"
        ) from error

    root = _json_mapping(root_response, label="root")
    health = {**root, **_json_mapping(health_response, label="health")}
    payload = _json_mapping(models_response, label="model catalog")
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise ValidationError("Model catalog response does not contain a data list")

    models: list[CatalogModel] = []
    for index, value in enumerate(raw_models):
        if not isinstance(value, dict):
            raise ValidationError(f"Model catalog entry {index} is not an object")
        wire_id = value.get("id")
        provider = value.get("owned_by")
        if not isinstance(wire_id, str) or not wire_id:
            raise ValidationError(f"Model catalog entry {index} has no valid id")
        if not isinstance(provider, str) or not provider:
            raise ValidationError(f"Model catalog entry {index} has no valid owned_by")
        prefix = f"{provider}/"
        bare_id = wire_id[len(prefix) :] if wire_id.startswith(prefix) else wire_id
        models.append(CatalogModel(wire_id, provider, bare_id, dict(value)))
    if not models:
        raise ValidationError("The selected context returned no models")
    return health, models


def _json_mapping(response: httpx.Response, *, label: str) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as error:
        raise ValidationError(f"The {label} response is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValidationError(f"The {label} response is not a JSON object")
    return value


def select_models(
    models: Sequence[CatalogModel],
    *,
    providers: Sequence[str],
    exact_models: Sequence[str],
    patterns: Sequence[str],
    max_models: int | None,
) -> list[CatalogModel]:
    if max_models is not None and max_models <= 0:
        raise ValidationError("--max-models must be positive")

    provider_filter = {value.casefold() for value in providers}
    exact_filter = {value.casefold() for value in exact_models}
    selected: list[CatalogModel] = []
    matched_exact: set[str] = set()
    matched_patterns: set[str] = set()

    for model in models:
        if provider_filter and model.provider.casefold() not in provider_filter:
            continue
        qualified = model.wire_id.casefold()
        bare = model.bare_id.casefold()
        exact_matches = {value for value in exact_filter if value in {qualified, bare}}
        pattern_matches = {
            pattern
            for pattern in patterns
            if fnmatch.fnmatchcase(model.wire_id, pattern)
            or fnmatch.fnmatchcase(model.bare_id, pattern)
        }
        if (exact_filter or patterns) and not (exact_matches or pattern_matches):
            continue
        matched_exact.update(exact_matches)
        matched_patterns.update(pattern_matches)
        selected.append(model)

    missing_exact = sorted(exact_filter - matched_exact)
    missing_patterns = sorted(set(patterns) - matched_patterns)
    if missing_exact:
        raise ValidationError(f"Exact model filters matched nothing: {', '.join(missing_exact)}")
    if missing_patterns:
        raise ValidationError(f"Model patterns matched nothing: {', '.join(missing_patterns)}")
    if providers:
        matched_providers = {model.provider.casefold() for model in selected}
        missing_providers = sorted(provider_filter - matched_providers)
        if missing_providers:
            raise ValidationError(
                f"Provider filters matched no selected model: {', '.join(missing_providers)}"
            )

    selected.sort(key=lambda model: (model.provider.casefold(), model.wire_id.casefold()))
    if max_models is not None:
        selected = selected[:max_models]
    if not selected:
        raise ValidationError("No models remain after applying filters")
    return selected


def codex_catalog_inputs(models: Sequence[CatalogModel]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for model in models:
        value = dict(model.raw)
        value.update(
            {
                "wire_key": model.wire_id,
                "provider": model.provider,
                "id": model.bare_id,
                "name": _display_name(model.bare_id),
            }
        )
        inputs.append(value)
    return inputs


def _display_name(model_id: str) -> str:
    words = re.split(r"[-_]", model_id)
    return " ".join(
        word.upper() if word.lower() in {"gpt", "mai"} else word.title() for word in words
    )


def write_codex_catalog(models: Sequence[CatalogModel], path: Path) -> None:
    catalog = _build_codex_model_catalog(codex_catalog_inputs(models))
    if catalog is None:
        raise ValidationError(
            "Codex bundled model metadata is unavailable; refusing to validate with "
            "fallback metadata"
        )
    write_json_owner_only(path, catalog)


def client_command_available(client: str) -> bool:
    return shutil.which(client) is not None


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _codex_config_args(runtime: Runtime) -> list[str]:
    if runtime.codex_catalog_path is None:
        raise ValidationError("Codex was selected without a generated model catalog")
    base_url = f"{runtime.target.endpoint}/api/openai/v1"
    return [
        "-c",
        f"model_provider={_toml_string(CODEX_PROVIDER_NAME)}",
        "-c",
        f"model_providers.{CODEX_PROVIDER_NAME}.name={_toml_string('Router-Maestro Live')}",
        "-c",
        f"model_providers.{CODEX_PROVIDER_NAME}.base_url={_toml_string(base_url)}",
        "-c",
        (f"model_providers.{CODEX_PROVIDER_NAME}.env_key={_toml_string('ROUTER_MAESTRO_API_KEY')}"),
        "-c",
        f"model_providers.{CODEX_PROVIDER_NAME}.wire_api={_toml_string('responses')}",
        "-c",
        f"model_catalog_json={_toml_string(str(runtime.codex_catalog_path))}",
        "-c",
        'web_search="disabled"',
    ]


def build_claude_command(
    model: str,
    prompt: str,
    *,
    session_id: str | None = None,
    resume: bool = False,
) -> list[str]:
    command = [
        "claude",
        "-p",
        "--safe-mode",
        "--no-chrome",
        "--model",
        model,
        "--prompt-suggestions=false",
        "--tools=",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
    ]
    if resume:
        if session_id is None:
            raise ValueError("A session ID is required when resuming Claude")
        command.extend(["--resume", session_id])
    elif session_id is not None:
        command.extend(["--session-id", session_id])
    else:
        command.append("--no-session-persistence")
    command.append(prompt)
    return command


def build_codex_command(
    runtime: Runtime,
    model: str,
    prompt: str,
    output_path: Path,
    *,
    session_id: str | None = None,
) -> list[str]:
    if session_id is None:
        command = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-C",
            str(runtime.project),
        ]
    else:
        command = [
            "codex",
            "exec",
            "resume",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
        ]
    command.extend(_codex_config_args(runtime))
    command.extend(["-m", model, "-o", str(output_path)])
    if session_id is not None:
        command.append(session_id)
    command.append(prompt)
    return command


def client_environment(runtime: Runtime, client: str) -> dict[str, str]:
    environment = os.environ.copy()
    if client == "claude":
        environment.pop("ANTHROPIC_API_KEY", None)
        environment["ANTHROPIC_AUTH_TOKEN"] = runtime.target.auth_value
        environment["ANTHROPIC_BASE_URL"] = f"{runtime.target.endpoint}/api/anthropic"
        environment["CLAUDE_CONFIG_DIR"] = str(runtime.claude_config_dir)
        environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    elif client == "codex":
        environment["ROUTER_MAESTRO_API_KEY"] = runtime.target.auth_value
        environment["CODEX_HOME"] = str(runtime.codex_home)
    else:
        raise ValueError(f"Unknown client: {client}")
    return environment


def run_process(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    timeout_seconds: float,
) -> ProcessResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return ProcessResult(
            tuple(command),
            124,
            _coerce_text(error.stdout),
            _coerce_text(error.stderr),
            time.monotonic() - started,
            timed_out=True,
        )
    except OSError as error:
        return ProcessResult(
            tuple(command),
            127,
            "",
            str(error),
            time.monotonic() - started,
        )
    return ProcessResult(
        tuple(command),
        completed.returncode,
        completed.stdout,
        completed.stderr,
        time.monotonic() - started,
    )


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def parse_claude_text(output: str) -> str:
    deltas: list[str] = []
    aggregate_result: str | None = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "stream_event":
            stream_event = event.get("event")
            if not isinstance(stream_event, dict):
                continue
            delta = stream_event.get("delta")
            if (
                stream_event.get("type") == "content_block_delta"
                and isinstance(delta, dict)
                and delta.get("type") == "text_delta"
                and isinstance(delta.get("text"), str)
            ):
                deltas.append(delta["text"])
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            aggregate_result = event["result"]
    if deltas:
        return "".join(deltas)
    return aggregate_result or ""


def parse_codex_session_id(output: str) -> str | None:
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        thread_id = event.get("thread_id")
        if event.get("type") == "thread.started" and isinstance(thread_id, str):
            return thread_id
    return None


def normalize_answer(value: str) -> str:
    return ANSI_ESCAPE.sub("", value).strip()


def is_explicit_transient_5xx(output: str) -> bool:
    if HTTP_400_PATTERN.search(output):
        return False
    return HTTP_5XX_PATTERN.search(output) is not None


def has_codex_fallback_warning(processes: Iterable[ProcessResult]) -> bool:
    return any(
        FALLBACK_METADATA_WARNING in process.combined_output.casefold() for process in processes
    )


def run_claude_case(runtime: Runtime, model: CatalogModel, phase: str) -> CaseAttempt:
    environment = client_environment(runtime, "claude")
    token = f"RMCLAUDE{phase.upper()}{secrets.token_hex(8).upper()}"
    processes: list[ProcessResult] = []

    if phase == "smoke":
        prompt = f"Reply with exactly this token and nothing else: {token}"
        process = run_process(
            build_claude_command(model.wire_id, prompt),
            environment=environment,
            cwd=runtime.project,
            timeout_seconds=runtime.timeout_seconds,
        )
        processes.append(process)
        actual = normalize_answer(parse_claude_text(process.stdout))
        return _single_expected_attempt(token, actual, processes)

    if phase != "recall":
        raise ValueError(f"Unknown phase: {phase}")

    session_id = str(uuid.uuid4())
    acknowledgement = f"ACK{token}"
    first_prompt = (
        f"Remember the verification token {token} for this session. "
        f"Reply with exactly {acknowledgement} and nothing else."
    )
    first = run_process(
        build_claude_command(model.wire_id, first_prompt, session_id=session_id),
        environment=environment,
        cwd=runtime.project,
        timeout_seconds=runtime.timeout_seconds,
    )
    processes.append(first)
    first_actual = normalize_answer(parse_claude_text(first.stdout))
    if first.returncode != 0 or first_actual != acknowledgement:
        return _single_expected_attempt(acknowledgement, first_actual, processes, label="turn 1")

    second_prompt = (
        "What verification token did I ask you to remember? "
        "Reply with only the token and no punctuation."
    )
    second = run_process(
        build_claude_command(model.wire_id, second_prompt, session_id=session_id, resume=True),
        environment=environment,
        cwd=runtime.project,
        timeout_seconds=runtime.timeout_seconds,
    )
    processes.append(second)
    actual = normalize_answer(parse_claude_text(second.stdout))
    return _single_expected_attempt(token, actual, processes, label="turn 2")


def run_codex_case(runtime: Runtime, model: CatalogModel, phase: str) -> CaseAttempt:
    environment = client_environment(runtime, "codex")
    token = f"RMCODEX{phase.upper()}{secrets.token_hex(8).upper()}"
    processes: list[ProcessResult] = []
    output_one = runtime.output_dir / f"codex-{uuid.uuid4().hex}-one.txt"

    if phase == "smoke":
        prompt = f"Reply with exactly this token and nothing else: {token}"
        process = run_process(
            build_codex_command(runtime, model.wire_id, prompt, output_one),
            environment=environment,
            cwd=runtime.project,
            timeout_seconds=runtime.timeout_seconds,
        )
        processes.append(process)
        actual = normalize_answer(_read_and_unlink(output_one))
        if has_codex_fallback_warning(processes):
            return CaseAttempt(
                False,
                token,
                actual,
                processes,
                "Codex emitted a fallback-model-metadata warning",
            )
        return _single_expected_attempt(token, actual, processes)

    if phase != "recall":
        raise ValueError(f"Unknown phase: {phase}")

    acknowledgement = f"ACK{token}"
    first_prompt = (
        f"Remember the verification token {token} for this session. "
        f"Reply with exactly {acknowledgement} and nothing else."
    )
    first = run_process(
        build_codex_command(runtime, model.wire_id, first_prompt, output_one),
        environment=environment,
        cwd=runtime.project,
        timeout_seconds=runtime.timeout_seconds,
    )
    processes.append(first)
    first_actual = normalize_answer(_read_and_unlink(output_one))
    if has_codex_fallback_warning(processes):
        return CaseAttempt(
            False,
            acknowledgement,
            first_actual,
            processes,
            "Codex emitted a fallback-model-metadata warning",
        )
    session_id = parse_codex_session_id(first.stdout)
    if first.returncode != 0 or first_actual != acknowledgement or session_id is None:
        detail = "turn 1 did not expose a resumable thread" if session_id is None else "turn 1"
        return _single_expected_attempt(acknowledgement, first_actual, processes, label=detail)

    output_two = runtime.output_dir / f"codex-{uuid.uuid4().hex}-two.txt"
    second_prompt = (
        "What verification token did I ask you to remember? "
        "Reply with only the token and no punctuation."
    )
    second = run_process(
        build_codex_command(
            runtime,
            model.wire_id,
            second_prompt,
            output_two,
            session_id=session_id,
        ),
        environment=environment,
        cwd=runtime.project,
        timeout_seconds=runtime.timeout_seconds,
    )
    processes.append(second)
    actual = normalize_answer(_read_and_unlink(output_two))
    if has_codex_fallback_warning(processes):
        return CaseAttempt(
            False,
            token,
            actual,
            processes,
            "Codex emitted a fallback-model-metadata warning",
        )
    return _single_expected_attempt(token, actual, processes, label="turn 2")


def _single_expected_attempt(
    expected: str,
    actual: str,
    processes: list[ProcessResult],
    *,
    label: str = "response",
) -> CaseAttempt:
    failed_process = next((process for process in processes if process.returncode != 0), None)
    if failed_process is not None:
        detail = (
            f"{label} timed out"
            if failed_process.timed_out
            else f"{label} exited with code {failed_process.returncode}"
        )
        return CaseAttempt(False, expected, actual, processes, detail)
    if actual != expected:
        return CaseAttempt(False, expected, actual, processes, f"{label} was not an exact match")
    return CaseAttempt(True, expected, actual, processes, "exact match")


def _read_and_unlink(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
    finally:
        path.unlink(missing_ok=True)


def run_case_with_retries(
    runtime: Runtime,
    model: CatalogModel,
    *,
    client: str,
    phase: str,
    transient_retries: int,
) -> CaseResult:
    started = time.monotonic()
    logs: list[str] = []
    final_attempt: CaseAttempt | None = None
    for attempt_index in range(transient_retries + 1):
        if client == "claude":
            attempt = run_claude_case(runtime, model, phase)
        elif client == "codex":
            attempt = run_codex_case(runtime, model, phase)
        else:
            raise ValueError(f"Unknown client: {client}")
        final_attempt = attempt
        log_name = _write_attempt_log(
            runtime,
            model,
            client=client,
            phase=phase,
            attempt_number=attempt_index + 1,
            attempt=attempt,
        )
        logs.append(log_name)
        if attempt.passed:
            break
        if attempt_index >= transient_retries or not attempt.transient_5xx:
            break
        print(f"[RETRY] {client:<6} {phase:<6} {model.wire_id} explicit transient 5xx")

    if final_attempt is None:
        raise AssertionError("Case loop did not execute")
    return CaseResult(
        client=client,
        phase=phase,
        model=model.wire_id,
        provider=model.provider,
        passed=final_attempt.passed,
        attempts=len(logs),
        duration_seconds=time.monotonic() - started,
        detail=final_attempt.detail,
        attempt_logs=logs,
    )


def _write_attempt_log(
    runtime: Runtime,
    model: CatalogModel,
    *,
    client: str,
    phase: str,
    attempt_number: int,
    attempt: CaseAttempt,
) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model.wire_id)
    name = f"{client}-{phase}-{safe_model}-attempt-{attempt_number}.json"
    path = runtime.output_dir / name
    secret = runtime.target.api_key
    data = {
        "client": client,
        "phase": phase,
        "model": model.wire_id,
        "provider": model.provider,
        "passed": attempt.passed,
        "detail": attempt.detail,
        "expected": attempt.expected,
        "actual": attempt.actual,
        "processes": [
            {
                "command": shlex.join(process.command),
                "returncode": process.returncode,
                "timed_out": process.timed_out,
                "duration_seconds": round(process.duration_seconds, 3),
                "diagnostic": sanitize_process_output(process, secret=secret),
            }
            for process in attempt.processes
        ],
    }
    write_json_owner_only(path, data)
    return name


def sanitize_process_output(process: ProcessResult, *, secret: str | None) -> str:
    lines: list[str] = []
    for source, text in (("stdout", process.stdout), ("stderr", process.stderr)):
        for line in text.splitlines():
            sanitized = _sanitize_log_line(line, secret=secret)
            if sanitized:
                lines.append(f"{source}: {sanitized}")
    return "\n".join(lines)[-12_000:]


def _sanitize_log_line(line: str, *, secret: str | None) -> str:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return _sanitize_text(line, secret=secret)
    return json.dumps(
        _sanitize_json(value, secret=secret),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _sanitize_json(value: Any, *, secret: str | None) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).replace("_", "").casefold()
            if (
                normalized in {field.replace("_", "") for field in SENSITIVE_LOG_FIELDS}
                or "signature" in normalized
                or "encrypted" in normalized
                or "capsule" in normalized
            ):
                result[str(key)] = "[redacted]"
            else:
                result[str(key)] = _sanitize_json(item, secret=secret)
        return result
    if isinstance(value, list):
        return [_sanitize_json(item, secret=secret) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value, secret=secret)
    return value


def _sanitize_text(value: str, *, secret: str | None) -> str:
    sanitized = value.replace(secret, "[redacted-api-key]") if secret else value
    sanitized = CAPSULE_PATTERN.sub("[redacted-capsule]", sanitized)
    return TOKEN_PATTERN.sub("[redacted-token]", sanitized)


def _write_summary(
    output_dir: Path,
    *,
    target: TargetContext,
    health: Mapping[str, Any],
    selected_models: Sequence[CatalogModel],
    results: Sequence[CaseResult],
    started_at: str,
) -> tuple[Path, Path]:
    passed = sum(result.passed for result in results)
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "context": target.name,
        "endpoint": target.endpoint,
        "server": {
            "status": health.get("status"),
            "name": health.get("name"),
            "version": health.get("version"),
        },
        "models": [model.wire_id for model in selected_models],
        "totals": {"passed": passed, "failed": len(results) - passed, "total": len(results)},
        "results": [asdict(result) for result in results],
    }
    json_path = output_dir / "summary.json"
    tsv_path = output_dir / "summary.tsv"
    write_json_owner_only(json_path, summary)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=output_dir,
        prefix=".summary.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        writer = csv.writer(temporary, delimiter="\t")
        writer.writerow(
            [
                "client",
                "phase",
                "provider",
                "model",
                "status",
                "attempts",
                "duration_seconds",
                "detail",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.client,
                    result.phase,
                    result.provider,
                    result.model,
                    "PASS" if result.passed else "FAIL",
                    result.attempts,
                    f"{result.duration_seconds:.3f}",
                    result.detail,
                ]
            )
    temporary_path.chmod(0o600)
    os.replace(temporary_path, tsv_path)
    return json_path, tsv_path


def _output_directory(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.output_dir is not None:
        path = args.output_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise ValidationError(f"--output-dir must be empty: {path}")
        path.chmod(0o700)
        return path, True
    path = Path(tempfile.mkdtemp(prefix="router-maestro-live-validation-"))
    path.chmod(0o700)
    return path, bool(args.keep_logs)


def _validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0:
        raise ValidationError("--timeout must be positive")
    if args.transient_retries < 0:
        raise ValidationError("--transient-retries cannot be negative")
    project = args.project.expanduser().resolve()
    if not project.is_dir():
        raise ValidationError(f"Project directory does not exist: {project}")
    args.project = project


def _selected_values(choice: str, values: Sequence[str]) -> tuple[str, ...]:
    return tuple(values) if choice == "all" else (choice,)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path | None = None
    state_dir: Path | None = None
    retain_output = False
    try:
        _validate_args(args)
        target = resolve_target_context(args.context)
        clients = _selected_values(args.client, CLIENTS)
        phases = _selected_values(args.phase, PHASES)
        for client in clients:
            if not client_command_available(client):
                raise ValidationError(f"Required client command is not installed: {client}")

        print(f"Context: {target.name} ({target.endpoint})")
        health, catalog = fetch_server_catalog(target, timeout_seconds=args.timeout)
        selected = select_models(
            catalog,
            providers=args.provider,
            exact_models=args.model,
            patterns=args.model_pattern,
            max_models=args.max_models,
        )
        print(
            f"Server: {health.get('name', 'Router-Maestro')} "
            f"v{health.get('version', 'unknown')} ({health.get('status', 'unknown')})"
        )
        print(
            f"Models: {len(selected)}; clients: {', '.join(clients)}; phases: {', '.join(phases)}"
        )

        output_dir, retain_output = _output_directory(args)
        state_dir = Path(tempfile.mkdtemp(prefix="router-maestro-live-state-"))
        state_dir.chmod(0o700)
        claude_config_dir = state_dir / "claude-config"
        claude_config_dir.mkdir(mode=0o700)
        codex_home = state_dir / "codex-home"
        codex_home.mkdir(mode=0o700)
        catalog_path = state_dir / "codex-model-catalog.json" if "codex" in clients else None
        if catalog_path is not None:
            write_codex_catalog(catalog, catalog_path)

        runtime = Runtime(
            target=target,
            project=args.project,
            output_dir=output_dir,
            claude_config_dir=claude_config_dir,
            codex_home=codex_home,
            codex_catalog_path=catalog_path,
            timeout_seconds=args.timeout,
        )
        started_at = datetime.now(UTC).isoformat()
        results: list[CaseResult] = []
        for client in clients:
            for phase in phases:
                for model in selected:
                    result = run_case_with_retries(
                        runtime,
                        model,
                        client=client,
                        phase=phase,
                        transient_retries=args.transient_retries,
                    )
                    results.append(result)
                    status = "PASS" if result.passed else "FAIL"
                    print(
                        f"[{status}] {client:<6} {phase:<6} {model.wire_id} "
                        f"({result.duration_seconds:.1f}s, {result.detail})"
                    )

        json_path, tsv_path = _write_summary(
            output_dir,
            target=target,
            health=health,
            selected_models=selected,
            results=results,
            started_at=started_at,
        )
        passed = sum(result.passed for result in results)
        failed = len(results) - passed
        print(f"Total: {passed} passed, {failed} failed, {len(results)} cases")
        if retain_output:
            print(f"Sanitized results: {json_path} and {tsv_path}")
        return 0 if failed == 0 else 1
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    finally:
        if state_dir is not None:
            shutil.rmtree(state_dir, ignore_errors=True)
        if output_dir is not None and not retain_output:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(run())
