"""Tests for the reusable live client validation runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "router-maestro-live-validation"
    / "scripts"
    / "live_model_matrix.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("router_maestro_live_model_matrix", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _model(wire_id: str, provider: str = "github-copilot"):
    prefix = f"{provider}/"
    bare_id = wire_id[len(prefix) :] if wire_id.startswith(prefix) else wire_id
    return runner.CatalogModel(
        wire_id=wire_id,
        provider=provider,
        bare_id=bare_id,
        raw={
            "id": wire_id,
            "owned_by": provider,
            "max_output_tokens": 10_000,
            "max_context_window_tokens": 100_000,
        },
    )


def _process(output: str, *, returncode: int = 1):
    return runner.ProcessResult(
        command=("client",),
        returncode=returncode,
        stdout=output,
        stderr="",
        duration_seconds=0.1,
    )


def test_claude_parser_joins_all_text_deltas_instead_of_using_truncated_result() -> None:
    output = "\n".join(
        [
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "PREFIX"},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "SUFFIX"},
                    },
                }
            ),
            json.dumps({"type": "result", "result": "SUFFIX"}),
        ]
    )

    assert runner.parse_claude_text(output) == "PREFIXSUFFIX"


def test_claude_parser_falls_back_to_aggregate_result_without_partial_events() -> None:
    output = json.dumps({"type": "result", "result": "COMPLETE"})

    assert runner.parse_claude_text(output) == "COMPLETE"


def test_codex_parser_extracts_thread_started_id() -> None:
    output = "\n".join(
        [
            json.dumps({"type": "item.started", "item": {"type": "reasoning"}}),
            json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
        ]
    )

    assert runner.parse_codex_session_id(output) == "thread-123"


def test_model_selection_accepts_provider_exact_ids_and_globs() -> None:
    models = [
        _model("github-copilot/gemini-3.6-flash"),
        _model("github-copilot/grok-4.6"),
        _model("openai/gpt-5.6", provider="openai"),
    ]

    selected = runner.select_models(
        models,
        providers=["github-copilot"],
        exact_models=["grok-4.6"],
        patterns=["gemini-*"],
        max_models=None,
    )

    assert [model.wire_id for model in selected] == [
        "github-copilot/gemini-3.6-flash",
        "github-copilot/grok-4.6",
    ]


def test_model_selection_rejects_filters_that_match_nothing() -> None:
    with pytest.raises(runner.ValidationError, match="matched nothing"):
        runner.select_models(
            [_model("github-copilot/grok-4.6")],
            providers=[],
            exact_models=["missing-model"],
            patterns=[],
            max_models=None,
        )


def test_codex_catalog_input_preserves_wire_id_and_context_metadata() -> None:
    model = _model("github-copilot/grok-4.6")

    [catalog_input] = runner.codex_catalog_inputs([model])

    assert catalog_input["wire_key"] == "github-copilot/grok-4.6"
    assert catalog_input["provider"] == "github-copilot"
    assert catalog_input["id"] == "grok-4.6"
    assert catalog_input["max_context_window_tokens"] == 100_000


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("API Error: 503 Service Unavailable", True),
        ("status_code=502", True),
        ("HTTP 504 Gateway Timeout", True),
        ("API Error: 400 invalid_request", False),
        ("API Error: Server error mid-response", False),
    ],
)
def test_transient_retry_requires_explicit_5xx(output: str, expected: bool) -> None:
    assert runner.is_explicit_transient_5xx(output) is expected


def test_codex_command_uses_temporary_provider_and_catalog_without_secret(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    runtime = runner.Runtime(
        target=runner.TargetContext("hk", "https://rm.example", "sk-rm-secret-value"),
        project=tmp_path,
        output_dir=tmp_path,
        claude_config_dir=tmp_path / "claude-config",
        codex_home=tmp_path / "codex-home",
        codex_catalog_path=catalog_path,
        timeout_seconds=30,
    )

    command = runner.build_codex_command(
        runtime,
        "github-copilot/grok-4.6",
        "PROMPT",
        tmp_path / "last.txt",
    )
    rendered = " ".join(command)

    assert "router-maestro-live" in rendered
    assert "model_catalog_json" in rendered
    assert 'wire_api="responses"' in rendered
    assert "sk-rm-secret-value" not in rendered


def test_claude_command_uses_stream_deltas_and_safe_mode() -> None:
    command = runner.build_claude_command("github-copilot/gpt-5.6", "PROMPT")

    assert "--safe-mode" in command
    assert "--include-partial-messages" in command
    assert "stream-json" in command
    assert "--tools=" in command
    assert "--no-session-persistence" in command


def test_claude_model_argument_applies_server_default_1m_context() -> None:
    model = _model("github-copilot/gemini-3.6-flash")
    model.raw["context_window_options"] = [
        {"tier": "default", "max_prompt_tokens": 200_000, "is_default": False},
        {"tier": "long_context", "max_prompt_tokens": 936_000, "is_default": True},
    ]

    assert runner.claude_model_argument(model) == "github-copilot/gemini-3.6-flash[1m]"


def test_claude_model_argument_keeps_nondefault_1m_tier_unsuffixed() -> None:
    model = _model("github-copilot/gpt-5.6-terra")
    model.raw["context_window_options"] = [
        {"tier": "default", "max_prompt_tokens": 200_000, "is_default": True},
        {"tier": "long_context", "max_prompt_tokens": 936_000, "is_default": False},
    ]

    assert runner.claude_model_argument(model) == "github-copilot/gpt-5.6-terra"


def test_claude_environment_isolates_session_state(tmp_path: Path) -> None:
    runtime = runner.Runtime(
        target=runner.TargetContext("hk", "https://rm.example", "secret"),
        project=tmp_path,
        output_dir=tmp_path,
        claude_config_dir=tmp_path / "claude-config",
        codex_home=tmp_path / "codex-home",
        codex_catalog_path=None,
        timeout_seconds=30,
    )

    environment = runner.client_environment(runtime, "claude")

    assert environment["CLAUDE_CONFIG_DIR"] == str(tmp_path / "claude-config")
    assert environment["ANTHROPIC_BASE_URL"] == "https://rm.example/api/anthropic"


def test_retry_wrapper_retries_only_explicit_transient_5xx(tmp_path: Path, monkeypatch) -> None:
    runtime = runner.Runtime(
        target=runner.TargetContext("hk", "https://rm.example", None),
        project=tmp_path,
        output_dir=tmp_path,
        claude_config_dir=tmp_path / "claude-config",
        codex_home=tmp_path / "codex-home",
        codex_catalog_path=None,
        timeout_seconds=30,
    )
    attempts = iter(
        [
            runner.CaseAttempt(
                passed=False,
                expected="TOKEN",
                actual="",
                processes=[_process("API Error: 503 Service Unavailable")],
                detail="response exited with code 1",
            ),
            runner.CaseAttempt(
                passed=True,
                expected="TOKEN",
                actual="TOKEN",
                processes=[_process("", returncode=0)],
                detail="exact match",
            ),
        ]
    )
    monkeypatch.setattr(runner, "run_claude_case", lambda *_args, **_kwargs: next(attempts))

    result = runner.run_case_with_retries(
        runtime,
        _model("github-copilot/grok-4.6"),
        client="claude",
        phase="smoke",
        transient_retries=1,
    )

    assert result.passed is True
    assert result.attempts == 2


def test_retry_wrapper_does_not_retry_deterministic_400(tmp_path: Path, monkeypatch) -> None:
    runtime = runner.Runtime(
        target=runner.TargetContext("hk", "https://rm.example", None),
        project=tmp_path,
        output_dir=tmp_path,
        claude_config_dir=tmp_path / "claude-config",
        codex_home=tmp_path / "codex-home",
        codex_catalog_path=None,
        timeout_seconds=30,
    )
    calls = 0

    def fail_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return runner.CaseAttempt(
            passed=False,
            expected="TOKEN",
            actual="",
            processes=[_process("API Error: 400 invalid_request")],
            detail="response exited with code 1",
        )

    monkeypatch.setattr(runner, "run_claude_case", fail_once)

    result = runner.run_case_with_retries(
        runtime,
        _model("github-copilot/grok-4.6"),
        client="claude",
        phase="smoke",
        transient_retries=3,
    )

    assert result.passed is False
    assert result.attempts == 1
    assert calls == 1


def test_sanitized_log_redacts_api_keys_capsules_and_encrypted_fields() -> None:
    secret = "sk-rm-secret-value"
    output = json.dumps(
        {
            "authorization": secret,
            "signature": "rmr1.key.payload",
            "encrypted_content": "opaque-provider-state",
            "message": f"failed with {secret}",
        }
    )

    sanitized = runner.sanitize_process_output(_process(output), secret=secret)

    assert secret not in sanitized
    assert "opaque-provider-state" not in sanitized
    assert "rmr1.key.payload" not in sanitized
    assert sanitized.count("[redacted]") >= 3


def test_fallback_metadata_warning_is_always_a_failure_signal() -> None:
    process = _process(
        "Model metadata for github-copilot/grok-4.6 not found. Defaulting to fallback metadata",
        returncode=0,
    )

    assert runner.has_codex_fallback_warning([process]) is True


def test_retained_results_delete_raw_client_state(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "results"
    state_dir = tmp_path / "raw-state"
    model = _model("github-copilot/gpt-5.6-luna")

    def make_state_dir(*_args, **_kwargs):
        state_dir.mkdir()
        return str(state_dir)

    monkeypatch.setattr(runner.tempfile, "mkdtemp", make_state_dir)
    monkeypatch.setattr(
        runner,
        "resolve_target_context",
        lambda _name: runner.TargetContext("hk", "https://rm.example", "secret"),
    )
    monkeypatch.setattr(runner, "client_command_available", lambda _client: True)
    monkeypatch.setattr(
        runner,
        "fetch_server_catalog",
        lambda *_args, **_kwargs: ({"status": "healthy", "version": "test"}, [model]),
    )
    monkeypatch.setattr(
        runner,
        "write_codex_catalog",
        lambda _models, path: path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        runner,
        "run_case_with_retries",
        lambda *_args, **_kwargs: runner.CaseResult(
            client="codex",
            phase="smoke",
            model=model.wire_id,
            provider=model.provider,
            passed=True,
            attempts=1,
            duration_seconds=0.1,
            detail="exact match",
        ),
    )

    exit_code = runner.run(
        [
            "--context",
            "hk",
            "--client",
            "codex",
            "--phase",
            "smoke",
            "--project",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "summary.json").exists()
    assert not (output_dir / "codex-home").exists()
    assert not (output_dir / "codex-model-catalog.json").exists()
    assert not state_dir.exists()


def test_explicit_output_directory_must_be_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "stale.json").write_text("{}", encoding="utf-8")

    args = runner.parse_args(["--output-dir", str(output_dir)])

    with pytest.raises(runner.ValidationError, match="must be empty"):
        runner._output_directory(args)
