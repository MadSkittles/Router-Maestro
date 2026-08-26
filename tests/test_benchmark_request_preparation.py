"""Focused contracts for the offline request-preparation benchmark."""

from __future__ import annotations

import pytest

from router_maestro.protocols import WireProtocol
from router_maestro.providers.bindings import (
    COPILOT_ANTHROPIC_MESSAGES_BINDING,
    EndpointBinding,
)
from scripts.benchmark_request_preparation import (
    IDENTITY_CPU_REGRESSION_LIMIT_PERCENT,
    PREPARATION_TIMING_BATCH_SIZE,
    BenchmarkReport,
    BenchmarkResult,
    benchmark_request_preparation,
    format_report,
    main,
)


def _result(
    label: str,
    cpu_time_ns: tuple[int, ...],
    *,
    ir_materializations: tuple[int, ...] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        label=label,
        wall_time_ns=cpu_time_ns,
        cpu_time_ns=cpu_time_ns,
        ir_materializations=ir_materializations or tuple(0 for _ in cpu_time_ns),
    )


def _report(
    *,
    baseline_cpu_ns: tuple[int, ...] = (100, 100, 100),
    identity_cpu_ns: tuple[int, ...] = (100, 100, 100),
    identity_ir: tuple[int, ...] = (0, 0, 0),
    equivalent: bool = True,
) -> BenchmarkReport:
    return BenchmarkReport(
        legacy_beta_identity=_result("legacy", baseline_cpu_ns),
        identity=_result(
            "identity",
            identity_cpu_ns,
            ir_materializations=identity_ir,
        ),
        cross_protocol=_result(
            "cross",
            tuple(1_000 for _ in baseline_cpu_ns),
            ir_materializations=tuple(1 for _ in baseline_cpu_ns),
        ),
        identity_outputs_equivalent=equivalent,
    )


@pytest.mark.asyncio
async def test_benchmark_reports_statistics_and_lazy_ir_counts() -> None:
    report = await benchmark_request_preparation(iterations=3)

    assert report.iterations == 3
    assert report.legacy_beta_identity.iterations == 3
    assert report.identity.iterations == 3
    assert report.cross_protocol.iterations == 3
    assert report.legacy_beta_identity.median_wall_time_us >= 0
    assert report.legacy_beta_identity.median_cpu_time_us >= 0
    assert report.identity.median_wall_time_us >= 0
    assert report.identity.median_cpu_time_us >= 0
    assert report.cross_protocol.median_wall_time_us >= 0
    assert report.cross_protocol.median_cpu_time_us >= 0
    assert report.legacy_beta_identity.ir_materializations == (0, 0, 0)
    assert report.identity.ir_materializations == (0, 0, 0)
    assert report.identity.total_ir_materializations == 0
    assert report.cross_protocol.ir_materializations == (1, 1, 1)
    assert report.cross_protocol.total_ir_materializations == 3
    assert report.identity_outputs_equivalent is True
    assert PREPARATION_TIMING_BATCH_SIZE > 1

    output = format_report(report)
    assert "Offline request preparation benchmark (3 iterations)" in output
    assert f"Timing batch size: {PREPARATION_TIMING_BATCH_SIZE} requests per iteration" in output
    assert "baseline: retired beta native preparation" in output
    assert "identity: Anthropic -> Copilot PreparedAttempt" in output
    assert "cross: Anthropic -> OpenAI Responses" in output
    assert "0 total; 0..0 per request" in output
    assert "3 total; 1..1 per request" in output
    assert "payload/path/model/stream contract equivalent to legacy beta: yes" in output
    assert "allowed median CPU regression: <=5.0%" in output


@pytest.mark.asyncio
async def test_benchmark_prepares_real_copilot_endpoint_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = EndpointBinding.prepare_attempt
    attempts = []

    async def track_attempt(self: EndpointBinding, **kwargs):
        attempt = await original(self, **kwargs)
        attempts.append(attempt)
        return attempt

    monkeypatch.setattr(EndpointBinding, "prepare_attempt", track_attempt)

    report = await benchmark_request_preparation(iterations=1)

    assert report.identity_outputs_equivalent is True
    assert len(attempts) >= PREPARATION_TIMING_BATCH_SIZE
    assert all(
        attempt.binding_id == COPILOT_ANTHROPIC_MESSAGES_BINDING
        and attempt.protocol is WireProtocol.ANTHROPIC_MESSAGES
        and attempt.url.endswith("/v1/messages")
        and attempt.model.upstream_id == "claude-sonnet-4.5"
        and attempt.payload["model"] == "claude-sonnet-4.5"
        and attempt.payload["stream"] is False
        and attempt.stream is False
        and attempt.method == "POST"
        for attempt in attempts
    )


@pytest.mark.parametrize(
    ("baseline_cpu_ns", "identity_cpu_ns", "expected_regression", "expected_pass"),
    [
        ((100, 100, 100), (105, 105, 105), 5.0, True),
        ((100, 100, 100), (106, 106, 106), 6.0, False),
        ((0, 0, 0), (0, 0, 0), 0.0, True),
        ((0, 0, 0), (1, 1, 1), float("inf"), False),
    ],
)
def test_identity_cpu_gate_is_deterministic_at_hard_boundary(
    baseline_cpu_ns: tuple[int, ...],
    identity_cpu_ns: tuple[int, ...],
    expected_regression: float,
    expected_pass: bool,
) -> None:
    report = _report(
        baseline_cpu_ns=baseline_cpu_ns,
        identity_cpu_ns=identity_cpu_ns,
    )

    assert IDENTITY_CPU_REGRESSION_LIMIT_PERCENT == 5.0
    assert report.identity_cpu_regression_percent == expected_regression
    assert report.identity_cpu_gate_passed is expected_pass
    assert report.identity_gate_passed is expected_pass


@pytest.mark.parametrize(
    ("equivalent", "identity_ir", "expected_failure"),
    [
        (False, (0, 0, 0), "lazy identity contract differs"),
        (True, (0, 1, 0), "materialized semantic IR"),
    ],
)
def test_identity_gate_includes_equivalence_and_lazy_ir_invariants(
    equivalent: bool,
    identity_ir: tuple[int, ...],
    expected_failure: str,
) -> None:
    report = _report(equivalent=equivalent, identity_ir=identity_ir)

    assert report.identity_gate_passed is False
    assert any(expected_failure in failure for failure in report.identity_gate_failures)


@pytest.mark.parametrize("iterations", [0, -1])
async def test_benchmark_rejects_non_positive_iteration_count(iterations: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        await benchmark_request_preparation(iterations)


def test_main_prints_passing_report_without_using_wall_clock(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def benchmark_runner(iterations: int) -> BenchmarkReport:
        assert iterations == 3
        return _report()

    assert main(["--iterations", "3"], benchmark_runner=benchmark_runner) == 0

    captured = capsys.readouterr()
    assert "Offline request preparation benchmark (3 iterations)" in captured.out
    assert "IR materializations: 0 total; 0..0 per request" in captured.out
    assert "result: PASS" in captured.out
    assert captured.err == ""


def test_main_returns_nonzero_when_identity_cpu_gate_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def benchmark_runner(_iterations: int) -> BenchmarkReport:
        return _report(identity_cpu_ns=(106, 106, 106))

    assert main(["--iterations", "3"], benchmark_runner=benchmark_runner) == 1

    captured = capsys.readouterr()
    assert "result: FAIL" in captured.out
    assert "benchmark gate failed: lazy identity median CPU regression exceeded 5.0%" in (
        captured.err
    )
