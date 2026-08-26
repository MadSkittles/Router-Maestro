"""Offline benchmark for identity and cross-protocol request preparation."""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from router_maestro.protocols import (
    AnthropicMessagesRuntime,
    OpenAIResponsesRuntime,
    RequestEnvelope,
    WireProtocol,
)
from router_maestro.providers.bindings import (
    COPILOT_ANTHROPIC_MESSAGES_BINDING,
    AttemptRequestContext,
    EndpointBinding,
    PreparedAttempt,
)
from router_maestro.providers.copilot import COPILOT_MESSAGES_PATH, CopilotProvider
from router_maestro.routing.capabilities import RequestFeatures
from router_maestro.routing.model_ref import ModelRef

_DEFAULT_ITERATIONS = 1_000
IDENTITY_CPU_REGRESSION_LIMIT_PERCENT = 5.0
PREPARATION_TIMING_BATCH_SIZE = 16
_WARMUP_REQUESTS = 32
_IDENTITY_PATH = "/api/anthropic/v1/messages"
_IDENTITY_UPSTREAM_MODEL = "claude-sonnet-4.5"
_CROSS_UPSTREAM_MODEL = "gpt-5.4-mini"
_ANTHROPIC_PAYLOAD = {
    "model": f"github-copilot/{_IDENTITY_UPSTREAM_MODEL}",
    "max_tokens": 1_024,
    "stream": False,
    "system": [
        {
            "type": "text",
            "text": "You are a coding assistant. Be precise and propose actionable fixes.",
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Inspect the build result and summarize the failures.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_build_1",
                    "name": "read_build",
                    "input": {"job_id": 42},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_build_1",
                    "content": "2 tests failed in the protocol compatibility suite",
                    "is_error": False,
                },
                {"type": "text", "text": "Focus on the next steps for the maintainer."},
            ],
        },
    ],
    "tools": [
        {
            "name": "read_build",
            "description": "Read a CI build result",
            "input_schema": {
                "type": "object",
                "properties": {"job_id": {"type": "integer"}},
                "required": ["job_id"],
            },
        }
    ],
    "tool_choice": {"type": "auto"},
    "temperature": 0.2,
}


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing samples and lazy-IR observations for one preparation path."""

    label: str
    wall_time_ns: tuple[int, ...]
    cpu_time_ns: tuple[int, ...]
    ir_materializations: tuple[int, ...]

    @property
    def iterations(self) -> int:
        return len(self.wall_time_ns)

    @property
    def median_wall_time_us(self) -> float:
        return statistics.median(self.wall_time_ns) / 1_000

    @property
    def median_cpu_time_us(self) -> float:
        return statistics.median(self.cpu_time_ns) / 1_000

    @property
    def median_cpu_time_ns(self) -> float:
        return statistics.median(self.cpu_time_ns)

    @property
    def total_ir_materializations(self) -> int:
        return sum(self.ir_materializations)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Legacy baseline, full identity attempt, and cross-protocol preparation results."""

    legacy_beta_identity: BenchmarkResult
    identity: BenchmarkResult
    cross_protocol: BenchmarkResult
    identity_outputs_equivalent: bool

    def __post_init__(self) -> None:
        iterations = {
            self.legacy_beta_identity.iterations,
            self.identity.iterations,
            self.cross_protocol.iterations,
        }
        if len(iterations) != 1:
            raise ValueError("all benchmark paths must contain the same number of samples")

    @property
    def iterations(self) -> int:
        return self.identity.iterations

    @property
    def identity_cpu_regression_percent(self) -> float:
        """Return the lazy identity median CPU delta from the old beta baseline."""
        baseline = self.legacy_beta_identity.median_cpu_time_ns
        current = self.identity.median_cpu_time_ns
        if baseline == 0:
            return 0.0 if current == 0 else math.inf
        return ((current - baseline) / baseline) * 100

    @property
    def identity_cpu_gate_passed(self) -> bool:
        """Apply the hard median CPU regression limit without rounding artifacts."""
        baseline = self.legacy_beta_identity.median_cpu_time_ns
        current = self.identity.median_cpu_time_ns
        if baseline == 0:
            return current == 0
        limit = 1 + (IDENTITY_CPU_REGRESSION_LIMIT_PERCENT / 100)
        return current <= baseline * limit

    @property
    def identity_gate_failures(self) -> tuple[str, ...]:
        """Return stable failure reasons consumed by the benchmark CLI gate."""
        failures: list[str] = []
        if not self.identity_outputs_equivalent:
            failures.append("lazy identity contract differs from the legacy beta baseline")
        if self.identity.total_ir_materializations != 0:
            failures.append("lazy identity materialized semantic IR")
        if not self.identity_cpu_gate_passed:
            failures.append(
                "lazy identity median CPU regression exceeded "
                f"{IDENTITY_CPU_REGRESSION_LIMIT_PERCENT:.1f}%"
            )
        return tuple(failures)

    @property
    def identity_gate_passed(self) -> bool:
        return not self.identity_gate_failures


@dataclass(frozen=True, slots=True)
class _LegacyIdentityAttempt:
    """Transport contract assembled by the retired native beta route."""

    path: str
    url: str
    payload: dict[str, Any]
    model: ModelRef
    stream: bool
    method: str = "POST"


def _legacy_beta_identity_payload(
    payload: Mapping[str, Any],
    *,
    provider: CopilotProvider,
    model: ModelRef,
) -> dict[str, Any]:
    """Reproduce the retired beta route through candidate-owned transport shaping.

    JSON parsing and provider I/O are intentionally outside this preparation
    benchmark. The old beta path inspected routing features, validated the
    native option pair, deep-copied the request body, applied the Copilot
    thinking policy, and rewrote the model before transport.
    """
    RequestFeatures.for_anthropic_native(payload)
    if not isinstance(payload, dict):
        raise TypeError("legacy beta preparation requires a dictionary payload")
    provider.outbound_contract.reject_unpreservable_native_options(payload)
    body = deepcopy(payload)
    provider.outbound_contract.apply_native_anthropic_thinking(
        body,
        model.upstream_id,
        provider._catalog_effort_values(model.upstream_id),
    )
    body["model"] = model.upstream_id
    return body


def _legacy_beta_identity_contract(
    payload: dict[str, Any],
    *,
    provider: CopilotProvider,
    model: ModelRef,
    stream: bool,
) -> _LegacyIdentityAttempt:
    """Describe the retired route's fixed transport values outside timing."""
    return _LegacyIdentityAttempt(
        path=COPILOT_MESSAGES_PATH,
        url=provider._url(COPILOT_MESSAGES_PATH),
        payload=payload,
        model=model,
        stream=stream,
    )


async def _prepare_identity_attempt(
    payload: Mapping[str, Any],
    *,
    runtime: AnthropicMessagesRuntime,
    binding: EndpointBinding,
    model: ModelRef,
    stream: bool,
) -> tuple[RequestEnvelope, PreparedAttempt]:
    """Run the production lazy identity path through a real endpoint binding."""
    envelope = RequestEnvelope(
        runtime,
        payload,
        path=_IDENTITY_PATH,
        take_ownership=True,
    )
    request_context = AttemptRequestContext(
        path=envelope.path,
        query=envelope.query,
        headers=envelope.headers,
        _mappings_owned=True,
    )
    attempt = await binding.prepare_attempt(
        model=model,
        payload=envelope.native_payload(),
        stream=stream,
        request_context=request_context,
    )
    return envelope, attempt


def _identity_contracts_match(
    legacy: _LegacyIdentityAttempt,
    current: PreparedAttempt,
) -> bool:
    """Compare every transport-relevant field without timing comparison work."""
    return (
        current.binding_id == COPILOT_ANTHROPIC_MESSAGES_BINDING
        and current.protocol is WireProtocol.ANTHROPIC_MESSAGES
        and current.url == legacy.url
        and current.model == legacy.model
        and current.stream is legacy.stream
        and current.method == legacy.method
        and dict(current.headers) == {}
        and dict(current.payload) == legacy.payload
        and legacy.path == COPILOT_MESSAGES_PATH
    )


def _per_request_ns(elapsed_ns: int) -> int:
    """Normalize one batch duration while retaining sub-microsecond resolution."""
    return elapsed_ns // PREPARATION_TIMING_BATCH_SIZE


async def benchmark_request_preparation(iterations: int) -> BenchmarkReport:
    """Measure complete attempt-time preparation with real bindings and no I/O.

    Input allocation is excluded from every sample, matching production where
    Starlette has already produced one request-scoped JSON dictionary.  The lazy
    path takes ownership of that dictionary exactly as ``build_generation_pipeline``
    does, then runs the real Copilot Messages ``EndpointBinding`` and
    ``ProviderDialect`` through immutable ``PreparedAttempt`` construction.

    Each timing sample is a small batch whose elapsed CPU/wall time is normalized
    per request. This avoids a one-microsecond process-clock quantum deciding a
    five-percent gate for a preparation path that itself takes only a few
    microseconds. Contract comparisons and input allocation stay outside timing.
    """
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    ingress_runtime = AnthropicMessagesRuntime()
    responses_runtime = OpenAIResponsesRuntime(
        provider_name="github-copilot",
        binding_id="responses",
    )
    legacy_wall: list[int] = []
    legacy_cpu: list[int] = []
    legacy_ir: list[int] = []
    identity_wall: list[int] = []
    identity_cpu: list[int] = []
    identity_ir: list[int] = []
    cross_wall: list[int] = []
    cross_cpu: list[int] = []
    cross_ir: list[int] = []
    identity_outputs_equivalent = True
    provider = CopilotProvider()
    identity_binding = next(
        binding
        for binding in provider.bindings()
        if binding.id == COPILOT_ANTHROPIC_MESSAGES_BINDING
    )
    identity_model = ModelRef(provider=provider.name, upstream_id=_IDENTITY_UPSTREAM_MODEL)

    try:
        # Warm caches, imports, and the global reasoning policy before either side
        # is sampled. Warm-up uses the same production functions but is not timed.
        for _ in range(_WARMUP_REQUESTS):
            warm_input = deepcopy(_ANTHROPIC_PAYLOAD)
            _legacy_beta_identity_payload(
                warm_input,
                provider=provider,
                model=identity_model,
            )
            await _prepare_identity_attempt(
                deepcopy(_ANTHROPIC_PAYLOAD),
                runtime=ingress_runtime,
                binding=identity_binding,
                model=identity_model,
                stream=False,
            )

        for sample_index in range(iterations):
            # Each path receives request-scoped objects allocated by the JSON
            # parser. Copies are deliberately outside the measured region.
            legacy_inputs = [
                deepcopy(_ANTHROPIC_PAYLOAD) for _ in range(PREPARATION_TIMING_BATCH_SIZE)
            ]
            identity_inputs = [
                deepcopy(_ANTHROPIC_PAYLOAD) for _ in range(PREPARATION_TIMING_BATCH_SIZE)
            ]
            cross_inputs = [
                deepcopy(_ANTHROPIC_PAYLOAD) for _ in range(PREPARATION_TIMING_BATCH_SIZE)
            ]

            legacy_payloads: list[dict[str, Any]] = []
            identity_attempts: list[tuple[RequestEnvelope, PreparedAttempt]] = []

            async def measure_identity() -> None:
                wall_start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                for identity_input in identity_inputs:
                    identity_attempts.append(
                        await _prepare_identity_attempt(
                            identity_input,
                            runtime=ingress_runtime,
                            binding=identity_binding,
                            model=identity_model,
                            stream=False,
                        )
                    )
                identity_cpu.append(_per_request_ns(time.process_time_ns() - cpu_start))
                identity_wall.append(_per_request_ns(time.perf_counter_ns() - wall_start))

            def measure_legacy() -> None:
                wall_start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                for legacy_input in legacy_inputs:
                    legacy_payloads.append(
                        _legacy_beta_identity_payload(
                            legacy_input,
                            provider=provider,
                            model=identity_model,
                        )
                    )
                legacy_cpu.append(_per_request_ns(time.process_time_ns() - cpu_start))
                legacy_wall.append(_per_request_ns(time.perf_counter_ns() - wall_start))

            # Alternate order to prevent a consistent first/second cache bias.
            if sample_index % 2:
                await measure_identity()
                measure_legacy()
            else:
                measure_legacy()
                await measure_identity()

            per_request_ir = [
                envelope.materialization_count for envelope, _attempt in identity_attempts
            ]
            if len(set(per_request_ir)) != 1:
                raise RuntimeError("identity batch produced inconsistent IR counts")
            identity_ir.append(per_request_ir[0])
            legacy_ir.append(0)
            legacy_attempts = [
                _legacy_beta_identity_contract(
                    payload,
                    provider=provider,
                    model=identity_model,
                    stream=False,
                )
                for payload in legacy_payloads
            ]
            identity_outputs_equivalent = identity_outputs_equivalent and all(
                _identity_contracts_match(legacy, current)
                for legacy, (_envelope, current) in zip(
                    legacy_attempts,
                    identity_attempts,
                    strict=True,
                )
            )

            wall_start = time.perf_counter_ns()
            cpu_start = time.process_time_ns()
            cross_counts: list[int] = []
            for cross_input in cross_inputs:
                cross_envelope = RequestEnvelope(
                    ingress_runtime,
                    cross_input,
                    take_ownership=True,
                )
                semantic_request = await cross_envelope.semantic_ir()
                await responses_runtime.encode_request(
                    replace(semantic_request, model=_CROSS_UPSTREAM_MODEL)
                )
                cross_counts.append(cross_envelope.materialization_count)
            cross_cpu.append(_per_request_ns(time.process_time_ns() - cpu_start))
            cross_wall.append(_per_request_ns(time.perf_counter_ns() - wall_start))
            if len(set(cross_counts)) != 1:
                raise RuntimeError("cross-protocol batch produced inconsistent IR counts")
            cross_ir.append(cross_counts[0])
    finally:
        await provider.close()

    return BenchmarkReport(
        legacy_beta_identity=BenchmarkResult(
            label="baseline: retired beta native preparation",
            wall_time_ns=tuple(legacy_wall),
            cpu_time_ns=tuple(legacy_cpu),
            ir_materializations=tuple(legacy_ir),
        ),
        identity=BenchmarkResult(
            label="identity: Anthropic -> Copilot PreparedAttempt",
            wall_time_ns=tuple(identity_wall),
            cpu_time_ns=tuple(identity_cpu),
            ir_materializations=tuple(identity_ir),
        ),
        cross_protocol=BenchmarkResult(
            label="cross: Anthropic -> OpenAI Responses",
            wall_time_ns=tuple(cross_wall),
            cpu_time_ns=tuple(cross_cpu),
            ir_materializations=tuple(cross_ir),
        ),
        identity_outputs_equivalent=identity_outputs_equivalent,
    )


def format_report(report: BenchmarkReport) -> str:
    """Render stable, human-readable benchmark statistics."""
    iteration_label = "iteration" if report.iterations == 1 else "iterations"
    lines = [
        f"Offline request preparation benchmark ({report.iterations} {iteration_label})",
        f"Timing batch size: {PREPARATION_TIMING_BATCH_SIZE} requests per iteration",
    ]
    for result in (
        report.legacy_beta_identity,
        report.identity,
        report.cross_protocol,
    ):
        lines.extend(
            (
                "",
                result.label,
                f"  median wall time: {result.median_wall_time_us:.3f} us",
                f"  median CPU time:  {result.median_cpu_time_us:.3f} us",
                "  IR materializations: "
                f"{result.total_ir_materializations} total; "
                f"{min(result.ir_materializations)}..{max(result.ir_materializations)} per request",
            )
        )
    regression = report.identity_cpu_regression_percent
    regression_text = "infinite" if math.isinf(regression) else f"{regression:.3f}%"
    lines.extend(
        (
            "",
            "Identity hard gate",
            "  payload/path/model/stream contract equivalent to legacy beta: "
            f"{'yes' if report.identity_outputs_equivalent else 'no'}",
            f"  median CPU regression: {regression_text}",
            f"  allowed median CPU regression: <={IDENTITY_CPU_REGRESSION_LIMIT_PERCENT:.1f}%",
            f"  result: {'PASS' if report.identity_gate_passed else 'FAIL'}",
        )
    )
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=_DEFAULT_ITERATIONS,
        help=f"number of samples per preparation path (default: {_DEFAULT_ITERATIONS})",
    )
    return parser.parse_args(argv)


BenchmarkRunner = Callable[[int], Coroutine[Any, Any, BenchmarkReport]]


def main(
    argv: Sequence[str] | None = None,
    *,
    benchmark_runner: BenchmarkRunner | None = None,
) -> int:
    args = _parse_args(argv)
    runner = benchmark_runner or benchmark_request_preparation
    report = asyncio.run(runner(args.iterations))
    print(format_report(report))
    if report.identity_gate_failures:
        for failure in report.identity_gate_failures:
            print(f"benchmark gate failed: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
