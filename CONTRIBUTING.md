# Contributing to Router-Maestro

Thank you for contributing. Router-Maestro sits between stateful coding clients
and multiple provider protocols, so a change that looks local can affect tool
history, streaming terminals, context limits, fallback, or reasoning replay.
This guide defines the development and validation workflow expected for a pull
request.

## Before You Start

- Read [README.md](README.md) for the product contract.
- Read [API Translation and Protocol Contracts](docs/api-translation.md) before
  changing request/response translation, streaming, tools, or reasoning.
- Read [Configuration](docs/configuration.md) before changing CLI or portal
  generation.
- Read [Metrics and Observability](docs/observability.md) before adding labels,
  request metadata, or traces.
- Search for a more specific `AGENTS.md` before editing a subtree.

Do not develop directly on `master`. Start from an up-to-date branch:

```bash
git switch master
git pull --ff-only
git switch -c feat/short-description
```

Use `fix/`, `docs/`, or `chore/` when they describe the change better. Preserve
unrelated user changes in a dirty worktree; do not reset or clean them away.

## Development Setup

Router-Maestro requires Python 3.14 and uses [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.14
uv sync --extra dev
uv run router-maestro --version
```

Start a development server:

```bash
uv run router-maestro server start --port 8080
```

For reload-on-source-change during local development:

```bash
uv run router-maestro server start --port 8080 --reload
```

The server reads and writes normal XDG Router-Maestro state. Use a disposable
`XDG_CONFIG_HOME` and `XDG_DATA_HOME` when a test must not touch your normal
contexts or credentials. Never commit generated credentials, contexts, capsule
keys, client configuration, or audit traces.

## Repository Map

```text
src/router_maestro/
├── protocols/       # Wire runtimes, typed semantic conversion, stream events
├── providers/       # Provider handlers, endpoint bindings, auth and contracts
├── routing/         # Model selection, Auto, capability and fallback planning
├── server/          # FastAPI routes, dispatcher, schemas and middleware
├── cli/             # CLI and client-config generators
├── web/             # Loopback-only configuration portal
├── config/          # XDG config models, repositories and paths
└── utils/           # Shared utilities, including audit support

tests/               # Offline unit and boundary tests
integration_tests/   # Local live-provider suite and controlled boundaries
skills/              # Repeatable live-client validation workflow
docs/                # User, operator and protocol reference
```

## Architecture Rules

### Keep protocol and provider responsibilities separate

- Routes are thin protocol entry points; they must not acquire provider URLs,
  headers, auth, transport ordering, or Copilot quirks.
- A provider handler owns its model catalog, authentication, endpoint bindings,
  model rewrite, provider-specific contract, and error classification.
- Transport selection for one model is not model fallback.
- Route planning selects models; it must not become the home of protocol
  conversion.

### Preserve the lazy identity path

- Same-protocol requests should stay copy-on-write and must not materialize the
  semantic IR.
- Cross-protocol requests materialize a typed semantic request lazily and at
  most once per client request; fallback attempts share it read-only.
- Cross-protocol streaming converts frames/events incrementally. Do not buffer
  a complete response merely to translate it.
- An explicit field that cannot be represented must fail before provider I/O
  with its precise parameter path. Do not add silent-drop branches.

### Respect stream commitment and terminal semantics

- A candidate may be retried only before the first valid upstream frame.
- Never replay a transport or model after the stream is committed.
- Produce one canonical terminal outcome. A clean EOF without a terminal is an
  `unexpected_eof`, not success.
- Provider business terminals such as failed, incomplete, cancelled, or safety
  outcomes are not generic retryable transport errors.

### Preserve model identity and capability metadata

- Public IDs should round-trip as `provider/model` without double qualification.
- Do not infer capabilities only from model names when catalog metadata exists.
- Keep context-window options, reasoning tiers, tools, vision, and transport
  capability metadata aligned across public model endpoints and Codex catalogs.
- Auto selects only configured task/priority models after capability and context
  filtering. It must not silently append arbitrary catalog entries.

### Handle reasoning state as sensitive data

- Preserve supported opaque continuation state through the reasoning capsule.
- Do not log capsule keys, plaintext, decrypted provider state, or hidden
  reasoning content.
- Fail closed on tampering, unknown keys, unsupported versions, or provenance
  mismatch.

## Implementing a Change

1. Reproduce the behavior at the observable boundary.
2. Trace the production path, not only a legacy provider facade. For generation
   this normally means route → dispatcher → provider binding → executor.
3. Add the narrowest failing test before or with the implementation.
4. Reuse current abstractions and translation state machines; avoid parallel
   token, auth, transport, or retry implementations.
5. Verify stream and non-stream behavior when either can be affected.
6. Update user documentation when endpoints, config, routing semantics, model
   metadata, deployment, or compatibility changes.
7. Run focused tests, then the complete offline gate.

## Offline Test Gate

Run a focused test while iterating:

```bash
uv run pytest tests/test_<area>.py -v
```

Before opening or updating a pull request, run:

```bash
uv run pytest tests/ -q
uv run pytest integration_tests/test_controlled_boundaries.py -v
uv run ruff check src/ tests/ integration_tests/ \
  skills/router-maestro-live-validation/scripts/
uv run ruff format --check src/ tests/ integration_tests/ \
  skills/router-maestro-live-validation/scripts/
npx -y basedpyright@1.39.10
```

CI runs the unit suite, Ruff, formatting, and deterministic controlled-boundary
tests. BasedPyright is also required locally for changed typed code and tests.
Do not weaken or exclude a test merely to make the gate green.

Build artifacts when packaging or container behavior changes:

```bash
uv build
docker build -t router-maestro:contributor-test .
```

When request preparation or the identity fast path changes, run the offline
benchmark as well:

```bash
uv run python scripts/benchmark_request_preparation.py --iterations 1000
```

Identity preparation must not materialize semantic IR. Treat a median identity
CPU regression above the benchmark's 5% gate as a release blocker until it is
explained and resolved.

## Audit Tracing for Development

Audit tracing is opt-in and intended for a bounded investigation. It can record
prompts, tool inputs/results, model output, URLs, status codes, routing attempts,
and terminal outcomes. Known credential headers and common credential-shaped
payload keys are redacted, but application-specific secrets may still appear.
Treat the entire trace directory as sensitive.

### Fastest local method

Enable tracing only for one development server process:

```bash
ROUTER_MAESTRO_TRACE=1 \
  uv run router-maestro server start --port 8080 --log-level DEBUG
```

Stop that process to disable the environment override. Setting
`audit.enabled=false` does **not** disable tracing while
`ROUTER_MAESTRO_TRACE=1` remains in the server environment.

Use this method when it is acceptable to restart a local development server.
Do not add `ROUTER_MAESTRO_TRACE=1` permanently to a production Compose file.

### Revision-aware runtime method

For a running local or remote context, update only the `audit` member through
the versioned admin API. The following script uses the active Router-Maestro
context and never prints its key or the returned configuration:

```bash
uv run python - <<'PY'
import asyncio

from router_maestro.cli.client import get_admin_client


async def main() -> None:
    client = get_admin_client()
    config = await client.get_runtime_config()
    revision = config.pop("revision")
    audit = dict(config.get("audit", {}))
    audit["enabled"] = True
    config["audit"] = audit
    await client.patch_runtime_config(config=config, revision=revision)


asyncio.run(main())
PY
```

This compare-and-swap fails instead of overwriting a concurrent config update.
Re-read and decide how to reconcile a conflict; do not blindly retry an old
snapshot.

To use a custom directory, add `audit["trace_dir"] = "/absolute/server/path"`.
The path is interpreted by the server process. Inside the packaged container,
the default trace path is:

```text
/home/maestro/.local/share/router-maestro/traces
```

With the repository Compose mounts, the same files appear on the host under:

```text
~/.local/share/router-maestro/traces
```

### Reproduce and correlate

Use a fresh, non-sensitive prompt and a unique request marker. When you control
the HTTP request, also set a valid request ID:

```text
X-Request-ID: req-dev-<unique-suffix>
```

The response returns the accepted/generated `X-Request-ID`. Correlate that
exact ID with server logs and:

```text
<trace-root>/<request-id>/
```

Do not assume that the most recent directory belongs to the request under
investigation: clients can send title, memory, compacting, or other auxiliary
traffic.

Possible artifacts include:

| Artifact | Meaning |
| --- | --- |
| `inbound.json` | Client method, path, headers, and body after redaction |
| `upstream.json`, `upstream_2.json`, ... | Ordered upstream request observations |
| `upstream_resp.json`, `upstream_resp_2.json`, ... | Ordered upstream response observations |
| `attempt.json`, `attempt_2.json`, ... | Provider, model, binding, protocols, conversion mode, outcome, and IR state |
| `outbound.json` | Wire status, duration, termination, response status, and safe terminal error |

The artifact set is intentionally dynamic. Validation can fail before provider
I/O; auth retry or fallback can create multiple attempts; streaming traces may
contain summaries instead of complete upstream bodies. A client-visible HTTP
`200` does not prove semantic success after stream commitment—check the final
event and `outbound.json`.

### Disable tracing

First remove any `ROUTER_MAESTRO_TRACE=1` environment override and restart the
server if necessary. Then set runtime audit to false using the same
revision-aware method:

```bash
uv run python - <<'PY'
import asyncio

from router_maestro.cli.client import get_admin_client


async def main() -> None:
    client = get_admin_client()
    config = await client.get_runtime_config()
    revision = config.pop("revision")
    audit = dict(config.get("audit", {}))
    audit["enabled"] = False
    config["audit"] = audit
    await client.patch_runtime_config(config=config, revision=revision)


asyncio.run(main())
PY
```

Send one harmless request and verify that it does not create a new trace
directory before considering audit disabled.

### Retention and cleanup

- Keep only the exact request directories needed for the investigation.
- Sanitize traces before sharing; redaction at write time is not a guarantee
  that arbitrary application secrets are absent.
- Never attach raw traces from a real user session to a public issue or PR.
- Disable audit before cleanup and confirm no process is still writing.
- Resolve and inspect the exact trace root and request ID before deletion.
- Never run a recursive delete against a home directory, XDG data root, volume
  root, wildcard, or unresolved environment variable.
- Prefer moving a small, explicitly named request directory to recoverable
  trash. If irreversible deletion is necessary, obtain explicit authorization
  and delete only that verified request directory.

For Prometheus labels, request-ID behavior, and detailed terminal semantics,
read [Metrics and Observability](docs/observability.md).

## Live Provider and Client Validation

Live tests consume provider quota and require real credentials. Run them only
when the change affects provider contracts, model catalogs, protocol conversion,
streaming, tools, reasoning, client configuration, or deployment behavior.

Authenticate a local test context first:

```bash
uv run router-maestro auth login github-copilot
```

Run the full local live-backend integration suite:

```bash
make integration-test
```

Bound only an intentional canary:

```bash
RM_INTEGRATION_MAX_MODELS=8 make integration-test
RM_INTEGRATION_MODEL=github-copilot/gpt-4o make integration-test
RM_INTEGRATION_RESPONSES_MODEL=github-copilot/gpt-5.4-mini make integration-test
```

The default suite covers the full available Copilot catalog and the OpenAI
Chat/Responses, Anthropic Messages/count-tokens, and Gemini generation/stream/
count-tokens surfaces, including tools, usage, and reasoning representatives.
It is intentionally not part of GitHub Actions.

For an already deployed test context, use the repeatable Claude Code and Codex
runner:

```bash
make live-validation \
  RM_LIVE_ARGS='--context <test-context> --client all --phase all'
```

The automated recall phase is a real two-request session-resume check, but it
does not replace interactive file and harmless read-only MCP rounds when a
change affects tool history. Follow
[`skills/router-maestro-live-validation/SKILL.md`](skills/router-maestro-live-validation/SKILL.md)
and its referenced runbook. A complete live result requires:

- Claude Code and Codex success on the same final image;
- expected audit transport/conversion records correlated by request ID;
- no Codex fallback-metadata warning;
- clean offline gates;
- fresh post-fix sessions; and
- exited clients and removed temporary test files.

Never run live validation against a production instance unless the operator
explicitly chose that target and scope.

## Documentation Changes

Keep the layers distinct:

- `README.md` is the product page and shortest successful path.
- `docs/configuration.md` owns client, context, provider, and Auto details.
- `docs/deployment.md` owns installation, HTTPS, upgrade, and rollback.
- `docs/observability.md` owns metrics, request IDs, traces, and diagnosis.
- `docs/api-translation.md` owns protocol contracts and conversion semantics.
- `CONTRIBUTING.md` owns developer workflow and validation.

Use stable endpoint paths in new documentation. `/api/gemini/v1beta` is the
Gemini protocol version; Router-Maestro's old `/beta/v1` aliases are not the
recommended paths. Verify all relative links and commands from the repository
root.

## Pull Requests

Before opening a PR:

1. rebase or merge the latest `master` according to the repository's preferred
   workflow without discarding unrelated work;
2. review `git diff --check` and the full diff;
3. run the required offline gate;
4. run scoped live validation when applicable;
5. update documentation and changelog entries for user-visible behavior; and
6. remove temporary payloads, logs, traces, and test files.

The PR description should include:

- the problem and observable impact;
- the architectural layer changed;
- compatibility or migration notes;
- focused and full test results;
- live-validation scope and deployed image/commit when applicable; and
- remaining risks or intentionally unsupported cases.

Do not include credentials, raw audit payloads, hidden reasoning, or sensitive
client history in commits, test fixtures, issues, or PR descriptions.
