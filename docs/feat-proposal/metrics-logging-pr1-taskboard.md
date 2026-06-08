# Metrics + Logging PR1 Taskboard

## PR1 Goal

Land the HTTP-layer observability foundation:

- Top-level `/metrics` endpoint.
- HTTP request metrics.
- Request ID middleware with `X-Request-ID` response header.
- Minimal request entry/exit logging.
- Tests and docs for the HTTP layer.

PR1 must not change routing/provider behavior.

## PR1 Scope

### In Scope

- `prometheus-client` dependency and lockfile update.
- `server/observability/metrics.py` or equivalent metrics module.
- Top-level `/metrics` endpoint.
- `/metrics` auth behavior using optional `ROUTER_MAESTRO_METRICS_TOKEN`.
- HTTP middleware for:
  - request ID extraction/generation
  - `X-Request-ID` response header
  - HTTP request counter
  - HTTP request duration histogram
  - request entry/exit logging
- Tests for `/metrics`, middleware behavior, registry isolation, 404 label behavior, and exception-path metrics.
- `README.md`, `docs/observability.md`, and `CHANGELOG.md` updates.

### Out of Scope

- No changes to `routing/router.py`.
- No provider/fallback metrics instrumentation.
- No route handler logging field unification.
- No streaming/provider duration metrics.
- No rate limiting.

## Work Items

| ID | Status | Task | Scope | Verification |
|---|---|---|---|---|
| PR1-01 | Done | Add Prometheus dependency | Add `prometheus-client>=0.20.0` to `pyproject.toml`; run `uv lock`. | `uv lock` succeeds; dependency is present in `uv.lock`. |
| PR1-02 | Done | Add metrics module | Create `src/router_maestro/server/observability/metrics.py` with low-level metric definitions and helper functions. | Module imports cleanly; no default test pollution; metrics use `router_maestro_` prefix. |
| PR1-03 | Done | Define metrics constants | Add constants/helpers for LLM buckets, bool labels (`"true"`/`"false"`), `path_template="unmatched"`, and the full 10-value `api_kind` enum for PR2 reuse. | Unit tests cover bool label conversion and unmatched route fallback; all 10 `api_kind` values are visible in constants. |
| PR1-04 | Done | Implement `/metrics` endpoint | Expose top-level `/metrics` from `server/app.py` or a dedicated route module. | `GET /metrics` returns Prometheus text format and includes `router_maestro_http_requests_total` plus `router_maestro_http_request_duration_seconds`. |
| PR1-05 | Todo | Implement `/metrics` auth behavior | Support default public `/metrics`; if `ROUTER_MAESTRO_METRICS_TOKEN` is set, require that token. | Tests cover four branches: no token configured = 200; token configured + correct token = 200; token configured + missing token = 401; token configured + wrong token = 401. |
| PR1-06 | Todo | Add HTTP observability middleware | Add middleware for `request_id`, response header, HTTP metrics, and entry/exit logs. Entry log fields: `request_id`, `method`, `path_template`. Exit log fields: `request_id`, `method`, `path_template`, `status`, `elapsed_ms`. Do not include `model`, `provider`, or `outcome` in PR1 middleware logs. | Responses include `X-Request-ID`; provided header is preserved; generated ID exists when absent; unauthenticated 401 requests still record HTTP counter/duration and include `X-Request-ID`. |
| PR1-07 | Todo | Preserve exception-path metrics | Ensure HTTP duration/count metrics are recorded when route handlers raise. | Test route raising an exception still records a 5xx duration/count metric; exception behavior is unchanged. |
| PR1-08 | Todo | Handle unmatched routes safely | Use FastAPI route templates for labels; use `path_template="unmatched"` for 404 routes. | A random 404 path does not appear as a Prometheus label; `unmatched` does. |
| PR1-09 | Todo | Test `/metrics` endpoint | Add focused tests for endpoint availability and key metric presence. | Tests assert text format response and expected `router_maestro_` metric names. |
| PR1-10 | Todo | Test HTTP middleware metrics | Add tests for counter increments, duration output, status labels, request ID response header, and 401 metric/header behavior. | Tests pass without calling real providers or external networks. |
| PR1-11 | Todo | Isolate Prometheus tests | Use independent `CollectorRegistry()` or equivalent reset mechanism for tests. | Tests are order-independent and do not fail when run repeatedly. |
| PR1-12 | Todo | Update docs | Add a short `README.md` "Metrics & Observability" section that links to a new `docs/observability.md`; put scrape examples, token behavior, key HTTP metrics, labels, and troubleshooting guidance in `docs/observability.md`. | README links to `docs/observability.md`; the docs include local scrape instructions and explain metric labels. |
| PR1-13 | Todo | Update changelog | Add a `CHANGELOG.md` entry for the new HTTP observability foundation. | Changelog mentions `/metrics`, HTTP metrics, and request ID header. |
| PR1-14 | Todo | Run validation | Run focused pytest targets and lint for touched code. | Focused tests pass; `uv run ruff check src/ tests/` passes; any skip/xfail is explained in the PR description. |

## Suggested Implementation Order

1. PR1-01: Add dependency and lockfile.
2. PR1-02 and PR1-03: Build the metrics module and constants.
3. PR1-04 and PR1-05: Wire `/metrics` and its auth behavior.
4. PR1-06, PR1-07, and PR1-08: Add middleware behavior.
5. PR1-09, PR1-10, and PR1-11: Add focused tests.
6. PR1-12 and PR1-13: Update docs and changelog.
7. PR1-14: Run validation and prepare the PR.

## Acceptance Criteria

- `/metrics` is available at the top-level path.
- `/metrics` auth behavior is deterministic and tested.
- HTTP metrics use route templates and never raw random paths.
- 404 requests use `path_template="unmatched"`.
- Every response includes `X-Request-ID`.
- Route handlers raising exceptions still record HTTP request counter/duration with `status=5xx`.
- Unauthenticated 401 requests still record HTTP request counter/duration and include `X-Request-ID`.
- Existing route behavior and status codes are unchanged.
- PR1 does not modify routing/provider internals.
- Tests are isolated from Prometheus global registry state.
