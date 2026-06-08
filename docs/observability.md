# Metrics and Observability

Router-Maestro exposes HTTP-layer observability for local or self-hosted
deployments. The first observability layer is intentionally narrow:

- a top-level Prometheus scrape endpoint at `/metrics`
- HTTP request counters
- HTTP request duration histograms
- request IDs propagated through `X-Request-ID`

Provider, fallback, and streaming-specific metrics are planned separately. In
this layer, provider-side failures are visible through the HTTP status and
request ID that connects the client response to server logs. Streaming response
duration is measured when the response body finishes, not when the stream object
is created.

## Scraping `/metrics`

For a local server:

```bash
curl http://localhost:8080/metrics
```

For a Docker container published on port 8080:

```bash
curl http://localhost:8080/metrics
```

For a remote HTTPS deployment:

```bash
curl https://api.example.com/metrics
```

The endpoint returns Prometheus text format. A minimal Prometheus scrape config
for a local server looks like this:

```yaml
scrape_configs:
  - job_name: router-maestro
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:8080"]
```

## Metrics Token

By default, `/metrics` is public so local Prometheus can scrape it without
sharing the Router-Maestro API key. To require a dedicated token, set
`ROUTER_MAESTRO_METRICS_TOKEN` on the server:

```bash
ROUTER_MAESTRO_METRICS_TOKEN="metrics-secret" router-maestro server start --port 8080
```

Then send the token in the `Authorization` header:

```bash
curl http://localhost:8080/metrics \
  -H "Authorization: Bearer metrics-secret"
```

The metrics token is independent from the Router-Maestro server API key. It
does not grant access to model, auth, or admin APIs.

Prometheus can send the token like this:

```yaml
scrape_configs:
  - job_name: router-maestro
    metrics_path: /metrics
    bearer_token: metrics-secret
    static_configs:
      - targets: ["localhost:8080"]
```

## HTTP Metrics

### `router_maestro_http_requests_total`

Counter for completed HTTP requests handled by Router-Maestro.

Labels:

| Label | Meaning |
| --- | --- |
| `method` | HTTP method, such as `GET` or `POST`; unknown methods are folded to `OTHER`. |
| `path_template` | FastAPI route template, such as `/api/openai/v1/models`. |
| `status` | HTTP status code as a string, such as `200`, `401`, or `500`. |

Unmatched routes use `path_template="unmatched"` so random 404 paths do not
create high-cardinality Prometheus labels.

### `router_maestro_http_request_duration_seconds`

Histogram for completed HTTP request duration. For streaming responses, the
duration includes the streamed body and is recorded after the final body frame.

Labels:

| Label | Meaning |
| --- | --- |
| `method` | HTTP method, such as `GET` or `POST`; unknown methods are folded to `OTHER`. |
| `path_template` | FastAPI route template, or `unmatched` for 404 routes. |
| `status` | HTTP status code as a string. |

Buckets are tuned for LLM proxy traffic:

```text
0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60, 120 seconds
```

## Request IDs

Router-Maestro accepts an inbound `X-Request-ID` header when it matches
`[A-Za-z0-9._:-]{1,128}`. If the client does not provide one, or provides an
invalid value, the server generates a request ID. The same accepted or generated
value is:

- stored on the request for route and middleware code
- written to request entry and exit logs
- returned to the client in the `X-Request-ID` response header

This includes framework-generated 500 responses. When reporting a failed
request, include the `X-Request-ID` value from the response so operators can
find the matching server logs.

Example:

```bash
curl http://localhost:8080/api/openai/v1/models \
  -H "Authorization: Bearer sk-rm-..." \
  -H "X-Request-ID: req-debug-001" \
  -i
```

## Troubleshooting

### `/metrics` returns 401

`ROUTER_MAESTRO_METRICS_TOKEN` is set on the server. Send the same value with
`Authorization: Bearer <token>`.

Do not use the Router-Maestro server API key unless you deliberately set the
metrics token to the same value. The two tokens are independent.

### Random 404 paths are increasing

Use the `unmatched` label to watch aggregate 404 volume:

```promql
sum(rate(router_maestro_http_requests_total{path_template="unmatched"}[5m]))
```

Random request paths should not appear as `path_template` label values. If they
do, treat it as a label-cardinality bug.

### Clients report 401 responses

Check the authenticated API route labels:

```promql
sum by (path_template) (
  rate(router_maestro_http_requests_total{status="401"}[5m])
)
```

401s usually mean the client sent the wrong Router-Maestro server API key. Run:

```bash
router-maestro server show-key
```

Then compare it with the key configured in the client.

### Clients report 5xx responses

Start with status and route labels:

```promql
sum by (path_template, status) (
  rate(router_maestro_http_requests_total{status=~"5.."}[5m])
)
```

Then ask the client for the `X-Request-ID` response header and search the
server logs for that request ID.

### Requests are slow

Use the duration histogram by route:

```promql
histogram_quantile(
  0.95,
  sum by (le, path_template) (
    rate(router_maestro_http_request_duration_seconds_bucket[5m])
  )
)
```

This shows Router-Maestro HTTP-level latency. Provider and streaming-specific
latency metrics are planned for the next observability layer.
