"""Tests for the Prometheus metrics route."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from starlette.responses import StreamingResponse

from router_maestro.server.app import METRICS_TOKEN_ENV, create_app
from router_maestro.server.middleware import REQUEST_ID_HEADER
from router_maestro.server.observability import (
    CONTENT_TYPE_LATEST,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
)


def metric_sample_value(metrics_text: str, metric_name: str, labels: dict[str, str]) -> float:
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name == metric_name and sample.labels == labels:
                return float(sample.value)
    raise AssertionError(f"metric sample not found: {metric_name}{labels}")


def test_metrics_endpoint_is_public_when_token_is_not_configured(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert HTTP_REQUESTS_TOTAL in response.text
    assert HTTP_REQUEST_DURATION_SECONDS in response.text


def test_metrics_endpoint_accepts_correct_metrics_token(monkeypatch):
    monkeypatch.setenv(METRICS_TOKEN_ENV, "metrics-secret")
    client = TestClient(create_app())

    response = client.get("/metrics", headers={"Authorization": "Bearer metrics-secret"})

    assert response.status_code == 200
    assert HTTP_REQUESTS_TOTAL in response.text
    assert HTTP_REQUEST_DURATION_SECONDS in response.text


def test_metrics_endpoint_rejects_missing_metrics_token(monkeypatch):
    monkeypatch.setenv(METRICS_TOKEN_ENV, "metrics-secret")
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid metrics token"


def test_metrics_endpoint_rejects_wrong_metrics_token(monkeypatch):
    monkeypatch.setenv(METRICS_TOKEN_ENV, "metrics-secret")
    client = TestClient(create_app())

    response = client.get("/metrics", headers={"Authorization": "Bearer wrong-token"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid metrics token"


def test_metrics_endpoint_rejects_wrong_token_when_configured_token_is_non_ascii(monkeypatch):
    monkeypatch.setenv(METRICS_TOKEN_ENV, "métrica")
    client = TestClient(create_app())

    response = client.get("/metrics", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid metrics token"


def test_metrics_endpoint_returns_prometheus_text_format(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    assert f"# TYPE {HTTP_REQUESTS_TOTAL} counter" in response.text
    assert f"# TYPE {HTTP_REQUEST_DURATION_SECONDS} histogram" in response.text


def test_http_middleware_generates_request_id_header(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]


def test_http_middleware_preserves_request_id_header(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.get("/health", headers={REQUEST_ID_HEADER: "req-test-123"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "req-test-123"


def test_http_middleware_replaces_invalid_request_id_header(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.get("/health", headers={REQUEST_ID_HEADER: "bad request id"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]
    assert response.headers[REQUEST_ID_HEADER] != "bad request id"


def test_http_middleware_replaces_overlong_request_id_header(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())
    overlong_request_id = "r" * 129

    response = client.get("/health", headers={REQUEST_ID_HEADER: overlong_request_id})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]
    assert response.headers[REQUEST_ID_HEADER] != overlong_request_id


def test_http_middleware_records_successful_request_metrics(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.get("/health")
    metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert (
        'router_maestro_http_requests_total{method="GET",path_template="/health",status="200"}'
        in metrics_response.text
    )
    assert (
        'router_maestro_http_request_duration_seconds_count{method="GET",'
        'path_template="/health",status="200"}' in metrics_response.text
    )


def test_http_middleware_metrics_counter_increments_and_duration_uses_labels(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    first_response = client.get("/health", headers={REQUEST_ID_HEADER: "req-first"})
    first_metrics = client.get("/metrics")
    second_response = client.get("/health", headers={REQUEST_ID_HEADER: "req-second"})
    second_metrics = client.get("/metrics")

    labels = {"method": "GET", "path_template": "/health", "status": "200"}
    assert first_response.headers[REQUEST_ID_HEADER] == "req-first"
    assert second_response.headers[REQUEST_ID_HEADER] == "req-second"
    assert metric_sample_value(first_metrics.text, HTTP_REQUESTS_TOTAL, labels) == 1
    assert metric_sample_value(second_metrics.text, HTTP_REQUESTS_TOTAL, labels) == 2
    assert (
        metric_sample_value(
            second_metrics.text,
            f"{HTTP_REQUEST_DURATION_SECONDS}_count",
            labels,
        )
        == 2
    )
    assert (
        metric_sample_value(
            second_metrics.text,
            f"{HTTP_REQUEST_DURATION_SECONDS}_bucket",
            labels | {"le": "+Inf"},
        )
        == 2
    )


def test_http_middleware_records_unauthenticated_request_metrics(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    monkeypatch.setenv("ROUTER_MAESTRO_API_KEY", "server-secret")
    client = TestClient(create_app())

    response = client.get("/api/openai/v1/models")
    metrics_response = client.get("/metrics")

    assert response.status_code == 401
    assert response.headers[REQUEST_ID_HEADER]
    assert (
        'router_maestro_http_requests_total{method="GET",path_template="/api/openai/v1/models",'
        'status="401"}' in metrics_response.text
    )


def test_http_middleware_records_exception_path_metrics(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    app = create_app()

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom", headers={REQUEST_ID_HEADER: "req-boom-123"})
    metrics_response = client.get("/metrics")

    assert response.status_code == 500
    assert response.headers[REQUEST_ID_HEADER] == "req-boom-123"
    assert (
        'router_maestro_http_requests_total{method="GET",path_template="/boom",status="500"}'
        in metrics_response.text
    )
    assert (
        'router_maestro_http_request_duration_seconds_count{method="GET",'
        'path_template="/boom",status="500"}' in metrics_response.text
    )

    with pytest.raises(RuntimeError, match="boom"):
        TestClient(app).get("/boom")


def test_http_middleware_records_stream_duration_after_final_body(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    app = create_app()

    @app.get("/slow-stream")
    async def slow_stream():
        async def generate():
            yield b"first\n"
            await asyncio.sleep(0.05)
            yield b"second\n"

        return StreamingResponse(generate(), media_type="text/plain")

    client = TestClient(app)

    response = client.get("/slow-stream")
    metrics_response = client.get("/metrics")

    labels = {"method": "GET", "path_template": "/slow-stream", "status": "200"}
    assert response.status_code == 200
    assert metric_sample_value(metrics_response.text, HTTP_REQUESTS_TOTAL, labels) == 1
    assert (
        metric_sample_value(metrics_response.text, f"{HTTP_REQUEST_DURATION_SECONDS}_sum", labels)
        >= 0.05
    )


def test_http_middleware_records_mid_stream_exception_as_500(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    app = create_app()

    @app.get("/broken-stream")
    async def broken_stream():
        async def generate():
            yield b"first\n"
            raise RuntimeError("stream exploded")

        return StreamingResponse(generate(), media_type="text/plain")

    client = TestClient(app, raise_server_exceptions=False)

    with client.stream("GET", "/broken-stream") as response:
        assert response.status_code == 200
        list(response.iter_bytes())
    metrics_response = client.get("/metrics")

    assert (
        'router_maestro_http_requests_total{method="GET",path_template="/broken-stream",'
        'status="500"}' in metrics_response.text
    )
    assert (
        'router_maestro_http_requests_total{method="GET",path_template="/broken-stream",'
        'status="200"}' not in metrics_response.text
    )


def test_http_middleware_records_cors_preflight(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.options(
        "/api/openai/v1/chat/completions",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]
    assert (
        'router_maestro_http_requests_total{method="OPTIONS",'
        'path_template="/api/openai/v1/chat/completions",status="200"}' in metrics_response.text
    )


def test_http_middleware_folds_unknown_method_label(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.request("PROPFIND", "/health")
    metrics_response = client.get("/metrics")

    assert response.status_code == 405
    assert 'method="PROPFIND"' not in metrics_response.text
    assert (
        'router_maestro_http_requests_total{method="OTHER",path_template="/health",'
        'status="405"}' in metrics_response.text
    )


def test_http_middleware_uses_unmatched_label_for_404(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.get("/this/path/does/not/exist/abc123xyz")
    metrics_response = client.get("/metrics")

    assert response.status_code == 404
    assert 'path_template="/this/path/does/not/exist/abc123xyz"' not in metrics_response.text
    assert (
        'router_maestro_http_requests_total{method="GET",path_template="unmatched",status="404"}'
        in metrics_response.text
    )


def test_metrics_registry_is_isolated_per_app(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    first_client = TestClient(create_app())
    second_client = TestClient(create_app())

    first_client.get("/health")
    first_metrics = first_client.get("/metrics")
    second_metrics = second_client.get("/metrics")

    labels = {"method": "GET", "path_template": "/health", "status": "200"}
    assert metric_sample_value(first_metrics.text, HTTP_REQUESTS_TOTAL, labels) == 1
    with pytest.raises(AssertionError, match="metric sample not found"):
        metric_sample_value(second_metrics.text, HTTP_REQUESTS_TOTAL, labels)
