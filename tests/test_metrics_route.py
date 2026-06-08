"""Tests for the Prometheus metrics route."""

import pytest
from fastapi.testclient import TestClient

from router_maestro.server.app import METRICS_TOKEN_ENV, create_app
from router_maestro.server.middleware import REQUEST_ID_HEADER
from router_maestro.server.observability import (
    CONTENT_TYPE_LATEST,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
)


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


def test_http_middleware_uses_unmatched_label_for_404(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.get("/this/path/does/not/exist/abc123xyz")
    metrics_response = client.get("/metrics")

    assert response.status_code == 404
    assert (
        'path_template="/this/path/does/not/exist/abc123xyz"' not in metrics_response.text
    )
    assert (
        'router_maestro_http_requests_total{method="GET",path_template="unmatched",status="404"}'
        in metrics_response.text
    )
