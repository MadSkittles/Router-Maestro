"""Tests for the Prometheus metrics route."""

from fastapi.testclient import TestClient

from router_maestro.server.app import METRICS_TOKEN_ENV, create_app
from router_maestro.server.observability import (
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
