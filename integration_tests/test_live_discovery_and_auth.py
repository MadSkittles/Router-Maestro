"""Live service discovery and authenticated model-list checks."""

from __future__ import annotations

from typing import Any

import httpx

from integration_tests.conftest import assert_http_success


def test_public_health_endpoints(live_server):
    """The local service should start and expose public health metadata."""
    response = httpx.get(f"{live_server.base_url}/health", timeout=10.0)
    assert_http_success(response)
    assert response.json() == {"status": "healthy"}

    root = httpx.get(f"{live_server.base_url}/", timeout=10.0)
    assert_http_success(root)
    assert root.json()["status"] == "running"


def test_model_endpoint_requires_authentication(unauthenticated_client: httpx.Client):
    """Model endpoints are part of the authenticated model-call surface."""
    response = unauthenticated_client.get("/api/openai/v1/models")
    assert response.status_code == 401


def test_openai_models_include_github_copilot_models(client: httpx.Client):
    """OpenAI-compatible model listing should expose Copilot model metadata."""
    response = client.get("/api/openai/v1/models")
    assert_http_success(response)
    data = response.json()

    assert data["object"] == "list"
    copilot_models = [
        model for model in data["data"] if model.get("owned_by") == "github-copilot"
    ]
    assert copilot_models, data
    assert all(model["object"] == "model" for model in copilot_models)
    assert all(model["id"] for model in copilot_models)
    assert any(
        model.get("max_context_window_tokens") or model.get("max_prompt_tokens")
        for model in copilot_models
    )


def test_anthropic_models_include_github_copilot_models(client: httpx.Client):
    """Anthropic-compatible model listing should expose Copilot model metadata."""
    response = client.get("/api/anthropic/v1/models")
    assert_http_success(response)
    data: dict[str, Any] = response.json()

    assert data["data"], data
    assert any(model["id"] for model in data["data"])
    assert all(model["type"] == "model" for model in data["data"])
    assert any(
        model.get("max_context_window_tokens") or model.get("max_prompt_tokens")
        for model in data["data"]
    )
