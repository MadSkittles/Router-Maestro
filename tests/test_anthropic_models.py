"""Tests for the Anthropic models endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from router_maestro.providers.base import ModelInfo
from router_maestro.server.routes.anthropic import (
    _generate_display_name,
    list_models,
    router,
)
from router_maestro.server.schemas.anthropic import AnthropicModelInfo, AnthropicModelList


class TestGenerateDisplayName:
    """Tests for display name generation."""

    def test_simple_model_name(self):
        """Test display name for simple model ID."""
        result = _generate_display_name("claude-sonnet-4")
        assert result == "Claude Sonnet 4"

    def test_model_with_provider(self):
        """Test display name includes provider."""
        result = _generate_display_name("github-copilot/claude-sonnet-4")
        assert result == "Claude Sonnet 4 (github-copilot)"

    def test_model_with_version_number(self):
        """Test display name preserves version numbers."""
        result = _generate_display_name("openai/gpt-4o")
        assert result == "Gpt 4o (openai)"

    def test_model_with_underscores(self):
        """Test display name handles underscores."""
        result = _generate_display_name("provider/some_model_name")
        assert result == "Some Model Name (provider)"


class TestAnthropicModelInfoSchema:
    """Tests for AnthropicModelInfo schema."""

    def test_model_info_fields(self):
        """Test that model info has required fields."""
        model = AnthropicModelInfo(
            id="claude-sonnet-4",
            created_at="2025-02-02T00:00:00Z",
            display_name="Claude Sonnet 4",
            type="model",
        )
        assert model.id == "claude-sonnet-4"
        assert model.created_at == "2025-02-02T00:00:00Z"
        assert model.display_name == "Claude Sonnet 4"
        assert model.type == "model"

    def test_model_info_default_type(self):
        """Test that type defaults to 'model'."""
        model = AnthropicModelInfo(
            id="test-model",
            created_at="2025-02-02T00:00:00Z",
            display_name="Test Model",
        )
        assert model.type == "model"


class TestAnthropicModelListSchema:
    """Tests for AnthropicModelList schema."""

    def test_model_list_with_data(self):
        """Test model list with data."""
        models = AnthropicModelList(
            data=[
                AnthropicModelInfo(
                    id="model-1",
                    created_at="2025-02-02T00:00:00Z",
                    display_name="Model 1",
                )
            ],
            first_id="model-1",
            last_id="model-1",
            has_more=False,
        )
        assert len(models.data) == 1
        assert models.first_id == "model-1"
        assert models.last_id == "model-1"
        assert models.has_more is False

    def test_model_list_empty(self):
        """Test empty model list."""
        models = AnthropicModelList(data=[])
        assert len(models.data) == 0
        assert models.first_id is None
        assert models.last_id is None
        assert models.has_more is False

    def test_model_list_pagination(self):
        """Test model list with pagination."""
        models = AnthropicModelList(
            data=[],
            first_id="first",
            last_id="last",
            has_more=True,
        )
        assert models.has_more is True


@pytest.fixture
def mock_router():
    """Create a mock router."""
    mock = AsyncMock()
    mock.list_models = AsyncMock(
        return_value=[
            ModelInfo(
                id="github-copilot/claude-sonnet-4",
                name="claude-sonnet-4",
                provider="github-copilot",
            ),
            ModelInfo(
                id="github-copilot/gpt-4o",
                name="gpt-4o",
                provider="github-copilot",
            ),
            ModelInfo(id="openai/gpt-4o", name="gpt-4o", provider="openai"),
        ]
    )
    return mock


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestListModelsEndpoint:
    """Tests for the /api/anthropic/v1/models endpoint."""

    @pytest.mark.anyio
    async def test_list_models_response_format(self, mock_router):
        """Test that response matches Anthropic format."""
        with patch("router_maestro.server.routes.anthropic.get_router", return_value=mock_router):
            response = await list_models()

        assert isinstance(response, AnthropicModelList)
        assert len(response.data) == 3
        assert response.first_id == "github-copilot/claude-sonnet-4"
        assert response.last_id == "openai/gpt-4o"
        assert response.has_more is False

    @pytest.mark.anyio
    async def test_list_models_model_fields(self, mock_router):
        """Test that each model has required Anthropic fields."""
        with patch("router_maestro.server.routes.anthropic.get_router", return_value=mock_router):
            response = await list_models()

        model = response.data[0]
        assert model.id == "github-copilot/claude-sonnet-4"
        assert model.type == "model"
        assert model.display_name == "Claude Sonnet 4 (github-copilot)"
        # created_at should be ISO 8601 format
        assert "T" in model.created_at
        assert model.created_at.endswith("Z")

    @pytest.mark.anyio
    async def test_list_models_pagination_limit(self, mock_router):
        """Test pagination with limit parameter."""
        with patch("router_maestro.server.routes.anthropic.get_router", return_value=mock_router):
            response = await list_models(limit=2)

        assert len(response.data) == 2
        assert response.has_more is True
        assert response.first_id == "github-copilot/claude-sonnet-4"
        assert response.last_id == "github-copilot/gpt-4o"

    @pytest.mark.anyio
    async def test_list_models_pagination_after_id(self, mock_router):
        """Test pagination with after_id parameter."""
        with patch("router_maestro.server.routes.anthropic.get_router", return_value=mock_router):
            response = await list_models(after_id="github-copilot/claude-sonnet-4")

        assert len(response.data) == 2
        assert response.data[0].id == "github-copilot/gpt-4o"
        assert response.has_more is False

    @pytest.mark.anyio
    async def test_list_models_empty(self):
        """Test response when no models available."""
        mock = AsyncMock()
        mock.list_models = AsyncMock(return_value=[])

        with patch("router_maestro.server.routes.anthropic.get_router", return_value=mock):
            response = await list_models()

        assert len(response.data) == 0
        assert response.first_id is None
        assert response.last_id is None
        assert response.has_more is False

    def test_http_endpoint(self, client, mock_router):
        """Test the HTTP endpoint via test client."""
        with patch("router_maestro.server.routes.anthropic.get_router", return_value=mock_router):
            response = client.get("/api/anthropic/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "first_id" in data
        assert "last_id" in data
        assert "has_more" in data
        assert len(data["data"]) == 3
