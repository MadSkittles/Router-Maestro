"""Provider and model configuration models."""

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    name: str = Field(default="", description="Display name for the model")


class CustomProviderConfig(BaseModel):
    """Configuration for a custom (OpenAI-compatible) provider."""

    type: str = Field(default="openai-compatible", description="Provider type")
    baseURL: str = Field(..., description="Base URL for API requests")  # noqa: N815
    models: dict[str, ModelConfig] = Field(default_factory=dict, description="Model configurations")
    options: dict[str, Any] = Field(default_factory=dict, description="Additional provider options")


class ProvidersConfig(BaseModel):
    """Root configuration for custom providers only."""

    providers: dict[str, CustomProviderConfig] = Field(
        default_factory=dict,
        description="Custom provider configurations (not including built-in providers)",
    )

    @classmethod
    def get_default(cls) -> "ProvidersConfig":
        """Get default empty configuration."""
        return cls(providers={})
