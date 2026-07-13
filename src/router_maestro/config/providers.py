"""Provider and model configuration models."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from router_maestro.routing.model_ref import validate_provider_id


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

    @field_validator("providers")
    @classmethod
    def validate_provider_names(
        cls,
        providers: dict[str, CustomProviderConfig],
    ) -> dict[str, CustomProviderConfig]:
        """Reject names that cannot round-trip through public model IDs."""
        canonical_names: dict[str, str] = {}
        for provider_name in providers:
            validate_provider_id(provider_name)
            canonical_name = provider_name.casefold()
            duplicate = canonical_names.get(canonical_name)
            if duplicate is not None:
                raise ValueError(
                    "provider names must be unique case-insensitively: "
                    f"'{duplicate}' conflicts with '{provider_name}'"
                )
            canonical_names[canonical_name] = provider_name
        return providers

    @classmethod
    def get_default(cls) -> "ProvidersConfig":
        """Get default empty configuration."""
        return cls(providers={})
