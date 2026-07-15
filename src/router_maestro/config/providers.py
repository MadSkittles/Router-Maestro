"""Provider and model configuration models."""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from router_maestro.routing.model_ref import validate_provider_id

RESERVED_PROVIDER_NAMES = frozenset({"github-copilot", "openai", "anthropic"})


def default_custom_api_key_env(provider: str) -> str:
    """Derive ``MY_PROVIDER_API_KEY`` from one public provider identifier."""
    validate_provider_id(provider)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", provider).strip("_").upper()
    if not normalized:
        raise ValueError("provider ID must contain a letter or digit for env-key derivation")
    return f"{normalized}_API_KEY"


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    name: str = Field(default="", description="Display name for the model")


class CustomProviderOptions(BaseModel):
    """Runtime options supported by an OpenAI-compatible custom provider."""

    model_config = ConfigDict(extra="allow")

    api_key_env: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Environment variable that may supply the provider API key",
    )
    allow_unauthenticated: bool = Field(
        default=False,
        description="Permit requests without an Authorization header",
    )


class CustomProviderConfig(BaseModel):
    """Configuration for a custom (OpenAI-compatible) provider."""

    type: str = Field(default="openai-compatible", description="Provider type")
    baseURL: str = Field(..., description="Base URL for API requests")  # noqa: N815
    models: dict[str, ModelConfig] = Field(default_factory=dict, description="Model configurations")
    options: CustomProviderOptions = Field(
        default_factory=CustomProviderOptions,
        description="Typed provider runtime options",
    )


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
            if canonical_name in RESERVED_PROVIDER_NAMES:
                raise ValueError(
                    f"provider name '{provider_name}' is reserved for a built-in provider"
                )
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
