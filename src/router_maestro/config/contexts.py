"""Context configuration for remote deployments."""

from pydantic import BaseModel, Field


class ContextConfig(BaseModel):
    """Configuration for a single deployment context."""

    endpoint: str = Field(..., description="API endpoint URL")
    api_key: str | None = Field(default=None, description="API key for authentication")


class ContextsConfig(BaseModel):
    """Root configuration for deployment contexts."""

    current: str = Field(default="local", description="Currently active context name")
    contexts: dict[str, ContextConfig] = Field(
        default_factory=dict, description="Available contexts"
    )

    @classmethod
    def get_default(cls) -> "ContextsConfig":
        """Get default configuration with local context."""
        return cls(
            current="local",
            contexts={
                "local": ContextConfig(endpoint="http://localhost:8080"),
            },
        )
