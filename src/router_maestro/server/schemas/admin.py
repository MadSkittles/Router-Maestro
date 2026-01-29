"""Admin API schemas for remote management."""

from pydantic import BaseModel, Field


class AuthProviderInfo(BaseModel):
    """Information about an authenticated provider."""

    provider: str = Field(..., description="Provider name")
    auth_type: str = Field(..., description="Authentication type: 'oauth' or 'api'")
    status: str = Field(..., description="Status: 'active' or 'expired'")


class AuthListResponse(BaseModel):
    """Response for listing authenticated providers."""

    providers: list[AuthProviderInfo] = Field(default_factory=list)


class LoginRequest(BaseModel):
    """Request to initiate login."""

    provider: str = Field(..., description="Provider to authenticate with")
    api_key: str | None = Field(default=None, description="API key for API key auth")


class OAuthInitResponse(BaseModel):
    """Response for OAuth initialization (device flow)."""

    session_id: str = Field(..., description="Session ID for polling status")
    user_code: str = Field(..., description="Code to enter at verification URL")
    verification_uri: str = Field(..., description="URL to visit for authorization")
    expires_in: int = Field(..., description="Seconds until expiration")


class OAuthStatusResponse(BaseModel):
    """Response for OAuth status polling."""

    status: str = Field(..., description="Status: 'pending', 'complete', 'expired', or 'error'")
    error: str | None = Field(default=None, description="Error message if status is 'error'")


class ModelInfo(BaseModel):
    """Information about a model."""

    provider: str = Field(..., description="Provider name")
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Display name")


class ModelsResponse(BaseModel):
    """Response for listing models."""

    models: list[ModelInfo] = Field(default_factory=list)


class PrioritiesResponse(BaseModel):
    """Response for getting priorities."""

    priorities: list[str] = Field(default_factory=list, description="Model priorities in order")
    fallback: dict = Field(default_factory=dict, description="Fallback configuration")


class PrioritiesUpdateRequest(BaseModel):
    """Request to update priorities."""

    priorities: list[str] = Field(..., description="New priority list")
    fallback: dict | None = Field(default=None, description="Optional fallback config update")
