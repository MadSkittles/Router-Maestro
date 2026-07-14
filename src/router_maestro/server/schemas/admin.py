"""Admin API schemas for remote management."""

from pydantic import BaseModel, Field

from router_maestro.auth.discovery import ProviderAuthSource
from router_maestro.auth.storage import AuthType
from router_maestro.config.priorities import PrioritiesConfig


class AuthProviderInfo(BaseModel):
    """Information about an authenticated provider."""

    provider: str = Field(..., description="Provider name")
    auth_type: str = Field(..., description="Authentication type: 'oauth' or 'api'")
    status: str = Field(..., description="Status: 'active' or 'expired'")


class AuthListResponse(BaseModel):
    """Response for listing authenticated providers."""

    providers: list[AuthProviderInfo] = Field(default_factory=list)


class AuthProviderDefinitionInfo(BaseModel):
    """Non-secret provider authentication metadata exposed by this server."""

    provider: str
    display_name: str
    auth_type: AuthType
    credential_required: bool
    source: ProviderAuthSource
    api_key_env: str | None = None


class AuthProviderDefinitionsResponse(BaseModel):
    """Provider login definitions configured for this server."""

    providers: list[AuthProviderDefinitionInfo] = Field(default_factory=list)


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
    max_prompt_tokens: int | None = Field(default=None, description="Maximum prompt tokens (input)")
    max_output_tokens: int | None = Field(default=None, description="Maximum output tokens")
    max_context_window_tokens: int | None = Field(
        default=None, description="Maximum total context window (input + output)"
    )


class ModelsResponse(BaseModel):
    """Response for listing models."""

    models: list[ModelInfo] = Field(default_factory=list)


REVISION_PATTERN = r"^[0-9a-f]{64}$"


class RuntimeConfigResponse(PrioritiesConfig):
    """Complete runtime configuration at one content revision."""

    revision: str = Field(pattern=REVISION_PATTERN)


class RuntimeConfigPatchRequest(PrioritiesConfig):
    """Complete replacement runtime configuration with its expected revision."""

    revision: str = Field(pattern=REVISION_PATTERN)


PrioritiesResponse = RuntimeConfigResponse
PrioritiesUpdateRequest = RuntimeConfigPatchRequest
