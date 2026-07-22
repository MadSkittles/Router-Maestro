"""Auth storage for credentials."""

import json
import logging
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from router_maestro.config.paths import AUTH_FILE
from router_maestro.config.settings import write_json_owner_only

logger = logging.getLogger("router_maestro.auth.storage")


class AuthType(StrEnum):
    """Authentication type."""

    OAUTH = "oauth"
    API_KEY = "api"


class OAuthCredential(BaseModel):
    """OAuth credential storage."""

    type: AuthType = AuthType.OAUTH
    refresh: str = Field(..., description="Refresh token")
    access: str = Field(..., description="Access token")
    expires: int = Field(default=0, description="Expiration timestamp (0 = never)")
    api_endpoint: str | None = Field(
        default=None,
        description="Provider API endpoint returned with the OAuth token",
    )


class ApiKeyCredential(BaseModel):
    """API key credential storage."""

    type: AuthType = AuthType.API_KEY
    key: str = Field(..., description="API key")


Credential = OAuthCredential | ApiKeyCredential


class AuthStorage(BaseModel):
    """Root storage for all credentials."""

    credentials: dict[str, Credential] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = AUTH_FILE) -> AuthStorage:
        """Load credentials from file."""
        if not path.exists():
            return cls()

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read auth file %s (%s); starting with no credentials", path, e)
            return cls()

        # A valid JSON file whose top level is not an object (e.g. `[]`, `null`,
        # a bare string) would otherwise crash on `.items()`.
        if not isinstance(data, dict):
            logger.error(
                "Auth file %s top-level is %s, expected an object; starting with no credentials",
                path,
                type(data).__name__,
            )
            return cls()

        # Parse credentials based on type
        credentials = {}
        for name, cred_data in data.items():
            cred_type = cred_data.get("type") if isinstance(cred_data, dict) else None
            try:
                if cred_type == "oauth":
                    credentials[name] = OAuthCredential.model_validate(cred_data)
                elif cred_type == "api":
                    credentials[name] = ApiKeyCredential.model_validate(cred_data)
                else:
                    logger.warning(
                        "Unknown credential type %r for provider %r; skipping", cred_type, name
                    )
            except ValidationError:
                # Do NOT log the ValidationError — it echoes the offending field's
                # input value, which may be a token/key. Log only the provider name.
                logger.warning("Invalid credential for provider %r; skipping", name)

        return cls(credentials=credentials)

    def save(self, path: Path = AUTH_FILE) -> None:
        """Save credentials to file."""
        # Convert to dict format matching the spec
        data = {}
        for name, cred in self.credentials.items():
            data[name] = cred.model_dump(mode="json")

        write_json_owner_only(path, data)

    def get(self, provider: str) -> Credential | None:
        """Get credential for a provider."""
        return self.credentials.get(provider)

    def set(self, provider: str, credential: Credential) -> None:
        """Set credential for a provider."""
        self.credentials[provider] = credential

    def remove(self, provider: str) -> bool:
        """Remove credential for a provider. Returns True if removed."""
        if provider in self.credentials:
            del self.credentials[provider]
            return True
        return False

    def list_providers(self) -> list[str]:
        """List all authenticated providers."""
        return list(self.credentials.keys())
