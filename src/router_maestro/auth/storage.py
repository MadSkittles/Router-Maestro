"""Auth storage for credentials."""

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from router_maestro.config.paths import AUTH_FILE


class AuthType(str, Enum):
    """Authentication type."""

    OAUTH = "oauth"
    API_KEY = "api"


class OAuthCredential(BaseModel):
    """OAuth credential storage."""

    type: AuthType = AuthType.OAUTH
    refresh: str = Field(..., description="Refresh token")
    access: str = Field(..., description="Access token")
    expires: int = Field(default=0, description="Expiration timestamp (0 = never)")


class ApiKeyCredential(BaseModel):
    """API key credential storage."""

    type: AuthType = AuthType.API_KEY
    key: str = Field(..., description="API key")


Credential = OAuthCredential | ApiKeyCredential


class AuthStorage(BaseModel):
    """Root storage for all credentials."""

    credentials: dict[str, Credential] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = AUTH_FILE) -> "AuthStorage":
        """Load credentials from file."""
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Parse credentials based on type
        credentials = {}
        for name, cred_data in data.items():
            if cred_data.get("type") == "oauth":
                credentials[name] = OAuthCredential.model_validate(cred_data)
            elif cred_data.get("type") == "api":
                credentials[name] = ApiKeyCredential.model_validate(cred_data)

        return cls(credentials=credentials)

    def save(self, path: Path = AUTH_FILE) -> None:
        """Save credentials to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict format matching the spec
        data = {}
        for name, cred in self.credentials.items():
            data[name] = cred.model_dump(mode="json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

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
