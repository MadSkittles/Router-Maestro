"""Auth module for router-maestro."""

from router_maestro.auth.discovery import (
    BUILTIN_PROVIDER_AUTH_DEFINITIONS,
    ProviderAuthDefinition,
    ProviderAuthSource,
    provider_auth_definitions,
)
from router_maestro.auth.manager import AuthManager, run_async
from router_maestro.auth.repository import CredentialRepository
from router_maestro.auth.storage import (
    ApiKeyCredential,
    AuthStorage,
    AuthType,
    Credential,
    OAuthCredential,
)

__all__ = [
    "AuthManager",
    "CredentialRepository",
    "AuthStorage",
    "AuthType",
    "Credential",
    "OAuthCredential",
    "ApiKeyCredential",
    "run_async",
    "ProviderAuthDefinition",
    "ProviderAuthSource",
    "BUILTIN_PROVIDER_AUTH_DEFINITIONS",
    "provider_auth_definitions",
]
