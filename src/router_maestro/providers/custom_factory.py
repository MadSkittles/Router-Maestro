"""Construction and credential resolution for custom providers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from router_maestro.auth.repository import CredentialRepository
from router_maestro.auth.storage import ApiKeyCredential
from router_maestro.config.providers import CustomProviderConfig, default_custom_api_key_env
from router_maestro.providers.openai_compat import OpenAICompatibleProvider


class CustomCredentialSource(StrEnum):
    """Configured source of a custom provider credential."""

    ENVIRONMENT = "environment"
    REPOSITORY = "repository"
    ANONYMOUS = "anonymous"


@dataclass(frozen=True, slots=True)
class ResolvedCustomCredential:
    """A resolved credential without exposing its value in repr output."""

    source: CustomCredentialSource
    api_key: str | None = field(repr=False)


def resolve_custom_provider_credential(
    provider_name: str,
    provider_config: CustomProviderConfig,
    *,
    credential_repository: CredentialRepository,
    environ: Mapping[str, str] | None = None,
) -> ResolvedCustomCredential | None:
    """Resolve environment, repository, then explicit anonymous access."""
    environment = os.environ if environ is None else environ
    env_name = provider_config.options.api_key_env or default_custom_api_key_env(provider_name)
    env_key = environment.get(env_name)
    if env_key:
        return ResolvedCustomCredential(CustomCredentialSource.ENVIRONMENT, env_key)

    credential = credential_repository.get_provider(provider_name)
    if isinstance(credential, ApiKeyCredential) and credential.key:
        return ResolvedCustomCredential(CustomCredentialSource.REPOSITORY, credential.key)

    if provider_config.options.allow_unauthenticated:
        return ResolvedCustomCredential(CustomCredentialSource.ANONYMOUS, None)
    return None


def create_custom_provider(
    provider_name: str,
    provider_config: CustomProviderConfig,
    *,
    credential_repository: CredentialRepository,
    environ: Mapping[str, str] | None = None,
) -> OpenAICompatibleProvider | None:
    """Build one supported provider when its credential policy is satisfied."""
    if provider_config.type != "openai-compatible":
        return None
    credential = resolve_custom_provider_credential(
        provider_name,
        provider_config,
        credential_repository=credential_repository,
        environ=environ,
    )
    if credential is None:
        return None
    return OpenAICompatibleProvider(
        name=provider_name,
        base_url=provider_config.baseURL,
        api_key=credential.api_key,
        models={model_id: config.name for model_id, config in provider_config.models.items()},
        allow_unauthenticated=credential.source is CustomCredentialSource.ANONYMOUS,
    )
