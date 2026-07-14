"""OpenAI-compatible provider for custom endpoints."""

import httpx

from router_maestro.providers.base import ModelInfo
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.utils import get_logger

logger = get_logger("providers.openai_compat")


class OpenAICompatibleProvider(OpenAIChatProvider):
    """OpenAI-compatible provider for custom endpoints."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None,
        models: dict[str, str] | None = None,
        *,
        allow_unauthenticated: bool = False,
    ) -> None:
        """Initialize the provider.

        Args:
            name: Provider name
            base_url: Base URL for API requests
            api_key: API key for authentication
            models: Dict of model_id -> display_name
            allow_unauthenticated: Whether the configured endpoint explicitly permits no key
        """
        self.name = name
        super().__init__(base_url=base_url, logger=logger)
        self.api_key = api_key
        self.allow_unauthenticated = allow_unauthenticated
        self._models = models or {}

    def is_authenticated(self) -> bool:
        """Check whether a key exists or anonymous access was explicitly allowed."""
        return bool(self.api_key) or self.allow_unauthenticated

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _error_label(self) -> str:
        return self.name

    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
        if self._models:
            return [
                ModelInfo(id=model_id, name=name, provider=self.name)
                for model_id, name in self._models.items()
            ]

        # Try to fetch from API
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                model_ids = self._parse_model_catalog(response)

                return [
                    ModelInfo(id=model_id, name=model_id, provider=self.name)
                    for model_id in model_ids
                ]
            except httpx.HTTPError:
                return []
