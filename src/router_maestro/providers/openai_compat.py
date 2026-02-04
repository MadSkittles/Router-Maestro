"""OpenAI-compatible provider for custom endpoints."""

import httpx

from router_maestro.providers.base import ChatRequest, ModelInfo
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.utils import get_logger

logger = get_logger("providers.openai_compat")


class OpenAICompatibleProvider(OpenAIChatProvider):
    """OpenAI-compatible provider for custom endpoints."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        models: dict[str, str] | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            name: Provider name
            base_url: Base URL for API requests
            api_key: API key for authentication
            models: Dict of model_id -> display_name
        """
        self.name = name
        super().__init__(base_url=base_url, logger=logger)
        self.api_key = api_key
        self._models = models or {}

    def is_authenticated(self) -> bool:
        """Check if authenticated (always true for custom providers)."""
        return bool(self.api_key)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_payload_extra(self, request: ChatRequest) -> dict:
        return request.extra

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
                data = response.json()

                return [
                    ModelInfo(id=model["id"], name=model["id"], provider=self.name)
                    for model in data.get("data", [])
                ]
            except httpx.HTTPError:
                return []
