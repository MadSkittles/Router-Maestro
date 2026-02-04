"""OpenAI provider implementation."""

import httpx

from router_maestro.auth import AuthManager, AuthType
from router_maestro.providers.base import ModelInfo, ProviderError
from router_maestro.providers.openai_base import OpenAIChatProvider
from router_maestro.utils import get_logger

logger = get_logger("providers.openai")

OPENAI_API_URL = "https://api.openai.com/v1"


class OpenAIProvider(OpenAIChatProvider):
    """OpenAI official provider."""

    name = "openai"

    def __init__(self, base_url: str = OPENAI_API_URL) -> None:
        super().__init__(base_url=base_url, logger=logger)
        self.auth_manager = AuthManager()

    def is_authenticated(self) -> bool:
        """Check if authenticated with OpenAI."""
        cred = self.auth_manager.get_credential("openai")
        return cred is not None and cred.type == AuthType.API_KEY

    def _get_api_key(self) -> str:
        """Get the API key."""
        cred = self.auth_manager.get_credential("openai")
        if not cred or cred.type != AuthType.API_KEY:
            logger.error("Not authenticated with OpenAI")
            raise ProviderError("Not authenticated with OpenAI", status_code=401)
        return cred.key

    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenAI API requests."""
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    async def list_models(self) -> list[ModelInfo]:
        """List available OpenAI models."""
        logger.debug("Fetching OpenAI models")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                models = []
                for model in data.get("data", []):
                    model_id = model["id"]
                    # Filter to chat models
                    if any(x in model_id for x in ["gpt-", "o1-", "o3-"]):
                        models.append(
                            ModelInfo(
                                id=model_id,
                                name=model_id,
                                provider=self.name,
                            )
                        )
                logger.info("Fetched %d OpenAI models", len(models))
                return models
            except httpx.HTTPError as e:
                logger.warning("Failed to list OpenAI models, using defaults: %s", e)
                # Return default models on error
                return [
                    ModelInfo(id="gpt-4o", name="GPT-4o", provider=self.name),
                    ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini", provider=self.name),
                    ModelInfo(id="gpt-4-turbo", name="GPT-4 Turbo", provider=self.name),
                ]
