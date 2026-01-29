"""Admin client for CLI operations.

All CLI commands use HTTP API to communicate with the server.
This ensures consistent behavior between local and remote contexts.
"""

import httpx

from router_maestro.config import load_contexts_config


class AdminClientError(Exception):
    """Error from admin client operations."""

    pass


class ServerNotRunningError(AdminClientError):
    """Server is not running."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        super().__init__(
            f"Server is not running at {endpoint}. Start it with: router-maestro server start"
        )


class AdminClient:
    """HTTP client for server admin operations.

    All operations go through the HTTP API, ensuring consistent behavior
    whether connecting to a local or remote server.
    """

    def __init__(self, endpoint: str, api_key: str | None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _handle_connection_error(self, e: Exception) -> None:
        """Handle connection errors with helpful messages."""
        if isinstance(e, httpx.ConnectError):
            raise ServerNotRunningError(self.endpoint) from e
        raise AdminClientError(f"Request failed: {e}") from e

    async def list_auth(self) -> list[dict]:
        """List authenticated providers.

        Returns:
            List of dicts with provider, auth_type, status
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/api/admin/auth",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()
                return data.get("providers", [])
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return []  # unreachable, for type checker

    async def login_oauth(self, provider: str) -> dict:
        """Initiate OAuth login.

        Args:
            provider: Provider name (e.g., 'github-copilot')

        Returns:
            Dict with session_id, user_code, verification_uri, expires_in
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoint}/api/admin/auth/login",
                    headers=self._get_headers(),
                    json={"provider": provider},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return {}

    async def login_api_key(self, provider: str, api_key: str) -> bool:
        """Login with API key.

        Args:
            provider: Provider name
            api_key: API key

        Returns:
            True if successful
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoint}/api/admin/auth/login",
                    headers=self._get_headers(),
                    json={"provider": provider, "api_key": api_key},
                )
                response.raise_for_status()
                data = response.json()
                return data.get("success", False)
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return False

    async def logout(self, provider: str) -> bool:
        """Logout from a provider.

        Args:
            provider: Provider name

        Returns:
            True if successful
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.endpoint}/api/admin/auth/{provider}",
                    headers=self._get_headers(),
                )
                if response.status_code == 404:
                    return False
                response.raise_for_status()
                data = response.json()
                return data.get("success", False)
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return False

    async def poll_oauth_status(self, session_id: str) -> dict:
        """Poll OAuth session status.

        Args:
            session_id: Session ID from login_oauth

        Returns:
            Dict with status ('pending', 'complete', 'expired', 'error') and optional error
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/api/admin/auth/oauth/status/{session_id}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return {}

    async def list_models(self) -> list[dict]:
        """List available models.

        Returns:
            List of dicts with provider, id, name
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/api/admin/models",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return []

    async def refresh_models(self) -> bool:
        """Refresh the models cache on the server.

        Returns:
            True if successful
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoint}/api/admin/models/refresh",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return True
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return False

    async def get_priorities(self) -> dict:
        """Get priority configuration.

        Returns:
            Dict with priorities list and fallback config
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/api/admin/priorities",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return {}

    async def set_priorities(self, priorities: list[str], fallback: dict | None = None) -> bool:
        """Set priority configuration.

        Args:
            priorities: List of model keys (provider/model)
            fallback: Optional fallback configuration

        Returns:
            True if successful
        """
        try:
            async with httpx.AsyncClient() as client:
                payload: dict = {"priorities": priorities}
                if fallback is not None:
                    payload["fallback"] = fallback

                response = await client.put(
                    f"{self.endpoint}/api/admin/priorities",
                    headers=self._get_headers(),
                    json=payload,
                )
                response.raise_for_status()
                return True
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return False

    async def test_connection(self) -> dict:
        """Test connection to the server.

        Returns:
            Dict with server info (name, version, status)

        Raises:
            ServerNotRunningError: If server is not running
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.endpoint}/",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            self._handle_connection_error(e)
            return {}


def get_admin_client() -> AdminClient:
    """Get admin client for current context.

    Returns:
        AdminClient configured with current context's endpoint and API key
    """
    config = load_contexts_config()
    ctx = config.contexts.get(config.current)

    if not ctx:
        # Fallback to default local endpoint
        return AdminClient("http://localhost:8080", None)

    return AdminClient(ctx.endpoint, ctx.api_key)


def get_current_endpoint() -> str:
    """Get the endpoint for the current context.

    Returns:
        The endpoint URL
    """
    config = load_contexts_config()
    ctx = config.contexts.get(config.current)
    return ctx.endpoint if ctx else "http://localhost:8080"
