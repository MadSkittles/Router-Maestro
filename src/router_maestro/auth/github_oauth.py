"""GitHub OAuth Device Flow implementation for Copilot."""

import time
from dataclasses import dataclass

import httpx

# GitHub OAuth constants (from copilot-api)
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

DEFAULT_POLL_INTERVAL = 5  # seconds


@dataclass
class DeviceCodeResponse:
    """Response from device code request."""

    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclass
class AccessTokenResponse:
    """Response from access token request."""

    access_token: str
    token_type: str
    scope: str


@dataclass
class CopilotTokenResponse:
    """Response from Copilot token request."""

    token: str
    expires_at: int
    refresh_in: int


class GitHubOAuthError(Exception):
    """Error during GitHub OAuth flow."""

    pass


async def request_device_code(client: httpx.AsyncClient) -> DeviceCodeResponse:
    """Request a device code from GitHub.

    Args:
        client: HTTP client

    Returns:
        Device code response with user_code and verification_uri
    """
    response = await client.post(
        GITHUB_DEVICE_CODE_URL,
        json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    response.raise_for_status()
    data = response.json()

    return DeviceCodeResponse(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_uri=data["verification_uri"],
        expires_in=data["expires_in"],
        interval=data.get("interval", DEFAULT_POLL_INTERVAL),
    )


async def poll_access_token(
    client: httpx.AsyncClient,
    device_code: str,
    interval: int = DEFAULT_POLL_INTERVAL,
    timeout: int = 900,
) -> AccessTokenResponse:
    """Poll GitHub for access token after user authorization.

    Args:
        client: HTTP client
        device_code: Device code from request_device_code
        interval: Polling interval in seconds
        timeout: Maximum time to wait in seconds

    Returns:
        Access token response

    Raises:
        GitHubOAuthError: If authorization fails or times out
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = await client.post(
            GITHUB_ACCESS_TOKEN_URL,
            json={
                "client_id": GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        if "access_token" in data:
            return AccessTokenResponse(
                access_token=data["access_token"],
                token_type=data["token_type"],
                scope=data["scope"],
            )

        error = data.get("error")
        if error == "authorization_pending":
            # User hasn't authorized yet, keep polling
            await _async_sleep(interval)
        elif error == "slow_down":
            # We're polling too fast, increase interval
            interval += 5
            await _async_sleep(interval)
        elif error == "expired_token":
            raise GitHubOAuthError("Device code expired. Please try again.")
        elif error == "access_denied":
            raise GitHubOAuthError("Authorization denied by user.")
        else:
            raise GitHubOAuthError(f"Unknown error: {error}")

    raise GitHubOAuthError("Authorization timed out. Please try again.")


async def get_copilot_token(
    client: httpx.AsyncClient,
    github_token: str,
) -> CopilotTokenResponse:
    """Exchange GitHub token for Copilot token.

    Args:
        client: HTTP client
        github_token: GitHub access token

    Returns:
        Copilot token response
    """
    # Headers matching copilot-api's githubHeaders
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Editor-Version": "vscode/1.104.3",
        "Editor-Plugin-Version": "copilot-chat/0.26.7",
        "User-Agent": "GitHubCopilotChat/0.26.7",
        "X-GitHub-Api-Version": "2025-04-01",
        "X-Vscode-User-Agent-Library-Version": "electron-fetch",
    }

    response = await client.get(
        COPILOT_TOKEN_URL,
        headers=headers,
    )
    response.raise_for_status()
    data = response.json()

    return CopilotTokenResponse(
        token=data["token"],
        expires_at=data["expires_at"],
        refresh_in=data.get("refresh_in", 1800000),  # Default 30 minutes in ms
    )


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio

    await asyncio.sleep(seconds)
