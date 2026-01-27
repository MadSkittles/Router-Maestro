"""Authentication middleware for API key validation."""

import os

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def get_server_api_key() -> str | None:
    """Get the server API key from environment variable."""
    return os.environ.get("ROUTER_MAESTRO_API_KEY")


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    """Verify the API key from the Authorization header.

    Accepts both:
    - Authorization: Bearer <api_key>
    - Authorization: <api_key>
    """
    server_api_key = get_server_api_key()

    if server_api_key is None:
        # No API key configured, allow all requests
        return

    # Skip auth for health and root endpoints
    if request.url.path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
        return

    # Get API key from header
    api_key: str | None = None

    if credentials:
        api_key = credentials.credentials
    else:
        # Try to get from Authorization header directly (without Bearer prefix)
        auth_header = request.headers.get("Authorization")
        if auth_header:
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]
            else:
                api_key = auth_header

    # Also check x-api-key header for Anthropic API compatibility
    if not api_key:
        api_key = request.headers.get("x-api-key")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Use 'Authorization: Bearer <api_key>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if api_key != server_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
