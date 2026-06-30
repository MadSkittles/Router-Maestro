"""Authentication middleware for API key validation."""

import hmac
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

    # Skip auth for health and root endpoints
    if request.url.path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
        return

    if server_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key is not configured",
        )

    # Collect every candidate key. Clients like Claude Code may send several
    # credential headers at once (e.g. an OAuth Authorization bearer alongside
    # the configured x-api-key); accept the request if ANY candidate matches so
    # the correct key is not ignored just because another header is present.
    candidates: list[str] = []

    if credentials:
        candidates.append(credentials.credentials)
    else:
        # Try to get from Authorization header directly (without Bearer prefix)
        auth_header = request.headers.get("Authorization")
        if auth_header:
            if auth_header.startswith("Bearer "):
                candidates.append(auth_header[7:])
            else:
                candidates.append(auth_header)

    # x-api-key (Anthropic) and x-goog-api-key (Gemini CLI) compatibility headers.
    for header_name in ("x-api-key", "x-goog-api-key"):
        value = request.headers.get(header_name)
        if value:
            candidates.append(value)

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Use 'Authorization: Bearer <api_key>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Compare on UTF-8 bytes: hmac.compare_digest raises TypeError on non-ASCII
    # str inputs, so a non-ASCII bearer token would otherwise 500 instead of 401.
    server_key_bytes = server_api_key.encode("utf-8")
    if not any(
        hmac.compare_digest(candidate.encode("utf-8"), server_key_bytes) for candidate in candidates
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
