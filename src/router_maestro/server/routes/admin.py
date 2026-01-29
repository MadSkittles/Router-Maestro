"""Admin API routes for remote management."""

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from router_maestro.auth import AuthManager, AuthType
from router_maestro.auth.github_oauth import (
    GitHubOAuthError,
    get_copilot_token,
    poll_access_token,
    request_device_code,
)
from router_maestro.auth.storage import OAuthCredential
from router_maestro.config import (
    load_priorities_config,
    save_priorities_config,
)
from router_maestro.routing import get_router, reset_router
from router_maestro.server.oauth_sessions import oauth_sessions
from router_maestro.server.schemas.admin import (
    AuthListResponse,
    AuthProviderInfo,
    LoginRequest,
    ModelInfo,
    ModelsResponse,
    OAuthInitResponse,
    OAuthStatusResponse,
    PrioritiesResponse,
    PrioritiesUpdateRequest,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ============================================================================
# Auth endpoints
# ============================================================================


@router.get("/auth", response_model=AuthListResponse)
async def list_auth() -> AuthListResponse:
    """List all authenticated providers."""
    manager = AuthManager()
    providers = []

    for provider_name in manager.list_authenticated():
        cred = manager.get_credential(provider_name)
        if cred:
            auth_type = "oauth" if cred.type == AuthType.OAUTH else "api"
            # For OAuth, check if token might be expired
            status = "active"
            if isinstance(cred, OAuthCredential) and cred.expires > 0:
                import time

                if cred.expires < time.time():
                    status = "expired"

            providers.append(
                AuthProviderInfo(
                    provider=provider_name,
                    auth_type=auth_type,
                    status=status,
                )
            )

    return AuthListResponse(providers=providers)


@router.post("/auth/login")
async def login(
    request: LoginRequest,
    background_tasks: BackgroundTasks,
) -> OAuthInitResponse | dict:
    """Initiate login for a provider.

    For OAuth providers (github-copilot): Returns session info for device flow polling.
    For API key providers: Saves the key and returns success.
    """
    manager = AuthManager()

    if request.provider == "github-copilot":
        # OAuth device flow
        async with httpx.AsyncClient() as client:
            try:
                device_code = await request_device_code(client)
            except httpx.HTTPError as e:
                raise HTTPException(status_code=502, detail=f"Failed to get device code: {e}")

        # Create session for polling
        session = oauth_sessions.create_session(
            provider=request.provider,
            device_code=device_code.device_code,
            user_code=device_code.user_code,
            verification_uri=device_code.verification_uri,
            expires_in=device_code.expires_in,
            interval=device_code.interval,
        )

        # Start background task to poll for token
        background_tasks.add_task(
            _poll_oauth_completion,
            session.session_id,
            device_code.device_code,
            device_code.interval,
        )

        return OAuthInitResponse(
            session_id=session.session_id,
            user_code=device_code.user_code,
            verification_uri=device_code.verification_uri,
            expires_in=device_code.expires_in,
        )

    elif request.api_key:
        # API key auth
        manager.login_api_key(request.provider, request.api_key)
        # Reset router to pick up new authentication
        reset_router()
        return {"success": True, "provider": request.provider}

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{request.provider}' requires an API key",
        )


async def _poll_oauth_completion(
    session_id: str,
    device_code: str,
    interval: int,
) -> None:
    """Background task to poll for OAuth completion and save credentials."""
    manager = AuthManager()

    async with httpx.AsyncClient() as client:
        try:
            # Poll for access token
            access_token = await poll_access_token(
                client,
                device_code,
                interval=interval,
                timeout=900,  # 15 minutes
            )

            # Get Copilot token
            copilot_token = await get_copilot_token(client, access_token.access_token)

            # Save credentials
            manager.storage.set(
                "github-copilot",
                OAuthCredential(
                    refresh=access_token.access_token,
                    access=copilot_token.token,
                    expires=copilot_token.expires_at,
                ),
            )
            manager.save()

            # Update session status
            oauth_sessions.update_session_status(
                session_id,
                status="complete",
                access_token=copilot_token.token,
                refresh_token=access_token.access_token,
            )

            # Reset router to pick up new authentication
            reset_router()

        except GitHubOAuthError as e:
            oauth_sessions.update_session_status(
                session_id,
                status="error",
                error=str(e),
            )
        except Exception as e:
            oauth_sessions.update_session_status(
                session_id,
                status="error",
                error=f"Unexpected error: {e}",
            )


@router.get("/auth/oauth/status/{session_id}", response_model=OAuthStatusResponse)
async def get_oauth_status(session_id: str) -> OAuthStatusResponse:
    """Get OAuth session status for polling."""
    session = oauth_sessions.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return OAuthStatusResponse(
        status=session.status,
        error=session.error,
    )


@router.delete("/auth/{provider}")
async def logout(provider: str) -> dict:
    """Log out from a provider."""
    manager = AuthManager()

    if manager.logout(provider):
        # Reset router to reflect authentication change
        reset_router()
        return {"success": True, "provider": provider}
    else:
        raise HTTPException(status_code=404, detail=f"Not authenticated with {provider}")


# ============================================================================
# Model endpoints
# ============================================================================


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List all available models from authenticated providers."""
    router_instance = get_router()

    try:
        models = await router_instance.list_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")

    model_list = [
        ModelInfo(
            provider=model.provider,
            id=model.id,
            name=model.name,
        )
        for model in models
    ]

    return ModelsResponse(models=model_list)


@router.post("/models/refresh")
async def refresh_models() -> dict:
    """Force refresh the models cache."""
    router_instance = get_router()
    router_instance.invalidate_cache()
    # Trigger re-population
    try:
        models = await router_instance.list_models()
        return {"success": True, "models_count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh models: {e}")


# ============================================================================
# Priority endpoints
# ============================================================================


@router.get("/priorities", response_model=PrioritiesResponse)
async def get_priorities() -> PrioritiesResponse:
    """Get current priority configuration."""
    config = load_priorities_config()

    return PrioritiesResponse(
        priorities=config.priorities,
        fallback=config.fallback.model_dump(),
    )


@router.put("/priorities", response_model=PrioritiesResponse)
async def update_priorities(request: PrioritiesUpdateRequest) -> PrioritiesResponse:
    """Update priority configuration."""
    config = load_priorities_config()

    # Update priorities
    config.priorities = request.priorities

    # Update fallback if provided
    if request.fallback is not None:
        from router_maestro.config import FallbackConfig

        config.fallback = FallbackConfig.model_validate(request.fallback)

    save_priorities_config(config)

    return PrioritiesResponse(
        priorities=config.priorities,
        fallback=config.fallback.model_dump(),
    )
