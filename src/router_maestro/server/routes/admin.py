"""Admin API routes for remote management."""

import asyncio
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response

from router_maestro.auth import (
    ApiKeyCredential,
    AuthManager,
    AuthType,
    Credential,
    CredentialRepository,
    provider_auth_definitions,
)
from router_maestro.auth.github_oauth import (
    GitHubOAuthError,
    get_copilot_token,
    poll_access_token,
    request_device_code,
)
from router_maestro.auth.storage import OAuthCredential
from router_maestro.config.repository import (
    RuntimeConfigConflictError,
    RuntimeConfigRepository,
    RuntimeConfigSnapshot,
)
from router_maestro.config.settings import load_providers_config
from router_maestro.routing.model_ref import catalog_model_public_id
from router_maestro.routing.router import Router
from router_maestro.server.dependencies import (
    get_app_router,
    get_router_owner,
    get_runtime_config_repository,
)
from router_maestro.server.oauth_sessions import oauth_sessions
from router_maestro.server.schemas.admin import (
    AuthListResponse,
    AuthProviderDefinitionInfo,
    AuthProviderDefinitionsResponse,
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

logger = logging.getLogger("router_maestro.server.routes.admin")


def _runtime_config_response(snapshot: RuntimeConfigSnapshot) -> PrioritiesResponse:
    return PrioritiesResponse(**snapshot.config.model_dump(mode="json"), revision=snapshot.revision)


def _set_revision_etag(response: Response, revision: str) -> None:
    response.headers["ETag"] = f'"{revision}"'


async def _wait_for_task_ignoring_cancellation(task: asyncio.Task) -> bool:
    """Wait for a task to finish and report whether its caller was cancelled."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return cancelled


async def _compensate_cancelled_credential_write(
    *,
    provider: str,
    previous: Credential | None,
    replacement: Credential,
    credential_repository: CredentialRepository,
) -> None:
    """Conditionally undo a cancelled write without letting more cancellation interrupt it."""
    compensation_task = asyncio.create_task(
        asyncio.to_thread(
            credential_repository.compare_and_swap_provider,
            provider,
            expected=replacement,
            replacement=previous,
        )
    )
    await _wait_for_task_ignoring_cancellation(compensation_task)
    try:
        restored = compensation_task.result()
        if not restored:
            logger.info(
                "Credential compensation skipped for %s because it changed concurrently",
                provider,
            )
    except Exception:
        logger.exception("Failed to compensate cancelled credential mutation for %s", provider)


async def _rebuild_after_credential_mutation(
    *,
    provider: str,
    previous: Credential | None,
    replacement: Credential | None,
    credential_repository: CredentialRepository,
    runtime_config_repository: RuntimeConfigRepository,
    router_owner,
) -> None:
    """Rebuild for one credential mutation, compensating only our own write on failure."""
    try:
        snapshot = runtime_config_repository.read()
        await router_owner.rebuild(config_snapshot=snapshot)
    except BaseException:
        try:
            restored = credential_repository.compare_and_swap_provider(
                provider,
                expected=replacement,
                replacement=previous,
            )
            if not restored:
                logger.info(
                    "Credential compensation skipped for %s because it changed concurrently",
                    provider,
                )
        except Exception:
            logger.exception("Failed to compensate credential mutation for %s", provider)
        raise


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


@router.get("/auth/providers", response_model=AuthProviderDefinitionsResponse)
def list_auth_providers() -> AuthProviderDefinitionsResponse:
    """List non-secret authentication definitions configured on this server."""
    definitions = provider_auth_definitions(load_providers_config())
    return AuthProviderDefinitionsResponse(
        providers=[
            AuthProviderDefinitionInfo(
                provider=definition.provider,
                display_name=definition.display_name,
                auth_type=definition.auth_type,
                credential_required=definition.credential_required,
                source=definition.source,
                api_key_env=definition.api_key_env,
            )
            for definition in definitions
        ]
    )


@router.post("/auth/login")
async def login(
    request: LoginRequest,
    background_tasks: BackgroundTasks,
    router_owner=Depends(get_router_owner),
    runtime_config_repository: RuntimeConfigRepository = Depends(get_runtime_config_repository),
) -> OAuthInitResponse | dict:
    """Initiate login for a provider.

    For OAuth providers (github-copilot): Returns session info for device flow polling.
    For API key providers: Saves the key and returns success.
    """
    credential_repository = CredentialRepository()
    manager = AuthManager(credential_repository)

    if request.provider == "github-copilot":
        # OAuth device flow
        async with httpx.AsyncClient() as client:
            try:
                device_code = await request_device_code(client)
            except httpx.HTTPError as e:
                raise HTTPException(status_code=502, detail=f"Failed to get device code: {e}")

        # Create session for polling
        session = await oauth_sessions.create_session(
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
            router_owner,
            credential_repository,
            runtime_config_repository,
        )

        return OAuthInitResponse(
            session_id=session.session_id,
            user_code=device_code.user_code,
            verification_uri=device_code.verification_uri,
            expires_in=device_code.expires_in,
        )

    elif request.api_key:
        # API key auth
        previous = credential_repository.get_provider(request.provider)
        replacement = ApiKeyCredential(key=request.api_key)
        manager.login_api_key(request.provider, request.api_key)
        await _rebuild_after_credential_mutation(
            provider=request.provider,
            previous=previous,
            replacement=replacement,
            credential_repository=credential_repository,
            runtime_config_repository=runtime_config_repository,
            router_owner=router_owner,
        )
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
    router_owner,
    credential_repository: CredentialRepository,
    runtime_config_repository: RuntimeConfigRepository,
) -> None:
    """Background task to poll for OAuth completion and save credentials."""
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
            credential = OAuthCredential(
                refresh=access_token.access_token,
                access=copilot_token.token,
                expires=copilot_token.expires_at,
                api_endpoint=copilot_token.api_endpoint,
            )
            previous = await asyncio.to_thread(
                credential_repository.get_provider,
                "github-copilot",
            )
            update_task = asyncio.create_task(
                asyncio.to_thread(
                    credential_repository.update_provider,
                    "github-copilot",
                    credential,
                )
            )
            cancelled = await _wait_for_task_ignoring_cancellation(update_task)
            if cancelled:
                await _compensate_cancelled_credential_write(
                    provider="github-copilot",
                    previous=previous,
                    replacement=credential,
                    credential_repository=credential_repository,
                )
                raise asyncio.CancelledError
            update_task.result()

            await _rebuild_after_credential_mutation(
                provider="github-copilot",
                previous=previous,
                replacement=credential,
                credential_repository=credential_repository,
                runtime_config_repository=runtime_config_repository,
                router_owner=router_owner,
            )

            # Update session status
            await oauth_sessions.update_session_status(
                session_id,
                status="complete",
                access_token=copilot_token.token,
                refresh_token=access_token.access_token,
            )

        except GitHubOAuthError as e:
            await oauth_sessions.update_session_status(
                session_id,
                status="error",
                error=str(e),
            )
        except Exception as e:
            await oauth_sessions.update_session_status(
                session_id,
                status="error",
                error=f"Unexpected error: {e}",
            )


@router.get("/auth/oauth/status/{session_id}", response_model=OAuthStatusResponse)
async def get_oauth_status(session_id: str) -> OAuthStatusResponse:
    """Get OAuth session status for polling."""
    session = await oauth_sessions.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return OAuthStatusResponse(
        status=session.status,
        error=session.error,
    )


@router.delete("/auth/{provider}")
async def logout(
    provider: str,
    router_owner=Depends(get_router_owner),
    runtime_config_repository: RuntimeConfigRepository = Depends(get_runtime_config_repository),
) -> dict:
    """Log out from a provider."""
    credential_repository = CredentialRepository()
    manager = AuthManager(credential_repository)
    previous = credential_repository.get_provider(provider)

    if manager.logout(provider):
        await _rebuild_after_credential_mutation(
            provider=provider,
            previous=previous,
            replacement=None,
            credential_repository=credential_repository,
            runtime_config_repository=runtime_config_repository,
            router_owner=router_owner,
        )
        return {"success": True, "provider": provider}
    else:
        raise HTTPException(status_code=404, detail=f"Not authenticated with {provider}")


# ============================================================================
# Model endpoints
# ============================================================================


@router.get("/models", response_model=ModelsResponse)
async def list_models(model_router: Router = Depends(get_app_router)) -> ModelsResponse:
    """List all available models from authenticated providers."""
    try:
        models = await model_router.list_models()
        model_list = [
            ModelInfo(
                provider=model.provider,
                id=catalog_model_public_id(
                    model.provider,
                    model.id,
                    id_is_qualified=model.id_is_qualified,
                ),
                name=model.name,
                max_prompt_tokens=model.max_prompt_tokens,
                max_output_tokens=model.max_output_tokens,
                max_context_window_tokens=model.max_context_window_tokens,
                operation_capabilities=dict(model.operation_capabilities),
            )
            for model in models
        ]
        return ModelsResponse(models=model_list)
    except Exception:
        logger.error("Failed to list models", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/models/refresh")
async def refresh_models(
    router_owner=Depends(get_router_owner),
    runtime_config_repository: RuntimeConfigRepository = Depends(get_runtime_config_repository),
) -> dict:
    """Force refresh the models cache."""
    lease = None
    try:
        snapshot = runtime_config_repository.read()
        await router_owner.rebuild(config_snapshot=snapshot)
        lease = await router_owner.acquire()
        models = await lease.router.list_models()
        return {"success": True, "models_count": len(models)}
    except Exception:
        logger.error("Failed to refresh models", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to refresh models")
    finally:
        if lease is not None:
            await lease.release()


# ============================================================================
# Priority endpoints
# ============================================================================


@router.get("/priorities", response_model=PrioritiesResponse)
async def get_priorities(
    response: Response,
    repository: RuntimeConfigRepository = Depends(get_runtime_config_repository),
) -> PrioritiesResponse:
    """Get current priority configuration."""
    snapshot = repository.read()
    _set_revision_etag(response, snapshot.revision)
    return _runtime_config_response(snapshot)


@router.patch("/priorities", response_model=PrioritiesResponse)
@router.put("/priorities", response_model=PrioritiesResponse)
async def update_priorities(
    request: PrioritiesUpdateRequest,
    response: Response,
    repository: RuntimeConfigRepository = Depends(get_runtime_config_repository),
    router_owner=Depends(get_router_owner),
) -> PrioritiesResponse:
    """Replace runtime configuration using content-revision compare-and-swap."""
    replacement = request.model_dump(exclude={"revision"})
    persisted_snapshot: RuntimeConfigSnapshot | None = None

    try:
        current = repository.read()
        if current.revision != request.revision:
            raise RuntimeConfigConflictError(
                expected_revision=request.revision,
                current_revision=current.revision,
            )
        replacement_config = current.config.model_validate(replacement)
        candidate = repository.prepare(replacement_config)
        if candidate.revision == current.revision:
            _set_revision_etag(response, current.revision)
            return _runtime_config_response(current)

        def commit_candidate() -> RuntimeConfigSnapshot:
            nonlocal persisted_snapshot
            persisted_snapshot = repository.compare_and_swap(
                expected_revision=request.revision,
                replacement=replacement_config,
            )
            return persisted_snapshot

        await router_owner.rebuild(
            config_snapshot=candidate,
            before_swap=commit_candidate,
        )
    except RuntimeConfigConflictError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "config_revision_conflict",
                "message": "Runtime configuration changed; refresh and retry",
                "current_revision": error.current_revision,
            },
            headers={"ETag": f'"{error.current_revision}"'},
        ) from error

    if persisted_snapshot is None:
        raise RuntimeError("Router owner did not commit the runtime configuration candidate")
    _set_revision_etag(response, persisted_snapshot.revision)
    return _runtime_config_response(persisted_snapshot)
