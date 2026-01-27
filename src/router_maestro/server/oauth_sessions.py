"""OAuth session management for remote OAuth flows."""

import secrets
import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class OAuthSession:
    """An OAuth session for device flow authentication."""

    session_id: str
    provider: str
    device_code: str
    user_code: str
    verification_uri: str
    expires_at: float
    interval: int
    status: str = "pending"  # pending, complete, expired, error
    error: str | None = None
    # Result data when complete
    access_token: str | None = None
    refresh_token: str | None = None


class OAuthSessionManager:
    """Manages OAuth sessions for remote device flow authentication."""

    def __init__(self, session_timeout: int = 900) -> None:
        """Initialize the session manager.

        Args:
            session_timeout: Default session timeout in seconds (default: 15 minutes)
        """
        self._sessions: dict[str, OAuthSession] = {}
        self._lock = Lock()
        self._session_timeout = session_timeout

    def create_session(
        self,
        provider: str,
        device_code: str,
        user_code: str,
        verification_uri: str,
        expires_in: int,
        interval: int = 5,
    ) -> OAuthSession:
        """Create a new OAuth session.

        Args:
            provider: Provider name (e.g., 'github-copilot')
            device_code: Device code from OAuth provider
            user_code: User code to display
            verification_uri: URL for user to visit
            expires_in: Seconds until expiration
            interval: Polling interval in seconds

        Returns:
            The created OAuth session
        """
        session_id = secrets.token_urlsafe(16)
        session = OAuthSession(
            session_id=session_id,
            provider=provider,
            device_code=device_code,
            user_code=user_code,
            verification_uri=verification_uri,
            expires_at=time.time() + expires_in,
            interval=interval,
        )

        with self._lock:
            self._sessions[session_id] = session
            self._cleanup_expired()

        return session

    def get_session(self, session_id: str) -> OAuthSession | None:
        """Get a session by ID.

        Args:
            session_id: Session ID to look up

        Returns:
            The session if found and not expired, None otherwise
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            # Check if expired
            if session.status == "pending" and time.time() > session.expires_at:
                session.status = "expired"

            return session

    def update_session_status(
        self,
        session_id: str,
        status: str,
        error: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
    ) -> bool:
        """Update session status.

        Args:
            session_id: Session ID to update
            status: New status
            error: Error message if status is 'error'
            access_token: Access token if status is 'complete'
            refresh_token: Refresh token if status is 'complete'

        Returns:
            True if session was updated, False if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False

            session.status = status
            session.error = error
            session.access_token = access_token
            session.refresh_token = refresh_token
            return True

    def remove_session(self, session_id: str) -> bool:
        """Remove a session.

        Args:
            session_id: Session ID to remove

        Returns:
            True if session was removed, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def _cleanup_expired(self) -> None:
        """Remove expired sessions. Must be called with lock held."""
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if session.status in ("complete", "error", "expired")
            or now > session.expires_at + 60  # Keep for 1 minute after expiry
        ]
        for sid in expired:
            del self._sessions[sid]


# Global session manager instance
oauth_sessions = OAuthSessionManager()
