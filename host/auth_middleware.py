"""Authentication middleware for session and token validation."""

from typing import Callable, Optional, List, Awaitable
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from config.settings import get_settings
from config.auth_config import AuthConfig
from shared.auth import TokenValidator
from shared.logging_config import get_logger
from host.session_manager import SessionManager, SessionState

logger = get_logger(__name__)

class AuthenticationError(Exception):
    """Authentication-related errors."""
    pass

class AuthorizationError(Exception):
    """Authorization-related errors."""
    pass

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for validating sessions and tokens.
    
    Validates Bearer tokens, attaches session info to request state, and checks permissions.
    """

    def __init__(
        self,
        app,
        auth_config: AuthConfig,
        session_manager: SessionManager,
        skip_paths: Optional[List[str]] = None,
    ):
        """
        Initialize authentication middleware.

        Args:
            app: ASGI application
            auth_config: Authentication configuration
            session_manager: Session manager instance
            skip_paths: List of paths to skip authentication
        """
        super().__init__(app)
        self.auth_config = auth_config
        self.session_manager = session_manager
        self.skip_paths = set(skip_paths or [])
        self.settings = get_settings()
        self.token_validator = TokenValidator(auth_config)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[JSONResponse]],
    ) -> JSONResponse:
        """
        Process request and enforce authentication.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint

        Returns:
            HTTP response
        """
        # Skip authentication for specified paths
        if request.url.path in self.skip_paths:
            logger.debug("auth_skipped", path=request.url.path)
            return await call_next(request)

        # Check if authentication is enabled
        if not self.settings.auth_enabled:
            logger.debug("auth_disabled")
            return await call_next(request)

        try:
            # Authenticate request
            session = await self.authenticate_request(
                session_id=request.headers.get("X-Session-ID"),
                token=self._extract_token(request),
            )

            # Attach session info to request state
            request.state.session_info = session
            logger.debug(
                "auth_successful",
                session_id=session.session_id,
                user_id=session.user_id,
                path=request.url.path,
            )
        except AuthenticationError as e:
            logger.warning("authentication_failed", error=str(e))
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={"error": str(e)},
            )
        except AuthorizationError as e:
            logger.warning("authorization_failed", error=str(e))
            return JSONResponse(
                status_code=HTTP_403_FORBIDDEN,
                content={"error": str(e)},
            )

        return await call_next(request)

    async def authenticate_request(
        self,
        session_id: Optional[str] = None,
        token: Optional[str] = None,
        required_scopes: Optional[List[str]] = None,
    ) -> SessionState:
        """
        Authenticate a request.

        Args:
            session_id: Session identifier
            token: Bearer token
            required_scopes: List of required scopes

        Returns:
            SessionState if authentication is successful

        Raises:
            AuthenticationError: If authentication fails
            AuthorizationError: If authorization fails
        """
        if not session_id:
            raise AuthenticationError("Session ID required")

        is_valid, session = await self.session_manager.validate_session(session_id, token)
        if not is_valid or not session:
            raise AuthenticationError("Invalid session or token")

        if required_scopes and session.token_data:
            missing_scopes = [
                scope for scope in required_scopes if scope not in session.token_data.scopes
            ]
            if missing_scopes:
                raise AuthorizationError(f"Missing scopes: {missing_scopes}")

        return session

    def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract Bearer token from Authorization header.

        Args:
            request: HTTP request

        Returns:
            Token string or None
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        try:
            scheme, token = auth_header.split(maxsplit=1)
            if scheme.lower() != "bearer":
                return None
            return token
        except ValueError:
            return None

    async def check_permission(
        self, session_id: str, token: Optional[str], resource: str, action: str
    ) -> bool:
        """
        Check if session has permission for resource/action.

        Args:
            session_id: Session identifier
            token: Bearer token
            resource: Resource name
            action: Action name

        Returns:
            True if permission is granted, False otherwise
        """
        try:
            session = await self.authenticate_request(session_id=session_id, token=token)
            required_scope = f"{resource}:{action}"

            if session.token_data:
                has_wildcard = f"{resource}:*" in session.token_data.scopes
                has_specific = required_scope in session.token_data.scopes
                return has_wildcard or has_specific
            return False
        except (AuthenticationError, AuthorizationError):
            return False