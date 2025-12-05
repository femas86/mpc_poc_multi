"""Authentication utilities and decorators."""

import functools
from typing import Any, Callable, Optional

from config.auth_config import AuthConfig, TokenData
from shared.logging_config import get_logger

logger = get_logger(__name__)


class TokenValidator:
    """Token validation utility."""

    def __init__(self, auth_config: AuthConfig):
        """
        Initialize token validator.

        Args:
            auth_config: Authentication configuration instance
        """
        self.auth_config = auth_config

    async def validate(self, token: str) -> Optional[TokenData]:
        """
        Validate an authentication token.

        Args:
            token: Token string to validate

        Returns:
            TokenData if valid, None otherwise
        """
        try:
            token_data = self.auth_config.verify_token(token)
            if token_data:
                logger.debug(
                    "token_validated",
                    user_id=token_data.user_id,
                    session_id=token_data.session_id,
                )
                return token_data
            else:
                logger.warning("invalid_token_validation_failed")
                return None
        except Exception as e:
            logger.error("token_validation_error", error=str(e))
            return None


def require_auth(scopes: Optional[list[str]] = None) -> Callable:
    """
    Decorator to require authentication for a function.

    Args:
        scopes: Required permission scopes

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract token from kwargs
            token = kwargs.get("token")
            validator = kwargs.get("token_validator")

            if not token or not validator:
                raise PermissionError("Authentication required")

            token_data = await validator.validate(token)
            if not token_data:
                raise PermissionError("Invalid or expired token")

            # Check scopes if required
            if scopes and not all(scope in token_data.scopes for scope in scopes):
                raise PermissionError(f"Missing required scopes: {scopes}")

            # Add token_data to kwargs for use in function
            kwargs["token_data"] = token_data

            return await func(*args, **kwargs)

        return wrapper

    return decorator