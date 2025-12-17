"""Authentication configuration and token management."""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field


class TokenData(BaseModel):
    """Token data structure."""

    user_id: str
    session_id: str
    expires_at: datetime
    scopes: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class AuthConfig:
    """Authentication configuration and utilities."""

    def __init__(self, secret_key: str, token_expiry: int = 3600):
        """
        Initialize authentication configuration.

        Args:
            secret_key: Secret key for token encryption
            token_expiry: Token expiry time in seconds
        """
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        # Generate Fernet key from secret
        self._fernet = Fernet(self._derive_key(secret_key))

    @staticmethod
    def _derive_key(secret: str) -> bytes:
        """Derive a Fernet key from secret string."""
        import hashlib
        import base64

        # Use SHA256 to derive 32 bytes, then base64 encode for Fernet
        hash_digest = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hash_digest)

    def generate_session_id(self) -> str:
        """Generate a secure session ID."""
        return secrets.token_urlsafe(32)

    def generate_token(self, token_data: TokenData) -> str:
        """
        Generate an encrypted token.

        Args:
            token_data: Token data to encrypt

        Returns:
            Encrypted token string
        """
        payload = token_data.model_dump_json()
        encrypted = self._fernet.encrypt(payload.encode())
        return encrypted.decode()

    def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decrypt a token.

        Args:
            token: Encrypted token string

        Returns:
            TokenData if valid, None otherwise
        """
        import json

        try:
            decrypted = self._fernet.decrypt(token.encode())
            data = json.loads(decrypted.decode())
            token_data = TokenData(**data)

            # Check expiration
            if token_data.expires_at < datetime.now():
                return None

            return token_data
        except Exception:
            return None

    def create_access_token(
        self, user_id: str, session_id: str, scopes: Optional[list[str]] = None
    ) -> str:
        """
        Create an access token for a user session.

        Args:
            user_id: User identifier
            session_id: Session identifier
            scopes: List of permission scopes

        Returns:
            Encrypted access token
        """
        expires_at = datetime.now() + timedelta(seconds=self.token_expiry)
        token_data = TokenData(
            user_id=user_id,
            session_id=session_id,
            expires_at=expires_at,
            scopes=scopes or [],
        )
        return self.generate_token(token_data)