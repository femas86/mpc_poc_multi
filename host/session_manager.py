"""Session management with authentication, lifecycle, and cleanup."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
from uuid import uuid4

from pydantic import BaseModel, Field

from config.auth_config import AuthConfig, TokenData
from config.settings import get_settings
from shared.auth import TokenValidator
from shared.logging_config import get_logger

logger = get_logger(__name__)

class SessionState(BaseModel):
    """Session state information."""

    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(default=0, description="Number of accesses")
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = Field(default=True)
    token_data: Optional[TokenData] = Field(default=None)

class SessionManager:
    """Manages user sessions with authentication and automatic cleanup."""

    def __init__(
        self,
        auth_config: Optional[AuthConfig] = None,
        max_age: Optional[int] = None,
        cleanup_interval: Optional[int] = None,
    ):
        settings = get_settings()
        
        if auth_config is None and settings.auth_enabled:
            auth_config = AuthConfig(
                secret_key=settings.auth_secret_key,
                token_expiry=settings.auth_token_expiry,
            )
        self.auth_config = auth_config
        self.token_validator = TokenValidator(auth_config) if auth_config else None
        
        self.max_age = max_age or settings.session_max_age
        self.cleanup_interval = cleanup_interval or settings.session_cleanup_interval
        
        self._sessions: Dict[str, SessionState] = {}
        self._user_sessions: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the session manager and cleanup task."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("session_manager_started")

    async def stop(self) -> None:
        """Stop the session manager."""
        if not self._running:
            return
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await asyncio.shield(self._cleanup_task)
            except asyncio.CancelledError:
                logger.debug("cleanup_task_cancelled_successfully")
            except Exception as e:
                logger.error(f"error_during_cleanup_task_stop: {e}")
       
        logger.info("session_manager_stopped")
    

    async def create_session(
        self, user_id: str, metadata: Optional[dict[str, Any]] = None, require_auth: bool = True
    ) -> tuple[str, Optional[str]]:
        """Create a new session with optional authentication."""
        if require_auth and not self.auth_config:
            raise RuntimeError("Authentication required but not configured")
        
        async with self._lock:
            session_id = f"session_{uuid4().hex}"
            token = None
            token_data = None
            
            if self.auth_config and require_auth:
                token = self.auth_config.create_access_token(
                    user_id=user_id, session_id=session_id, scopes=["read", "write"]
                )
                token_data = self.auth_config.verify_token(token)
            
            session = SessionState(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {},
                token_data=token_data,
            )
            self._sessions[session_id] = session
            
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
            
            logger.info("session_created", session_id=session_id, user_id=user_id)
            return session_id, token

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or not self._is_session_valid(session):
                return None
            session.last_accessed = datetime.now()
            session.access_count += 1
            return session

    async def validate_session(
        self, session_id: str, token: Optional[str] = None
    ) -> tuple[bool, Optional[SessionState]]:
        """Validate session and optional token."""
        session = await self.get_session(session_id)
        if not session:
            return False, None
        
        if token and self.token_validator:
            token_data = await self.token_validator.validate(token)
            if not token_data or token_data.session_id != session_id:
                return False, None
        
        return True, session

    def _is_session_valid(self, session: SessionState) -> bool:
        """Check if session is valid."""
        if not session.is_active:
            return False
        age = (datetime.now() - session.last_accessed).total_seconds()
        return age <= self.max_age

    async def refresh_session(self, session_id: str) -> Optional[str]:
        """Refresh an existing session token."""
        session_info = await self.get_session(session_id)
        if not session_info:
            return None
        
        # Creare un nuovo token con gli stessi scope e metadata
        new_token = self.auth_config.create_access_token(
            user_id=session_info.user_id,
            session_id=session_info.session_id,
            scopes=session_info.token_data.scopes, # Mantenere gli scope
        )
        
        # Aggiornare il token_data nella sessione
        new_token_data = self.auth_config.verify_token(new_token)
        if new_token_data:
            session_info.token_data = new_token_data
            session_info.updated_at = datetime.now()
            logger.info("session_token_refreshed", session_id=session_id)
            return new_token
        return None
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                async with self._lock:
                    expired = [
                        sid for sid, s in self._sessions.items() 
                        if not self._is_session_valid(s)
                    ]
                    for sid in expired:
                        del self._sessions[sid]
                    if expired:
                        logger.info("sessions_cleaned", count=len(expired))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cleanup_error", error=str(e))