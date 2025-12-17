"""Context management for conversation history and token tracking."""

import asyncio
from collections import deque
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from config.settings import get_settings
from shared.logging_config import get_logger

# import os
# from collections.abc import AsyncIterator
# from contextlib import asynccontextmanager
# from dataclasses import dataclass

# from neo4j import AsyncGraphDatabase, AsyncDriver


logger = get_logger(__name__)


# @dataclass
# class AppContext:
#     """Application context with Neo4j driver."""
#     driver: AsyncDriver
#     database: str

# @asynccontextmanager
# async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
#     """Manage Neo4j driver lifecycle."""

#     # Read connection details from environment
#     uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
#     username = os.getenv("NEO4J_USER", "neo4j")
#     password = os.getenv("NEO4J_PASSWORD", "password")
#     database = os.getenv("NEO4J_DATABASE", "neo4j")

#     # Initialize driver on startup
#     driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

#     try:
#         # Yield context with driver
#         yield AppContext(driver=driver, database=database)
#     finally:
#         # Close driver on shutdown
#         await driver.close()

class Message(BaseModel):
    """Individual message in conversation."""

    role: str = Field(..., description="Message role: system, user, assistant, or tool")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = Field(default=None, description="Approximate token count")


class ConversationContext(BaseModel):
    """Complete conversation context with history and metadata."""

    session_id: str = Field(..., description="Unique session identifier")
    messages: list[Message] = Field(default_factory=list)
    total_tokens: int = Field(default=0, description="Total tokens in context")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class ContextManager:
    """
    Manages conversation context with automatic pruning and token tracking.
    
    Features:
    - Rolling window of conversation history
    - Token count estimation
    - Automatic context pruning
    - System message preservation
    - Context compression
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        retention_count: Optional[int] = None,
    ):
        """
        Initialize context manager.

        Args:
            max_tokens: Maximum tokens to maintain in context
            retention_count: Maximum number of messages to retain
        """
        settings = get_settings()
        self.max_tokens = max_tokens or settings.context_max_tokens
        self.retention_count = retention_count or settings.context_retention_count
        
        # Active contexts keyed by session_id
        self._contexts: dict[str, ConversationContext] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        logger.info(
            "context_manager_initialized",
            max_tokens=self.max_tokens,
            retention_count=self.retention_count,
        )

    async def create_context(self, session_id: str, system_message: Optional[str] = None) -> ConversationContext:
        """
        Create a new conversation context.

        Args:
            session_id: Unique session identifier
            system_message: Optional system message to initialize context

        Returns:
            New ConversationContext instance
        """
        async with self._lock:
            context = ConversationContext(session_id=session_id)
            
            if system_message:
                await self._add_message_internal(
                    context,
                    Message(
                        role="system",
                        content=system_message,
                        token_count=self._estimate_tokens(system_message),
                    ),
                )
            
            self._contexts[session_id] = context
            
            logger.info(
                "context_created",
                session_id=session_id,
                has_system_message=bool(system_message),
            )
            
            return context

    async def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get existing conversation context.

        Args:
            session_id: Session identifier

        Returns:
            ConversationContext if exists, None otherwise
        """
        async with self._lock:
            return self._contexts.get(session_id)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationContext:
        """
        Add a message to conversation context.

        Args:
            session_id: Session identifier
            role: Message role
            content: Message content
            metadata: Optional metadata

        Returns:
            Updated ConversationContext

        Raises:
            ValueError: If context doesn't exist
        """
        async with self._lock:
            context = self._contexts.get(session_id)
            if not context:
                raise ValueError(f"Context not found for session: {session_id}")
            
            message = Message(
                role=role,
                content=content,
                token_count=self._estimate_tokens(content),
                metadata=metadata or {},
            )
            
            await self._add_message_internal(context, message)
            
            # Prune if necessary
            await self._prune_context(context)
            
            context.updated_at = datetime.now()
            
            logger.debug(
                "message_added",
                session_id=session_id,
                role=role,
                content_length=len(content),
                total_tokens=context.total_tokens,
                message_count=len(context.messages),
            )
            
            return context

    async def _add_message_internal(self, context: ConversationContext, message: Message) -> None:
        """
        Internal method to add message without lock.

        Args:
            context: Conversation context
            message: Message to add
        """
        context.messages.append(message)
        context.total_tokens += message.token_count or 0

    async def _prune_context(self, context: ConversationContext) -> None:
        """
        Prune context to stay within limits.

        Args:
            context: Context to prune
        """
        # Keep system message if present
        system_messages = [msg for msg in context.messages if msg.role == "system"]
        other_messages = [msg for msg in context.messages if msg.role != "system"]
        
        # Prune by retention count
        if len(other_messages) > self.retention_count:
            other_messages = other_messages[-self.retention_count:]
        
        # Prune by token count
        while context.total_tokens > self.max_tokens and len(other_messages) > 1:
            removed = other_messages.pop(0)
            context.total_tokens -= removed.token_count or 0
        
        # Reconstruct messages
        context.messages = system_messages + other_messages
        
        # Recalculate total tokens
        context.total_tokens = sum(msg.token_count or 0 for msg in context.messages)
        
        logger.debug(
            "context_pruned",
            session_id=context.session_id,
            message_count=len(context.messages),
            total_tokens=context.total_tokens,
        )

    async def get_messages_for_llm(self, session_id: str) -> list[dict[str, str]]:
        """
        Get messages formatted for LLM consumption.

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries with role and content

        Raises:
            ValueError: If context doesn't exist
        """
        context = await self.get_context(session_id)
        if not context:
            raise ValueError(f"Context not found for session: {session_id}")
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in context.messages
        ]

    async def clear_context(self, session_id: str) -> bool:
        """
        Clear conversation context.

        Args:
            session_id: Session identifier

        Returns:
            True if context existed and was cleared
        """
        async with self._lock:
            if session_id in self._contexts:
                del self._contexts[session_id]
                logger.info("context_cleared", session_id=session_id)
                return True
            return False

    async def get_context_summary(self, session_id: str) -> dict[str, Any]:
        """
        Get summary of context state.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with context statistics
        """
        context = await self.get_context(session_id)
        if not context:
            return {"exists": False}
        
        return {
            "exists": True,
            "session_id": session_id,
            "message_count": len(context.messages),
            "total_tokens": context.total_tokens,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "roles": [msg.role for msg in context.messages],
        }

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses simple heuristic: ~4 characters per token.
        For production, consider using tiktoken or similar.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    async def compress_context(self, session_id: str, summary: str) -> ConversationContext:
        """
        Compress context by replacing old messages with summary.

        Args:
            session_id: Session identifier
            summary: Summary of conversation to preserve

        Returns:
            Updated context

        Raises:
            ValueError: If context doesn't exist
        """
        async with self._lock:
            context = self._contexts.get(session_id)
            if not context:
                raise ValueError(f"Context not found for session: {session_id}")
            
            # Keep system message and last N messages
            system_messages = [msg for msg in context.messages if msg.role == "system"]
            recent_messages = [msg for msg in context.messages if msg.role != "system"][-3:]
            
            # Create summary message
            summary_message = Message(
                role="system",
                content=f"[Conversation Summary]: {summary}",
                token_count=self._estimate_tokens(summary),
                metadata={"compressed": True},
            )
            
            # Reconstruct context
            context.messages = system_messages + [summary_message] + recent_messages
            context.total_tokens = sum(msg.token_count or 0 for msg in context.messages)
            context.updated_at = datetime.now()
            
            logger.info(
                "context_compressed",
                session_id=session_id,
                new_message_count=len(context.messages),
                new_token_count=context.total_tokens,
            )
            
            return context

    async def list_active_contexts(self) -> list[str]:
        """
        List all active context session IDs.

        Returns:
            List of session IDs
        """
        async with self._lock:
            return list(self._contexts.keys())

    async def cleanup_old_contexts(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up contexts older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of contexts cleaned up
        """
        async with self._lock:
            now = datetime.now()
            to_remove = []
            
            for session_id, context in self._contexts.items():
                age = (now - context.updated_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                del self._contexts[session_id]
            
            if to_remove:
                logger.info(
                    "contexts_cleaned_up",
                    count=len(to_remove),
                    max_age_seconds=max_age_seconds,
                )
            
            return len(to_remove)