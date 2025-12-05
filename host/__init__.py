"""MCP Host package - orchestrates multiple MCP servers and LLM interaction."""

from .context_manager import ContextManager, ConversationContext
from .ollama_client import OllamaClient, OllamaMessage, OllamaResponse

__all__ = [
    "ContextManager",
    "ConversationContext",
    "OllamaClient",
    "OllamaMessage",
    "OllamaResponse",
]