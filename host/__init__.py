"""MCP Host package - orchestrates multiple MCP servers and LLM interaction."""

from .context_manager import ContextManager, ConversationContext
from .ollama_client import OllamaClient, OllamaMessage, OllamaResponse
from .mcp_host import MCPHost
from .server_discovery import ServerDiscovery, ServerMetadata
__all__ = [
    "ContextManager",
    "ConversationContext",
    "OllamaClient",
    "OllamaMessage",
    "OllamaResponse",
    "ServerDiscovery",
    "ServerMetadata",
    "MCPHost",
]