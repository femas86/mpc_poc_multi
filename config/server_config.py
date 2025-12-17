from typing import Optional    
import os
from dataclasses import dataclass

@dataclass 
class ServerConfig:
    """Configuration for MCP server."""
    
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: Optional[dict[str, str]] = None,
        # available_tools: Optional[list[str]] = None,
        # available_resources: Optional[list[str]] = None,    
        description: str = "",
    ):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or os.getenv()
        self.description = description
        # self.available_tools = available_tools or []
        # self.available_resources = available_resources or []