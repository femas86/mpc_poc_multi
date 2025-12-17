"""
Automatic discovery and registration of MCP servers.

Features:
- Scans servers/ directory
- Detects valid MCP servers
- Auto-registers available servers
- Validates server configuration
- Skips disabled/broken servers
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from shared.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


@dataclass
class ServerMetadata:
    """Metadata for a discovered MCP server."""
    
    name: str
    path: Path
    description: str
    enabled: bool = True
    requires_config: list[str] = None
    
    def __post_init__(self):
        if self.requires_config is None:
            self.requires_config = []


class ServerDiscovery:
    """Automatic MCP server discovery and validation."""
    
    def __init__(self, servers_dir: Optional[Path] = None):
        """
        Initialize server discovery.
        
        Args:
            servers_dir: Path to servers directory (defaults to ./servers)
        """
        self.servers_dir = servers_dir or Path.cwd() / "servers"
        
        if not self.servers_dir.exists():
            raise RuntimeError(f"Servers directory not found: {self.servers_dir}")
        
        logger.info("server_discovery_initialized", path=str(self.servers_dir))
    
    def discover_servers(self) -> list[ServerMetadata]:
        """
        Discover all valid MCP servers in servers directory.
        
        Returns:
            List of discovered server metadata
        """
        logger.info("discovering_servers", directory=str(self.servers_dir))
        
        discovered = []
        
        # Scan subdirectories in servers/
        for server_dir in self.servers_dir.iterdir():
            if not server_dir.is_dir():
                continue
            
            # Skip private/hidden directories
            if server_dir.name.startswith(("_", ".")):
                continue
            
            # Try to discover server
            metadata = self._discover_server(server_dir)
            if metadata:
                discovered.append(metadata)
        
        logger.info("servers_discovered", count=len(discovered))
        return discovered
    
    def _discover_server(self, server_dir: Path) -> Optional[ServerMetadata]:
        """
        Discover MCP server in a directory.
        
        Args:
            server_dir: Server directory path
            
        Returns:
            ServerMetadata if valid server found, None otherwise
        """
        logger.debug("checking_directory", path=str(server_dir))
        
        # Look for server.py or server_*.py
        server_files = list(server_dir.glob("server*.py"))
        
        if not server_files:
            logger.debug("no_server_file", directory=server_dir.name)
            return None
        
        server_file = server_files[0]  # Use first match
        
        # Check for server.json metadata (optional)
        metadata_file = server_dir / "server.json"
        
        if metadata_file.exists():
            try:
                metadata = self._load_metadata(metadata_file)
                metadata.path = server_file
                logger.info("server_discovered_with_metadata", name=metadata.name)
                return metadata
            except Exception as e:
                logger.warning("metadata_load_failed", file=str(metadata_file), error=str(e))
        
        # Fallback: Create metadata from directory structure
        metadata = self._infer_metadata(server_dir, server_file)
        
        if metadata:
            logger.info("server_discovered_inferred", name=metadata.name)
        
        return metadata
    
    def _load_metadata(self, metadata_file: Path) -> ServerMetadata:
        """
        Load server metadata from JSON file.
        
        Args:
            metadata_file: Path to server.json
            
        Returns:
            ServerMetadata instance
        """
        with open(metadata_file, "r") as f:
            data = json.load(f)
        
        return ServerMetadata(
            name=data["name"],
            path=Path(),  # Will be set by caller
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            requires_config=data.get("requires_config", []),
        )
    
    def _infer_metadata(self, server_dir: Path, server_file: Path) -> Optional[ServerMetadata]:
        """
        Infer server metadata from directory structure and naming.
        
        Args:
            server_dir: Server directory
            server_file: Server Python file
            
        Returns:
            Inferred ServerMetadata or None
        """
        # Extract name from directory
        name = server_dir.name
        
        # Infer description from common patterns
        description_map = {
            "weather_italy": "Italian weather forecasts using Open-Meteo API",
            "weather_usa": "US weather forecasts using National Weather Service",
            "neo4j_graph": "Neo4j graph database with semantic querying",
            "neo4j": "Neo4j graph database operations",
        }
        
        description = description_map.get(name, f"{name.replace('_', ' ').title()} MCP Server")
        
        # Infer required config
        requires_config = []
        if "neo4j" in name.lower():
            requires_config = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
        
        return ServerMetadata(
            name=name,
            path=server_file,
            description=description,
            enabled=True,
            requires_config=requires_config,
        )
    
    def validate_server(self, metadata: ServerMetadata) -> tuple[bool, Optional[str]]:
        """
        Validate server configuration and requirements.
        
        Args:
            metadata: Server metadata to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if explicitly disabled
        if not metadata.enabled:
            return False, f"Server '{metadata.name}' is disabled in metadata"
        
        # Check if server file exists
        if not metadata.path.exists():
            return False, f"Server file not found: {metadata.path}"
        
        # Check required configuration
        settings = get_settings()
        missing_config = []
        
        for config_key in metadata.requires_config:
            # Check if environment variable exists
            import os
            if not os.getenv(config_key):
                missing_config.append(config_key)
        
        if missing_config:
            return False, f"Missing required config: {', '.join(missing_config)}"
        
        return True, None
