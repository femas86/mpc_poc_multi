"""Central configuration management using Pydantic."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MCP Host Configuration
    mcp_host_name: str = Field(default="LocalMCPHost", description="MCP Host name")
    mcp_host_port: int = Field(default=8000, ge=1024, le=65535)
    mcp_debug: bool = Field(default=False)
    mcp_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Ollama Configuration
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:1b")
    ollama_timeout: int = Field(default=120, ge=10, le=600)

    # Authentication
    access_token_expire_minutes: int = 30
    auth_enabled: bool = Field(default=True)
    auth_secret_key: str = Field(default="change-me-in-production", min_length=32)
    algorithm: str = "HS256"
    auth_token_expiry: int = Field(default=3600, ge=300, le=86400)

    # Weather.gov Configuration
    weathergov_base_url : str = Field(
        default="https://api.weather.gov"
        )
    weathergov_user_agent: str = Field(
        default="weather-app/1.0",
        description="Required User-Agent for Weather.gov API",
    )
    weathergov_timeout: int = Field(default=30, ge=5, le=120)

    # Open-Meteo Configuration
    openmeteo_base_url: str = Field(default="https://api.open-meteo.com/v1")
    openmeteo_timeout: int = Field(default=30, ge=5, le=120)

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    neo4j_database: str = Field(default="neo4j")

    # Session Management
    session_max_age: int = Field(default=1800, ge=60, le=7200)
    session_cleanup_interval: int = Field(default=300, ge=60, le=3600)

    # Context Management
    context_max_tokens: int = Field(default=8000, ge=1000, le=32000)
    context_retention_count: int = Field(default=10, ge=1, le=100)

    @field_validator("auth_secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate that secret key is not default in production."""
        if v == "change-me-in-production" and not os.getenv("MCP_DEBUG", "").lower() == "true":
            raise ValueError("AUTH_SECRET_KEY must be changed in production!")
        if len(v) < 32:
            raise ValueError("AUTH_SECRET_KEY must be at least 32 characters long")
        return v

    @field_validator("ollama_host")
    @classmethod
    def validate_ollama_host(cls, v: str) -> str:
        """Ensure Ollama host has proper URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_HOST must start with http:// or https://")
        return v.rstrip("/")

    def get_server_url(self) -> str:
        """Get the full server URL."""
        return f"http://localhost:{self.mcp_host_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings
