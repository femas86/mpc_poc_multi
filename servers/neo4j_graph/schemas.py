"""Data schemas for Neo4j graph operations."""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class Node(BaseModel):
    """Graph node representation."""
    
    id: Optional[str] = Field(None, description="Node internal ID")
    labels: list[str] = Field(default_factory=list, description="Node labels")
    properties: dict[str, Any] = Field(default_factory=dict, description="Node properties")


class Relationship(BaseModel):
    """Graph relationship representation."""
    
    id: Optional[str] = Field(None, description="Relationship internal ID")
    type: str = Field(..., description="Relationship type")
    start_node: str = Field(..., description="Start node ID or reference")
    end_node: str = Field(..., description="End node ID or reference")
    properties: dict[str, Any] = Field(default_factory=dict, description="Relationship properties")


class GraphPattern(BaseModel):
    """Graph pattern for matching."""
    
    nodes: list[Node] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    where_clauses: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Result from Cypher query execution."""
    
    success: bool = Field(..., description="Whether query succeeded")
    records: list[dict[str, Any]] = Field(default_factory=list, description="Result records")
    summary: dict[str, Any] = Field(default_factory=dict, description="Query summary")
    count: int = Field(default=0, description="Number of records returned")
    error: Optional[str] = Field(None, description="Error message if failed")


class SemanticQuery(BaseModel):
    """Semantic query request."""
    
    question: str = Field(..., description="Natural language question")
    context: Optional[dict[str, Any]] = Field(None, description="Additional context")
    max_results: int = Field(default=10, ge=1, le=100)


class NodeCreate(BaseModel):
    """Request to create a node."""
    
    labels: list[str] = Field(..., description="Node labels", min_length=1)
    properties: dict[str, Any] = Field(..., description="Node properties")


class RelationshipCreate(BaseModel):
    """Request to create a relationship."""
    
    type: str = Field(..., description="Relationship type")
    from_node_id: str = Field(..., description="Source node ID")
    to_node_id: str = Field(..., description="Target node ID")
    properties: dict[str, Any] = Field(default_factory=dict)


class PathResult(BaseModel):
    """Result of path finding query."""
    
    path_length: int
    nodes: list[Node]
    relationships: list[Relationship]
    total_cost: Optional[float] = None
