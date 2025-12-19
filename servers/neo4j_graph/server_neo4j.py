"""MCP Server for Neo4j Graph Database."""

from typing import Any
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from shared.logging_config import get_logger
from servers.neo4j_graph.db_client import Neo4jClient
from servers.neo4j_graph.queries import CypherQueryBuilder
from servers.neo4j_graph.schemas import (
    NodeCreate,
    RelationshipCreate,
    SemanticQuery,
    QueryResult,
)

logger = get_logger(__name__)

# Create MCP server
mcp = FastMCP(
    name="Neo4jGraph",
    instructions="Provides graph database operations for Neo4j with semantic querying"
)

# Initialize Neo4j client
neo4j_client = Neo4jClient()
query_builder = CypherQueryBuilder()


@mcp.tool()
async def execute_cypher_query(
    query: str,
    params: dict[str, Any] = None,
    ctx: Context[ServerSession, None] = None,
) -> dict[str, Any]:
    """
    Execute a raw Cypher query.
    
    Args:
        query: Cypher query string
        parameters: Query parameters
        
    Returns:
        Query results with records and metadata
    """
    await ctx.info(f"Executing Cypher query: {query[:100]}...")
    
    try:
        result = await neo4j_client.execute_query(query, params)
        
        await ctx.info(f"Found {result.count} recommendations")
        
        return {
            "success": result.success,
            "recommendations": result.records,
            "count": result.count,
        }
        
    except Exception as e:
        await ctx.error(f"Recommendation failed: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_database_schema(ctx: Context[ServerSession, None] = None) -> dict[str, Any]:
    """
    Get the database schema with node labels and relationship types.
    
    Returns:
        Schema information
    """
    await ctx.info("Retrieving database schema")
    
    try:
        result = await neo4j_client.get_schema()
        
        await ctx.info("Schema retrieved successfully")
        
        return {
            "success": result.success,
            "schema": result.records,
        }
        
    except Exception as e:
        await ctx.error(f"Schema retrieval failed: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def aggregate_nodes(
    label: str,
    aggregation: str,
    property_name: str = None,
    ctx: Context[ServerSession, None] = None,
) -> dict[str, Any]:
    """
    Perform aggregation on nodes.
    
    Args:
        label: Node label
        aggregation: Aggregation function (count, sum, avg, max, min)
        property_name: Property to aggregate (not needed for count)
        
    Returns:
        Aggregation result
    """
    await ctx.info(f"Aggregating {label} nodes: {aggregation}({property_name})")
    
    try:
        result = await neo4j_client.execute_aggregation(label, aggregation, property_name)
        
        if result.success:
            await ctx.info(f"Aggregation completed")
            return {
                "success": True,
                "result": result.records[0] if result.records else None,
            }
        else:
            await ctx.error(f"Aggregation failed: {result.error}")
            return {"success": False, "error": result.error}
            
    except Exception as e:
        await ctx.error(f"Aggregation error: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.resource("graph://schema")
async def schema_resource() -> str:
    """
    Resource endpoint for database schema.
    
    Returns:
        Formatted schema information
    """
    try:
        result = await neo4j_client.get_schema()
        
        if not result.success or not result.records:
            return "Schema information not available"
        
        output = ["Neo4j Database Schema", "=" * 50, ""]
        
        schema_data = result.records[0]
        
        if "nodes" in schema_data:
            output.append("Node Labels:")
            for node in schema_data["nodes"]:
                labels = node.get("labels", [])
                output.append(f"  • {', '.join(labels)}")
        
        if "relationships" in schema_data:
            output.append("\nRelationship Types:")
            for rel in schema_data["relationships"]:
                rel_type = rel.get("type", "UNKNOWN")
                output.append(f"  • {rel_type}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error("schema_resource_error", error=str(e))
        return f"Error retrieving schema: {str(e)}"


class Neo4jGraphServer:
    """Wrapper class for Neo4j Graph MCP Server."""
    
    def __init__(self):
        """Initialize Neo4j Graph server."""
        self.mcp = mcp
        self.neo4j_client = neo4j_client
        self.query_builder = query_builder
        logger.info("neo4j_graph_server_initialized")
    
    async def start(self):
        """Start the server and connect to Neo4j."""
        await self.neo4j_client.connect()
        logger.info("neo4j_graph_server_started")
    
    async def stop(self):
        """Stop the server and disconnect from Neo4j."""
        await self.neo4j_client.close()
        logger.info("neo4j_graph_server_stopped")
    
    def get_mcp_server(self) -> FastMCP:
        """Get the FastMCP server instance."""
        return self.mcp

if __name__ == "__main__":
    """
    Run server in stdio mode for MCP host connection.
    
    This allows the MCP host to:
    1. Connect via stdio
    2. Discover tools with list_tools()
    3. Discover resources with list_resources()
    4. Call tools and read resources
    """
    import asyncio
    from shared.logging_config import setup_logging
    
    # 1. Initialize the logging system to use stderr
    setup_logging()
    
    async def run_server():
        """Run the MCP server in stdio mode."""
        server = Neo4jGraphServer()
        await server.start()
        
        # Run FastMCP server in streamable http async mode
        # This is what the MCP host connects to
        await mcp.run_streamable_http_async()
        
        await server.stop()
    
    # Run the server
    asyncio.run(run_server())
