"""Neo4j database client with connection pooling."""

from typing import Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError

from config.settings import get_settings
from shared.logging_config import get_logger
from shared.utils import retry_async

import asyncio

from servers.neo4j_graph.schemas import Node, Relationship, QueryResult

logger = get_logger(__name__)


class Neo4jClient:
    """Async client for Neo4j graph database operations."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
            database: Database name
        """
        settings = get_settings()
        
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        
        self.driver: Optional[AsyncDriver] = None
        self._driver_lock = asyncio.Lock()
        self._initialized = False
        
        logger.info("neo4j_client_initialized", uri=self.uri, database=self.database)
    
    async def _ensure_driver(self):
        """Ensure driver is initialized (lazy)."""
        if self._initialized and self._driver:
            return
        
        async with self._driver_lock:
            if self._initialized and self._driver:
                return
            
            logger.info("initializing_neo4j_driver", uri=self.uri)
            
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            
            await self._driver.verify_connectivity()
            self._initialized = True
            
            logger.info("neo4j_driver_ready")

    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connectivity
            await self.driver.verify_connectivity()
            logger.info("neo4j_connected", database=self.database)
        except Neo4jError as e:
            logger.error("neo4j_connection_failed", error=str(e))
            raise
    
    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self._driver = None
            self._initialized = False
            logger.info("neo4j_disconnected")
    
    @retry_async(max_attempts=3, delay=1.0)
    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            QueryResult with records and metadata
        """

        await self._ensure_driver()
        async with self._driver.session(database=self.database) as session:
            if not self.driver:
                await self.connect()
        
            try:
                logger.debug("executing_cypher", query=query[:200], params=parameters)
                
                async with self.driver.session(database=self.database) as session:
                    result = await session.run(query, parameters or {})
                    records = await result.data()
                    summary = await result.consume()
                    
                    logger.info(
                        "query_executed",
                        records_returned=len(records),
                        query_type=summary.query_type,
                    )
                    
                    return QueryResult(
                        success=True,
                        records=records,
                        summary={
                            "query_type": summary.query_type,
                            "counters": {
                                "nodes_created": summary.counters.nodes_created,
                                "nodes_deleted": summary.counters.nodes_deleted,
                                "relationships_created": summary.counters.relationships_created,
                                "relationships_deleted": summary.counters.relationships_deleted,
                                "properties_set": summary.counters.properties_set,
                            },
                        },
                        count=len(records),
                    )
                    
            except Neo4jError as e:
                logger.error("query_execution_failed", error=str(e), query=query[:100])
                return QueryResult(
                    success=False,
                    error=str(e),
                )
            pass
            
    
    async def create_node(
        self,
        labels: list[str],
        properties: dict[str, Any],
    ) -> QueryResult:
        """
        Create a node with labels and properties.
        
        Args:
            labels: Node labels
            properties: Node properties
            
        Returns:
            QueryResult with created node
        """
        labels_str = ":".join(labels)
        query = f"CREATE (n:{labels_str} $properties) RETURN n"
        
        return await self.execute_query(query, {"properties": properties})
    
    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        rel_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Create a relationship between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            rel_type: Relationship type
            properties: Relationship properties
            
        Returns:
            QueryResult with created relationship
        """
        query = """
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        CREATE (a)-[r:%s $properties]->(b)
        RETURN r
        """ % rel_type
        
        params = {
            "from_id": int(from_node_id),
            "to_id": int(to_node_id),
            "properties": properties or {},
        }
        
        return await self.execute_query(query, params)
    
    async def find_nodes(
        self,
        label: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> QueryResult:
        """
        Find nodes by label and/or properties.
        
        Args:
            label: Node label to filter
            properties: Properties to match
            limit: Maximum results
            
        Returns:
            QueryResult with matching nodes
        """
        label_clause = f":{label}" if label else ""
        where_clauses = []
        params = {"limit": limit}
        
        if properties:
            for key, value in properties.items():
                where_clauses.append(f"n.{key} = ${key}")
                params[key] = value
        
        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (n{label_clause})
        {where_str}
        RETURN n
        LIMIT $limit
        """
        
        return await self.execute_query(query, params)
    
    async def find_shortest_path(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_types: Optional[list[str]] = None,
        max_depth: int = 5,
    ) -> QueryResult:
        """
        Find shortest path between two nodes.
        
        Args:
            from_node_id: Start node ID
            to_node_id: End node ID
            relationship_types: Relationship types to traverse
            max_depth: Maximum path length
            
        Returns:
            QueryResult with path information
        """
        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)
        
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        MATCH path = shortestPath((a)-[{rel_filter}*1..{max_depth}]-(b))
        RETURN path,
               length(path) as path_length,
               [node in nodes(path) | node] as nodes,
               [rel in relationships(path) | rel] as relationships
        """
        
        params = {
            "from_id": int(from_node_id),
            "to_id": int(to_node_id),
        }
        
        return await self.execute_query(query, params)
    
    async def get_node_neighbors(
        self,
        node_id: str,
        direction: str = "BOTH",
        relationship_types: Optional[list[str]] = None,
        max_depth: int = 1,
    ) -> QueryResult:
        """
        Get neighbors of a node.
        
        Args:
            node_id: Node ID
            direction: Direction (OUTGOING, INCOMING, BOTH)
            relationship_types: Filter by relationship types
            max_depth: How many hops
            
        Returns:
            QueryResult with neighbor nodes
        """
        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)
        
        direction_map = {
            "OUTGOING": f"-[{rel_filter}*1..{max_depth}]->",
            "INCOMING": f"<-[{rel_filter}*1..{max_depth}]-",
            "BOTH": f"-[{rel_filter}*1..{max_depth}]-",
        }
        
        pattern = direction_map.get(direction.upper(), direction_map["BOTH"])
        
        query = f"""
        MATCH (n){pattern}(neighbor)
        WHERE id(n) = $node_id
        RETURN DISTINCT neighbor, labels(neighbor) as labels
        """
        
        return await self.execute_query(query, {"node_id": int(node_id)})
    
    async def execute_aggregation(
        self,
        label: str,
        aggregation: str,
        property_name: Optional[str] = None,
    ) -> QueryResult:
        """
        Execute aggregation query.
        
        Args:
            label: Node label
            aggregation: Aggregation function (count, sum, avg, max, min)
            property_name: Property to aggregate
            
        Returns:
            QueryResult with aggregation result
        """
        if aggregation.lower() == "count":
            query = f"MATCH (n:{label}) RETURN count(n) as result"
            params = {}
        else:
            if not property_name:
                raise ValueError(f"Property name required for {aggregation} aggregation")
            
            query = f"""
            MATCH (n:{label})
            RETURN {aggregation}(n.{property_name}) as result
            """
            params = {}
        
        return await self.execute_query(query, params)
    
    async def get_schema(self) -> QueryResult:
        """
        Get database schema information.
        
        Returns:
            QueryResult with schema details
        """
        query = """
        CALL db.schema.visualization()
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        
        return await self.execute_query(query)
