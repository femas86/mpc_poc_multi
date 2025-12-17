"""Cypher query builder for semantic queries."""

from typing import Any, Optional
from shared.logging_config import get_logger

logger = get_logger(__name__)


class CypherQueryBuilder:
    """Builder for complex Cypher queries from natural language."""
    
    @staticmethod
    def build_semantic_query(question: str, context: Optional[dict] = None) -> tuple[str, dict]:
        """
        Build Cypher query from natural language question.
        
        Args:
            question: Natural language question
            context: Additional context
            
        Returns:
            Tuple of (cypher_query, parameters)
        """
        question_lower = question.lower()
        
        # Pattern: "Find all X"
        if "find all" in question_lower or "list all" in question_lower:
            return CypherQueryBuilder._build_list_query(question_lower, context)
        
        # Pattern: "How many X"
        if "how many" in question_lower or "count" in question_lower:
            return CypherQueryBuilder._build_count_query(question_lower, context)
        
        # Pattern: "What are the relationships"
        if "relationship" in question_lower or "connected" in question_lower:
            return CypherQueryBuilder._build_relationship_query(question_lower, context)
        
        # Pattern: "Path between X and Y"
        if "path" in question_lower or "route" in question_lower:
            return CypherQueryBuilder._build_path_query(question_lower, context)
        
        # Pattern: "Similar to X"
        if "similar" in question_lower or "like" in question_lower:
            return CypherQueryBuilder._build_similarity_query(question_lower, context)
        
        # Default: Return all nodes with limit
        return "MATCH (n) RETURN n LIMIT 10", {}
    
    @staticmethod
    def _build_list_query(question: str, context: Optional[dict]) -> tuple[str, dict]:
        """Build query to list entities."""
        # Extract label from question
        # Simple pattern matching - can be enhanced with NLP
        label = "Node"  # Default
        
        if "person" in question or "people" in question:
            label = "Person"
        elif "company" in question or "companies" in question:
            label = "Company"
        elif "product" in question:
            label = "Product"
        elif "city" in question or "cities" in question:
            label = "City"
        
        query = f"MATCH (n:{label}) RETURN n LIMIT 50"
        return query, {}
    
    @staticmethod
    def _build_count_query(question: str, context: Optional[dict]) -> tuple[str, dict]:
        """Build count query."""
        label = "Node"
        
        if "person" in question or "people" in question:
            label = "Person"
        elif "company" in question or "companies" in question:
            label = "Company"
        elif "relationship" in question:
            return "MATCH ()-[r]->() RETURN count(r) as count", {}
        
        query = f"MATCH (n:{label}) RETURN count(n) as count"
        return query, {}
    
    @staticmethod
    def _build_relationship_query(question: str, context: Optional[dict]) -> tuple[str, dict]:
        """Build relationship query."""
        # Pattern: "What are X's relationships"
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, type(r) as relationship, m
        LIMIT 20
        """
        return query, {}
    
    @staticmethod
    def _build_path_query(question: str, context: Optional[dict]) -> tuple[str, dict]:
        """Build path finding query."""
        # This would need entity extraction in production
        query = """
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        MATCH path = shortestPath((a)-[*]-(b))
        RETURN path
        """
        
        params = {
            "from_id": context.get("from_id", 0) if context else 0,
            "to_id": context.get("to_id", 1) if context else 1,
        }
        
        return query, params
    
    @staticmethod
    def _build_similarity_query(question: str, context: Optional[dict]) -> tuple[str, dict]:
        """Build similarity query."""
        # Find nodes with similar properties
        query = """
        MATCH (n)
        WHERE n.name =~ '.*' + $search_term + '.*'
        RETURN n
        LIMIT 10
        """
        
        params = {"search_term": context.get("search_term", "") if context else ""}
        return query, params
    
    @staticmethod
    def build_recommendation_query(
        node_id: str,
        relationship_type: Optional[str] = None,
        max_results: int = 10,
    ) -> tuple[str, dict]:
        """
        Build collaborative filtering recommendation query.
        
        Args:
            node_id: Source node ID
            relationship_type: Type of relationship to follow
            max_results: Maximum recommendations
            
        Returns:
            Tuple of (query, parameters)
        """
        rel_filter = f":{relationship_type}" if relationship_type else ""
        
        query = f"""
        MATCH (source)-[{rel_filter}]->(item)<-[{rel_filter}]-(other)
        WHERE id(source) = $node_id
        MATCH (other)-[{rel_filter}]->(recommendation)
        WHERE NOT (source)-[{rel_filter}]->(recommendation)
        RETURN recommendation, count(*) as score
        ORDER BY score DESC
        LIMIT $max_results
        """
        
        return query, {"node_id": int(node_id), "max_results": max_results}
    
    @staticmethod
    def build_community_detection_query(min_connections: int = 3) -> tuple[str, dict]:
        """
        Build query to detect communities.
        
        Args:
            min_connections: Minimum connections for community
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (n)-[r]->(m)
        WITH n, count(r) as connections
        WHERE connections >= $min_connections
        RETURN n, connections
        ORDER BY connections DESC
        """
        
        return query, {"min_connections": min_connections}