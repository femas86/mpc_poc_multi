"""
MCP Host with ReAct reasoning for multi-server orchestration.

Features:
- Multi-client management (Weather USA, Weather Italy, Neo4j)
- ReAct (Reasoning + Acting) loop for query planning
- Tool selection and chaining
- Context aggregation across servers
- Session management integration
"""

import asyncio
import json
from typing import Any, Optional
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from shared.logging_config import get_logger
from config.settings import get_settings
from .ollama_client import OllamaClient, OllamaMessage
from .context_manager import ContextManager
from .session_manager import SessionManager
from .auth_middleware import AuthMiddleware

logger = get_logger(__name__)


class ServerConfig:
    """Configuration for MCP server."""
    
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: Optional[dict[str, str]] = None,
        description: str = "",
    ):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.description = description


class MCPClientWrapper:
    """Wrapper for MCP client session with metadata."""
    
    def __init__(self, name: str, session: ClientSession, config: ServerConfig):
        self.name = name
        self.session = session
        self.config = config
        self.tools: list[dict] = []
        self.resources: list[dict] = []
        self.connected = False
    
    async def initialize(self):
        """Initialize client and discover capabilities."""
        try:
            await self.session.initialize()
            
            # Discover tools
            tools_response = await self.session.list_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                    "server": self.name,
                }
                for tool in tools_response.tools
            ]
            
            # Discover resources
            resources_response = await self.session.list_resources()
            self.resources = [
                {
                    "uri": resource.uri,
                    "name": resource.name or "",
                    "description": resource.description or "",
                    "server": self.name,
                }
                for resource in resources_response.resources
            ]
            
            self.connected = True
            logger.info(
                "mcp_client_initialized",
                server=self.name,
                tools=len(self.tools),
                resources=len(self.resources),
            )
            
        except Exception as e:
            logger.error("mcp_client_init_failed", server=self.name, error=str(e))
            raise


class ReActStep:
    """Single step in ReAct reasoning chain."""
    
    def __init__(
        self,
        step_num: int,
        thought: str,
        action: Optional[str] = None,
        action_input: Optional[dict] = None,
        observation: Optional[str] = None,
    ):
        self.step_num = step_num
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation
        self.timestamp = datetime.now()


class MCPHost:
    """
    MCP Host with ReAct reasoning for intelligent query routing.
    
    Implements:
    - Multi-server client management
    - ReAct reasoning loop (Thought -> Action -> Observation)
    - Dynamic tool selection
    - Query decomposition
    - Result aggregation
    """
    
    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        context_manager: Optional[ContextManager] = None,
        ollama_client: Optional[OllamaClient] = None,
        auth_middleware: Optional[AuthMiddleware] = None,
    ):
        """Initialize MCP Host."""
        self.settings = get_settings()
        
        # Core components
        self.session_manager = session_manager or SessionManager()
        self.context_manager = context_manager or ContextManager()
        self.ollama_client = ollama_client or OllamaClient()
        self.auth_middleware = auth_middleware
        
        # MCP clients
        self.clients: dict[str, MCPClientWrapper] = {}
        self.server_configs: list[ServerConfig] = []
        
        # ReAct configuration
        self.max_reasoning_steps = 5
        self.react_system_prompt = self._build_react_system_prompt()
        
        logger.info("mcp_host_initialized")
    
    def _build_react_system_prompt(self) -> str:
        """Build system prompt for ReAct reasoning."""
        return """You are an intelligent assistant with access to multiple specialized tools.

You use ReAct (Reasoning + Acting) to solve queries:
1. **Thought**: Analyze the query and plan your approach
2. **Action**: Choose the most appropriate tool(s) to use
3. **Observation**: Process the tool results
4. Repeat if needed, or provide final answer

Available tool categories:
- Weather Italy: Italian weather forecasts (cities in Italy)
- Weather USA: US weather forecasts (US locations)
- Neo4j Graph: Graph database queries (relationships, patterns)

**CRITICAL INSTRUCTIONS**:
- For weather queries, identify the location FIRST
- Use Weather Italy for Italian cities (Roma, Milano, Napoli, etc.)
- Use Weather USA for US locations (Seattle, NYC, Los Angeles, etc.)
- You can chain multiple tools if needed
- Always provide clear, concise final answers
- If uncertain about location, ASK for clarification

**Response Format**:
Thought: [Your reasoning about what to do]
Action: [tool_name]
Action Input: [JSON arguments]

OR when done:
Final Answer: [Your complete response to the user]
"""
    
    async def start(self):
        """Start the MCP host and all components."""
        logger.info("mcp_host_starting")
        
        # Start session manager
        await self.session_manager.start()
        
        # Initialize configured MCP servers
        await self._initialize_servers()
        
        logger.info("mcp_host_started", servers=len(self.server_configs))
    
    async def stop(self):
        """Stop the MCP host and cleanup."""
        logger.info("mcp_host_stopping")
        
        # Close all MCP clients
        for client in self.clients.values():
            try:
                # Cleanup handled by context managers
                pass
            except Exception as e:
                logger.error("client_close_error", server=client.name, error=str(e))
        
        # Stop session manager
        await self.session_manager.stop()
        
        # Close Ollama client
        await self.ollama_client.close()
        
        logger.info("mcp_host_stopped")
    
    def register_server(self, config: ServerConfig):
        """Register an MCP server configuration."""
        self.server_configs.append(config)
        logger.info("server_registered", name=config.name)
    
    async def _initialize_servers(self):
        """Initialize all registered MCP servers."""
        for config in self.server_configs:
            try:
                # Store config - actual connection happens per-query
                logger.info("server_config_loaded", server=config.name)
            except Exception as e:
                logger.error("server_init_failed", server=config.name, error=str(e))
    
    async def process_query(
        self,
        session_id: str,
        query: str,
        token: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Process user query with ReAct reasoning.
        
        Args:
            session_id: Session identifier
            query: User query
            token: Optional auth token
            
        Returns:
            Query result with reasoning trace
        """
        logger.info("query_received", session_id=session_id, query=query[:100])
        
        # Validate session
        if self.auth_middleware:
            try:
                await self.auth_middleware.authenticate_request(
                    session_id=session_id,
                    token=token,
                )
            except Exception as e:
                return {"error": f"Authentication failed: {e}"}
        
        # Add query to context
        await self.context_manager.add_message(
            session_id=session_id,
            role="user",
            content=query,
        )
        
        # Execute ReAct loop
        react_steps: list[ReActStep] = []
        final_answer = None
        
        for step_num in range(1, self.max_reasoning_steps + 1):
            logger.debug("react_step", step=step_num, session=session_id)
            
            # Get reasoning from LLM
            step = await self._reasoning_step(session_id, query, react_steps)
            react_steps.append(step)
            
            # Check if we have final answer
            if "Final Answer:" in step.thought:
                final_answer = step.thought.split("Final Answer:")[-1].strip()
                break
            
            # Execute action if specified
            if step.action:
                observation = await self._execute_action(
                    session_id=session_id,
                    action=step.action,
                    action_input=step.action_input or {},
                )
                step.observation = observation
        
        # If no final answer after max steps, synthesize one
        if not final_answer:
            final_answer = await self._synthesize_answer(session_id, react_steps)
        
        # Add response to context
        await self.context_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=final_answer,
        )
        
        logger.info("query_completed", session_id=session_id, steps=len(react_steps))
        
        return {
            "answer": final_answer,
            "reasoning_steps": [
                {
                    "step": s.step_num,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation,
                }
                for s in react_steps
            ],
            "session_id": session_id,
        }
    
    async def _reasoning_step(
        self,
        session_id: str,
        original_query: str,
        previous_steps: list[ReActStep],
    ) -> ReActStep:
        """Execute one ReAct reasoning step."""
        
        # Build prompt with previous steps
        prompt_parts = [f"Original Query: {original_query}\n"]
        
        if previous_steps:
            prompt_parts.append("Previous Steps:")
            for step in previous_steps:
                prompt_parts.append(f"\nStep {step.step_num}:")
                prompt_parts.append(f"Thought: {step.thought}")
                if step.action:
                    prompt_parts.append(f"Action: {step.action}")
                    prompt_parts.append(f"Action Input: {json.dumps(step.action_input)}")
                if step.observation:
                    prompt_parts.append(f"Observation: {step.observation}")
        
        prompt_parts.append("\nWhat should you do next?")
        prompt = "\n".join(prompt_parts)
        
        # Get available tools
        tools_description = await self._get_tools_description()
        full_prompt = f"{tools_description}\n\n{prompt}"
        
        # Get LLM response
        messages = [
            OllamaMessage(role="system", content=self.react_system_prompt),
            OllamaMessage(role="user", content=full_prompt),
        ]
        
        response = await self.ollama_client.chat(
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused reasoning
        )
        
        # Parse response
        step = self._parse_reasoning_response(
            step_num=len(previous_steps) + 1,
            response_text=response.content,
        )
        
        return step
    
    def _parse_reasoning_response(self, step_num: int, response_text: str) -> ReActStep:
        """Parse LLM response into ReActStep."""
        
        thought = ""
        action = None
        action_input = None
        
        lines = response_text.strip().split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Thought:"):
                current_section = "thought"
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                current_section = "action"
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                current_section = "action_input"
                input_str = line.replace("Action Input:", "").strip()
                try:
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    action_input = {"raw": input_str}
            elif line.startswith("Final Answer:"):
                thought = response_text  # Keep full text for final answer
                break
            elif current_section == "thought":
                thought += " " + line
            elif current_section == "action_input":
                try:
                    action_input = json.loads(line)
                except json.JSONDecodeError:
                    pass
        
        return ReActStep(
            step_num=step_num,
            thought=thought,
            action=action,
            action_input=action_input,
        )
    
    async def _execute_action(
        self,
        session_id: str,
        action: str,
        action_input: dict[str, Any],
    ) -> str:
        """Execute a tool action."""
        
        logger.info("executing_action", action=action, input=action_input)
        
        try:
            # Parse action to determine server and tool
            if "." in action:
                server_name, tool_name = action.split(".", 1)
            else:
                # Try to infer server from action name
                tool_name = action
                server_name = self._infer_server(tool_name)
            
            # Get appropriate server config
            server_config = next(
                (c for c in self.server_configs if c.name.lower() == server_name.lower()),
                None
            )
            
            if not server_config:
                return f"Error: Server '{server_name}' not found"
            
            # Execute tool via MCP
            result = await self._call_mcp_tool(
                server_config=server_config,
                tool_name=tool_name,
                arguments=action_input,
            )
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error("action_execution_error", error=str(e), action=action)
            return f"Error executing action: {str(e)}"
    
    async def _call_mcp_tool(
        self,
        server_config: ServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call an MCP tool via stdio connection."""
        
        server_params = StdioServerParameters(
            command=server_config.command,
            args=server_config.args,
            env=server_config.env,
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Call tool
                result = await session.call_tool(tool_name, arguments)
                
                # Extract content
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        try:
                            return json.loads(content.text)
                        except json.JSONDecodeError:
                            return content.text
                    return str(content)
                
                return result
    
    def _infer_server(self, tool_name: str) -> str:
        """Infer server name from tool name."""
        tool_lower = tool_name.lower()
        
        if "italy" in tool_lower or "italian" in tool_lower:
            return "weather_italy"
        elif "usa" in tool_lower or "us_" in tool_lower or "current_conditions" in tool_lower:
            return "weather_usa"
        elif "graph" in tool_lower or "neo4j" in tool_lower or "cypher" in tool_lower:
            return "neo4j"
        
        # Default to first available server
        return self.server_configs[0].name if self.server_configs else "unknown"
    
    async def _get_tools_description(self) -> str:
        """Get description of all available tools."""
        
        descriptions = ["Available Tools:\n"]
        
        for config in self.server_configs:
            descriptions.append(f"\n{config.name}: {config.description}")
            
            # Common tools based on server type
            if "italy" in config.name.lower():
                descriptions.append("  - search_italian_city(city_name)")
                descriptions.append("  - get_weather_italy(city_name, forecast_days, include_hourly)")
            elif "usa" in config.name.lower():
                descriptions.append("  - search_us_location(location)")
                descriptions.append("  - get_weather_usa(location, include_hourly, include_alerts)")
                descriptions.append("  - get_current_conditions_usa(location)")
            elif "neo4j" in config.name.lower():
                descriptions.append("  - execute_cypher_query(query)")
                descriptions.append("  - search_nodes(label, property, value)")
                descriptions.append("  - find_relationships(node_id, relationship_type)")
        
        return "\n".join(descriptions)
    
    async def _synthesize_answer(
        self,
        session_id: str,
        react_steps: list[ReActStep],
    ) -> str:
        """Synthesize final answer from reasoning steps."""
        
        # Collect all observations
        observations = [
            step.observation
            for step in react_steps
            if step.observation
        ]
        
        if not observations:
            return "I couldn't find enough information to answer your question."
        
        # Ask LLM to synthesize
        synthesis_prompt = f"""Based on the following information gathered:

{chr(10).join(f"- {obs}" for obs in observations)}

Please provide a clear, concise answer to the user's original question."""
        
        messages = [
            OllamaMessage(role="system", content="Synthesize a helpful answer from the given information."),
            OllamaMessage(role="user", content=synthesis_prompt),
        ]
        
        response = await self.ollama_client.chat(messages=messages, temperature=0.5)
        
        return response.content
    
    async def get_session_info(self, session_id: str) -> dict[str, Any]:
        """Get information about a session."""
        
        session = await self.session_manager.get_session(session_id)
        context_summary = await self.context_manager.get_context_summary(session_id)
        
        return {
            "session": {
                "session_id": session.session_id if session else None,
                "user_id": session.user_id if session else None,
                "created_at": session.created_at.isoformat() if session else None,
                "access_count": session.access_count if session else 0,
            },
            "context": context_summary,
            "registered_servers": [c.name for c in self.server_configs],
        }
