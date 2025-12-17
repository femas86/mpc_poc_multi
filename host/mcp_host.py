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
import os
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from shared.logging_config import get_logger
from config.settings import get_settings
from host.ollama_client import OllamaClient, OllamaMessage
from host.context_manager import ContextManager
from host.session_manager import SessionManager
from host.auth_middleware import AuthMiddleware
from host.health_checker import HealthChecker
from config.server_config import ServerConfig
from host.server_discovery import ServerDiscovery, ServerMetadata


logger = get_logger(__name__)

# class MCPClientWrapper:
    # """Wrapper for MCP client session with metadata."""
    
    # def __init__(self, name: str, session: ClientSession, config: ServerConfig):
    #     self.name = name
    #     self.session = session
    #     self.config = config
    #     self.tools: list[dict] = []
    #     self.resources: list[dict] = []
    #     self.connected = False
    
    # async def initialize(self):
    #     """Initialize client and discover capabilities."""
    #     try:
    #         await self.session.initialize()
            
    #         # Discover tools
    #         tools_response = await self.session.list_tools()
    #         self.tools = [
    #             {
    #                 "name": tool.name,
    #                 "description": tool.description or "",
    #                 "input_schema": tool.inputSchema,
    #                 "server": self.name,
    #             }
    #             for tool in tools_response.tools
    #         ]
            
    #         # Discover resources
    #         resources_response = await self.session.list_resources()
    #         self.resources = [
    #             {
    #                 "uri": resource.uri,
    #                 "name": resource.name or "",
    #                 "description": resource.description or "",
    #                 "server": self.name,
    #             }
    #             for resource in resources_response.resources
    #         ]
            
    #         self.connected = True
    #         logger.info(
    #             "mcp_client_initialized",
    #             server=self.name,
    #             tools=len(self.tools),
    #             resources=len(self.resources),
    #         )
            
    #     except Exception as e:
    #         logger.error("mcp_client_init_failed", server=self.name, error=str(e))
    #         raise

class DiscoveredTool:
    """Tool discovered from MCP server."""
    
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        server_name: str,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.server_name = server_name
        
        # Extract required parameters
        self.required_params = input_schema.get("required", [])
        self.properties = input_schema.get("properties", {})

class DiscoveredResource:
    """Resource discovered from MCP server."""
    
    def __init__(
        self,
        name: str,
        uri: str,
        description: str,
        # input_schema: dict,
        server_name: str,
    ):

        self.name = name
        self.uri = uri
        self.description = description
        self.server_name = server_name
        # self.input_schema = input_schema
    
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
    - Dynamic tool or resource selection
    - Query decomposition
    - Result aggregation
    """
    
    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        context_manager: Optional[ContextManager] = None,
        ollama_client: Optional[OllamaClient] = None,
        auth_middleware: Optional[AuthMiddleware] = None,
        health_checker: Optional[HealthChecker] = None,
    ):
        """Initialize MCP Host."""
        self.settings = get_settings()
        
        # Core components
        self.session_manager = session_manager or SessionManager()
        self.context_manager = context_manager or ContextManager()
        self.ollama_client = ollama_client or OllamaClient()
        self.auth_middleware = auth_middleware
        self.health_checker = health_checker or HealthChecker(self)
        
        # MCP clients
        # self.clients: dict[str, MCPClientWrapper] = {}
        #MCP servers
        self.server_configs: list[ServerConfig] = []
        self.discovered_tools: dict[str, DiscoveredTool] = {}  # tool_name -> DiscoveredTool
        self.discovered_resources: dict[str, DiscoveredResource] = {}  # resource_name -> DiscoveredResource


        # ReAct configuration
        self.max_reasoning_steps = 5
        self.react_system_prompt = self._build_react_system_prompt()
        
        logger.info("mcp_host_initialized")
    
    def _build_react_system_prompt(self) -> str:
        """Build system prompt for ReAct reasoning."""
        return """You are an intelligent assistant which uses multiple specialized tools, sometimes combined, to answer user queries.

CRITICAL: You MUST follow this EXACT format for EVERY response:
You MUST use ReAct (Reasoning + Acting) to solve queries:
1. **Thought**: Analyze the query and plan your approach, explain what you are going to do
2. **Action**: Choose the most appropriate tool(s) to use
3. **Observation**: Process the tool results
4. Repeat if needed, or provide final answer

**If you need to use a TOOL:**

Thought: [explain what you need to do]

Action: [exact_tool_name]

Action Input: {"arg1": "value1", "arg2": "value2"}

**To read a RESOURCE (for formatted data):**

Thought: [explain why you need this data]

Action: read_resource

Action Input: {"uri": "resource://path/here"}

**If you have completed your reasoning and/or have enough information:**

Final Answer: [your complete answer to the user]


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

RULES:

1. ALWAYS start with "Thought:" or "Final Answer:"
2. Action MUST be EXACTLY one of the available tools OR "read_resource"
3. Action Input MUST be valid JSON
4. For weather queries, identify the LOCATION first
5. You MAY chain multiple tools to get the final answer
6. NEVER fabricate tool names or actions
7. NEVER provide incomplete JSON in Action Input
8. ALWAYS provide a Final Answer when you have enough information
9. NEVER make up data - use only tool or resources results
10. Resources provide read-only formatted data (like current weather displays)
11. Tools perform queries and return structured data

EXAMPLE:

User: "What's the weather in Rome?"

Thought: I need to get formatted weather data for Rome, Italy. I should use the Italian city weather resource.

Action: read_resource

Action Input: {"uri": "weather://italy/current/Rome"}

EXAMPLE:

User: "Is it warmer in Florence or in San Francisco today"

Thought: I need to get weather data for Rome, Italy, and for San Francisco, CA, USA. I should use the both the Italian weather and the USA Weather tool, and compare results.

Action: get_weather_italy

Action Input: {"city_name": "Roma", "forecast_days": 1, "include_hourly": false}

Action: get_weather_usa

Action Input: {"city_name": "San Francisco", "forecast_days": 1, "include_hourly": false}

Observation: The weather in Rome today is 75°F and sunny. The weather in San Francisco today is 65°F and cloudy.

Thought: I have the weather data for both cities. Now I need to compare the temperatures and provide the answer.
"""
    
    def auto_register_servers(self, servers_dir: Optional[Path] = None):
        """
        Automatically discover and register MCP servers.
        
        Args:
            servers_dir: Path to servers directory (optional)
        """
        logger.info("starting_auto_discovery")
        
        # Discover servers
        discovery = ServerDiscovery(servers_dir)
        discovered = discovery.discover_servers()
        
        if not discovered:
            logger.warning("no_servers_discovered")
            return
        
        # Validate and register
        registered_count = 0
        skipped_count = 0
        
        for metadata in discovered:
            is_valid, error_msg = discovery.validate_server(metadata)
            
            if not is_valid:
                logger.warning(
                    "server_skipped",
                    name=metadata.name,
                    reason=error_msg,
                )
                skipped_count += 1
                continue
            
            # Register server
            try:
                config = ServerConfig(
                    name=metadata.name,
                    command="python",
                    args=[str(metadata.path.absolute())],
                    env={"PYTHONPATH": str(metadata.path.parent.parent)},
                    description=metadata.description,
                )
                
                self.register_server(config)
                registered_count += 1
                
                logger.info(
                    "server_auto_registered",
                    name=metadata.name,
                    description=metadata.description,
                )
                
            except Exception as e:
                logger.error(
                    "server_registration_failed",
                    name=metadata.name,
                    error=str(e),
                )
                skipped_count += 1
        
        logger.info(
            "auto_discovery_complete",
            registered=registered_count,
            skipped=skipped_count,
            total=len(discovered),
        )

    async def start(self):
        """Start the MCP host and all components."""
        logger.info("mcp_host_starting")
        
        # Start session manager
        await self.session_manager.start()

        # Avviare un task periodico per i health check
        asyncio.create_task(self._health_check_loop())
        
        # Initialize configured MCP servers
        # await self._initialize_servers()

        # Discover capabilities from all servers
        await self._discover_all_capabilities()
        
        logger.info("mcp_host_started", 
                    servers=len(self.server_configs),
                    tools_discovered=len(self.discovered_tools),
                    resources_discovered=len(self.discovered_resources),
                    )

    async def _health_check_loop(self):
        while True:
            await asyncio.sleep(300) # Ogni 5 minuti
            await self.health_checker.check_all_servers()

    async def stop(self):
        """Stop the MCP host and cleanup."""
        logger.info("mcp_host_stopping")
        
        # Close all MCP clients
        # for client in self.clients.values():
        #     try:
        #         # Cleanup handled by context managers
        #         pass
        #     except Exception as e:
        #         logger.error("client_close_error", server=client.name, error=str(e))
        
        # Stop session manager
        await self.session_manager.stop()
        
        # Close Ollama client
        await self.ollama_client.close()
        
        logger.info("mcp_host_stopped")
    
    def register_server(self, config: ServerConfig):
        """Register an MCP server configuration."""
        if not isinstance(config, ServerConfig):
            logger.error("invalid_server_config_type", config=vars(config))
            raise TypeError("Server configuration must be a ServerConfig instance.")
        self.server_configs.append(config)
        logger.info("server_registered", name=config.name)
    
    # async def _initialize_servers(self):
    #     """Initialize all registered MCP servers."""
    #     for config in self.server_configs:
    #         try:
    #             # Store config - actual connection happens per-query
    #             logger.info("server_config_loaded", server=config.name)
    #         except Exception as e:
    #             logger.error("server_init_failed", server=config.name, error=str(e))
    
    async def _discover_all_capabilities(self):
        """Discover tools and resources from all registered servers."""
        for config in self.server_configs:
            try:
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env,
                )
                
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        
                        # Discover tools
                        tools_response = await session.list_tools()
                        for tool in tools_response.tools:
                            discovered_tool = DiscoveredTool(
                                name=tool.name,
                                description=tool.description or "",
                                input_schema=tool.inputSchema or {},
                                server_name=config.name,
                            )
                            self.discovered_tools[tool.name] = discovered_tool
                        
                        # Discover resources
                        resources_response = await session.list_resources()
                        for resource in resources_response.resources:
                            discovered_resource = DiscoveredResource(
                                uri=resource.uri,
                                name=resource.name or "",
                                description=resource.description or "",
                                server_name=config.name,
                            )
                            self.discovered_resources[resource.uri] = discovered_resource
                        
                        logger.info(
                            "capabilities_discovered",
                            server=config.name,
                            tools=len(tools_response.tools),
                            resources=len(resources_response.resources),
                        )
            except Exception as e:
                logger.error("capability_discovery_failed", server=config.name, error=str(e))


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
        logger.info("query_received", session_id=session_id, query=query)
        
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
            if step.thought.startswith("Final Answer:"):
                final_answer = step.thought.replace("Final Answer:", "").strip()
                logger.info("final_answer_reached", step=step_num)
                break

            # Execute action if specified

            if step.action and step.action_input:
                if step.action not in self.discovered_tools:
                    logger.error(
                        "tool_not_found_in_discovered",
                        tool=step.action,
                        available=list(self.discovered_tools.keys())[:5],
                    )
                    step.observation = f"Error: Tool '{step.action}' not found. Available tools: {', '.join(list(self.discovered_tools.keys())[:5])}"
                else:
                    logger.info("executing_tool", tool=step.action, input=step.action_input)
                    observation = await self._execute_action(
                        session_id=session_id,
                        action=step.action,
                        action_input=step.action_input,
                    )
                    step.observation = observation
                    logger.debug("tool_result_received", length=len(observation))
            else:
                logger.warning("no_action_parsed", thought=step.thought[:100])
        
        # If no final answer after max steps, synthesize one
        if not final_answer:
            logger.warning("max_steps_reached_synthesizing")
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
                    "action_input": s.action_input,
                    "observation": s.observation[:500] if s.observation else None,  # Truncate
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
        
        # Build context from previous steps
        context_parts = []

        if previous_steps:
            context_parts.append("Previous Steps:")
            for step in previous_steps:
                context_parts.append(f"\nStep {step.step_num}:")
                context_parts.append(f"Thought: {step.thought}")
                if step.action:
                    context_parts.append(f"Action: {step.action}")
                    context_parts.append(f"Action Input: {json.dumps(step.action_input)}")
                if step.observation:
                    # Truncate long observations
                    obs_preview = step.observation[:500]
                    context_parts.append(f"Observation: {obs_preview}")

        # Build prompt
        tools_desc = await self._get_tools_description()

        if previous_steps:
            prompt = f"""Question: {original_query}
{chr(10).join(context_parts)}
Based on the observations above, what should you do next?
Remember: Start with "Thought:" or "Final Answer:"
{tools_desc}"""
        else:
            prompt = f"""Question: {original_query}
What is your first step?
{tools_desc}
Remember: Start with "Thought:" and then specify Action and Action Input."""
        
        # Get LLM response
        messages = [
            OllamaMessage(role="system", content=self.react_system_prompt),
            OllamaMessage(role="user", content=prompt),
        ]
        logger.debug("calling_llm", prompt_length=len(prompt))
        response = await self.ollama_client.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=500  # Lower temperature for more focused reasoning
        )
        
        # Parse response
        step = self._parse_reasoning_response(
            step_num=len(previous_steps) + 1,
            response_text=response.content,
        )
        
        return step
    
    def _parse_reasoning_response(self, step_num: int, response_text: str) -> ReActStep:
        """Parse LLM response into ReActStep."""

        logger.debug("parsing_response", text=response_text)
        # Clean response
        text = response_text.strip()
        
        thought = ""
        action = None
        action_input = None
        
        # Check for Final Answer first

        if "Final Answer:" in text:
            thought = text
            return ReActStep(step_num=step_num, thought=thought)

        # Split by lines
        lines = text.split("\n")

        current_section = None
        action_input_lines=[]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove markdown formatting
            line = line.replace("**", "")
            if line.startswith("Thought:"):
                current_section = "thought"
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                current_section = "action"
                action_text = line.replace("Action:", "").strip()
                # Clean action name
                action = action_text.split()[0] if action_text else None
            elif line.startswith("Action Input:"):
                current_section = "action_input"
                input_text = line.replace("Action Input:", "").strip()
                action_input_lines.append(input_text)
            elif current_section == "thought" and not any(line.startswith(x) for x in ["Action:", "Final", "Observation:"]):
                thought += " " + line
            elif current_section == "action_input" and not any(line.startswith(x) for x in ["Thought:", "Final", "Observation:"]):
                action_input_lines.append(line)
        # Parse action input
        if action_input_lines:
            input_text = " ".join(action_input_lines)
            try:
                # Try to extract JSON
                if "{" in input_text and "}" in input_text:
                    json_start = input_text.index("{")
                    json_end = input_text.rindex("}") + 1
                    json_str = input_text[json_start:json_end]
                    action_input = json.loads(json_str)
                    logger.debug("parsed_action_input", input=action_input)
                else:
                    # Fallback: create simple dict
                    action_input = {"query": input_text}
            except json.JSONDecodeError as e:
                logger.warning("json_parse_failed", error=str(e), text=input_text[:100])
                action_input = {"raw": input_text}
        logger.info(
            "step_parsed",
            thought_length=len(thought),
            action=action,
            has_input=bool(action_input),
        )
        
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
        """Execute a tool action OR read a resource."""
        
        logger.info("executing_action", action=action, input=action_input)
        
        try:
            # Check if it's a resource read request
            if action == "read_resource" and "uri" in action_input:
                return await self._read_resource(action_input["uri"])

            # Otherwise, it's a tool call
            # Get tool info
            tool = self.discovered_tools.get(action)
            if not tool:
                logger.error("tool_not_found", action=action)
                return f"Error: Tool '{action}' not found"         

            # Get server config
            server_config = next(
                (c for c in self.server_configs if c.name == tool.server_name),
                None
            )
            
            if not server_config:
                return f"Error: Server '{tool.server_name}' not configured"
            
            logger.info("calling_mcp_tool", server=server_config.name, tool=action)

            # Execute tool via MCP
            result = await self._call_mcp_tool(
                server_config=server_config,
                tool_name=action,
                arguments=action_input,
            )
            
            # Format result

            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            return str(result)
            
        # except Exception as e:
        #     logger.error("action_execution_error", error=str(e), action=action)
        #     return f"Error executing action: {str(e)}"
        # except self.mcp.errors.McpError as e:
        #     logger.error("mcp_protocol_error", server=tool.server_name, error=str(e))
        #     return f"Error (MCP Protocol): {e.message}"
        except ConnectionRefusedError as e: # O simili, a seconda del client
            logger.error("server_connection_refused", server=tool.server_name, error=str(e))
            return f"Error: Could not connect to server {tool.server_name}"
        except asyncio.TimeoutError as e:
            logger.error("server_timeout", serveer=tool.server_name, error=str(e))
            return f"Error: Server {tool.server_name} timed out"
        except json.JSONDecodeError as e:
            logger.error("server_response_json_error", server=tool.server_name, response_text=str(result), error=str(e))
            return f"Error: Invalid JSON response from {tool.server_name}"
        except Exception as e:
            logger.error("unexpected_tool_execution_error", tool=action, server=tool.server_name, error=str(e))
            return f"Error: An unexpected error occurred while executing {action} on {tool.server_name}"

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
                    for content in result.content:
                        if hasattr(content, 'text'):
                            try:
                                return json.loads(content.text)
                            except json.JSONDecodeError:
                                return content.text
                        return str(content)
                return str(result)
    
    async def _read_resource(self, uri: str) -> str:
        """Read a resource from MCP server."""
        try:
            # Find which server has this resource
            resource = self.discovered_resources.get(uri)
            if not resource:
                logger.error("resource_not_found", uri=uri)
                return f"Error: Resource '{uri}' not found"
            
            # Get server config
            server_config = next(
                (c for c in self.server_configs if c.name == resource.server_name),
                None
            )
            if not server_config:
                return f"Error: Server '{resource.server_name}' not configured"
            logger.info("reading_mcp_resource", server=server_config.name, uri=uri)

            # Read resource via MCP
            result = await self._call_mcp_resource(
                server_config=server_config,
                resource_uri=uri,
            )
            return str(result)

        except Exception as e:
            logger.error("resource_read_error", error=str(e), uri=uri)

            return f"Error reading resource {uri}: {str(e)}"

    async def _call_mcp_resource(
        self,
        server_config: ServerConfig,
        resource_uri: str,
    ) -> str:
        """Read a resource via MCP connection."""
        server_params = StdioServerParameters(
            command=server_config.command,
            args=server_config.args,
            env=server_config.env,
        )
        logger.debug("stdio_connection_for_resource", uri=resource_uri)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Read resource
                result = await session.read_resource(resource_uri)

                # Extract content
                if hasattr(result, 'contents') and result.contents:
                    for content in result.contents:
                        if hasattr(content, 'text'):
                            return content.text
                        elif hasattr(content, 'blob'):
                            return f"Binary data: {len(content.blob)} bytes"
                        return str(content)
                return str(result)
            
    # def _infer_server(self, tool_name: str) -> str:
    #     """Infer server name from tool name."""
    #     tool_lower = tool_name.lower()
        
    #     if "italy" in tool_lower or "italian" in tool_lower:
    #         return "weather_italy"
    #     elif "usa" in tool_lower or "us_" in tool_lower or "current_conditions" in tool_lower:
    #         return "weather_usa"
    #     elif "graph" in tool_lower or "neo4j" in tool_lower or "cypher" in tool_lower:
    #         return "neo4j"
        
    #     # Default to first available server
    #     return self.server_configs[0].name if self.server_configs else "unknown"
    
    async def _get_tools_description(self) -> str:
        """Get description of all available tools AND resources."""
        if not self.discovered_tools and not self.discovered_resources:
            return "No tools or resources available."

        descriptions = []
        
        # === TOOLS SECTION ===

        if self.discovered_tools:
            descriptions.append("Available Tools (use EXACT names):")
            # Group tools by server
            tools_by_server: dict[str, list[DiscoveredTool]] = {}
            for tool in self.discovered_tools.values():
                if tool.server_name not in tools_by_server:
                    tools_by_server[tool.server_name] = []
                tools_by_server[tool.server_name].append(tool)

            for server_name, tools in tools_by_server.items():
                descriptions.append(f"\n{server_name}:")
                for tool in tools:
                    descriptions.append(f"  • {tool.name}")
                    descriptions.append(f"    {tool.description}")
                    # Show required parameters
                    if tool.required_params:
                        descriptions.append(f"    Required: {', '.join(tool.required_params)}")
                    # Show parameter types
                    if tool.properties:
                        param_details = []
                        for param, schema in tool.properties.items():
                            param_type = schema.get('type', 'any')
                            param_details.append(f"{param}:{param_type}")
                        descriptions.append(f"    Parameters: {', '.join(param_details)}")

         # === RESOURCES SECTION ===

        if self.discovered_resources:

            descriptions.append("\n\nAvailable Resources (read-only data):")

            # Group resources by server
            resources_by_server: dict[str, list[DiscoveredResource]] = {}
            for resource in self.discovered_resources.values():
                if resource.server_name not in resources_by_server:
                    resources_by_server[resource.server_name] = []
                resources_by_server[resource.server_name].append(resource)

            for server_name, resources in resources_by_server.items():
                descriptions.append(f"\n{server_name}:")
                for resource in resources:
                    descriptions.append(f"  • {resource.uri}")
                    if resource.name:
                        descriptions.append(f"    Name: {resource.name}")
                    descriptions.append(f"    {resource.description}")
                    descriptions.append(f"    Access via: read_resource('{resource.uri}')")
        descriptions.append("\n\nNote: Use tools for actions/queries. Use resources for reading static/formatted data.")

        return "\n".join(descriptions)
    
    async def _synthesize_answer(
        self,
        session_id: str,
        react_steps: list[ReActStep],
    ) -> str:
        """Synthesize final answer from reasoning steps."""
        
        # Collect all observations
        observations = [
            f"Step {step.step_num}: {step.observation}"
            for step in react_steps
            if step.observation
        ]
        
        if not observations:
            return "I couldn't gather enough information to answer your question. Please try rephrasing."
        
        # Ask LLM to synthesize
        synthesis_prompt = f"""Based on this information:
{chr(10).join(observations)}
Provide a clear, direct answer to the user's question."""
        messages = [
            OllamaMessage(role="user", content=synthesis_prompt),
        ]
        response = await self.ollama_client.chat(messages=messages, temperature=0.3)        
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
            "discovered_tools": list(self.discovered_tools.keys()),
            "discovered_resources": list(self.discovered_resources.keys()),
        
        }
