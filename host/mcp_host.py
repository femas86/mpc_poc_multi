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
import re
import difflib
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from shared.logging_config import get_logger
from config.settings import get_settings
from host.ollama_client import OllamaClient, ToolDefinition
from host.context_manager import ContextManager, Message
from host.session_manager import SessionManager
from host.auth_middleware import AuthMiddleware
from host.health_checker import HealthChecker
from config.server_config import ServerConfig
from host.server_discovery import ServerDiscovery, ServerMetadata


logger = get_logger(__name__)

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

    def get(self, attribute: str, default: str = None) -> Any:
        return getattr(self, attribute, default)

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
        #self.react_system_prompt = self._build_react_system_prompt()
        
        logger.info("mcp_host_initialized")
    
    def _build_react_system_prompt(self) -> str:
        """Build system prompt for ReAct reasoning."""
        return """
You are a ReAct multi-tool assistant. When receive a query, think about the tools you have at disposal and USE them.
You operate in a Thought -> Action -> Observation loop. For every step, first explain your reasoning (Thought). 
Then, if you NEED data, call a tool (Action). BE DECISIVE, CALL IT. After receiving the data (Observation), analyze it and decide if you have the final answer or if you need another tool.
RULES:
1. To provide weather information, you MUST have coordinates (lat/lon).
2. If the user provides a city or a State name, FIRST use the geolocation correct tool.
   - For Italy: use 'search_italy_location'
   - For USA: use 'search_us_location'
3. AFTER receiving the coordinates, USE the relative weather tool.
5. If a tool returns an error, inform the user clearly. NEVER INVENT answers
6. If you decide an action is needed, you MUST provide the tool call in the same turn. Do not just say you will do it.
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
        
    async def _discover_all_capabilities(self):
        #TIMEOUT_SECONDS = 20  # Tempo massimo per ogni server
        """Discover tools and resources from all registered servers."""
        for config in self.server_configs:
            try:
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env,
                )
                #async with asyncio.timeout(TIMEOUT_SECONDS):
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                    
                        # Discover tools
                        tools_response = await session.list_tools()
                        for tool in tools_response.tools:
                            self.discovered_tools[tool.name] = DiscoveredTool(
                                name=tool.name,
                                description=tool.description or "",
                                input_schema=tool.inputSchema or {},
                                server_name=config.name,
                            )

                        
                        # Discover resources
                        try:
                            static_resources_response = await session.list_resources()
                            for resource in static_resources_response.resources:
                                discovered_resource = DiscoveredResource(
                                    uri=resource.uri,
                                    name=resource.name or "",
                                    description=resource.description or "",
                                    server_name=config.name,
                                )
                                self.discovered_resources[resource.uri] = discovered_resource
                        except Exception as e:
                            logger.warning(
                                "static_resource_discovery_failed",
                                server=config.name,
                                error=str(e),
                            )

                        try:
                            template_resources_response = await session.list_resource_templates()
                            for resource in template_resources_response.resourceTemplates:
                                discovered_resource = DiscoveredResource(
                                    uri=resource.uriTemplate,
                                    name=resource.name or "",
                                    description=resource.description or "",
                                    server_name=config.name,
                                )
                                self.discovered_resources[resource.uriTemplate] = discovered_resource
                        except Exception as e:
                            logger.warning(
                                "resource_template_discovery_failed",
                                server=config.name,
                                error=str(e),
                            )

                        logger.info(
                            "capabilities_discovered",
                            server=config.name,
                            tools=len(tools_response.tools),
                            resources=len(static_resources_response.resources) + len(template_resources_response.resourceTemplates),
                        )
            # except asyncio.TimeoutError:
            #     logger.error("discovery_timeout", server=config.name, timeout=TIMEOUT_SECONDS)                
            except Exception as e:
                logger.error("capability_discovery_failed", server=config.name, error=str(e))

    async def process_query(
        self,
        session_id: str,
        query: str,
        token: Optional[str] = None,
    ) :
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
                yield {"type":"error", "content": f"Authentication failed: {e}"}
                return
        
        # Add query to context
        await self.context_manager.add_message(
            session_id=session_id,
            role="user",
            content=query,
            tool_calls= None
        )
        
        available_tools = self._convert_to_ollama_tools()
        
        for step_num in range(1, self.max_reasoning_steps + 1):
            logger.debug("react_step", step=step_num, session=session_id)
            
            response = await self.ollama_client.chat(
                messages= await self.context_manager.get_messages_for_llm(session_id),
                tools= available_tools,
                temperature=0.0,
                max_tokens=1000,
            )

            thought = response.content
            if thought:
                logger.info(f"Step {step_num} - Thought: {thought}")
                yield {"type": "thought", "content": thought, "step": step_num}

            if response.tool_calls:
                # Aggiungiamo alla history il pensiero + l'intenzione di agire
                await self.context_manager.add_message(
                    session_id=session_id,
                    role="assistant", 
                    content=thought, 
                    tool_calls=response.tool_calls
                )

                for t_call in response.tool_calls:
                    action_name = t_call['function']['name']
                    action_input = t_call['function']['arguments']
                    
                    yield {"type": "action", "content": action_name, "input": action_input}
                    
                    # --- [3. OBSERVATION] ---
                    observation = await self._execute_mcp_tool(action_name, action_input)
                    logger.info(f"Step {step_num} - Observation: {observation}")
                    yield {"type": "observation", "content": str(observation)}

                    # Aggiungiamo l'osservazione alla history per il prossimo Thought
                    await self.context_manager.add_message(Message(
                        role="tool",
                        name=action_name,
                        content=str(observation),
                        tool_calls= response.tool_calls
                    ))
                
                continue # Torna a pensare basandosi sull'osservazione
            else:
                if thought:
                    # Add response to context
                    await self.context_manager.add_message(
                        session_id=session_id,
                        role="assistant",
                        content=thought,
                        tool_calls=response.tool_calls
                    )
                    
                    logger.info("query_completed", session_id=session_id, steps=step_num)
                    yield {"type":"answer", "content": thought}
                    return
                else:
                    await self.context_manager.add_message(
                        session_id=session_id,
                        role="assistant",
                        content="error",
                        tool_calls= response.tool_calls or None
                    )
                    yield {"type":"error", "content": "il modello non ha prodotto una risposta"}
                    return
        
    def _convert_to_ollama_tools(self) -> list:
        """
        Converte i tool scoperti dai server MCP nel formato JSON Schema 
        richiesto dall'API 'tools' di Ollama.
        """
        ollama_tools = []
        
        # self.discovered_tools è il dizionario dove hai salvato i tool dei server
        # Ogni tool MCP ha tipicamente: 'name', 'description', 'inputSchema'
        for name, tool in self.discovered_tools.items():
            # Costruiamo la struttura richiesta da Ollama/OpenAI standard
            schema = tool.get('input_schema', {})
            props = schema.get('properties', {})
            required = schema.get('required', [])
            tool_definition = ToolDefinition(
                type = "function",
                function = {
                    'name': name,
                    'description': tool.get('description', 'Nessuna descrizione fornita').strip(),
                    'parameters': {
                        'type': 'object',
                        'properties': props,
                        'required': required
                    }
                },
            )
            
            ollama_tools.append(tool_definition)
        
        # Lo standard Ollama Tool Calling non ha un concetto nativo di "Resources". Per loro esiste solo Chat e Tools. Aggiungiamo le risorse come un tool speciale
        if self.discovered_resources: # Se ci sono mcp.resources
            ollama_tools.append(ToolDefinition(
                type = "function",
                function = {
                    'name': 'read_resource',
                    'description': 'Leggi il contenuto di una risorsa specifica (file, log, dati statici).',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'uri': {
                                'type': 'string',
                                'description': 'URI della risorsa da leggere',
                                'enum': list(self.discovered_resources.keys()) # Opzionale: limita ai soli URI esistenti
                            }
                        },
                        'required': ['uri']
                    }
                }
            ))

        return ollama_tools

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
    
    async def _get_resources_description(self) -> str:
        """Get description of all available resources, both static and templates"""
        if not self.discovered_resources:
            return "No resources available."

        descriptions = []
        if self.discovered_resources:

            descriptions.append("\n\nAvailable Resources & Templates (read-only data):")
        
            # Group resources by server
            resources_by_server: dict[str, list[DiscoveredResource]] = {}
            for resource in self.discovered_resources.values():
                if resource.server_name not in resources_by_server:
                    resources_by_server[resource.server_name] = []
                resources_by_server[resource.server_name].append(resource)

            for server_name, resources in resources_by_server.items():
                descriptions.append(f"\n{server_name}:")
                for res in resources:
                    # Rileviamo se è un template cercando le parentesi graffe
                    is_template = "{" in res.uri and "}" in res.uri
                    
                    if is_template:
                        descriptions.append(f"  • TEMPLATE: {res.uri}")
                        descriptions.append(f"    Desc: {res.description or 'Dynamic resource'}")
                        descriptions.append(f"    INSTRUCTION: Replace the placeholder (e.g. {{city}}) with a real value.")
                        descriptions.append(f"    Access via: read_resource(uri=\"{res.uri.replace('{city}', 'rome')}\") <-- Example")
                    else:
                        descriptions.append(f"  • STATIC: {res.uri}")
                        descriptions.append(f"    Desc: {res.description or 'Fixed resource'}")
                        descriptions.append(f"    Access via: read_resource(uri=\"{res.uri}\")")
                    descriptions.append("")
        descriptions.append("\n\nNote: Use resources for reading formatted data.")    
        descriptions.append("\nIMPORTANT: When using a TEMPLATE, you MUST substitute all {parameters} with actual values before calling read_resource.")
        return "\n".join(descriptions)
    
    async def _get_tools_description(self) -> str:
        """Get description of all available tools."""
        if not self.discovered_tools:
            return "No tools available."

        descriptions = []
        
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
        descriptions.append("\n\nNote: Use tools for actions that change state or require computation and/or multiple steps.")
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
        response = await self.ollama_client.chat(messages=messages, temperature=0.1)        
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
