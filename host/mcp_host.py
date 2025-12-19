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
from host.ollama_client import OllamaClient, OllamaMessage
from host.context_manager import ContextManager
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
        return """
    ### ROLE
    You are a reasoning engine that solves queries by looping through Thought, Action, and Action Input. You have access to a dynamic set of Tools and Resources provided in the user prompt.

    CRITICAL: You MUST follow this EXACT format for EVERY response:
    1. **Thought**: Analyze the query, the "Available Tools" and "Available Resources" sections below. Plan your next move.
    2. **Action**: Choose exactly ONE tool name or use "read_resource"
    3. **Action Input**: Provide arguments in valid JSON format matching the tool's schema.

    **CRITICAL: STOP after writing the Action Input. Do NOT write an "Observation". The system will provide the Observation to you in the next turn.**

    ### EXIT CONDITION
    If the available information is sufficient:
    **Final Answer**: [Your complete, helpful response to the user]

    **If you need to use a TOOL:**

    Thought: [explain what you need to do]
    Action: [exact_tool_name]
    Action Input: {"arg1": "value1", "arg2": "value2"}

    **To read a RESOURCE (for formatted data):**

    Thought: [explain why you need this data]
    Action: read_resource
    Action Input: {"uri": "resource://path/here"}

    ### CRITICAL INSTRUCTIONS:
    - For weather queries, identify the location FIRST
    - Use Weather Italy for Italian cities
    - Use Weather USA for US locations
    - You can chain multiple tools if needed
    - Always provide clear, concise final answers
    - If uncertain about location, ASK for clarification

    ### RULES:

    - NEVER fabricate data. Only use what is provided in an "Observation:".
    - NEVER fabricate tools or resource, use only available.
    - You can only perform ONE action per turn.
    - Action Input MUST be a valid JSON object.
    - If a location is ambiguous, use "Thought:" to explain why and then "Final Answer:" to ask the user for clarity.
    - If the user says "Hello" or "Hi", respond with "Final Answer:". Do NOT call a tool.
    - If no action can help, use "Final Answer:" to explain why
    - CRITICAL: If you receive an error from a tool, DO NOT invent tools. Use "Final Answer:" to inform the user the service is unavailable.

"""
#  esempi rimossi dal prompt qui sopra per rimpicciolire la finestra di contesto del modello utilizzata per il system prompt
#  EXAMPLE (resource use):
    # User: "What's the weather in Rome?"
    # Thought: I need to get formatted weather data for Rome, Italy. I should use the Italian city weather resource.
    # Action: read_resource
    # Action Input: {"uri": "weather://italy/current/Rome"}
    # [STOP]
# 
    # EXAMPLE (Chained tool use):
    # User: "Is it warmer in Florence or in San Francisco today"
    # Thought: I need to get weather data for Rome, Italy, and for San Francisco, CA, USA. I should use the both the Italian weather and the USA Weather tool, and compare results.
    # Action: get_weather_italy
    # Action Input: {"city_name": "Roma", "forecast_days": 1, "include_hourly": false}
    # Action: get_weather_usa
    # Action Input: {"city_name": "San Francisco", "forecast_days": 1, "include_hourly": false}
    # Observation: The weather in Rome today is 75°F and sunny. The weather in San Francisco today is 65°F and cloudy.
    # Thought: I have the weather data for both cities. Now I need to compare the temperatures and provide the answer.
# 
    # EXAMPLE (Final Answer):
    # Observation: {"temp": "12°C", "condition": "Rain"}
    # Thought: I now have the weather data for Venice. I can provide the final answer.
    # Final Answer: The current weather in Venice is 12°C with rain.
    
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
    
    def extract_json_robust(text: str) -> dict:
        """
        Finds the first JSON-like object in a string and attempts to parse it.
        Useful for small models that 'leak' prose before or after JSON.
        """
        try:
            # Regex to find everything between the first '{' and the last '}'
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM output")
                
            json_str = match.group(0)
            
            # Clean up potential common LLM formatting issues
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted JSON: {e}")
            # Fallback: if the model leaked text AFTER the JSON, 
            # try to find the last valid '}' and cut there.
            try:
                last_bracket = json_str.rfind('}')
                if last_bracket != -1:
                    return json.loads(json_str[:last_bracket + 1])
            except:
                pass
            raise

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
                    # Look for a close match
                    close_matches = difflib.get_close_matches(step.action, list(self.discovered_tools.keys()), n=1, cutoff=0.6)
                    if close_matches:
                        suggested = close_matches[0]
                        logger.warning(f"Fuzzy matching '{step.action}' to '{suggested}'")
                        step.observation = f"Note: '{step.action}' not found. Did you mean '{suggested}'?"
                    else:
                        logger.error(
                            "tool_not_found_in_discovered",
                            tool=step.action,
                            available=list(self.discovered_tools.keys()),
                        )
                        step.observation = (
                            f"Error: Tool '{step.action}' not found."
                            f"Available tools: {', '.join(list(self.discovered_tools.keys()))}"
                        )
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
        resources_desc = await self._get_resources_description()
        #prompts_desc = await self._get_mcp_prompts_description() # TODO Optional: MCP Prompts

        if previous_steps:
            prompt = f"""Question: {original_query}
{chr(10).join(context_parts)}

{tools_desc}
{resources_desc}

Based on the observations, tools and resources above, what should you do next?
Remember: Start with "Thought:" or "Final Answer:" 
"""
        else:
            prompt = f"""Question: {original_query}
What is your first step?
{tools_desc}
{resources_desc}
Based on the query and the available tools and resources, what should you do next?
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
        
        # Se il contenuto non contiene 'Action:', allora è probabilmente una risposta finale
        if "Action:" not in response.content and "Final Answer:" not in response.content:
            # Se il 1B è confuso, forziamo noi la chiusura
            return response.content

        if "Final Answer:" in response.content:
            return response.content.split("Final Answer:")[-1].strip()

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
            # try:
            #     # Try to extract JSON
            #     if "{" in input_text and "}" in input_text:
            #         json_start = input_text.index("{")
            #         json_end = input_text.rindex("}") + 1
            #         json_str = input_text[json_start:json_end]
            #         action_input = json.loads(json_str)
            #         logger.debug("parsed_action_input", input=action_input)
            #     else:
            #         # Fallback: create simple dict
            #         action_input = {"query": input_text}
            # except json.JSONDecodeError as e:
            #     logger.warning("json_parse_failed", error=str(e), text=input_text[:100])
            #     action_input = {"raw": input_text}
            try:
                # Use a more aggressive "First-Match" approach
                match = re.search(r'(\{.*?\})', input_text, re.DOTALL) # The '?' makes it non-greedy
                if match:
                    json_str = match.group(1)
                    action_input = json.loads(json_str)
                else:
                    action_input = {"query": input_text}
            except json.JSONDecodeError:
                # Fallback to your rindex logic if the non-greedy match fails
                try:
                    json_start = input_text.index("{")
                    json_end = input_text.rindex("}") + 1
                    action_input = json.loads(input_text[json_start:json_end])
                except:
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
