"""
Complete initialization and usage example demonstrating:
- Multi-server registration
- Session management
- ReAct reasoning with tool chaining
- Context management
"""
import asyncio
from pathlib import Path



async def main():
    from host.mcp_host import MCPHost, ServerConfig
    from host.session_manager import SessionManager
    from host.context_manager import ContextManager
    from host.ollama_client import OllamaClient
    from host.auth_middleware import AuthMiddleware
    from config.auth_config import AuthConfig

    # Get project root

    project_root = Path(__file__).parent
    
    # Initialize core components
    auth_config = AuthConfig(secret_key="your-secret-key-change-in-production", token_expiry=3600)
    session_manager = SessionManager(auth_config=auth_config)
    context_manager = ContextManager()
    ollama_client = OllamaClient()
    auth_middleware = AuthMiddleware(None, auth_config, session_manager=session_manager)
    
    # Create MCP Host
    host = MCPHost(
        session_manager=session_manager,
        context_manager=context_manager,
        ollama_client=ollama_client,
        auth_middleware=auth_middleware,
    )
    
    # Register MCP servers
    host.register_server(ServerConfig(
        name="weather_italy",
        command="python",
        args=[str(project_root / "servers/weather_ita/server_it.py")],
        description="Italian weather forecasts using Open-Meteo API with geocoding",
    ))
    
    host.register_server(ServerConfig(
        name="weather_usa",
        command="python",
        args=[str(project_root / "servers/weather_us/server_us.py")],
        description="US weather forecasts using National Weather Service API",
    ))
    
    # host.register_server(ServerConfig(
    #     name="neo4j",
    #     command="python",
    #     args=["servers/neo4j_graph/server.py"],
    #     description="Neo4j graph database queries and relationship analysis",
    # ))
    
    # Start host
    await host.start()
    
    # Create user session
    session_id, token = await session_manager.create_session(
        user_id="user_demo",
        metadata={"client": "cli", "version": "1.0"},
    )
    
    # Create conversation context
    await context_manager.create_context(
        session_id=session_id,
        system_message="You are a helpful assistant with access to weather data and graph databases.",
    )
    
    print("=" * 80)
    print("MCP HOST DEMO - ReAct Reasoning with Multi-Server Orchestration")
    print("=" * 80)
    
    # Example 1: Simple single-tool query
    print("\n[EXAMPLE 1] Simple Weather Query")
    print("-" * 80)
    result1 = await host.process_query(
        session_id=session_id,
        query="What's the weather like in Rome today?",
        token=token,
    )
    print(f"Answer: {result1['answer']}")
    print(f"Reasoning Steps: {len(result1['reasoning_steps'])}")
    for step in result1['reasoning_steps']:
        print(f"  Step {step['step']}: {step['thought'][:200]}...")
    
    # Example 2: Comparison requiring multiple tools
    print("\n[EXAMPLE 2] Multi-Tool Comparison Query")
    print("-" * 80)
    result2 = await host.process_query(
        session_id=session_id,
        query="Compare the weather in New York and Milan this week",
        token=token,
    )
    print(f"Answer: {result2['answer']}")
    print(f"Reasoning Steps: {len(result2['reasoning_steps'])}")
    for step in result2['reasoning_steps']:
        print(f"  Step {step['step']}: {step['thought'][:200]}...")
    
    # Example 3: Temperature comparison
    print("\n[EXAMPLE 3] Complex Comparison Query")
    print("-" * 80)
    result3 = await host.process_query(
        session_id=session_id,
        query="Is it warmer in San Francisco or Florence right now?",
        token=token,
    )
    print(f"Answer: {result3['answer']}")
    for step in result3['reasoning_steps']:
        print(f"  Step {step['step']}: {step['thought'][:200]}...")
    
    # Example 4: Forecast query with specifics
    print("\n[EXAMPLE 4] Detailed Forecast Query")
    print("-" * 80)
    result4 = await host.process_query(
        session_id=session_id,
        query="Will it rain in Seattle tomorrow?",
        token=token,
    )
    print(f"Answer: {result4['answer']}")
    for step in result4['reasoning_steps']:
        print(f"  Step {step['step']}: {step['thought'][:200]}...")
    
    # Get session info
    print("\n[SESSION INFO]")
    print("-" * 80)
    info = await host.get_session_info(session_id)
    print(f"Session ID: {info['session']['session_id']}")
    print(f"Messages: {info['context']['message_count']}")
    print(f"Registered Servers: {', '.join(info['registered_servers'])}")
    
    # Cleanup
    await host.stop()
    print("\n" + "=" * 80)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
