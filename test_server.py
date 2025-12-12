"""
Test script to verify servers expose tools/resources correctly.

Run this to verify discovery works:
"""

import os
from pathlib import Path
from host.mcp_host import MCPHost, ServerConfig

# Get project root
project_root = Path(__file__).parent  # Se main.py è nella root
# oppure
project_root = Path.cwd()  # Current working directory


import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_server_discovery(server_path: str, server_name: str):
    """Test that a server properly exposes tools and resources."""
    
    print(f"\n{'=' * 80}")
    print(f"Testing {server_name} Discovery")
    print('=' * 80)
    
    server_params = StdioServerParameters(
        command="python",
        args=[server_path],
        env={
            "PYTHONPATH": str(project_root),
        },
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize
                await session.initialize()
                print(f"✓ Connected to {server_name}")
                
                # List tools
                tools_response = await session.list_tools()
                print(f"\n📦 Discovered {len(tools_response.tools)} tools:")
                for tool in tools_response.tools:
                    print(f"  • {tool.name}")
                    print(f"    {tool.description}")
                    if tool.inputSchema and "required" in tool.inputSchema:
                        print(f"    Required: {tool.inputSchema['required']}")
                
                # List resources
                resources_response = await session.list_resources()
                print(f"\n📚 Discovered {len(resources_response.resources)} resources:")
                for resource in resources_response.resources:
                    print(f"  • {resource.uri}")
                    print(f"    {resource.description}")
                
                print(f"\n✓ {server_name} discovery successful!")
                
    except Exception as e:
        print(f"\n✗ {server_name} discovery failed: {e}")
        raise


async def test_all_servers():
    """Test discovery for all weather servers."""
    
    # Test Italy server
    # await test_server_discovery(
    #     str(project_root /"servers"/"weather_ita"/"server_it.py"),
    #     "Weather Italy"
    # )
    
    # Test USA server
    await test_server_discovery(
        str(project_root /"servers"/"weather_us"/"server_us.py"),
        "Weather USA"
    )
    
    print("\n" + "=" * 80)
    print("✓ All servers passed discovery test!")
    print("=" * 80)


if __name__ == "__main__":
    # Run discovery tests
    asyncio.run(test_all_servers())
