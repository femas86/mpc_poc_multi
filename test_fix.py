import asyncio
import os
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_with_pythonpath():
    project_root = Path.cwd()
    
    server_params = StdioServerParameters(
        command="python",
        args=[str(project_root / "servers" / "weather_us" / "server_us.py")],
        env={
            "PYTHONPATH": str(project_root),  # ← FIX!
        },
    )
    
    print("Testing with PYTHONPATH fix...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected!")
            
            tools = await session.list_tools()
            print(f"✓ Discovered {len(tools.tools)} tools")
            for tool in tools.tools:
                print(f"  • {tool.name}")

asyncio.run(test_with_pythonpath())