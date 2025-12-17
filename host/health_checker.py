
import asyncio
from typing import Dict, List
from config.server_config import ServerConfig
from shared.logging_config import get_logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = get_logger(__name__)

class HealthChecker:
    def __init__(self, mcp_host_instance):
        self.mcp_host = mcp_host_instance

    async def check_server(self, config: ServerConfig) -> bool:
        logger.info("checking_server_health", server=config.name)
        try:
            # Logica simile a _discover_all_capabilities ma solo per un ping/health
            # o tentare di listare gli strumenti e vedere se va a buon fine
            server_params = StdioServerParameters(command=config.command, args=config.args, env=config.env)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await asyncio.wait_for(session.list_tools(), timeout=10.0) # Timeout breve
                    logger.info("server_healthy", server=config.name)
                    return True
        except Exception as e:
            logger.warning("server_unhealthy", server=config.name, error=str(e))
            return False
    
    async def check_all_servers(self) -> Dict[str, bool]:
        statuses = {}
        tasks = [self.check_server(config) for config in self.mcp_host.server_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, config in enumerate(self.mcp_host.server_configs):
            if isinstance(results[i], Exception):
                logger.error("health_check_failed", server=config.name, error=str(results[i]))
                statuses[config.name] = False
            else:
                statuses[config.name] = results[i]
        return statuses

# Integrare in MCPHost
# class MCPHost:
#     def __init__(self, ...):
#         # ...
#         self.health_checker = HealthChecker(self)
#
#     async def start(self):
#         # ...
#         # Avviare un task periodico per i health check
#         asyncio.create_task(self._health_check_loop())
#
#     async def _health_check_loop(self):
#         while True:
#             await asyncio.sleep(300) # Ogni 5 minuti
#             await self.health_checker.check_all_servers()