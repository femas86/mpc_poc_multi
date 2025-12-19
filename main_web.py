"""
Web interface for MCP Multi-Server with dynamic server management.

Features:
- Browser-based chat interface
- Real-time conversation with MCP Host
- Admin panel for server management
- Add/remove servers dynamically
- Server configuration editor
- Session management
- Statistics dashboard
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

# Gradio for web interface
import gradio as gr

# Project imports
from host.mcp_host import MCPHost, ServerConfig
from host.session_manager import SessionManager
from host.context_manager import ContextManager
from host.ollama_client import OllamaClient
from host.auth_middleware import AuthMiddleware
from host.server_discovery import ServerDiscovery, ServerMetadata
from config.auth_config import AuthConfig
from config.settings import get_settings
from shared.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class WebInterface:
    """Web interface for MCP Host with dynamic server management."""
    
    def __init__(self):
        """Initialize web interface."""
        self.host: Optional[MCPHost] = None
        self.session_id: Optional[str] = None
        self.token: Optional[str] = None
        self.project_root = Path(__file__).parent
        
        # Chat history for UI
        self.chat_history: List[Tuple[str, str]] = []
        
        logger.info("web_interface_initialized")
    
    async def initialize_host(self):
        """Initialize MCP Host with auto-discovery."""
        if self.host:
            logger.warning("host_already_initialized")
            return
        
        logger.info("initializing_mcp_host")
        
        settings = get_settings()
        
        # Initialize components
        auth_config = AuthConfig(
            secret_key=settings.auth_secret_key,
            token_expiry=settings.auth_token_expiry,
        )
        session_manager = SessionManager(auth_config=auth_config)
        context_manager = ContextManager()
        ollama_client = OllamaClient()
        auth_middleware = AuthMiddleware(app=None, auth_config=auth_config, session_manager=session_manager)
        
        # Create MCP Host
        self.host = MCPHost(
            session_manager=session_manager,
            context_manager=context_manager,
            ollama_client=ollama_client,
            auth_middleware=auth_middleware,
        )
        
        # Auto-discover and register servers
        self.host.auto_register_servers(servers_dir=self.project_root / "servers")
        
        # Start host
        await self.host.start()
        
        # Create session
        self.session_id, self.token = await session_manager.create_session(
            user_id="web_user",
            metadata={"client": "web", "interface": "gradio"},
        )
        
        # Create context
        await context_manager.create_context(
            session_id=self.session_id,
            system_message="You are a helpful assistant with access to weather data and graph databases.",
        )
        
        logger.info("mcp_host_initialized")
    
    async def shutdown(self):
        """Shutdown MCP Host."""
        if self.host:
            await self.host.stop()
            self.host = None
            logger.info("mcp_host_shutdown")
    
    async def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Process chat message.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple of (empty string for input box, updated history)
        """
        if not self.host:
            await self.initialize_host()
        
        try:
            # Add user message to history
            history.append((message, None))
            
            # Process query
            result = await self.host.process_query(
                session_id=self.session_id,
                query=message,
                token=self.token,
            )
            
            # Update history with assistant response
            history[-1] = (message, result["answer"])
            
            return "", history
            
        except Exception as e:
            logger.error("chat_error", error=str(e))
            error_msg = f"❌ Error: {str(e)}"
            history[-1] = (message, error_msg)
            return "", history
    
    def get_registered_servers(self) -> str:
        """Get list of registered servers as formatted text."""
        if not self.host or not self.host.server_configs:
            return "No servers registered yet."
        
        lines = ["# Registered Servers\n"]
        for config in self.host.server_configs:
            lines.append(f"### {config.name}")
            lines.append(f"- **Description**: {config.description}")
            lines.append(f"- **Command**: `{config.command} {' '.join(config.args)}`")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_discovered_tools(self) -> str:
        """Get list of discovered tools."""
        if not self.host: #or not self.host.discovered_tools:
            return "No tools discovered yet. Start the host first."
        
        lines = ["# Discovered Tools\n"]
        
        # Group by server
        by_server = {}
        for tool_name, tool in self.host.discovered_tools.items():
            if tool.server_name not in by_server:
                by_server[tool.server_name] = []
            by_server[tool.server_name].append(tool)
        
        for server_name, tools in by_server.items():
            lines.append(f"## {server_name}")
            for tool in tools:
                lines.append(f"- **{tool.name}**: {tool.description}")
            lines.append("")
        
        return "\n".join(lines)
    
    def scan_available_servers(self) -> str:
        """Scan servers directory for available servers."""
        try:
            discovery = ServerDiscovery(self.project_root / "servers")
            discovered = discovery.discover_servers()
            
            if not discovered:
                return "No servers found in servers/ directory."
            
            lines = ["# Available Servers\n"]
            for metadata in discovered:
                is_valid, error = discovery.validate_server(metadata)
                
                status = "✅ Ready" if is_valid else f"❌ {error}"
                
                lines.append(f"### {metadata.name}")
                lines.append(f"- **Status**: {status}")
                lines.append(f"- **Description**: {metadata.description}")
                lines.append(f"- **Path**: `{metadata.path}`")
                if metadata.requires_config:
                    lines.append(f"- **Required Config**: {', '.join(metadata.requires_config)}")
                lines.append("")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error scanning servers: {str(e)}"
    
    async def add_server_from_discovery(self, server_name: str) -> str:
        """
        Add a server that was discovered in servers/ directory.
        
        Args:
            server_name: Name of server to add
            
        Returns:
            Status message
        """
        if not self.host:
            return "❌ Please start the host first"
        
        try:
            # Discover servers
            discovery = ServerDiscovery(self.project_root / "servers")
            discovered = discovery.discover_servers()
            
            # Find requested server
            metadata = next((s for s in discovered if s.name == server_name), None)
            
            if not metadata:
                return f"❌ Server '{server_name}' not found"
            
            # Validate
            is_valid, error = discovery.validate_server(metadata)
            if not is_valid:
                return f"❌ Cannot add server: {error}"
            
            # Check if already registered
            if any(c.name == server_name for c in self.host.server_configs):
                return f"⚠️ Server '{server_name}' is already registered"
            
            # Register server
            config = ServerConfig(
                name=metadata.name,
                command="python",
                args=[str(metadata.path.absolute())],
                env={"PYTHONPATH": str(self.project_root)},
                description=metadata.description,
            )
            
            self.host.register_server(config)
            
            # Discover capabilities
            await self.host._discover_server_capabilities(config)
            
            return f"✅ Server '{server_name}' added successfully!"
            
        except Exception as e:
            logger.error("add_server_error", error=str(e))
            return f"❌ Error adding server: {str(e)}"
    
    async def add_custom_server(
        self,
        name: str,
        description: str,
        server_path: str,
    ) -> str:
        """
        Add a custom server with manual configuration.
        
        Args:
            name: Server name
            description: Server description
            server_path: Path to server.py file
            
        Returns:
            Status message
        """
        if not self.host:
            return "❌ Please start the host first"
        
        try:
            # Validate inputs
            if not name or not server_path:
                return "❌ Name and server path are required"
            
            # Validate path
            path = Path(server_path)
            if not path.exists():
                return f"❌ Server file not found: {server_path}"
            
            # Check if already registered
            if any(c.name == name for c in self.host.server_configs):
                return f"⚠️ Server '{name}' is already registered"
            
            # Create config
            config = ServerConfig(
                name=name,
                command="python",
                args=[str(path.absolute())],
                env={"PYTHONPATH": str(self.project_root)},
                description=description or f"Custom server: {name}",
            )
            
            # Register
            self.host.register_server(config)
            
            # Discover capabilities
            await self.host._discover_server_capabilities(config)
            
            return f"✅ Custom server '{name}' added successfully!"
            
        except Exception as e:
            logger.error("add_custom_server_error", error=str(e))
            return f"❌ Error adding server: {str(e)}"
    
    async def remove_server(self, server_name: str) -> str:
        """
        Remove a registered server.
        
        Args:
            server_name: Name of server to remove
            
        Returns:
            Status message
        """
        if not self.host:
            return "❌ No host initialized"
        
        try:
            # Find server
            config = next((c for c in self.host.server_configs if c.name == server_name), None)
            
            if not config:
                return f"❌ Server '{server_name}' not found"
            
            # Remove from configs
            self.host.server_configs.remove(config)
            
            # Remove discovered tools from this server
            tools_to_remove = [
                name for name, tool in self.host.discovered_tools.items()
                if tool.server_name == server_name
            ]
            
            for tool_name in tools_to_remove:
                del self.host.discovered_tools[tool_name]
            
            return f"✅ Server '{server_name}' removed successfully!"
            
        except Exception as e:
            logger.error("remove_server_error", error=str(e))
            return f"❌ Error removing server: {str(e)}"
    
    def get_statistics(self) -> str:
        """Get system statistics."""
        if not self.host:
            return "No statistics available yet."
        
        try:
            lines = ["# System Statistics\n"]
            
            lines.append(f"## Servers")
            lines.append(f"- Registered: {len(self.host.server_configs)}")
            
            lines.append(f"\n## Tools")
            lines.append(f"- Discovered: {len(self.host.discovered_tools)}")
            
            lines.append(f"\n## Resources")
            lines.append(f"- Discovered: {len(self.host.discovered_resources)}")
            
            # Session stats
            if self.session_id:
                lines.append(f"\n## Current Session")
                lines.append(f"- Session ID: `{self.session_id[:16]}...`")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error retrieving stats: {str(e)}"


# Global interface instance
interface = WebInterface()

def create_gradio_interface():
    """Create Gradio web interface."""
    
    with gr.Blocks(
        title="MCP Multi-Server Assistant",
        theme=gr.themes.Soft(),
        css="""
        #chatbot {height: 600px; overflow-y: auto;}
        .server-panel {border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin: 10px 0;}
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 MCP Multi-Server Assistant
        
        Chat with an AI assistant that has access to multiple specialized servers:
        - 🌤️ Weather data (Italy & USA)
        - 🕸️ Graph database (Neo4j)
        - 🧠 Semantic reasoning
        """)
        
        with gr.Tabs():
            
            # ===== TAB 1: CHAT =====
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    elem_id="chatbot",
                    height=600,
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me anything... (e.g., 'What's the weather in Rome?')",
                        scale=4,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                gr.Examples(
                    examples=[
                        "What's the weather in Rome today?",
                        "Compare weather in New York and Milan",
                        "Is it raining in Seattle right now?",
                        "How many nodes are in the graph database?",
                        "Find the shortest path between node 1 and node 5",
                    ],
                    inputs=msg,
                )
                
                # Chat event handlers
                async def chat_fn(message, history):
                    return await interface.chat(message, history)
                
                msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[msg, chatbot])
                submit_btn.click(chat_fn, inputs=[msg, chatbot], outputs=[msg, chatbot])
            
            # ===== TAB 2: SERVER MANAGEMENT =====
            with gr.Tab("⚙️ Server Management"):
                
                gr.Markdown("## 📋 Registered Servers")
                
                registered_display = gr.Markdown(value="Loading...")
                refresh_registered = gr.Button("🔄 Refresh List")
                
                gr.Markdown("---")
                gr.Markdown("## ➕ Add Server")
                
                with gr.Tabs():
                    
                    # Add from discovered
                    with gr.Tab("From Available"):
                        gr.Markdown("Add a server that's already in the `servers/` directory")
                        
                        available_display = gr.Markdown(value="Click scan to see available servers")
                        
                        with gr.Row():
                            scan_btn = gr.Button("🔍 Scan Available Servers")
                            
                        with gr.Row():
                            server_dropdown = gr.Dropdown(
                                label="Select Server",
                                choices=[],
                                interactive=True,
                            )
                            add_discovered_btn = gr.Button("➕ Add Server", variant="primary")
                        
                        add_result = gr.Textbox(label="Result", interactive=False)
                    
                    # Add custom
                    with gr.Tab("Custom Server"):
                        gr.Markdown("Add a server from a custom location")
                        
                        custom_name = gr.Textbox(label="Server Name", placeholder="my_custom_server")
                        custom_desc = gr.Textbox(label="Description", placeholder="My custom MCP server")
                        custom_path = gr.Textbox(
                            label="Server Path",
                            placeholder="/path/to/server.py"
                        )
                        
                        add_custom_btn = gr.Button("➕ Add Custom Server", variant="primary")
                        custom_result = gr.Textbox(label="Result", interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("## ➖ Remove Server")
                
                remove_dropdown = gr.Dropdown(
                    label="Select Server to Remove",
                    choices=[],
                    interactive=True,
                )
                remove_btn = gr.Button("🗑️ Remove Server", variant="stop")
                remove_result = gr.Textbox(label="Result", interactive=False)
                
                # Event handlers for server management
                def refresh_registered_list():
                    return interface.get_registered_servers()
                
                def scan_servers():
                    discovered_text = interface.scan_available_servers()
                    # Extract server names for dropdown
                    names = []
                    try:
                        discovery = ServerDiscovery(interface.project_root / "servers")
                        discovered = discovery.discover_servers()
                        names = [s.name for s in discovered]
                    except:
                        pass
                    return discovered_text, gr.Dropdown(choices=names)
                
                async def add_discovered(server_name):
                    result = await interface.add_server_from_discovery(server_name)
                    registered = interface.get_registered_servers()
                    return result, registered
                
                async def add_custom(name, desc, path):
                    result = await interface.add_custom_server(name, desc, path)
                    registered = interface.get_registered_servers()
                    return result, registered
                
                def update_remove_dropdown():
                    if not interface.host:
                        return gr.Dropdown(choices=[])
                    names = [c.name for c in interface.host.server_configs]
                    return gr.Dropdown(choices=names)
                
                async def remove_server_fn(server_name):
                    result = await interface.remove_server(server_name)
                    registered = interface.get_registered_servers()
                    names = [c.name for c in interface.host.server_configs] if interface.host else []
                    return result, registered, gr.Dropdown(choices=names)
                
                # Connect events
                refresh_registered.click(
                    refresh_registered_list,
                    outputs=registered_display
                )
                
                scan_btn.click(
                    scan_servers,
                    outputs=[available_display, server_dropdown]
                )
                
                add_discovered_btn.click(
                    add_discovered,
                    inputs=server_dropdown,
                    outputs=[add_result, registered_display]
                )
                
                add_custom_btn.click(
                    add_custom,
                    inputs=[custom_name, custom_desc, custom_path],
                    outputs=[custom_result, registered_display]
                )
                
                remove_btn.click(
                    update_remove_dropdown,
                    outputs=remove_dropdown
                ).then(
                    remove_server_fn,
                    inputs=remove_dropdown,
                    outputs=[remove_result, registered_display, remove_dropdown]
                )
            
            # ===== TAB 3: TOOLS & RESOURCES =====
            with gr.Tab("🔧 Tools & Resources"):
                
                gr.Markdown("## 🛠️ Discovered Tools")
                tools_display = gr.Markdown(value="Start the host to see discovered tools")
                refresh_tools = gr.Button("🔄 Refresh Tools")
                
                gr.Markdown("---")
                gr.Markdown("## 📚 Discovered Resources")
                resources_display = gr.Markdown(value="Resources will appear here")
                
                def refresh_tools_fn():
                    return interface.get_discovered_tools()
                
                refresh_tools.click(refresh_tools_fn, outputs=tools_display)
            
            # ===== TAB 4: STATISTICS =====
            with gr.Tab("📊 Statistics"):
                stats_display = gr.Markdown(value="Loading statistics...")
                refresh_stats = gr.Button("🔄 Refresh Statistics")
                
                def refresh_stats_fn():
                    return interface.get_statistics()
                
                refresh_stats.click(refresh_stats_fn, outputs=stats_display)
        
        # Initialize on load
        demo.load(
            lambda: (
                interface.get_registered_servers(),
                interface.get_discovered_tools(),
                interface.get_statistics(),
            ),
            outputs=[registered_display, tools_display, stats_display]
        )
    
    return demo

async def main():
    """Main entry point."""
    
    # Setup logging
    settings = get_settings()
    # print( settings)
    setup_logging(log_level=settings.mcp_log_level, json_logs=False)
    
    logger.info("starting_web_interface")
    
    try:
        # Initialize host
        await interface.initialize_host()
        
        # Create and launch Gradio interface
        demo = create_gradio_interface()
        
        logger.info("launching_gradio_interface")
        
        demo.launch(
            server_name="0.0.0.0",  # Listen on all interfaces
            server_port=7860,
            share=False,  # Set True for public URL
            show_error=True,
        )
        
    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    except Exception as e:
        logger.error("fatal_error", error=str(e))
        raise
    finally:
        # Cleanup
        await interface.shutdown()
        logger.info("web_interface_stopped")


if __name__ == "__main__":
    asyncio.run(main())
