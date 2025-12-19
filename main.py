"""
Main entry point with conversational interface.

Features:
- Interactive CLI chat interface
- MCP Host initialization with all servers
- Session management with authentication
- Conversation history
- Rich formatting with colors
- Command system (/help, /quit, /clear, etc.)
- Error handling and graceful shutdown
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Install 'rich' for better UI: pip install rich")

from host.mcp_host import MCPHost, ServerConfig
from host.session_manager import SessionManager
from host.context_manager import ContextManager
from host.ollama_client import OllamaClient
from host.auth_middleware import AuthMiddleware
from config.auth_config import AuthConfig
from config.settings import get_settings
from shared.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

class ConversationalInterface:
    """Interactive conversational interface for MCP Host."""
    
    def __init__(self, host: MCPHost, session_id: str, token: str):
        """
        Initialize conversational interface.
        
        Args:
            host: MCP Host instance
            session_id: User session ID
            token: Authentication token
        """
        self.host = host
        self.session_id = session_id
        self.token = token
        
        # Rich console for beautiful output
        self.console = Console() if RICH_AVAILABLE else None
        
        # Command handlers
        self.commands = {
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/clear": self._cmd_clear,
            "/history": self._cmd_history,
            "/servers": self._cmd_servers,
            "/tools": self._cmd_tools,
            "/stats": self._cmd_stats,
            "/reset": self._cmd_reset,
        }
        
        self.running = True
    
    def print(self, message: str, style: Optional[str] = None):
        """Print message with optional styling."""
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def print_panel(self, content: str, title: str, style: str = "cyan"):
        """Print content in a panel."""
        if self.console:
            self.console.print(Panel(content, title=title, border_style=style))
        else:
            print(f"\n=== {title} ===")
            print(content)
            print("=" * (len(title) + 8))
    
    def print_markdown(self, content: str):
        """Print markdown content."""
        if self.console:
            self.console.print(Markdown(content))
        else:
            print(content)
    
    async def run(self):
        """Run the conversational interface loop."""
        
        # Welcome message
        self._print_welcome()
        
        # Main loop
        while self.running:
            try:
                # Get user input
                if self.console:
                    user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                else:
                    user_input = input("\nYou: ")
                
                # Skip empty input
                if not user_input.strip():
                    continue
                
                # Check for commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue
                
                # Process as query
                await self._process_query(user_input)
                
            except KeyboardInterrupt:
                self.print("\n\n[yellow]Interrupted by user[/yellow]")
                break
            except EOFError:
                break
            except Exception as e:
                logger.error("interface_error", error=str(e))
                self.print(f"[red]Error: {str(e)}[/red]")
    
    async def _process_query(self, query: str):
        """Process user query through MCP Host."""
        
        # Show thinking indicator
        if self.console:
            with self.console.status("[bold green]Thinking...", spinner="dots"):
                result = await self.host.process_query(
                    session_id=self.session_id,
                    query=query,
                    token=self.token,
                )
        else:
            print("Thinking...")
            result = await self.host.process_query(
                session_id=self.session_id,
                query=query,
                token=self.token,
            )
        
        # Display answer
        self.print("\n[bold green]Assistant:[/bold green]")
        self.print_markdown(result["answer"])
        
        # Optionally show reasoning steps
        if result.get("reasoning_steps"):
            self._show_reasoning_steps(result["reasoning_steps"])
    
    def _show_reasoning_steps(self, steps: list):
        """Show reasoning steps if verbose mode."""
        # Only show if explicitly requested
        # Could add a /verbose command to toggle this
        pass
    
    async def _handle_command(self, command: str):
        """Handle special commands."""
        
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in self.commands:
            await self.commands[cmd](args)
        else:
            self.print(f"[red]Unknown command: {cmd}[/red]")
            self.print("Type [cyan]/help[/cyan] for available commands")
    
    async def _cmd_help(self, args):
        """Show help message."""
        help_text = """
# Available Commands

- **/help** - Show this help message
- **/quit** or **/exit** - Exit the application
- **/clear** - Clear the screen
- **/history** - Show conversation history
- **/servers** - List registered MCP servers
- **/tools** - List available tools
- **/stats** - Show session statistics
- **/reset** - Reset conversation context

# Usage Tips

- Ask natural questions: "What's the weather in Rome?"
- Request comparisons: "Compare weather in NYC and Milan"
- Query the graph: "Find all people in the database"
- Get recommendations: "Recommend similar items to node 5"

# Examples

```
What's the weather in Florence, Italy?
Is it raining in Seattle right now?
Create a node with label Person and name Alice
Find the shortest path between node 1 and node 5
How many companies are in the graph database?
```
        """
        self.print_markdown(help_text)
    
    async def _cmd_quit(self, args):
        """Quit the application."""
        self.print("\n[yellow]Goodbye! 👋[/yellow]")
        self.running = False
    
    async def _cmd_clear(self, args):
        """Clear the screen."""
        os.system('clear' if os.name != 'nt' else 'cls')
        self._print_welcome()
    
    async def _cmd_history(self, args):
        """Show conversation history."""
        try:
            summary = await self.host.context_manager.get_context_summary(self.session_id)
            
            if not summary.get("exists"):
                self.print("[yellow]No conversation history[/yellow]")
                return
            
            context = await self.host.context_manager.get_context(self.session_id)
            
            self.print_panel(
                f"Messages: {summary['message_count']}\n"
                f"Total tokens: {summary['total_tokens']}\n"
                f"Created: {summary['created_at']}",
                "Conversation History",
                "cyan"
            )
            
            # Show recent messages
            if context and context.messages:
                self.print("\n[bold]Recent messages:[/bold]\n")
                for msg in context.messages[-5:]:
                    role_color = "cyan" if msg.role == "user" else "green"
                    self.print(f"[{role_color}]{msg.role.upper()}:[/{role_color}] {msg.content[:100]}...")
        
        except Exception as e:
            self.print(f"[red]Error retrieving history: {e}[/red]")
    
    async def _cmd_servers(self, args):
        """List registered servers."""
        info = await self.host.get_session_info(self.session_id)
        
        if self.console:
            table = Table(title="Registered MCP Servers")
            table.add_column("Server", style="cyan")
            table.add_column("Description", style="white")
            
            for server in self.host.server_configs:
                table.add_row(server.name, server.description)
            
            self.console.print(table)
        else:
            self.print("\n=== Registered MCP Servers ===")
            for server in self.host.server_configs:
                self.print(f"• {server.name}: {server.description}")
    
    async def _cmd_tools(self, args):
        """List available tools."""
        info = await self.host.get_session_info(self.session_id)
        
        discovered_tools = info.get("discovered_tools", [])
        
        if self.console:
            table = Table(title="Available Tools")
            table.add_column("Tool Name", style="cyan")
            table.add_column("Server", style="yellow")
            
            for tool_name in discovered_tools:
                tool = self.host.discovered_tools.get(tool_name)
                if tool:
                    table.add_row(tool_name, tool.server_name)
            
            self.console.print(table)
        else:
            self.print("\n=== Available Tools ===")
            for tool_name in discovered_tools:
                tool = self.host.discovered_tools.get(tool_name)
                if tool:
                    self.print(f"• {tool_name} ({tool.server_name})")
    
    async def _cmd_stats(self, args):
        """Show session statistics."""
        try:
            info = await self.host.get_session_info(self.session_id)
            stats = await self.host.session_manager.get_statistics()
            
            stats_text = f"""
**Session Information:**
- Session ID: `{info['session']['session_id']}`
- User ID: `{info['session']['user_id']}`
- Created: {info['session']['created_at']}

**Context Statistics:**
- Messages: {info['context']['message_count']}
- Total tokens: {info['context']['total_tokens']}

**System Statistics:**
- Active sessions: {stats.active_sessions}
- Total sessions: {stats.total_sessions}
- Average duration: {stats.avg_session_duration:.1f}s

**Discovered Capabilities:**
- Tools: {len(info['discovered_tools'])}
- Resources: {len(info['discovered_resources'])}
            """
            
            self.print_markdown(stats_text)
        
        except Exception as e:
            self.print(f"[red]Error retrieving stats: {e}[/red]")
    
    async def _cmd_reset(self, args):
        """Reset conversation context."""
        try:
            await self.host.context_manager.clear_context(self.session_id)
            
            # Recreate context with system message
            await self.host.context_manager.create_context(
                session_id=self.session_id,
                system_message="You are a helpful assistant with access to weather data and graph databases.",
            )
            
            self.print("[green]✓ Conversation reset[/green]")
        
        except Exception as e:
            self.print(f"[red]Error resetting context: {e}[/red]")
    
    def _print_welcome(self):
        """Print welcome message."""
        welcome = """
# MCP Multi-Server Assistant

Welcome! I have access to:
- 🌤️  **Weather data** for Italian and US cities
- 🕸️  **Graph database** (Neo4j) for complex queries
- 🧠 **Semantic reasoning** with Ollama LLM

Type your questions naturally, or use **/help** for commands.
        """
        
        if self.console:
            self.console.clear()
            self.print_panel(welcome, "Welcome", "green")
        else:
            print("\n" + "=" * 60)
            print(welcome)
            print("=" * 60)


async def initialize_host() -> tuple[MCPHost, str, str]:
    """
    Initialize MCP Host with automatic server discovery.
    
    Returns:
        Tuple of (host, session_id, token)
    """
    logger.info("initializing_mcp_host_with_auto_discovery")
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Load settings
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
    host = MCPHost(
        session_manager=session_manager,
        context_manager=context_manager,
        ollama_client=ollama_client,
        auth_middleware=auth_middleware,
    )
    
    # AUTO-DISCOVER AND REGISTER SERVERS
    # Instead of manual registration, use auto-discovery!
    host.auto_register_servers(servers_dir=project_root / "servers")
    
    # Optional: Manual override for specific servers
    # host.register_server(ServerConfig(...))  # Only if needed
    
    # Start host
    logger.info("starting_mcp_host")
    await host.start()
    
    # Create user session
    logger.info("creating_user_session")
    session_id, token = await session_manager.create_session(
        user_id="cli_user",
        metadata={"client": "cli", "interface": "conversational"},
    )
    
    # Create conversation context
    await context_manager.create_context(
        session_id=session_id,
        system_message="You are a helpful assistant with access to weather data and graph databases.",
    )
    
    logger.info("mcp_host_initialized", session_id=session_id)
    
    return host, session_id, token


async def main():
    """Main entry point."""
    
    # Setup logging
    settings = get_settings()
    setup_logging(
        log_level=settings.mcp_log_level,
        json_logs=False,  # Human-readable for CLI
    )
    
    logger.info("application_starting")
    
    host = None
    
    try:
        # Initialize host
        host, session_id, token = await initialize_host()
        
        # Run conversational interface
        interface = ConversationalInterface(host, session_id, token)
        await interface.run()
        
    except KeyboardInterrupt:
        logger.info("application_interrupted")
    except Exception as e:
        logger.error("application_error", error=str(e))
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if host:
            logger.info("shutting_down")
            await host.stop()
        
        logger.info("application_stopped")


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
