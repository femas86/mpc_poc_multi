# The MCP (Model Context Protocol) framework is the backbone of this project, enabling seamless multi-server orchestration and advanced reasoning capabilities. By leveraging MCP, the project integrates diverse services, such as weather data APIs and geocoding tools, into a unified, modular architecture. This ensures scalability, maintainability, and efficient communication between components, making MCP a critical enabler for the project's success.

host/ollama_client.py - Client Ollama Completo

✅ Chat completion con retry automatico
✅ Streaming support con AsyncIterator[str]
✅ Function calling / tool use ready
✅ Embedding generation
✅ Health checks e model listing
✅ Full type safety con Pydantic models

host/context_manager.py - Gestione Contesto Conversazionale

✅ Token tracking e stima
✅ Auto-pruning per window management
✅ System message preservation
✅ Context compression per long conversations
✅ Thread-safe con asyncio locks
✅ Cleanup automatico contexts vecchi

🎯 Caratteristiche Chiave:

Integration Ready: Entrambi i componenti pronti per integrazione con MCP servers
Production Quality: Error handling, retry logic, health checks
Observability: Structured logging su ogni operazione
Type Safety: Pydantic models per validation
Performance: Async-first, context pruning automatico

📊 Metriche Implementate:

Token count per messaggio
Total tokens nel context
Message count tracking
Duration metrics (Ollama)
Health status monitoring

host/session_manager.py - Gestione Sessioni Completa

✅ Lifecycle management (start/stop)
✅ Background cleanup task automatico
✅ Token generation e validation integrati
✅ Activity tracking (access count, timestamps)
✅ Session statistics aggregate
✅ Multi-user support
✅ Metadata storage per session
✅ Thread-safe con async locks

host/auth_middleware.py - Middleware Autenticazione

✅ Request authentication completo
✅ Scope-based authorization (resource:action format)
✅ Context enrichment per requests
✅ Decorator @require_scopes per protezione routes
✅ Permission checking granulare
✅ Token refresh automatico

🎯 Caratteristiche Chiave:

Production-Ready Security: Token encryption, scope validation, audit logging
Automatic Cleanup: Background task per sessioni expired
Graceful Shutdown: Proper cleanup di background tasks
Fine-Grained Permissions: Format resource:action (es. "weather:read", "neo4j:write")
Activity Monitoring: Statistics real-time su sessioni attive

📊 Statistiche Trackate:

Total/active/expired sessions
Average session duration
Total accesses across all sessions
Per-session access counts

SessionManager: Gestione completa sessioni con lifecycle, cleanup automatico, token validation
AuthMiddleware: Authentication/authorization con scope checking e permission control

Geocoding Integrato:

Client per Open-Meteo Geocoding API
Ricerca città italiane con filtro country="IT"
Supporto region/province (admin1/admin2)
Population data inclusa


API Client Open-Meteo:

Forecast fino a 16 giorni
Current weather + Daily + Hourly
WMO Weather codes mapping (95 condizioni)
Retry automatico con backoff


Tools MCP:

search_italian_city(): Cerca città italiane
get_weather_italy(): Forecast completo con geocoding automatico


Resource Endpoint:

weather://italy/current/{city}: Current weather formattato


Schemas Pydantic:

LocationInfo: Dati geografici completi
WeatherCurrent: Condizioni attuali
HourlyForecast: Previsioni orarie
DailyForecast: Previsioni giornaliere con sunrise/sunset
WeatherForecast: Forecast completo

 MCP Host Completo:
Architettura Chiave:

MCPClientWrapper: Wrapper per ogni client MCP con metadata e capability discovery
ReActStep: Rappresenta singolo step nel reasoning chain (Thought → Action → Observation)
MCPHost: Orchestratore principale con loop ReAct

Funzionalità Core:
1. Multi-Client Management

Registrazione dinamica server MCP
Connection pooling via stdio
Capability discovery automatico (tools + resources)

2. ReAct Reasoning Loop
Query → Thought → Action → Observation → Thought → ... → Final Answer
Processo:

Thought: LLM ragiona sul prossimo passo
Action: Seleziona tool appropriato (es. weather_italy.get_weather)
Observation: Esegue tool e raccoglie risultato
Iterate: Fino a max_reasoning_steps o Final Answer

3. Intelligent Tool Selection

Parse query per identificare intent
Inferenza server automatica da tool name
Supporto chaining multi-tool

4. Query Examples:
Simple Query:
python"What's the weather in Rome?"
→ Thought: Need Italian weather
→ Action: weather_italy.get_weather_italy
→ Observation: {forecast data}
→ Final Answer: "Rome has..."
Comparison Query:
python"Compare NYC and Milan weather"
→ Thought: Need USA weather
→ Action: weather_usa.get_weather_usa("NYC")
→ Observation: {NYC data}
→ Thought: Need Italy weather
→ Action: weather_italy.get_weather_italy("Milan")
→ Observation: {Milan data}
→ Final Answer: "NYC is X, Milan is Y..."
Integration Points:
✅ Session Manager: Auth e lifecycle
✅ Context Manager: Conversation history
✅ Ollama Client: LLM reasoning
✅ Auth Middleware: Permission checking
Configuration:
pythonhost.register_server(ServerConfig(
    name="weather_italy",
    command="uv",
    args=["run", "servers/weather_italy/server.py"],
    description="Italian weather forecasts",
))
Key Methods:

process_query(): Main entry point con ReAct loop
_reasoning_step(): Single ReAct iteration
_execute_action(): Tool execution via MCP
_synthesize_answer(): Final answer generation

Funzionalità Chiave:
ReAct Loop:
Query → Thought → Action → Observation → [repeat] → Final Answer
Multi-Tool Chaining:

Query: "Compare NYC and Milan weather"
Step 1: Call weather_usa (NYC)
Step 2: Call weather_italy (Milan)
Synthesize comparison

Intelligent Routing:

Inferenza automatica server da tool name
Pattern matching per location (Italy/USA)
Fallback su primo server disponibile

1. Tool Discovery Automatico
pythonasync def _discover_all_capabilities(self):
    """Discover tools and resources from ALL registered servers."""
    for config in self.server_configs:
        async with stdio_client(...) as (read, write):
            tools_response = await session.list_tools()
            # Populate self.discovered_tools automatically
2. Fixed Async _get_tools_description()
python# PRIMA: tools_desc = self._get_tools_description()  # ❌ Non awaited
# DOPO:
tools_desc = await self._get_tools_description()  # ✅ Properly awaited
3. Tool Matching con Discovered Tools
pythonif step.action not in self.discovered_tools:
    # Error con lista tool disponibili
    step.observation = f"Error: Tool '{step.action}' not found. Available: {list(self.discovered_tools.keys())}"
4. Prevent Tool Invention

Lista EXACT tool names nel prompt
Validazione prima dell'esecuzione
Error message con tool disponibili

5. Registration Semplificata
python# NO MANUAL TOOLS!
host.register_server(ServerConfig(
    name="weather_italy",
    command="python",
    args=["servers/weather_italy/server.py"],
    description="Italian weather forecasts",
    # NO available_tools parameter!
))
🔍 Strutture Dati Chiave:
pythonself.discovered_tools: dict[str, DiscoveredTool]
# {
#   "get_weather_italy": DiscoveredTool(...),
#   "search_italian_city": DiscoveredTool(...),
#   "get_weather_usa": DiscoveredTool(...),
# }

self.discovered_resources: dict[str, DiscoveredResource]
# {
#   "weather://italy/current/roma": DiscoveredResource(...),
# }
```

## 📊 Output Startup Atteso:
```
[info] mcp_host_starting
[debug] discovering_server server=weather_italy
[debug] tool_discovered tool=search_italian_city server=weather_italy
[debug] tool_discovered tool=get_weather_italy server=weather_italy
[info] server_discovery_complete server=weather_italy tools=2
[info] mcp_host_started servers=2 tools_discovered=5 resources_discovered=2
🎯 Vantaggi:

Zero Configuration - Nessuna definizione manuale tool
Always Up-to-Date - Tool list sincronizzata automaticamente
Error Prevention - LLM non può inventare tool inesistenti
Better Prompts - Descrizioni tool estratte dai server

 Schema URI Standardizzato:
Weather Italy:  weather://italy/current/{city}
                → weather://italy/current/roma

Weather USA:    weather://usa/forecast/{location}
                → weather://usa/forecast/seattle,wa
📝 Output Tool Description Atteso:
Available Resources (read-only formatted data):
Use Action: read_resource with Action Input: {"uri": "..."}

weather_italy:
  • URI: weather://italy/current/{city}
    Resource endpoint for current weather
    Example: read_resource with uri="weather://italy/current/roma"

IMPORTANT:
- Tools: Use for dynamic queries (e.g., get_weather_italy)
- Resources: Use for pre-formatted display (e.g., weather://italy/current/roma)
- Resource URIs must match EXACTLY the format shown above

NEO4J GRAPH DB CLIENT/SERVER
1. Schemas (schemas.py) - 9 modelli Pydantic:

Node, Relationship, GraphPattern
QueryResult, SemanticQuery
NodeCreate, RelationshipCreate
PathResult

2. Database Client (db_client.py) - Client Neo4j completo:

Connection pooling async
10 metodi principali:

execute_query() - Raw Cypher
create_node(), create_relationship()
find_nodes(), find_shortest_path()
get_node_neighbors(), execute_aggregation()
get_schema()



3. Query Builder (queries.py) - Semantic query generation:

build_semantic_query() - Natural language → Cypher
Pattern matching per:

"Find all X"
"How many X"
"Path between X and Y"
"Similar to X"


build_recommendation_query() - Collaborative filtering
build_community_detection_query()

4. MCP Server (server.py) - 10 Tools + 1 Resource:
Tools:

execute_cypher_query - Raw Cypher execution
create_node - Create nodes with labels/properties
create_relationship - Create relationships
find_nodes - Search nodes
find_shortest_path - Path finding
semantic_query - Natural language queries
get_node_neighbors - Neighbor discovery
recommend_nodes - Recommendations
get_database_schema - Schema inspection
aggregate_nodes - Aggregations (count, sum, avg, max, min)

Resource:

graph://neo4j/schema - Formatted schema display

🎯 Capacità Semantiche:
L'LLM può fare domande tipo:
"Find all people who work at companies"
"How many products are in the catalog?"
"What's the shortest path between node 5 and node 10?"
"Give me recommendations similar to node 3"
"Show me all relationships in the graph"
Il CypherQueryBuilder converte automaticamente in Cypher!

✅ Lazy Init nel Server
Il server Neo4j gestisce il proprio driver con lazy initialization interna.
Flusso:
User query → MCP Host → Call Neo4j tool → 
  → Neo4jClient._ensure_driver() (lazy init primo uso) → 
  → Execute query
PRO:

✅ Semplice da implementare
✅ Nessun accoppiamento
✅ Lazy init automatico
✅ Ogni server autosufficiente

┌─────────────────────────────────────┐
│          MCP HOST                   │
│                                     │
│  ┌──────────────┐                   │
│  │ Ollama Client│ ← Usato dal host  │
│  └──────────────┘    direttamente   │
│                                     │
│  [NO Neo4j Client qui!]             │
└──────────────┬──────────────────────┘
               │ stdio
               │
     ┌─────────▼──────────┐
     │ Neo4j MCP Server   │
     │                    │
     │ ┌────────────────┐ │
     │ │ Neo4j Client   │ │ ← Usato dal server
     │ └────────────────┘ │
     └────────────────────┘
 main.py - Interfaccia Conversazionale Completa
Caratteristiche:

💬 Chat interattiva in stile conversazione naturale
🎨 UI ricca con Rich library (colori, panel, tabelle, markdown)
🔧 Sistema di comandi (10 comandi built-in)
🔄 Loop conversazionale con gestione errori
📊 Statistiche real-time su sessioni e contesto
🧹 Graceful shutdown con cleanup completo

Comandi Disponibili:
/help      - Mostra aiuto
/quit      - Esci
/clear     - Pulisci schermo
/history   - Cronologia conversazione
/servers   - Lista server registrati
/tools     - Lista tool disponibili
/stats     - Statistiche sessione
/reset     - Reset conversazione

📋 Setup Richiesto:
1. Install Rich:
bashpip install rich>=13.7.0
2. Configure .env:
bashOLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-password
3. Avvia Servizi:
bash# Ollama
ollama serve

# Neo4j (opzionale)
neo4j start
4. Run:
bashpython main.py

-------------------------------------------------------------------
🎨 UI Preview:
╭─────────────────── Welcome ────────────────────╮
│ # MCP Multi-Server Assistant                  │
│                                                │
│ Welcome! I have access to:                    │
│ - 🌤️  Weather data for Italian and US cities  │
│ - 🕸️  Graph database (Neo4j)                  │
│ - 🧠 Semantic reasoning with Ollama           │
╰────────────────────────────────────────────────╯

You: What's the weather in Rome?

⠋ Thinking...
-----------------------------------------------------------------------------
# main.py
host.auto_register_servers()  # ✨ FATTO!
```

## 🔍 Come Funziona:
```
1. Scansiona servers/ directory
   ├── servers/weather_italy/
   ├── servers/weather_usa/
   └── servers/neo4j_graph/

2. Per ogni directory:
   - Cerca server.py o server_*.py
   - Legge server.json (se esiste)
   - Inferisce metadata se manca JSON

3. Valida configurazione:
   - Controlla ENABLE_* flags
   - Verifica config richiesta (NEO4J_PASSWORD, etc.)
   - Skip server con missing requirements

4. Registra automaticamente:
   - Crea ServerConfig
   - Registra nel host
   - Log dettagliato

5. Start discovery automatico tools
```

## 📋 Struttura File (Opzionale):
```
servers/
├── weather_italy/
│   ├── server.py           ✅ Required
│   ├── server.json         ⭐ Optional (metadata)
│   ├── api_client.py
│   └── schemas.py
├── weather_usa/
│   ├── server.py           ✅ Required
│   └── ...
└── neo4j_graph/
    ├── server.py           ✅ Required
    ├── server.json         ⭐ Optional
    └── ...

🎯 Per Aggiungere un Nuovo Server:
bash
# 1. Crea directory
mkdir servers/my_new_server

# 2. Aggiungi server.py
touch servers/my_new_server/server.py

# 3. (Optional) Aggiungi metadata
touch servers/my_new_server/server.json

# 4. Run
python main.py

# ✨ Server auto-discovered!


"""
# Start web interface
python main_web.py

# Open browser to:
http://localhost:7860

# Features:
- Chat tab: Talk with the assistant
- Server Management: Add/remove servers dynamically
- Tools & Resources: See discovered capabilities
- Statistics: System metrics

# Adding a new server:
1. Go to "Server Management" tab
2. Click "Scan Available Servers"
3. Select server from dropdown
4. Click "Add Server"
5. Server is immediately available!

# Or add custom server:
1. Go to "Custom Server" sub-tab
2. Fill in name, description, path
3. Click "Add Custom Server"
"""
