# MCP Dynamic Host - Proof of Concept (PoC)

Questo progetto è un **Host MCP (Model Context Protocol)** flessibile e agnostico, progettato come Proof of Concept per l'integrazione dinamica di tool e risorse. A differenza degli host statici, questo sistema è preparato per accettare e connettersi a qualsiasi MCP Client o Server di cui l'azienda necessiti, orchestrando la comunicazione tra LLM locali e servizi esterni in tempo reale.

## Visione del Progetto
L'obiettivo è creare un ecosistema in cui l'intelligenza artificiale non sia limitata dai dati di addestramento, ma possa estendere le proprie capacità collegandosi dinamicamente a:
- **Sistemi legacy** tramite server MCP dedicati.
- **Database a grafo** per il recupero di relazioni complesse.
- **API esterne** (Meteo, CRM, Slack, ecc.) configurate a runtime.

---

## 🛠 Prerequisiti

Prima di iniziare, assicuratevi di avere installato:

1. **Python 3.10+**
2. **Ollama**: Necessario per il serving dei modelli locali (es. Llama 3.2, Granite, ecc.).
3. **Neo4j**: Un'istanza attiva (locale o cloud) per la gestione del contesto a grafo.
4. **UV (Consigliato)**: Per una gestione rapida delle dipendenze e degli ambienti virtuali.

---

## 📦 Installazione e Setup

1. **Clona il repository:**
   ```bash
   git clone [https://github.com/tuo-username/mcp-dynamic-host.git](https://github.com/tuo-username/mcp-dynamic-host.git)
   cd mcp-dynamic-host

2. **Crea e attiva l'ambiente virtuale:**
    ```bash
    uv venv
    source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
    uv sync

3. **Configurazione ambiente:** *Copia il file di esempio e modificalo con le tue credenziali:*
    ```bash
    cp .env.example .env

4. **Popola il file *.env*: Assicurati di compilare i seguenti campi:**
    # LLM Config
        OLLAMA_BASE_URL=http://localhost:11434
        MODEL_NAME=granite3-dense:2b  # o llama3.2

    # Neo4j Config
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=la_tua_password

## 🧠 Guida al Prompt Engineering
Se desideri modificare il comportamento del motore di ragionamento ReAct o ottimizzare la risposta dei modelli Small (1B/2B), segui queste linee guida:

Dove intervenire
La logica dei prompt è centralizzata in mcp_host.py (o nel modulo dedicato ai prompt).

Metodo: build_react_system_prompt()

Obiettivo: Definire i tag Thought:, Action:, Action Input: e Final Answer:.

Suggerimenti per il Prompting (Modelli Small)
I modelli piccoli (necessari visto l'uso di Ollama per il serving locale dei modelli) tendono a "saltare" passaggi, o a essere *overeager* nel tentativo di dare una risposta, a discapito delle istruzioni nel system prompt; soprattutto se esso è troppo verboso e articolato. Se riscontri problemi:

- **Sii conciso**: Usa istruzioni brevi e imperative.

- **Esempi Few-Shot**: Inserisci nel prompt esempi reali di come il modello deve scrivere l'azione.

- **Guardrails**: Specifica esplicitamente di NON scrivere "Final Answer" finché un'azione non è stata completata.

- **Formato JSON**: Ricorda al modello che l'Action Input deve essere solo JSON valido, senza testo aggiuntivo.

## 🛠 Sviluppo di nuovi Tool
Per aggiungere un nuovo server MCP, basta aggiungerlo alla configurazione di avvio. L'host rileverà automaticamente i nuovi tool tramite il protocollo di list_tools e li renderà disponibili all'LLM senza modifiche al codice core. Si possono caricare direttamente a runtime tramite l'interfaccia di Server Management messa a disposizione.