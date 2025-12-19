import asyncio
import json
from host.ollama_client import OllamaClient, OllamaMessage
from shared.logging_config import get_logger

# Configura un logger rapido per il test
logger = get_logger("test_ollama")

async def test_greeting():
    # 1. Inizializza il client (usa il modello llama3.2:1b)
    client = OllamaClient()
    
    # 2. Definisci un System Prompt ReAct "Pulito"
    system_prompt = """You are a helpful assistant. 
    If you need to use a tool, use the format:
    Thought: [reasoning]
    Action: [tool_name]
    Action Input: {"key": "value"}
    
    If you can answer directly, use:
    Final Answer: [your response]
    """

    # 3. Messaggi di test
    messages = [
        OllamaMessage(role="system", content=system_prompt),
        OllamaMessage(role="user", content="Ciao, come va?")
    ]

    print("\n--- TEST: INVIO SALUTO ---")
    print(f"User: {messages[1].content}")
    
    try:
        # 4. Chiamata al client (Temperature 0 e Stop tokens applicati)
        response = await client.chat(
            messages=messages,
            temperature=0.0
        )
        
        print(f"\nRisposta LLM Raw:\n{response.content}")
        
        # 5. Verifica logica
        if "Action:" in response.content:
            print("\n❌ ERRORE: Il modello ha provato a usare un tool per un saluto!")
        elif "Final Answer:" in response.content or "Ciao" in response.content:
            print("\n✅ SUCCESSO: Il modello ha risposto correttamente.")
        else:
            print("\n⚠️ ATTENZIONE: Risposta ambigua, controlla il formato.")

    except Exception as e:
        print(f"\n❌ CRASH: {str(e)}")

async def test_tool_failure():
    client = OllamaClient()
    prompt="""
     You are a reasoning engine that solves queries by looping through Thought, Action, and Action Input. You have access to a dynamic set of Tools and Resources provided in the user prompt.
     **CRITICAL: STOP after writing the Action Input. Do NOT write an "Observation". The system will provide the Observation to you in the next turn.**
     CRITICAL: If you receive an error from a tool, DO NOT invent tools. Use "Final Answer:" to inform the user the service is unavailable.
    """
    # Simuliamo la history dove il modello ha chiesto il meteo 
    # e il sistema ha risposto con un ERRORE
    messages = [
        OllamaMessage(role="system", content=prompt),
        OllamaMessage(role="user", content="Che tempo fa a Roma?"),
        OllamaMessage(role="assistant", content="Thought: I need to check the weather.\nAction: get_weather\nAction Input: {'location': 'Rome'}"),
        # Simuliamo il fallimento del server MCP
        OllamaMessage(role="user", content="Observation: Error: Weather service is offline (500).")
    ]

    print("\n--- TEST: FALLIMENTO TOOL ---")
    
    response = await client.chat(messages=messages, temperature=0.1)
    print(f"Risposta LLM post-errore:\n{response.content}")

    if "Final Answer:" in response.content:
        print("\n✅ SUCCESSO: Il modello ha gestito l'errore e ha informato l'utente.")
    else:
        print("\n⚠️ ATTENZIONE: Il modello potrebbe essere entrato in loop o aver allucinato.")

if __name__ == "__main__":
    asyncio.run(test_greeting())
    asyncio.run(test_tool_failure())