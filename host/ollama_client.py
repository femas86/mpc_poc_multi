"""Ollama client for LLM interaction with full streaming and function calling support."""

from ollama import AsyncClient as oaClient, ChatResponse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, AsyncIterator, Optional
import json

import asyncio

from pydantic import BaseModel, Field

from config.settings import get_settings
from shared.logging_config import get_logger
from shared.utils import retry_async
from host.context_manager import ContextManager, Message


logger = get_logger(__name__)

class ToolDefinition(BaseModel):
    """Tool definition for function calling."""

    type: str = Field(default="function", description="Tool type")
    function: dict[str, Any] = Field(..., description="Function specification")

class OllamaResponse(BaseModel):
    """Structured response from Ollama."""

    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model used")
    tool_calls: Optional[list[ToolDefinition]] = Field(default=None, description="Tool call details")
    done: bool = Field(..., description="Whether generation is complete")
    total_duration: Optional[int] = Field(default=None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(default=None, description="Load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(default=None, description="Number of tokens in prompt")
    eval_count: Optional[int] = Field(default=None, description="Number of tokens generated")
    eval_duration: Optional[int] = Field(default=None, description="Evaluation duration in nanoseconds")

class OllamaClient:
    """Client asincrono per interagire con Ollama locale
        Supports:
        - Standard chat completion
        - Streaming responses
        - Function calling / tool use
        - Context management
        - Automatic retries
    
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Args:
            host: Ollama server URL (defaults to settings)
            model: Model name (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)

        """
        settings = get_settings()
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout
        
        self.client = oaClient(host=self.host, timeout=self.timeout)
        
        logger.info(
            "ollama_client_initialized",
            host=self.host,
            model=self.model,
            timeout=self.timeout,
        )
   
    @retry_async(max_attempts=3, delay=1.0, backoff=2.0)
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        stream: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> OllamaResponse | AsyncIterator[str]:
        """
        Send chat request to Ollama.

        Args:
            messages: List of conversation messages
            tools: Optional tool definitions for function calling
            stream: Whether to stream the response
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            OllamaResponse or async iterator of response chunks

        Raises:
            RuntimeError: If Ollama request fails
        """
        
        try:
            # Convert messages to dict format
            message_dicts = []
            for msg in messages:
                if hasattr(msg, "model_dump"):
                    # È un oggetto Pydantic (OllamaMessage)
                    message_dicts.append(msg.model_dump(include={'role', 'content', 'name', 'tool_calls'}, exclude_none=True))
                elif isinstance(msg, dict):
                    # È già un dizionario
                    message_dicts.append(msg)
                else:
                    # Fallback estremo per altri tipi di oggetti
                    message_dicts.append({"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", ""), 
                                         "name": getattr(msg, "name", None), 'tool_calls': getattr(msg, "tool_calls", None)})
            
            tool_dict= None
            if tools:
                tool_dict = [t.model_dump() if isinstance(t, ToolDefinition) else t for t in tools]

            # Prepare request options
            options: dict[str, Any] = {
                "temperature": temperature,
                "num_predict": max_tokens or 512,
                "num_ctx": 4096,            # Focalizza l'attenzione su una finestra gestibile
            }

            logger.debug(
                "ollama_chat_request",
                model=self.model,
                message_count=len(messages),
                has_tools=bool(tools),
                stream=stream,
            )

            # Handle streaming response
            if stream:
                return self._stream_chat(message_dicts, tools, options)

            # Standard request
            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": message_dicts,
                "stream": False,
                "options": options,
            }
            
            if tool_dict:
                request_params["tools"] = tool_dict

            
            if tool_dict:
                logger.info("DEBUG_TOOLS_SENT", count=len(tool_dict), names=[t['function']['name'] for t in tool_dict])
            else:
                logger.warning("DEBUG_NO_TOOLS_SENT_TO_OLLAMA")

            response: ChatResponse = await self.client.chat(**request_params)
            
            resp_msg = getattr(response, "message", response.get("message", {}))
            content = getattr(resp_msg, "content", resp_msg.get("content", ""))
            t_calls = getattr(resp_msg, "tool_calls", resp_msg.get("tool_calls", None))
            
            final_tool_calls = None
            if t_calls:
                final_tool_calls = [tc if isinstance(tc, dict) else tc.model_dump() for tc in t_calls]
            
            ollama_response = OllamaResponse(
                content=content.strip(),
                tool_calls=final_tool_calls,
                model=self.model,
                done=True,
                total_duration=getattr(response, "total_duration", None),
                load_duration=getattr(response, "load_duration", None),
                prompt_eval_count=getattr(response, "prompt_eval_count", None),
                eval_count=getattr(response, "eval_count", None,),
                eval_duration=getattr(response, "eval_duration", None),
            )                    
                        
            logger.info(
                "ollama_chat_completed",
                model=self.model,
                content_length=len(content),
                prompt_tokens=ollama_response.prompt_eval_count,
                completion_tokens=ollama_response.eval_count,
            )

            return ollama_response

        except Exception as e:
            logger.error("ollama_chat_error", error=str(e), model=self.model)
            raise RuntimeError(f"Ollama chat failed: {e}") from e

    async def _stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[ToolDefinition]],
        options: dict[str, Any],
    ) -> AsyncIterator[str]:
        """
        Stream chat responses from Ollama.

        Args:
            messages: Message dictionaries
            tools: Optional tool definitions
            options: Request options

        Yields:
            Response content chunks
        """
        try:
            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": options,
            }
            
            if tools:
                request_params["tools"] = [tool.model_dump() for tool in tools]

            async for chunk in await self.client.chat(**request_params):
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    content = chunk.message.content
                    if content:
                        yield content
                elif isinstance(chunk, dict) and "message" in chunk:
                    content = chunk["message"].get("content")
                    if content:
                        yield content

        except Exception as e:
            logger.error("ollama_stream_error", error=str(e))
            raise RuntimeError(f"Ollama streaming failed: {e}") from e

    async def check_health(self) -> bool:
        """
        Check if Ollama server is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to list models as a health check
            await asyncio.wait_for(self.client.list(), timeout=5.0)
            logger.debug("ollama_health_check_passed")
            return True
        except Exception as e:
            logger.warning("ollama_health_check_failed", error=str(e))
            return False

    """2 metodi della versione vecchia da stabilire se servono o no"""

    # def _build_prompt(
    #         self,
    #         query: str,
    #         context: Dict,
    #         data: Dict,
    #         history: List[Dict]
    # ) -> str:
    #     """Costruisce un prompt strutturato"""
    #     prompt_parts = [
    #         "Sei un assistente che risponde a domande usando dati da varie fonti.",
    #         f"\nContesto utente: {json.dumps(context, indent=2)}",
    #         f"\nDati raccolti: {json.dumps(data, indent=2)}"
    #     ]

    #     if history:
    #         prompt_parts.append("\nStorico conversazione: ")
    #         for item in history[-4:]:   #Ultimi 4 scambi
    #             prompt_parts.append(f"Q: {item['query']}")
    #             prompt_parts.append(f"A: {item['response']}")

    #     prompt_parts.append(f"\nDomanda corrente: {query}")
    #     prompt_parts.append(f"\nRisposta:")

    #     return "\n".join(prompt_parts)
    
    # async def get_routing(self, routing_prompt: str) -> List[str]:
    #     """Determina routing usando Ollama"""
        
    #     response = await self.client.chat(
    #         model = self.model,
    #         messages=routing_prompt,
    #         think= 'high',
    #         format= 'json',
    #         stream= False
    #     )
    #     routing = json.loads(response.get('response', '[]'))
    #     return routing
    
    async def close(self) -> None:
        """Close the Ollama client connection."""
        # AsyncClient doesn't require explicit cleanup in current version
        logger.debug("ollama_client_closed")

    # """Vecchia versione perché non mi fido del cleanup automatico"""    
    # async def close_old(self):
    #     await self.client.aclose()