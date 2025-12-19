"""Ollama client for LLM interaction with full streaming and function calling support."""

from ollama import AsyncClient as oaClient, ChatResponse, chat as Ochat

from typing import Dict, List, Any, AsyncIterator, Optional
import json

import asyncio

from pydantic import BaseModel, Field

from config.settings import get_settings
from shared.logging_config import get_logger
from shared.utils import retry_async

logger = get_logger(__name__)

class OllamaMessage(BaseModel):
    """Message structure for Ollama chat."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    images: Optional[list[str]] = Field(default=None, description="Optional image data")

class OllamaResponse(BaseModel):
    """Structured response from Ollama."""

    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model used")
    done: bool = Field(..., description="Whether generation is complete")
    total_duration: Optional[int] = Field(default=None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(default=None, description="Load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(default=None, description="Number of tokens in prompt")
    eval_count: Optional[int] = Field(default=None, description="Number of tokens generated")
    eval_duration: Optional[int] = Field(default=None, description="Evaluation duration in nanoseconds")

class ToolDefinition(BaseModel):
    """Tool definition for function calling."""

    type: str = Field(default="function", description="Tool type")
    function: dict[str, Any] = Field(..., description="Function specification")


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
        messages: list[OllamaMessage],
        tools: Optional[list[ToolDefinition]] = None,
        stream: bool = False,
        temperature: float = 0.2,
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
        
        # 1. TRIMMING: Mantieni il System Prompt + solo gli ultimi 3 messaggi
        # Questo evita che il modello "impazzisca" leggendo troppi errori passati
        if len(messages) > 4:
            system_msg = messages[0] # Il primo è sempre il System Prompt
            last_messages = messages[-3:] # Gli ultimi 3 scambi
            messages = [system_msg] + last_messages
        try:
            # Convert messages to dict format
            message_dicts = [msg.model_dump(exclude_none=True) for msg in messages]
            
            # Prepare request options
            options: dict[str, Any] = {
                "temperature": temperature,
                "num_predict": max_tokens or 512,
                "stop": [
                    "Observation:",         # Fondamentale: ferma l'LLM dopo l'azione
                    "User:",                # Evita che l'LLM simuli l'utente
                    "Thought: Observation:", # Cattura loop comuni
                    "\n\n",
                    "\n"
                ],
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
            
            if tools:
                request_params["tools"] = [tool.model_dump() for tool in tools]

            response: ChatResponse = await self.client.chat(**request_params)
            
            # Extract response content
            content = ""
            if hasattr(response, "message") and hasattr(response.message, "content"):
                content = response.message.content
            elif isinstance(response, dict) and "message" in response:
                content = response["message"].get("content", "")

            content = content.strip()
            if "Observation:" in content:
                content = content.split("Observation:")[0].strip()
            if content.count("Thought:") > 1:
                content = "Thought:" + content.split("Thought:")[1].split("Thought:")[0]

            ollama_response = OllamaResponse(
                content=content,
                model=self.model,
                done=True,
                total_duration=getattr(response, "total_duration", None),
                load_duration=getattr(response, "load_duration", None),
                prompt_eval_count=getattr(response, "prompt_eval_count", None),
                eval_count=getattr(response, "eval_count", None),
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

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            response = await self.client.embeddings(model=self.model, prompt=text)
            
            if hasattr(response, "embedding"):
                return response.embedding
            elif isinstance(response, dict) and "embedding" in response:
                return response["embedding"]
            else:
                raise RuntimeError("Invalid embedding response format")

        except Exception as e:
            logger.error("ollama_embedding_error", error=str(e))
            raise RuntimeError(f"Embedding generation failed: {e}") from e

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

    async def list_models(self) -> list[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names
        """
        try:
            response = await self.client.list()
            if hasattr(response, "models"):
                return [model.model for model in response.models]
            elif isinstance(response, dict) and "models" in response:
                return [model["model"] for model in response["models"]]
            return []
        except Exception as e:
            logger.error("ollama_list_models_error", error=str(e))
            return []

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