import logging
from abc import ABC, abstractmethod

from npc_conversation_engine.defaults import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM chat clients."""

    @abstractmethod
    async def chat(self, messages: list[dict], model: str, temperature: float) -> str:
        """Send messages to an LLM and return the response content.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model identifier string.
            temperature: Sampling temperature.

        Returns:
            The LLM's response content as a string.
        """


class OpenAIChat(BaseLLMClient):
    """LLM client using the OpenAI-compatible API (works with OpenRouter)."""

    def __init__(self, api_key: str, base_url: str, timeout: float = DEFAULT_TIMEOUT):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._timeout = timeout

    async def chat(self, messages: list[dict], model: str, temperature: float) -> str:
        logger.info("--- LLM Request ---")
        logger.info("Model: %s | Temperature: %s", model, temperature)
        for msg in messages:
            logger.debug("[%s]: %s", msg["role"], msg["content"])
        logger.info("--------------------")

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=self._timeout,
        )

        content = response.choices[0].message.content
        logger.info("--- LLM Response ---")
        logger.debug("%s", content)
        logger.info("---------------------")
        return content
