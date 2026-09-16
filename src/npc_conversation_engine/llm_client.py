import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from npc_conversation_engine.defaults import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A single tool/function call requested by the LLM."""

    id: str
    name: str
    arguments: str  # raw JSON string exactly as returned by the model

    @property
    def parsed_arguments(self) -> dict:
        return json.loads(self.arguments or "{}")


@dataclass
class LLMResponse:
    """Structured response from an LLM chat call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class BaseLLMClient(ABC):
    """Abstract base class for LLM chat clients."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send messages to an LLM and return the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model identifier string.
            temperature: Sampling temperature.
            tools: Optional list of OpenAI-format tool specs.

        Returns:
            An LLMResponse with content and/or tool calls.
        """


class OpenAIChat(BaseLLMClient):
    """LLM client using the OpenAI-compatible API (works with OpenRouter)."""

    def __init__(self, api_key: str, base_url: str, timeout: float = DEFAULT_TIMEOUT):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._timeout = timeout

    async def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        logger.info("--- LLM Request ---")
        logger.info("Model: %s | Temperature: %s", model, temperature)
        for msg in messages:
            logger.debug("[%s]: %s", msg["role"], msg["content"])
        logger.info("--------------------")

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        content = message.content
        raw = getattr(message, "tool_calls", None) or []
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in raw
        ]

        logger.info("--- LLM Response ---")
        logger.debug("%s", content)
        logger.info("---------------------")
        return LLMResponse(content=content, tool_calls=tool_calls)
