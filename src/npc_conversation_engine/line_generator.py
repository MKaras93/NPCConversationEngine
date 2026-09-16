from abc import ABC, abstractmethod

from npc_conversation_engine.context_builder import BaseContextBuilder
from npc_conversation_engine.defaults import (
    DEFAULT_MAX_HISTORY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)
from npc_conversation_engine.llm_client import BaseLLMClient
from npc_conversation_engine.models.conversation import Conversation
from npc_conversation_engine.prompt_manager import PromptManager


class BaseLineGenerator(ABC):
    """Abstract base class for generating character dialogue lines."""

    @abstractmethod
    async def generate(self, conversation: Conversation) -> str:
        """Generate a line of dialogue for the speaker in the given conversation.

        Args:
            conversation: The conversation context.

        Returns:
            The generated line of dialogue.
        """


class LLMLineGenerator(BaseLineGenerator):
    """Generates dialogue lines using an LLM.

    Builds context from the conversation, renders a system prompt template,
    assembles the message list (system + conversation history), and calls
    the LLM client asynchronously.
    """

    def __init__(
        self,
        context_builder: BaseContextBuilder,
        prompt_manager: PromptManager,
        llm_client: BaseLLMClient,
        prompt_path: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_history: int | None = DEFAULT_MAX_HISTORY,
    ):
        self._context_builder = context_builder
        self._prompt_manager = prompt_manager
        self._llm_client = llm_client
        self._prompt_path = prompt_path
        self._model = model
        self._temperature = temperature
        self._max_history = max_history

    async def generate(self, conversation: Conversation) -> str:
        context = self._context_builder.get_context(conversation)

        system_path = self._prompt_manager.resolve_prompt_path(
            f"{self._prompt_path}/system.j2"
        )
        system_msg = self._prompt_manager.render(system_path, **context)

        history_messages = conversation.to_llm_messages
        if self._max_history is not None:
            history_messages = history_messages[-self._max_history :]

        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
        messages.extend(history_messages)

        response = await self._llm_client.chat(messages, self._model, self._temperature)
        return response.content or ""


class HumanLineGenerator(BaseLineGenerator):
    """Stub for human-controlled character input."""

    async def generate(self, conversation: Conversation) -> str:
        return ""


LINE_GENERATOR_REGISTRY: dict[str, type[BaseLineGenerator]] = {
    "LLMLineGenerator": LLMLineGenerator,
    "HumanLineGenerator": HumanLineGenerator,
}
