import json
import logging
from abc import ABC, abstractmethod

from pydantic import ValidationError

from npc_conversation_engine.context_builder import BaseContextBuilder
from npc_conversation_engine.defaults import (
    DEFAULT_MAX_HISTORY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_TOOL_ITERATIONS,
)
from npc_conversation_engine.llm_client import BaseLLMClient, ToolCall
from npc_conversation_engine.models.conversation import Conversation
from npc_conversation_engine.prompt_manager import PromptManager
from npc_conversation_engine.tools import BaseTool

logger = logging.getLogger(__name__)


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
    the LLM client asynchronously.  Supports an auto-execute tool loop when
    tools are provided.
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
        tools: list[BaseTool] | None = None,
    ):
        self._context_builder = context_builder
        self._prompt_manager = prompt_manager
        self._llm_client = llm_client
        self._prompt_path = prompt_path
        self._model = model
        self._temperature = temperature
        self._max_history = max_history
        self._tools = tools or []

    async def generate(self, conversation: Conversation) -> str:
        context = self._context_builder.get_context(conversation)

        system_path = self._prompt_manager.resolve_prompt_path(
            f"{self._prompt_path}/system.j2"
        )
        system_msg = self._prompt_manager.render(system_path, **context)

        history_messages = conversation.to_llm_messages
        if self._max_history is not None:
            history_messages = history_messages[-self._max_history :]

        messages: list[dict] = [{"role": "system", "content": system_msg}]
        messages.extend(history_messages)

        tool_specs = [t.to_openai_spec() for t in self._tools]

        for _ in range(MAX_TOOL_ITERATIONS):
            response = await self._llm_client.chat(
                messages, self._model, self._temperature, tools=tool_specs or None
            )
            if not response.tool_calls:
                return response.content or ""

            fed_back_calls: list[ToolCall] = []
            results: dict[str, str] = {}
            for tool_call in response.tool_calls:
                tool = self._resolve_tool(tool_call.name)
                if tool is None:
                    results[tool_call.id] = f"Error: unknown tool '{tool_call.name}'."
                    fed_back_calls.append(tool_call)
                    continue

                try:
                    arguments = tool_call.parsed_arguments
                    if tool.arguments_model is not None:
                        arguments = tool.arguments_model.model_validate(arguments)
                except (json.JSONDecodeError, ValidationError) as exc:
                    results[tool_call.id] = (
                        f"Invalid arguments for tool '{tool_call.name}': {exc}"
                    )
                    fed_back_calls.append(tool_call)
                    continue

                try:
                    result = await tool.execute(arguments, conversation)
                except Exception:
                    logger.exception(
                        "Tool '%s' failed during execution", tool_call.name
                    )
                    raise

                if conversation.ended:
                    return ""
                if tool.returns_to_llm:
                    results[tool_call.id] = result
                    fed_back_calls.append(tool_call)

            if not fed_back_calls:
                return ""  # all tools were fire-and-forget: end the turn

            messages.append(self._assistant_tool_call_message(fed_back_calls))
            for tool_call in fed_back_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": results[tool_call.id],
                    }
                )
        return ""

    @staticmethod
    def _assistant_tool_call_message(tool_calls: list[ToolCall]) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in tool_calls
            ],
        }

    def _resolve_tool(self, name: str) -> BaseTool | None:
        return next((t for t in self._tools if t.name == name), None)


class HumanLineGenerator(BaseLineGenerator):
    """Stub for human-controlled character input."""

    async def generate(self, conversation: Conversation) -> str:
        return ""


LINE_GENERATOR_REGISTRY: dict[str, type[BaseLineGenerator]] = {
    "LLMLineGenerator": LLMLineGenerator,
    "HumanLineGenerator": HumanLineGenerator,
}
