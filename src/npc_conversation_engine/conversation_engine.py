import logging

from npc_conversation_engine.context_builder import BaseContextBuilder
from npc_conversation_engine.line_generator import (
    LINE_GENERATOR_REGISTRY,
    BaseLineGenerator,
    HumanLineGenerator,
    LLMLineGenerator,
)
from npc_conversation_engine.llm_client import BaseLLMClient
from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.conversation import Conversation, ConversationMsg
from npc_conversation_engine.models.location import Location
from npc_conversation_engine.prompt_manager import PromptManager
from npc_conversation_engine.tools import BaseTool, EndConversationTool

logger = logging.getLogger(__name__)


class ConversationEngine:
    """Orchestrates multi-turn conversations between characters.

    Manages conversation state, message history, role switching, and
    line generation for NPC and human characters.
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        template_path: str,
        context_builder: BaseContextBuilder,
        llm_client: BaseLLMClient,
        model: str | None = None,
        temperature: float | None = None,
        max_history: int | None = None,
        tools: list[BaseTool] | None = None,
    ):
        self.prompt_manager = prompt_manager
        self.template_path = template_path
        self.context_builder = context_builder
        self.llm_client = llm_client
        self.conversation: Conversation | None = None
        self._generator_cache: dict[tuple, BaseLineGenerator] = {}
        self._model = model
        self._temperature = temperature
        self._max_history = max_history
        self._tools: dict[str, BaseTool] = {"end_conversation": EndConversationTool()}
        for tool in tools or []:
            self._tools[tool.name] = tool

    def initialize_conversation(
        self, location: Location, speaker: Character, listener: Character
    ) -> None:
        """Create a new conversation between two characters at a location.

        Args:
            location: The location where the conversation takes place.
            speaker: The character who speaks first (typically the NPC).
            listener: The character who listens (typically the player).

        Raises:
            RuntimeError: If a conversation is already active.
        """
        if self.conversation is not None:
            raise RuntimeError(
                "A conversation is already active. "
                "Start a new engine or finish the current conversation first."
            )
        self.conversation = Conversation(
            speaker=speaker, listener=listener, location=location
        )
        self._generator_cache.clear()
        logger.info(
            "Conversation initialized: %s and %s at %s",
            speaker.name,
            listener.name,
            location.name,
        )

    def add_message(self, speaker: str | Character, msg: str) -> None:
        """Append a message to the conversation history.

        Args:
            speaker: The speaker's name (string) or Character instance.
            msg: The message content.

        Raises:
            RuntimeError: If no conversation has been initialized.
        """
        self._require_conversation()
        speaker_name = speaker.name if isinstance(speaker, Character) else speaker
        self.conversation.history.append(
            ConversationMsg(speaker=speaker_name, content=msg)
        )

    def add_event(self, event: str) -> None:
        """Append a system event to the conversation history.

        Events use the speaker '<event>' and are mapped to the 'system'
        role in LLM message formatting.

        Args:
            event: The event description.

        Raises:
            RuntimeError: If no conversation has been initialized.
        """
        self._require_conversation()
        self.conversation.history.append(
            ConversationMsg(speaker="<event>", content=event)
        )

    def switch_roles(self) -> None:
        """Swap the speaker and listener roles.

        Raises:
            RuntimeError: If no conversation has been initialized.
        """
        self._require_conversation()
        self.conversation.speaker, self.conversation.listener = (
            self.conversation.listener,
            self.conversation.speaker,
        )

    async def speak_next(self) -> str:
        """Generate the next line for the current speaker and add it to history.

        Resolves the speaker's line generator type from their Character model,
        instantiates (and caches) the appropriate generator, generates a line,
        and appends it to the conversation history.

        Does NOT switch roles — call switch_roles() separately if needed.

        Returns:
            The generated line of dialogue, or "" if the conversation ended.

        Raises:
            RuntimeError: If no conversation has been initialized.
            ValueError: If the speaker's line_generator type is unknown.
        """
        self._require_conversation()
        if self.conversation.ended:
            return ""
        speaker = self.conversation.speaker
        generator = self._get_generator(speaker)
        line = await generator.generate(self.conversation)
        if line:
            self.conversation.history.append(
                ConversationMsg(speaker=speaker.name, content=line)
            )
        logger.info("%s: %s", speaker.name, line)
        return line

    async def run_conversation(self, rounds: int = 2) -> list[str]:
        """Run the conversation for a number of rounds.

        Each round calls speak_next() then switch_roles().

        Args:
            rounds: Number of speak_next + switch_roles cycles.

        Returns:
            List of all generated lines in order.

        Raises:
            RuntimeError: If no conversation has been initialized.
        """
        self._require_conversation()
        lines: list[str] = []
        for _ in range(rounds):
            if self.conversation.ended:
                break
            line = await self.speak_next()
            lines.append(line)
            self.switch_roles()
        return lines

    async def use_tool(self, tool_name: str, arguments: dict | None = None) -> str:
        """Execute a registered tool programmatically.

        Allows the host app to invoke a tool outside of the LLM loop.

        Args:
            tool_name: The name of the tool to execute.
            arguments: Optional arguments dict for the tool.

        Returns:
            The tool's result string.

        Raises:
            RuntimeError: If no conversation has been initialized.
            ValueError: If the tool name is not registered.
        """
        self._require_conversation()
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(
                f"Unknown tool '{tool_name}'. Available: {sorted(self._tools)}"
            )
        return await tool.execute(arguments or {}, self.conversation)

    def _get_generator(self, character: Character) -> BaseLineGenerator:
        """Get or create a cached line generator for a character."""
        gen_type = character.line_generator
        if not gen_type:
            raise ValueError(f"Character '{character.name}' has no line_generator set.")
        tool_names = tuple(character.tools)
        cache_key = (gen_type, tool_names)
        if cache_key not in self._generator_cache:
            tools = self._resolve_tools(tool_names)
            self._generator_cache[cache_key] = self._create_generator(gen_type, tools)
        return self._generator_cache[cache_key]

    def _resolve_tools(self, tool_names: tuple[str, ...]) -> list[BaseTool]:
        """Resolve tool names to BaseTool instances."""
        tools = []
        for name in tool_names:
            if name not in self._tools:
                raise ValueError(
                    f"Unknown tool '{name}'. Available: {sorted(self._tools)}"
                )
            tools.append(self._tools[name])
        return tools

    def _create_generator(
        self, gen_type: str, tools: list[BaseTool] | None = None
    ) -> BaseLineGenerator:
        """Instantiate a line generator by type name."""
        if gen_type == "HumanLineGenerator":
            return HumanLineGenerator()

        if gen_type not in LINE_GENERATOR_REGISTRY:
            raise ValueError(
                f"Unknown line generator type '{gen_type}'. "
                f"Available: {list(LINE_GENERATOR_REGISTRY.keys())}"
            )
        kwargs: dict = {
            "context_builder": self.context_builder,
            "prompt_manager": self.prompt_manager,
            "llm_client": self.llm_client,
            "prompt_path": self.template_path,
        }
        if self._model is not None:
            kwargs["model"] = self._model
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._max_history is not None:
            kwargs["max_history"] = self._max_history
        if tools:
            kwargs["tools"] = tools
        return LLMLineGenerator(**kwargs)

    def _require_conversation(self) -> None:
        """Raise if no conversation has been initialized."""
        if self.conversation is None:
            raise RuntimeError(
                "No active conversation. Call initialize_conversation() first."
            )
