import os

import pytest

from npc_conversation_engine.context_builder import ExampleContextBuilder
from npc_conversation_engine.line_generator import (
    LINE_GENERATOR_REGISTRY,
    HumanLineGenerator,
    LLMLineGenerator,
)
from npc_conversation_engine.llm_client import BaseLLMClient, LLMResponse, ToolCall
from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.conversation import Conversation, ConversationMsg
from npc_conversation_engine.models.location import Location
from npc_conversation_engine.prompt_manager import PromptManager
from npc_conversation_engine.storage import Storage
from npc_conversation_engine.tools import BaseTool

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "prompts")


class FakeLLMClient(BaseLLMClient):
    """A fake LLM client that returns pre-configured responses in sequence."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = list(responses or [LLMResponse(content="Hello there!")])
        self._call_count = 0
        self.last_messages: list[dict] | None = None
        self.last_model: str | None = None
        self.last_temperature: float | None = None
        self.last_tools: list[dict] | None = None
        # History of all calls for assertions
        self.all_messages: list[list[dict]] = []
        self.all_tools: list[list[dict] | None] = []

    async def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self.last_messages = messages
        self.last_model = model
        self.last_temperature = temperature
        self.last_tools = tools
        self.all_messages.append(messages)
        self.all_tools.append(tools)
        self._call_count += 1
        return self._responses[idx]

    @property
    def call_count(self) -> int:
        return self._call_count


@pytest.fixture
def gandalf():
    return Character(
        name="Gandalf",
        gender="male",
        age=2000,
        appearance="tall wizard",
        bio="a wizard",
        type="npc",
        line_generator="LLMLineGenerator",
    )


@pytest.fixture
def frodo():
    return Character(
        name="Frodo",
        gender="male",
        age=50,
        appearance="short hobbit",
        bio="a hobbit",
        type="player",
    )


@pytest.fixture
def location():
    return Location(name="The Shire")


@pytest.fixture
def conversation(gandalf, frodo, location):
    return Conversation(speaker=gandalf, listener=frodo, location=location)


@pytest.fixture
def prompt_manager():
    return PromptManager(language="en", prompts_dir=FIXTURES_DIR)


@pytest.fixture
def context_builder():
    return ExampleContextBuilder()


@pytest.fixture
def llm_client():
    return FakeLLMClient(responses=[LLMResponse(content="A wizard is never late.")])


@pytest.fixture
def llm_generator(context_builder, prompt_manager, llm_client):
    return LLMLineGenerator(
        context_builder=context_builder,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        prompt_path="test_dialogue",
        model="test-model",
        temperature=0.5,
    )


class TestLLMLineGeneratorGenerate:
    @pytest.mark.asyncio
    async def test_returns_llm_response(self, llm_generator, conversation):
        result = await llm_generator.generate(conversation)
        assert result == "A wizard is never late."

    @pytest.mark.asyncio
    async def test_system_message_is_first(self, llm_generator, conversation):
        await llm_generator.generate(conversation)
        messages = llm_generator._llm_client.last_messages
        assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_system_message_contains_context(self, llm_generator, conversation):
        await llm_generator.generate(conversation)
        system_content = llm_generator._llm_client.last_messages[0]["content"]
        assert "Gandalf" in system_content
        assert "Frodo" in system_content
        assert "The Shire" in system_content

    @pytest.mark.asyncio
    async def test_history_appended_after_system(self, gandalf, frodo, location):
        conversation = Conversation(speaker=gandalf, listener=frodo, location=location)
        conversation.history = [
            ConversationMsg(speaker="Frodo", content="What is the ring?"),
            ConversationMsg(speaker="Gandalf", content="It is the One Ring."),
        ]

        client = FakeLLMClient()
        gen = LLMLineGenerator(
            context_builder=ExampleContextBuilder(),
            prompt_manager=PromptManager(language="en", prompts_dir=FIXTURES_DIR),
            llm_client=client,
            prompt_path="test_dialogue",
        )
        await gen.generate(conversation)

        messages = client.last_messages
        assert messages[0] == {"role": "system", "content": messages[0]["content"]}
        assert messages[1] == {"role": "user", "content": "What is the ring?"}
        assert messages[2] == {"role": "assistant", "content": "It is the One Ring."}
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_passes_model_and_temperature(self, llm_generator, conversation):
        await llm_generator.generate(conversation)
        assert llm_generator._llm_client.last_model == "test-model"
        assert llm_generator._llm_client.last_temperature == 0.5

    @pytest.mark.asyncio
    async def test_empty_history(self, llm_generator, conversation):
        await llm_generator.generate(conversation)
        messages = llm_generator._llm_client.last_messages
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


class TestLLMLineGeneratorMaxHistory:
    @pytest.mark.asyncio
    async def test_max_history_truncates_messages(self, gandalf, frodo, location):
        conversation = Conversation(speaker=gandalf, listener=frodo, location=location)
        conversation.history = [
            ConversationMsg(speaker="Gandalf", content="First"),
            ConversationMsg(speaker="Frodo", content="Second"),
            ConversationMsg(speaker="Gandalf", content="Third"),
            ConversationMsg(speaker="Frodo", content="Fourth"),
        ]
        client = FakeLLMClient()
        gen = LLMLineGenerator(
            context_builder=ExampleContextBuilder(),
            prompt_manager=PromptManager(language="en", prompts_dir=FIXTURES_DIR),
            llm_client=client,
            prompt_path="test_dialogue",
            max_history=2,
        )
        await gen.generate(conversation)

        messages = client.last_messages
        assert len(messages) == 3  # system + 2 history
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Third"
        assert messages[2]["content"] == "Fourth"

    @pytest.mark.asyncio
    async def test_max_history_none_sends_all(self, gandalf, frodo, location):
        conversation = Conversation(speaker=gandalf, listener=frodo, location=location)
        conversation.history = [
            ConversationMsg(speaker="Gandalf", content="First"),
            ConversationMsg(speaker="Frodo", content="Second"),
            ConversationMsg(speaker="Gandalf", content="Third"),
        ]
        client = FakeLLMClient()
        gen = LLMLineGenerator(
            context_builder=ExampleContextBuilder(),
            prompt_manager=PromptManager(language="en", prompts_dir=FIXTURES_DIR),
            llm_client=client,
            prompt_path="test_dialogue",
            max_history=None,
        )
        await gen.generate(conversation)

        messages = client.last_messages
        assert len(messages) == 4  # system + 3 history

    @pytest.mark.asyncio
    async def test_max_history_preserves_conversation_history(
        self, gandalf, frodo, location
    ):
        conversation = Conversation(speaker=gandalf, listener=frodo, location=location)
        conversation.history = [
            ConversationMsg(speaker="Gandalf", content="First"),
            ConversationMsg(speaker="Frodo", content="Second"),
            ConversationMsg(speaker="Gandalf", content="Third"),
        ]
        client = FakeLLMClient()
        gen = LLMLineGenerator(
            context_builder=ExampleContextBuilder(),
            prompt_manager=PromptManager(language="en", prompts_dir=FIXTURES_DIR),
            llm_client=client,
            prompt_path="test_dialogue",
            max_history=1,
        )
        await gen.generate(conversation)

        assert len(conversation.history) == 3  # unchanged


class TestHumanLineGenerator:
    @pytest.mark.asyncio
    async def test_returns_empty_string(self, conversation):
        gen = HumanLineGenerator()
        result = await gen.generate(conversation)
        assert result == ""


class TestLineGeneratorRegistry:
    def test_contains_llm_generator(self):
        assert LINE_GENERATOR_REGISTRY["LLMLineGenerator"] is LLMLineGenerator

    def test_contains_human_generator(self):
        assert LINE_GENERATOR_REGISTRY["HumanLineGenerator"] is HumanLineGenerator


class TestCharacterLineGeneratorField:
    def test_default_empty(self):
        char = Character(
            name="Bilbo",
            gender="male",
            age=111,
            appearance="old hobbit",
            bio="adventurer",
            type="npc",
        )
        assert char.line_generator == ""

    def test_stores_class_name(self, gandalf):
        assert gandalf.line_generator == "LLMLineGenerator"

    def test_round_trip_through_storage(self, gandalf, tmp_path):
        storage = Storage(str(tmp_path / "characters"))
        storage.save(gandalf, "gandalf")
        loaded = storage.load("gandalf", Character)
        assert loaded.line_generator == "LLMLineGenerator"
        assert loaded == gandalf


# --- Tool-loop tests ---


class _GreetTool(BaseTool):
    """A simple tool that returns a greeting. Used for testing the tool loop."""

    name = "greet"
    description = "Greet someone."
    returns_to_llm = True

    async def execute(self, arguments, conversation):
        return f"Hello, {arguments.get('name', 'stranger')}!"


class _FireAndForgetTool(BaseTool):
    """A tool with returns_to_llm=False for fire-and-forget testing."""

    name = "ping"
    description = "Send a ping."
    returns_to_llm = False

    async def execute(self, arguments, conversation):
        return "pong"


class _ExplodingTool(BaseTool):
    """A tool whose execute() raises an error."""

    name = "explode"
    description = "Explode."
    returns_to_llm = True

    async def execute(self, arguments, conversation):
        raise RuntimeError("boom")


class _EndConvTool(BaseTool):
    """A tool that ends the conversation."""

    name = "end_conversation"
    description = "End the current conversation."
    returns_to_llm = False

    async def execute(self, arguments, conversation):
        conversation.end()
        return "ended"


class _ValidatedTool(BaseTool):
    """A tool that uses arguments_model for validation."""

    from pydantic import BaseModel as _BaseModel

    class _Args(_BaseModel):
        name: str

    name = "validated_greet"
    description = "Greet with validated args."
    arguments_model = _Args
    returns_to_llm = True

    async def execute(self, arguments, conversation):
        return f"Validated hello, {arguments.name}!"


class TestToolLoop:
    @pytest.mark.asyncio
    async def test_no_tools_sends_tools_none(
        self, context_builder, prompt_manager, conversation
    ):
        """Generator with no tools passes tools=None to the client."""
        client = FakeLLMClient(responses=[LLMResponse(content="Hi")])
        gen = LLMLineGenerator(
            context_builder=context_builder,
            prompt_manager=prompt_manager,
            llm_client=client,
            prompt_path="test_dialogue",
        )
        await gen.generate(conversation)
        assert client.all_tools[0] is None

    @pytest.mark.asyncio
    async def test_tools_advertised(
        self, context_builder, prompt_manager, conversation
    ):
        """Generator tools are all advertised: the client call receives tools=[spec, ...]."""
        tool = _GreetTool()
        client = FakeLLMClient(responses=[LLMResponse(content="Hi")])
        gen = LLMLineGenerator(
            context_builder=context_builder,
            prompt_manager=prompt_manager,
            llm_client=client,
            prompt_path="test_dialogue",
            tools=[tool],
        )
        await gen.generate(conversation)
        assert client.all_tools[0] is not None
        assert len(client.all_tools[0]) == 1
        assert client.all_tools[0][0]["function"]["name"] == "greet"

    @pytest.mark.asyncio
    async def test_tool_call_loop(self, context_builder, prompt_manager, conversation):
        """A tool call is executed and the result is fed back to the LLM."""
        tool = _GreetTool()
        tool_call = ToolCall(id="tc1", name="greet", arguments='{"name": "Frodo"}')
        client = FakeLLMClient(
            responses=[
                LLMResponse(content=None, tool_calls=[tool_call]),
                LLMResponse(content="The wizard greets you."),
            ]
        )
        gen = LLMLineGenerator(
            context_builder=context_builder,
            prompt_manager=prompt_manager,
            llm_client=client,
            prompt_path="test_dialogue",
            tools=[tool],
        )
        result = await gen.generate(conversation)

        assert result == "The wizard greets you."
        assert client.call_count == 2
        # Second call messages should include the assistant tool_calls message
        second_call_messages = client.all_messages[1]
        assistant_tc_msg = next(
            m
            for m in second_call_messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        assert assistant_tc_msg["tool_calls"][0]["id"] == "tc1"
        # And a tool result message
        tool_result_msg = next(
            m for m in second_call_messages if m.get("role") == "tool"
        )
        assert tool_result_msg["tool_call_id"] == "tc1"
        assert "Hello, Frodo!" in tool_result_msg["content"]

    @pytest.mark.asyncio
    async def test_invalid_arguments_fed_back(
        self, context_builder, prompt_manager, conversation
    ):
        """Invalid arguments from the LLM are fed back as a sanitized error message."""
        tool = _ValidatedTool()
        tool_call = ToolCall(
            id="tc1", name="validated_greet", arguments='{"wrong": 123}'
        )
        client = FakeLLMClient(
            responses=[
                LLMResponse(content=None, tool_calls=[tool_call]),
                LLMResponse(content="Let me try again."),
            ]
        )
        gen = LLMLineGenerator(
            context_builder=context_builder,
            prompt_manager=prompt_manager,
            llm_client=client,
            prompt_path="test_dialogue",
            tools=[tool],
        )
        result = await gen.generate(conversation)

        # Should not raise; should return the second LLM response
        assert result == "Let me try again."
        # execute should never have been called (validated_greet would fail
        # if execute ran with invalid args)
        # The second call should contain a tool message with the error
        second_call_messages = client.all_messages[1]
        tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
        assert tool_msg["content"].startswith("Invalid arguments for tool")

    @pytest.mark.asyncio
    async def test_execution_error_crashes_loudly(
        self, context_builder, prompt_manager, conversation
    ):
        """A tool whose execute raises re-raises the exception."""
        tool = _ExplodingTool()
        tool_call = ToolCall(id="tc1", name="explode", arguments="{}")
        client = FakeLLMClient(
            responses=[
                LLMResponse(content=None, tool_calls=[tool_call]),
                LLMResponse(content="Should not reach here."),
            ]
        )
        gen = LLMLineGenerator(
            context_builder=context_builder,
            prompt_manager=prompt_manager,
            llm_client=client,
            prompt_path="test_dialogue",
            tools=[tool],
        )
        with pytest.raises(RuntimeError, match="boom"):
            await gen.generate(conversation)
        # Only the first LLM call should have been made
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_fire_and_forget_tool_ends_turn(
        self, context_builder, prompt_manager, conversation
    ):
        """A tool with returns_to_llm=False ends the turn immediately."""
        tool = _FireAndForgetTool()
        tool_call = ToolCall(id="tc1", name="ping", arguments="{}")
        client = FakeLLMClient(
            responses=[
                LLMResponse(content=None, tool_calls=[tool_call]),
                LLMResponse(content="Should not reach here."),
            ]
        )
        gen = LLMLineGenerator(
            context_builder=context_builder,
            prompt_manager=prompt_manager,
            llm_client=client,
            prompt_path="test_dialogue",
            tools=[tool],
        )
        result = await gen.generate(conversation)

        assert result == ""
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_end_conversation_stops_loop(
        self, context_builder, prompt_manager, conversation
    ):
        """EndConversationTool sets conversation.ended; generate returns '' without further LLM call."""
        tool = _EndConvTool()
        tool_call = ToolCall(id="tc1", name="end_conversation", arguments="{}")
        client = FakeLLMClient(
            responses=[
                LLMResponse(content=None, tool_calls=[tool_call]),
                LLMResponse(content="Should not reach here."),
            ]
        )
        gen = LLMLineGenerator(
            context_builder=context_builder,
            prompt_manager=prompt_manager,
            llm_client=client,
            prompt_path="test_dialogue",
            tools=[tool],
        )
        result = await gen.generate(conversation)

        assert result == ""
        assert conversation.ended is True
        assert client.call_count == 1
