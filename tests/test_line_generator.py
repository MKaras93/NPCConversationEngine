import os

import pytest

from npc_conversation_engine.context_builder import ExampleContextBuilder
from npc_conversation_engine.line_generator import (
    LINE_GENERATOR_REGISTRY,
    HumanLineGenerator,
    LLMLineGenerator,
)
from npc_conversation_engine.llm_client import BaseLLMClient
from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.conversation import Conversation, ConversationMsg
from npc_conversation_engine.models.location import Location
from npc_conversation_engine.prompt_manager import PromptManager
from npc_conversation_engine.storage import Storage

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "prompts")


class FakeLLMClient(BaseLLMClient):
    def __init__(self, response: str = "Hello there!"):
        self._response = response
        self.last_messages: list[dict] | None = None
        self.last_model: str | None = None
        self.last_temperature: float | None = None

    async def chat(self, messages: list[dict], model: str, temperature: float) -> str:
        self.last_messages = messages
        self.last_model = model
        self.last_temperature = temperature
        return self._response


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
    return FakeLLMClient(response="A wizard is never late.")


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
