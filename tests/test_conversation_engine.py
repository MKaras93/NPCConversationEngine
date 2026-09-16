import os

import pytest

from npc_conversation_engine.context_builder import ExampleContextBuilder
from npc_conversation_engine.conversation_engine import ConversationEngine
from npc_conversation_engine.llm_client import BaseLLMClient
from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.location import Location
from npc_conversation_engine.prompt_manager import PromptManager

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "prompts")


class FakeLLMClient(BaseLLMClient):
    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["Hello there!"]
        self._call_count = 0

    async def chat(self, messages: list[dict], model: str, temperature: float) -> str:
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response


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
        line_generator="LLMLineGenerator",
    )


@pytest.fixture
def location():
    return Location(name="The Shire")


@pytest.fixture
def prompt_manager():
    return PromptManager(language="en", prompts_dir=FIXTURES_DIR)


@pytest.fixture
def context_builder():
    return ExampleContextBuilder()


@pytest.fixture
def llm_client():
    return FakeLLMClient(responses=["A wizard is never late.", "Indeed."])


@pytest.fixture
def engine(prompt_manager, context_builder, llm_client):
    return ConversationEngine(
        prompt_manager=prompt_manager,
        template_path="test_dialogue",
        context_builder=context_builder,
        llm_client=llm_client,
    )


class TestInitializeConversation:
    def test_creates_conversation(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        assert engine.conversation is not None
        assert engine.conversation.speaker is gandalf
        assert engine.conversation.listener is frodo
        assert engine.conversation.location is location
        assert engine.conversation.history == []

    def test_raises_if_already_initialized(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        with pytest.raises(RuntimeError, match="already active"):
            engine.initialize_conversation(location, frodo, gandalf)


class TestAddMessage:
    def test_adds_with_string_speaker(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        engine.add_message("Frodo", "What is the ring?")
        assert len(engine.conversation.history) == 1
        assert engine.conversation.history[0].speaker == "Frodo"
        assert engine.conversation.history[0].content == "What is the ring?"

    def test_adds_with_character_speaker(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        engine.add_message(frodo, "What is the ring?")
        assert engine.conversation.history[0].speaker == "Frodo"

    def test_raises_without_conversation(self, engine):
        with pytest.raises(RuntimeError, match="No active conversation"):
            engine.add_message("Frodo", "Hello")


class TestAddEvent:
    def test_adds_event_with_special_speaker(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        engine.add_event("A thunderstorm begins.")
        assert len(engine.conversation.history) == 1
        assert engine.conversation.history[0].speaker == "<event>"
        assert engine.conversation.history[0].content == "A thunderstorm begins."

    def test_event_appears_as_system_in_llm_messages(
        self, engine, location, gandalf, frodo
    ):
        engine.initialize_conversation(location, gandalf, frodo)
        engine.add_event("The ground shakes.")
        messages = engine.conversation.to_llm_messages
        assert messages[0] == {
            "role": "user",
            "content": "<event> The ground shakes. </event>",
        }

    def test_raises_without_conversation(self, engine):
        with pytest.raises(RuntimeError, match="No active conversation"):
            engine.add_event("Something happened")


class TestSwitchRoles:
    def test_swaps_speaker_and_listener(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        assert engine.conversation.speaker is gandalf
        assert engine.conversation.listener is frodo
        engine.switch_roles()
        assert engine.conversation.speaker is frodo
        assert engine.conversation.listener is gandalf

    def test_double_switch_restores(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        engine.switch_roles()
        engine.switch_roles()
        assert engine.conversation.speaker is gandalf
        assert engine.conversation.listener is frodo

    def test_raises_without_conversation(self, engine):
        with pytest.raises(RuntimeError, match="No active conversation"):
            engine.switch_roles()


class TestSpeakNext:
    @pytest.mark.asyncio
    async def test_generates_and_adds_to_history(
        self, engine, location, gandalf, frodo
    ):
        engine.initialize_conversation(location, gandalf, frodo)
        line = await engine.speak_next()
        assert line == "A wizard is never late."
        assert len(engine.conversation.history) == 1
        assert engine.conversation.history[0].speaker == "Gandalf"
        assert engine.conversation.history[0].content == "A wizard is never late."

    @pytest.mark.asyncio
    async def test_does_not_switch_roles(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        await engine.speak_next()
        assert engine.conversation.speaker is gandalf
        assert engine.conversation.listener is frodo

    @pytest.mark.asyncio
    async def test_raises_without_conversation(self, engine):
        with pytest.raises(RuntimeError, match="No active conversation"):
            await engine.speak_next()

    @pytest.mark.asyncio
    async def test_raises_for_unknown_generator(
        self, prompt_manager, context_builder, llm_client, location
    ):
        bad_char = Character(
            name="Mystery",
            gender="unknown",
            age=0,
            appearance="",
            bio="",
            type="npc",
            line_generator="NonexistentGenerator",
        )
        other = Character(
            name="Other",
            gender="unknown",
            age=0,
            appearance="",
            bio="",
            type="player",
        )
        engine = ConversationEngine(
            prompt_manager=prompt_manager,
            template_path="test_dialogue",
            context_builder=context_builder,
            llm_client=llm_client,
        )
        engine.initialize_conversation(location, bad_char, other)
        with pytest.raises(ValueError, match="Unknown line generator"):
            await engine.speak_next()

    @pytest.mark.asyncio
    async def test_raises_for_empty_generator(
        self, prompt_manager, context_builder, llm_client, location
    ):
        no_gen = Character(
            name="NoGen",
            gender="unknown",
            age=0,
            appearance="",
            bio="",
            type="npc",
            line_generator="",
        )
        other = Character(
            name="Other",
            gender="unknown",
            age=0,
            appearance="",
            bio="",
            type="player",
        )
        engine = ConversationEngine(
            prompt_manager=prompt_manager,
            template_path="test_dialogue",
            context_builder=context_builder,
            llm_client=llm_client,
        )
        engine.initialize_conversation(location, no_gen, other)
        with pytest.raises(ValueError, match="no line_generator set"):
            await engine.speak_next()

    @pytest.mark.asyncio
    async def test_caches_generator(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        await engine.speak_next()
        await engine.speak_next()
        assert "LLMLineGenerator" in engine._generator_cache


class TestEngineDIForwarding:
    def test_forwards_model_and_temperature(
        self, prompt_manager, context_builder, llm_client, location, gandalf, frodo
    ):
        engine = ConversationEngine(
            prompt_manager=prompt_manager,
            template_path="test_dialogue",
            context_builder=context_builder,
            llm_client=llm_client,
            model="custom-model",
            temperature=0.3,
        )
        engine.initialize_conversation(location, gandalf, frodo)
        generator = engine._get_generator(gandalf)
        assert generator._model == "custom-model"
        assert generator._temperature == 0.3

    def test_forwards_max_history(
        self, prompt_manager, context_builder, llm_client, location, gandalf, frodo
    ):
        engine = ConversationEngine(
            prompt_manager=prompt_manager,
            template_path="test_dialogue",
            context_builder=context_builder,
            llm_client=llm_client,
            max_history=5,
        )
        engine.initialize_conversation(location, gandalf, frodo)
        generator = engine._get_generator(gandalf)
        assert generator._max_history == 5

    def test_none_values_use_generator_defaults(
        self, prompt_manager, context_builder, llm_client, location, gandalf, frodo
    ):
        engine = ConversationEngine(
            prompt_manager=prompt_manager,
            template_path="test_dialogue",
            context_builder=context_builder,
            llm_client=llm_client,
        )
        engine.initialize_conversation(location, gandalf, frodo)
        generator = engine._get_generator(gandalf)
        assert generator._model == "openrouter/free"
        assert generator._temperature == 0.7
        assert generator._max_history is None


class TestRunConversation:
    @pytest.mark.asyncio
    async def test_runs_two_rounds_by_default(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        lines = await engine.run_conversation()
        assert len(lines) == 2
        assert lines[0] == "A wizard is never late."
        assert lines[1] == "Indeed."
        assert len(engine.conversation.history) == 2

    @pytest.mark.asyncio
    async def test_roles_alternate(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        await engine.run_conversation(rounds=2)
        # After 2 rounds (speak + switch each), speaker should be back to gandalf
        assert engine.conversation.speaker is gandalf
        assert engine.conversation.listener is frodo

    @pytest.mark.asyncio
    async def test_history_has_correct_speakers(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        await engine.run_conversation(rounds=2)
        speakers = [msg.speaker for msg in engine.conversation.history]
        assert speakers == ["Gandalf", "Frodo"]

    @pytest.mark.asyncio
    async def test_custom_rounds(self, engine, location, gandalf, frodo):
        engine.initialize_conversation(location, gandalf, frodo)
        lines = await engine.run_conversation(rounds=4)
        assert len(lines) == 4
        assert len(engine.conversation.history) == 4

    @pytest.mark.asyncio
    async def test_raises_without_conversation(self, engine):
        with pytest.raises(RuntimeError, match="No active conversation"):
            await engine.run_conversation()
