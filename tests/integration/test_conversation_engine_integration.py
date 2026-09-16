import logging
import os

import pytest

from npc_conversation_engine.context_builder import ExampleContextBuilder
from npc_conversation_engine.conversation_engine import ConversationEngine
from npc_conversation_engine.llm_client import OpenAIChat
from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.location import Location
from npc_conversation_engine.prompt_manager import PromptManager

FIXTURES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "fixtures", "prompts")
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _enable_logging():
    loggers = [
        logging.getLogger("npc_conversation_engine.llm_client"),
        logging.getLogger(__name__),
    ]
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    for log in loggers:
        log.setLevel(logging.INFO)
        log.addHandler(handler)
    yield
    for log in loggers:
        log.removeHandler(handler)


@pytest.fixture
def engine():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    llm_client = OpenAIChat(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return ConversationEngine(
        prompt_manager=PromptManager(language="en", prompts_dir=FIXTURES_DIR),
        template_path="integration_test",
        context_builder=ExampleContextBuilder(),
        llm_client=llm_client,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_conversation_with_event(engine):
    speaker = Character(
        name="Scott Thomas",
        gender="male",
        age=18,
        appearance="tall red-hair guy with long beard",
        bio="A teenager from Ohio, came to Europe for vacation.",
        type="npc",
        line_generator="LLMLineGenerator",
    )
    listener = Character(
        name="Mark",
        gender="male",
        age=21,
        appearance="Short, fat, bald guy",
        bio="",
        type="player",
        line_generator="LLMLineGenerator",
    )
    location = Location(name="Vatican City")

    engine.initialize_conversation(location, speaker, listener)

    first_lines = await engine.run_conversation(rounds=2)
    assert len(first_lines) == 2
    assert all(isinstance(line, str) and len(line) > 0 for line in first_lines)

    engine.add_event("The ground shakes.")

    second_lines = await engine.run_conversation(rounds=3)
    assert len(second_lines) == 3
    assert all(isinstance(line, str) and len(line) > 0 for line in second_lines)

    assert len(engine.conversation.history) == 6
    assert engine.conversation.history[2].speaker == "<event>"
    assert engine.conversation.history[2].content == "The ground shakes."

    logger.info("\n--- Full Conversation ---")
    logger.info(engine.conversation.to_text())
