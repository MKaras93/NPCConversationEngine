import logging
import os

import pytest

from npc_conversation_engine.context_builder import ExampleContextBuilder
from npc_conversation_engine.line_generator import LLMLineGenerator
from npc_conversation_engine.llm_client import OpenAIChat
from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.conversation import Conversation
from npc_conversation_engine.models.location import Location
from npc_conversation_engine.prompt_manager import PromptManager

FIXTURES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "fixtures", "prompts")
)


@pytest.fixture(autouse=True)
def _enable_llm_logging():
    logger = logging.getLogger("npc_conversation_engine.llm_client")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    yield
    logger.removeHandler(handler)


@pytest.fixture
def conversation():
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
    )
    location = Location(name="Vatican City")
    return Conversation(speaker=speaker, listener=listener, location=location)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_generator_returns_response(conversation):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    llm_client = OpenAIChat(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    generator = LLMLineGenerator(
        context_builder=ExampleContextBuilder(),
        prompt_manager=PromptManager(language="en", prompts_dir=FIXTURES_DIR),
        llm_client=llm_client,
        prompt_path="integration_test",
        model="openrouter/free",
        temperature=0.7,
    )

    response = await generator.generate(conversation)

    assert isinstance(response, str)
    assert len(response) > 0
