from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from npc_conversation_engine.tools import BaseTool, EndConversationTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleTool(BaseTool):
    name = "simple"
    description = "A simple tool."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "additionalProperties": False,
    }

    async def execute(self, arguments, conversation):
        return "done"


class GiveItemArgs(BaseModel):
    item_name: str
    quantity: int = 1


class GiveItemTool(BaseTool):
    name = "give_item"
    description = "Give an item to the listener."
    arguments_model = GiveItemArgs

    async def execute(self, arguments, conversation):
        return f"Gave {arguments.quantity}x {arguments.item_name}"


def _stub_conversation():
    conv = MagicMock()
    conv.ended = False

    def end():
        conv.ended = True

    conv.end = end
    return conv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseToolToOpenaiSpec:
    def test_returns_openai_function_spec(self):
        tool = SimpleTool()
        spec = tool.to_openai_spec()
        assert spec == {
            "type": "function",
            "function": {
                "name": "simple",
                "description": "A simple tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
        }

    def test_returns_to_llm_defaults_true(self):
        tool = SimpleTool()
        assert tool.returns_to_llm is True


class TestBaseToolArgumentsModel:
    def test_arguments_model_produces_schema(self):
        tool = GiveItemTool()
        spec = tool.to_openai_spec()
        assert spec["function"]["parameters"] == GiveItemArgs.model_json_schema()

    def test_arguments_model_name_in_spec(self):
        tool = GiveItemTool()
        spec = tool.to_openai_spec()
        assert spec["function"]["name"] == "give_item"


class TestEndConversationTool:
    @pytest.mark.asyncio
    async def test_execute_sets_ended(self):
        tool = EndConversationTool()
        conv = _stub_conversation()
        result = await tool.execute({}, conv)
        assert conv.ended is True
        assert result == "Conversation ended."

    @pytest.mark.asyncio
    async def test_execute_returns_farewell(self):
        tool = EndConversationTool()
        conv = _stub_conversation()
        result = await tool.execute({"farewell": "See you!"}, conv)
        assert conv.ended is True
        assert result == "See you!"

    def test_returns_to_llm_is_false(self):
        tool = EndConversationTool()
        assert tool.returns_to_llm is False

    def test_spec_has_name_and_description(self):
        tool = EndConversationTool()
        spec = tool.to_openai_spec()
        assert spec["function"]["name"] == "end_conversation"
        assert spec["function"]["description"] == "End the current conversation."
