from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from npc_conversation_engine.models.conversation import Conversation


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    arguments_model: type[BaseModel] | None = None
    returns_to_llm: bool = True

    def to_openai_spec(self) -> dict:
        params = (
            self.arguments_model.model_json_schema()
            if self.arguments_model is not None
            else self.parameters
        )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    @abstractmethod
    async def execute(self, arguments, conversation: Conversation) -> str:
        """Execute the tool and return a short result string for the LLM.

        ``arguments`` is the parsed JSON dict, or a validated instance of
        ``arguments_model`` when that attribute is set.
        """


class EndConversationTool(BaseTool):
    name = "end_conversation"
    description = "End the current conversation."
    returns_to_llm = False
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "farewell": {"type": "string", "description": "Optional parting words."},
        },
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict, conversation: Conversation) -> str:
        conversation.end()
        return arguments.get("farewell") or "Conversation ended."
