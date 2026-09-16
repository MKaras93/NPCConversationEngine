from abc import ABC, abstractmethod
from typing import Any

from npc_conversation_engine.models.conversation import Conversation


class BaseContextBuilder(ABC):
    """Abstract base class for building context dictionaries from a conversation."""

    @abstractmethod
    def get_context(self, conversation: Conversation) -> dict[str, Any]:
        """Build and return a context dictionary from the given conversation.

        Args:
            conversation: The conversation to extract context from.

        Returns:
            A dictionary of context key-value pairs.
        """


class ExampleContextBuilder(BaseContextBuilder):
    """Builds context with speaker, listener, and location fields."""

    def get_context(self, conversation: Conversation) -> dict[str, Any]:
        speaker_fields = conversation.speaker.__class__.model_fields
        listener_fields = conversation.listener.__class__.model_fields
        context: dict[str, Any] = {}
        for field in speaker_fields:
            context[f"speaker_{field}"] = getattr(conversation.speaker, field)
        for field in listener_fields:
            context[f"listener_{field}"] = getattr(conversation.listener, field)
        context["location"] = conversation.location.name
        return context
