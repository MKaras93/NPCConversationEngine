from dataclasses import dataclass

from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.location import Location


@dataclass
class ConversationMsg:
    speaker: str
    content: str


class Conversation:
    def __init__(self, speaker: Character, listener: Character, location: Location):
        self.speaker = speaker
        self.listener = listener
        self.location = location
        self.history: list[ConversationMsg] = []
        self.ended: bool = False

    def end(self) -> None:
        self.ended = True

    def to_text(self) -> str:
        lines = [f"{msg.speaker}: {msg.content}" for msg in self.history]
        return "\n".join(lines)

    @property
    def to_llm_messages(self) -> list[dict[str, str]]:
        messages = []
        for msg in self.history:
            if msg.speaker == "<event>":
                role = "user"
                content = f"<event> {msg.content} </event>"
            elif msg.speaker == self.speaker.name:
                role = "assistant"
                content = msg.content
            else:
                role = "user"
                content = msg.content
            messages.append({"role": role, "content": content})
        return messages
