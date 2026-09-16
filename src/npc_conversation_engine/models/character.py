from pydantic import BaseModel


class Character(BaseModel):
    """Represents a character in the conversation engine.

    Contains information about the character and links to a line generator
    type (human or npc) for producing dialogue.
    """

    name: str
    gender: str
    age: int
    appearance: str
    bio: str
    type: str
    line_generator: str = ""
