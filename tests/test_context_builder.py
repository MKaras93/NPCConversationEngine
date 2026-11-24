from npc_conversation_engine.context_builder import ExampleContextBuilder
from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.conversation import Conversation
from npc_conversation_engine.models.location import Location


def test_get_context():
    speaker = Character(
        name="Gandalf",
        gender="male",
        age=2000,
        appearance="tall wizard with a grey beard",
        bio="a wizard",
        type="npc",
    )
    listener = Character(
        name="Frodo",
        gender="male",
        age=50,
        appearance="short hobbit with curly hair",
        bio="a hobbit from the Shire",
        type="player",
    )
    location = Location(name="The Shire")
    conversation = Conversation(speaker=speaker, listener=listener, location=location)

    result = ExampleContextBuilder().get_context(conversation)

    expected_result = {
        "speaker_name": "Gandalf",
        "speaker_gender": "male",
        "speaker_age": 2000,
        "speaker_appearance": "tall wizard with a grey beard",
        "speaker_bio": "a wizard",
        "speaker_type": "npc",
        "speaker_line_generator": "",
        "speaker_tools": [],
        "listener_name": "Frodo",
        "listener_gender": "male",
        "listener_age": 50,
        "listener_appearance": "short hobbit with curly hair",
        "listener_bio": "a hobbit from the Shire",
        "listener_type": "player",
        "listener_line_generator": "",
        "listener_tools": [],
        "location": "The Shire",
    }
    assert result == expected_result
