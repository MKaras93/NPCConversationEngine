import pytest

from npc_conversation_engine.models.character import Character
from npc_conversation_engine.models.conversation import Conversation, ConversationMsg
from npc_conversation_engine.models.location import Location


@pytest.fixture
def gandalf():
    return Character(
        name="Gandalf", gender="", age=0, appearance="", bio="", type="npc"
    )


@pytest.fixture
def frodo():
    return Character(
        name="Frodo", gender="", age=0, appearance="", bio="", type="player"
    )


@pytest.fixture
def location():
    return Location(name="The Shire")


@pytest.mark.parametrize(
    "history, expected",
    [
        pytest.param(
            [],
            [],
            id="empty-history",
        ),
        pytest.param(
            [
                ConversationMsg(speaker="Frodo", content="What is the ring?"),
                ConversationMsg(speaker="Gandalf", content="It is the One Ring."),
            ],
            [
                {"role": "user", "content": "What is the ring?"},
                {"role": "assistant", "content": "It is the One Ring."},
            ],
            id="listener-then-speaker",
        ),
        pytest.param(
            [
                ConversationMsg(speaker="Gandalf", content="You shall not pass!"),
                ConversationMsg(speaker="Frodo", content="I will take the ring."),
                ConversationMsg(speaker="Gandalf", content="So be it."),
            ],
            [
                {"role": "assistant", "content": "You shall not pass!"},
                {"role": "user", "content": "I will take the ring."},
                {"role": "assistant", "content": "So be it."},
            ],
            id="multiple-exchanges",
        ),
        pytest.param(
            [
                ConversationMsg(speaker="<event>", content="The ground shakes."),
                ConversationMsg(speaker="Gandalf", content="Run!"),
            ],
            [
                {"role": "user", "content": "<event> The ground shakes. </event>"},
                {"role": "assistant", "content": "Run!"},
            ],
            id="event-maps-to-user-role-with-tags",
        ),
    ],
)
def test_to_llm_messages_maps_roles(gandalf, frodo, location, history, expected):
    conversation = Conversation(speaker=gandalf, listener=frodo, location=location)
    conversation.history = history

    assert conversation.to_llm_messages == expected


@pytest.mark.parametrize(
    "history, expected",
    [
        pytest.param(
            [],
            "",
            id="empty-history",
        ),
        pytest.param(
            [
                ConversationMsg(speaker="Frodo", content="What is the ring?"),
                ConversationMsg(speaker="Gandalf", content="It is the One Ring."),
            ],
            "Frodo: What is the ring?\nGandalf: It is the One Ring.",
            id="two-messages",
        ),
        pytest.param(
            [
                ConversationMsg(speaker="Gandalf", content="You shall not pass!"),
                ConversationMsg(speaker="Frodo", content="I will take the ring."),
                ConversationMsg(speaker="<event>", content="The ground shakes."),
                ConversationMsg(speaker="Gandalf", content="So be it."),
            ],
            "Gandalf: You shall not pass!\nFrodo: I will take the ring.\n<event>: The ground shakes.\nGandalf: So be it.",
            id="mixed-speakers-and-events",
        ),
    ],
)
def test_to_text(gandalf, frodo, location, history, expected):
    conversation = Conversation(speaker=gandalf, listener=frodo, location=location)
    conversation.history = history

    assert conversation.to_text() == expected


def test_conversation_starts_not_ended(gandalf, frodo, location):
    conversation = Conversation(speaker=gandalf, listener=frodo, location=location)
    assert conversation.ended is False


def test_conversation_end_sets_ended(gandalf, frodo, location):
    conversation = Conversation(speaker=gandalf, listener=frodo, location=location)
    conversation.end()
    assert conversation.ended is True
