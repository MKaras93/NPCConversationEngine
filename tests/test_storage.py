import pytest

from npc_conversation_engine.models import Character
from npc_conversation_engine.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path / "characters"))


@pytest.fixture
def sample_character():
    return Character(
        name="Aldric",
        gender="male",
        age=32,
        appearance="tall, scarred",
        bio="former knight",
        type="human",
    )


class TestStorageSaveAndLoad:
    def test_save_and_load_returns_same_characters(self, storage, sample_character):
        other = Character(
            name="Elara",
            gender="female",
            age=28,
            appearance="short, dark hair",
            bio="mage",
            type="npc",
        )
        storage.save(sample_character, "aldric")
        storage.save(other, "elara")
        assert storage.load("aldric", Character) == sample_character
        assert storage.load("elara", Character) == other

    def test_load_nonexistent_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.load("missing", Character)


class TestStorageListFiles:
    def test_list_empty_directory(self, storage):
        assert storage.list_files() == []

    def test_list_multiple_files(self, storage, sample_character):
        other = Character(
            name="Elara",
            gender="female",
            age=28,
            appearance="short, dark hair",
            bio="mage",
            type="npc",
        )
        storage.save(sample_character, "aldric")
        storage.save(other, "elara")
        files = storage.list_files()
        assert sorted(files) == ["aldric", "elara"]


class TestStorageDelete:
    def test_delete_removes_file(self, storage, sample_character):
        storage.save(sample_character, "aldric")
        storage.delete("aldric")
        assert storage.list_files() == []

    def test_delete_nonexistent_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.delete("missing")
