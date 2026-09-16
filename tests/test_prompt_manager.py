import os

import jinja2
import pytest

from npc_conversation_engine.prompt_manager import PromptManager

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "prompts")


@pytest.fixture
def prompt_manager():
    return PromptManager(language="en", prompts_dir=FIXTURES_DIR)


class TestPromptManagerRender:
    def test_render_hello_prompt(self, prompt_manager):
        path = os.path.join(FIXTURES_DIR, "en", "hello_prompt", "system.j2")
        result = prompt_manager.render(path, character_name="Aldric")
        assert result == "You are Aldric, a brave adventurer."

    def test_render_with_context_dict(self, prompt_manager):
        path = os.path.join(FIXTURES_DIR, "en", "hello_prompt", "system.j2")
        context = {"character_name": "Elara"}
        result = prompt_manager.render(path, **context)
        assert result == "You are Elara, a brave adventurer."

    def test_render_uses_cache(self, prompt_manager, tmp_path):
        prompt_file = tmp_path / "cached_prompt.j2"
        prompt_file.write_text("Hello {{ name }}, welcome!")
        path = str(prompt_file)

        first = prompt_manager.render(path, name="Aldric")
        assert first == "Hello Aldric, welcome!"

        prompt_file.unlink()  # delete file, to see if template was cached

        second = prompt_manager.render(path, name="Aldric")
        assert second == "Hello Aldric, welcome!"

    def test_render_nonexistent_path(self, prompt_manager):
        with pytest.raises(jinja2.exceptions.TemplateNotFound):
            prompt_manager.render("/nonexistent/path.j2")

    def test_render_missing_context_key(self, prompt_manager):
        path = os.path.join(FIXTURES_DIR, "en", "hello_prompt", "system.j2")
        with pytest.raises(jinja2.exceptions.UndefinedError):
            prompt_manager.render(path)

    def test_render_extra_context_key_ignored(self, prompt_manager):
        path = os.path.join(FIXTURES_DIR, "en", "hello_prompt", "system.j2")
        result = prompt_manager.render(
            path, character_name="Aldric", unused_key="ignored"
        )
        assert result == "You are Aldric, a brave adventurer."


class TestPromptManagerResolvePath:
    def test_resolve_prompt_path(self, prompt_manager):
        result = prompt_manager.resolve_prompt_path("hello_prompt/system.j2")
        expected = os.path.join(FIXTURES_DIR, "en", "hello_prompt", "system.j2")
        assert result == expected

    def test_resolve_path_not_found(self, prompt_manager):
        with pytest.raises(FileNotFoundError):
            prompt_manager.resolve_prompt_path("nonexistent/prompt.j2")

    def test_resolve_path_language_dir_missing(self):
        pm = PromptManager(language="fr", prompts_dir=FIXTURES_DIR)
        result = pm.resolve_prompt_path("hello_prompt/system.j2")
        expected = os.path.join(FIXTURES_DIR, "technical", "hello_prompt", "system.j2")
        assert result == expected
