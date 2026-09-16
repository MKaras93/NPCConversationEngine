from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from npc_conversation_engine.llm_client import OpenAIChat


@pytest.fixture
def mock_openai():
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


def _make_completion(content: str = "Hello"):
    choice = MagicMock()
    choice.message.content = content
    completion = MagicMock()
    completion.choices = [choice]
    return completion


class TestOpenAIChatTimeout:
    def test_default_timeout(self, mock_openai):
        chat = OpenAIChat(api_key="key", base_url="http://test")
        assert chat._timeout == 60.0

    def test_custom_timeout(self, mock_openai):
        chat = OpenAIChat(api_key="key", base_url="http://test", timeout=30.0)
        assert chat._timeout == 30.0

    @pytest.mark.asyncio
    async def test_timeout_passed_to_create(self, mock_openai):
        mock_openai.chat.completions.create = AsyncMock(return_value=_make_completion())
        chat = OpenAIChat(api_key="key", base_url="http://test", timeout=15.0)
        await chat.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.5,
        )
        mock_openai.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            timeout=15.0,
        )
