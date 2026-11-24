from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from npc_conversation_engine.llm_client import LLMResponse, OpenAIChat, ToolCall


@pytest.fixture
def mock_openai():
    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


def _make_completion(content: str = "Hello", tool_calls=None):
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls if tool_calls is not None else []
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
        result = await chat.chat(
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
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello"
        assert result.tool_calls == []


class TestOpenAIChatTools:
    @pytest.mark.asyncio
    async def test_tools_passed_when_non_empty(self, mock_openai):
        mock_openai.chat.completions.create = AsyncMock(
            return_value=_make_completion("Sure")
        )
        chat = OpenAIChat(api_key="key", base_url="http://test")
        tools = [{"type": "function", "function": {"name": "do_thing"}}]
        await chat.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.5,
            tools=tools,
        )
        mock_openai.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            timeout=60.0,
            tools=tools,
        )

    @pytest.mark.asyncio
    async def test_tools_not_passed_when_none(self, mock_openai):
        mock_openai.chat.completions.create = AsyncMock(
            return_value=_make_completion("Sure")
        )
        chat = OpenAIChat(api_key="key", base_url="http://test")
        await chat.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.5,
            tools=None,
        )
        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_tools_not_passed_when_empty_list(self, mock_openai):
        mock_openai.chat.completions.create = AsyncMock(
            return_value=_make_completion("Sure")
        )
        chat = OpenAIChat(api_key="key", base_url="http://test")
        await chat.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.5,
            tools=[],
        )
        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs


class TestOpenAIChatToolCalls:
    @pytest.mark.asyncio
    async def test_tool_calls_parsed(self, mock_openai):
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "Berlin"}'
        mock_openai.chat.completions.create = AsyncMock(
            return_value=_make_completion(None, tool_calls=[tc])
        )
        chat = OpenAIChat(api_key="key", base_url="http://test")
        result = await chat.chat(
            messages=[{"role": "user", "content": "What is the weather?"}],
            model="test-model",
            temperature=0.5,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert result.content is None
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id == "call_123"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == '{"city": "Berlin"}'
        assert tool_call.parsed_arguments == {"city": "Berlin"}

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_empty_list(self, mock_openai):
        mock_openai.chat.completions.create = AsyncMock(
            return_value=_make_completion("Plain text")
        )
        chat = OpenAIChat(api_key="key", base_url="http://test")
        result = await chat.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.5,
        )
        assert result.tool_calls == []
