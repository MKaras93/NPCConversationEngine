# Stage 1 — LLM client tool support + tool types

## Files to touch

- `src/npc_conversation_engine/llm_client.py` (modify)
- `src/npc_conversation_engine/tools.py` (new)
- `src/npc_conversation_engine/defaults.py` (modify)
- `tests/test_llm_client.py` (modify)
- `tests/test_tools.py` (new)

## Implementation Details

### `defaults.py`

Add one constant:

```python
MAX_TOOL_ITERATIONS: int = 5
```

### `llm_client.py`

Add two dataclasses and change the abstract interface + `OpenAIChat`:

```python
from dataclasses import dataclass, field
import json


@dataclass
class ToolCall:
    """A single tool/function call requested by the LLM."""

    id: str
    name: str
    arguments: str  # raw JSON string exactly as returned by the model

    @property
    def parsed_arguments(self) -> dict:
        return json.loads(self.arguments or "{}")


@dataclass
class LLMResponse:
    """Structured response from an LLM chat call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
```

`BaseLLMClient.chat` signature becomes (breaking change):

```python
async def chat(
    self,
    messages: list[dict],
    model: str,
    temperature: float,
    tools: list[dict] | None = None,
) -> LLMResponse:
```

`OpenAIChat.chat`:
- Build kwargs with `model`, `messages`, `temperature`, `timeout`; add `tools=tools`
  **only when `tools` is non-empty** (so the plain-text path and existing tests stay
  valid).
- Read `message = response.choices[0].message`.
- `content = message.content`.
- Parse tool calls defensively (mock-safe):
  `raw = getattr(message, "tool_calls", None) or []`, then map each entry to
  `ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)`.
- Return `LLMResponse(content=content, tool_calls=tool_calls)`.
- Keep the existing `logger.info`/`logger.debug` request/response logging.

### `tools.py` (new)

```python
from abc import ABC, abstractmethod

from pydantic import BaseModel

from npc_conversation_engine.models.conversation import Conversation


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict = {
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
    parameters = {
        "type": "object",
        "properties": {
            "farewell": {"type": "string", "description": "Optional parting words."},
        },
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict, conversation: Conversation) -> str:
        conversation.end()
        return arguments.get("farewell") or "Conversation ended."
```

`EndConversationTool` requires `Conversation.end()` (added in Stage 2), so Stage 1
tests for `EndConversationTool` will use a stub conversation or a `MagicMock` with
an `ended` attribute; the real `Conversation.end()` lands in Stage 2.

Prefer `arguments_model` (a Pydantic model) over a raw `parameters` dict for typed,
auto-generated schemas; keep `parameters` as the escape hatch for exotic schemas.
Simple flat Pydantic models produce OpenAI-compatible schemas; deeply nested models
may emit `$defs`/`title` fields that a few strict providers reject — use `parameters`
in that case. Using `arguments_model` also enables automatic validation: invalid
LLM arguments are fed back to the LLM for correction (Stage 2).

### `tests/test_llm_client.py` (update)

- The mock `choice.message` must expose `content` and `tool_calls`. Set
  `choice.message.tool_calls = []` in `_make_completion` so parsing works.
- Assert `chat` returns an `LLMResponse` with the expected `content`.
- Add a test that when `tools=[...]` is passed, `create` is called with
  `tools=...`; and that when `tools` is empty/None, `tools` is **not** passed.
- Add a test where `message.tool_calls` is a list of mock entries and verify the
  returned `LLMResponse.tool_calls` contains parsed `ToolCall` objects.

### `tests/test_tools.py` (new)

- `TestBaseToolToOpenaiSpec` — subclass a `BaseTool` and assert `to_openai_spec()`
  returns `{"type": "function", "function": {...name, description, parameters}}`,
  and that `returns_to_llm` defaults to `True`.
- `TestBaseToolArgumentsModel` — a tool with `arguments_model = GiveItemArgs`
  produces a spec whose `parameters` equals `GiveItemArgs.model_json_schema()`.
- `TestEndConversationTool` — with a stub conversation (object with
  `ended = False` + `end()` method), assert `execute({}, conv)` sets `ended` and
  returns `"Conversation ended."`; assert a `farewell` argument is returned; and
  assert `returns_to_llm is False`.

## Verification Step

```powershell
uv run pytest tests/test_llm_client.py tests/test_tools.py
```
