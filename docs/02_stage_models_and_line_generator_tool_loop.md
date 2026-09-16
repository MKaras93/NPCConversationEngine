# Stage 2 — Models + auto-execute tool loop in LLMLineGenerator

## Files to touch

- `src/npc_conversation_engine/models/character.py` (modify)
- `src/npc_conversation_engine/models/conversation.py` (modify)
- `src/npc_conversation_engine/line_generator.py` (modify)
- `tests/test_conversation.py` (modify)
- `tests/test_line_generator.py` (modify)

## Implementation Details

### `models/character.py`

Add a field to `Character`:

```python
tools: list[str] = []
```

Names reference tool instances registered on the engine (Stage 3).

### `models/conversation.py`

Add termination state to `Conversation`:

```python
def __init__(self, speaker, listener, location):
    ...
    self.ended: bool = False


def end(self) -> None:
    self.ended = True
```

No changes to `to_text()` or `to_llm_messages`.

### `line_generator.py`

Import `MAX_TOOL_ITERATIONS`, `ToolCall`, `BaseTool`, plus `json`, `logging` (add a
module-level `logger = logging.getLogger(__name__)`), and `from pydantic import
ValidationError`.

`LLMLineGenerator.__init__` gains a trailing param:

```python
tools: list[BaseTool] | None = None
```

stored as `self._tools = tools or []`.

`generate()` becomes an auto-execute loop:

```python
async def generate(self, conversation):
    # ... existing context/system/history assembly unchanged ...
    messages: list[dict] = [{"role": "system", "content": system_msg}]
    messages.extend(history_messages)

    tool_specs = [t.to_openai_spec() for t in self._tools]

    for _ in range(MAX_TOOL_ITERATIONS):
        response = await self._llm_client.chat(
            messages, self._model, self._temperature, tools=tool_specs or None
        )
        if not response.tool_calls:
            return response.content or ""

        fed_back_calls: list[ToolCall] = []
        results: dict[str, str] = {}
        for tool_call in response.tool_calls:
            tool = self._resolve_tool(tool_call.name)
            if tool is None:
                results[tool_call.id] = f"Error: unknown tool '{tool_call.name}'."
                fed_back_calls.append(tool_call)
                continue

            try:
                arguments = tool_call.parsed_arguments
                if tool.arguments_model is not None:
                    arguments = tool.arguments_model.model_validate(arguments)
            except (json.JSONDecodeError, ValidationError) as exc:
                # Bad input from the LLM — feed back a sanitized message so it can
                # retry with corrected arguments.
                results[tool_call.id] = (
                    f"Invalid arguments for tool '{tool_call.name}': {exc}"
                )
                fed_back_calls.append(tool_call)
                continue

            try:
                result = await tool.execute(arguments, conversation)
            except Exception:
                # Genuine execution failure — crash loudly, never feed to the LLM.
                logger.exception("Tool '%s' failed during execution", tool_call.name)
                raise

            if conversation.ended:
                return ""
            if tool.returns_to_llm:
                results[tool_call.id] = result
                fed_back_calls.append(tool_call)

        if not fed_back_calls:
            return ""  # all tools were fire-and-forget: end the turn

        messages.append(self._assistant_tool_call_message(fed_back_calls))
        for tool_call in fed_back_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": results[tool_call.id],
                }
            )
    return ""
```

Two private helpers:

```python
@staticmethod
def _assistant_tool_call_message(tool_calls: list[ToolCall]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ],
    }


def _resolve_tool(self, name: str) -> BaseTool | None:
    return next((t for t in self._tools if t.name == name), None)
```

Notes:
- `messages` type widens from `list[dict[str, str]]` to `list[dict]` (tool messages
  carry `tool_calls` / `tool_call_id`).
- Tool calls/results are transient (local `messages` list only); they are NOT
  written to `conversation.history`.
- `tool_specs or None` means the plain-text path (no tools) sends no `tools` param.
- All tools are advertised to the LLM; `returns_to_llm` only controls whether their
  result is fed back and the loop continues, or the turn ends immediately.
- Invalid LLM arguments (JSON decode / Pydantic `ValidationError`) are fed back to
  the LLM as a sanitized message so it can retry. Execution errors from `execute`
  are logged and re-raised (crash loudly) — never fed to the LLM, to avoid leaking
  internal data or burning tokens. Use `arguments_model` to get this validation for
  free; with a raw `parameters` dict, invalid args surface as an execution error.

### `tests/test_conversation.py`

Add a test that `Conversation.end()` sets `ended`, and that a fresh conversation
starts with `ended is False`.

### `tests/test_line_generator.py` (update)

- Update the existing `FakeLLMClient.chat` to the new signature and return
  `LLMResponse(content=self._response)`.
- Update any assertions that inspect `last_messages`/`last_model`/etc. to also
  capture `last_tools`.
- New tests:
  - `test_no_tools_sends_tools_none` — generator with no tools passes
    `tools=None`.
  - `test_tools_advertised` — generator tools are all advertised: the client call
    receives `tools=[spec, ...]` for every tool.
  - `test_tool_call_loop` — a `FakeLLMClient` returns a `LLMResponse` with a
    `ToolCall` (tool has default `returns_to_llm=True`), then a text
    `LLMResponse`; assert the tool's `execute` was called with parsed arguments,
    that the second call's messages include an assistant `tool_calls` message and
    a `tool` result message, and the final return value is the text.
  - `test_invalid_arguments_fed_back` — a tool with `arguments_model` whose
    `model_validate` raises; assert `generate` does not raise, that a second LLM
    call includes a `tool` message whose content starts with
    `"Invalid arguments for tool"`, and `execute` is never called.
  - `test_execution_error_crashes_loudly` — a tool whose `execute` raises
    `RuntimeError`; assert `generate` re-raises it (via `pytest.raises`) and no
    follow-up LLM call is made (client `call_count == 1`).
  - `test_fire_and_forget_tool_ends_turn` — a tool with `returns_to_llm=False`;
    assert `execute` is called once, `generate` returns `""`, and no follow-up LLM
    call is made (client `call_count == 1`).
  - `test_end_conversation_stops_loop` — `EndConversationTool` sets
    `conversation.ended`; assert `generate` returns `""` without a further LLM
    call.

## Verification Step

```powershell
uv run pytest tests/test_conversation.py tests/test_line_generator.py
```
