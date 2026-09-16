# Stage 3 — ConversationEngine tool resolution + use_tool + early termination

## Files to touch

- `src/npc_conversation_engine/conversation_engine.py` (modify)
- `tests/test_conversation_engine.py` (modify)

## Implementation Details

### `conversation_engine.py`

Imports: add `BaseTool` and `EndConversationTool` from `npc_conversation_engine.tools`.

**Constructor** gains an optional `tools: list[BaseTool] | None = None` param:

```python
def __init__(self, ..., tools: list[BaseTool] | None = None):
    ...
    self._tools: dict[str, BaseTool] = {"end_conversation": EndConversationTool()}
    for tool in tools or []:
        self._tools[tool.name] = tool
```

The built-in `end_conversation` is always present; a developer-provided tool with
the same `name` overrides it.

**Tool resolution + generator cache key** (replaces `_get_generator`):

```python
def _get_generator(self, character):
    gen_type = character.line_generator
    if not gen_type:
        raise ValueError(f"Character '{character.name}' has no line_generator set.")
    tool_names = tuple(character.tools)
    cache_key = (gen_type, tool_names)
    if cache_key not in self._generator_cache:
        tools = self._resolve_tools(tool_names)
        self._generator_cache[cache_key] = self._create_generator(gen_type, tools)
    return self._generator_cache[cache_key]


def _resolve_tools(self, tool_names) -> list[BaseTool]:
    tools = []
    for name in tool_names:
        if name not in self._tools:
            raise ValueError(f"Unknown tool '{name}'. Available: {sorted(self._tools)}")
        tools.append(self._tools[name])
    return tools
```

`_create_generator(gen_type, tools)` passes `tools=tools` into `LLMLineGenerator`
(the `HumanLineGenerator` branch ignores tools).

**Early termination** in `speak_next`:

```python
async def speak_next(self):
    self._require_conversation()
    if self.conversation.ended:
        return ""
    speaker = self.conversation.speaker
    generator = self._get_generator(speaker)
    line = await generator.generate(self.conversation)
    if line:
        self.conversation.history.append(
            ConversationMsg(speaker=speaker.name, content=line)
        )
    logger.info("%s: %s", speaker.name, line)
    return line
```

(`if line:` avoids appending an empty entry when a tool ends the conversation
mid-turn.)

**Early termination** in `run_conversation`:

```python
async def run_conversation(self, rounds: int = 2):
    self._require_conversation()
    lines: list[str] = []
    for _ in range(rounds):
        if self.conversation.ended:
            break
        line = await self.speak_next()
        lines.append(line)
        self.switch_roles()
    return lines
```

**Programmatic tool invocation** (for the player / host app):

```python
async def use_tool(self, tool_name: str, arguments: dict | None = None) -> str:
    self._require_conversation()
    tool = self._tools.get(tool_name)
    if tool is None:
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available: {sorted(self._tools)}"
        )
    return await tool.execute(arguments or {}, self.conversation)
```

### `tests/test_conversation_engine.py` (update)

- Update `FakeLLMClient.chat` to the new `LLMResponse`-returning signature.
- `test_caches_generator` — the cache key is now a `(gen_type, tool_names)` tuple;
  assert the expected key is present.
- New tests:
  - `test_forwards_tools_to_generator` — a character with `tools=["give_item"]` and
    a registered `give_item` tool produces a generator whose `_tools` contains it.
  - `test_unknown_tool_raises` — `character.tools=["missing"]` raises `ValueError`
    with an "Unknown tool" message.
  - `test_use_tool_executes` — `await engine.use_tool("end_conversation")` sets
    `engine.conversation.ended` and returns the result string.
  - `test_use_tool_unknown_raises` — raises `ValueError`.
  - `test_use_tool_requires_conversation` — raises `RuntimeError` with no active
    conversation.
  - `test_speak_next_returns_empty_when_ended` — after ending, `speak_next()`
    returns `""` without calling the client.
  - `test_run_conversation_stops_early` — when the LLM returns a tool call to
    `end_conversation`, `run_conversation(rounds=5)` returns early (fewer than 5
    lines) and `conversation.ended` is `True`.

## Verification Step

```powershell
uv run pytest tests/test_conversation_engine.py
```
