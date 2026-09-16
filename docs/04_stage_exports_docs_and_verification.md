# Stage 4 — Public exports, AGENTS.md, full verification

## Files to touch

- `src/npc_conversation_engine/__init__.py` (modify)
- `AGENTS.md` (modify)
- (verify) `scripts/test_as_dependency.py` still imports cleanly — no change expected

## Implementation Details

### `src/npc_conversation_engine/__init__.py`

Expose the new tool types at the package top level:

```python
from npc_conversation_engine.settings import NpcEngineSettings
from npc_conversation_engine.tools import BaseTool, EndConversationTool

__all__ = ["NpcEngineSettings", "BaseTool", "EndConversationTool", "__version__"]
```

(`ToolCall` / `LLMResponse` remain importable from
`npc_conversation_engine.llm_client`.)

### `AGENTS.md`

Update the Architecture and Configuration sections to reflect the new design:

- **Architecture**: add `src/npc_conversation_engine/tools.py` (`BaseTool`,
  `EndConversationTool`) and describe the auto-execute tool loop in
  `LLMLineGenerator`, the transient tool messages, and `Conversation.ended`.
- **Configuration (DI) table**: add a `tools` row for `ConversationEngine`
  (`tools=[...]` → indexed by name, used by `Character.tools`), and note
  `BaseLLMClient.chat` now returns `LLMResponse` (breaking change) with an optional
  `tools` parameter.
- **Tools section** (new): document `BaseTool` fields (`name`, `description`,
  `parameters`, `arguments_model`, `returns_to_llm`, `async execute`), the
  `to_openai_spec()` method, constructor-injected instances (e.g.
  `GiveItemTool(game_state)`), `Character.tools` name selection, and the
  exception-boundary split (invalid arguments are fed back to the LLM; execution
  errors crash loudly).

## Verification Step

Full suite + lint/format (no integration tests by default):

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

If formatting is flagged, run `uv run ruff format .` and `uv run ruff check --fix .`,
then re-run the verification commands.
