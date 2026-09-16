# Feature: Tools for NPCs and Player

## Goal

Extend the library so that **both NPCs and the player can invoke tools**, with the
LLM able to call OpenAI-style function/tool calls and the engine able to execute
them against application logic (e.g. a game's `GameState`). The flagship example is
a built-in `end_conversation` tool that an NPC (or the player) can trigger to
terminate the current conversation.

This lets a consuming game wire arbitrary side effects — `game_state.give_item(...)`,
quest updates, etc. — into dialogue, driven either by the LLM (NPC) or by the host
application (player / human-controlled characters).

## Scope

**In scope**

- A `BaseTool` ABC + OpenAI function-spec generation (`to_openai_spec()`).
- A built-in `EndConversationTool` that flips a `Conversation.ended` flag.
- LLM tool-calling support in the client (`BaseLLMClient` / `OpenAIChat`):
  pass `tools`, parse `tool_calls`.
- An auto-execute + loop in `LLMLineGenerator`: when the LLM returns tool calls,
  execute them, feed results back and loop (for `returns_to_llm=True` tools) or end
  the turn (for `returns_to_llm=False` tools), capped at `MAX_TOOL_ITERATIONS`.
- `Character.tools: list[str]` to declare which tools a character may use.
- `Conversation.ended` + early termination of `speak_next()` / `run_conversation()`.
- `ConversationEngine.use_tool(...)` for programmatic (player/human) invocation.
- Constructor-injected tool instances so tools can capture application services
  (e.g. `GameState`).
- Exception-safe tool execution: invalid LLM arguments (JSON/Schema validation
  failures) are fed back to the LLM so it can correct itself; genuine execution
  failures crash loudly instead of leaking internals or burning tokens.
- Optional Pydantic `arguments_model` for typed, auto-generated JSON schemas (the
  raw `parameters` dict remains as an escape hatch).

**Out of scope**

- Persisting tool calls/results into `Conversation.history` (they are transient,
  fed back to the LLM within a single turn only). The `ended` flag is the durable
  outcome; a tool that wants a history record can append to history itself.
- Multi-step reasoning/planning agents, streaming, or parallel tool fan-out beyond
  the OpenAI `tool_calls` list that the SDK already returns.
- Non-OpenAI LLM providers (the existing `OpenAIChat` remains the only client).

## Approach

Follow the library's existing dependency-injection and ABC style. Introduce tool
types, extend the LLM client to return structured responses, run the tool loop
inside `LLMLineGenerator.generate()`, and let `ConversationEngine` own the set of
available tool instances (resolving `Character.tools` names against them).

### Architectural decisions (recorded in AGENTS.md)

1. **`BaseLLMClient.chat` becomes structured.** Return type changes from `str` to
   `LLMResponse(content, tool_calls)`, and the signature gains an optional `tools`
   parameter. This is a deliberate breaking change for any custom `BaseLLMClient`
   subclass (acceptable at v0.1.0).
2. **Tools are constructor-injected instances, not auto-instantiated classes.**
   The engine receives `tools: list[BaseTool]` and indexes them by `name`.
   `Character.tools` holds names that select from that set. This is how
   `GiveItemTool(game_state)` captures application state.
3. **`returns_to_llm` flag.** Every tool in a character's `tools` list is advertised
   to the LLM. `returns_to_llm=True` (default) feeds the tool result back and loops
   to a final text line; `returns_to_llm=False` executes the tool and then ends the
   turn without a follow-up completion (fire-and-forget). `end_conversation` uses
   `returns_to_llm=False` plus sets `Conversation.ended`.
4. **Tools are an exception boundary with a clear split.** Invalid arguments from
   the LLM (JSON decode / Pydantic `ValidationError`) are fed back to the LLM as a
   sanitized message so it can retry. Any other exception raised inside `execute`
   is logged and re-raised (crash loudly) — never fed to the LLM, to avoid leaking
   internal data or burning tokens.

### Stage plan

- `01_stage_llm_client_and_tool_types.md` — `ToolCall`, `LLMResponse`, `BaseTool`,
  `EndConversationTool`, client changes, `MAX_TOOL_ITERATIONS`.
- `02_stage_models_and_line_generator_tool_loop.md` — `Character.tools`,
  `Conversation.ended`, and the auto-execute loop in `LLMLineGenerator`.
- `03_stage_engine_integration.md` — tool resolution, `use_tool()`, and early
  termination in the engine.
- `04_stage_exports_docs_and_verification.md` — public exports, AGENTS.md update,
  full test + lint verification.
