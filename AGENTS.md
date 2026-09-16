# AGENTS.md

## Project Overview

Python library for multi-turn NPC dialogue generation using LLMs. Uses **uv** (dependency manager), **ruff** (formatter/linter), **pytest** (testing), and **pydantic-settings** (config).

## Commands

| Task | Command |
|---|---|
| Install dependencies | `make install-dependencies` (runs `uv sync --extra dev`) |
| Run tests | `make test` (runs `uv run pytest`) |
| Run all tests | `uv run pytest` |
| Run a single test | `uv run pytest tests/test_conversation_engine.py::TestSpeakNext::test_generates_and_adds_to_history` |
| Check formatting | `uv run ruff format --check .` |
| Auto-format | `uv run ruff format .` |
| Lint | `uv run ruff check .` |
| Lint (auto-fix) | `uv run ruff check --fix .` |
| Install pre-commit hooks | `make init-pre-commit` |
| Test as dependency | `make test-package` (builds wheel + runs `scripts/test_as_dependency.py`) |

## Architecture

- **`src/npc_conversation_engine/`** — library code
- **`src/npc_conversation_engine/settings/`** — `NpcEngineSettings(BaseSettings)` with `env_prefix="NPC_ENGINE_"`, no `.env` assumption
- **`src/npc_conversation_engine/conversation_engine.py`** — `ConversationEngine`: orchestrates multi-turn dialogue
- **`src/npc_conversation_engine/llm_client.py`** — `BaseLLMClient` (ABC) / `OpenAIChat` (OpenAI-compatible); also defines `ToolCall` and `LLMResponse`
- **`src/npc_conversation_engine/line_generator.py`** — `BaseLineGenerator` / `LLMLineGenerator` (auto-execute tool loop) / `HumanLineGenerator`
- **`src/npc_conversation_engine/tools.py`** — `BaseTool` (ABC) / `EndConversationTool`; tool definitions and the OpenAI spec adapter
- **`src/npc_conversation_engine/models/`** — `Character`, `Conversation` (with `.ended` flag), `ConversationMsg`, `Location`
- **`tests/`** — pytest tests; import from `npc_conversation_engine` package
- **`inputs/`, `outputs/`, `local/`** — data/log directories; `local/` and `inputs/local/` are gitignored
- **`.env`** — dev/test convenience only (e.g. `OPENROUTER_API_KEY`); not required by the library

### Tool loop

`LLMLineGenerator.generate()` runs an auto-execute tool loop. When the LLM
returns tool calls instead of text, the generator resolves each tool by name,
validates arguments (feeding validation errors back to the LLM), executes the
tool, and appends transient assistant + tool messages to the LLM context. The
loop repeats up to `MAX_TOOL_ITERATIONS` times. If the LLM's final response
is text (no tool calls), that text is returned as the dialogue line.

Invalid arguments (JSON decode errors, pydantic `ValidationError`) are fed
back to the LLM as error strings so it can retry. Execution errors raise
immediately — they are not caught or retried.

When a tool sets `conversation.end()` (e.g. `EndConversationTool`), the
generator returns `""` and the conversation's `.ended` flag becomes `True`.
These tool-induced messages are transient: they appear in the LLM context but
not in the persisted conversation history.

## Configuration (DI)

The engine uses constructor dependency injection. No central config singleton.

| Field | Where set | Default |
|---|---|---|
| `model` | `ConversationEngine(model="...")` → `LLMLineGenerator` | `"openrouter/free"` |
| `temperature` | `ConversationEngine(temperature=0.7)` → `LLMLineGenerator` | `0.7` |
| `max_history` | `ConversationEngine(max_history=20)` → `LLMLineGenerator` | `None` (no truncation) |
| `tools` | `ConversationEngine(tools=[...])` → indexed by name, used by `Character.tools` | `[EndConversationTool]` (always present) |
| `llm_timeout_seconds` | `OpenAIChat(timeout=60.0)` only | `60.0` |

`NpcEngineSettings` provides reference defaults via `NPC_ENGINE_*` env vars. Consumers read it when constructing `OpenAIChat` for the timeout; the engine does not forward timeout.

`BaseLLMClient.chat` returns `LLMResponse` (breaking change from raw `str`) and accepts an optional `tools` parameter for OpenAI-format tool specs.

## Tools

`BaseTool` is the abstract base class for all tools. Key fields:

- **`name`** — unique tool identifier (used in `Character.tools` and OpenAI function calls)
- **`description`** — human-readable purpose (included in the LLM tool spec)
- **`parameters`** — class-variable dict (OpenAI JSON Schema format) defining accepted arguments
- **`arguments_model`** — optional `pydantic.BaseModel` subclass; when set, its schema replaces `parameters` and validated instances are passed to `execute()`
- **`returns_to_llm`** — if `True` (default), the result is fed back to the LLM; if `False` (e.g. `EndConversationTool`), the tool is fire-and-forget

`to_openai_spec()` converts the tool into an OpenAI function-call dict for the LLM API.

Constructor-injected instances let tools carry runtime state (e.g.
`GiveItemTool(game_state)`). The engine indexes tools by `name` and resolves
`Character.tools` entries against them. `EndConversationTool` is always
available.

Invalid arguments (JSON decode errors, pydantic `ValidationError`) are fed
back to the LLM as error strings so it can retry. Execution errors
(e.g. database failures) raise immediately — they are not caught or retried.

## Formatting

Ruff is the sole formatter and linter. CI enforces `ruff check .` and `ruff format --check .` on every push.
