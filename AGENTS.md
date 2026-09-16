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
- **`src/npc_conversation_engine/llm_client.py`** — `BaseLLMClient` (ABC) / `OpenAIChat` (OpenAI-compatible)
- **`src/npc_conversation_engine/line_generator.py`** — `BaseLineGenerator` / `LLMLineGenerator` / `HumanLineGenerator`
- **`tests/`** — pytest tests; import from `npc_conversation_engine` package
- **`inputs/`, `outputs/`, `local/`** — data/log directories; `local/` and `inputs/local/` are gitignored
- **`.env`** — dev/test convenience only (e.g. `OPENROUTER_API_KEY`); not required by the library

## Configuration (DI)

The engine uses constructor dependency injection. No central config singleton.

| Field | Where set | Default |
|---|---|---|
| `model` | `ConversationEngine(model="...")` → `LLMLineGenerator` | `"openrouter/free"` |
| `temperature` | `ConversationEngine(temperature=0.7)` → `LLMLineGenerator` | `0.7` |
| `max_history` | `ConversationEngine(max_history=20)` → `LLMLineGenerator` | `None` (no truncation) |
| `llm_timeout_seconds` | `OpenAIChat(timeout=60.0)` only | `60.0` |

`NpcEngineSettings` provides reference defaults via `NPC_ENGINE_*` env vars. Consumers read it when constructing `OpenAIChat` for the timeout; the engine does not forward timeout.

## Formatting

Ruff is the sole formatter and linter. CI enforces `ruff check .` and `ruff format --check .` on every push.
