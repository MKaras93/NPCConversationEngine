# NPC Conversation Engine

A Python library for generating multi-turn NPC dialogue using LLMs. It handles conversation state, role switching, event injection, and prompt templating — so you can focus on building game logic, not plumbing.

## Features

- Multi-turn conversation engine with automatic role alternation
- LLM-backed dialogue via any OpenAI-compatible API (OpenRouter, OpenAI, etc.)
- Human-controlled character stub for player input
- Jinja2 prompt templates with language fallback
- JSON storage for persisting characters and locations
- Event injection into active conversations
- Extensible: custom context builders, line generators, LLM clients

## Getting Started

1. Install uv if you haven't already:
   - **Linux/macOS**: `make install-uv-unix`
   - **Windows**: `make install-uv-win`
   - Or see https://docs.astral.sh/uv/getting-started/installation/
2. Install dependencies: `make install-dependencies` (or `uv sync --extra dev`)
3. For integration tests, set your API key:
   ```bash
   export OPENROUTER_API_KEY=sk-or-...
   ```
4. Run tests: `make test`

## Usage Examples

### Define Characters and Location

```python
from npc_conversation_engine.models import Character, Location

gandalf = Character(
    name="Gandalf",
    gender="male",
    age=2000,
    appearance="tall wizard with a long grey beard and pointed hat",
    bio="A wise and powerful wizard who guides the Fellowship.",
    type="npc",
    line_generator="LLMLineGenerator",
)

frodo = Character(
    name="Frodo",
    gender="male",
    age=50,
    appearance="short hobbit with curly brown hair",
    bio="A hobbit from the Shire, bearer of the One Ring.",
    type="player",
    line_generator="LLMLineGenerator",
)

location = Location(name="The Shire")
```

### Set Up the Engine

```python
from npc_conversation_engine.context_builder import ExampleContextBuilder
from npc_conversation_engine.conversation_engine import ConversationEngine
from npc_conversation_engine.llm_client import OpenAIChat
from npc_conversation_engine.prompt_manager import PromptManager

llm_client = OpenAIChat(
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1",
)

engine = ConversationEngine(
    prompt_manager=PromptManager(language="en", prompts_dir="prompts"),
    template_path="my_dialogue",
    context_builder=ExampleContextBuilder(),
    llm_client=llm_client,
)
```

### Run a Multi-Round Conversation

```python
import asyncio


async def main():
    engine.initialize_conversation(location, gandalf, frodo)
    lines = await engine.run_conversation(rounds=4)
    # lines[0]: Gandalf speaks (LLM-generated)
    # lines[1]: Frodo speaks (LLM-generated)
    # lines[2]: Gandalf speaks
    # lines[3]: Frodo speaks

    # Print the full conversation
    print(engine.conversation.to_text())


asyncio.run(main())
```

### Manual Control — Events and Role Switching

```python
async def main():
    engine.initialize_conversation(location, gandalf, frodo)

    line = await engine.speak_next()  # Gandalf speaks
    engine.switch_roles()

    line = await engine.speak_next()  # Frodo speaks
    engine.switch_roles()

    # Inject a world event — appears in conversation history as <event>
    engine.add_event("A thunderstorm begins. Rain pours down on the Shire.")

    # Continue — the LLM sees the event in context
    lines = await engine.run_conversation(rounds=2)


asyncio.run(main())
```

### Human-Controlled Characters

Use `HumanLineGenerator` for player-controlled characters. It returns an empty string — feed player input externally via `add_message()`.

```python
player = Character(
    name="Frodo",
    gender="male",
    age=50,
    appearance="short hobbit",
    bio="A hobbit from the Shire.",
    type="player",
    line_generator="HumanLineGenerator",
)


async def main():
    engine.initialize_conversation(location, gandalf, player)

    await engine.speak_next()  # Gandalf (LLM) speaks
    engine.switch_roles()

    await engine.speak_next()  # Returns "" for human — skip or display prompt
    engine.add_message(player, "I seek the ring.")  # Inject player input manually
    engine.switch_roles()

    await engine.speak_next()  # Gandalf responds to what the player said


asyncio.run(main())
```

### Persist Characters with Storage

```python
from npc_conversation_engine.storage import Storage

storage = Storage("data/characters")
storage.save(gandalf, "gandalf")

loaded = storage.load("gandalf", Character)  # Returns a Character instance
storage.list_files()  # ["gandalf"]
storage.delete("gandalf")
```

### Custom Context Builder

Subclass `BaseContextBuilder` to control what variables are available in your prompt templates:

```python
from typing import Any
from npc_conversation_engine.context_builder import BaseContextBuilder
from npc_conversation_engine.models.conversation import Conversation


class GameContextBuilder(BaseContextBuilder):
    def __init__(self, time_of_day: str, weather: str):
        self.time_of_day = time_of_day
        self.weather = weather

    def get_context(self, conversation: Conversation) -> dict[str, Any]:
        context = {
            "speaker_name": conversation.speaker.name,
            "speaker_bio": conversation.speaker.bio,
            "listener_name": conversation.listener.name,
            "location": conversation.location.name,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
        }
        return context
```

Then use it in a template:

```jinja2
You are {{ speaker_name }}. {{ speaker_bio }}
You are speaking to {{ listener_name }} at {{ location }}.
It is currently {{ time_of_day }} and the weather is {{ weather }}.
Respond in character.
```

## Prompt Templates

Templates use Jinja2 and are organized by language with a technical fallback:

```
prompts/
  en/                          # English templates
    my_dialogue/
      system.j2                # System prompt for the LLM
  fr/                          # French templates (example)
    my_dialogue/
      system.j2
  technical/                   # Fallback — used if language-specific template is missing
    shared_prompt/
      system.j2
```

When resolving `my_dialogue/system.j2` with language `en`, the engine checks:
1. `prompts/en/my_dialogue/system.j2`
2. `prompts/technical/my_dialogue/system.j2`

The `ExampleContextBuilder` exposes these variables to templates:
- `speaker_name`, `speaker_gender`, `speaker_age`, `speaker_appearance`, `speaker_bio`, `speaker_type`, `speaker_line_generator`
- `listener_name`, `listener_gender`, `listener_age`, `listener_appearance`, `listener_bio`, `listener_type`, `listener_line_generator`
- `location`

## Configuration

The engine uses constructor dependency injection. Configure at construction time:

| Field | Where | Default |
|---|---|---|
| `model` | `ConversationEngine(model="...")` | `"openrouter/free"` |
| `temperature` | `ConversationEngine(temperature=0.7)` | `0.7` |
| `max_history` | `ConversationEngine(max_history=20)` | `None` (no truncation) |
| `llm_timeout_seconds` | `OpenAIChat(timeout=60.0)` | `60.0` |

`NpcEngineSettings` provides reference defaults via `NPC_ENGINE_*` environment variables
(e.g. `NPC_ENGINE_MODEL=gpt-4`). It is not required by the engine — use it if you want
env-driven defaults when constructing your client.

## Project Structure

```
src/
  conversation_engine.py   — ConversationEngine: orchestrates multi-turn dialogue
  context_builder.py       — BaseContextBuilder / ExampleContextBuilder
  line_generator.py        — BaseLineGenerator / LLMLineGenerator / HumanLineGenerator
  llm_client.py            — BaseLLMClient / OpenAIChat (OpenAI-compatible)
  prompt_manager.py        — Jinja2 template rendering with language fallback
  storage.py               — JSON persistence for Pydantic models
  models/
    character.py           — Character model
    conversation.py        — Conversation and ConversationMsg
    location.py            — Location model
  settings/                — NpcEngineSettings (env-driven reference defaults)
tests/
  test_conversation_engine.py — Unit tests with FakeLLMClient
  integration/                — Integration tests (requires OPENROUTER_API_KEY)
```

## Testing

```bash
# Run unit tests
make test

# Run integration tests (requires OPENROUTER_API_KEY)
make test-integration
```
