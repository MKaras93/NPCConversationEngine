"""Library-safe settings for NPCConversationEngine.

Usage:
    from npc_conversation_engine.settings import NpcEngineSettings
    settings = NpcEngineSettings()  # reads NPC_ENGINE_* env vars
    print(settings.model)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from npc_conversation_engine.defaults import (
    DEFAULT_MAX_HISTORY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)


class NpcEngineSettings(BaseSettings):
    """Settings for NPCConversationEngine, read from environment variables.

    All fields are optional and have sensible defaults. Consumers can override
    via NPC_ENGINE_* environment variables or by passing values to the constructor.
    """

    model_config = SettingsConfigDict(env_prefix="NPC_ENGINE_", extra="ignore")

    llm_timeout_seconds: float = DEFAULT_TIMEOUT
    max_history: int | None = DEFAULT_MAX_HISTORY
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE


__all__ = ["NpcEngineSettings"]
