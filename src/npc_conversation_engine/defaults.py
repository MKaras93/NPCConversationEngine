"""Centralized default values for NPCConversationEngine.

Import these constants wherever defaults are needed to maintain
a single source of truth.
"""

DEFAULT_MODEL: str = "openrouter/free"
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_HISTORY: int | None = None
DEFAULT_TIMEOUT: float = 60.0
