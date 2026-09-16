import logging

from npc_conversation_engine.settings import NpcEngineSettings

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = ["NpcEngineSettings", "__version__"]
