# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Smoke test: verifies the package works when installed as an external dependency.

Run via: make test-package
"""

import logging
import os

import npc_conversation_engine
from npc_conversation_engine import __version__

# Imported to verify they're available in the package (smoke test)
from npc_conversation_engine.context_builder import (
    BaseContextBuilder,
    ExampleContextBuilder,  # noqa: F401
)
from npc_conversation_engine.line_generator import (
    LINE_GENERATOR_REGISTRY,
    BaseLineGenerator,
)
from npc_conversation_engine.llm_client import BaseLLMClient  # noqa: F401
from npc_conversation_engine.models import (
    Character,
    Conversation,
    ConversationMsg,
    Location,
)
from npc_conversation_engine.prompt_manager import PromptManager  # noqa: F401
from npc_conversation_engine.settings import NpcEngineSettings
from npc_conversation_engine.storage import Storage  # noqa: F401

# === Edge Case 1: Verify we're testing the installed wheel, not local source ===
package_path = str(npc_conversation_engine.__file__)
assert "site-packages" in package_path, (
    f"FAIL: Imported local source instead of installed wheel!\nPath: {package_path}"
)
print(f"[OK] Loaded from site-packages: {package_path}")

# === Basic imports ===
print(f"[OK] Package version: {__version__}")

# === Model instantiation ===
char = Character(
    name="Gandalf",
    gender="male",
    age=2000,
    appearance="wizard",
    bio="wizard",
    type="npc",
)
loc = Location(name="The Shire")
print(f"[OK] Model instantiation: {char.name}, {loc.name}")

# === Conversation creation ===
frodo = Character(
    name="Frodo",
    gender="male",
    age=50,
    appearance="hobbit",
    bio="hobbit",
    type="player",
)
conv = Conversation(speaker=char, listener=frodo, location=loc)
msg = ConversationMsg(speaker="Gandalf", content="Hello, Frodo!")
conv.history.append(msg)
print(
    f"[OK] Conversation: {len(conv.history)} message(s), LLM format: {conv.to_llm_messages}"
)

# === Settings instantiation (no .env, no side effects) ===
cwd_before = set(os.listdir("."))
settings = NpcEngineSettings()
assert settings.llm_timeout_seconds == 60.0
assert settings.max_history is None
assert settings.model == "openrouter/free"
assert settings.temperature == 0.7
cwd_after = set(os.listdir("."))
assert cwd_before == cwd_after, (
    f"FAIL: Settings instantiation created files: {cwd_after - cwd_before}"
)
print(f"[OK] Settings: model={settings.model}, timeout={settings.llm_timeout_seconds}")

# === Logging: library logger has a NullHandler (no "no handler" warnings) ===
import contextlib
import io

stderr_buf = io.StringIO()
with contextlib.redirect_stderr(stderr_buf):
    test_logger = logging.getLogger("npc_conversation_engine.test")
    test_logger.warning("test warning")
    test_logger.info("test info")

stderr_output = stderr_buf.getvalue()
assert "No handlers could be found" not in stderr_output, (
    f"FAIL: NullHandler not working — got: {stderr_output}"
)
print("[OK] Logging: library logger has a NullHandler")

# === Line generator registry ===
assert "LLMLineGenerator" in LINE_GENERATOR_REGISTRY
assert "HumanLineGenerator" in LINE_GENERATOR_REGISTRY
print(f"[OK] Line generators: {list(LINE_GENERATOR_REGISTRY.keys())}")

# === Abstract base classes are importable ===
assert issubclass(BaseLineGenerator, BaseContextBuilder) is False  # different ABCs
assert issubclass(LINE_GENERATOR_REGISTRY["LLMLineGenerator"], BaseLineGenerator)
print("[OK] Abstract base classes verified")

print("\n=== All smoke tests passed! ===")
