import os
from pathlib import Path

import jinja2
from jinja2 import Environment, StrictUndefined

PROMPTS_DIR = os.path.dirname(__file__)


class PromptManager:
    """Renders Jinja2 templates from file paths with context.

    Templates are resolved via resolve_prompt_path(), which checks the
    language-specific directory first, then falls back to technical/.
    """

    def __init__(self, language: str = "en", prompts_dir: str = PROMPTS_DIR):
        self._env = Environment(
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._cache: dict[str, jinja2.Template] = {}
        self.language = language
        self._prompts_dir = prompts_dir

    def resolve_prompt_path(self, relative_path: str) -> str:
        """Resolve a prompt template path with language fallback.

        Checks: prompts/{language}/{relative_path}
        Falls back to: prompts/technical/{relative_path}

        Args:
            relative_path: Path relative to language dir, with .j2 extension.
                           e.g. "narrator/system.j2", "travel_destination/input.j2"

        Returns:
            Absolute path to the template file.

        Raises:
            FileNotFoundError: If not found in language dir or technical/.
        """
        base = self._prompts_dir

        primary = os.path.normpath(os.path.join(base, self.language, relative_path))
        if os.path.exists(primary):
            return primary

        fallback = os.path.normpath(os.path.join(base, "technical", relative_path))
        if os.path.exists(fallback):
            return fallback

        raise FileNotFoundError(
            f"Prompt template not found: {relative_path} "
            f"(checked paths: {primary}, {fallback})"
        )

    def render(self, path: str, **context) -> str:
        """Load a Jinja2 template from absolute path and render with context.

        Templates are cached after first load.

        Args:
            path: Absolute path to the .j2 template file.
            **context: Template variables.

        Returns:
            Rendered template string.

        Raises:
            jinja2.TemplateNotFound: If the template file does not exist.
        """
        if path not in self._cache:
            try:
                content = Path(path).read_text(encoding="utf-8")
            except FileNotFoundError:
                raise jinja2.TemplateNotFound(path)
            self._cache[path] = self._env.from_string(content)
        return self._cache[path].render(**context)
