import json
from pathlib import Path

from pydantic import BaseModel


class Storage:
    """Generic JSON storage for Pydantic models.

    Saves each entity as a separate JSON file in the given directory.
    Works with any Pydantic BaseModel subclass.
    """

    def __init__(self, directory: str):
        """Initialize storage with a target directory.

        Creates the directory if it doesn't exist.
        """
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, filename: str) -> Path:
        """Resolve a filename to a full path, appending .json if needed."""
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        return self._directory / filename

    def save(self, entity: BaseModel, filename: str) -> None:
        """Save a Pydantic model to a JSON file."""
        path = self._resolve_path(filename)
        path.write_text(json.dumps(entity.model_dump(), indent=2), encoding="utf-8")

    def load(self, filename: str, model_class: type[BaseModel]) -> BaseModel:
        """Load a Pydantic model from a JSON file.

        Raises FileNotFoundError if the file doesn't exist.
        """
        path = self._resolve_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return model_class.model_validate(data)

    def list_files(self) -> list[str]:
        """List all stored filenames (without .json extension)."""
        return [p.stem for p in self._directory.glob("*.json")]

    def delete(self, filename: str) -> None:
        """Delete a stored JSON file.

        Raises FileNotFoundError if the file doesn't exist.
        """
        path = self._resolve_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        path.unlink()
