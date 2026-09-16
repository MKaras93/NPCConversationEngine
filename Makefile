# Example Makefile with most often used commands.


#  loads .env file so that env variables are available in commands
-include .env
export

init-pre-commit:
	pip3 install pre-commit
	pre-commit install

install-uv-unix:
	curl -LsSf https://astral.sh/uv/install.sh | sh

install-uv-win:
	powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

install-dependencies:
	@which uv > /dev/null 2>&1 || (echo "Error: uv is not installed. Please run 'make install-uv-unix' (Linux/macOS) or 'make install-uv-win' (Windows)" && exit 1)
	uv sync --extra dev

test:
	uv run pytest

test-integration:
	uv run pytest -m integration -v -s

test-package:
	powershell -c "if (Test-Path dist) { Remove-Item -Recurse -Force dist }"
	uv cache clean npc-conversation-engine
	uv build
	@echo "--- Testing Wheel ---"
	powershell -c "uv run --with $$(Get-ChildItem dist\*.whl | Select-Object -First 1 -ExpandProperty FullName) scripts/test_as_dependency.py"
	@echo "--- Testing Source Distribution (sdist) ---"
	powershell -c "uv run --with $$(Get-ChildItem dist\*.tar.gz | Select-Object -First 1 -ExpandProperty FullName) scripts/test_as_dependency.py"

