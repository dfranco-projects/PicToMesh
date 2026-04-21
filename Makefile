.PHONY: install uninstall lint format test build clean
.DEFAULT_GOAL := install

# ── Environment ──────────────────────────────────────────────────────────────

install:
	@echo "\033[1;33m=> Installing dev environment...\033[0m"
	@if [ -d ".venv" ]; then \
		echo "Virtual environment exists — syncing dependencies..."; \
		uv sync; \
	else \
		bash bin/install.sh; \
	fi
	@echo "\033[0;32m✨ Ready. Run: source .venv/bin/activate\033[0m"

uninstall:
	@echo "\033[1;31m=> Removing virtual environment...\033[0m"
	@rm -rf .venv .uv.cache
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[0;32m✅ Cleaned.\033[0m"

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -x --no-header -q
