.PHONY: install uninstall lint format test test-fast api worker web local docker down
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
	@uv run --no-sync pre-commit install
	@echo "\033[0;32m✨ Ready. Run: source .venv/bin/activate\033[0m"

uninstall:
	@echo "\033[1;31m=> Removing virtual environment...\033[0m"
	@rm -rf .venv .uv.cache
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[0;32m✅ Cleaned.\033[0m"

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/ backend/

format:
	uv run ruff format src/ tests/ backend/
	uv run ruff check --fix src/ tests/ backend/

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -x --no-header -q

# ── Local run (no Docker) ─────────────────────────────────────────────────────

API_PORT ?= 8000
EXTRAS := --extra single-image --extra depth --extra multi-view

api:
	uv run uvicorn backend.api.main:app --reload --port $(API_PORT) --log-config backend/log_config.json

worker:
	uv run arq backend.worker.settings.WorkerSettings --custom-log-dict backend.log_config.LOG_CONFIG

web:
	@test -d frontend/node_modules || (cd frontend && npm install)
	cd frontend && npm run dev

local:
	@command -v uv > /dev/null || { echo "uv is not installed. macOS: brew install uv node redis"; exit 1; }
	uv sync $(EXTRAS)
	@bash scripts/dev.sh

# ── Docker ────────────────────────────────────────────────────────────────────

docker:
	docker compose up --build

down:
	docker compose down
