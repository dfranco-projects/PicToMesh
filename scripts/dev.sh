#!/usr/bin/env bash
#
# Runs the full PicToMesh stack locally, without Docker:
# Redis + FastAPI (:8000) + ARQ worker + Vite dev server (:3000).
#
# Usage:
#   make dev                 # idiomatic shortcut
#   bash scripts/dev.sh

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export API_PORT="${API_PORT:-8000}"

if [ ! -d .venv ]; then
    echo "No .venv found. Run 'make install' first." >&2
    exit 1
fi

if ! uv run python -c "import rembg" > /dev/null 2>&1; then
    echo "Model extras are not installed. Run: uv sync --extra single-image --extra depth --extra multi-view" >&2
    exit 1
fi

pids=()
cleanup() {
    trap - INT TERM EXIT
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if ! redis-cli ping > /dev/null 2>&1; then
    if command -v redis-server > /dev/null 2>&1; then
        echo "Starting redis-server on :6379..."
        redis-server --port 6379 --save "" --appendonly no > /dev/null 2>&1 &
        pids+=("$!")
        sleep 1
    else
        echo "Redis is not running and redis-server is not installed." >&2
        echo "  macOS:         brew install redis" >&2
        echo "  Debian/Ubuntu: sudo apt install redis-server" >&2
        exit 1
    fi
fi

if [ ! -d frontend/node_modules ]; then
    echo "Installing frontend dependencies..."
    (cd frontend && npm install)
fi

uv run uvicorn backend.api.main:app --reload --port "${API_PORT}" &
pids+=("$!")

uv run arq backend.worker.settings.WorkerSettings &
pids+=("$!")

(cd frontend && npm run dev) &
pids+=("$!")

echo ""
echo "PicToMesh is up:"
echo "  frontend  http://localhost:3000"
echo "  api       http://localhost:${API_PORT}"
echo "Press Ctrl-C to stop everything."

wait
