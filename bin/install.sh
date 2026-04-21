#!/usr/bin/env bash
#
# Sets up the PicToMesh development environment.
# Requires Python 3.10+ on the PATH.
#
# Usage:
#   bash bin/install.sh          # first-time setup
#   make install                 # idiomatic shortcut (uses uv sync if .venv exists)

set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_DIR="${PROJECT_ROOT}/.venv"

readonly GREEN=$'\033[0;32m'
readonly YELLOW=$'\033[1;33m'
readonly NC=$'\033[0m'

log_info()    { echo -e "${YELLOW}INFO:${NC} $1"; }
log_success() { echo -e "${GREEN}SUCCESS:${NC} $1"; }

ensure_uv() {
    if ! command -v uv &> /dev/null; then
        log_info "uv not found — installing via pip..."
        pip install -q uv
    fi
}

main() {
    log_info "Setting up PicToMesh dev environment..."

    ensure_uv

    cd "${PROJECT_ROOT}"

    log_info "Creating virtual environment and syncing dependencies..."
    uv sync

    echo ""
    log_success "Done. Activate with: . .venv/bin/activate"
}

main "$@"
