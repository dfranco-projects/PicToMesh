FROM --platform=linux/amd64 python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies (cached layer)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Copy source
COPY src/ ./src/
COPY backend/ ./backend/

# Install the project itself
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
