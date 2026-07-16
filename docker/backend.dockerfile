FROM --platform=linux/amd64 python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies — Linux resolver picks torch+cpu automatically (see pyproject.toml)
# Extras: single-image ships rembg (default segmentor), depth ships Depth Anything v2
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra single-image --extra depth

# Copy source
COPY src/ ./src/
COPY backend/ ./backend/

# Install the project itself
RUN uv sync --frozen --extra single-image --extra depth

ENV PATH="/app/.venv/bin:$PATH"
