FROM --platform=linux/amd64 python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Optional corporate CA certificates (e.g. Zscaler): drop *.crt into docker/certs/ (gitignored).
# UV_SYSTEM_CERTS makes uv trust the system store; the other two cover the model downloads
# rembg (requests) and huggingface-hub (httpx) do inside the worker.
COPY docker/certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV UV_SYSTEM_CERTS=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Native libraries opencv-python and open3d link against at import time.
# python:3.12-slim ships none of them, so both fail with ImportError without this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libx11-6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

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
