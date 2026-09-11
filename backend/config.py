from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PICTOMESH_", env_file=".env", extra="ignore")

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Storage
    media_dir: Path = Path("media")

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Pipeline
    image_threshold: int = 2


settings = Settings()

# A live worker's WorkerStatus (loading or ready), refreshed while it lives; expires if it dies.
WORKER_HEARTBEAT_KEY = "pictomesh:worker"
# The last failed worker startup (WorkerError); kept until a worker becomes ready.
WORKER_ERROR_KEY = "pictomesh:worker:error"
