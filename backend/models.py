from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class JobStatus(StrEnum):
    queued = "queued"
    in_progress = "in_progress"
    complete = "complete"
    failed = "failed"


class JobResponse(BaseModel):
    """Returned immediately after a job is enqueued (202)."""

    job_id: str
    status: JobStatus = JobStatus.queued


class JobResult(BaseModel):
    """Full job state returned by GET /jobs/{id}."""

    job_id: str
    status: JobStatus
    mesh_url: str | None = None
    error: str | None = None


class ProgressEvent(BaseModel):
    """Payload published to Redis pub/sub for SSE streaming."""

    job_id: str
    status: JobStatus
    message: str = ""
    progress: float = 0.0  # 0.0–1.0
