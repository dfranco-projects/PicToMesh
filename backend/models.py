from __future__ import annotations

from datetime import datetime
from enum import StrEnum

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


class WorkerState(StrEnum):
    loading = "loading"
    ready = "ready"
    failed = "failed"
    offline = "offline"


class WorkerError(BaseModel):
    """Why the last worker startup failed."""

    message: str  # one sentence for the UI
    detail: str  # technical, for logs and bug reports
    at: datetime


class WorkerStatus(BaseModel):
    """Worker readiness, returned by GET /health/worker."""

    state: WorkerState
    since: datetime | None = None
    error: WorkerError | None = None  # the last startup failure, until a worker is ready
