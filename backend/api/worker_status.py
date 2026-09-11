from __future__ import annotations

from redis.asyncio import Redis

from backend.config import WORKER_ERROR_KEY, WORKER_HEARTBEAT_KEY
from backend.models import WorkerError, WorkerState, WorkerStatus


async def read_worker_status(redis: Redis) -> WorkerStatus:
    """What the worker is doing, from its heartbeat and its last startup failure."""
    heartbeat, error = await redis.mget(WORKER_HEARTBEAT_KEY, WORKER_ERROR_KEY)
    last_error = WorkerError.model_validate_json(error) if error else None
    if heartbeat:
        status = WorkerStatus.model_validate_json(heartbeat)
        if status.state == WorkerState.ready:
            return status
        # Loading again after a failure: keep showing why the last attempt failed.
        return status.model_copy(update={"error": last_error})
    if last_error:
        return WorkerStatus(state=WorkerState.failed, since=last_error.at, error=last_error)
    return WorkerStatus(state=WorkerState.offline)


def unavailable_reason(status: WorkerStatus) -> str | None:
    """A sentence explaining why jobs can't run, or None if they can."""
    if status.state == WorkerState.failed and status.error:
        return f"{status.error.message} Restart PicToMesh once that's fixed."
    if status.state == WorkerState.offline:
        return "The processing worker isn't running. Restart PicToMesh."
    return None
