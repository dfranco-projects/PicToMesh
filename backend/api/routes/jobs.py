from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Literal

import aiofiles
from arq import ArqRedis
from arq.jobs import Job
from arq.jobs import JobStatus as ArqJobStatus
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis

from backend.api.worker_status import read_worker_status, unavailable_reason
from backend.config import settings
from backend.models import JobResponse, JobResult, JobStatus, ProgressEvent

router = APIRouter()

MAX_FILES = 50
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # per file
_CHUNK = 1024 * 1024
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

_TERMINAL = (JobStatus.complete, JobStatus.failed)

SSE_POLL_SECONDS = 1.0
SSE_HEARTBEAT_SECONDS = 15.0
SSE_MAX_SECONDS = 1860.0  # just over the worker's job_timeout


# ── helpers ───────────────────────────────────────────────────────────────────


def _upload_name(index: int, filename: str | None) -> str:
    """Name the file ourselves — the client's filename can traverse out of the dir."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        suffix = ".jpg"
    return f"{index:03d}{suffix}"


async def _save_upload(file: UploadFile, dest: Path) -> None:
    """Stream *file* to *dest*, rejecting anything over MAX_UPLOAD_BYTES."""
    written = 0
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(_CHUNK):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Each file must be under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            await out.write(chunk)


def _arq_to_job_status(arq_status: ArqJobStatus) -> JobStatus:
    return {
        ArqJobStatus.queued: JobStatus.queued,
        ArqJobStatus.in_progress: JobStatus.in_progress,
        ArqJobStatus.complete: JobStatus.complete,
        ArqJobStatus.deferred: JobStatus.queued,
    }.get(arq_status, JobStatus.failed)


def _is_terminal(data: str) -> bool:
    """True if a published progress payload says the job is over."""
    try:
        return json.loads(data).get("status") in _TERMINAL
    except (TypeError, ValueError, AttributeError):
        return False


async def _job_result(job_id: str, arq: ArqRedis, redis: Redis) -> JobResult | None:
    """Current state of *job_id*, or None if arq has never heard of it."""
    job = Job(job_id, arq)
    arq_status = await job.status()
    if arq_status == ArqJobStatus.not_found:
        return None

    job_status = _arq_to_job_status(arq_status)
    mesh_url: str | None = None
    error: str | None = None

    # Nothing would ever pick it up. In-progress jobs are owned by a worker already.
    if job_status == JobStatus.queued:
        reason = unavailable_reason(await read_worker_status(redis))
        if reason:
            return JobResult(job_id=job_id, status=JobStatus.failed, error=reason)

    if arq_status == ArqJobStatus.complete:
        try:
            result = await job.result()
        except Exception as e:
            return JobResult(job_id=job_id, status=JobStatus.failed, error=str(e))
        if isinstance(result, dict):
            mesh_url = result.get("mesh_url")
            error = result.get("error")
            if error:
                job_status = JobStatus.failed

    return JobResult(job_id=job_id, status=job_status, mesh_url=mesh_url, error=error)


async def _terminal_event(job_id: str, arq: ArqRedis, redis: Redis) -> ProgressEvent | None:
    """A synthesised final event if the job is already over, else None."""
    result = await _job_result(job_id, arq, redis)
    if result is None or result.status not in _TERMINAL:
        return None
    failed = result.status == JobStatus.failed
    return ProgressEvent(
        job_id=job_id,
        status=result.status,
        message=result.error or ("Processing failed" if failed else "Done"),
        progress=0.0 if failed else 1.0,
    )


# ── routes ────────────────────────────────────────────────────────────────────


@router.post("", status_code=202, response_model=JobResponse)
async def create_job(
    request: Request,
    files: list[UploadFile],
    fmt: Literal["glb", "obj", "stl"] = "glb",
) -> JobResponse:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=413, detail=f"At most {MAX_FILES} files per job.")

    # A loading worker will get to the job; a failed or missing one never will.
    reason = unavailable_reason(await read_worker_status(request.app.state.redis))
    if reason:
        raise HTTPException(status_code=503, detail=reason)

    job_id = str(uuid.uuid4())
    job_dir = settings.media_dir / job_id
    upload_dir = job_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    try:
        for index, file in enumerate(files):
            dest = upload_dir / _upload_name(index, file.filename)
            await _save_upload(file, dest)
            saved_paths.append(str(dest))
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    arq: ArqRedis = request.app.state.arq
    await arq.enqueue_job("process_images", job_id, saved_paths, fmt, _job_id=job_id)

    return JobResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobResult)
async def get_job(job_id: str, request: Request) -> JobResult:
    result = await _job_result(job_id, request.app.state.arq, request.app.state.redis)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return result


@router.get("/{job_id}/stream")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """SSE endpoint — streams progress events published by the ARQ task."""
    redis = request.app.state.redis
    arq: ArqRedis = request.app.state.arq
    channel = f"job:{job_id}:progress"

    if await _job_result(job_id, arq, redis) is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        started = last_beat = time.monotonic()
        try:
            # Subscribed first, so this only has to catch a job that finished earlier.
            settled = await _terminal_event(job_id, arq, redis)
            if settled is not None:
                yield f"data: {settled.model_dump_json()}\n\n"
                return

            while True:
                now = time.monotonic()
                if now - started > SSE_MAX_SECONDS or await request.is_disconnected():
                    return

                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=SSE_POLL_SECONDS
                )
                if message is None:
                    if now - last_beat >= SSE_HEARTBEAT_SECONDS:
                        last_beat = now
                        yield ": keepalive\n\n"
                    continue

                data = message["data"]
                yield f"data: {data}\n\n"
                if _is_terminal(data):
                    return
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
