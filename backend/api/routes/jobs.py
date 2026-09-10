from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Literal

import aiofiles
from arq import ArqRedis
from arq.jobs import Job
from arq.jobs import JobStatus as ArqJobStatus
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.models import JobResponse, JobResult, JobStatus

router = APIRouter()

MAX_FILES = 50
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # per file
_CHUNK = 1024 * 1024
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


# ── helpers ───────────────────────────────────────────────────────────────────


def _upload_name(index: int, filename: str | None) -> str:
    """Build the on-disk name ourselves.

    The multipart filename is attacker controlled: a value like
    ``../../../../etc/cron.d/x`` would otherwise escape the upload directory and
    write anywhere the API process can reach. Only the extension is taken from
    the client, and only if it is one we accept.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        suffix = ".jpg"
    return f"{index:03d}{suffix}"


async def _save_upload(file: UploadFile, dest: Path) -> None:
    """Stream *file* to *dest*, rejecting anything over MAX_UPLOAD_BYTES.

    Streamed rather than read() in one go so a large upload cannot be buffered
    into memory before the size is known.
    """
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
    arq: ArqRedis = request.app.state.arq
    job = Job(job_id, arq)
    arq_status = await job.status()

    if arq_status == ArqJobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found.")

    job_status = _arq_to_job_status(arq_status)
    mesh_url: str | None = None
    error: str | None = None

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


@router.get("/{job_id}/stream")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """SSE endpoint — streams progress events published by the ARQ task."""
    redis = request.app.state.redis
    channel = f"job:{job_id}:progress"

    async def event_generator():
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                data = message["data"]
                yield f"data: {data}\n\n"
                parsed = json.loads(data)
                if parsed.get("status") in (JobStatus.complete, JobStatus.failed):
                    break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
