from __future__ import annotations

import json
import uuid

import aiofiles
from arq import ArqRedis
from arq.jobs import Job
from arq.jobs import JobStatus as ArqJobStatus
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.models import JobResponse, JobResult, JobStatus

router = APIRouter()


# ── helpers ───────────────────────────────────────────────────────────────────


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
    fmt: str = "glb",
) -> JobResponse:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required.")

    job_id = str(uuid.uuid4())
    upload_dir = settings.media_dir / job_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for file in files:
        dest = upload_dir / (file.filename or f"{uuid.uuid4()}.jpg")
        async with aiofiles.open(dest, "wb") as f:
            await f.write(await file.read())
        saved_paths.append(str(dest))

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
