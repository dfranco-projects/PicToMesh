from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
from arq import ArqRedis

from backend.config import WORKER_ERROR_KEY, WORKER_HEARTBEAT_KEY, settings
from backend.models import JobStatus, ProgressEvent, WorkerError, WorkerState, WorkerStatus
from backend.worker.diagnostics import describe_startup_failure
from pictomesh.filtering.service import FilteringService
from pictomesh.mesh.service import MeshService
from pictomesh.mesh.triposr import TripoSRReconstructor
from pictomesh.pipeline import Pipeline
from pictomesh.reconstruction.da3 import Da3Reconstructor
from pictomesh.reconstruction.service import (
    DepthAnythingEstimator,
    FlatDepthEstimator,
    ReconstructionService,
)
from pictomesh.segmentation.service import RembgSegmentor, SegmentationService

logger = logging.getLogger(__name__)

_MAX_SIDE = 512

_HEARTBEAT_SECONDS = 10
# Well above the refresh interval: GIL-holding steps such as open3d's Poisson
# freeze the event loop for 10 s or more, and a busy worker must not look dead.
_HEARTBEAT_TTL_SECONDS = 60


def _load_image(path: str) -> np.ndarray | None:
    """Read an image and cap its longest side at _MAX_SIDE.

    Full-resolution photos lift to point clouds with millions of points,
    which Ball Pivoting cannot mesh within the job timeout.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = _MAX_SIDE / max(h, w)
    if scale < 1.0:
        size = (round(w * scale), round(h * scale))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _build_pipeline() -> Pipeline:
    """Construct a default pipeline with real services."""
    seg = SegmentationService(RembgSegmentor())
    from pictomesh.filtering.clip_encoder import CLIPEncoder

    flt = FilteringService(CLIPEncoder())
    rec = ReconstructionService()
    msh = MeshService()
    try:
        dep = DepthAnythingEstimator()
    except ImportError:
        dep = FlatDepthEstimator()
    try:
        single = TripoSRReconstructor()
    except ImportError:
        single = None
    try:
        multi = Da3Reconstructor()
    except ImportError:
        multi = None
    return Pipeline(
        seg,
        flt,
        rec,
        msh,
        dep,
        reconstructor=multi,
        image_threshold=settings.image_threshold,
        single_image_reconstructor=single,
    )


async def _publish(redis: ArqRedis, event: ProgressEvent) -> None:
    channel = f"job:{event.job_id}:progress"
    await redis.publish(channel, event.model_dump_json())


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _announce(ctx: dict) -> None:
    status: WorkerStatus = ctx["status"]
    await ctx["redis"].set(
        WORKER_HEARTBEAT_KEY, status.model_dump_json(), ex=_HEARTBEAT_TTL_SECONDS
    )


async def _heartbeat(ctx: dict) -> None:
    while True:
        await _announce(ctx)
        await asyncio.sleep(_HEARTBEAT_SECONDS)


async def startup(ctx: dict) -> None:
    """Build the pipeline once per worker process, reporting progress as it goes.

    Model weights load here rather than per job, where they otherwise dominate
    the runtime of every single job. A first start downloads several GB, so the
    heartbeat runs throughout and the API can tell a loading worker from a dead one.
    """
    redis: ArqRedis = ctx["redis"]
    ctx["status"] = WorkerStatus(state=WorkerState.loading, since=_now())
    ctx["heartbeat"] = asyncio.create_task(_heartbeat(ctx))
    try:
        ctx["pipeline"] = await asyncio.to_thread(_build_pipeline)
    except Exception as exc:
        message, detail = await asyncio.to_thread(describe_startup_failure, exc)
        logger.exception("Worker startup failed. %s", message)
        error = WorkerError(message=message, detail=detail, at=_now())
        await redis.set(WORKER_ERROR_KEY, error.model_dump_json())
        # Already logged; exit without Python printing the traceback a second time.
        raise SystemExit(1) from None
    ctx["status"] = WorkerStatus(state=WorkerState.ready, since=_now())
    await _announce(ctx)
    await redis.delete(WORKER_ERROR_KEY)
    logger.info("Worker ready")


async def shutdown(ctx: dict) -> None:
    """Stop announcing readiness. arq also calls this after a failed startup."""
    if heartbeat := ctx.get("heartbeat"):
        heartbeat.cancel()
    await ctx["redis"].delete(WORKER_HEARTBEAT_KEY)


async def process_images(
    ctx: dict,
    job_id: str,
    image_paths: list[str],
    fmt: str = "glb",
) -> dict:
    """ARQ task: run the pipeline and return the output mesh path."""
    redis: ArqRedis = ctx["redis"]

    await _publish(
        redis,
        ProgressEvent(
            job_id=job_id, status=JobStatus.in_progress, message="Loading images", progress=0.05
        ),
    )

    images: list[np.ndarray] = []
    for p in image_paths:
        img = _load_image(p)
        if img is not None:
            images.append(img)

    if not images:
        await _publish(
            redis,
            ProgressEvent(
                job_id=job_id, status=JobStatus.failed, message="No readable images", progress=0.0
            ),
        )
        return {"status": JobStatus.failed, "error": "No readable images"}

    await _publish(
        redis,
        ProgressEvent(
            job_id=job_id, status=JobStatus.in_progress, message="Running pipeline", progress=0.2
        ),
    )

    output_dir = settings.media_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mesh"

    try:
        # Off the event loop, so the heartbeat keeps beating and job_timeout can fire.
        out_file = await asyncio.to_thread(ctx["pipeline"].run, images, output_path, fmt=fmt)
    except Exception as e:
        await _publish(
            redis,
            ProgressEvent(job_id=job_id, status=JobStatus.failed, message=str(e), progress=0.0),
        )
        return {"status": JobStatus.failed, "error": str(e)}

    mesh_url = f"/meshes/{job_id}/mesh.{fmt}"
    await _publish(
        redis, ProgressEvent(job_id=job_id, status=JobStatus.complete, message="Done", progress=1.0)
    )

    return {"status": JobStatus.complete, "mesh_url": mesh_url, "output_path": str(out_file)}
