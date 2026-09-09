from __future__ import annotations

import cv2
import numpy as np
from arq import ArqRedis

from backend.config import settings
from backend.models import JobStatus, ProgressEvent
from pictomesh.filtering.service import FilteringService
from pictomesh.mesh.service import MeshService
from pictomesh.mesh.triposr import TripoSRReconstructor
from pictomesh.pipeline import Pipeline
from pictomesh.reconstruction.service import (
    DepthAnythingEstimator,
    FlatDepthEstimator,
    ReconstructionService,
)
from pictomesh.segmentation.service import RembgSegmentor, SegmentationService

_MAX_SIDE = 512


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
    return Pipeline(
        seg,
        flt,
        rec,
        msh,
        dep,
        image_threshold=settings.image_threshold,
        single_image_reconstructor=single,
    )


async def _publish(redis: ArqRedis, event: ProgressEvent) -> None:
    channel = f"job:{event.job_id}:progress"
    await redis.publish(channel, event.model_dump_json())


async def startup(ctx: dict) -> None:
    """Build the pipeline once per worker process.

    Model weights (rembg u2net, Depth Anything) load here rather than per job,
    where they otherwise dominate the runtime of every single job.
    """
    ctx["pipeline"] = _build_pipeline()


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
        out_file = ctx["pipeline"].run(images, output_path, fmt=fmt)
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
