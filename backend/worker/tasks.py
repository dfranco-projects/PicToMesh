from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from arq import ArqRedis

from backend.config import settings
from backend.models import JobStatus, ProgressEvent
from pictomesh.filtering.service import FilteringService
from pictomesh.mesh.service import MeshService
from pictomesh.pipeline import Pipeline
from pictomesh.reconstruction.service import FlatDepthEstimator, ReconstructionService
from pictomesh.segmentation.service import RembgSegmentor, SegmentationService


def _build_pipeline() -> Pipeline:
    """Construct a default pipeline with real services."""
    seg = SegmentationService(RembgSegmentor())
    from pictomesh.filtering.clip_encoder import CLIPEncoder

    flt = FilteringService(CLIPEncoder())
    rec = ReconstructionService()
    msh = MeshService()
    dep = FlatDepthEstimator()
    return Pipeline(seg, flt, rec, msh, dep, image_threshold=settings.image_threshold)


async def _publish(redis: ArqRedis, event: ProgressEvent) -> None:
    channel = f"job:{event.job_id}:progress"
    await redis.publish(channel, event.model_dump_json())


async def process_images(
    ctx: dict,
    job_id: str,
    image_paths: list[str],
    fmt: str = "glb",
) -> dict:
    """ARQ task: run the pipeline and return the output mesh path."""
    redis: ArqRedis = ctx["redis"]

    await _publish(redis, ProgressEvent(job_id=job_id, status=JobStatus.in_progress, message="Loading images", progress=0.05))

    images: list[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)

    if not images:
        await _publish(redis, ProgressEvent(job_id=job_id, status=JobStatus.failed, message="No readable images", progress=0.0))
        return {"status": JobStatus.failed, "error": "No readable images"}

    await _publish(redis, ProgressEvent(job_id=job_id, status=JobStatus.in_progress, message="Running pipeline", progress=0.2))

    output_dir = settings.media_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mesh"

    pipeline = _build_pipeline()
    out_file = pipeline.run(images, output_path, fmt=fmt)

    mesh_url = f"/meshes/{job_id}/mesh.{fmt}"
    await _publish(redis, ProgressEvent(job_id=job_id, status=JobStatus.complete, message="Done", progress=1.0))

    return {"status": JobStatus.complete, "mesh_url": mesh_url, "output_path": str(out_file)}
