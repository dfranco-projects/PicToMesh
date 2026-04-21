from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import settings

router = APIRouter()

_MEDIA_TYPES = {
    "glb": "model/gltf-binary",
    "obj": "text/plain",
    "stl": "model/stl",
}


@router.get("/{job_id}/{filename}")
async def get_mesh(job_id: str, filename: str) -> FileResponse:
    mesh_path = settings.media_dir / job_id / filename
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="Mesh not found.")

    suffix = mesh_path.suffix.lstrip(".")
    media_type = _MEDIA_TYPES.get(suffix, "application/octet-stream")
    return FileResponse(str(mesh_path), media_type=media_type, filename=filename)
