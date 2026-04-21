from fastapi import APIRouter

router = APIRouter()


@router.get("/{mesh_id}")
async def get_mesh(mesh_id: str) -> dict:
    # TODO: serve mesh file from storage
    return {"mesh_id": mesh_id}
