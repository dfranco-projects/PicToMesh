from fastapi import APIRouter

router = APIRouter()


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    # TODO: query ARQ job status from Redis
    return {"job_id": job_id, "status": "pending"}
