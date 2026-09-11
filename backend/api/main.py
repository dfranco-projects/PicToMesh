from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import jobs, meshes
from backend.api.worker_status import read_worker_status
from backend.config import settings
from backend.models import WorkerStatus


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
    app.state.arq = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    yield
    await app.state.arq.aclose()
    await app.state.redis.aclose()


app = FastAPI(title="PicToMesh API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(meshes.router, prefix="/meshes", tags=["meshes"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/worker", response_model=WorkerStatus)
async def worker_health(request: Request) -> WorkerStatus:
    return await read_worker_status(request.app.state.redis)
