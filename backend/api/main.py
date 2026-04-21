from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import jobs, meshes

app = FastAPI(title="PicToMesh API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(meshes.router, prefix="/meshes", tags=["meshes"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
