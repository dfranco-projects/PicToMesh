# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**PicToMesh** converts images into downloadable 3D meshes (GLB, OBJ, STL). Pipeline:
1. Upload → 2. SAM2 background removal → 3a. TripoSR (1 image) or 3b. DUSt3R + Open3D Poisson (2+ images) → 4. SSE progress stream → 5. React Three Fiber viewer + download.

Jobs run asynchronously via ARQ (Redis-backed queue). The browser polls job status and streams progress via SSE.

## Tech Stack

**Backend**
- **FastAPI** — async API framework (not Django; too heavy for this use case)
- **ARQ + Redis** — async job queue for reconstruction tasks (not Celery; lighter, async-native)
- **PyTorch** — ML runtime, auto-detects MPS / CUDA / CPU
- **SAM2** — background removal
- **TripoSR** — feed-forward single-image → mesh
- **DUSt3R** — neural SfM for multi-image (no camera calibration); vendored, CC BY-NC-SA 4.0 (non-commercial)
- **Depth Anything v2** — monocular depth fallback for CPU
- **Open3D + trimesh** — point cloud processing and mesh I/O

**Frontend**
- **React 19 + Vite 6** — app framework
- **React Three Fiber + @react-three/drei** — declarative Three.js (not raw Three.js)
- **shadcn/ui + Tailwind v4** — component library; Radix UI primitives, code lives in repo
- **TanStack Query v5** — server state, job polling, SSE
- **Zustand** — client state
- **react-dropzone** — file upload

**Infra**
- Docker Compose: `api` (FastAPI/uvicorn), `worker` (ARQ, same image), `redis`, `frontend` (nginx)
- **uv** — Python package manager; never edit `uv.lock` manually
- **ruff** — linting and formatting

## Architecture

```
Browser
  │ HTTP + SSE
  ▼
FastAPI (backend/api/)
  ├── POST /jobs              enqueue job → return job_id
  ├── GET  /jobs/{id}         poll status + result URL
  ├── GET  /jobs/{id}/stream  SSE progress
  └── GET  /meshes/{id}       serve GLB/OBJ/STL
  │
  └──► ARQ worker (backend/worker/)
         ├── segmentation task  (SAM2)
         ├── single_image task  (TripoSR)
         └── multi_image task   (MASt3R → Poisson)

src/pictomesh/             ML logic (imported by worker tasks)
  image_io/manager.py      ImageManager: load images from folder
  filtering/               CLIP encoder + Louvain outlier detection
  segmentation/            SAM2 wrapper (to implement)
  reconstruction/          DUSt3R (vendored under _vendor/dust3r) + Depth Anything fallback
  mesh/                    TripoSR (vendored under _vendor/tsr) + Open3D Poisson/BPA

frontend/src/
  components/Viewer3D      R3F scene with OrbitControls
  components/Dropzone      react-dropzone upload
  components/JobStatus     SSE progress bar
  lib/api.ts               TanStack Query hooks
  store/useStore.ts        Zustand
```

## Active Code

All modules are fully implemented. Notable callouts:

- `src/pictomesh/filtering/clip_encoder.py` — `CLIP.encode_images()` → `{name: feature_vector}`; has dead code after `return` (unreachable similarity block)
- `src/pictomesh/reconstruction/service.py` — `ReconstructionService`: `from_depth`, `from_rgbd`, `from_images`; `FlatDepthEstimator` (constant-depth fallback); `CameraIntrinsics.estimate` (60° FoV heuristic)
- `src/pictomesh/segmentation/service.py` — `SegmentationService` + `RembgSegmentor` (u2net, requires `--extra single-image`)
- `src/pictomesh/reconstruction/dust3r.py` — `Dust3rReconstructor`: 2+ photos + subject masks → coloured cloud in the first camera's frame, unit scale; `prepare_view` mirrors upstream load_images
- `src/pictomesh/_vendor/dust3r/`, `_vendor/croco/` — vendored DUSt3R (CC BY-NC-SA 4.0, non-commercial); patches listed in `_vendor/dust3r/__init__.py`
- `src/pictomesh/mesh/triposr.py` — `TripoSRReconstructor`: single RGBA image → vertex-coloured, y-up mesh; `prepare_image` mirrors upstream preprocessing
- `src/pictomesh/_vendor/tsr/` — vendored TripoSR (MIT) with PyMCubes marching cubes; excluded from ruff; local patches listed in its `__init__.py`
- `src/pictomesh/pipeline.py` — routes: < `image_threshold` (default 2) images → TripoSR on the first image (depth-lift + BPA if not installed); ≥ threshold → DUSt3R on the photos + Poisson (density-trimmed, coloured) if installed, else the same as below. Clouds are rotated from the OpenCV camera frame to glTF y-up before meshing

## Rollout Plan

Progress is tracked here. Each phase should be fully tested and committed before moving to the next.

### Phase 1 — Core ML Pipeline ✅
> Pure Python, no server. Each module has a service class + unit tests.

- [x] `ImageManager` — image loading, resize, metadata
- [x] `MeshService` — Poisson, Ball Pivoting, export (GLB/OBJ/STL)
- [x] `ReconstructionService` — depth map → point cloud; `FlatDepthEstimator` fallback; MASt3R protocol
- [x] `SegmentationService` — `RembgSegmentor` backend (real); SAM2 interface (protocol only)
- [x] `FilteringService` — CLIP encoder + Louvain graph clustering
- [x] `Pipeline` — top-level orchestrator with routing, depth fallback, E2E tests (`@pytest.mark.slow`)

### Phase 2 — Backend API ✅
> FastAPI + ARQ + Redis. Pipeline becomes a background job.

- [x] `backend/worker/tasks.py` — ARQ task definitions wrapping Pipeline
- [x] `POST /jobs` — accept images, validate, enqueue job, return `job_id`
- [x] `GET /jobs/{id}` — return job status + result URL when done
- [x] `GET /jobs/{id}/stream` — SSE real-time progress events (Redis pub/sub)
- [x] `GET /meshes/{id}/{filename}` — serve mesh file (GLB/OBJ/STL) from storage
- [x] `backend/models.py` — Pydantic schemas: `JobStatus`, `JobResponse`, `JobResult`, `ProgressEvent`

### Phase 3 — Frontend ✅
> React 19 + Vite + shadcn/ui + React Three Fiber.

- [x] Vite + React + TypeScript scaffold with Tailwind v4 + shadcn/ui
- [x] `Dropzone` component — react-dropzone, multi-file, format validation
- [x] `JobStatus` component — SSE progress + TanStack Query polling fallback
- [x] `Viewer3D` component — R3F scene with OrbitControls + Environment, loads GLB
- [x] `api.ts` — `submitJob`, `fetchJob`, `meshDownloadUrl`
- [x] `useStore.ts` — Zustand store (jobId, meshUrl)
- [x] Download button — format picker (GLB/OBJ/STL) pre-submission; download on completion

### Phase 4 — Infra ✅
> Containerised full stack runnable with `docker-compose up --build`.

- [x] `docker/backend.dockerfile` — python:3.12-slim, uv install, CPU torch on Linux
- [x] `docker/frontend.dockerfile` — Node 22 build + nginx serve
- [x] `docker/nginx/default.conf` — reverse proxy `/jobs`, `/meshes`, `/health`; SSE buffering off
- [x] `docker-compose.yml` — `api`, `worker`, `redis`, `frontend` services
- [x] `.env.example` — all required environment variables documented

### Phase 5 — Polish & Deploy
- [ ] End-to-end smoke test (upload → job → mesh download)
- [ ] CLI (`cli/`) for local batch processing without the server
- [ ] Deployment config for Fly.io / Render
- [ ] README demo GIF

## Commands

```bash
make install          # first-time setup (creates .venv via uv)
uv sync               # sync deps after pulling
make lint             # ruff check src/ tests/
make format           # ruff format + fix src/ tests/
make test             # pytest tests/ -v
make test-fast        # pytest -x (stop on first failure)
make dev              # local full stack, no docker (redis + api + worker + vite)
make api              # uvicorn on :8000 (local)
make worker           # arq worker (local)
make web              # vite dev server on :3000
```

### Add dependencies
```bash
uv add <package>                        # core dep
uv add --dev <package>                  # dev-only
uv add --optional <group> <package>     # optional group (single-image, segmentation)
```

### Optional model extras
```bash
uv sync --extra single-image    # rembg + TripoSR deps (einops, omegaconf, huggingface-hub, pymcubes, transformers)
uv sync --extra segmentation    # SAM2
uv sync --extra depth           # Depth Anything v2 (transformers)
uv sync --extra multi-view      # DUSt3R (roma, einops, huggingface-hub); CC BY-NC-SA 4.0, non-commercial
```

The worker's default pipeline needs `single-image` (rembg segmentor + TripoSR, ~1.7 GB of weights downloaded on first start), uses `multi-view` (DUSt3R, ~2.4 GB) for two or more photos when present, and uses `depth` when present, falling back to FlatDepthEstimator otherwise. The Docker image installs all three. After any `uv add`/`uv remove`, re-run `uv sync` with the extras: uv strips them from the venv otherwise.

## Python version

Pinned to **3.12** (open3d has no wheels for 3.13+). See `.python-version`.

## Git
Follow conventional commits, always lowercase. Minimal but informative messages.

Commit atomically, check local and staged changes, categorize and do multiple commits if there are many different types of changes (never add a feature and tests at the same time, first feature then tests).

Never add co-authored by Claude parts. Just main message and bullet points.


## Coding Guidelines

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
