# PicToMesh – Open-Source 3D Meshes From Your Images

**PicToMesh** is a full-stack, open-source tool that lets you drag-and-drop images and receive a downloadable 3D mesh of the object — all in the browser, with no signups, no API keys, and no black-box models.

---

## What It Does

- Drag and drop one or many images of an object
- Automatic background removal before reconstruction
- Feed-forward mesh generation from a single image (no photogrammetry)
- Neural multi-view reconstruction from many images (no camera calibration)
- Interactive 3D viewer in-browser (rotate, zoom, pan)
- Download the mesh in GLB, OBJ, or STL
- Real-time progress updates during reconstruction
- Fully open-source — no API keys, no paywalls, just local compute

---

## How It Works

| Scenario        | Pipeline                                                                           |
|-----------------|------------------------------------------------------------------------------------|
| Single image    | SAM2 background removal → **TripoSR** feed-forward mesh (~5s on GPU)              |
| 2–4 images      | SAM2 → TripoSR with multi-view hint                                                |
| 5+ images       | SAM2 → **MASt3R** neural SfM (no calibration) → Open3D Poisson mesh              |
| CPU / no GPU    | SAM2 → **Depth Anything v2** monocular depth → point cloud lift                   |

Jobs are processed asynchronously — the browser streams real-time progress via SSE.

---

## Tech Stack

**Backend**

| Layer          | Choice                        | Why                                                    |
|----------------|-------------------------------|--------------------------------------------------------|
| API framework  | FastAPI + uvicorn             | Async-native, lightweight, excellent for ML APIs       |
| Job queue      | ARQ + Redis                   | Async Redis queue; pairs naturally with FastAPI        |
| Progress       | SSE (Server-Sent Events)      | Simple one-way streaming, no WebSocket overhead        |
| ML runtime     | PyTorch (MPS / CUDA / CPU)    | Auto-detects best device                               |
| 3D engine      | Open3D + trimesh              | Point cloud processing and mesh I/O                   |
| Segmentation   | SAM2                          | State-of-the-art open-source segmentation              |
| Single-image   | TripoSR                       | Feed-forward image→mesh transformer, ~5s on GPU       |
| Multi-image    | MASt3R                        | Neural SfM, no camera calibration required             |
| Depth fallback | Depth Anything v2             | Monocular depth, runs on CPU                           |

**Frontend**

| Layer          | Choice                        | Why                                                    |
|----------------|-------------------------------|--------------------------------------------------------|
| Framework      | React 19 + Vite 6             | Fast builds, great DX                                  |
| 3D viewer      | React Three Fiber + drei      | Declarative Three.js for React                         |
| UI components  | shadcn/ui + Tailwind v4       | Radix primitives, accessible, code lives in your repo  |
| Server state   | TanStack Query v5             | Job polling, caching, SSE integration                  |
| Client state   | Zustand                       | Lightweight, no boilerplate                            |
| File upload    | react-dropzone                | Drag-and-drop with validation                          |

**Infrastructure**

| Layer          | Choice                        |
|----------------|-------------------------------|
| Container      | Docker + Docker Compose       |
| Reverse proxy  | nginx                         |
| Queue backend  | Redis 7                       |
| Package mgr    | uv                            |
| Linting        | ruff                          |

---

## Architecture

```
Browser (React + R3F + shadcn/ui)
  │  HTTP + SSE
  ▼
FastAPI  ──────────────────────────────────────────────
  ├── POST /jobs          enqueue reconstruction job
  ├── GET  /jobs/{id}     poll status + result URL
  ├── GET  /jobs/{id}/stream   SSE progress stream
  └── GET  /meshes/{id}  download GLB/OBJ/STL
  │
  └──► ARQ worker (Redis queue)
         ├── SAM2 segmentation
         ├── TripoSR   (1–4 images)
         └── MASt3R + Poisson  (5+ images)
```

---

## Project Structure

```
PicToMesh/
├── src/pictomesh/          # Core ML package
│   ├── image_io/           # Image loading (ImageManager)
│   ├── filtering/          # CLIP embedding + outlier detection
│   ├── segmentation/       # SAM2 background removal
│   ├── reconstruction/     # MASt3R + Depth Anything fallback
│   └── mesh/               # TripoSR + Open3D Poisson
│
├── backend/                # FastAPI app + ARQ worker
│   ├── api/                # Routes (jobs, meshes)
│   ├── worker/             # ARQ task definitions
│   └── models.py           # Pydantic / SQLModel schemas
│
├── frontend/               # React/Vite app
│   ├── src/
│   │   ├── components/     # Viewer3D (R3F), Dropzone, JobStatus, shadcn/ui
│   │   ├── lib/            # API client (submitJob, fetchJob)
│   │   └── store/          # Zustand store (jobId, meshUrl)
│   └── package.json
│
├── docker/                 # Dockerfiles + nginx config
├── demos/                  # Jupyter notebooks
├── tests/
│   └── assets/             # Test images (chairs, cats)
├── docs/
├── bin/
├── pyproject.toml
├── Makefile
└── docker-compose.yml
```

---

## Run Locally

### With Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine with the Compose v2 plugin. The command is `docker compose` (with a space); the legacy `docker-compose` binary is not required.

```bash
git clone https://github.com/dfranco-projects/PicToMesh.git
cd PicToMesh
docker compose up --build
```

Open `http://localhost:3000`. The API is at `http://localhost:8000`.

A `.env` file is optional. Defaults work out of the box; copy `.env.example` to `.env` only if you want to override them.

The worker downloads model weights on its first start (about 2.5 GB across TripoSR, CLIP, Depth Anything v2 and rembg) and keeps them in the `model_cache` volume, so rebuilds and restarts do not fetch them again.

Behind a corporate TLS proxy such as Zscaler the builds fail inside the containers with certificate errors (`uv` prints a hint about `--system-certs`). Drop your root CA as `docker/certs/<name>.crt` before building: both images install everything in that folder into their system trust store, and the folder is gitignored.

### Without Docker

Requires [uv](https://docs.astral.sh/uv/), Node 22+, and Redis. uv fetches the pinned Python 3.12 automatically.

```bash
# macOS
brew install uv node redis

# Debian/Ubuntu
sudo apt install redis-server nodejs npm && curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/dfranco-projects/PicToMesh.git
cd PicToMesh
make install                                  # create .venv, sync deps, install git hooks
uv sync --extra single-image --extra depth    # model deps (rembg, Depth Anything v2)
make dev                                      # starts redis, api, worker, frontend
```

Open `http://localhost:5173`.

Prefer separate terminals? Run `make api`, `make worker`, and `make web` individually (needs a running Redis).

If port 8000 is already taken, pick another one with `API_PORT=8010 make dev`; the Vite proxy follows automatically.

The worker downloads model weights on its first start (TripoSR ~1.7 GB, plus CLIP, Depth Anything v2 Small and u2net), so that start is slow once; later runs reuse the cache.

Commits run `ruff check --fix` and `ruff format` through pre-commit. If a hook rewrites a file or an unfixable lint error remains, the commit is rejected: re-stage and commit again.

---

## Roadmap

- [x] FastAPI backend with ARQ job queue
- [x] Segmentation pipeline (rembg / SAM2 protocol)
- [x] Depth-lift path — monocular depth → point cloud (FlatDepthEstimator + Open3D)
- [x] MASt3R multi-image protocol (injected reconstructor interface)
- [x] React frontend with shadcn/ui
- [x] React Three Fiber mesh viewer
- [x] SSE real-time progress
- [x] GLB / OBJ / STL download
- [x] Docker Compose full-stack setup
- [x] Depth Anything v2 real model integration
- [ ] TripoSR single-image path (real model integration)
- [ ] MASt3R real model integration
- [ ] CLI for local batch processing
- [ ] Deployment config (Fly.io / Render)

---

## Contributing

Contributions are welcome. Open a PR or an issue — whether it's a bug fix, new feature, or refactor.

---

## License

**MIT License** – Free to use, modify, and redistribute with credit.

---

## Connect

- [daniel.franco.inbox@gmail.com](mailto:daniel.franco.inbox@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/daniel-abrantes-franco/)
