# Contributing

PRs and issues welcome, whether it's a bug fix, a new feature or a refactor.

## Run without Docker

You need [uv](https://docs.astral.sh/uv/), Node 22+ and Redis. uv fetches the pinned Python 3.12 on its own.

```bash
# macOS
brew install uv node redis

# Debian/Ubuntu
sudo apt install redis-server nodejs npm && curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then run `make local`. It creates the venv, syncs deps with the model extras (rembg, TripoSR, Depth Anything v2, DUSt3R) and starts redis, api, worker and frontend. Open http://localhost:3000.

- Committing? Run `make install` once to set up the git hooks.
- Prefer separate terminals? Run `make api`, `make worker` and `make web` (needs a running Redis).
- Port 8000 taken? Run `API_PORT=8010 make local` and the Vite proxy follows.
- A `.env` is optional. Copy `.env.example` to `.env` only if you want to override the defaults.
- The worker downloads model weights on its first start (TripoSR ~1.7 GB, DUSt3R ~2.4 GB, plus CLIP, Depth Anything v2 Small and u2net). Later runs reuse the cache.

## Commit hooks

Commits run `ruff check --fix` and `ruff format` through pre-commit. If a hook rewrites a file or an unfixable lint error remains, the commit is rejected: re-stage and commit again. A commit-msg hook also rejects `Co-Authored-By: Claude` trailers.

## How it's built

```
Browser (React + React Three Fiber)
  │  HTTP + SSE
  ▼
FastAPI
  ├── POST /jobs                     enqueue a job
  ├── GET  /jobs/{id}                status + result URL
  ├── GET  /jobs/{id}/stream         SSE progress
  └── GET  /meshes/{id}/{filename}   GLB / OBJ / STL
  │
  └──► ARQ worker (Redis queue)
         ├── rembg background removal
         ├── TripoSR            1 photo
         └── DUSt3R + Poisson   2+ photos
```

- Backend: FastAPI, ARQ + Redis, PyTorch (CUDA / MPS / CPU), Open3D + trimesh
- Frontend: React 19 + Vite, React Three Fiber + drei, shadcn/ui + Tailwind v4, TanStack Query, Zustand
- Infra: Docker Compose, nginx, uv, ruff

## Model licences

PicToMesh itself is MIT. Two vendored models carry their own terms: TripoSR is MIT, DUSt3R is CC BY-NC-SA 4.0, non-commercial use only. Installing the `multi-view` extra therefore limits that deployment to non-commercial use.
