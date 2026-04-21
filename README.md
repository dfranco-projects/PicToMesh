# 📸 PicToMesh – Open-Source 3D Meshes From Your Images

**PicToMesh** is a full-stack, open-source tool that lets you drag-and-drop images and receive a downloadable 3D mesh of the object — all in the browser, with no signups, no API keys, and no black-box models. Built for hackers, researchers, and hobbyists who want to own their pipeline.

---

## 🎯 Utilities & Why It Exists

Creating 3D meshes from images is traditionally a task locked behind expensive software or complex pipelines. **PicToMesh** aims to break that barrier by offering a fully open-source, easy-to-use tool for:

- ✅ **Hobbyists & Makers:** Turn your DIY projects or handmade objects into printable 3D files.
- 🏫 **Educators & Students:** Learn photogrammetry, 3D reconstruction, and computer vision with a visual, hackable tool.
- 📦 **Developers & Researchers:** Extend it as a base for experiments in image processing, scene understanding, or shape analysis.
- 🖼️ **Designers & Artists:** Convert real-world imagery into interactive meshes for use in 3D scenes or assets.
- 🔍 **Open Science Advocates:** Transparent algorithms, no vendor lock-in, and reproducible outputs — perfect for academic use.

---

## 🚀 What It Does

- 🖼️ **Drag and drop** one or many images of an object
- 🧠 Uses **CLIP embeddings** to auto-cluster and filter mismatched or outlier images (if multi-image)
- 🔎 Applies **point cloud reconstruction** from images using open-source algorithms
- 🧱 Generates a **3D mesh** based on input image count (different algo for 1 vs many)
- 🌀 View your 3D model interactively (rotate, zoom, pan) in-browser
- 💾 **Download** the final mesh (STL, OBJ, or GLB)
- 🔁 Supports algorithm switching and re-generation
- 🔓 Fully open-source — no API keys, no paywalls, just local compute
- 💡 Designed to be readable, hackable, and educational

---

## 🧠 How It Works

| Scenario         | What Happens Under the Hood                             |
|------------------|----------------------------------------------------------|
| Single Image     | SAM2 background removal → **TripoSR** feed-forward mesh (no photogrammetry needed) |
| Multiple Images  | SAM2 background removal → **MASt3R** neural SfM (no camera calibration needed) → Open3D Poisson mesh |
| Depth Fallback   | **Depth Anything v2** monocular depth → point cloud lift (CPU-friendly path) |
| Mesh Interaction | Three.js + OrbitControls for interactive in-browser viewing |

---

## 🧰 Tech Stack

| Layer              | Tools Used                                                        |
|--------------------|-------------------------------------------------------------------|
| Frontend           | React, Vite, Three.js, Tailwind                                   |
| Backend            | Django, OpenCV, Open3D, NumPy, PyTorch                            |
| Single-image → 3D  | TripoSR (feed-forward transformer, no photogrammetry)             |
| Multi-image → 3D   | MASt3R neural SfM + Open3D Poisson reconstruction                 |
| Segmentation       | SAM2 (automatic background removal before reconstruction)         |
| Depth fallback     | Depth Anything v2 (monocular depth, CPU-friendly)                 |
| Mesh Viewer        | Three.js + OrbitControls                                          |
| Container          | Docker, Docker Compose                                            |
| Package manager    | uv                                                                |
| Linting            | ruff                                                              |

---

## 📁 Project Structure

```bash
PicToMesh/
├── src/pictomesh/              # Core Python package
│   ├── image_io/               # Image loading and metadata (ImageManager)
│   ├── filtering/              # CLIP embedding + Louvain outlier detection
│   ├── segmentation/           # SAM2 background removal
│   ├── reconstruction/         # MASt3R neural SfM + Depth Anything fallback
│   ├── mesh/                   # TripoSR (single-image) + Open3D Poisson (multi-image)
│   └── uploader/               # File validation and drag-and-drop handling
│
├── backend/                    # Django API (routes, views, serializers, settings)
├── frontend/                   # React/Vite (components, pages, Three.js viewer)
├── cli/                        # CLI tools for local batch processing
├── scripts/                    # One-off utility scripts
├── demos/                      # Jupyter notebooks for experimenting
│
├── media/                      # Runtime: uploaded images and generated meshes
├── output/                     # Runtime: intermediate results
├── static/                     # Static assets served by Django/nginx
│
├── tests/                      # Pytest tests
│   └── assets/                 # All test images (chairs, cats)
│
├── docs/                       # Documentation and diagrams
├── docker/                     # Dockerfiles + nginx config
├── bin/                        # Dev scripts (install.sh)
│
├── pyproject.toml              # Dependencies and tool config (uv)
├── Makefile                    # Dev commands
└── README.md
```

---

## 🖼️ Live Demo

Coming soon – GIF and link to deployed version (no login required).

---

## 🛠️ Run Locally

```bash
# clone the repo
git clone https://github.com/dfranco-projects/PicToMesh.git
cd PicToMesh

# spin up the app
docker-compose up --build

```
---

## 🧪 Roadmap

- [x] Upload and drag-and-drop interface
- [x] CLIP-based image similarity + Louvain clustering
- [x] Point cloud generation (Open3D)
- [x] Mesh creation (single vs multi-image support)
- [x] 3D viewer with orbit, zoom, and lighting
- [x] Mesh file download (.OBJ or .STL)
- [ ] Retry / Re-select processing pipeline
- [ ] Algorithm selection dropdown (Poisson, Ball Pivoting, etc.)
- [ ] GLB format export (web-optimized)
- [ ] Light/dark theme toggle for frontend
- [ ] Mesh denoising or refinement with open-source tools
- [ ] Replace all remaining proprietary dependencies (if any)
- [ ] Add image preprocessing options (resize, background removal)
- [ ] CLI support for local batch processing

---

## 🤝 Contributing

Contributions are welcome! Whether you’re improving performance, fixing bugs, adding features, or refactoring code — just open a PR or an issue.

If you're unsure where to start, check out the `issues` tab for ideas, or reach out!

---

## 📜 License

**MIT License** – Free to use, modify, and redistribute with credit.

---

## 💬 Connect

- 📫 [daniel.franco.inbox@gmail.com](mailto:daniel.franco.inbox@gmail.com)  
- 💼 [LinkedIn](https://www.linkedin.com/in/daniel-abrantes-franco/)