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
| Single Image     | Skips filtering, runs 1-image mesh reconstruction flow  |
| Multiple Images  | Uses **CLIP + cosine similarity + Louvain clustering** to group consistent images and discard outliers |
| After Filtering  | Generates a point cloud via photogrammetry (OpenMVG/OpenMVS, or Open3D) |
| Mesh Creation    | Converts point cloud to 3D mesh using Poisson or Ball Pivoting |
| Mesh Interaction | Uses Three.js for an interactive in-browser experience  |

---

## 🧰 Tech Stack

| Layer         | Tools Used                                                  |
|---------------|-------------------------------------------------------------|
| Frontend      | React, Vite, Three.js, Tailwind                             |
| Backend       | Django, OpenCV, Open3D, NumPy, CLIP (open source, no API)   |
| Mesh Viewer   | Three.js + OrbitControls                                    |
| 3D Engine     | Open3D or custom Poisson / Ball Pivoting meshers           |
| Container     | Docker, Docker Compose                                     |
| Deployment    | Fly.io / Render (infra ready with no vendor lock-in)       |
| Styling       | PEP8 with `black`, `flake8`, and `isort`                   |

---

## 📁 Project Structure

```bash
PicToMesh/
├── backend/                    # Django app: filtering, 3D processing, mesh export
│   ├── clustering/             # CLIP-based similarity + Louvain clustering
│   ├── reconstruction/         # Point cloud and mesh generation
│   ├── views.py                # Upload endpoints and processing logic
│   └── models/                 # (Optional future ML support)
│
├── frontend/                   # React + Vite + Three.js interface
│   ├── components/             # Drag-and-drop UI, mesh viewer
│   ├── pages/                  # Routes and main layout
│   └── main.jsx                # Frontend entry point
│
├── docker/                     # Dockerfiles and container configs
│
├── static/                     # Test images and sample meshes
├── tests/                      # Unit tests for core image + mesh logic
├── README.md                   # This file :)
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