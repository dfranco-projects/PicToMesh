# 📸 PicToMesh – From Images to Interactive 3D Meshes

**PicToMesh** is a fully open-source, drag-and-drop interface that takes one or more images of an object and turns them into a 3D mesh. It's fast, visual, and optimized for real-world use—powered by classic computer vision and modern 3D rendering.

---

## 🚀 What It Does

- ✨ Drag and drop a folder containing one or more images
- 🔍 Quickly filters out irrelevant images using ORB (Oriented FAST and Rotated BRIEF)
- 🌀 Generates a clean point cloud of the main object (backgrounds? nope.)
- 🧱 Builds a 3D mesh you can rotate, inspect, and save
- 🎛️ Not satisfied? Switch mesh generation algorithms on the fly

---

## 💡 Why It Exists

3D scanning and modeling is cool—but most tools are bloated, closed-source, or locked behind expensive software. PicToMesh is:

- Fast (thanks to ORB)
- Clean (PEP8-compliant, Dockerized)
- Visual (interact with your model in-browser)
- Open (MIT licensed and ready to extend)

---

## 🧰 Tech Stack

| Layer        | Tools Used                                   |
|--------------|----------------------------------------------|
| Frontend     | React, Three.js, Vite                        |
| Backend      | FastAPI, OpenCV, Open3D                      |
| 3D Viewer    | Three.js + OrbitControls                     |
| ML/Refinement| PyTorch (for future super-resolution)        |
| Container    | Docker, Docker Compose                       |
| Infra as Code| Terraform (optional, but infra-ready)        |
| CI/CD        | GitHub Actions (with linting, formatting)    |
| Styling      | PEP8 with `flake8`, `black`, and `isort`     |

---

## 📁 Project Structure

```bash
PicToMesh/
├── backend/                        # FastAPI app: ORB matcher, point cloud, mesh gen
│   ├── api/                        # API routes and controllers
│   ├── core/                       # ORB matching, Open3D processing logic
│   ├── models/                     # PyTorch super-resolution modules
│   └── main.py                     # FastAPI entry
│
├── frontend/                       # React + Vite + Three.js UI
│   ├── components/                 # Drag-n-drop, viewer, controls
│   ├── pages/                      # Main layout and routes
│   └── main.jsx                    # Frontend entry point
│
├── demos/                          # Jupyter notebooks to explain core processes visually
│   ├── orb_matcher.ipynb           # Interactive, beginner-friendly ORB guide
│   ├── point_cloud_gen.ipynb       # Open3D and PCD explanations
│   └── mesh_refinement.ipynb       # (Planned) Super-resolution experiments
│
├── tests/                          # Automated testing with pytest
│   ├── test_orb_matcher.py         # Unit tests for ORB logic
│   ├── test_point_cloud.py         # Tests for Open3D pipelines
│   └── conftest.py                 # Fixtures and test setup
│
├── test_images/                    # Images for testing
│
├── docker-compose.yml              # Runs frontend + backend together
├── .gitignore                      # Ignores common files, Python, Node, etc.
├── requirements.txt                # Backend Python deps (manually or pip freeze)
├── frontend/package.json           # Auto-generated when you init Vite frontend
├── README.md                       # You're here :)                
```

---

## 🖥️ Demo

Coming soon – GIF + link to live version on Render/Fly.io

---

## 🛠️ How to Run Locally

```bash
# clone the repo
git clone https://github.com/dfranco-projects/PicToMesh.git
cd PicToMesh

# spin up frontend and backend
docker-compose up --build
```

Then head to `http://localhost:3000` and start dragging in images.

The app will:

- Run ORB feature matching to identify relevant images
- Generate a point cloud, filtering out background noise
- Build a 3D mesh of the main object
- Show the result in an interactive viewer—spin, zoom, inspect

If the mesh isn’t quite right, no problem—you can select a different reconstruction algorithm from a dropdown and try again. Once you're happy, save the result locally.

---

## 🧪 Roadmap & Stretch Goals

- [x] ORB-based image matcher
- [x] Point cloud generation from matched images
- [x] Mesh creation & interactive 3D viewer
- [ ] Dropdown to select different mesh generation algorithms
- [ ] Mesh super-resolution using ML-based refinement
- [ ] STL/OBJ export for 3D printing or CAD tools
- [ ] Fully containerized deployment using Docker
- [ ] Infrastructure setup using Terraform (Fly.io / Render)
- [ ] Add automated CI/CD pipeline with GitHub Actions
- [ ] Add PyTorch-based mesh refinement (super-resolution or denoising)
- [ ] Write a blog post explaining the full pipeline and tech choices

---

## 🤝 Contributing

Want to help build the future of easy 3D reconstruction? Contributions are super welcome!

Whether it's fixing a bug, adding a feature, improving the UI, or cleaning up code—we'd love to have you on board. Please open an issue or submit a pull request with a clear description.

---

## 📜 License

MIT License – free to use, modify, and distribute with no strings attached.

---

## 💬 Say Hi

This project was built to blend AI, vision, and practical dev skills into something fun and useful.

If you want to chat, collab, hire, or just geek out about computer vision and 3D stuff—reach out!

📫 [Mail](mailto:daniel.franco.inbox@gmail.com)  
💼 [LinkedIn](https://www.linkedin.com/in/daniel-abrantes-franco/)
