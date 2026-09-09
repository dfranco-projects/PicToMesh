"""Vendored TripoSR (https://github.com/VAST-AI-Research/TripoSR, MIT, commit 107cefd).

Upstream ships no package metadata, so it cannot be installed with uv. Local changes:
- models/isosurface.py: PyMCubes replaces torchmcubes (a C++ extension with no wheels).
- utils.py: find_class() maps the ``tsr.`` paths in config.yaml onto this package;
  rembg / imageio helpers and to_gradio_3d_orientation() removed.
- bake_texture.py (xatlas, moderngl) not vendored.
"""
