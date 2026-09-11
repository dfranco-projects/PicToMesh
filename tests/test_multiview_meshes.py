"""
Multi-view mesh samples
=======================
Runs the real multi-photo pipeline (rembg masks, CLIP filtering, Depth Anything 3, hull
fill, Poisson) on every photo set in tests/assets/ and saves the results in tests/output/
for a visual check: open <set>.glb (the mesh) or <set>_points.ply (the cloud it came from)
in VS Code or any 3D viewer.

A set is every image sharing a name before its last "_": chair_1.jpeg, chair_2.jpeg, ...
Sets with two or more photos are meshed, so dropping name_1.jpg, name_2.jpg, ... anywhere
in tests/assets/ adds one. Marked slow: downloads and runs the models.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import trimesh

ASSETS = Path(__file__).parent / "assets"
OUTPUT = Path(__file__).parent / "output"


def _photo_sets() -> dict[str, list[Path]]:
    sets = defaultdict(list)
    for path in sorted(ASSETS.rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            sets[path.stem.rsplit("_", 1)[0]].append(path)
    return {name: paths for name, paths in sets.items() if len(paths) >= 2}


class _KeepsCloud:
    """Wraps the reconstructor to keep the cloud it hands to the mesher."""

    def __init__(self, reconstructor):
        self._reconstructor = reconstructor
        self.cloud = None

    def reconstruct(self, images, masks=None):
        self.cloud = self._reconstructor.reconstruct(images, masks)
        return self.cloud


@pytest.fixture(scope="module")
def pipeline():
    from pictomesh.filtering.clip_encoder import CLIPEncoder
    from pictomesh.filtering.service import FilteringService
    from pictomesh.mesh.service import MeshService
    from pictomesh.pipeline import Pipeline
    from pictomesh.reconstruction.da3 import Da3Reconstructor
    from pictomesh.reconstruction.service import FlatDepthEstimator, ReconstructionService
    from pictomesh.segmentation.service import RembgSegmentor, SegmentationService

    recorder = _KeepsCloud(Da3Reconstructor())
    built = Pipeline(
        SegmentationService(RembgSegmentor()),
        FilteringService(CLIPEncoder()),
        ReconstructionService(),
        MeshService(),
        FlatDepthEstimator(),
        reconstructor=recorder,
    )
    return built, recorder


def _open_edges(mesh: trimesh.Trimesh) -> int:
    _, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    return int((counts == 1).sum())


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_photo_sets()))
def test_multiview_mesh_is_saved_and_closed(pipeline, name):
    from backend.worker.tasks import _load_image  # the worker's resize, for parity

    built, recorder = pipeline
    images = [_load_image(str(p)) for p in _photo_sets()[name]]

    out = built.run(images, OUTPUT / name, fmt="glb")
    o3d.io.write_point_cloud(str(OUTPUT / f"{name}_points.ply"), recorder.cloud)
    mesh = trimesh.load(out, force="mesh", process=False)
    print(f"\n{name}: {out} ({len(mesh.faces)} faces), cloud {OUTPUT / f'{name}_points.ply'}")

    assert mesh.body_count == 1
    # closed apart from where the photos' frames cut the object off
    assert _open_edges(mesh) <= 0.0005 * len(mesh.edges)
    assert mesh.visual.kind == "vertex"
