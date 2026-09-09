from typing import Optional, Tuple

import mcubes
import numpy as np
import torch
import torch.nn as nn


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        # PyMCubes returns vertices in grid-index (x, y, z) order with outward faces,
        # so no axis swap is needed (torchmcubes returned them reversed).
        v_pos, t_pos_idx = mcubes.marching_cubes(level.detach().cpu().numpy(), 0.0)
        v_pos = torch.from_numpy(v_pos.astype(np.float32)) / (self.resolution - 1.0)
        t_pos_idx = torch.from_numpy(t_pos_idx.astype(np.int64))
        return v_pos.to(level.device), t_pos_idx.to(level.device)
