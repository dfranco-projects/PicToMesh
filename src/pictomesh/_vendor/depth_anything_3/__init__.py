"""Vendored Depth Anything 3 inference code (https://github.com/ByteDance-Seed/depth-anything-3,
commit 3d835ec). The code is Apache-2.0 (LICENSE next to this file). The DA3-LARGE-1.1 weights
it loads are CC BY-NC 4.0, non-commercial.

Upstream pins numpy<2 and pulls in xformers, pycolmap, evo, gsplat and more, so only the
modules inference needs are vendored. Local changes:
- absolute imports rewritten to pictomesh._vendor.depth_anything_3; only the da3-large config kept
- exporters (utils.export), pose alignment (utils.pose_align), Gaussian-splat heads, the app,
  benchmarks and CLI not vendored; calling export or passing input extrinsics raises
- autocast only on CUDA: on MPS/CPU it forced fp16, now fp32
- utils.logger routes messages through the logging module instead of print()
- imageio imported lazily in utils.parallel_utils (only image saving uses it)
"""
