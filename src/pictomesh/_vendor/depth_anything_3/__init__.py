"""Vendored Depth Anything 3 inference code (github.com/ByteDance-Seed/depth-anything-3, 3d835ec)

Code Apache-2.0 (LICENSE here); the DA3-LARGE-1.1 weights it loads are CC BY-NC 4.0.
Only inference is vendored: upstream pins numpy<2 and pulls in xformers, pycolmap, evo and
gsplat. Local changes:
- imports rewritten to pictomesh._vendor.depth_anything_3; only the da3-large config kept
- no exporters, pose alignment, Gaussian heads, app, benchmarks or CLI; export and input
  extrinsics raise
- autocast on CUDA only (MPS/CPU run fp32)
- utils.logger logs through the logging module instead of print()
- imageio imported lazily in utils.parallel_utils
"""
