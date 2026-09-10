"""Vendored DUSt3R (https://github.com/naver/dust3r, commit 4c24a6e) and the CroCo modules it
needs (../croco). Both are CC BY-NC-SA 4.0, non-commercial: see the LICENSE files next to them.

Upstream is not packaged and imports CroCo through sys.path tricks. Local changes:
- absolute imports rewritten to pictomesh._vendor.{dust3r,croco}; path_to_croco removed
- visualisation (dust3r.viz, matplotlib) and training modules not vendored; base_opt.show()
  is left in place but unusable
- the CUDA RoPE kernel is not built; the PyTorch fallback runs silently
"""
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
