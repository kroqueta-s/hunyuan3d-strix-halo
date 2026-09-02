# SPDX-License-Identifier: MIT
"""Replace Hunyuan3D's `custom_rasterizer` with **a z-buffer written in torch**.

Upstream's texture stage (`hy3dpaint`) renders the mesh from several viewpoints
and bakes what it sees into a texture. The only thing standing in the way on
this machine is `custom_rasterizer`, **a CUDA extension with a `.cu` kernel**
that cannot be built for Windows + ROCm.

**The surface to replace is tiny.** Of the two functions upstream calls,
`interpolate` is already pure torch and is reused verbatim; only `rasterize`
needs an implementation:

```python
findices, barycentric = custom_rasterizer_kernel.rasterize_image(pos[0], tri, ...)
```

- `pos` is in **clip space** (`MeshRender.raster_rasterize`), which is what this
  rasterizer takes.
- `findices` is the face index per pixel, **one-based, 0 for background** --
  the same convention `runners/trellis/raster.py` already returns.
- `barycentric` is `[H, W, 3]`, the weight of each of the winning face's three
  vertices. Upstream multiplies it straight onto the vertex values, so
  **background pixels must be zero**.

**Screen-space barycentrics are exact here** because the texture stage renders
orthographically (`MeshRender(camera_type="orth")`, which the pipeline does not
override), so there is no perspective term to correct for.

The rasterization core is the same as `runners/trellis/raster.py`: pixel centres
are tested against the triangle by the sign of the edge functions, never by
sampling. **The two runners deliberately keep their own copy** (they ship as
separate repositories and never import each other).

Verified by `tests/test_hunyuan3d_raster.py`.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import torch

# Cap on triangle-times-pixel tests evaluated at once. This sets the VRAM peak.
TILE_BUDGET = 16_000_000

# Bits used to pack the depth. The low 32 bits hold the face index.
_DEPTH_BITS = 21
_FACE_BITS = 32
_EMPTY = (1 << 62) - 1


def _rasterize(
    screen: torch.Tensor, depth: torch.Tensor, keep: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """Fill screen-space triangles into a z-buffer and return the packed key per pixel.

    Args:
        screen: `[F, 3, 2]` screen coordinates, in pixels.
        depth: `[F, 3]` NDC z (-1 near, 1 far).
        keep: `[F]` booleans; false faces are discarded.
        width: Width.
        height: Height.

    Returns:
        `[H*W]` of `int64`, depth in the high bits and face index in the low
        bits. Empty pixels hold `_EMPTY`.
    """
    device = screen.device
    buffer = torch.full((height * width,), _EMPTY, dtype=torch.int64, device=device)
    index = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if index.numel() == 0:
        return buffer

    tri = screen[index]  # [f, 3, 2]
    lo = tri.amin(dim=1)
    hi = tri.amax(dim=1)
    x0 = lo[:, 0].floor().clamp(0, width - 1).long()
    y0 = lo[:, 1].floor().clamp(0, height - 1).long()
    x1 = hi[:, 0].ceil().clamp(0, width - 1).long()
    y1 = hi[:, 1].ceil().clamp(0, height - 1).long()
    span = torch.maximum(x1 - x0, y1 - y0) + 1  # [f]

    depth_scale = float((1 << _DEPTH_BITS) - 1)
    # Bucket by bounding-box size rounded up to a power of two.
    bucket = torch.pow(2, torch.ceil(torch.log2(span.float().clamp(min=1.0)))).long()
    for size in torch.unique(bucket).tolist():
        sel = torch.nonzero(bucket == size, as_tuple=False).squeeze(1)
        per_chunk = max(1, TILE_BUDGET // (size * size))
        for start in range(0, sel.numel(), per_chunk):
            part = sel[start : start + per_chunk]
            faces_here = index[part]
            v = tri[part]  # [n, 3, 2]
            grid = torch.arange(size, device=device)
            n_here = part.numel()
            shape = (n_here, size, size)
            px = (x0[part][:, None, None] + grid[None, :, None]).expand(shape)
            py = (y0[part][:, None, None] + grid[None, None, :]).expand(shape)
            sx = px.to(v.dtype) + 0.5
            sy = py.to(v.dtype) + 0.5

            w0, w1, w2, area = _edges(v, sx, sy)
            sign = torch.where(area >= 0, 1.0, -1.0)
            inside = (w0 * sign >= 0) & (w1 * sign >= 0) & (w2 * sign >= 0) & (area.abs() > 1e-12)
            inside &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
            if not bool(inside.any()):
                continue

            safe_area = torch.where(area.abs() > 1e-12, area, torch.ones_like(area))
            z = depth[faces_here]  # [n, 3]
            zz = (
                w0 * z[:, 0][:, None, None]
                + w1 * z[:, 1][:, None, None]
                + w2 * z[:, 2][:, None, None]
            ) / safe_area
            inside &= (zz >= -1.0) & (zz <= 1.0)
            if not bool(inside.any()):
                continue

            pixel = (py * width + px)[inside]
            depth_q = (((zz.clamp(-1.0, 1.0) + 1.0) * 0.5) * depth_scale).long()[inside]
            face_idx = faces_here[:, None, None].expand(shape)[inside]
            # Depth high, face index low, so taking the minimum wins the front face.
            key = (depth_q << _FACE_BITS) | face_idx
            buffer.scatter_reduce_(0, pixel, key, reduce="amin")
    return buffer


def _edges(
    v: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edge functions of a triangle at the given sample points.

    Each `w` is twice the area of the sub-triangle opposite that vertex, so
    `w / area` is the barycentric weight of the vertex with the same index.

    Args:
        v: `[n, 3, 2]` screen-space triangle corners.
        sx: Sample x, broadcastable against `v[:, 0, 0][:, None, None]`.
        sy: Sample y, likewise.

    Returns:
        `(w0, w1, w2, area)`.
    """
    ax, ay = v[:, 0, 0][:, None, None], v[:, 0, 1][:, None, None]
    bx, by = v[:, 1, 0][:, None, None], v[:, 1, 1][:, None, None]
    cx, cy = v[:, 2, 0][:, None, None], v[:, 2, 1][:, None, None]
    w0 = (cx - bx) * (sy - by) - (cy - by) * (sx - bx)
    w1 = (ax - cx) * (sy - cy) - (ay - cy) * (sx - cx)
    w2 = (bx - ax) * (sy - ay) - (by - ay) * (sx - ax)
    return w0, w1, w2, w0 + w1 + w2


def rasterize(
    pos: torch.Tensor,
    tri: torch.Tensor,
    resolution: Any,
    clamp_depth: torch.Tensor | None = None,
    use_depth_prior: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stand-in for `custom_rasterizer.rasterize`.

    Args:
        pos: `[1, N, 4]` (or `[N, 4]`) clip-space vertices.
        tri: `[F, 3]` triangle indices.
        resolution: `(height, width)`.
        clamp_depth: Unused; upstream passes an empty tensor.
        use_depth_prior: Unused; upstream passes 0.

    Returns:
        `findices` `[H, W]` int32, **one-based with 0 for background**, and
        `barycentric` `[H, W, 3]` float, **zero on background pixels**.
    """
    if use_depth_prior:
        raise NotImplementedError("depth prior is not part of this replacement")
    verts = pos[0] if pos.dim() == 3 else pos
    verts = verts.float()
    height, width = int(resolution[0]), int(resolution[1])
    device = verts.device

    idx = tri.long()
    tri_clip = verts[idx]  # [F, 3, 4]
    w = tri_clip[..., 3]
    # Discard triangles that wrap in front of the near plane, where the division
    # breaks down. The texture stage looks at the object from outside, so they
    # do not actually occur.
    keep = (w > 1e-6).all(dim=1)
    inv_w = 1.0 / w.clamp(min=1e-6)
    ndc = tri_clip[..., :3] * inv_w.unsqueeze(-1)
    screen = torch.stack(
        [(ndc[..., 0] * 0.5 + 0.5) * width, (ndc[..., 1] * 0.5 + 0.5) * height], dim=-1
    )

    buffer = _rasterize(screen, ndc[..., 2], keep, width, height)

    hit = buffer != _EMPTY
    findices = torch.zeros(height * width, dtype=torch.int32, device=device)
    barycentric = torch.zeros(height * width, 3, dtype=torch.float32, device=device)
    if bool(hit.any()):
        won = (buffer[hit] & ((1 << _FACE_BITS) - 1)).long()  # winning face per hit pixel
        findices[hit] = (won + 1).to(torch.int32)
        # Recompute the barycentric weights for the face that won each pixel.
        flat = torch.nonzero(hit, as_tuple=False).squeeze(1)
        sx = (flat % width).to(torch.float32) + 0.5
        sy = (flat // width).to(torch.float32) + 0.5
        v = screen[won]  # [n, 3, 2]
        w0, w1, w2, area = _edges(v, sx[:, None, None], sy[:, None, None])
        safe_area = torch.where(area.abs() > 1e-12, area, torch.ones_like(area))
        weights = torch.cat([w0, w1, w2], dim=-1).squeeze(1) / safe_area.squeeze(1)
        barycentric[hit] = weights

    return findices.view(height, width), barycentric.view(height, width, 3)


def interpolate(
    col: torch.Tensor, findices: torch.Tensor, barycentric: torch.Tensor, tri: torch.Tensor
) -> torch.Tensor:
    """Upstream's `custom_rasterizer.interpolate`, **which is already pure torch**.

    It is reproduced here unchanged so that the stand-in module is complete.
    """
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)


def install() -> None:
    """Put the stand-in `custom_rasterizer` into `sys.modules`.

    **Call this before importing `hy3dpaint`**, whose `MeshRender` imports the
    module inside `__init__` when `raster_mode == "cr"` (the only mode upstream
    supports).
    """
    module = types.ModuleType("custom_rasterizer")
    module.__file__ = "<hearth shim: custom_rasterizer>"
    module.rasterize = rasterize  # type: ignore[attr-defined]
    module.interpolate = interpolate  # type: ignore[attr-defined]
    sys.modules["custom_rasterizer"] = module
