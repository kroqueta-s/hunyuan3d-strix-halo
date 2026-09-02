# SPDX-License-Identifier: MIT
"""Stand in for `mesh_inpaint_processor` (**a C++ extension upstream, pure python here**).

The bake leaves holes: **texels no viewpoint could see**. Upstream fills them in
`meshVerticeInpaint`, walking the mesh edges to spread nearby colour by a
distance weight and writing the result back into the UV texture.

Upstream builds `hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp`
with pybind11, but **there is no C++ compiler on this machine** (no Visual
Studio). Even a successful build would be at risk: **Smart App Control has
blocked freshly built binaries here** (`WinError 4551`). So, as `raster.py`
does, **upstream's procedure is reproduced in python exactly**. **Nothing
clever is added.**

Details of upstream that are deliberately kept:

- **Adjacency is directed and duplicates are not collapsed.** `G[a].push_back(b)`
  runs three times per face, so an edge shared by two faces is counted twice.
  That changes the weights, so it must not become a `set`.
- **Smoothing is sequential (Gauss-Seidel).** A vertex coloured earlier in a
  pass feeds the average of a later one. Updating every vertex at once gives a
  different answer.
- **`uncolored_vtxs` holds duplicates, and can hold already-coloured vertices.**
  Upstream does not distinguish them and overwrites.
- The stopping rule: if the number of vertices that could not be coloured equals
  the previous pass, `smooth_count` goes down, otherwise up; it ends at zero,
  starting from 2.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

# Keeps the weight finite for coincident vertices (upstream's 1E-4).
_MIN_DISTANCE = 1e-4


def _uv_texels(vtx_uv: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn UV coordinates into texel (row, column) pairs.

    **Upstream's rounding is kept exactly**: the column is `u * (W - 1)` and the
    row is `(1 - v) * (H - 1)`.

    Returns:
        Integer arrays of (rows, columns).
    """
    col = np.round(vtx_uv[:, 0] * (width - 1)).astype(np.int64)
    row = np.round((1.0 - vtx_uv[:, 1]) * (height - 1)).astype(np.int64)
    return row, col


def _build_graph(pos_idx: np.ndarray, vtx_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-vertex adjacency as two CSR-style arrays (**duplicates kept**).

    Returns:
        (offsets, neighbours). Vertex `v`'s neighbours are
        `neighbours[offsets[v]:offsets[v + 1]]`.
    """
    src = np.concatenate([pos_idx[:, k] for k in range(3)])
    dst = np.concatenate([pos_idx[:, (k + 1) % 3] for k in range(3)])
    order = np.argsort(src, kind="stable")
    neighbours = dst[order]
    counts = np.bincount(src, minlength=vtx_num)
    offsets = np.zeros(vtx_num + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, neighbours


def _initialize(
    texture: np.ndarray,
    mask: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vtx_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read each vertex's colour, and whether it has one, out of the baked texture.

    Returns:
        (vtx_mask, vtx_color, uncolored). `uncolored` is in face-corner order
        and **contains duplicates**.
    """
    channels = texture.shape[2]
    vtx_mask = np.zeros(vtx_num, dtype=np.float32)
    vtx_color = np.zeros((vtx_num, channels), dtype=np.float32)

    corner_vtx = pos_idx.reshape(-1)
    corner_uv = uv_idx.reshape(-1)
    corner_row = rows[corner_uv]
    corner_col = cols[corner_uv]
    lit = mask[corner_row, corner_col] > 0

    # **The last face wins** - upstream simply assigns, face after face.
    vtx_mask[corner_vtx[lit]] = 1.0
    vtx_color[corner_vtx[lit]] = texture[corner_row[lit], corner_col[lit]]
    uncolored = corner_vtx[~lit]
    return vtx_mask, vtx_color, uncolored


def _smooth(
    vtx_pos: np.ndarray,
    vtx_mask: np.ndarray,
    vtx_color: np.ndarray,
    uncolored: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
) -> None:
    """Spread neighbouring colour into uncoloured vertices (**modifies in place**).

    The weight is upstream's `1 / max(distance, 1e-4) ** 2`. Distances never
    change, so they are computed once instead of once per pass.
    """
    # One weight per edge, in the same order as the adjacency array.
    heads = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    delta = vtx_pos[heads] - vtx_pos[neighbours]
    distance = np.maximum(np.sqrt((delta * delta).sum(axis=1)), _MIN_DISTANCE)
    weights = (1.0 / distance) ** 2

    # Walked in the original order, duplicates included, as upstream does.
    todo = [int(v) for v in uncolored]
    smooth_count = 2
    last_uncolored = 0
    while smooth_count > 0:
        remaining = 0
        for vtx in todo:
            start, end = offsets[vtx], offsets[vtx + 1]
            nbrs = neighbours[start:end]
            colored = vtx_mask[nbrs] > 0
            if not colored.any():
                remaining += 1
                continue
            w = weights[start:end][colored]
            total = w.sum()
            if total <= 0.0:
                remaining += 1
                continue
            vtx_color[vtx] = (vtx_color[nbrs[colored]] * w[:, None]).sum(axis=0) / total
            vtx_mask[vtx] = 1.0
        smooth_count += -1 if last_uncolored == remaining else 1
        last_uncolored = remaining


def _write_back(
    texture: np.ndarray,
    mask: np.ndarray,
    vtx_mask: np.ndarray,
    vtx_color: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Write the vertex colours back into the UV texture.

    Returns:
        New (texture, mask) arrays. The originals are left alone.
    """
    new_texture = texture.copy()
    new_mask = mask.copy()

    corner_vtx = pos_idx.reshape(-1)
    corner_uv = uv_idx.reshape(-1)
    filled = vtx_mask[corner_vtx] == 1.0
    row = rows[corner_uv][filled]
    col = cols[corner_uv][filled]
    new_texture[row, col] = vtx_color[corner_vtx[filled]]
    new_mask[row, col] = 255
    return new_texture, new_mask


def meshVerticeInpaint(  # noqa: N802 - upstream's name, kept as-is
    texture: np.ndarray,
    mask: np.ndarray,
    vtx_pos: np.ndarray,
    vtx_uv: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
    method: str = "smooth",
) -> tuple[np.ndarray, np.ndarray]:
    """Fill texels no view could see, by walking the mesh's connectivity.

    Args:
        texture: `[H, W, C]` float texture.
        mask: `[H, W]` uint8. Zero means "not painted yet".
        vtx_pos: `[V, 3]` vertex positions.
        vtx_uv: `[U, 2]` UV coordinates.
        pos_idx: `[F, 3]` vertex indices.
        uv_idx: `[F, 3]` UV indices.
        method: Only `smooth` is supported. Upstream's `forward` is never
            reached from `uv_inpaint`, so it is not implemented.

    Returns:
        (texture, mask).

    Raises:
        NotImplementedError: For anything but `smooth`. **It never quietly
            returns a different result.**
    """
    if method != "smooth":
        raise NotImplementedError(
            f"method={method!r} is not implemented in the stand-in "
            "(upstream's uv_inpaint only ever uses smooth)"
        )

    texture = np.ascontiguousarray(texture, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    vtx_pos = np.ascontiguousarray(vtx_pos, dtype=np.float32)
    vtx_uv = np.ascontiguousarray(vtx_uv, dtype=np.float32)
    pos_idx = np.ascontiguousarray(pos_idx, dtype=np.int64)
    uv_idx = np.ascontiguousarray(uv_idx, dtype=np.int64)

    height, width = texture.shape[0], texture.shape[1]
    rows, cols = _uv_texels(vtx_uv, height, width)

    # **`vtx_pos` decides the vertex count**, as upstream does, so a vertex no
    # face refers to does not shift the numbering.
    vtx_num = int(vtx_pos.shape[0])
    vtx_mask, vtx_color, uncolored = _initialize(
        texture, mask, pos_idx, uv_idx, rows, cols, vtx_num
    )
    offsets, neighbours = _build_graph(pos_idx, vtx_num=vtx_num)
    _smooth(vtx_pos, vtx_mask, vtx_color, uncolored, offsets, neighbours)
    return _write_back(texture, mask, vtx_mask, vtx_color, pos_idx, uv_idx, rows, cols)


def install() -> None:
    """Install the stand-in as `DifferentiableRenderer.mesh_inpaint_processor`.

    **Call this before `MeshRender` is imported.** Upstream wraps
    `from .mesh_inpaint_processor import meshVerticeInpaint` in a `try` and
    **swallows the failure**. The missing name then surfaces only inside the
    final `uv_inpaint`, after minutes of baking, as a `NameError`.

    Call it with the upstream clone on `sys.path` (`texture._in_upstream()`
    arranges that).
    """
    module = types.ModuleType("mesh_inpaint_processor")
    module.__file__ = "<hearth stand-in: mesh_inpaint_processor>"
    module.meshVerticeInpaint = meshVerticeInpaint  # type: ignore[attr-defined]

    # Upstream imports both from `hy3dpaint` and from inside it. **Register both.**
    for package in ("DifferentiableRenderer", "hy3dpaint.DifferentiableRenderer"):
        sys.modules[f"{package}.mesh_inpaint_processor"] = module
        parent: Any = sys.modules.get(package)
        if parent is not None:
            parent.mesh_inpaint_processor = module
