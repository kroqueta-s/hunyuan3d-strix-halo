# SPDX-License-Identifier: MIT
"""Verify the `mesh_inpaint_processor` stand-in (**it fills the holes, not the picture**).

The bake leaves texels that no camera saw. Upstream's C++ extension walks the
mesh edges and spreads nearby colour into them. This machine has no compiler, so
`runners/hunyuan3d/inpaint.py` reproduces the same walk in Python, and these
tests pin the properties the texture depends on:

- **Texels that were already painted are never touched.** Inpainting must not
  repaint what the diffusion model produced.
- **A hole surrounded by one colour is filled with that colour**, and its mask
  flips to 255 so the later Navier-Stokes pass leaves it alone.
- **Distance decides.** A vertex between a near red neighbour and a far blue one
  comes out closer to red, because the weight is `1 / distance ** 2`.
- **An island with no painted neighbour stays empty** rather than picking up a
  colour from nowhere, and the walk still terminates.

Run it with any python that has numpy (torch is not needed)::

    <venv>\\Scripts\\python.exe .\\tests\\test_hunyuan3d_inpaint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.hunyuan3d import inpaint  # noqa: E402

SIZE = 8


def _strip(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A chain of vertices along x, triangulated so every neighbour is an edge.

    Returns:
        (vtx_pos, vtx_uv, faces). Vertex `i` owns texel `(0, i)`.
    """
    pos = np.zeros((count, 3), dtype=np.float32)
    pos[:, 0] = np.arange(count, dtype=np.float32)
    uv = np.zeros((count, 2), dtype=np.float32)
    # Column i of row 0: u = i / (SIZE - 1), v = 1 so that row = 0.
    uv[:, 0] = np.arange(count, dtype=np.float32) / (SIZE - 1)
    uv[:, 1] = 1.0
    faces = np.array([[i, i + 1, i + 2] for i in range(count - 2)], dtype=np.int64)
    return pos, uv, faces


def _blank() -> tuple[np.ndarray, np.ndarray]:
    """An empty texture and its mask."""
    return np.zeros((SIZE, SIZE, 3), dtype=np.float32), np.zeros((SIZE, SIZE), dtype=np.uint8)


def test_painted_texels_are_left_alone() -> None:
    """What the diffusion model painted survives untouched."""
    pos, uv, faces = _strip(4)
    texture, mask = _blank()
    for i, colour in enumerate([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]):
        texture[0, i] = colour
        mask[0, i] = 255
    out_texture, out_mask = inpaint.meshVerticeInpaint(texture, mask, pos, uv, faces, faces)
    for i in range(3):
        assert np.allclose(out_texture[0, i], texture[0, i]), f"texel {i} was repainted"
        assert out_mask[0, i] == 255


def test_a_hole_takes_the_surrounding_colour() -> None:
    """A gap between two red vertices comes out red, and the mask closes."""
    pos, uv, faces = _strip(3)
    texture, mask = _blank()
    texture[0, 0] = (1.0, 0.0, 0.0)
    texture[0, 2] = (1.0, 0.0, 0.0)
    mask[0, 0] = mask[0, 2] = 255
    out_texture, out_mask = inpaint.meshVerticeInpaint(texture, mask, pos, uv, faces, faces)
    assert np.allclose(out_texture[0, 1], (1.0, 0.0, 0.0)), out_texture[0, 1]
    assert out_mask[0, 1] == 255, "the filled texel is still marked as a hole"


def test_the_nearer_neighbour_weighs_more() -> None:
    """`1 / distance ** 2` means the close red beats the distant blue."""
    pos, uv, _ = _strip(3)
    # vertex 1 sits next to vertex 0 and far from vertex 2.
    pos[0] = (0.0, 0.0, 0.0)
    pos[1] = (1.0, 0.0, 0.0)
    pos[2] = (9.0, 0.0, 0.0)
    # **Adjacency is directed**: a face contributes k -> k+1 only. Two faces with
    # opposite winding are needed for vertex 1 to see both of its neighbours.
    faces = np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int64)
    texture, mask = _blank()
    texture[0, 0] = (1.0, 0.0, 0.0)
    texture[0, 2] = (0.0, 0.0, 1.0)
    mask[0, 0] = mask[0, 2] = 255
    out_texture, _ = inpaint.meshVerticeInpaint(texture, mask, pos, uv, faces, faces)
    red, blue = float(out_texture[0, 1][0]), float(out_texture[0, 1][2])
    assert red > blue, f"the far neighbour won: red={red}, blue={blue}"
    # 1/1 against 1/64: red should dominate by roughly that ratio.
    assert red > 0.9, red


def test_adjacency_is_directed() -> None:
    """**A face only links k to k+1.** Upstream never symmetrises the graph.

    Vertex 1's single out-edge in a lone triangle points at vertex 2, so vertex 1
    takes vertex 2's colour even though vertex 0 is nearer. Symmetrising the
    graph would quietly change every baked texture, so pin the asymmetry here.
    """
    pos, uv, faces = _strip(3)
    pos[0] = (0.0, 0.0, 0.0)
    pos[1] = (1.0, 0.0, 0.0)
    pos[2] = (9.0, 0.0, 0.0)
    texture, mask = _blank()
    texture[0, 0] = (1.0, 0.0, 0.0)
    texture[0, 2] = (0.0, 0.0, 1.0)
    mask[0, 0] = mask[0, 2] = 255
    out_texture, _ = inpaint.meshVerticeInpaint(texture, mask, pos, uv, faces, faces)
    assert np.allclose(out_texture[0, 1], (0.0, 0.0, 1.0)), out_texture[0, 1]


def test_an_unreachable_hole_stays_empty() -> None:
    """No painted neighbour means no colour invented, and the loop still ends."""
    pos, uv, faces = _strip(4)
    texture, mask = _blank()
    out_texture, out_mask = inpaint.meshVerticeInpaint(texture, mask, pos, uv, faces, faces)
    assert out_mask.sum() == 0, "a texel was marked painted with nothing to paint from"
    assert np.allclose(out_texture, 0.0)


def test_only_smooth_is_accepted() -> None:
    """An unimplemented method fails loudly instead of returning something else."""
    pos, uv, faces = _strip(3)
    texture, mask = _blank()
    try:
        inpaint.meshVerticeInpaint(texture, mask, pos, uv, faces, faces, method="forward")
    except NotImplementedError:
        return
    raise AssertionError("method='forward' was accepted")


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
