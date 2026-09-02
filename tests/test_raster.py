# SPDX-License-Identifier: MIT
"""Verify the `custom_rasterizer` stand-in (**get the numbers right before baking a texture**).

The texture stage bakes what the rasterizer reports, so a wrong face index puts
colour on the wrong triangle and wrong barycentric weights smear it. Both are
checkable without the original CUDA extension:

- **Barycentric weights sum to 1** on every covered pixel.
- **Interpolating the vertex positions with those weights returns the pixel's
  own position.** That is the definition of barycentric coordinates, so it
  catches a wrong vertex order or a missing division just as well as a
  reference implementation would.
- A box hidden inside a box is never visible (the z-buffer works).
- Background pixels report face 0 **and zero weights**, which upstream's
  `interpolate` relies on.

Run it with a runner's virtual environment (torch is required)::

    <venv>\\Scripts\\python.exe .\\tests\\test_hunyuan3d_raster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.hunyuan3d import raster  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = (256, 256)


def _ortho(scale: float = 1.2) -> torch.Tensor:
    """Orthographic projection, which is what the texture stage uses."""
    m = torch.eye(4, device=DEVICE, dtype=torch.float32)
    m[0, 0] = 1.0 / scale
    m[1, 1] = 1.0 / scale
    m[2, 2] = -1.0 / 10.0
    return m


def _quad(z: float = 0.0, half: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """A square facing the camera, as two triangles."""
    verts = torch.tensor(
        [
            [-half, -half, z, 1.0],
            [half, -half, z, 1.0],
            [half, half, z, 1.0],
            [-half, half, z, 1.0],
        ],
        device=DEVICE,
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=DEVICE, dtype=torch.int64)
    return verts, faces


def _project(verts: torch.Tensor) -> torch.Tensor:
    return (verts @ _ortho().transpose(-1, -2))[None]


def test_barycentric_sums_to_one() -> None:
    """**Weights sum to 1 wherever a face was hit** (and are 0 on background)."""
    verts, faces = _quad()
    findices, bary = raster.rasterize(_project(verts), faces, RES)
    hit = findices > 0
    assert bool(hit.any()), "nothing was drawn"
    total = bary[hit].sum(dim=-1)
    err = (total - 1.0).abs().max().item()
    assert err < 1e-4, f"weights do not sum to 1: max error {err}"
    assert bary[~hit].abs().max().item() == 0.0, "background weights are not zero"


def test_barycentric_reconstructs_the_pixel_position() -> None:
    """**Interpolating the corner positions returns the pixel's own position.**

    This is what makes the weights usable for baking: upstream feeds them UVs
    and expects the surface point under the pixel.
    """
    verts, faces = _quad()
    findices, bary = raster.rasterize(_project(verts), faces, RES)
    hit = findices > 0
    height, width = RES

    # Where each hit pixel says the surface is, in clip space.
    clip = _project(verts)[0]
    tri = clip[faces.long()][(findices[hit].long() - 1)]  # [n, 3, 4]
    got = (bary[hit].unsqueeze(-1) * tri).sum(dim=-2)  # [n, 4]
    ndc = got[:, :3] / got[:, 3:4]
    sx = (ndc[:, 0] * 0.5 + 0.5) * width
    sy = (ndc[:, 1] * 0.5 + 0.5) * height

    flat = torch.nonzero(hit.reshape(-1), as_tuple=False).squeeze(1)
    want_x = (flat % width).float() + 0.5
    want_y = (flat // width).float() + 0.5
    err = max((sx - want_x).abs().max().item(), (sy - want_y).abs().max().item())
    assert err < 1e-2, f"interpolated position is off by {err} pixels"


def test_interpolate_matches_upstream_shapes() -> None:
    """`interpolate` returns what upstream expects, and background reads as 0."""
    verts, faces = _quad()
    findices, bary = raster.rasterize(_project(verts), faces, RES)
    # A colour per vertex; the first channel is the vertex index.
    col = torch.zeros(1, verts.shape[0], 3, device=DEVICE)
    col[0, :, 0] = torch.arange(verts.shape[0], device=DEVICE, dtype=torch.float32)
    out = raster.interpolate(col, findices, bary, faces)
    assert tuple(out.shape) == (1, RES[0], RES[1], 3), out.shape
    hit = findices > 0
    assert out[0][~hit].abs().max().item() == 0.0, "background did not interpolate to 0"
    inside = out[0][hit][:, 0]
    assert inside.min().item() >= -1e-4, "vertex index went negative"
    assert inside.max().item() <= verts.shape[0] - 1 + 1e-4, "vertex index overshot"


def test_hidden_box_is_never_visible() -> None:
    """A quad behind another is never the winner (the z-buffer works)."""
    front_v, faces = _quad(z=0.0, half=0.5)
    back_v, _ = _quad(z=-2.0, half=0.5)
    verts = torch.cat([front_v, back_v], dim=0)
    all_faces = torch.cat([faces, faces + front_v.shape[0]], dim=0)
    findices, _ = raster.rasterize(_project(verts), all_faces, RES)
    seen = torch.unique(findices[findices > 0]).tolist()
    assert seen and max(seen) <= 2, f"a hidden face won a pixel: {seen}"


def main() -> int:
    """Run every test."""
    print(f"device: {DEVICE}")
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
