# SPDX-License-Identifier: MIT
"""The texture stage's files, turned into a PLY and a GLB. **No graphics card needed.**

What is pinned:

- **The PLY carries the texture on its vertices**, sampled where each vertex's UV
  points, and the same faces as the `.obj`.
- **The GLB carries both textures**: the colour one, and a metallic-roughness one
  with metalness in blue and roughness in green, as glTF defines it. trimesh does
  not read `map_Pm` from an `.mtl`, so this is the part that would silently go
  missing.
- **A roughness map that is named but not there** falls back to the default
  rather than failing: upstream's own output is exactly that (2026-09-15).

Run it with this repository's virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_export.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.hunyuan3d.export import DEFAULT_ROUGHNESS, export_textured  # noqa: E402

# Two triangles over the unit square, UVs matching positions. The colour image is
# red on its left half and blue on its right, so a vertex at x = 0 samples red
# and one at x = 1 samples blue.
_OBJ = """mtllib textured.mtl
o square
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0.02 0.02
vt 0.98 0.02
vt 0.98 0.98
vt 0.02 0.98
usemtl Material
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""

_MTL = """newmtl Material
Kd 0.8 0.8 0.8
map_Kd textured.jpg
map_Pm textured_metallic.jpg
map_Pr textured_roughness.jpg
"""


def _write_set(where: Path, metal_value: int) -> Path:
    """An upstream-shaped set: `.obj`, `.mtl`, colour and metallic images, no roughness."""
    from PIL import Image

    colour = np.zeros((64, 64, 3), dtype=np.uint8)
    colour[:, :32] = (255, 0, 0)
    colour[:, 32:] = (0, 0, 255)
    Image.fromarray(colour, mode="RGB").save(where / "textured.jpg", quality=100)
    metal = np.full((64, 64), metal_value, dtype=np.uint8)
    Image.fromarray(metal, mode="L").save(where / "textured_metallic.jpg", quality=100)
    (where / "textured.mtl").write_text(_MTL, encoding="utf-8")
    obj = where / "textured.obj"
    obj.write_text(_OBJ, encoding="utf-8")
    return obj


def test_the_ply_carries_the_texture_on_its_vertices() -> None:
    import trimesh

    with tempfile.TemporaryDirectory() as raw:
        where = Path(raw)
        out = export_textured(_write_set(where, 200), where)
        ply = trimesh.load(str(out.ply), process=False)
        assert out.ply.name == "textured.ply", out.ply
        assert len(ply.faces) == 2 and out.n_faces == 2, (len(ply.faces), out.n_faces)
        colours = np.asarray(ply.visual.vertex_colors)[:, :3].astype(int)
        left = colours[np.isclose(ply.vertices[:, 0], 0.0)]
        right = colours[np.isclose(ply.vertices[:, 0], 1.0)]
        assert (left[:, 0] > 200).all() and (left[:, 2] < 60).all(), left
        assert (right[:, 2] > 200).all() and (right[:, 0] < 60).all(), right


def test_the_glb_carries_colour_and_packed_metal_roughness() -> None:
    import trimesh

    with tempfile.TemporaryDirectory() as raw:
        where = Path(raw)
        out = export_textured(_write_set(where, 200), where)
        assert out.metallic_map is True and out.roughness_map is False, out
        glb = trimesh.load(str(out.glb), process=False, force="mesh")
        material = glb.visual.material
        assert material.baseColorTexture is not None, "the colour texture is missing"
        packed = material.metallicRoughnessTexture
        assert packed is not None, "the metallic-roughness texture is missing"
        pixels = np.asarray(packed.convert("RGB")).astype(int)
        # JPEG round trips shift a flat value by a little; blue is metalness.
        assert abs(int(np.median(pixels[..., 2])) - 200) <= 4, np.median(pixels[..., 2])
        expected_rough = round(255 * DEFAULT_ROUGHNESS)
        assert abs(int(np.median(pixels[..., 1])) - expected_rough) <= 4, np.median(pixels[..., 1])
        assert len(glb.faces) == 2, len(glb.faces)


def test_nothing_half_written_is_left_behind() -> None:
    with tempfile.TemporaryDirectory() as raw:
        where = Path(raw)
        export_textured(_write_set(where, 0), where)
        left = sorted(p.name for p in where.glob("*.part"))
        assert not left, left


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  OK   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
