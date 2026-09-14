# SPDX-License-Identifier: MIT
"""The texture stage's result, in the two files a caller can use: a PLY and a GLB.

Upstream writes `textured.obj` with a `.mtl` and images beside it. That is three
or four files that refer to each other, and **the runner contract names a PLY as
`mesh_path`** (§5), with anything else in `extra` (`hearth/docs/protocol.md`
§3.1a). So the set is turned into:

- `textured.ply` - the same geometry, **with the texture sampled onto the
  vertices**. Colours per vertex survive every later step, because they follow
  position.
- `textured.glb` - the same geometry with its UVs, the colour texture, and a
  metallic-roughness texture, in one file.

**Everything is done with trimesh and Pillow**, never `bpy`: upstream's own GLB
conversion imports it, and that would cost this repository its licence.

**trimesh reads only the colour map from an `.mtl`** (measured 2026-09-15 on
upstream's output: `map_Pm` and `map_Pr` are ignored, and the material comes back
with a colour texture alone), so the metallic and roughness maps are read here,
by name, and packed the way glTF wants them: **metalness in blue, roughness in
green**.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Roughness when upstream wrote no roughness map. **Not measured**: the `.mtl`
#: names `textured_roughness.jpg` and the file was not there (2026-09-15), so
#: nothing says what it should be. Fully rough is chosen because a model shown
#: glossier than it was painted is the more misleading of the two mistakes.
DEFAULT_ROUGHNESS = 1.0


@dataclass
class Exported:
    """What was written, and what was found to write it from."""

    ply: Path
    glb: Path
    n_faces: int
    n_vertices: int
    metallic_map: bool
    roughness_map: bool


def _maps(mtl: Path) -> dict[str, Path]:
    """The image each map statement in an `.mtl` names, when the file exists."""
    found: dict[str, Path] = {}
    if not mtl.is_file():
        return found
    for raw in mtl.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split()
        if len(parts) >= 2 and parts[0] in ("map_Kd", "map_Pm", "map_Pr"):
            image = mtl.parent / parts[-1]
            if image.is_file():
                found[parts[0]] = image
    return found


def _write_atomically(data: bytes, target: Path) -> Path:
    """Write beside the name, then rename (runner contract §9)."""
    staging = target.with_name(target.name + ".part")
    staging.write_bytes(data)
    os.replace(staging, target)
    return target


def export_textured(obj_path: Path, out_dir: Path) -> Exported:
    """Turn upstream's textured `.obj` set into `textured.ply` and `textured.glb`.

    Args:
        obj_path: `textured.obj`, with its `.mtl` and images beside it.
        out_dir: Where the two files go.

    Returns:
        The paths, the counts, and which maps were found.

    Raises:
        FileNotFoundError: If the `.obj` is not there.
        ValueError: If it carries no texture to sample.
    """
    import trimesh
    from PIL import Image

    if not obj_path.is_file():
        raise FileNotFoundError(f"textured mesh not found: {obj_path}")
    mesh = trimesh.load(str(obj_path), process=False, force="mesh")
    visual = mesh.visual
    if not isinstance(visual, trimesh.visual.TextureVisuals) or visual.uv is None:
        raise ValueError(f"{obj_path.name} carries no texture coordinates to sample")

    maps = _maps(obj_path.with_suffix(".mtl"))
    colour_path = maps.get("map_Kd")
    colour = (
        Image.open(colour_path).convert("RGB")
        if colour_path is not None
        else getattr(visual.material, "image", None)
    )
    if colour is None:
        raise ValueError(f"{obj_path.name} names no colour texture")

    # --- The PLY: the texture sampled onto each vertex ---------------------
    painted = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        visual=trimesh.visual.TextureVisuals(uv=visual.uv, image=colour).to_color(),
        process=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = _write_atomically(painted.export(file_type="ply"), out_dir / "textured.ply")

    # --- The GLB: colour, and metalness/roughness packed for glTF ------------
    width, height = colour.size
    metal = (
        np.asarray(Image.open(maps["map_Pm"]).convert("L").resize((width, height)))
        if "map_Pm" in maps
        else np.zeros((height, width), dtype=np.uint8)
    )
    rough = (
        np.asarray(Image.open(maps["map_Pr"]).convert("L").resize((width, height)))
        if "map_Pr" in maps
        else np.full((height, width), round(255 * DEFAULT_ROUGHNESS), dtype=np.uint8)
    )
    packed = np.zeros((height, width, 3), dtype=np.uint8)
    packed[..., 1] = rough
    packed[..., 2] = metal
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=colour,
        metallicRoughnessTexture=Image.fromarray(packed, mode="RGB"),
        metallicFactor=1.0,
        roughnessFactor=1.0,
    )
    textured = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        visual=trimesh.visual.TextureVisuals(uv=visual.uv, material=material),
        process=False,
    )
    glb = _write_atomically(textured.export(file_type="glb"), out_dir / "textured.glb")

    return Exported(
        ply=ply,
        glb=glb,
        n_faces=int(len(mesh.faces)),
        n_vertices=int(len(mesh.vertices)),
        metallic_map="map_Pm" in maps,
        roughness_map="map_Pr" in maps,
    )
