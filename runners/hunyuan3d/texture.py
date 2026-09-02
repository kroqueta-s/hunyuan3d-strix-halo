# SPDX-License-Identifier: MIT
"""Hunyuan3D 2.1's texture stage (`hy3dpaint`). **Mesh plus image to a textured mesh.**

Upstream looks at the mesh from several viewpoints, has a diffusion model paint
what each view should look like, and bakes the result into a UV texture. **There
is a single entry point**:

```python
pipeline(mesh_path=..., image_path=..., output_mesh_path=...)
```

so **one function covers both jobs**:

1. **Texturing at generation time** - hand it the mesh the shape stage produced.
2. **Texturing an existing mesh** - hand it a mesh from anywhere.

**Seven things have to be arranged first**, all of them at launch time.
**Upstream code is never patched.**

1. `custom_rasterizer` (a CUDA extension) is replaced by the pure-torch
   `raster.install()`.
2. `mesh_inpaint_processor` (a C++ extension) is replaced by the pure-python
   `inpaint.install()`. **There is no C++ compiler on this machine.**
3. `snapshot_download` is pointed at the local weights. Upstream goes to the hub
   even when the files are already on disk.
4. `trust_remote_code=True` is passed for upstream's own custom pipeline, and
   only for that one.
5. The positional argument of the decimation call is read as a face count again
   (trimesh 5 reordered it).
6. `torchvision.transforms.functional_tensor` is put back, because `basicsr`
   imports it and torchvision 0.17 removed it.
7. The `bpy` stand-in, below.

**Upstream resolves its config and some weights by relative path**
(`hy3dpaint/cfgs/...`, `ckpt/RealESRGAN_x4plus.pth`), so the working directory
moves into the upstream clone before it is called and is restored in a
`finally`.

## **`bpy` is not installed** (the one compromise here)

`textureGenPipeline` imports `convert_obj_to_glb` at module level, and its
`mesh_utils.py` does an unconditional `import bpy` (Blender). **Installing `bpy`
would mean this repository could no longer call itself MIT**, so it is not
installed.

`bpy` is only genuinely needed to convert to GLB (`save_glb=True`), so a
stand-in goes in that **imports fine and raises when called**, and the result
comes out as `.obj` plus texture images. **A real `bpy` is not used even if one
is present** - its presence is never checked, because checking means importing.
This is the same approach `runners/trellis/shims.py` takes for `open3d`. If GLB
is wanted, convert downstream (in `forge` or the Blender add-on).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config, inpaint, raster

_PIPELINE: Any = None
_LOAD_SEC: float = 0.0


@dataclass(frozen=True)
class TextureResult:
    """What the texture stage produced, and what it measured.

    mesh_path: The textured mesh (`.obj`, plus `.glb` if it was asked for).
    load_sec: Seconds spent loading the pipeline. Zero after the first call.
    texture_sec: Seconds spent baking.
    n_faces: Faces in the mesh that came out. **Far fewer than went in**:
        upstream decimates to 40,000 faces before unwrapping UVs
        (`mesh_simplify_trimesh`). Use `image_to_mesh`'s `raw.ply` when the
        geometry is what matters.
    """

    mesh_path: Path
    load_sec: float
    texture_sec: float
    n_faces: int


class _AbsentBpy:
    """A stand-in for `bpy` that **follows attribute lookups but raises when called**.

    Upstream imports `bpy` at module level and only actually uses it to convert
    to GLB. **Installing `bpy` would cost this repository its MIT licence**, so
    the import is allowed to succeed and any use of it is stopped.
    **It never quietly returns something else.**
    """

    def __init__(self, path: str = "bpy") -> None:
        self._path = path

    def __getattr__(self, attr: str) -> _AbsentBpy:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _AbsentBpy(f"{self._path}.{attr}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"{self._path} is not available: bpy (Blender) is GPL and is not installed. "
            "Convert downstream if GLB is needed - the texture is already written "
            "as .obj plus image files."
        )

    def __repr__(self) -> str:
        return f"<absent {self._path}>"


def _install_local_snapshot() -> None:
    """Point `snapshot_download` at the weights that are already on disk.

    Upstream's `multiview_utils` **always** writes it this way:

    ```python
    model_path = huggingface_hub.snapshot_download(
        repo_id=config.multiview_pretrained_path,
        allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
    )
    ```

    There is no `local_dir`, so it **ignores the 6.4 GB the installer put there
    and goes to the hub**. Unauthenticated downloads stall at zero bytes here,
    and did (2026-09-02: six minutes at zero, the Xet `.incomplete` file never
    growing). Not making the call at all is the only reliable answer.

    **Only the repository root is returned**; the sub-directory below it is
    upstream's business. `HUNYUAN3D_TEXTURE_SUBFOLDER` is used for the stock
    check alone, so **keep it equal to the name upstream joins on**.

    Raises:
        FileNotFoundError: If the texture weights are not on disk.
    """
    local = config.SHAPE_MODELS_DIR / config.SHAPE_MODEL_ID.replace("/", os.sep)
    if not (local / config.TEXTURE_SUBFOLDER).is_dir():
        raise FileNotFoundError(
            f"texture weights not found: {local / config.TEXTURE_SUBFOLDER} "
            "(run install.ps1 -Texture to fetch hunyuan3d-paintpbr-v2-1)"
        )

    import huggingface_hub

    original = huggingface_hub.snapshot_download
    # An unload followed by a reload would otherwise wrap the wrapper.
    if getattr(original, "_hearth_local", False):
        return

    def snapshot_download(*args: Any, **kwargs: Any) -> str:
        repo_id = kwargs.get("repo_id") or (args[0] if args else None)
        if repo_id == config.SHAPE_MODEL_ID and "local_dir" not in kwargs:
            return str(local)
        return original(*args, **kwargs)

    snapshot_download._hearth_local = True  # type: ignore[attr-defined]
    huggingface_hub.snapshot_download = snapshot_download


def _install_face_count_decimation() -> None:
    """Read the decimation's positional argument the way upstream meant it.

    Upstream calls `mesh.simplify_quadric_decimation(40000)` before unwrapping
    UVs. **In trimesh 4 the first argument was `face_count`; in 5 it is
    `percent`**, so the call dies with `target_reduction must be between 0 and 1`.

    A positional argument **greater than 1** is therefore passed on as a face
    count. A ratio is always between 0 and 1, so there is nothing to confuse,
    and a correct call such as `percent=0.5` is left alone.
    """
    import trimesh

    original = trimesh.Trimesh.simplify_quadric_decimation
    if getattr(original, "_hearth_face_count", False):
        return

    def simplify_quadric_decimation(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], (int, float)) and args[0] > 1:
            return original(self, face_count=int(args[0]), **kwargs)
        return original(self, *args, **kwargs)

    simplify_quadric_decimation._hearth_face_count = True  # type: ignore[attr-defined]
    trimesh.Trimesh.simplify_quadric_decimation = simplify_quadric_decimation


def _install_trusted_custom_pipeline(root: Path) -> None:
    """Allow `trust_remote_code=True`, but only for upstream's own custom pipeline.

    Upstream loads its `hy3dpaint/hunyuanpaintpbr/pipeline.py` as a diffusers
    "custom pipeline", and **diffusers 0.40 refuses to execute that without
    consent**. Upstream's call has no `trust_remote_code`, so it is added here.

    **It is added only when the code lives inside the upstream clone.** Trusting
    whatever arrives from the hub would turn this stand-in into the hole it is
    meant to avoid.

    Args:
        root: The upstream clone. Nothing outside it is trusted.
    """
    from diffusers import DiffusionPipeline

    original = DiffusionPipeline.from_pretrained.__func__  # type: ignore[attr-defined]
    if getattr(original, "_hearth_trusted", False):
        return

    def from_pretrained(cls: Any, *args: Any, **kwargs: Any) -> Any:
        custom = kwargs.get("custom_pipeline")
        if custom is not None and "trust_remote_code" not in kwargs:
            try:
                inside = Path(str(custom)).resolve().is_relative_to(root.resolve())
            except (OSError, ValueError):
                inside = False
            if inside:
                kwargs["trust_remote_code"] = True
        return original(cls, *args, **kwargs)

    from_pretrained._hearth_trusted = True  # type: ignore[attr-defined]
    DiffusionPipeline.from_pretrained = classmethod(from_pretrained)


def _install_functional_tensor() -> None:
    """Put `torchvision.transforms.functional_tensor` back.

    Upstream upscales each view with RealESRGAN, whose `basicsr` does
    `from torchvision.transforms.functional_tensor import rgb_to_grayscale`.
    **That private module was removed in torchvision 0.17** (its contents moved
    into `functional`). basicsr is unmaintained, so the alias is made here.
    """
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401

        return
    except ImportError:
        pass
    import torchvision.transforms.functional as tv_functional

    sys.modules[name] = tv_functional


def _install_absent_bpy() -> None:
    """Put the `bpy` stand-in into `sys.modules`.

    **A real `bpy` is not used even if one is installed.** Importing `bpy` at
    all would cost this repository its MIT licence, so its presence is never
    checked - checking means importing. `test_nothing_imports_bpy` pins this.
    """
    if "bpy" in sys.modules:
        return
    module = types.ModuleType("bpy")
    module.__file__ = "<hearth shim: bpy>"

    def _absent(attr: str) -> Any:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _AbsentBpy(f"bpy.{attr}")

    module.__getattr__ = _absent  # type: ignore[attr-defined]
    sys.modules["bpy"] = module


@contextlib.contextmanager
def _in_upstream() -> Iterator[Path]:
    """Make the upstream clone the working directory.

    **Upstream opens its config and the RealESRGAN weights by relative path**,
    so without this `hy3dpaint/cfgs/hunyuan-paint-pbr.yaml` is not found.

    Yields:
        The upstream root.

    Raises:
        FileNotFoundError: If the clone is missing.
    """
    root = config.SHAPE_REPO.parent  # .../Hunyuan3D-2.1/hy3dshape -> .../Hunyuan3D-2.1
    paint = root / "hy3dpaint"
    if not paint.is_dir():
        raise FileNotFoundError(f"hy3dpaint not found under {root} (check HUNYUAN3D_SHAPE_REPO)")
    previous = Path.cwd()
    os.chdir(root)
    # **Both are needed.** The root makes `hy3dpaint.textureGenPipeline`
    # importable; hy3dpaint itself makes the `from DifferentiableRenderer...`
    # inside it work, because upstream imports relative to its own directory.
    for path in (str(root), str(paint)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield root
    finally:
        os.chdir(previous)


def load_pipeline(progress: Callable[[str, str], None] | None = None) -> Any:
    """Load the texture pipeline once per process.

    **`raster.install()` runs before `hy3dpaint` is imported.**
    `MeshRender.__init__` does `import custom_rasterizer` when `raster_mode` is
    `"cr"`, so installing it afterwards is too late.

    Returns:
        A `Hunyuan3DPaintPipeline`.
    """
    global _PIPELINE, _LOAD_SEC
    if _PIPELINE is not None:
        return _PIPELINE

    def say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    say("texture_shim", "replacing custom_rasterizer with the pure-torch stand-in")
    raster.install()
    _install_local_snapshot()
    _install_functional_tensor()
    _install_absent_bpy()

    started = time.perf_counter()
    with _in_upstream() as root:
        # **Before `MeshRender`.** Upstream swallows the inpaint import failure,
        # so missing this shows up only as a `NameError` after minutes of baking.
        inpaint.install()
        _install_trusted_custom_pipeline(root)
        _install_face_count_decimation()
        say("texture_load", "loading the texture weights (tens of seconds on the first run)")
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        cfg = Hunyuan3DPaintConfig(
            max_num_view=config.TEXTURE_MAX_VIEWS,
            resolution=config.TEXTURE_VIEW_RESOLUTION,
        )
        cfg.render_size = config.TEXTURE_RENDER_SIZE
        cfg.texture_size = config.TEXTURE_SIZE
        # **Upstream's default is the hub id `facebook/dinov2-giant`**, which it
        # looks up every time. A local directory is read as-is by
        # `AutoModel.from_pretrained`.
        if config.TEXTURE_DINO_DIR.is_dir():
            cfg.dino_ckpt_path = str(config.TEXTURE_DINO_DIR)
        _PIPELINE = Hunyuan3DPaintPipeline(cfg)
    _LOAD_SEC = time.perf_counter() - started
    say("texture_loaded", f"weights loaded ({_LOAD_SEC:.1f}s)")
    return _PIPELINE


def unload_pipeline() -> bool:
    """Release the texture stage and give the VRAM back.

    Returns:
        Whether anything was actually released.
    """
    global _PIPELINE, _LOAD_SEC
    if _PIPELINE is None:
        return False
    _PIPELINE = None
    _LOAD_SEC = 0.0
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def texture_mesh(
    mesh_path: Path,
    image_path: Path,
    out_dir: Path,
    *,
    save_glb: bool = False,
    progress: Callable[[str, str], None] | None = None,
) -> TextureResult:
    """Turn one mesh and one image into a textured mesh.

    Args:
        mesh_path: The mesh to texture.
        image_path: The image the appearance is taken from. **Its background
            should already be removed.**
        out_dir: Where the results go.
        save_glb: Whether to also write `.glb`. **False by default**: upstream's
            GLB conversion goes through `bpy` (GPL), a path this repository
            cannot take.
        progress: Where stage notifications go.

    Returns:
        A `TextureResult`.

    Raises:
        FileNotFoundError: If the mesh or the image is missing.
    """
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(progress)
    output = out_dir / "textured.obj"

    # **Upstream writes its remesh next to the input mesh**
    # (`white_mesh_remesh.obj`). Copy the input into the output directory first,
    # so texturing someone's own mesh does not litter their directory.
    staged = out_dir / f"input{mesh_path.suffix}"
    if staged.resolve() != mesh_path.resolve():
        shutil.copyfile(mesh_path, staged)

    if progress is not None:
        progress("texture", "painting the views and baking them into a texture (minutes)")
    started = time.perf_counter()
    # **The bake writes no intermediate files at all.** Without a heartbeat
    # there is no way to tell progress from a hang across a run measured at
    # over 40 minutes (on 2026-09-02 that was 13 minutes of not knowing). The
    # shape stage's watcher does the job unchanged.
    from .shape import _DeviceWatch

    with _in_upstream(), _DeviceWatch(
        progress=progress,
        stage="texture",
        heartbeat_sec=config.HEARTBEAT_SEC,
        limit_gb=config.VRAM_LIMIT_GB,
    ):
        pipeline(
            mesh_path=str(staged),
            image_path=str(image_path),
            output_mesh_path=str(output),
            save_glb=save_glb,
        )
    texture_sec = time.perf_counter() - started

    if not output.is_file():
        raise RuntimeError(f"no textured mesh was written: {output}")

    import trimesh

    baked = trimesh.load(str(output), force="mesh")
    return TextureResult(
        mesh_path=output,
        load_sec=_LOAD_SEC,
        texture_sec=texture_sec,
        n_faces=int(len(baked.faces)),
    )
