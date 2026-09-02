# SPDX-License-Identifier: MIT
"""The Hunyuan3D 2.1 runner (an implementation of the runner contract).

**This process is the only one that holds torch.** Neither hearth itself nor the
Blender add-on imports it.

The runner imports nothing from hearth, so this repository is self-contained.

Start it (hearth normally spawns it as a child process)::

    .venv\\Scripts\\python.exe -m runners.hunyuan3d
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from . import config, displaykeep, gfxlight

# **Has no effect unless it precedes torch** (setting os.environ later is
# ignored). It makes the flash and memory-efficient kernels available on
# gfx1151, so the slowest path (fp32 over chunked heads) is never needed
# (measured: seq=4096 from 0.135s to 0.012s). Importing config here does not
# pull in torch, because config only reads dotenv.
if config.FAST_ATTENTION:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Same rule: **only effective before torch is imported.** hipBLASLt runs the
# DiT's unbiased projections ~24% faster than rocBLAS (measured 2026-09-02,
# see docs/gemm_profile.md). `metrics.blas_backend` records what was used.
if config.PREFER_HIPBLASLT:
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "1")

NAME = "hunyuan3d"
VERSION = "2.1"


# --- Protocol (same format as hearth's rpc.py, but with no dependency on it) ---
def install_stdout_guard() -> TextIO:
    """Duplicate and hide the real stdout, **redirecting fd 1 itself to stderr**.

    **Call this first.** Upstream code prints freely, and replacing `sys.stdout`
    is not enough: **C extensions write straight to fd 1**, bypassing the Python
    side and corrupting the protocol stream. Measured 2026-09-01: `pymeshfix`
    emitted `Loading ..0%` hundreds of times directly to fd 1.

    So fd 1 is duplicated and reserved for the protocol, and **fd 1 itself is
    pointed at fd 2**. Everything that is not protocol, from Python or from
    native code, then lands on stderr.

    Returns:
        The protocol-only writer.
    """
    fd = os.dup(1)
    os.dup2(2, 1)  # **fd 1 to stderr; output from C extensions lands there too.**
    protocol = os.fdopen(fd, "w", encoding="utf-8", newline="\n", buffering=1)
    sys.stdout = sys.stderr
    return protocol


# **A lock is required because the heartbeat thread writes too.**
# The contract is one JSON object per line; interleaving breaks the reader.
_EMIT_LOCK = threading.Lock()


def emit(out: TextIO, payload: dict[str, Any]) -> None:
    """Write one message per line and always flush. **Thread-safe.**"""
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _EMIT_LOCK:
        out.write(line)
        out.flush()


# --- Methods -----------------------------------------------------------------
def m_capabilities(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Report capabilities. **Answers immediately, without loading the weights.**"""
    return {
        "name": NAME,
        "version": VERSION,
        "capabilities": {
            "image_to_mesh": True,
            # There is no direct text-to-3D path; the image comes from elsewhere.
            "text_to_mesh": False,
            "multi_image_to_mesh": False,
            # The texture stage runs here: its CUDA-only `custom_rasterizer` is
            # replaced by the pure-torch one in `raster.py`.
            "texture": True,
            # Texture an existing mesh from a reference image.
            "texture_mesh": True,
        },
        "params": {
            "steps": {"type": "int", "default": 30, "min": 1, "max": 200},
            "octree_resolution": {"type": "int", "default": 384, "min": 64, "max": 768},
            "guidance_scale": {"type": "float", "default": 5.0, "min": 0.0, "max": 20.0},
            "seed": {"type": "int", "default": 0, "min": 0},
            "rembg": {"type": "bool", "default": True},
            "texture": {"type": "bool", "default": False},
        },
        "notes": (
            "On gfx1151, SDPA falls back to fp32 over 4 chunked heads when fast attention "
            "is unavailable, and enable_flashvdm is required. rembg cannot be skipped "
            "(ImageProcessorV2.recenter does nothing useful on a 3-channel image). "
            "Generation time varies between 235 and 921 seconds for identical settings, "
            "so never use it as a pass/fail signal."
        ),
    }


def m_load(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Load the weights (measured 77.6 s cold, 0 s on later calls)."""
    from . import shape

    progress("load", "loading the Hunyuan3D weights (about 80 seconds on the first run)")
    started = time.perf_counter()
    shape.load_pipeline()
    return {"loaded": True, "elapsed_sec": round(time.perf_counter() - started, 2)}


def m_unload(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Release the weights and give the VRAM back.

    **Both stages are released.** The texture stage keeps its own weights
    resident once used, and hearth expects `unload` to hand the whole GPU back.
    """
    from . import shape, texture

    freed = shape.unload_pipeline()
    freed_texture = texture.unload_pipeline()
    used_gb, _ = shape.device_memory_gb()
    return {
        "unloaded": freed or freed_texture,
        "vram_used_gb": round(used_gb, 2),
    }


_ALLOWED = frozenset(
    {"steps", "octree_resolution", "guidance_scale", "seed", "rembg", "rembg_model", "texture"}
)


def m_texture_mesh(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Texture an existing mesh from a reference image.

    The mesh can come from anywhere: this runner's own output, another
    generator, or a hand-made model.

    **The geometry does change.** Upstream decimates to 40,000 faces before
    unwrapping UVs, so a dense input comes back much lighter. `n_faces` in the
    result says what came out; keep the original if you need the detail.
    """
    from . import background, shape, texture

    # **The shape stage is not needed here.** hearth loads a runner before
    # calling it, so its weights are resident; they would just crowd the texture
    # stage out of the 32 GB of dedicated VRAM.
    if shape.is_loaded():
        progress("unload_shape", "releasing the shape weights (texturing does not need them)")
        shape.unload_pipeline()

    mesh_path = Path(str(params["mesh_path"]))
    image_path = Path(str(params["image_path"]))
    out_dir = Path(str(params["out_dir"]))
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # The reference image wants its background gone, for the same reason the
    # shape stage does: the model otherwise paints the backdrop onto the object.
    use_rembg = bool(params.get("rembg", True))
    if use_rembg:
        progress("rembg", "removing the background from the reference image")
        foreground, _ = background.prepare_image(image_path, rembg=True)
        image_path = out_dir / "texture_reference.png"
        foreground.save(image_path)

    result = texture.texture_mesh(
        mesh_path,
        image_path,
        out_dir,
        # GLB is off by default: upstream converts with bpy (GPL), which this
        # repository does not install.
        save_glb=bool(params.get("save_glb", False)),
        progress=progress,
    )
    return {
        "mesh_path": str(result.mesh_path),
        "source_mesh": str(params["mesh_path"]),
        "input_image": str(image_path),
        "n_faces": result.n_faces,
        "metrics": {
            "load_sec": round(result.load_sec, 2),
            "texture_sec": round(result.texture_sec, 2),
        },
    }


def m_image_to_mesh(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """One image to a raw mesh.

    **Preprocessing (background removal) is this runner's job** and cannot be
    skipped for Hunyuan3D.
    **Scaling to real-world size is not done here.** Millimetres are downstream
    work (meshforge's forge).
    """
    from . import background, shape

    image_path = Path(str(params["image_path"]))
    out_dir = Path(str(params["out_dir"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = set(params) - _ALLOWED - {"image_path", "out_dir"}
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)} (accepted: {sorted(_ALLOWED)})")

    progress("rembg", "removing the background")
    foreground, fraction = background.prepare_image(
        image_path,
        rembg=params.get("rembg"),
        model=params.get("rembg_model"),
    )
    foreground_path = out_dir / "foreground.png"
    foreground.save(foreground_path)

    progress("shape", "generating the 3D shape (several minutes; the time is not consistent)")
    result = shape.generate_mesh(
        foreground,
        steps=params.get("steps"),
        octree_resolution=params.get("octree_resolution"),
        guidance_scale=params.get("guidance_scale"),
        seed=int(params.get("seed", 0)),
        progress=progress,
    )

    progress("export", "writing the mesh")
    mesh_path = out_dir / "raw.ply"
    result.mesh.export(str(mesh_path))

    # **The texture stage is opt-in**: it costs several more minutes and its own
    # weights, and downstream printing does not need it.
    textured_path: str | None = None
    texture_metrics: dict[str, Any] = {}
    if params.get("texture"):
        from . import texture as texture_stage

        # The shape stage keeps its weights resident; the texture stage needs
        # the VRAM, and only one of them fits at a time.
        progress("unload_shape", "releasing the shape weights before texturing")
        shape.unload_pipeline()
        textured = texture_stage.texture_mesh(
            mesh_path, foreground_path, out_dir, progress=progress
        )
        textured_path = str(textured.mesh_path)
        texture_metrics = {
            "texture_load_sec": round(textured.load_sec, 2),
            "texture_sec": round(textured.texture_sec, 2),
            # **The textured mesh is much lighter than raw.ply**: upstream
            # decimates to 40,000 faces before unwrapping UVs.
            "textured_n_faces": textured.n_faces,
        }

    return {
        "mesh_path": str(mesh_path),
        "textured_mesh_path": textured_path,
        "n_vertices": int(len(result.mesh.vertices)),
        "n_faces": int(len(result.mesh.faces)),
        "extra": {
            "foreground": str(foreground_path),
            "foreground_fraction": round(float(fraction), 4),
        },
        "metrics": {
            "load_sec": round(result.load_sec, 2),
            # **Never use this as a pass/fail signal.** It varies between 235 and
            # 921 seconds for identical settings.
            "gen_sec": round(result.gen_sec, 2),
            "vram_peak_gb": round(result.vram_peak_gb, 2),
            # **Whether fast attention is in effect.** Without it generation is
            # several times slower.
            "fast_attention": result.fast_attention,
            # **Which BLAS backend served the GEMMs.** Timings cannot be
            # compared without knowing which one produced them.
            "blas_backend": shape.blas_backend(),
            **texture_metrics,
        },
        "params": {
            "steps": result.steps,
            "octree_resolution": result.octree_resolution,
            "guidance_scale": result.guidance_scale,
            "seed": result.seed,
        },
    }


METHODS = {
    "capabilities": m_capabilities,
    "load": m_load,
    "unload": m_unload,
    "image_to_mesh": m_image_to_mesh,
    "texture_mesh": m_texture_mesh,
}


def main() -> int:
    """Handle requests one at a time, in order.

    Returns:
        The exit code. 0 on a clean exit.
    """
    out = install_stdout_guard()
    print(f"[{NAME}] runner started.", file=sys.stderr)

    for raw in sys.stdin:
        line = raw.lstrip("﻿").strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = int(request["id"])
            method_name = str(request["method"])
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[{NAME}] skipped an unparsable request: {exc}", file=sys.stderr)
            continue

        if method_name == "shutdown":
            emit(out, {"id": request_id, "event": "result", "result": {"bye": True}})
            break

        method = METHODS.get(method_name)
        if method is None:
            emit(
                out,
                {
                    "id": request_id,
                    "event": "error",
                    "error": {"type": "ValueError", "message": f"unknown method: {method_name}"},
                },
            )
            continue

        def progress(
            stage: str,
            message: str = "",
            _id: int = request_id,
            **extra: Any,
        ) -> None:
            # `extra` carries `step` and, when the length is known, `total`.
            # **Nothing estimated ever goes in here** (see steps.py).
            emit(
                out,
                {"id": _id, "event": "progress", "stage": stage, "message": message, **extra},
            )

        # **Render-loop keepalive** (gfxlight.py). Measured to change nothing
        # on the current driver; kept because it costs nothing.
        light: gfxlight.GfxLight | None = None
        if method_name == "image_to_mesh" and config.GFX_KEEPALIVE:
            light = gfxlight.GfxLight()
            light.start()
        # **Display keepalive** (displaykeep.py). With the console display off
        # the driver pins the GPU near 600 MHz and generation runs ~4x slower;
        # holding the display awake prevents that. **Off by default** - it
        # keeps the panel lit (see config.py and gfx1151-gemm
        # docs/displayoff.md).
        keep: displaykeep.DisplayKeep | None = None
        if method_name == "image_to_mesh" and config.DISPLAY_KEEPALIVE:
            keep = displaykeep.DisplayKeep()
            keep.start()
        try:
            result = method(dict(request.get("params") or {}), progress)
            if light is not None and isinstance(result.get("metrics"), dict):
                # Whether it stayed alive to the end. False means it may not
                # have taken effect.
                result["metrics"]["gfx_keepalive"] = light.is_lit()
            if keep is not None and isinstance(result.get("metrics"), dict):
                result["metrics"]["display_keepalive"] = keep.is_held()
            emit(out, {"id": request_id, "event": "result", "result": result})
        except Exception as exc:  # noqa: BLE001 - always answer, whatever happens
            import traceback

            traceback.print_exc()
            emit(
                out,
                {
                    "id": request_id,
                    "event": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        finally:
            if light is not None:
                light.stop()
            if keep is not None:
                keep.stop()

    print(f"[{NAME}] runner exiting.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
