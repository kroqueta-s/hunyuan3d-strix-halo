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
        # The version of `docs/runner_contract.md` this was written against.
        # **A caller uses it to explain an absence**, never to refuse a runner.
        "contract": 3,
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
        # **Every method that is not `image_to_mesh` declares its own settings**
        # (contract §3). Without this a caller has to guess, and the guess it
        # made was "the same as the shape stage" - which this runner then threw
        # away without a word.
        "method_params": {
            "texture_mesh": {
                "rembg": {"type": "bool", "default": True},
                "save_glb": {"type": "bool", "default": False},
            },
        },
        "params": {
            "steps": {"type": "int", "default": 30, "min": 1, "max": 200},
            "octree_resolution": {"type": "int", "default": 384, "min": 64, "max": 768},
            "guidance_scale": {"type": "float", "default": 5.0, "min": 0.0, "max": 20.0},
            "seed": {"type": "int", "default": 0, "min": 0},
            "rembg": {"type": "bool", "default": True},
            # **Accepted, so declared** (contract §3). It was read by
            # `image_to_mesh` and left out of this table, which is the same
            # mistake as declaring one that does nothing, pointing the other
            # way: a caller had no way to know it existed. Empty means the
            # `.env` default.
            "rembg_model": {"type": "str", "default": ""},
            "texture": {"type": "bool", "default": False},
        },
        "notes": (
            "On gfx1151, SDPA falls back to fp32 over 4 chunked heads when fast attention "
            "is unavailable, and enable_flashvdm is required. rembg cannot be skipped "
            "(ImageProcessorV2.recenter does nothing useful on a 3-channel image). "
            "Generation time varies between 235 and 921 seconds for identical settings, "
            "so never use it as a pass/fail signal. "
            "The mesh comes back Y-up (measured 2026-09-03 against the image it was "
            "generated from), at the scale of the input. Which way is forward has not "
            "been measured, so forward_axis is null rather than guessed."
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

# **What `texture_mesh` takes, which is not what `image_to_mesh` takes.** It is a
# separate method on a mesh from anywhere, so a shape model's `steps` and
# `octree_resolution` mean nothing to it. They used to be accepted and silently
# dropped, which is worse than refusing them: a caller could change a setting,
# see no error, and get the same result.
_ALLOWED_TEXTURE = frozenset({"rembg", "save_glb"})


def m_texture_mesh(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Texture an existing mesh from a reference image.

    The mesh can come from anywhere: this runner's own output, another
    generator, or a hand-made model.

    **The geometry does change.** Upstream decimates to 40,000 faces before
    unwrapping UVs, so a dense input comes back much lighter. `n_faces` in the
    result says what came out; keep the original if you need the detail.
    """
    # **Refuse what this method never declared** (contract §4). A shape setting
    # sent here used to be accepted and dropped without a word, so a caller could
    # change `steps`, see no error, and get an identical bake.
    unknown = set(params) - _ALLOWED_TEXTURE - {"mesh_path", "image_path", "out_dir"}
    if unknown:
        raise ValueError(
            f"unknown parameters: {sorted(unknown)} "
            f"(texture_mesh accepts: {sorted(_ALLOWED_TEXTURE)})"
        )

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
        # **This stage keeps the frame it is given** (measured 2026-09-03: a bake
        # returned its input's bounding box to within 0.07 %, axes in the same
        # order). But it takes a mesh from anywhere - that is the point of it
        # being a separate method - so **this runner does not know what that
        # frame was** and will not invent one. The caller has it: whatever the
        # mesh going in was reported as, the mesh coming out is the same.
        "up_axis": None,
        "forward_axis": None,
        "params_used": {
            "rembg": bool(params.get("rembg", True)),
            "save_glb": bool(params.get("save_glb", False)),
        },
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
    # **Written beside its final name, then renamed** (contract §9). A cancel
    # ends this process outright, and a run killed halfway through writing a
    # million faces otherwise leaves a truncated file that looks finished.
    staging = out_dir / "raw.ply.part"
    result.mesh.export(str(staging), file_type="ply")
    os.replace(staging, mesh_path)

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
        # **Measured on 2026-09-03, and it is not what anyone would have
        # guessed.** Upstream returns a Y-up mesh: a generated model taken as
        # Z-up arrives lying on its back, which renders perfectly correctly and
        # prints with every joint in the wrong plane. The method was to turn the
        # mesh every way a right-handed frame allows and compare its silhouette
        # to the very image it was generated from; `up: y` won by 0.29 IoU over
        # the best candidate that disagreed, and hi3dgen - which reports `z` on
        # its own say-so - measured `z` by the same method, as a control.
        #
        # **Which way is forward is still not known.** The object is nearly
        # symmetric front to back, so the two candidates were separated by 0.01
        # IoU, which is not a measurement. `null` says so; a caller stands the
        # mesh upright and leaves the facing alone.
        "up_axis": "y",
        "forward_axis": None,
        "params_used": {
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


def watch_parent(interval_sec: float = 2.0) -> None:
    """End this process if the caller that started it goes away.

    **This is the orphan case nothing else covers.** hearth stops its runners
    when it shuts down, and a caller that kills hearth kills the whole tree -
    but a hearth that *crashes* does neither. On Windows the child simply
    carries on, holding the entire card, and **nothing anywhere errors**:
    everything afterwards is several times slower for a reason nobody can see.

    Reporting progress or reading stdin is not enough on its own. Both fail once
    the caller's pipes close, which covers most of a run - but not the middle of
    a long kernel, which is exactly when there is most to lose.

    Two things about how this is done, both measured rather than assumed:

    - **The process to watch is the one `HEARTH_PARENT_PID` names**, not this
      process's own parent. A venv's `python.exe` re-executes the base
      interpreter, so the runner's parent is a launcher that outlives hearth by
      design; watching it would never fire. `os.getppid()` is the fallback for
      being run by hand.
    - **`os.getppid()` cannot detect a dead parent on Windows.** A process whose
      parent dies is not reparented there, so the field keeps naming the dead
      one. Holding a handle from the start and waiting on it does work: the
      handle stays valid after the process exits, and a reused id cannot fool
      it.

    `os._exit` rather than a clean exit on purpose: this fires on a thread while
    a generation may be mid-kernel, and unwinding a model from another thread is
    not something to attempt. The weights are in VRAM, not on disk, so there is
    nothing to lose by leaving abruptly.

    Args:
        interval_sec: How often to look, where waiting on a handle is not
            available. Two seconds is far below the cost of noticing an orphan
            any other way.
    """
    named = os.environ.get("HEARTH_PARENT_PID", "").strip()
    watched = int(named) if named.isdigit() else os.getppid()

    def gone() -> None:
        # **Saying so must never stop it leaving.** stderr is a pipe to the
        # process that just died, so writing to it raises - and an exception
        # here would kill this thread and leave the runner holding the card,
        # which is the entire failure being prevented.
        try:
            print(
                f"[{NAME}] the process that started this runner is gone; "
                "exiting so the card is freed",
                file=sys.stderr,
                flush=True,
            )
        except OSError:
            pass
        os._exit(0)

    def watch() -> None:
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415 - only needed here, and only on Windows
            from ctypes import wintypes  # noqa: PLC0415 - absent on other platforms

            synchronize = 0x00100000
            infinite = 0xFFFFFFFF
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            handle = kernel32.OpenProcess(synchronize, False, watched)
            if handle:
                # Blocks until that process exits, however long that takes.
                kernel32.WaitForSingleObject(handle, infinite)
                gone()
                return
            # No handle: fall through to polling, which is worse but not nothing.
        while True:
            time.sleep(interval_sec)
            if os.getppid() != watched:
                gone()

    threading.Thread(target=watch, name=f"{NAME}-parent-watch", daemon=True).start()


def main() -> int:
    """Handle requests one at a time, in order.

    Returns:
        The exit code. 0 on a clean exit.
    """
    out = install_stdout_guard()
    # **Before anything is loaded.** A runner that has already taken the card is
    # exactly the one worth ending.
    watch_parent()
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
