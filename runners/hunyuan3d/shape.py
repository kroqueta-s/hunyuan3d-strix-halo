# SPDX-License-Identifier: MIT
"""The Hunyuan3D 2.1 shape stage, the core of this runner.

One foreground image goes to 3D-native latent diffusion, to an SDF, through
marching cubes, and out as a watertight mesh. The texture stage is not used:
the intended output is 3D printing, and skipping it avoids a CUDA dependency.

**This module runs only inside this runner's own virtual environment**, which
needs torch with ROCm and the Hunyuan3D repository.

Two things are required on gfx1151 / Windows / ROCm: `_install_sdpa_shim` and
`enable_flashvdm`. Dropping either does not produce a failure but **silent
corruption or a run that never ends**, so the reasoning is recorded in the
docstrings where each one lives.
"""

from __future__ import annotations

import contextlib
import gc
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from . import config

# Imported by name, not as a module: `generate_mesh` has a local called `steps`.
from .steps import StepCounter, count_scheduler

BACKEND = "hunyuan3d21"

_PIPELINE: Any = None
_LOAD_SEC: float = 0.0
# Counts the denoising loop. One per pipeline, rebound for each request.
_STEPS = StepCounter()
# Whether fast attention (AOTriton) is in effect. **Recorded in metrics.**
_FAST_ATTENTION: bool = False


class _DeviceWatch:
    """Watcher thread that tracks device memory and **reports liveness at a fixed interval**.

    It used to only take a peak reading. That alone leaves no way to tell from
    outside whether a long stage is progressing or stuck; on 2026-09-01 runs
    were repeatedly allowed to continue silently for more than 12 minutes.

    Three things are watched:

    1. **Liveness** (`heartbeat`): elapsed time and VRAM every 10 seconds by
       default. The caller can treat a gap as "not progressing".
    2. **Dedicated VRAM overflow** (`vram_over`). **There are only 32 GB of
       dedicated VRAM.** The total reported by `torch.cuda.mem_get_info`
       (43.87 GB) is a lie that counts shared memory, so spilling raises nothing
       and **silently becomes several times slower**. Crossing the line is
       reported the moment it happens.
    3. The peak (still reported in `metrics`).

    **This thread calls `progress`, so the caller's emit must be lock-protected.**
    """

    def __init__(
        self,
        progress: Callable[[str, str], None] | None = None,
        stage: str = "",
        interval: float = 2.0,
        heartbeat_sec: float = 10.0,
        limit_gb: float = 0.0,
    ) -> None:
        self.interval = interval
        self.heartbeat_sec = heartbeat_sec
        self.limit_gb = limit_gb
        self.stage = stage
        self.peak_used_gb = 0.0
        self.exceeded = False
        self._progress = progress
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _say(self, stage: str, message: str) -> None:
        if self._progress is not None:
            self._progress(stage, message)

    def _run(self) -> None:
        started = time.perf_counter()
        last_beat = started
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024**3
            if used > self.peak_used_gb:
                self.peak_used_gb = used
            now = time.perf_counter()
            if self.limit_gb > 0 and used > self.limit_gb and not self.exceeded:
                self.exceeded = True
                self._say(
                    "vram_over",
                    f"**dedicated VRAM exceeded** ({used:.2f}GB > {self.limit_gb:.2f}GB). "
                    "It is spilling into shared memory, so waiting only means slower",
                )
            if now - last_beat >= self.heartbeat_sec:
                last_beat = now
                self._say(
                    "heartbeat",
                    f"{self.stage or 'running'} {now - started:.0f}s elapsed / "
                    f"VRAM {used:.2f}GB (peak {self.peak_used_gb:.2f}GB)",
                )
            self._stop.wait(self.interval)

    def __enter__(self) -> _DeviceWatch:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join(timeout=5)


# **Capture the real function before replacing it**, or a call from inside the shim
# would call the shim itself.
_TORCH_SDPA = F.scaled_dot_product_attention


def fast_attention_available() -> bool:
    """Test on a small tensor whether flash or mem-efficient **actually runs**.

    On gfx1151 the AOTriton implementations become available only when
    `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is set **before torch is
    imported** (measured: setting `os.environ` afterwards has no effect).
    """
    if not torch.cuda.is_available():
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return False
    probe = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(backend):
                _TORCH_SDPA(probe, probe, probe)
            return True
        except Exception:  # noqa: BLE001 - only the availability matters
            continue
    return False


def _install_sdpa_shim(head_chunk: int, fp32: bool | None = None) -> bool:
    """Replace SDPA for gfx1151 / Windows / ROCm.

    It was originally a fix for two problems at once:

    1. **No backend available.** hunyuandit.py calls
       ``sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)``
       in three places, and **that request blocked the only backend that worked
       (math)**.
    2. **fp16 math breaking down numerically.** The math backend has no online
       softmax and materialises ``q @ k^T`` (seq=4096) in the input dtype. It
       exceeds the fp16 limit of 65504 and turns the SDF field into noise
       (reproduced deterministically with a mirrored input).

    **The premise changed on 2026-09-01.** Setting
    `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` before torch **makes flash and
    mem-efficient available** (measured: seq=4096 from 0.135s to 0.012s). Flash
    uses an online softmax and never materialises `q @ k^T`, so **the cause of
    problem 2 disappears entirely.**

    So when the fast path is available, **only the `sdp_kernel` block is lifted
    and the real implementation takes over**; the fp32 head-chunked path is used
    only when it is not. **The decision is made by measurement, never assumed.**

    Args:
        head_chunk: Heads computed at once on the fp32 path. Unused on the fast
            path.
        fp32: None **decides by measurement**. True or False forces it, which is
            useful for reproducing the breakdown.

    Returns:
        Whether the fast path is in effect. **Recorded in `metrics`.**
    """
    # Keep upstream's flash-and-mem_efficient-only request from failing. When
    # the fast path is available, lifting the block is all that is needed for the
    # real flash implementation to be chosen.
    torch.backends.cuda.sdp_kernel = lambda *a, **kw: contextlib.nullcontext()

    use_fp32 = (not fast_attention_available()) if fp32 is None else fp32
    if not use_fp32:
        F.scaled_dot_product_attention = _TORCH_SDPA
        print("[shape] fast attention=yes (delegating to AOTriton flash)", file=sys.stderr)
        return True

    def sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if attn_mask is not None or is_causal or dropout_p:
            # Never used on this model's inference path; raise so it is noticed.
            raise NotImplementedError("patched SDPA supports plain attention only")
        out_dtype = query.dtype
        d = query.shape[-1]
        sc = scale if scale is not None else (1.0 / math.sqrt(d))
        heads = query.shape[1]
        outs = []
        for i in range(0, heads, head_chunk):
            q = query[:, i : i + head_chunk].float()
            k = key[:, i : i + head_chunk].float()
            v = value[:, i : i + head_chunk].float()
            attn = torch.matmul(q, k.transpose(-1, -2)) * sc
            attn = torch.softmax(attn, dim=-1)
            outs.append(torch.matmul(attn, v).to(out_dtype))
            del q, k, v, attn
        return torch.cat(outs, dim=1)

    F.scaled_dot_product_attention = sdpa
    print("[shape] fast attention=no (falling back to fp32 over chunked heads)", file=sys.stderr)
    return False


@dataclass(frozen=True)
class ShapeResult:
    """The generated shape and its measurements.

    mesh: The generated mesh, at normalized scale. Scaling to real-world size is
        downstream work.
    backend: Identifier of the backend used.
    load_sec: Seconds spent loading the pipeline. Later calls return the same
        value, since the pipeline is cached.
    gen_sec: Seconds spent generating. **An unstable figure on this hardware**,
        so never use it as a pass/fail signal.
    vram_peak_gb: Peak device memory in use, in GB.
    fast_attention: Whether fast attention (AOTriton flash) is in effect.
    steps: Inference steps.
    octree_resolution: Octree resolution for marching cubes.
    guidance_scale: Guidance scale.
    seed: Random seed.
    """

    mesh: trimesh.Trimesh
    backend: str
    load_sec: float
    gen_sec: float
    vram_peak_gb: float
    fast_attention: bool
    steps: int
    octree_resolution: int
    guidance_scale: float
    seed: int


def load_pipeline() -> Any:
    """Load the Hunyuan3D 2.1 shape pipeline once per process and return it.

    The order of the steps matters. In particular, `_install_sdpa_shim` must be
    called **before hy3dshape is imported**: once the import binds
    `scaled_dot_product_attention`, the replacement no longer takes effect.

    Returns:
        A Hunyuan3DDiTFlowMatchingPipeline instance.

    Raises:
        FileNotFoundError: If the repository or the weights directory is missing.
    """
    global _PIPELINE, _LOAD_SEC
    if _PIPELINE is not None:
        return _PIPELINE
    if not config.SHAPE_REPO.is_dir():
        raise FileNotFoundError(
            f"shape repository not found: {config.SHAPE_REPO} (check HUNYUAN3D_SHAPE_REPO)"
        )
    if not config.SHAPE_MODELS_DIR.is_dir():
        raise FileNotFoundError(
            f"weights directory not found: {config.SHAPE_MODELS_DIR} "
            "(check HUNYUAN3D_MODELS_DIR)"
        )

    # Hunyuan finds its weights through this environment variable
    # (utils.py: os.environ.get('HY3DGEN_MODELS', ...)). `.env` is authoritative
    # here, so any existing value is overwritten.
    os.environ["HY3DGEN_MODELS"] = str(config.SHAPE_MODELS_DIR)
    repo = str(config.SHAPE_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    global _FAST_ATTENTION
    _FAST_ATTENTION = _install_sdpa_shim(head_chunk=config.ATTN_HEAD_CHUNK)
    apply_vram_limit()

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    t0 = time.perf_counter()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        config.SHAPE_MODEL_ID, device="cuda", dtype=torch.float16
    )
    # flashvdm is required: the default VanillaVolumeDecoder queries every point
    # of a 385^3 grid and does not finish in 30 minutes. FlashVDM is pure PyTorch
    # and uses no custom CUDA kernels. mc_algo="mc" is skimage on the CPU; "dmc"
    # needs diso (CUDA) and is unavailable here.
    pipe.enable_flashvdm(enabled=True, topk_mode="mean", mc_algo="mc")
    _LOAD_SEC = time.perf_counter() - t0
    _PIPELINE = pipe
    return pipe


def generate_mesh(
    image: Image.Image,
    *,
    steps: int | None = None,
    octree_resolution: int | None = None,
    guidance_scale: float | None = None,
    seed: int = 0,
    progress: Callable[[str, str], None] | None = None,
) -> ShapeResult:
    """Generate a shape mesh from one foreground image.

    - **Pass RGBA with an alpha channel.** With plain RGB, Hunyuan's
      `ImageProcessorV2.recenter()` sets the whole mask to 255 and the cut-out
      never happens.
    - Generation time is unstable on this hardware (235 to 921 seconds for
      identical settings). Never use it as a pass/fail signal.

    Args:
        image: The input image with its background removed (RGBA).
        steps: Inference steps, or None for the `.env` default.
        octree_resolution: Octree resolution for marching cubes, or None for the
            `.env` default.
        guidance_scale: Guidance scale, or None for the `.env` default.
        seed: Random seed.

    Returns:
        A ShapeResult holding the mesh and the measurements.

    Raises:
        TypeError: If the pipeline returns something other than a Trimesh.
    """
    steps = config.SHAPE_STEPS if steps is None else steps
    octree_resolution = config.MC_RESOLUTION if octree_resolution is None else octree_resolution
    guidance_scale = config.GUIDANCE_SCALE if guidance_scale is None else guidance_scale

    pipe = load_pipeline()
    load_sec = _LOAD_SEC

    # **Report the denoising steps.** The scheduler is the authority on how many
    # there are, so the count follows whatever upstream actually asked for.
    _STEPS.bind(progress, "shape", "denoising")
    count_scheduler(pipe.scheduler, _STEPS)

    torch.cuda.reset_peak_memory_stats()
    sampler = _DeviceWatch(
        progress=progress,
        stage="generation",
        heartbeat_sec=config.HEARTBEAT_SEC,
        limit_gb=config.VRAM_LIMIT_GB,
    )
    t0 = time.perf_counter()
    try:
        with sampler:
            meshes = pipe(
                image=image,
                num_inference_steps=steps,
                octree_resolution=octree_resolution,
                guidance_scale=guidance_scale,
                mc_algo="mc",
                generator=torch.Generator(device="cuda").manual_seed(seed),
                output_type="trimesh",
            )
    finally:
        # **Let go of this request's sink.** The hook stays on the scheduler for
        # the next generation, but a step must never be reported against a
        # request that has already been answered.
        _STEPS.bind(None, "shape")
    gen_sec = time.perf_counter() - t0

    mesh = meshes[0]
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"the shape pipeline returned a non-Trimesh: {type(mesh)}")

    return ShapeResult(
        mesh=mesh,
        backend=BACKEND,
        load_sec=float(load_sec),
        gen_sec=float(gen_sec),
        vram_peak_gb=float(sampler.peak_used_gb),
        fast_attention=_FAST_ATTENTION,
        steps=int(steps),
        octree_resolution=int(octree_resolution),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
    )


def unload_pipeline() -> bool:
    """Release the loaded pipeline and give the VRAM back.

    **This is what makes exclusive GPU use possible.** The runner stays resident
    to keep the model warm (a cold load measured 77.6 seconds), but releases
    explicitly when switching models or handing the GPU to another process.

    `torch`'s caching allocator does not return memory to the OS on `del` alone,
    so **`empty_cache()` is called too**. Any surviving reference would prevent
    the release, which is why the module-level variables are cleared here.

    Returns:
        True if something was released, False if nothing was loaded.
    """
    global _PIPELINE, _LOAD_SEC
    if _PIPELINE is None:
        return False
    _PIPELINE = None
    _LOAD_SEC = 0.0
    gc.collect()
    torch.cuda.empty_cache()
    return True


def is_loaded() -> bool:
    """Return whether the pipeline is loaded."""
    return _PIPELINE is not None


def load_seconds() -> float:
    """Return the seconds the last cold load took, or 0.0 when nothing is loaded."""
    return _LOAD_SEC


def apply_vram_limit() -> float:
    """Make exceeding dedicated VRAM **fail immediately instead of silently slowing down**.

    The total from `torch.cuda.mem_get_info` includes shared memory (43.87 GB on
    gfx1151), so passing the 32 GB of dedicated VRAM raises nothing. The excess
    lands in host memory and **becomes several times slower with no exception and
    no warning** (measured 2026-09-01 in sparse convolution, which eventually
    reached 42.02 GB and a `torch.OutOfMemoryError`).

    Passing an allocation cap to torch as a fraction of the total makes any
    allocation beyond it a `torch.OutOfMemoryError`, so **it surfaces at once**.

    Returns:
        The cap actually applied, in GB, or 0.0 if none could be applied.
    """
    limit = float(config.VRAM_LIMIT_GB)
    if limit <= 0 or not torch.cuda.is_available():
        return 0.0
    _, total = torch.cuda.mem_get_info()
    total_gb = total / 1024**3
    fraction = min(max(limit / total_gb, 0.05), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction)
    return limit


def device_memory_gb() -> tuple[float, float]:
    """Return (used GB, total GB).

    **The total includes shared memory**: gfx1151 reports 43.87 GB while the
    dedicated VRAM is 32 GB. Say so wherever the figure is displayed.
    """
    free, total = torch.cuda.mem_get_info()
    return ((total - free) / 1024**3, total / 1024**3)
