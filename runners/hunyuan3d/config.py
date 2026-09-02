# SPDX-License-Identifier: MIT
"""Configuration for the Hunyuan3D runner, read from `.env`.

**This runner is self-contained.** It never reads hearth's configuration, so it
works unchanged as the standalone `hunyuan3d-strix-halo` repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# runners/hunyuan3d/config.py -> the repository root.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


def _str(key: str, default: str = "") -> str:
    raw = os.getenv(key)
    return raw.strip() if raw is not None and raw.strip() != "" else default


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None and raw.strip() != "" else default


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw is not None and raw.strip() != "" else default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


SHAPE_REPO: Path = Path(_str("HUNYUAN3D_SHAPE_REPO"))
SHAPE_MODELS_DIR: Path = Path(_str("HUNYUAN3D_MODELS_DIR"))
SHAPE_MODEL_ID: str = _str("HUNYUAN3D_MODEL_ID", "tencent/Hunyuan3D-2.1")

MC_RESOLUTION: int = _int("HUNYUAN3D_MC_RESOLUTION", 384)
SHAPE_STEPS: int = _int("HUNYUAN3D_SHAPE_STEPS", 30)
GUIDANCE_SCALE: float = _float("HUNYUAN3D_GUIDANCE_SCALE", 5.0)

# Attention heads computed at once in the SDPA replacement. Measured best on
# gfx1151 at 4 (302 s / 11.6 GB); 8 is faster but takes 17.7 GB, and 16 is worse
# on both counts (511 s / 30.0 GB). **Do not change it without evidence.**
ATTN_HEAD_CHUNK: int = _int("HUNYUAN3D_ATTN_HEAD_CHUNK", 4)

# **Do not turn this off.** With a 3-channel image, `ImageProcessorV2.recenter()`
# sets the entire mask to 255.
REMBG: bool = _bool("HUNYUAN3D_REMBG", True)
REMBG_MODEL: str = _str("HUNYUAN3D_REMBG_MODEL", "isnet-general-use")


# **Cap on dedicated VRAM (GB).** gfx1151 has 32 GB of dedicated VRAM, but the
# total from `torch.cuda.mem_get_info` is 43.87 GB because it counts shared
# memory. Overflow therefore raises nothing and **silently becomes several times
# slower** (measured on 2026-09-01). Passing the cap to torch as well turns that
# into an **immediate OOM**.
VRAM_LIMIT_GB: float = _float("HUNYUAN3D_VRAM_LIMIT_GB", 30.0)

# Heartbeat interval in seconds. It exists so that **nothing runs silently for a
# long time**.
HEARTBEAT_SEC: float = _float("HUNYUAN3D_HEARTBEAT_SEC", 10.0)

# Whether to set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL **before torch is
# imported**. It makes the flash and memory-efficient kernels available, so the
# fp32 head-chunked path is never needed. **Setting it afterwards has no
# effect**, so it goes at the top of `__main__.py`.
FAST_ATTENTION: bool = _bool("HUNYUAN3D_FAST_ATTENTION", True)

# Whether to run the clock keepalive during generation (`gfxlight.py`). The AMD
# Windows driver does not raise the clock for compute-only work (measured: GEMM
# alone 600 MHz, with 3D alongside 2.35 GHz, a 4.3x difference). Generation
# works as before if it fails to start; `metrics.gfx_keepalive` records whether
# it was alive.
GFX_KEEPALIVE: bool = _bool("HUNYUAN3D_GFX_KEEPALIVE", True)


# --- Texture stage (hy3dpaint) -----------------------------------------------
# It needs its own weights (hunyuan3d-paintpbr-v2-1, about 6.4 GB) on top of the
# shape stage. **The CUDA-only `custom_rasterizer` is replaced by the pure-torch
# `raster.py`**, so nothing has to be built.
TEXTURE_MAX_VIEWS: int = _int("HUNYUAN3D_TEXTURE_MAX_VIEWS", 6)
TEXTURE_VIEW_RESOLUTION: int = _int("HUNYUAN3D_TEXTURE_VIEW_RESOLUTION", 512)
# Upstream defaults: 2048 for rendering and 4096 for the texture. Both cost
# VRAM and time, so they are settings rather than constants.
TEXTURE_RENDER_SIZE: int = _int("HUNYUAN3D_TEXTURE_RENDER_SIZE", 2048)
TEXTURE_SIZE: int = _int("HUNYUAN3D_TEXTURE_SIZE", 4096)

# DINOv2 features condition the multi-view diffusion. **It is not optional**:
# upstream hardcodes `use_dino = True`. Its weights (about 4.5 GB) live beside
# the Hunyuan ones, so nothing is fetched at generation time.
TEXTURE_DINO_DIR: Path = Path(
    _str("HUNYUAN3D_TEXTURE_DINO_DIR", str(SHAPE_MODELS_DIR / "facebook" / "dinov2-giant"))
)
# Sub-directory of the weights repository holding the texture stage.
TEXTURE_SUBFOLDER: str = _str("HUNYUAN3D_TEXTURE_SUBFOLDER", "hunyuan3d-paintpbr-v2-1")
