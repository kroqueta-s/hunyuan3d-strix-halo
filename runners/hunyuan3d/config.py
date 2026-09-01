# SPDX-License-Identifier: MIT
"""Hunyuan3D ランナーの設定（`.env` から読み込む）。

**このランナーは自分の中で閉じている。** hearth の設定を参照しないので、
`hunyuan3d-strix-halo` として独立リポジトリへ出しても、そのまま動く。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# runners/hunyuan3d/config.py -> リポジトリのルート。
# 独立リポジトリへ出したときも、同じ位置関係になるよう配置する。
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

# SDPA を何ヘッドずつ計算するか。gfx1151 実測で 4 が最良（302 秒 / 11.6GB）。
# 8 は速いが 17.7GB、16 は速度も VRAM も悪化する（511 秒 / 30.0GB）。**根拠なく変えない。**
ATTN_HEAD_CHUNK: int = _int("HUNYUAN3D_ATTN_HEAD_CHUNK", 4)

# **off にしない。** ImageProcessorV2.recenter() は 3ch 画像だとマスクを全面 255 にする。
REMBG: bool = _bool("HUNYUAN3D_REMBG", True)
REMBG_MODEL: str = _str("HUNYUAN3D_REMBG_MODEL", "isnet-general-use")


# **専用 VRAM の上限（GB）。** gfx1151 の専用 VRAM は 32GB だが、
# `torch.cuda.mem_get_info` の total は共有メモリ込みの 43.87GB を返す。
# そのため溢れても例外にならず、**黙って数倍遅くなる**（2026-09-01 に実測で踏んだ）。
# ここを torch にも伝えて、超えたら OOM で**すぐ落ちる**ようにする。
VRAM_LIMIT_GB: float = _float("HUNYUAN3D_VRAM_LIMIT_GB", 30.0)

# 生存確認を流す間隔（秒）。**黙って長時間走らせない**ためのもの。
HEARTBEAT_SEC: float = _float("HUNYUAN3D_HEARTBEAT_SEC", 10.0)

# **torch を import する前に** TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL を立てるか。
# 立てると flash / mem-efficient が使えるようになり、fp32＋ヘッド分割の経路を通らずに済む。
# **後から os.environ へ入れても効かない**ので、`__main__.py` の先頭で置く。
FAST_ATTENTION: bool = _bool("HUNYUAN3D_FAST_ATTENTION", True)

# 生成中だけ「3D の常夜灯」を点けるか（`gfxlight.py`）。Windows の AMD ドライバは
# compute だけの負荷ではクロックを上げない（実測：GEMM 単独 600 MHz / 3D 併用 2.35 GHz・
# 4.3 倍）。点かなくても生成は従来どおり動く。効いたかは metrics.gfx_keepalive に載る。
GFX_KEEPALIVE: bool = _bool("HUNYUAN3D_GFX_KEEPALIVE", True)
