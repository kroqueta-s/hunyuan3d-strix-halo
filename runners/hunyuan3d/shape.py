# SPDX-License-Identifier: MIT
"""Hunyuan3D 2.1 の shape 段（このランナーの中核）。

1 枚の前景画像 → 3D ネイティブ潜在拡散 → SDF → marching cubes → watertight メッシュ。
テクスチャ段は使わない（用途が 3D プリント主体で、かつ CUDA 依存を避けられる）。

**このモジュールはこのランナー専用の venv でしか動かない**
（torch+ROCm と Hunyuan3D のリポジトリが要る）。ComfyUI の venv からは import できない。

gfx1151/Windows/ROCm での必須の細工は `_install_sdpa_shim` と `enable_flashvdm` の 2 つ。
どちらも外すと「動かない」ではなく「静かに壊れる／終わらない」ので、
根拠は各所の docstring に残してある。**実測は SwitchDeck の docs/25 §1.3。**

SwitchDeck -> meshforge -> hearth のランナーへと移設した（2026-09-01）。
変更は設定キー名だけで、**shim には触っていない**。
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

BACKEND = "hunyuan3d21"

_PIPELINE: Any = None
_LOAD_SEC: float = 0.0


class _DeviceWatch:
    """デバイスの実使用量を追い、**一定間隔で生存を知らせる**監視スレッド。

    以前は使用量のピークを取るだけだった。それだけだと、長い段の途中で
    「進んでいるのか、止まっているのか」が外から一切分からない。実際に
    2026-09-01、生成が黙って 12 分以上走るのを何度も待ってしまった。

    見ているのは 3 つ：

    1. **生存**（`heartbeat`）。既定 10 秒ごとに経過秒と VRAM を流す。
       呼び出し側はこれが止まったことで「進んでいない」を判定できる
    2. **専用 VRAM の超過**（`vram_over`）。**専用 VRAM は 32GB しかない。**
       `torch.cuda.mem_get_info` の total（43.87GB）は共有メモリ込みの嘘なので、
       溢れても例外にならず、**黙って数倍遅くなる**。ここを跨いだ瞬間に知らせる
    3. ピーク（従来どおり `metrics` へ載せる）

    **このスレッドから `progress` を呼ぶので、呼び出し側の emit は鍵で守ること。**
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
                    f"**専用 VRAM を超えた**（{used:.2f}GB > {self.limit_gb:.2f}GB）。"
                    "共有メモリへ溢れているので、このまま待っても遅いだけ",
                )
            if now - last_beat >= self.heartbeat_sec:
                last_beat = now
                self._say(
                    "heartbeat",
                    f"{self.stage or '実行中'} 経過 {now - started:.0f}s / "
                    f"VRAM {used:.2f}GB（ピーク {self.peak_used_gb:.2f}GB）",
                )
            self._stop.wait(self.interval)

    def __enter__(self) -> _DeviceWatch:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join(timeout=5)


def _install_sdpa_shim(head_chunk: int, fp32: bool = True) -> None:
    """gfx1151/Windows/ROCm 7.2.1 向けに SDPA を差し替える。

    2 つの問題を同時に解く:

    1. **バックエンド不在** — hunyuandit.py は 3 箇所で
       ``sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)``
       と呼ぶが、本機で実測すると flash も mem_efficient も「No available kernel」。
       使えるのは math だけなので、この指定は**唯一動くバックエンドを塞いでいる**。

    2. **fp16 math の数値破綻** — math backend は flash のような online softmax を持たず
       ``q @ k^T``（seq=4096）を入力 dtype のまま実体化する。fp16 の上限は 65504 で、
       入力によっては溢れて SDF 場がノイズ化する（鏡像入力で決定論的に再現した）。
       さらに全ヘッド同時だと中間テンソルが 4GB 級になり専用 VRAM を圧迫する。

    対策として ``F.scaled_dot_product_attention`` を fp32 計算＋ヘッド分割の実装に
    差し替える。**ベンダーコードは書き換えない**（再 clone・更新で壊れるため）。

    Args:
        head_chunk: 一度に計算するヘッド数。VRAM ピークと速度を決める。
            gfx1151 実測では 4 が最良（302 秒 / 11.6GB）。8 は 235 秒だが 17.7GB、
            16 は 511 秒 / 30.0GB で速度も VRAM も悪化する。
        fp32: False にすると fp16 のまま計算する（破綻の再現用。実運用では True）。
    """
    torch.backends.cuda.sdp_kernel = lambda *a, **kw: contextlib.nullcontext()

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
            # 本モデルの推論経路では使われない。来たら気付けるように落とす。
            raise NotImplementedError("patched SDPA supports plain attention only")
        out_dtype = query.dtype
        compute_dtype = torch.float32 if fp32 else out_dtype
        d = query.shape[-1]
        sc = scale if scale is not None else (1.0 / math.sqrt(d))
        heads = query.shape[1]
        outs = []
        for i in range(0, heads, head_chunk):
            q = query[:, i : i + head_chunk].to(compute_dtype)
            k = key[:, i : i + head_chunk].to(compute_dtype)
            v = value[:, i : i + head_chunk].to(compute_dtype)
            attn = torch.matmul(q, k.transpose(-1, -2)) * sc
            attn = torch.softmax(attn, dim=-1)
            outs.append(torch.matmul(attn, v).to(out_dtype))
            del q, k, v, attn
        return torch.cat(outs, dim=1)

    F.scaled_dot_product_attention = sdpa


@dataclass(frozen=True)
class ShapeResult:
    """shape 生成の結果と実測値。

    mesh: 生成されたメッシュ（正規化スケール。実寸化は printability.scale_to_mm）。
    backend: 使用したバックエンド識別子。
    load_sec: パイプライン読み込み秒。2 回目以降はキャッシュ済みなので同じ値が返る。
    gen_sec: 生成秒。**このハードでは不安定な指標**なので合否判定に使わない。
    vram_peak_gb: デバイス実使用量のピーク（GB）。
    steps: 推論ステップ数。
    octree_resolution: marching cubes の octree 解像度。
    guidance_scale: guidance scale。
    seed: 乱数シード。
    """

    mesh: trimesh.Trimesh
    backend: str
    load_sec: float
    gen_sec: float
    vram_peak_gb: float
    steps: int
    octree_resolution: int
    guidance_scale: float
    seed: int


def load_pipeline() -> Any:
    """Hunyuan3D 2.1 の shape パイプラインをプロセス内で 1 度だけ読み込んで返す。

    手順の順序に意味がある。特に `_install_sdpa_shim` は **hy3dshape を import する前**に
    呼ぶこと（import 時に `scaled_dot_product_attention` を束縛されると差し替えが効かない）。

    Returns:
        Hunyuan3DDiTFlowMatchingPipeline のインスタンス。

    Raises:
        FileNotFoundError: リポジトリまたは重みのディレクトリが見つからないとき。
    """
    global _PIPELINE, _LOAD_SEC
    if _PIPELINE is not None:
        return _PIPELINE
    if not config.SHAPE_REPO.is_dir():
        raise FileNotFoundError(
            f"shape リポジトリが無い: {config.SHAPE_REPO}（HUNYUAN3D_SHAPE_REPO を確認する）"
        )
    if not config.SHAPE_MODELS_DIR.is_dir():
        raise FileNotFoundError(
            f"重みのディレクトリが無い: {config.SHAPE_MODELS_DIR}"
            "（HUNYUAN3D_MODELS_DIR を確認する）"
        )

    # Hunyuan 側はこの環境変数で重みを探す（utils.py: os.environ.get('HY3DGEN_MODELS', ...)）。
    # .env が正典なので既存値があっても上書きする。
    os.environ["HY3DGEN_MODELS"] = str(config.SHAPE_MODELS_DIR)
    repo = str(config.SHAPE_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    _install_sdpa_shim(head_chunk=config.ATTN_HEAD_CHUNK)
    apply_vram_limit()

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    t0 = time.perf_counter()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        config.SHAPE_MODEL_ID, device="cuda", dtype=torch.float16
    )
    # flashvdm は必須。既定の VanillaVolumeDecoder は 385^3 点を全数クエリして
    # 30 分超でも終わらない。FlashVDM は純 PyTorch で独自 CUDA カーネルを使わない。
    # mc_algo="mc" は skimage（CPU）。"dmc" は diso（CUDA）が要るので使えない。
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
    """前景画像 1 枚から shape メッシュを生成する。

    - **入力はアルファ付き RGBA を渡すこと。** RGB のままだと Hunyuan の
      `ImageProcessorV2.recenter()` がマスクを全面 255 にして切り出しが効かない。
    - 生成時間はこのハードでは不安定（同一設定で 235〜921 秒）。合否判定に使わない。

    Args:
        image: 背景除去済みの入力画像（RGBA）。
        steps: 推論ステップ数。None なら .env の既定。
        octree_resolution: marching cubes の octree 解像度。None なら .env の既定。
        guidance_scale: guidance scale。None なら .env の既定。
        seed: 乱数シード。

    Returns:
        生成メッシュと実測値をまとめた ShapeResult。

    Raises:
        TypeError: パイプラインが Trimesh 以外を返したとき。
    """
    steps = config.SHAPE_STEPS if steps is None else steps
    octree_resolution = config.MC_RESOLUTION if octree_resolution is None else octree_resolution
    guidance_scale = config.GUIDANCE_SCALE if guidance_scale is None else guidance_scale

    pipe = load_pipeline()
    load_sec = _LOAD_SEC

    torch.cuda.reset_peak_memory_stats()
    sampler = _DeviceWatch(
        progress=progress,
        stage="生成",
        heartbeat_sec=config.HEARTBEAT_SEC,
        limit_gb=config.VRAM_LIMIT_GB,
    )
    t0 = time.perf_counter()
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
    gen_sec = time.perf_counter() - t0

    mesh = meshes[0]
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"shape パイプラインが Trimesh 以外を返した: {type(mesh)}")

    return ShapeResult(
        mesh=mesh,
        backend=BACKEND,
        load_sec=float(load_sec),
        gen_sec=float(gen_sec),
        vram_peak_gb=float(sampler.peak_used_gb),
        steps=int(steps),
        octree_resolution=int(octree_resolution),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
    )


def unload_pipeline() -> bool:
    """読み込み済みのパイプラインを解放して VRAM を返す。

    **VRAM 排他の要。** ランナーは常駐してモデルを温存するが（コールドロードは実測 77.6 秒）、
    別のモデルへ切り替えるときや `mode_dev` に GPU を譲るときは明示的に解放する。

    `torch` の caching allocator は `del` しただけでは OS へ返さないので、
    **`empty_cache()` まで呼ぶ**。参照が他から残っていると解放されないため、
    ここでモジュール変数を確実に落とす。

    Returns:
        解放したら True。もともと読み込んでいなければ False。
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
    """パイプラインを読み込み済みかを返す。"""
    return _PIPELINE is not None


def load_seconds() -> float:
    """直近のコールドロードに要した秒数を返す（未読み込みなら 0.0）。"""
    return _LOAD_SEC


def apply_vram_limit() -> float:
    """**専用 VRAM を超えたら黙って遅くなるのではなく、その場で落ちる**ようにする。

    `torch.cuda.mem_get_info` の総容量は共有メモリ込み（gfx1151 で 43.87GB）で、
    専用 VRAM の 32GB を超えても例外にならない。超えた分はホスト側のメモリへ落ちるので、
    **例外も警告も出ないまま数倍遅くなる**（2026-09-01、疎畳み込みで実測。最終的には
    42.02GB まで確保して `torch.OutOfMemoryError` に至った）。

    そこで割り当ての上限を総容量に対する割合で torch へ伝える。上限を超える確保は
    `torch.OutOfMemoryError` になるので、**待たされずに気付ける。**

    Returns:
        実際に設定した上限（GB）。設定できなければ 0.0。
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
    """(使用中 GB, 総容量 GB) を返す。

    **総容量は共有メモリ込みの値**（gfx1151 では 43.87GB と出るが専用VRAM は 32GB）。
    表示に使うときはその旨を添えること。
    """
    free, total = torch.cuda.mem_get_info()
    return ((total - free) / 1024**3, total / 1024**3)
