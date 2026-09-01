# SPDX-License-Identifier: MIT
"""背景除去（rembg・isnet-general-use・CPU・VRAM 不要）。

**このランナーの責任である**（契約 §4）。Hunyuan3D では省略できない。

**これは省略できない前処理。** Hunyuan の `ImageProcessorV2.recenter()` は 3ch 画像だと
マスクを全面 255 にするため、被写体の切り出しと再センタリングが一切効かない。
**モデル内蔵の前景抽出は存在しない**ので、`rembg=False` は「モデルに委ねる」ではなく
「既に背景除去済みの画像を渡す」という意味のバイパスである。

rembg のセッションは生成コストが高いので、モデル名ごとにキャッシュする。
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from . import config

_SESSIONS: dict[str, object] = {}

def _install_pymatting_stub(cause: BaseException) -> None:
    """`pymatting` の**使わない部分だけ**を差し替えて `rembg` を import 可能にする。

    `rembg/bg.py` は `pymatting` を**モジュール先頭で無条件に import する**が、
    実際に使うのは `alpha_matting=True` と `decontaminate=True` の経路だけで、
    どちらも既定では off である。ここは既定のまま呼ぶので pymatting は不要。

    それでも import が通らないと `rembg` ごと使えない。実測（2026-09-01）では
    **Smart App Control が `llvmlite.dll` をブロックし**（WinError 4551）、
    `pymatting -> numba -> llvmlite` の連鎖で `rembg` の import が失敗した。

    **ベンダーコードは書き換えない**（`shape.py` の SDPA 差し替えと同じ流儀）。
    本物が import できるときは何もしないので、SAC の判定が変われば自動的に元へ戻る。
    差し込んだ関数は**呼ばれたら必ず落ちる**（黙って違う結果を返さない）。

    Args:
        cause: 本物の import が失敗した理由。診断のため stderr へ出す。
    """
    import types

    def _unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "pymatting が使えない環境なので alpha matting は実行できない"
            "（背景除去は alpha_matting=False の経路だけを使うこと）"
        )

    print(f"[background] pymatting を代替に差し替えた: {cause}", file=sys.stderr)
    for name in ("pymatting", "pymatting.alpha", "pymatting.foreground", "pymatting.util"):
        sys.modules.setdefault(name, types.ModuleType(name))
    for name, attr in (
        ("pymatting.alpha.estimate_alpha_cf", "estimate_alpha_cf"),
        ("pymatting.foreground.estimate_foreground_ml", "estimate_foreground_ml"),
        ("pymatting.util.util", "stack_images"),
    ):
        module = types.ModuleType(name)
        setattr(module, attr, _unavailable)
        sys.modules[name] = module


def _rembg() -> tuple[Callable[..., object], Callable[..., object]]:
    """`rembg` の `remove` と `new_session` を返す（必要なら代替を差し込んでから）。

    Returns:
        `(remove, new_session)` の組。
    """
    try:
        import pymatting  # noqa: F401
    except (ImportError, OSError) as exc:
        _install_pymatting_stub(exc)
    from rembg import new_session, remove

    return remove, new_session



def _session(model: str) -> object:
    """rembg のセッションをモデル名ごとにキャッシュして返す。

    Args:
        model: rembg のモデル名。

    Returns:
        rembg のセッションオブジェクト。
    """
    if model not in _SESSIONS:
        _, new_session = _rembg()
        _SESSIONS[model] = new_session(model)
    return _SESSIONS[model]


def remove_background(image: Image.Image, *, model: str | None = None) -> Image.Image:
    """前景を切り出してアルファ付き RGBA を返す。

    Args:
        image: 入力画像（RGB でも RGBA でもよい）。
        model: rembg のモデル名。None なら .env の既定（isnet-general-use）。

    Returns:
        **必ず RGBA** の画像。アルファが前景マスクになる。
    """
    name = config.REMBG_MODEL if model is None else model
    remove, _ = _rembg()
    return remove(image.convert("RGB"), session=_session(name))


def foreground_fraction(image: Image.Image) -> float:
    """アルファが 127 を超える画素の割合を返す（背景除去が効いたかの目安）。

    Args:
        image: 判定する画像。RGBA でなければ 1.0 を返す。

    Returns:
        前景率（0.0〜1.0）。**0 に近いと切り出しに失敗している**（入力画像を疑うこと）。
    """
    if image.mode != "RGBA":
        return 1.0
    alpha = image.split()[-1]
    binary = alpha.point(lambda a: 255 if a > 127 else 0).convert("L")
    total = 255.0 * image.width * image.height
    return float(sum(binary.getdata())) / total if total > 0 else 0.0


def prepare_image(
    path: Path, *, rembg: bool | None = None, model: str | None = None
) -> tuple[Image.Image, float]:
    """画像ファイルを読み、必要なら背景除去して (RGBA 画像, 前景率) を返す。

    `rembg=False` は「既に背景除去済みの画像を渡す」場合のバイパスであって、
    モデルに前景抽出を委ねる意味ではない（内蔵の前景抽出は存在しない）。

    Args:
        path: 入力画像のパス。
        rembg: 背景除去を行うか。None なら .env の既定。
        model: rembg のモデル名。None なら .env の既定。

    Returns:
        (RGBA 画像, 前景率) の組。
    """
    use = config.REMBG if rembg is None else rembg
    loaded: Image.Image = Image.open(path)
    image = remove_background(loaded, model=model) if use else loaded.convert("RGBA")
    return image, foreground_fraction(image)
