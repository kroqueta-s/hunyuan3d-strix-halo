# SPDX-License-Identifier: MIT
"""Background removal (rembg, isnet-general-use, on the CPU, no VRAM needed).

**This is the runner's responsibility** and cannot be skipped for Hunyuan3D.

**It is not optional preprocessing.** With a 3-channel image, Hunyuan's
`ImageProcessorV2.recenter()` sets the entire mask to 255, so cropping and
recentring the subject never happen. **The model has no built-in foreground
extraction**, which makes `rembg=False` a bypass meaning "the image already has
its background removed", not "let the model handle it".

rembg sessions are expensive to create, so they are cached per model name.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from . import config

_SESSIONS: dict[str, object] = {}


def _install_pymatting_stub(cause: BaseException) -> None:
    """Replace **only the unused parts** of `pymatting` so that `rembg` can be imported.

    `rembg/bg.py` **imports `pymatting` unconditionally at module level**, but
    only uses it on the `alpha_matting=True` and `decontaminate=True` paths, both
    of which are off by default. The defaults are what run here, so pymatting is
    not needed.

    The import still has to succeed, or `rembg` is unusable. Measured on
    2026-09-01: **Smart App Control blocked `llvmlite.dll`** (WinError 4551), and
    the `pymatting -> numba -> llvmlite` chain broke the `rembg` import.

    **Upstream code is never modified** (the same approach as the SDPA
    replacement in `shape.py`). Nothing happens when the real package imports, so
    this reverts by itself once Smart App Control changes its mind. The
    substituted functions **always raise when called**, never returning a
    different result silently.

    Args:
        cause: Why the real import failed, printed to stderr for diagnosis.
    """
    import types

    def _unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "pymatting is unavailable in this environment, so alpha matting cannot run "
            "(background removal must stay on the alpha_matting=False path)"
        )

    print(f"[background] substituted a stand-in for pymatting: {cause}", file=sys.stderr)
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
    """Return `rembg`'s `remove` and `new_session`, installing the stand-in first if needed.

    Returns:
        The pair `(remove, new_session)`.
    """
    try:
        import pymatting  # noqa: F401
    except (ImportError, OSError) as exc:
        _install_pymatting_stub(exc)
    from rembg import new_session, remove

    return remove, new_session


def _session(model: str) -> object:
    """Return a rembg session, cached per model name.

    Args:
        model: The rembg model name.

    Returns:
        The rembg session object.
    """
    if model not in _SESSIONS:
        _, new_session = _rembg()
        _SESSIONS[model] = new_session(model)
    return _SESSIONS[model]


def remove_background(image: Image.Image, *, model: str | None = None) -> Image.Image:
    """Cut out the foreground and return RGBA with an alpha channel.

    Args:
        image: The input image; RGB or RGBA both work.
        model: The rembg model name, or None for the `.env` default
            (isnet-general-use).

    Returns:
        An image that is **always RGBA**, with alpha as the foreground mask.
    """
    name = config.REMBG_MODEL if model is None else model
    remove, _ = _rembg()
    return remove(image.convert("RGB"), session=_session(name))


def foreground_fraction(image: Image.Image) -> float:
    """Return the fraction of pixels with alpha above 127, as a check that removal worked.

    Args:
        image: The image to inspect. Anything that is not RGBA returns 1.0.

    Returns:
        The foreground fraction, from 0.0 to 1.0. **A value near 0 means the
        cut-out failed**, so suspect the input image.
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
    """Load an image file and, if asked, remove its background.

    Returns the RGBA image together with its foreground fraction.

    `rembg=False` is a bypass for images whose background is already removed. It
    does not hand foreground extraction to the model, which has none.

    Args:
        path: Path to the input image.
        rembg: Whether to remove the background, or None for the `.env` default.
        model: The rembg model name, or None for the `.env` default.

    Returns:
        The pair (RGBA image, foreground fraction).
    """
    use = config.REMBG if rembg is None else rembg
    loaded: Image.Image = Image.open(path)
    image = remove_background(loaded, model=model) if use else loaded.convert("RGBA")
    return image, foreground_fraction(image)
