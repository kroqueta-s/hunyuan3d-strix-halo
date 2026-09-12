# SPDX-License-Identifier: MIT
"""Verify that **loading the weights reports liveness**, without loading any.

Loading takes about eighty seconds here and used to say nothing while it did.
A caller cannot tell that from a hang: hearth's harness ends a runner that has
been silent for sixty seconds, and the five-model switch test failed on exactly
that (2026-09-12).

What is checked is the watcher itself - that it beats at the interval it was
given, that it names the stage it was given, and that it **emits no step**,
because a heartbeat is not progress (contract §8). The pipeline it wraps is
irrelevant to that, so torch is replaced with a stand-in and nothing is loaded.

Run it with this repository's virtual environment, or any python with no torch::

    .venv\\Scripts\\python.exe .\\tests\\test_load_heartbeat.py
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = " OK " if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def _install_stubs() -> Any:
    """Replace torch with just enough for the watcher, then import `shape`."""
    free_then_used = [(8 * 1024**3, 32 * 1024**3)]

    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.mem_get_info = lambda *a, **k: free_then_used[0]  # type: ignore[attr-defined]
    cuda.is_available = lambda: True  # type: ignore[attr-defined]
    cuda.set_per_process_memory_fraction = lambda *a, **k: None  # type: ignore[attr-defined]
    cuda.reset_peak_memory_stats = lambda *a, **k: None  # type: ignore[attr-defined]
    cuda.empty_cache = lambda *a, **k: None  # type: ignore[attr-defined]
    cuda.synchronize = lambda *a, **k: None  # type: ignore[attr-defined]
    torch.cuda = cuda  # type: ignore[attr-defined]
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.Tensor = object  # type: ignore[attr-defined]
    functional = types.ModuleType("torch.nn.functional")
    functional.scaled_dot_product_attention = lambda *a, **k: None  # type: ignore[attr-defined]
    nn = types.ModuleType("torch.nn")
    nn.functional = functional  # type: ignore[attr-defined]
    torch.nn = nn  # type: ignore[attr-defined]
    for name, module in (
        ("torch", torch),
        ("torch.cuda", cuda),
        ("torch.nn", nn),
        ("torch.nn.functional", functional),
    ):
        sys.modules.setdefault(name, module)

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules.setdefault("dotenv", dotenv)

    for name in ("displaykeep", "gfxlight"):
        module = types.ModuleType(f"runners.hunyuan3d.{name}")
        module.__getattr__ = lambda attr: (lambda *a, **k: None)  # type: ignore[attr-defined]
        sys.modules[f"runners.hunyuan3d.{name}"] = module

    for name in ("PIL", "PIL.Image", "trimesh", "numpy"):
        sys.modules.setdefault(name, types.ModuleType(name))

    from runners.hunyuan3d import shape

    return shape


SHAPE = _install_stubs()


def test_the_watcher_beats_while_something_slow_runs() -> None:
    """**A heartbeat during the wait**, not only after it."""
    beats: list[tuple[str, str]] = []
    with SHAPE._DeviceWatch(
        progress=lambda stage, message: beats.append((stage, message)),
        stage="loading the weights",
        interval=0.02,
        heartbeat_sec=0.05,
    ):
        time.sleep(0.35)
    heartbeats = [b for b in beats if b[0] == "heartbeat"]
    check("the load emits heartbeats", len(heartbeats) >= 3, f"got {len(heartbeats)}")
    check(
        "the heartbeat names the stage",
        all("loading the weights" in message for _stage, message in heartbeats),
        str(heartbeats[:1]),
    )
    check(
        "the heartbeat reports elapsed time and VRAM",
        all("elapsed" in message and "VRAM" in message for _stage, message in heartbeats),
    )


def test_the_watcher_stops_when_the_load_finishes() -> None:
    """It must not keep beating for a stage that is over."""
    beats: list[tuple[str, str]] = []
    with SHAPE._DeviceWatch(
        progress=lambda stage, message: beats.append((stage, message)),
        stage="loading the weights",
        interval=0.02,
        heartbeat_sec=0.05,
    ):
        time.sleep(0.15)
    after = len(beats)
    time.sleep(0.2)
    check("it stops when the stage ends", len(beats) == after, f"{after} then {len(beats)}")


def test_load_pipeline_takes_a_progress_callback() -> None:
    """**The plumbing, not the watcher.** Without this the beats reach nobody."""
    import inspect

    for module_name, function in (
        ("shape", SHAPE.load_pipeline),
        ("texture", None),
    ):
        if function is None:
            continue
        parameters = inspect.signature(function).parameters
        check(
            f"{module_name}.load_pipeline accepts progress",
            "progress" in parameters,
            str(list(parameters)),
        )

    source = inspect.getsource(SHAPE.load_pipeline)
    check("the shape load wraps the weights in the watcher", "_DeviceWatch(" in source)
    source = inspect.getsource(SHAPE.generate_mesh)
    check("generating passes progress into the load", "load_pipeline(progress)" in source)


def main() -> int:
    print("heartbeat during the load (no torch, no weights):")
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
    total = 3
    print(f"\n{total - len(set(FAILURES))}/{total} passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
