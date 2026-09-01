# hunyuan3d-strix-halo

**[Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) image-to-mesh
(shape stage) on AMD Strix Halo (gfx1151), Windows, ROCm.**

Upstream runs on this hardware only after two changes made from the launcher —
never by editing upstream code:

1. **Scaled dot-product attention is handled at launch.** With
   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` set before torch is imported
   (the runner does this), flash attention works on gfx1151 and is used as-is —
   measured 5× faster on the DiT than any fallback. Without it, the math
   backend materialises `q @ kᵀ` in fp16, overflows 65504 and turns the
   signed-distance field into noise; the runner then falls back to an fp32
   head-chunked replacement. The choice is made by measurement at startup and
   reported in `metrics.fast_attention`.
2. **`enable_flashvdm` is required.** The default volume decoder queries every
   point of a 385³ grid and does not finish in 30 minutes.

This is a runner for [hearth](https://github.com/kroqueta-s/hearth): it speaks
one JSON object per line over stdin/stdout. It also runs standalone.

| Input image | Mesh (4 views) |
|---|---|
| ![input](assets/sample.png) | ![mesh](assets/preview.png) |

*The bundled [`assets/sample.png`](assets/sample.png) (an SDXL-generated robot)
is the reference specimen for the measurements below.*

## Prerequisites

- Windows 11
- An AMD GPU supported by ROCm on Windows (verified on **Strix Halo / gfx1151**,
  Radeon 8060S)
- AMD Adrenalin driver with **ROCm 7.2.1** support
- **Python 3.12**
- ~20 GB of disk (venv + upstream clone + weights)
- ~15 GB of free VRAM at peak

## Install

```powershell
git clone https://github.com/kroqueta-s/hunyuan3d-strix-halo
cd hunyuan3d-strix-halo
.\install.ps1
```

## Quickstart

Generate a mesh from the bundled sample, no JSON required:

```powershell
.venv\Scripts\python.exe tools\run_single.py --image assets\sample.png --out C:\out
```

Progress streams to the console; the mesh lands in `C:\out\raw.ply`. To
reproduce the benchmark below, run the same command **twice and time the second
run**: the first run includes MIOpen's one-time kernel tuning, which says
nothing about steady-state speed.

## Use

```powershell
.venv\Scripts\python.exe -m runners.hunyuan3d
```

```json
{"id": 1, "method": "capabilities"}
{"id": 2, "method": "image_to_mesh", "params": {"image_path": "C:/in.png", "out_dir": "C:/out"}}
```

`image_to_mesh` writes `raw.ply` and `foreground.png`. Parameters: `steps`,
`octree_resolution`, `guidance_scale`, `seed`, `rembg`.

## The GPU idles at 600 MHz unless something renders

The AMD Windows driver does not raise the GPU power state for compute-only
work: at 99 % compute utilisation the clock sits at **600 MHz** (measured,
2026-09-01). With any 3D rendering alive alongside, the same workload sustains
**2.3–2.9 GHz** — a 4.3× difference on GEMM throughput. This is why generation
time on this machine used to vary between 235 s and 921 s for identical
settings: it depended on whether some UI happened to be animating on the
desktop.

The runner therefore keeps a **hidden 3D render loop** (`gfxlight.py`, pure
ctypes, ~0.4 % of the 3D engine) alive during `image_to_mesh`. It is on by
default (`HUNYUAN3D_GFX_KEEPALIVE`), costs nothing measurable, and whether it
was alive is reported in `metrics.gfx_keepalive`. Measured on the same image:
179 s without it, **86 s** with it.

## Measurements (gfx1151, Radeon 8060S, 32 GB dedicated VRAM)

One image (`assets/sample.png`), clock keepalive on, flash attention on,
2026-09-02:

| Setting | Load | Generate | Peak VRAM | Output |
|---|--:|--:|--:|---|
| `steps=30`, `octree=384` (default) | 32 s | **86 s** | 11.3 GB | 1,227,315 faces, watertight |

Lowering `octree_resolution` opens holes (low-resolution marching cubes); do
not trade quality for speed here.

## Troubleshooting

- **Out of VRAM.** The runner caps torch at `HUNYUAN3D_VRAM_LIMIT_GB` (default
  30 GB) so that overflow fails fast as `torch.OutOfMemoryError` instead of
  silently spilling into shared memory and becoming several times slower. If
  you hit it, close other GPU consumers (check dedicated-VRAM usage in Task
  Manager's Performance tab); peak use for the defaults is about 12 GB.
- **The first run looks hung.** It is not. MIOpen tunes kernels once per
  machine, with the GPU busy the whole time. Do not kill it; every later run
  reuses the tuned kernels. The runner emits a `heartbeat` line every 10 s —
  as long as those keep coming, it is working.

## Limits

- **Shape stage only.** The texture stage is untested here and likely needs
  CUDA-only components.
- **Background removal is not optional.** With a 3-channel image, upstream's
  `ImageProcessorV2.recenter()` sets the entire mask to 255, so the subject is
  never cropped. `rembg=False` means "I already removed the background", not
  "let the model handle it".
- **Smart App Control can block dependencies.** On 2026-09-01 it blocked
  `llvmlite.dll`, which broke `rembg → pymatting → numba → llvmlite`. The runner
  substitutes the unused part of `pymatting` when the real one cannot be
  imported. A freshly installed binary can also be blocked only briefly: retry
  once before concluding it is permanent. Disabling Smart App Control is
  irreversible and is not the answer.

## License

MIT (see [LICENSE](LICENSE)). Upstream Hunyuan3D 2.1 and its weights carry
Tencent's own licence — read it before commercial use. This repository contains
no upstream code and no weights.
