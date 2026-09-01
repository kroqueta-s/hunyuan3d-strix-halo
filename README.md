# hunyuan3d-strix-halo

**[Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) image-to-mesh
(shape stage) on AMD Strix Halo (gfx1151), Windows, ROCm.**

Upstream runs on this hardware only after two changes made from the launcher —
never by editing upstream code:

1. **Scaled dot-product attention is replaced.** Upstream asks for the flash and
   memory-efficient kernels while disabling the math one. On this GPU those two
   are unavailable by default, so the request blocks the only backend that
   works. The math backend then materialises `q @ kᵀ` in fp16, overflows 65504
   and turns the signed-distance field into noise. The replacement computes in
   fp32 and splits the heads.
2. **`enable_flashvdm` is required.** The default volume decoder queries every
   point of a 385³ grid and does not finish in 30 minutes.

This is a runner for [hearth](https://github.com/kroqueta-s/hearth): it speaks
one JSON object per line over stdin/stdout. It also runs standalone.

## Install

```powershell
git clone https://github.com/kroqueta-s/hunyuan3d-strix-halo
cd hunyuan3d-strix-halo
.\install.ps1
```

Requirements: Windows, an AMD GPU with ROCm 7.2.1 drivers, Python 3.12, and
about 15 GB of VRAM at peak.

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

## Measurements (gfx1151, Radeon 8060S, 32 GB dedicated VRAM)

| Setting | Load | Generate | Peak VRAM | Output |
|---|--:|--:|--:|---|
| `steps=30`, `octree=384` | 40–78 s | **235–921 s** | 9.8–14.2 GB | 1,275,718 faces, watertight |
| `steps=5`, `octree=128` | 42 s | 155 s | 9.4 GB | 218,995 faces, not watertight |

**Generation time varies by a factor of four for identical settings.** Do not
use it as a pass/fail signal. Lowering `octree_resolution` opens holes.

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
