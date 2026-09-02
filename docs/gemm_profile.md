# Where the GPU time goes in one shape generation

Measured with [`tools/profile_gemm.py`](../tools/profile_gemm.py) on gfx1151
(Radeon 8060S), Windows 11, ROCm 7.2.1, torch 2.9.1+rocm7.2.1, clock keepalive
on, fast attention on, 2026-09-02. Reference fp16 GEMM taken alongside:
24–25 TFLOPS at 2048³, 31 TFLOPS at 4096³ (rocBLAS). Sample:
`assets/sample.png`, defaults (`steps=30`, `octree_resolution=384`). The
texture stage is opt-in and was not profiled.

The shape pipeline is one upstream call (DiT denoising plus the FlashVDM
volume decode), so it is profiled as one stage. Shares are of profiled device
time; the wall is from an unprofiled run. This torch build has no Kineto, so
shares are decision-grade rather than exact.

| Stage | Wall | GEMM | Attention | Other |
|---|--:|--:|--:|--:|
| shape | 82.6 s | 42.7 s (52 %) | 18.8 s (23 %) | 20.5 s |

**Unlike the TRELLIS-family pipelines, this one is GEMM-bound**, and its
shapes are large and well-proportioned. The largest, all fp16 (the model is
loaded in fp16; the profiler's dispatch recorder does not see inside
`torch.inference_mode`, so dtypes here come from the model, not per call):

| Role | M | N | K | Calls | rocBLAS TFLOPS |
|---|--:|--:|--:|--:|--:|
| DiT attention proj | 8194 | 2048 | 2048 | 2520 | 21.7 |
| DiT MLP down | 8194 | 2048 | 8192 | 630 | 22.0 |
| DiT MLP up | 8194 | 8192 | 2048 | 630 | 26.1 |
| DiT fused proj | 8194 | 2048 | 4096 | 300 | 29.2 |
| Cross-attention | 2740 | 2048 | 1024 | 1260 | 23.5 |
| VAE / FlashVDM | 13824 | 1024 | 4096 | 64 | 20.3 |

rocBLAS already runs these at 21–27 TFLOPS; the headroom to a hand-tuned WMMA
kernel (41–46 TFLOPS at 4096³ on Linux) is roughly 1.7×.

## With hipBLASLt (now the default)

Same measurement with `TORCH_BLAS_PREFER_HIPBLASLT=1` and
`ROCBLAS_USE_HIPBLASLT=1` (what `HUNYUAN3D_PREFER_HIPBLASLT=on` sets): the
shape stage went 82.6 s → 80.4 s — within this machine's variance, but the
per-shape numbers move consistently in hipBLASLt's favour (the unbiased
`mm` projections improved 21.7 → 26.9 TFLOPS; the biased `addmm` rows did not
move). `metrics.blas_backend` records which backend a run used.

## After the ROCm 10.0 update (torch 2.13.0+rocm10.0.0)

Second-run shape wall, same sample and settings, 2026-09-02: **73.2 s**
(1.13× over the 7.2.1 baseline). Reference GEMM alongside: 30.4 / 30.8 TFLOPS
at 2048³ / 4096³ — this pipeline's fat shapes were already near the library
ceiling, so the update moves it least of the three runners. The install traps
and the full measurement set are in gfx1151-gemm's `docs/rocm10.md`.

The three-pipeline comparison, the shape-overlap analysis, and everything
about this GPU that does not depend on the model live in
[gfx1151-gemm](https://github.com/kroqueta-s/gfx1151-gemm).
