# GPU offload (CUDA)

depth-anything.cpp can offload model weights and the compute graph to a GPU
through ggml's backend layer. The C++ code calls **only ggml backend APIs** (no
direct CUDA), so the same sources build with or without CUDA — the only
difference is a CMake flag and which device the runtime selects.

> Status: the GPU path is **untested on the development box** (it has no GPU /
> CUDA). It is implemented "correct-by-construction", mirroring the verified
> sibling `locate-anything.cpp` offload, and is **validated on the NVIDIA GB10
> (Blackwell, ARM64, CUDA 13)** DGX via `scripts/validate_gpu.sh`. The CPU path
> is byte-for-byte unchanged (30/30 ctest + e2e corr=1.0).

## Build

CPU-only (default — no CUDA toolkit required):

```bash
cmake -B build -DDA_BUILD_CLI=ON
cmake --build build -j
```

With CUDA:

```bash
cmake -B build-cuda -DDA_BUILD_CLI=ON \
      -DDA_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda -j
```

- `DA_GGML_CUDA=ON` forwards to ggml's `GGML_CUDA` and links the CUDA backend in.
- `CMAKE_CUDA_ARCHITECTURES=native` targets the GPU on the build host (on the
  GB10, Blackwell `sm_121`). If you omit it while `DA_GGML_CUDA=ON`, CMake
  defaults it to `native`; you can override it (e.g. `-DCMAKE_CUDA_ARCHITECTURES=90`).
- All CUDA-specific CMake is guarded behind `if(DA_GGML_CUDA)`, so a CPU-only
  build never touches CUDA settings.

Metal / Vulkan backends are wired the same way (`-DDA_GGML_METAL=ON`,
`-DDA_GGML_VULKAN=ON`).

## Device selection (`DA_DEVICE`)

The compute device is chosen by `da::Backend` from the ggml device registry:

- **unset** — auto-pick the first GPU/iGPU device a compiled-in backend
  registers, else fall back to CPU.
- `DA_DEVICE=cpu` — force the CPU backend (numeric baseline / CPU-only box).
- `DA_DEVICE=<name>` — select a registry device by name, case-insensitive
  (e.g. `CUDA0`, `Vulkan0`, `Metal`).

At startup the backend logs the chosen device:
`da::Backend using device: <name>`. Read that line off a GPU run to learn the
exact device name to pin.

## Offload design

When a non-CPU device is selected (`Backend::is_offloading()` true),
`ModelLoader::offload_weights()` mirrors the GGUF weights onto the device:

- A `no_alloc` device `ggml_context` is created; for every weight tensor a
  device tensor of the same type/shape is added, the context is allocated on the
  backend (`ggml_backend_alloc_ctx_tensors`), bytes are uploaded
  (`ggml_backend_tensor_set`), and the loader's tensor map is repointed at the
  device tensors. Metric-branch aliases (`m_vit.*`/`vit.*` sharing one source
  tensor) are de-duplicated by pointer so each weight is uploaded once.
- **Four host-read tensors are deliberately kept host-resident** because they are
  read via `->data` on the CPU during graph build (they produce host-computed
  graph *inputs*, not graph nodes). Offloading them would turn `->data` into a
  device pointer and crash:
  - `vit.pos_embed` — host bicubic interpolation (`interp_pos_embed`)
  - `vit.camera_token` — host camera-token inject
  - `vit.norm.weight`, `vit.norm.bias` — host post-norm
  - (the metric branch aliases `m_vit.*` → `vit.*`, so the same names apply.)
- On the CPU backend `offload_weights` is a **no-op**: graphs keep referencing
  the GGUF host tensors directly (zero-copy), so the CPU path is byte-identical.
- `offload_weights` is idempotent; the device buffer + context are freed in the
  loader's destructor before the host context.

### GPU-friendly op routing

`Backend::compute()` scopes `gpu_mode()` to the backend executing that graph
(see `src/compute_mode.hpp`). The flag is thread-local, so an explicitly forced
CPU context and a Vulkan/CUDA/Metal context may coexist and execute concurrently
without changing each other's graph construction. In GPU mode the graph builders
route to **standard ggml ops that have accelerator kernels**, instead of the
CPU-tuned custom paths that would force GPU↔CPU round-trips:

- **Conv (`src/dpt_blocks.cpp`)** — 3×3 stride-1 convs use
  `ggml_conv_2d_direct` (CUDA kernel) instead of the CPU-only Winograd custom op
  (a `ggml_custom_4d`). 1×1 convs stay im2col GEMM either way.
- **Attention (`src/attention.cpp`)** — the manual `mul_mat` / `soft_max_ext`
  path (all CUDA-backed F32 ops) instead of `ggml_flash_attn_ext`, whose
  CPU-tuned F32-kv config may not map cleanly onto the CUDA flash kernel.

In both cases the explicit env override (`DA_CONV`, `DA_ATTN`) still takes
precedence. On CPU (`gpu_mode()` false) the defaults are unchanged — Winograd +
flash — so the CPU path is byte-identical.

Unsupported ops are additionally offloaded back to CPU automatically by the
`ggml_backend_sched` scheduler path in `src/backend.cpp`, so the graph runs even
if some op lacks a device kernel.

## Validation

`scripts/validate_gpu.sh` (run on the GB10 / any CUDA box):

1. Builds both a CPU-only (`build-cpu`) and a CUDA (`build-cuda`,
   `-DDA_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native`) `da3-cli`.
2. Runs `da3-cli depth` on the same image with `DA_DEVICE=cpu` and on the GPU,
   writing two PFMs.
3. Compares the depth maps — reports `max|d|`, `mean|d|`, correlation; parity
   passes when `max|d| ≤ 1e-2` and `corr ≥ 0.999` (GPU floating-point ordering
   differs slightly, so an exact bit match is not expected).
4. Benchmarks both with `--repeat 10` and reports the GPU speedup.
5. Prints a clear `PASS`/`FAIL`.

Required env: `DA_GGUF` (model gguf), `DA_IMAGE` (input image). Optional:
`DA_CUDA_DEV` (pin a GPU device name; unset = auto-pick first GPU),
`DA_REPEAT` (default 10), `DA_THREADS`, `DA_TOL`, `DA_CORR`.

```bash
DA_GGUF=models/depth-anything-giant-f32.gguf \
DA_IMAGE=dumps/native_input.png \
bash scripts/validate_gpu.sh
```

## Fused backbone+head graph (single-image depth)

`Engine::depth_native` runs the backbone and DPT head as ONE ggml graph (`build_feats_graph` →
`build_depth_graph`) so the out-layer features stay device-resident — eliminating a feats
GPU→host→GPU round-trip and a second graph setup. The out-layer post-processing
(`cat([local_x, vit.norm(x)])` + token-0 strip) runs as ggml ops instead of a host scalar loop.
`DA_FUSED=0` falls back to the two-graph path. depth_pose / multi-view / metric / gs stay unfused.

Parity: fused vs unfused depth max|d|=1.2e-7 (CPU); CPU-vs-GPU corr=0.999998. On the **unified**
GB10 it's latency-neutral (160 vs 160 ms — the round-trip was already cheap); the win is for
**discrete** (PCIe) GPUs where the feats round-trip is a real copy. No regression anywhere; 31/31 tests.

## Embedded media-player device selection

The C API also supports per-context selection without mutating `DA_DEVICE`:
`da_capi_load_ex()` accepts `DA_CAPI_DEVICE_DEFAULT`, `AUTO`, `CPU`, or `NAMED`.
This is useful for applications that expose an explicit "run depth on CPU/GPU"
setting, and allows CPU and accelerator contexts to coexist in one process.

For depth-only video/photo integration, see `docs/VR_MEDIA_INTEGRATION.md`.
