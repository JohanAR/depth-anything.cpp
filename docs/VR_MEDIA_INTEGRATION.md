# VR / real-time media depth integration

This document defines the media-player boundary for generated depth. The depth
library is a **depth side-stream producer**. It does not create left/right RGB,
SBS frames, or otherwise modify the presentation image.

The source video and generated depth remain independent streams:

```text
                         +-------------------------------> original RGB/video
                         |                                      to Godot/XR
GPU decoded video frame -+
                         |
                         +-> async Depth Anything producer
                                      |
                                      +-------------------> timestamped depth
                                                           to Godot/XR
```

Godot/OpenXR owns the stereo reconstruction policy. It may use cheap disparity
warping, a depth mesh, DIBR, or a future reconstruction method without changing
or rerunning depth inference.

## Responsibilities

The Depth Anything integration is responsible for:

* explicit CPU/accelerator selection;
* accepting a GPU video frame through a thin Vulkan adapter;
* creating a reduced-resolution normalized inference input;
* running depth inference without a full-frame CPU round-trip;
* publishing a GPU-resident depth result, preferably F16 for video;
* attaching the source frame ID and PTS to every depth result;
* publishing an explicit video-UV -> depth-UV transform and valid UV region;
* reporting whether depth is relative or metric and its representation;
* keeping inference off the playback/OpenXR critical path;
* bounded/latest-wins pending work when inference cannot keep up;
* retaining only a small timestamped ring of completed depth frames.

It is deliberately **not** responsible for:

* generating left/right RGB images;
* generating SBS/OU video frames;
* deciding eye separation or convergence;
* performing per-eye image warping or DIBR;
* delaying playback until depth is available.

## Device selection and CPU fallback

The legacy API keeps its current behavior:

```c
da_ctx * ctx = da_capi_load("model.gguf", 8);
```

It still honors `DA_DEVICE`. Embedders should normally use explicit per-context
selection:

```c
da_capi_load_options opts;
da_capi_load_options_init(&opts);
opts.n_threads = 8;

opts.device_policy = DA_CAPI_DEVICE_CPU;   // force CPU
opts.device_policy = DA_CAPI_DEVICE_AUTO;  // GPU/iGPU then CPU
opts.device_policy = DA_CAPI_DEVICE_NAMED;
opts.device_name = "Vulkan0";

da_ctx * ctx = da_capi_load_ex("model.gguf", &opts);
```

CPU execution remains a first-class path. A media player may expose backend
choice directly to the user and may fall back to CPU when Vulkan interop is not
available.

## Model and preprocessing metadata

Use `da_capi_get_depth_model_info()` to query:

* relative vs metric-metre depth;
* depth representation (`Z` for the current single-image paths);
* whether the model's full pipeline has a camera-intrinsics path.

Use `da_capi_get_preprocess_desc()` to query the model input size and ImageNet
normalization constants for a source frame. It also returns
`source_to_depth_uv[9]` and `valid_depth_uv[4]`.

The current `preprocess_real()` policy only resizes the complete image. It does
not crop, pad, rotate, or mirror it, so normalized source UV and normalized depth
UV are currently identical. The transform is nevertheless explicit so a future
preprocessing change cannot silently misalign RGB and depth.

The Vulkan media adapter must compose any transformation it performs before
inference -- decoder crop, display-aspect correction, rotation metadata,
mirroring, etc. -- into the final `video_to_depth_uv` matrix carried with the
result.

UV convention should be kept consistent across the media adapter and renderer:

```text
(0,0) = top-left image edge
(1,1) = bottom-right image edge
pixel centre = ((x + 0.5) / width, (y + 0.5) / height)
```

## GPU-resident depth-only entry point

The core library exposes a backend-resident inference hook rather than Vulkan
handles directly:

```c
da_capi_depth_request req;
da_capi_depth_request_init(&req);
req.frame_id = frame_id;
req.pts_ns = pts_ns;
req.output_element_type = DA_CAPI_DEPTH_ELEMENT_F16;
memcpy(req.video_to_depth_uv, video_to_depth_uv, sizeof(req.video_to_depth_uv));

int r = da_capi_depth_device_tensor_ex(
    ctx,
    input_tensor,      // ggml_tensor*: normalized F32 [W,H,3,1], already on backend
    H, W,
    &req,
    consume_depth,
    user);
```

The call runs backbone + DPT + final model-specific activation as one graph and
invokes the consumer while the final tensor is still backend-resident. No final
`ggml_backend_tensor_get()` is performed.

The result metadata includes:

```text
frame_id
pts_ns
width / height
F16 or F32 element type
relative / metric semantics
Z-depth representation
video_to_depth_uv
valid_depth_uv
optional intrinsics fields
```

The current depth-only fast path does not run the additional camera head, so its
per-frame result has `has_intrinsics == 0`. `camera_intrinsics_capable` from the
model-info query tells an adapter whether a more expensive/full model path could
provide them in the future.

F16 output is intended for the real-time bridge: the graph casts the final depth
tensor to F16 on-device, allowing a direct GPU copy into an external buffer that
can be interpreted as `R16_SFLOAT`. F32 remains available for debugging and
consumers that require it.

## Asynchronous producer

`da_capi_depth_device_tensor_ex()` is intentionally synchronous **only with
respect to the inference worker thread**. The Vulkan/media wrapper owns the
asynchronous producer around it.

Recommended producer state:

```text
1 active inference
1 replaceable pending candidate
3-5 completed timestamped depth results
```

When depth inference is keeping up, process buffered source frames in PTS order.
When it falls behind, replace stale pending work with the newest useful frame.
Never build an inference FIFO that attempts to catch up by processing old video
frames.

A completed result is immutable until its output-ring slot is recycled. For a
rendered video frame at PTS `T`, the renderer should normally select:

1. exact depth for `T`, if ready;
2. otherwise the newest completed depth with `depth_pts <= T`.

A future renderer may interpolate/reproject between entries in the small result
ring. It must never require OpenXR presentation to wait for depth.

Photos use the same producer without stale-frame dropping: one submitted image
can remain pending until its depth result is available.

## Recommended Vulkan wrapper contract

The thin player-facing wrapper built on the core API should expose a contract of
this shape (names are illustrative):

```cpp
struct VideoDepthInput {
    VkImage image;
    uint32_t width, height;
    VkImageLayout layout;
    uint64_t frame_id;
    int64_t pts_ns;
    VkSemaphore ready_semaphore;
    uint64_t ready_value;
    Mat3 video_to_image_uv;
};

struct VideoDepthResult {
    VkBuffer depth_buffer;       // F16 by default; image view may be layered on top
    VkDeviceSize offset;
    uint32_t width, height;
    VkFormat format;             // normally VK_FORMAT_R16_SFLOAT
    uint64_t frame_id;
    int64_t pts_ns;
    Mat3 video_to_depth_uv;
    Vec4 valid_depth_uv;
    DepthSemantics semantics;
    DepthRepresentation representation;
    bool has_intrinsics;
    Mat3 intrinsics;
    VkSemaphore ready_semaphore;
    uint64_t ready_value;
};

DepthSubmitResult submit(const VideoDepthInput& frame);
bool try_acquire_latest(int64_t not_after_pts_ns, VideoDepthResult& out);
void release(const VideoDepthResult& result);
```

`submit()` must not wait for inference. A pending frame may be replaced by a
newer one. `try_acquire_latest()` selects a completed depth result without
blocking and never returns a result from the future relative to the requested
video PTS. The returned Vulkan resource remains valid until `release()`.

The wrapper may preprocess every submitted frame immediately into a small input
slot, or delay preprocessing until a frame is selected for inference; either is
valid as long as source-image lifetime and queue synchronization are explicit.
The expensive model inference itself must remain latest-wins rather than FIFO.

## Vulkan bridge

For the current player architecture, assume three logical Vulkan devices on the
same physical GPU:

```text
A = media/decode device
D = ggml / Depth Anything device
G = Godot/OpenXR device
```

The intended GPU-only path is:

```text
full-resolution decoded image on A
    -> resize / RGB conversion / normalization on A
    -> small external input buffer
    -> GPU copy into persistent ggml input tensor on D
    -> depth-anything.cpp fused inference on D
    -> F16/F32 backend-resident depth tensor
    -> GPU copy into external output-ring buffer
    -> imported/bridged depth resource for G

original full-resolution RGB follows its existing A -> G path independently
```

Only the reduced model input and low-resolution depth stream cross the depth
bridge. The original video frame does not need to pass through ggml.

A GPU->GPU staging copy is fine. The normal path should avoid full-frame or
full-depth GPU->CPU->GPU transfers.

The previously developed generic ggml Vulkan interop hooks are a suitable basis:

* enable `VK_KHR_external_memory_fd` / `VK_KHR_external_semaphore_fd` when
  available on ggml's Vulkan device;
* expose the ggml `VkDevice`, physical device and compute queue family;
* submit bridge copies through ggml's own queue synchronization;
* expose the `VkBuffer`/offset/size backing a backend-resident ggml tensor.

Those hooks allow app-owned external bridge buffers while leaving ggml's own
allocator and graph memory private.

The source archive used for this patch contains an empty `third_party/ggml`
submodule directory. Therefore the model-library side of the contract is included
here, while the concrete external-FD/timeline-semaphore Vulkan adapter must be
built against the exact populated ggml submodule revision.

## Separation of concerns

The intended final layering is:

```text
DepthAnything model/core
    model loading
    CPU/GPU selection
    fused depth graph
    backend-resident depth tensor + metadata
             |
             v
Vulkan depth producer
    VkImage preprocessing
    A <-> D external-memory/semaphore bridge
    inference worker
    latest-wins pending job
    timestamped completed-depth ring
             |
             v
Godot/XR renderer
    independent original RGB stream
    depth selection/interpolation
    disparity warp / depth mesh / DIBR
    per-eye OpenXR rendering
```

No layer below Godot/XR creates stereo RGB output.
