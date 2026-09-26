#ifndef DA_CAPI_H
#define DA_CAPI_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct da_ctx da_ctx;
/* ABI version. 3: added da_capi_depth_dense, da_capi_points, da_capi_free_bytes.
   4: added da_capi_load_nested (two-branch metric model).
   5: added da_capi_points_multi (fused multi-view cloud) + da_capi_gaussians.
   6: added da_capi_points_stream (sliding-window streaming cloud); removed the
      unused da_capi_depth_pose_multi.
   7: da_capi_gaussians also returns the input camera intrinsics + processed size
      (out_intr[9], out_w, out_h) so the viewer can render from the input view.
   8: da_capi_points_stream gained a `metric` flag (rescale the streamed cloud to
      metres via the nested metric branch; requires a da_capi_load_nested ctx).
   9: added da_capi_stream_last_poses — per-input-frame camera poses from the last
      da_capi_points_stream call (for the viewer flythrough).
   10: added da_capi_set_fuse_params — scene-relative TSDF voxel/truncation knobs for
      the next streamed fuse (exposed as viewer sliders).
   11: added temporal exposed-face voxel mesh generation and retrieval for streamed
      fused scenes.
   12: added explicit per-context compute-device selection and raw/preprocessed
      single-image depth entry points for embedded media pipelines.
   13: added timestamped GPU depth-result metadata, UV transforms, depth semantics,
      model/preprocess descriptors, and F16 device output for asynchronous media
      depth side-stream producers.
   14: added intermediate resize geometry and relative inverse-depth metadata;
      device-tensor execution now rejects intermediate CPU fallback. */
int         da_capi_abi_version(void);

/* Device selection for da_capi_load_ex / da_capi_load_nested_ex. DEFAULT keeps
   the historical DA_DEVICE environment-variable behaviour. AUTO ignores it and
   chooses GPU/IGPU then CPU. CPU explicitly forces CPU even in a GPU-enabled
   build. NAMED selects a ggml registry device by name (e.g. "Vulkan0"). */
enum {
    DA_CAPI_DEVICE_DEFAULT = 0,
    DA_CAPI_DEVICE_AUTO    = 1,
    DA_CAPI_DEVICE_CPU     = 2,
    DA_CAPI_DEVICE_NAMED   = 3,
};
typedef struct da_capi_load_options {
    size_t      struct_size;
    int         n_threads;
    int         device_policy;
    const char* device_name; /* only used with DA_CAPI_DEVICE_NAMED */
} da_capi_load_options;
void        da_capi_load_options_init(da_capi_load_options* options);
da_ctx*     da_capi_load_ex(const char* gguf_path, const da_capi_load_options* options);
da_ctx*     da_capi_load_nested_ex(const char* anyview_gguf, const char* metric_gguf,
                                   const da_capi_load_options* options);
const char* da_capi_device_name(const da_ctx* ctx); /* owned by ctx */
int         da_capi_is_offloading(const da_ctx* ctx);

/* Depth metadata used by real-time media integrations. "Metric meters" means the
   produced scalar is intended to represent camera-space depth in metres. */
enum {
    DA_CAPI_DEPTH_SEMANTICS_UNKNOWN       = 0,
    DA_CAPI_DEPTH_SEMANTICS_RELATIVE      = 1,
    DA_CAPI_DEPTH_SEMANTICS_METRIC_METERS = 2,
};
enum {
    DA_CAPI_DEPTH_REPRESENTATION_UNKNOWN = 0,
    DA_CAPI_DEPTH_REPRESENTATION_Z       = 1,
    DA_CAPI_DEPTH_REPRESENTATION_INVERSE = 2,
};
enum {
    DA_CAPI_DEPTH_ELEMENT_F32 = 0,
    DA_CAPI_DEPTH_ELEMENT_F16 = 1,
};

typedef struct da_capi_depth_model_info {
    size_t struct_size;
    int depth_semantics;
    int depth_representation;
    /* Non-zero when the model's full image pipeline has a camera-intrinsics path.
       The depth-only device-tensor entry point does not run that additional path. */
    int camera_intrinsics_capable;
} da_capi_depth_model_info;
void da_capi_depth_model_info_init(da_capi_depth_model_info* info);
int  da_capi_get_depth_model_info(da_ctx* ctx, da_capi_depth_model_info* info);

/* Exact resize/normalization contract for a source image. source_to_depth_uv is a
   normalized homogeneous 3x3 transform from the source image UV domain to the
   model depth UV domain. Current preprocess_real() resizes the complete image and
   therefore returns identity; making it explicit prevents future crop/pad changes
   from silently misaligning renderer depth. valid_depth_uv = {min_u,min_v,max_u,max_v}. */
typedef struct da_capi_preprocess_desc {
    size_t struct_size;
    int output_width;
    int output_height;
    float mean[3];
    float std[3];
    float source_to_depth_uv[9];
    float valid_depth_uv[4];
} da_capi_preprocess_desc;
void da_capi_preprocess_desc_init(da_capi_preprocess_desc* desc);
int  da_capi_get_preprocess_desc(da_ctx* ctx, int src_w, int src_h,
                                 da_capi_preprocess_desc* desc);
/* Dimensions after boundary resize, before patch rounding. Each resize
   quantizes to RGB8; upsampling uses cubic (-0.75), downsampling uses area. */
int da_capi_get_preprocess_intermediate_size(da_ctx* ctx, int src_w, int src_h,
                                            int* width, int* height);

/* Advanced interop only: borrowed ggml_backend_t as an opaque pointer, valid until
   da_capi_free(ctx). This lets a backend-specific adapter import/allocate persistent
   device tensors without exposing ggml headers through this C API. */
void*       da_capi_backend_handle(da_ctx* ctx);
/* Set scene-relative TSDF surface-fusion knobs consumed by the NEXT
   da_capi_points_stream whose fuse flag is set. voxel_frac: voxel edge as a fraction
   of the reconstruction's bbox diagonal (fusion detail; <=0 => 0.004). trunc_mult:
   truncation as a multiple of the voxel edge (merge range = 2*trunc_mult voxels;
   <=0 => 4). Split out because da_capi_points_stream is at the purego arg ceiling. */
void        da_capi_set_fuse_params(da_ctx* ctx, double voxel_frac, double trunc_mult);
/* Enable/disable a DTVM temporal exposed-face mesh for the NEXT streamed fuse.
   The mesh retains exact TSDF integer voxel keys and face lifetimes derived from
   each voxel's first-observing frame. Disabled by default. */
void        da_capi_set_temporal_voxel_mesh(da_ctx* ctx, int enabled);
da_ctx*     da_capi_load(const char* gguf_path, int n_threads);  /* NULL on failure */
/* Load a NESTED metric model from its two branches: the anyview (GIANT) GGUF and
   the metric (ViT-L + DPT/sky) GGUF. The returned ctx runs the nested metric
   alignment: da_capi_depth_dense / da_capi_depth_path / da_capi_pose_path all
   produce the final metric-scale depth + scaled extrinsics (is_metric=1, conf/sky
   = NULL). NULL on failure. */
da_ctx*     da_capi_load_nested(const char* anyview_gguf, const char* metric_gguf, int n_threads);
void        da_capi_free(da_ctx* ctx);                           /* safe on NULL */
/* malloc'd JSON describing model config; free via da_capi_free_string. */
char*       da_capi_info_json(da_ctx* ctx);
void        da_capi_free_string(char* s);
const char* da_capi_last_error(da_ctx* ctx);                     /* owned by ctx, "" if none */
/* Run depth on an image file. On success writes *out_h,*out_w and returns a malloc'd
   float[H*W] depth map (row-major); caller frees via da_capi_free_floats. NULL on error. */
float* da_capi_depth_path(da_ctx* ctx, const char* image_path, int* out_h, int* out_w);
/* CPU/host-memory media path: HWC RGB8 with caller-provided row stride in bytes
   (0 => tightly packed width*3). The normal model preprocessing is applied. */
float* da_capi_depth_rgb8(da_ctx* ctx, const unsigned char* rgb, int width, int height,
                          size_t row_stride, int* out_h, int* out_w);
/* Already-preprocessed host input: normalized planar F32 CHW [3,H,W]. This skips
   image decode and resize/normalization and runs the fused depth-only graph. */
float* da_capi_depth_chw_f32(da_ctx* ctx, const float* chw, int h, int w);
/* Metadata attached to one asynchronous media-depth request. The Vulkan/media
   wrapper should populate frame_id and pts_ns from the decoded source frame and
   compose all decoder crop/rotation/aspect corrections into video_to_depth_uv.
   Use da_capi_depth_request_init() before overriding fields. */
typedef struct da_capi_depth_request {
    size_t struct_size;
    uint64_t frame_id;
    int64_t pts_ns;
    int output_element_type; /* DA_CAPI_DEPTH_ELEMENT_* */
    float video_to_depth_uv[9];
    float valid_depth_uv[4]; /* {min_u,min_v,max_u,max_v} */
} da_capi_depth_request;
void da_capi_depth_request_init(da_capi_depth_request* request);

/* Description of the backend-resident depth tensor passed to the consumer.
   The Vulkan bridge adds its own exported buffer/image and completion
   semaphore/value around this result; this structure deliberately contains no
   Vulkan handles and no stereo/RGB output. */
typedef struct da_capi_device_depth_result {
    size_t struct_size;
    uint64_t frame_id;
    int64_t pts_ns;
    int width;
    int height;
    int element_type;         /* DA_CAPI_DEPTH_ELEMENT_* */
    int depth_semantics;      /* DA_CAPI_DEPTH_SEMANTICS_* */
    int depth_representation; /* DA_CAPI_DEPTH_REPRESENTATION_* */
    float video_to_depth_uv[9];
    float valid_depth_uv[4];
    int has_intrinsics;
    float intrinsics[9];
} da_capi_device_depth_result;

/* Advanced zero-host-readback hook for GPU media pipelines. input_tensor is a
   backend-resident ggml_tensor* containing normalized F32 [W,H,3,1] data on the
   same device selected for ctx. Inference is synchronous with respect to this
   call (run it on the producer's inference worker), but all full-frame data stays
   on the backend. The consumer receives the final depth tensor plus timestamped
   side-stream metadata and must GPU-copy/consume the tensor before returning.
   It must not retain backend/tensor pointers after the callback. */
typedef int (*da_capi_device_depth_consumer_ex)(
    void* user, void* backend, const void* depth_tensor,
    const da_capi_device_depth_result* result);
int da_capi_depth_device_tensor_ex(da_ctx* ctx, void* input_tensor, int h, int w,
                                   const da_capi_depth_request* request,
                                   da_capi_device_depth_consumer_ex consume, void* user);

/* Compatibility form: F32 output, identity UV transform, no frame timestamp. */
typedef int (*da_capi_device_depth_consumer)(void* user, void* backend, const void* depth_tensor);
int da_capi_depth_device_tensor(da_ctx* ctx, void* input_tensor, int h, int w,
                                da_capi_device_depth_consumer consume, void* user);

/* Legacy geometry/normalization query retained for existing callers. New media
   integrations should use da_capi_get_preprocess_desc() so the UV mapping is
   explicit even when it is identity today. */
int da_capi_preprocess_info(da_ctx* ctx, int src_w, int src_h,
                            int* out_w, int* out_h, float mean3[3], float std3[3]);
void   da_capi_free_floats(float* p);
/* Run pose; fills ext[12] (3x4 row-major) and intr[9] (3x3). Returns 0 ok, -1 error. */
int da_capi_pose_path(da_ctx* ctx, const char* image_path, float out_ext[12], float out_intr[9]);
/* Single-image 3D export. Runs the native depth+pose pipeline, captures the
   processed-resolution RGB colors, and writes a glTF-2.0 binary point cloud to
   out_glb. Returns 0 ok, -1 error (see da_capi_last_error). */
int da_capi_export_glb(da_ctx* ctx, const char* image_path, const char* out_glb);
/* Single-image 3D export to a COLMAP sparse model (cameras/images/points3D) in
   directory out_dir. binary != 0 => .bin (default); 0 => .txt. Returns 0 ok, -1 error. */
int da_capi_export_colmap(da_ctx* ctx, const char* image_path, const char* out_dir, int binary);

/* Dense per-pixel output for a single image. Returns 0 ok, -1 error.
   Writes processed dims to *out_h,*out_w. Each non-NULL out_* float buffer is
   malloc'd [H*W] row-major and must be freed via da_capi_free_floats; buffers
   not produced by the model are set to NULL.
     - DualDPT model (camera-pose capable): *out_depth + *out_conf are filled,
       *out_sky = NULL, out_ext[12] (3x4 row-major) + out_intr[9] (3x3) filled.
     - mono model (DA3MONO): *out_depth + *out_sky are filled, *out_conf = NULL,
       out_ext/out_intr zeroed (mono has no camera pose).
     - nested model (da_capi_load_nested): *out_depth = final metric-scale depth,
       *out_conf = *out_sky = NULL, out_ext/out_intr = scaled extrinsics/intrinsics.
   *out_is_metric = 1 for metric/nested/mono variants (best-effort from config),
   else 0. Any of out_h/out_w/out_depth/out_conf/out_sky/out_is_metric may be NULL;
   out_ext/out_intr must point to 12/9 floats respectively. */
int da_capi_depth_dense(da_ctx* ctx, const char* image_path, int* out_h, int* out_w,
                        float** out_depth, float** out_conf, float** out_sky,
                        float out_ext[12], float out_intr[9], int* out_is_metric);

/* Single-image 3D point cloud (DualDPT/pose-capable models only; returns -1 for
   mono models with a clear last_error). Runs depth+pose+processed-RGB, back-projects
   to world space keeping pixels with conf >= conf_thresh. On success sets *out_n
   and writes a malloc'd *out_xyz[3*N float] + *out_rgb[3*N uint8]; free xyz via
   da_capi_free_floats and rgb via da_capi_free_bytes. Returns 0 ok, -1 error. */
int da_capi_points(da_ctx* ctx, const char* image_path, float conf_thresh,
                   int* out_n, float** out_xyz, unsigned char** out_rgb);
/* Free a uint8 buffer returned by da_capi_points (out_rgb). */
void da_capi_free_bytes(unsigned char* p);

/* Multi-view FUSED colored point cloud (DualDPT/pose-capable DA3 models; returns
   -1 for mono/DA2). Runs ONE cross-view depth+pose pass over n_images and back-
   projects every view into a single shared world frame, so the cloud is coherent
   across frames (no per-frame scale drift). Recovers per-pixel processed RGB and a
   perspective-correct per-point radius (~depth/focal * point_size). Keeps pixels
   with confidence >= percentile(conf, conf_pct) (conf_pct in [0,100]). Points are
   emitted grouped by source view (frames outer); if out_counts != NULL it must
   point to int[n_images] and receives the per-view point count (for progressive
   build-up rendering). On success sets *out_n and mallocs *out_xyz[3N] (world xyz,
   OpenCV frame), *out_rgb[3N] uint8, *out_radius[N]; free xyz/radius via
   da_capi_free_floats and rgb via da_capi_free_bytes. Returns 0 ok, -1 error. */
int da_capi_points_multi(da_ctx* ctx, const char** image_paths, int n_images,
                         double conf_pct, float point_size,
                         int* out_n, int* out_counts,
                         float** out_xyz, unsigned char** out_rgb, float** out_radius);

/* Sliding-window STREAMING colored point cloud (DualDPT/pose-capable DA3 models;
   returns -1 for mono/DA2). For long ordered frame sequences (time-lapse video):
   runs the fused cross-view pass on overlapping windows of chunk_size frames
   (sharing `overlap` frames) and stitches each window into ONE global frame via a
   weighted-Umeyama Sim3 solved on the overlap, so hundreds of frames fuse into a
   single coherent cloud at bounded per-pass memory. conf_pct in [0,100];
   global_budget caps total emitted points (<=0 = unlimited, subsampled per window).
   out_counts (if non-NULL) is int[n_images] receiving per-INPUT-frame point counts
   (frame-major, for progressive build-up). On success sets *out_n and mallocs
   *out_xyz[3N] (world xyz, OpenCV frame), *out_rgb[3N] uint8, *out_radius[N]; free
   xyz/radius via da_capi_free_floats and rgb via da_capi_free_bytes. 0 ok, -1 err.

   De-ghosting toggles (0=off, non-zero=on):
     icp_refine  : point-to-plane ICP refinement of each window seam;
     loop_close  : loop-closure detection + Sim3 pose-graph drift removal;
     fuse        : final voxel/surface fusion (collapses doubled sheets, de-densifies)
                   with cell fuse_voxel_m metres (<=0 => 0.03 m default). After fusion
                   the point order is spatial, so out_counts is rescaled to sum to N.
     metric      : rescale each window to absolute METRES via the nested metric branch
                   (per-window robust-median scale_factor applied to depth + pose).
                   REQUIRES a da_capi_load_nested ctx; returns -1 otherwise. Without it
                   the cloud is at an arbitrary relative scale.

   Per-frame camera poses (for the viewer flythrough) are stashed on the ctx and
   retrieved separately via da_capi_stream_last_poses — this function is already at the
   FFI argument-count ceiling, so they can't be added as out-params here. */
int da_capi_points_stream(da_ctx* ctx, const char** image_paths, int n_images,
                          int chunk_size, int overlap, double conf_pct,
                          float point_size, int global_budget,
                          int icp_refine, int loop_close, int fuse, int metric,
                          double fuse_voxel_m,
                          int* out_n, int* out_counts,
                          float** out_xyz, unsigned char** out_rgb, float** out_radius);

/* Per-input-frame camera poses from the most recent da_capi_points_stream call.
   Mallocs *out_pos[3F] (camera centre) and *out_fwd[3F] (unit view direction), OpenCV
   world axes; sets *out_nframes = F (== that call's n_images). Free via
   da_capi_free_floats. Returns 0 ok, -1 if none (no stream run / all windows failed). */
int da_capi_stream_last_poses(da_ctx* ctx, float** out_pos, float** out_fwd, int* out_nframes);

/* Retrieve the DTVM v1 temporal voxel mesh generated by the most recent fused
   da_capi_points_stream call. Mallocs *out_data[*out_size]; free with
   da_capi_free_bytes. Returns 0 on success, -1 when generation was disabled or
   no valid fused surface was produced. */
int da_capi_stream_last_voxel_mesh(da_ctx* ctx, unsigned char** out_data, size_t* out_size);

/* Single-image 3D GAUSSIANS (DA3-GIANT / GS-head models only; returns -1 with a
   clear last_error otherwise). Returns world-frame (OpenCV) gaussians as parallel
   arrays: *out_xyz[3N] means, *out_scale[3N], *out_quat[4N] (wxyz), *out_rgb[3N]
   linear colour in [0,1] (input-photo colour; see note), *out_opacity[N]. Sets
   *out_n. Also, when non-NULL, writes the input camera intrinsics to out_intr[9]
   (K 3x3 row-major, pixel units) and the processed resolution to *out_w,*out_h --
   the pose is canonical (identity) so a viewer placed at the origin looking down
   +z with this K reproduces the input view. Free every returned float buffer via
   da_capi_free_floats. Returns 0 ok, -1 error. */
int da_capi_gaussians(da_ctx* ctx, const char* image_path, int* out_n,
                      float** out_xyz, float** out_scale, float** out_quat,
                      float** out_rgb, float** out_opacity,
                      float* out_intr, int* out_w, int* out_h);
#ifdef __cplusplus
}
#endif
#endif
