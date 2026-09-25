// Verifies the C-API routes a Depth Anything V2 (relative ViT-L) GGUF through the
// DA2 depth-only path: da_capi_depth_dense returns depth only (no conf/sky/pose),
// is_metric==0 for the relative model, and da_capi_pose_path returns -1 (no pose).
#include "da_capi.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>

static bool finite_all(const float* p, int n){
    for (int i = 0; i < n; ++i) if (!std::isfinite(p[i])) return false;
    return true;
}

int main(){
    const char* gguf = std::getenv("DA_TEST_GGUF_DA2"); // relative ViT-L DA2 GGUF
    if (!gguf) return 77;                               // skip if fixture absent
    const char* png = "assets/samples/desk.jpg";        // WORKING_DIRECTORY = DA_ROOT
    FILE* f = std::fopen(png, "rb");
    if (!f){ std::fprintf(stderr, "sample image %s absent, skipping\n", png); return 77; }
    std::fclose(f);

    // Explicit CPU selection must override any GPU backend compiled into the
    // process (and any DA_DEVICE environment setting).
    da_capi_load_options opts;
    da_capi_load_options_init(&opts);
    opts.n_threads = 1;
    opts.device_policy = DA_CAPI_DEVICE_CPU;
    da_ctx* c = da_capi_load_ex(gguf, &opts);
    if (!c){ std::fprintf(stderr, "da2: explicit CPU load failed\n"); return 1; }
    if (da_capi_is_offloading(c) != 0 || std::string(da_capi_device_name(c)) != "cpu") {
        std::fprintf(stderr, "da2: CPU selection returned device=%s offload=%d\n",
                     da_capi_device_name(c), da_capi_is_offloading(c));
        da_capi_free(c);
        return 1;
    }

    da_capi_depth_model_info mi;
    da_capi_depth_model_info_init(&mi);
    if (da_capi_get_depth_model_info(c, &mi) != 0 ||
        mi.depth_semantics != DA_CAPI_DEPTH_SEMANTICS_RELATIVE ||
        mi.depth_representation != DA_CAPI_DEPTH_REPRESENTATION_Z ||
        mi.camera_intrinsics_capable != 0) {
        std::fprintf(stderr, "da2: bad media metadata sem=%d repr=%d camera=%d\n",
                     mi.depth_semantics, mi.depth_representation,
                     mi.camera_intrinsics_capable);
        da_capi_free(c);
        return 1;
    }

    da_capi_preprocess_desc pd;
    da_capi_preprocess_desc_init(&pd);
    if (da_capi_get_preprocess_desc(c, 1920, 1080, &pd) != 0 ||
        pd.output_width <= 0 || pd.output_height <= 0 ||
        pd.source_to_depth_uv[0] != 1.f || pd.source_to_depth_uv[4] != 1.f ||
        pd.source_to_depth_uv[8] != 1.f ||
        pd.valid_depth_uv[0] != 0.f || pd.valid_depth_uv[1] != 0.f ||
        pd.valid_depth_uv[2] != 1.f || pd.valid_depth_uv[3] != 1.f) {
        std::fprintf(stderr, "da2: bad preprocess descriptor\n");
        da_capi_free(c);
        return 1;
    }

    da_capi_depth_request dr;
    da_capi_depth_request_init(&dr);
    if (dr.output_element_type != DA_CAPI_DEPTH_ELEMENT_F16 ||
        dr.video_to_depth_uv[0] != 1.f || dr.video_to_depth_uv[4] != 1.f ||
        dr.video_to_depth_uv[8] != 1.f ||
        dr.valid_depth_uv[0] != 0.f || dr.valid_depth_uv[1] != 0.f ||
        dr.valid_depth_uv[2] != 1.f || dr.valid_depth_uv[3] != 1.f) {
        std::fprintf(stderr, "da2: bad depth request defaults\n");
        da_capi_free(c);
        return 1;
    }

    int H=0, W=0, is_metric=-1; float *depth=nullptr, *conf=nullptr, *sky=nullptr;
    float ext[12], intr[9];
    int r = da_capi_depth_dense(c, png, &H, &W, &depth, &conf, &sky, ext, intr, &is_metric);
    bool ok = (r == 0) && H>0 && W>0 && depth && !conf && !sky;
    if (ok) ok = (H*W > 0) && finite_all(depth, H*W);
    if (ok) ok = (is_metric == 0); // relative DA2 -> non-metric
    std::fprintf(stderr, "da2 dense: r=%d %dx%d depth=%p conf=%p sky=%p is_metric=%d -> %s\n",
                 r, W, H, (void*)depth, (void*)conf, (void*)sky, is_metric, ok?"OK":"FAIL");
    da_capi_free_floats(depth); da_capi_free_floats(conf); da_capi_free_floats(sky);

    // DA2 has no camera pose: pose_path must fail.
    int rp = da_capi_pose_path(c, png, ext, intr);
    bool okp = (rp == -1);
    std::fprintf(stderr, "da2 pose: r=%d (expect -1) err=\"%s\" -> %s\n",
                 rp, da_capi_last_error(c), okp?"OK":"FAIL");

    da_capi_free(c);
    return (ok && okp) ? 0 : 1;
}
