#include "engine.hpp"
#include "common.hpp"
#include "dino_backbone.hpp"
#include "image_io.hpp"
#include "preprocess.hpp"
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <string>
#include "dpt_head.hpp"
#include "cam_pose.hpp"
#include "ray_pose.hpp"
#include "gs_head.hpp"
#include "gs_adapter.hpp"
#include "nested.hpp"

namespace da {
std::unique_ptr<Engine> Engine::load(const std::string& path, int n_threads){
    EngineLoadOptions options;
    options.n_threads = n_threads;
    return load(path, options);
}

std::unique_ptr<Engine> Engine::load(const std::string& path,
                                     const EngineLoadOptions& options){
    std::unique_ptr<Engine> e(new Engine(options.backend));
    if (!e->ml_.load(path)) { DA_LOG("engine: load failed"); return nullptr; }
    e->be_.set_n_threads(options.n_threads > 0 ? options.n_threads : 1);
    if (!e->ml_.offload_weights(e->be_)) { DA_LOG("engine: offload failed"); return nullptr; }
    return e;
}

std::unique_ptr<Engine> Engine::load_nested(const std::string& anyview_gguf,
                                            const std::string& metric_gguf, int n_threads){
    EngineLoadOptions options;
    options.n_threads = n_threads;
    return load_nested(anyview_gguf, metric_gguf, options);
}

std::unique_ptr<Engine> Engine::load_nested(const std::string& anyview_gguf,
                                            const std::string& metric_gguf,
                                            const EngineLoadOptions& options){
    auto e = load(anyview_gguf, options);
    if (!e) { DA_LOG("engine: anyview load failed"); return nullptr; }
    e->metric_ml_.reset(new ModelLoader());
    e->metric_be_.reset(new Backend(options.backend));
    if (!e->metric_ml_->load(metric_gguf)) { DA_LOG("engine: metric load failed"); return nullptr; }
    e->metric_be_->set_n_threads(options.n_threads > 0 ? options.n_threads : 1);
    if (!e->metric_ml_->offload_weights(*e->metric_be_)) { DA_LOG("engine: metric offload failed"); return nullptr; }
    return e;
}
bool Engine::backbone_features(const std::vector<float>& input_chw, int H, int W,
                               std::vector<std::vector<float>>& feats_out){
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> cams;
    return bb.forward(input_chw, H, W, feats_out, cams);
}
bool Engine::depth(const std::string& image_path, std::vector<float>& depth_out,
                   std::vector<float>& conf_out, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth: load image failed"); return false; }
    return depth_image(img, depth_out, conf_out, H, W);
}
bool Engine::depth_image(const Image& img, std::vector<float>& depth_out,
                         std::vector<float>& conf_out, int& H, int& W){
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) { DA_LOG("depth: preprocess failed"); return false; }
    H = p.H; W = p.W;
    std::vector<std::vector<float>> feats;
    if (!backbone_features(p.chw, H, W, feats)) { DA_LOG("depth: backbone failed"); return false; }
    DptHead head(ml_, be_);
    return head.depth(feats, H, W, depth_out, conf_out);
}
bool Engine::is_mono() const {
    // out2b out-channels == 1 (depth only) AND a parallel sky head present.
    ggml_tensor* out2b = ml_.tensor("head.scratch.out2b.weight");
    ggml_tensor* sky   = ml_.tensor("head.scratch.sky_out2b.weight");
    return out2b && sky && (int)out2b->ne[3] == 1;
}
bool Engine::depth_mono(const Image& img, std::vector<float>& depth_out,
                        std::vector<float>& sky_out, int& H, int& W){
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("depth_mono: preprocess_real failed"); return false; }
    H = p.H; W = p.W;
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_mono: backbone failed"); return false; }
    DptHead head(ml_, be_);
    return head.depth_sky(feats, H, W, depth_out, sky_out);
}
bool Engine::depth_mono_path(const std::string& image_path, std::vector<float>& depth_out,
                             std::vector<float>& sky_out, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_mono: load image failed"); return false; }
    return depth_mono(img, depth_out, sky_out, H, W);
}
bool Engine::depth_relative(const Image& img, std::vector<float>& depth_out, int& H, int& W){
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("depth_relative: preprocess_real failed"); return false; }
    H = p.H; W = p.W;

    // Prefer the one-graph media path, but keep the old two-graph implementation
    // as an A/B escape hatch and backend-compatibility fallback.
    const char* fenv = std::getenv("DA_FUSED");
    const bool fused_off = fenv && std::string(fenv) == "0";
    if (!fused_off && depth_preprocessed(p.chw.data(), H, W, depth_out)) return true;

    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_relative: backbone failed"); return false; }
    DptHead head(ml_, be_);
    return head.depth_relative(feats, H, W, ml_.config().head_max_depth, depth_out);
}
bool Engine::depth_relative_path(const std::string& image_path, std::vector<float>& depth_out,
                                 int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_relative: load image failed"); return false; }
    return depth_relative(img, depth_out, H, W);
}


ggml_tensor* Engine::build_depth_only_graph(ggml_context* ctx, ggml_tensor* input_chw,
                                            int H, int W, GraphInputPool& pool,
                                            ggml_type output_type){
    if (!input_chw || H <= 0 || W <= 0) return nullptr;
    const int patch = (int)ml_.config().patch_size;
    if (patch <= 0 || (H % patch) != 0 || (W % patch) != 0) return nullptr;

    DinoBackbone bb(ml_, be_);
    DptHead head(ml_, be_);
    ggml_tensor* feat[4] = { nullptr, nullptr, nullptr, nullptr };
    if (!bb.build_feats_graph(ctx, input_chw, H, W, pool, feat)) return nullptr;
    ggml_tensor* logits = head.build_depth_graph(ctx, feat, H, W, pool,
                                                 nullptr, nullptr, nullptr);
    if (!logits) return nullptr;

    // DPT logits are [W,H,C,1]. Keep depth channel 0 and perform the final
    // model-specific activation in-graph so the device callback receives actual
    // depth, not logits.
    ggml_tensor* depth = ggml_view_2d(ctx, logits, W, H, logits->nb[1], 0);
    depth = ggml_cont(ctx, depth);
    if (is_da2()) {
        if (ml_.config().head_max_depth > 0.f)
            depth = ggml_scale(ctx, ggml_sigmoid(ctx, depth), ml_.config().head_max_depth);
        else
            depth = ggml_relu(ctx, depth);
    } else {
        depth = ggml_exp(ctx, depth);
    }
    if (output_type != GGML_TYPE_F32) {
        if (output_type != GGML_TYPE_F16) return nullptr;
        depth = ggml_cast(ctx, depth, output_type);
    }
    return depth;
}

bool Engine::depth_preprocessed(const float* chw, int H, int W,
                                std::vector<float>& depth_out){
    if (!chw || is_nested()) return false;
    GraphInputPool pool;
    return be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ne[4] = { W, H, 3, 1 };
        ggml_tensor* input = be_.add_graph_input_nd(ctx, pool, chw, ne, 4);
        return build_depth_only_graph(ctx, input, H, W, pool);
    }, depth_out);
}

bool Engine::depth_preprocessed_device(
        ggml_tensor* input_chw, int H, int W, ggml_type output_type,
        const std::function<bool(ggml_backend_t, const ggml_tensor*)>& consume){
    if (!input_chw || !consume || is_nested() || !be_.is_offloading()) return false;
    if (output_type != GGML_TYPE_F32 && output_type != GGML_TYPE_F16) return false;
    if (input_chw->type != GGML_TYPE_F32 || input_chw->ne[0] != W ||
        input_chw->ne[1] != H || input_chw->ne[2] != 3 || input_chw->ne[3] != 1)
        return false;

    // The input may be persistent metadata from another ggml_context, but its
    // storage must belong to the same backend device as this Engine. This catches
    // accidental cross-device pointers before the graph allocator reaches them.
    ggml_backend_buffer_t buf = input_chw->view_src ? input_chw->view_src->buffer
                                                     : input_chw->buffer;
    if (!buf) return false;
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    if (!buft || ggml_backend_buft_get_device(buft) != ggml_backend_get_device(be_.handle()))
        return false;

    GraphInputPool pool;
    return be_.compute_device([&](ggml_context* ctx) -> ggml_tensor* {
        return build_depth_only_graph(ctx, input_chw, H, W, pool, output_type);
    }, consume);
}
bool Engine::depth_native(const std::string& image_path, std::vector<float>& depth_out,
                          std::vector<float>& conf_out, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_native: load image failed"); return false; }
    return depth_native_image(img, depth_out, conf_out, H, W);
}
bool Engine::depth_native_image(const Image& img, std::vector<float>& depth_out,
                                std::vector<float>& conf_out, int& H, int& W){
    // Fused backbone+head graph by default (feats stay device-resident). DA_FUSED=0
    // forces the original two-graph path; cat_token=false models always use unfused.
    const char* fenv = std::getenv("DA_FUSED");
    const bool fused_off = fenv && std::string(fenv) == "0";
    if (ml_.config().cat_token && !fused_off)
        return depth_native_fused(img, depth_out, conf_out, H, W);
    return depth_native_unfused(img, depth_out, conf_out, H, W);
}
bool Engine::depth_native_unfused(const Image& img, std::vector<float>& depth_out,
                                  std::vector<float>& conf_out, int& H, int& W){
    const bool prof = std::getenv("DA_PROFILE") != nullptr;
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    auto ms = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };
    Preprocessed p;
    auto t0 = now();
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("depth_native: preprocess_real failed"); return false; }
    H = p.H; W = p.W;
    auto t1 = now();
    std::vector<std::vector<float>> feats;
    if (!backbone_features(p.chw, H, W, feats)) { DA_LOG("depth_native: backbone failed"); return false; }
    auto t2 = now();
    DptHead head(ml_, be_);
    bool ok = head.depth(feats, H, W, depth_out, conf_out);
    auto t3 = now();
    if (prof) DA_LOG("profile: [unfused] preprocess=%.1fms backbone=%.1fms head=%.1fms",
                     ms(t0,t1), ms(t1,t2), ms(t2,t3));
    return ok;
}
bool Engine::depth_native_fused(const Image& img, std::vector<float>& depth_out,
                                std::vector<float>& conf_out, int& H, int& W){
    const bool prof = std::getenv("DA_PROFILE") != nullptr;
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    auto ms = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };
    Preprocessed p;
    auto t0 = now();
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("depth_native: preprocess_real failed"); return false; }
    H = p.H; W = p.W;
    auto t1 = now();
    DinoBackbone bb(ml_, be_);
    DptHead head(ml_, be_);
    // output_dim from out2b out-channels (depth+conf for base/giant -> 2).
    ggml_tensor* out2b_w = ml_.tensor("head.scratch.out2b.weight");
    const int output_dim = out2b_w ? (int)out2b_w->ne[3] : 2;
    GraphInputPool pool;
    std::vector<float> logits;
    // ONE graph: backbone produces the 4 feat tensors in-graph, the head consumes
    // them directly -> feats never leave the device.
    bool ok = be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* feat[4] = { nullptr, nullptr, nullptr, nullptr };
        if (!bb.build_feats_graph(ctx, p.chw, H, W, pool, feat)) return nullptr;
        return head.build_depth_graph(ctx, feat, H, W, pool, nullptr, nullptr, nullptr);
    }, logits);
    auto t2 = now();
    if (!ok) { DA_LOG("depth_native_fused: compute failed"); return false; }
    const size_t HW = (size_t)H * W;
    if (logits.size() != (size_t)output_dim * HW) { DA_LOG("depth_native_fused: bad logits size"); return false; }
    // Same host post-process as DptHead::run -> bit-identical depth/conf given equal logits.
    depth_out.resize(HW);
    for (size_t i = 0; i < HW; ++i) depth_out[i] = std::exp(logits[i]);
    if (output_dim >= 2) {
        conf_out.resize(HW);
        for (size_t i = 0; i < HW; ++i) conf_out[i] = std::exp(logits[HW + i]) + 1.0f;
    } else {
        conf_out.clear();
    }
    if (prof) DA_LOG("profile: [fused] preprocess=%.1fms graph(backbone+head)=%.1fms",
                     ms(t0,t1), ms(t1,t2));
    return ok;
}
bool Engine::depth_pose_native(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                               std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W){
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("depth_pose_native: preprocess_real failed"); return false; }
    H = p.H; W = p.W;
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_pose_native: backbone failed"); return false; }
    DptHead head(ml_, be_);
    if (!head.depth(feats, H, W, depth, conf)) { DA_LOG("depth_pose_native: depth head failed"); return false; }
    if (cam_tokens.size() < 4) { DA_LOG("depth_pose_native: missing layer-11 cam token"); return false; }
    CamPose cam(ml_, be_);
    std::array<float,9> pe;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) { DA_LOG("depth_pose_native: cam pose failed"); return false; }
    return true;
}
bool Engine::depth_pose_native_path(const std::string& image_path, std::vector<float>& depth,
                                    std::vector<float>& conf, std::array<float,12>& ext,
                                    std::array<float,9>& intr, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_pose_native: load image failed"); return false; }
    return depth_pose_native(img, depth, conf, ext, intr, H, W);
}
bool Engine::has_aux() const {
    return ml_.tensor("head.scratch.rn1_aux.out.weight") != nullptr;
}
bool Engine::depth_pose_rays_native(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                                    std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W){
    if (!has_aux()) {
        DA_LOG("depth_pose_rays: this GGUF has no auxiliary ray head (rebuild with --with-aux for --ray-pose)");
        return false;
    }
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("depth_pose_rays: preprocess_real failed"); return false; }
    H = p.H; W = p.W;
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_pose_rays: backbone failed"); return false; }
    DptHead head(ml_, be_);
    if (!head.depth(feats, H, W, depth, conf)) { DA_LOG("depth_pose_rays: depth head failed"); return false; }
    // Ray field -> pose (use_ray_pose). The aux head runs on the SAME backbone feats.
    std::vector<float> ray, ray_conf; int ray_h = 0, ray_w = 0;
    if (!head.rays(feats, H, W, ray, ray_conf, ray_h, ray_w)) {
        DA_LOG("depth_pose_rays: aux ray head failed");
        return false;
    }
    RayPoseParams pp;
    RayPoseOut o;
    // Production path: no fed indices -> internal SEEDED deterministic RANSAC sampling.
    if (!solve_ray_pose(ray.data(), ray_conf.data(), ray_h, ray_w, H, W,
                        /*rand_idx=*/nullptr, /*sorted_idx=*/nullptr,
                        /*refit_idx=*/nullptr, /*n_refit=*/0, pp, o)) {
        DA_LOG("depth_pose_rays: ray->pose solver failed");
        return false;
    }
    ext = o.extrinsics;
    intr = o.intrinsics;
    return true;
}
bool Engine::depth_pose_rays_native_path(const std::string& image_path, std::vector<float>& depth,
                                         std::vector<float>& conf, std::array<float,12>& ext,
                                         std::array<float,9>& intr, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_pose_rays: load image failed"); return false; }
    return depth_pose_rays_native(img, depth, conf, ext, intr, H, W);
}
bool Engine::depth_pose(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                        std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W){
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) { DA_LOG("depth_pose: preprocess failed"); return false; }
    H = p.H; W = p.W;
    // Run the backbone ONCE; reuse feats for depth and cam_tokens[3] for pose.
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_pose: backbone failed"); return false; }
    DptHead head(ml_, be_);
    if (!head.depth(feats, H, W, depth, conf)) { DA_LOG("depth_pose: depth head failed"); return false; }
    if (cam_tokens.size() < 4) { DA_LOG("depth_pose: missing layer-11 cam token"); return false; }
    CamPose cam(ml_, be_);
    std::array<float,9> pe;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) { DA_LOG("depth_pose: cam pose failed"); return false; }
    return true;
}
bool Engine::depth_pose_multi(const std::vector<Image>& imgs, std::vector<ViewResult>& out, int& H, int& W){
    out.clear();
    if (imgs.empty()) { DA_LOG("depth_pose_multi: no images"); return false; }
    // Preprocess every image; all must yield identical H,W.
    std::vector<std::vector<float>> views_chw;
    views_chw.reserve(imgs.size());
    H = 0; W = 0;
    for (size_t i = 0; i < imgs.size(); ++i){
        Preprocessed p;
        // Real DA3 resize policy (longest side -> img_resize_target, round to patch),
        // same as the single-image native path. Bounds the input resolution, so a
        // full-res photo can't blow the graph up to tens of GB.
        if (!preprocess_real(imgs[i], ml_.config(), p)) { DA_LOG("depth_pose_multi: preprocess failed"); return false; }
        if (i == 0) { H = p.H; W = p.W; }
        else if (p.H != H || p.W != W) { DA_LOG("depth_pose_multi: views differ in H,W"); return false; }
        views_chw.push_back(std::move(p.chw));
    }
    const int S = (int)views_chw.size();
    // One backbone pass over all views (cross-view global attention).
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<std::vector<float>>> feats, cam_tokens;  // [L=4][S][...]
    if (!bb.forward_mv(views_chw, H, W, feats, cam_tokens)) { DA_LOG("depth_pose_multi: backbone failed"); return false; }
    if (feats.size() < 4 || cam_tokens.size() < 4) { DA_LOG("depth_pose_multi: missing out layers"); return false; }
    out.resize(S);
    for (int v = 0; v < S; ++v){
        ViewResult& r = out[v];
        std::vector<std::vector<float>> feats4_v = {
            feats[0][v], feats[1][v], feats[2][v], feats[3][v] };
        DptHead head(ml_, be_);
        if (!head.depth(feats4_v, H, W, r.depth, r.conf)) { DA_LOG("depth_pose_multi: depth head failed"); return false; }
        CamPose cam(ml_, be_);
        std::array<float,9> pe;
        if (!cam.pose(cam_tokens[3][v], H, W, pe, r.ext, r.intr)) { DA_LOG("depth_pose_multi: cam pose failed"); return false; }
    }
    return true;
}
bool Engine::reconstruct(const Image& img, Gaussians& g, int& H, int& W){
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) { DA_LOG("reconstruct: preprocess failed"); return false; }
    H = p.H; W = p.W;
    // One backbone pass: feats[4] feed depth + gs_head; cam_tokens[3] feeds pose.
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("reconstruct: backbone failed"); return false; }
    if (feats.size() < 4 || cam_tokens.size() < 4) { DA_LOG("reconstruct: missing out layers"); return false; }
    DptHead head(ml_, be_);
    std::vector<float> depth, conf;
    if (!head.depth(feats, H, W, depth, conf)) { DA_LOG("reconstruct: depth head failed"); return false; }
    CamPose cam(ml_, be_);
    std::array<float,9> pe; std::array<float,12> ext; std::array<float,9> intr;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) { DA_LOG("reconstruct: cam pose failed"); return false; }
    GsHead gs(ml_, be_);
    std::vector<float> raw_gs, gs_conf;
    if (!gs.raw_gaussians(feats, p.chw, H, W, raw_gs, gs_conf)) { DA_LOG("reconstruct: gs_head failed"); return false; }
    GsAdapter ad;
    if (!ad.build(raw_gs, depth, gs_conf, ext, intr, H, W, g)) { DA_LOG("reconstruct: gs_adapter failed"); return false; }
    g.ext = ext; g.intr = intr; g.H = H; g.W = W;  // input camera, for input-view rendering
    // Colour each gaussian by its own source pixel (de-normalize p.chw, the exact
    // model input). DA3's learned SH-DC colour is a near-grey flat base; the true
    // photo colour gives a faithful input-view presentation (same convention as the
    // point cloud). g.colors empty => callers fall back to SH-DC.
    {
        const auto& mean = ml_.config().img_mean;
        const auto& std  = ml_.config().img_std;
        if (mean.size() >= 3 && std.size() >= 3 && (int)p.chw.size() >= 3*H*W && g.N == H*W) {
            const size_t HW = (size_t)H * W;
            g.colors.resize((size_t)g.N * 3);
            for (size_t pix = 0; pix < HW; ++pix)
                for (int c = 0; c < 3; ++c) {
                    float v = p.chw[(size_t)c*HW + pix] * std[c] + mean[c];
                    g.colors[pix*3 + c] = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                }
        }
    }
    return true;
}
bool Engine::reconstruct_path(const std::string& image_path, Gaussians& g, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("reconstruct: load image failed"); return false; }
    return reconstruct(img, g, H, W);
}
bool Engine::depth_pose_path(const std::string& image_path, std::vector<float>& depth, std::vector<float>& conf,
                             std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_pose: load image failed"); return false; }
    return depth_pose(img, depth, conf, ext, intr, H, W);
}
bool Engine::metric_branch(const Image& img, std::vector<float>& depth_raw,
                           std::vector<float>& sky, int& H, int& W){
    if (!metric_ml_ || !metric_be_) { DA_LOG("metric_branch: engine not loaded via load_nested"); return false; }
    // Use the STREAMING resize policy (preprocess_real, longest side ->
    // img_resize_target ~504), NOT the single-image depth_metric() policy
    // (preprocess = near-full-res). Two reasons: (1) the raw metric depth must be
    // pixel-aligned with the multi-view anyview depth that depth_pose_multi()
    // already produced (same preprocess_real), so a per-pixel least-squares scale
    // is valid; (2) full-res would need a ~6 GB ViT-L compute buffer on top of the
    // giant's ~6.8 GB, blowing a 16 GB card — at 504 the pair fits comfortably.
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) { DA_LOG("metric_branch: preprocess_real failed"); return false; }
    H = p.H; W = p.W;
    DinoBackbone bb(*metric_ml_, *metric_be_);
    std::vector<std::vector<float>> feats_m, cams_m;
    if (!bb.forward(p.chw, H, W, feats_m, cams_m)) { DA_LOG("metric_branch: backbone failed"); return false; }
    DptHead head(*metric_ml_, *metric_be_);
    if (!head.depth_sky(feats_m, H, W, depth_raw, sky)) { DA_LOG("metric_branch: depth_sky failed"); return false; }
    // The metric branch applies its own sky-fill inside da3_metric(x) before alignment.
    NestedAligner::process_mono_sky(depth_raw, sky);
    return true;
}

bool Engine::depth_metric(const Image& img, NestedOut& out, int& H, int& W){
    if (!metric_ml_ || !metric_be_) { DA_LOG("depth_metric: engine not loaded via load_nested"); return false; }
    // Both branches consume the SAME preprocessed input x (da3.py NestedDepthAnything3Net).
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) { DA_LOG("depth_metric: preprocess failed"); return false; }
    H = p.H; W = p.W;

    // --- anyview (GIANT): backbone once -> depth + conf + cam pose ---
    AnyviewOut any;
    {
        DinoBackbone bb(ml_, be_);
        std::vector<std::vector<float>> feats, cam_tokens;
        if (!bb.forward(p.chw, H, W, feats, cam_tokens)) { DA_LOG("depth_metric: anyview backbone failed"); return false; }
        if (cam_tokens.size() < 4) { DA_LOG("depth_metric: missing cam token"); return false; }
        DptHead head(ml_, be_);
        if (!head.depth(feats, H, W, any.depth, any.depth_conf)) { DA_LOG("depth_metric: anyview depth head failed"); return false; }
        CamPose cam(ml_, be_);
        std::array<float,9> pe;
        if (!cam.pose(cam_tokens[3], H, W, pe, any.extrinsics, any.intrinsics)) { DA_LOG("depth_metric: cam pose failed"); return false; }
    }

    // --- metric (ViT-L + DPT/sky): backbone + depth_sky head ---
    MetricOut metric;
    {
        DinoBackbone bb(*metric_ml_, *metric_be_);
        std::vector<std::vector<float>> feats_m, cams_m;
        if (!bb.forward(p.chw, H, W, feats_m, cams_m)) { DA_LOG("depth_metric: metric backbone failed"); return false; }
        DptHead head(*metric_ml_, *metric_be_);
        if (!head.depth_sky(feats_m, H, W, metric.depth, metric.sky)) { DA_LOG("depth_metric: metric depth_sky failed"); return false; }
    }
    // The metric branch applies its own sky-fill inside da3_metric(x) before alignment.
    NestedAligner::process_mono_sky(metric.depth, metric.sky);

    out = NestedAligner::align(any, metric, H, W);
    return true;
}
bool Engine::depth_metric_path(const std::string& image_path, NestedOut& out, int& H, int& W){
    Image img; if (!load_image_rgb(image_path, img)) { DA_LOG("depth_metric: load image failed"); return false; }
    return depth_metric(img, out, H, W);
}
} // namespace da
