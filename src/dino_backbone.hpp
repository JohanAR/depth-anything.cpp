#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "rope2d.hpp"
#include <vector>
namespace da {
class DinoBackbone {
public:
    DinoBackbone(ModelLoader& ml, Backend& be) : ml_(ml), be_(be) {}
    // input_chw: [3,H,W] normalized. Returns tokens flattened [embed * (1+N_patch)],
    // ggml-order (embed fastest): element (token,e) at token*embed + e.
    bool prepare_tokens(const std::vector<float>& input_chw, int H, int W, std::vector<float>& out_tokens);
    bool forward(const std::vector<float>& input_chw, int H, int W,
                 std::vector<std::vector<float>>& feats,
                 std::vector<std::vector<float>>& cam_tokens);  // implemented in T15 - leave declared
    // Multi-view (S>=1) backbone with cross-view global attention.
    //   views_chw : S items, each [3,H,W] normalized.
    //   feats     : [L][S] each [256*1536] (cat[local_x, norm(x)], token0 stripped, per view).
    //   cam_tokens: [L][S] each [1536]     (cat[local_x[t0], x[t0]] RAW, per view).
    // For S >= THRESH_FOR_REF_SELECTION (=3) and no input cam-token, a reference
    // view is selected (saddle_balanced) at layer alt_start-1, views are reordered
    // (ref first), processed, and the ORIGINAL view order is restored before feats/
    // cam_tokens are produced. out_b_idx (optional) receives the selected reference
    // view index (0 when no selection is applied, e.g. S<3).
    bool forward_mv(const std::vector<std::vector<float>>& views_chw, int H, int W,
                    std::vector<std::vector<std::vector<float>>>& feats,
                    std::vector<std::vector<std::vector<float>>>& cam_tokens,
                    int* out_b_idx = nullptr);
    // Fused single-image path: produce all four intermediate feature tensors in
    // the caller's graph instead of capturing them to host. Supports both DA3's
    // cat([local_x,norm(x)]) features and DA2/mono/metric norm(x)-only features.
    bool build_feats_graph(ggml_context* ctx, const std::vector<float>& input_chw,
                           int H, int W, GraphInputPool& pool, ggml_tensor* out_feat[4]);
    // Same graph, but consume an already backend-resident normalized CHW tensor
    // [W,H,3,1]. This is the zero-host-copy hook used by Vulkan/media integrations.
    // The tensor may come from a different ggml_context as long as its metadata and
    // backend buffer stay alive through graph execution (weights work the same way).
    bool build_feats_graph(ggml_context* ctx, ggml_tensor* input_chw,
                           int H, int W, GraphInputPool& pool, ggml_tensor* out_feat[4]);
private:
    std::vector<float> interp_pos_embed(int gh, int gw) const;  // host bicubic -> [(1+gh*gw)*embed], token-major embed-minor
    // Full multi-view forward on views already in processing order (ref first when
    // selection is applied). Returns feats/cam_tokens in the SAME (input) order.
    bool forward_mv_ordered(const std::vector<std::vector<float>>& views_chw, int H, int W,
                            std::vector<std::vector<std::vector<float>>>& feats,
                            std::vector<std::vector<std::vector<float>>>& cam_tokens);
    // Pass-A selection helper: run the per-view LOCAL blocks [0, upto) and capture
    // each view's token-0 (cls) feature (un-normalized) -> cls_out[s] = [embed].
    bool capture_local_cls(const std::vector<std::vector<float>>& views_chw, int H, int W,
                           int upto, std::vector<std::vector<float>>& cls_out);
    // saddle_balanced reference-view selection from per-view cls features.
    int select_reference_view_saddle(const std::vector<std::vector<float>>& cls, int embed) const;
    ModelLoader& ml_; Backend& be_;
    // Largest view-count logged by the one-time shape diagnostic so far; the log
    // re-fires only when a bigger window arrives (that window sets the VRAM peak).
    int diag_max_s_ = 0;
};
}
