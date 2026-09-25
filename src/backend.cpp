#include "backend.hpp"
#include "common.hpp"
#include "compute_mode.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace da {

namespace {
// Number of graph nodes the metadata context must hold. Sized for the heaviest
// supported graph: a full multi-view streaming window on GIANT (40 layers) at the
// max chunk of 24 views (~33k nodes). 16384 fit base multi-view and single-image
// giant but overflowed giant x 12-view by a hair. The per-compute metadata context
// + graph hash-set scale with this (~408 B/node, ~20 MB here) — cheap.
constexpr size_t kGraphSize = 49152;

struct PendingInput {
    ggml_tensor* tensor;
    const float* host;
    size_t       nbytes;
};
// Extra graph tensors to read back after compute (besides the final output).
struct PendingCapture {
    ggml_tensor*        tensor;
    std::vector<float>* dst;
};
}  // namespace

struct Backend::Impl {
    ggml_backend_t       backend     = nullptr;  // primary device (GPU or CPU)
    ggml_backend_t       cpu_backend = nullptr;  // CPU fallback (GPU path only)
    ggml_gallocr_t       galloc      = nullptr;  // CPU / single-backend path (unchanged)
    ggml_backend_sched_t sched       = nullptr;  // GPU path: schedules over {backend, cpu_backend}
    size_t               sched_nodes = 0;        // graph_size the sched was created with (recreated when a bigger graph arrives)
    bool                 use_sched   = false;    // true only when `backend` is a GPU device
    bool                 flash_diag_logged = false; // one-time flash-attn placement diagnostic (GPU path)
    // Inputs registered by the build lambda for the in-flight compute. Copied
    // into the allocated tensors after the graph is allocated, then cleared.
    // Never overlaps across calls (compute is not re-entrant).
    std::vector<PendingInput>   pending;
    // Extra tensors to read back after compute (registered via capture()).
    std::vector<PendingCapture> captures;
    // Extra graph roots to expand (registered via add_graph_root()) but NOT read
    // back. Used for resident K/V write (ggml_cpy) nodes.
    std::vector<ggml_tensor*>   roots;
};

Backend::Backend(const BackendOptions& options) : impl_(new Impl()) {
    // Preserve the existing environment-driven behaviour by default, but allow
    // embedders (notably media players) to make device choice per-context.
    std::string want;
    bool force_cpu = false;
    switch (options.device_policy) {
        case DevicePolicy::Environment: {
            const char* force = std::getenv("DA_DEVICE");
            want = force ? force : "";
            force_cpu = want == "cpu" || want == "CPU";
            break;
        }
        case DevicePolicy::Auto:
            break;
        case DevicePolicy::Cpu:
            want = "cpu";
            force_cpu = true;
            break;
        case DevicePolicy::Named:
            want = options.device_name;
            break;
    }

    // Case-insensitive equality, used to match DA_DEVICE against the registry's
    // device names (upper-case like "CUDA0"/"Vulkan0").
    auto iequals = [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
                return false;
        return true;
    };

    if (!force_cpu) {
        // Walk the registry. Whatever backend was compiled in
        // (CUDA/Metal/Vulkan/HIP/SYCL) registers itself here, so this single path
        // covers them all with no backend-specific includes. try_pick inits the
        // first registry device matching `match` and returns whether it stuck.
        auto try_pick = [&](auto match) -> bool {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                const auto type = ggml_backend_dev_type(dev);
                const char* name = ggml_backend_dev_name(dev);
                if (!match(type, name)) continue;
                impl_->backend = ggml_backend_dev_init(dev, nullptr);
                if (!impl_->backend) continue;
                device_name_ = name ? name : "";
                // Route compute through ggml_backend_sched for any non-CPU device
                // so unsupported ops can fall back to CPU.
                impl_->use_sched = type != GGML_BACKEND_DEVICE_TYPE_CPU;
                offloading_      = impl_->use_sched;
                DA_LOG("da::Backend using device: %s", device_name_.c_str());
                return true;
            }
            return false;
        };

        if (!want.empty()) {
            // Explicit device request: match by name (e.g. "Vulkan1", "CUDA0").
            if (!try_pick([&](auto, const char* name) { return name && iequals(want, name); }))
                DA_LOG("da::Backend: device %s not found; falling back to CPU",
                       want.c_str());
        } else {
            // Auto-pick: prefer a DISCRETE GPU over an integrated one. On multi-GPU
            // boxes the iGPU often enumerates first (device 0) but is far weaker and
            // can wedge on big models — so only fall back to IGPU if no discrete GPU
            // is present.
            (void)(try_pick([](auto type, const char*) { return type == GGML_BACKEND_DEVICE_TYPE_GPU; }) ||
                   try_pick([](auto type, const char*) { return type == GGML_BACKEND_DEVICE_TYPE_IGPU; }));
        }
    }
    if (!impl_->backend) {              // CPU fallback (or CPU-only build)
        impl_->backend = ggml_backend_cpu_init();
        device_name_ = "cpu";
        offloading_  = false;
        if (!impl_->backend) {
            DA_LOG("da::Backend: ggml_backend_cpu_init failed");
            return;
        }
    }
    // GPU path: create a CPU fallback backend so ops the device lacks a kernel
    // for are offloaded to CPU by the scheduler instead of aborting. The
    // CPU/single-backend path keeps using the persistent gallocr and is
    // untouched.
    if (impl_->use_sched) {
        impl_->cpu_backend = ggml_backend_cpu_init();
        if (!impl_->cpu_backend) {
            DA_LOG("da::Backend: CPU fallback init failed; disabling sched");
            impl_->use_sched = false;
        }
    }
    set_n_threads(n_threads_);
}

Backend::~Backend() {
    if (impl_) {
        // Free allocators/scheduler BEFORE the backends they reference.
        if (impl_->sched)       ggml_backend_sched_free(impl_->sched);
        if (impl_->galloc)      ggml_gallocr_free(impl_->galloc);
        if (impl_->cpu_backend) ggml_backend_free(impl_->cpu_backend);
        if (impl_->backend)     ggml_backend_free(impl_->backend);
    }
}

void Backend::set_n_threads(int n) {
    n_threads_ = n > 0 ? n : 1;
    // Only the CPU backend(s) take a thread count; GPU backends ignore it.
    if (impl_ && impl_->backend && ggml_backend_is_cpu(impl_->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->backend, n_threads_);
    }
    if (impl_ && impl_->cpu_backend) {
        ggml_backend_cpu_set_n_threads(impl_->cpu_backend, n_threads_);
    }
}

ggml_backend_t Backend::handle() const {
    return impl_ ? impl_->backend : nullptr;
}

ggml_tensor* Backend::add_graph_input(ggml_context* ctx, GraphInputPool& pool,
                                      const float* host, size_t n) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)n);
    ggml_set_input(t);
    impl_->pending.push_back({t, pool.keep(host, n), n * sizeof(float)});
    return t;
}

ggml_tensor* Backend::add_graph_input_nd(ggml_context* ctx, GraphInputPool& pool,
                                         const float* host, const int64_t* ne,
                                         int n_dims) {
    size_t total = 1;
    for (int i = 0; i < n_dims; ++i) total *= (size_t)ne[i];
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    ggml_set_input(t);
    impl_->pending.push_back({t, pool.keep(host, total), total * sizeof(float)});
    return t;
}

ggml_tensor* Backend::add_graph_input_nd_borrow(ggml_context* ctx,
                                         const float* host, const int64_t* ne, int n_dims) {
    size_t total = 1;
    for (int i = 0; i < n_dims; ++i) total *= (size_t)ne[i];
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    ggml_set_input(t);
    impl_->pending.push_back({t, host, total * sizeof(float)});   // borrow: no copy
    return t;
}

ggml_tensor* Backend::add_int32_input_nd(ggml_context* ctx, GraphInputPool& pool,
                                         const int32_t* host, const int64_t* ne,
                                         int n_dims) {
    size_t total = 1;
    for (int i = 0; i < n_dims; ++i) total *= (size_t)ne[i];
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_I32, n_dims, ne);
    ggml_set_input(t);
    // The upload is a raw, type-agnostic ggml_backend_tensor_set; reuse the same
    // pending-upload list (host pointer reinterpreted as bytes).
    impl_->pending.push_back({t, (const float*)pool.keep_i32(host, total),
                              total * sizeof(int32_t)});
    return t;
}

void Backend::capture(ggml_tensor* t, std::vector<float>* dst) {
    impl_->captures.push_back({t, dst});
}

ggml_backend_buffer_t Backend::allocate_ctx_tensors(ggml_context* ctx) {
    return ggml_backend_alloc_ctx_tensors(ctx, impl_->backend);
}

void Backend::add_graph_root(ggml_tensor* t) {
    impl_->roots.push_back(t);
}

bool Backend::compute_impl(const std::function<ggml_tensor*(ggml_context*)>& build,
                           std::vector<float>* out,
                           const std::function<bool(ggml_backend_t, const ggml_tensor*)>* consume,
                           size_t graph_nodes) {
    if (!impl_ || !impl_->backend) {
        DA_LOG("Backend::compute called on an uninitialised backend");
        return false;
    }
    const size_t gs = graph_nodes ? graph_nodes : kGraphSize;

    // Metadata-only context: holds graph + tensor structs, no tensor data
    // (no_alloc=true). Tensor data lives in the gallocr's persistent buffer.
    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * gs +
                            ggml_graph_overhead_custom(gs, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        DA_LOG("Backend::compute: ggml_init failed");
        return false;
    }

    // Reset registrations for this compute, then run the build lambda. The build
    // lambda registers host inputs / captures via member calls on this Backend.
    impl_->pending.clear();
    impl_->captures.clear();
    impl_->roots.clear();
    // Upstream graph builders choose a few CPU/GPU-specific ops through gpu_mode().
    // Scope that decision to THIS backend invocation rather than model load, so a
    // CPU context and a Vulkan/CUDA/Metal context can execute concurrently.
    ScopedComputeMode compute_mode(offloading_);
    struct ggml_tensor* output = build(ctx);
    if (!output) {
        DA_LOG("Backend::compute: build() returned null output tensor");
        impl_->pending.clear();
        impl_->captures.clear();
        impl_->roots.clear();
        ggml_free(ctx);
        return false;
    }

    // Mark the output (and any captured tensors) so the gallocr does not recycle
    // their storage before we read them back, then expand the forward graph.
    ggml_set_output(output);
    for (const PendingCapture& pc : impl_->captures) ggml_set_output(pc.tensor);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, gs, false);
    // Expand captures FIRST so they are present in the graph even if the final
    // output's subgraph does not reach them.
    for (const PendingCapture& pc : impl_->captures)
        ggml_build_forward_expand(gf, pc.tensor);
    // Expand registered extra graph roots (e.g. resident K/V ggml_cpy writes) in
    // registration order, BEFORE the output, so their side-effect nodes land in
    // the graph. These are not read back.
    for (ggml_tensor* r : impl_->roots)
        ggml_build_forward_expand(gf, r);
    ggml_build_forward_expand(gf, output);

    // GPU devices default to the fast persistent-gallocr path (identical to a
    // single-backend run). Only route THIS graph through the scheduler (which
    // offloads unsupported ops to CPU) when the GPU backend actually lacks a
    // kernel for one of its ops. The per-graph check is a cheap O(nodes) scan,
    // far less than a sched re-plan. The CPU path never sets use_sched, so it is
    // byte-for-byte unchanged.
    bool need_sched = false;
    if (impl_->use_sched) {
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            if (!ggml_backend_supports_op(impl_->backend, ggml_graph_node(gf, i))) {
                need_sched = true;
                break;
            }
        }
    }

    // One-time diagnostic (GPU path): does the fused flash-attention node actually
    // run on THIS device, or does the scheduler silently offload it to CPU? On
    // Vulkan the flash kernel is gated on coopmat2 / subgroup shuffle+vote and the
    // head-dim being a multiple of 8; when the device lacks those, supports_op()
    // returns false and attention falls back to CPU (correct output, but the O(N^2)
    // score matrix materializes CPU-side and the VRAM/latency win is lost). This
    // fallback is otherwise invisible, so surface it once per backend instance.
    if (impl_->use_sched && !impl_->flash_diag_logged) {
        const int n_nodes = ggml_graph_n_nodes(gf);
        int n_flash = 0, n_flash_off = 0;
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_FLASH_ATTN_EXT) continue;
            ++n_flash;
            if (!ggml_backend_supports_op(impl_->backend, node)) ++n_flash_off;
        }
        if (n_flash > 0) {
            impl_->flash_diag_logged = true;
            if (n_flash_off == 0)
                DA_LOG("flash-attn: %d node(s) run on %s (fused, no N^2 score matrix)",
                       n_flash, device_name_.c_str());
            else
                DA_LOG("flash-attn: %d/%d node(s) UNSUPPORTED on %s -> offloaded to CPU "
                       "(scores materialize CPU-side; check coopmat2 / subgroup "
                       "shuffle+vote support, or run DA_ATTN=manual)",
                       n_flash_off, n_flash, device_name_.c_str());
        }
    }

    bool alloc_ok = false;
    if (need_sched) {
        // GPU path: schedule across {GPU, CPU}. Unsupported ops fall back to CPU.
        // (Re)create the scheduler if absent or too small for this graph. It's created
        // once and reused, so a later, larger window (more views => more nodes) needs a
        // bigger sched than the first call sized it for.
        if (!impl_->sched || gs > impl_->sched_nodes) {
            if (impl_->sched) ggml_backend_sched_free(impl_->sched);
            ggml_backend_t backs[2] = { impl_->backend, impl_->cpu_backend };
            impl_->sched = ggml_backend_sched_new(
                backs, /*bufts=*/nullptr, /*n_backends=*/2,
                /*graph_size=*/gs, /*parallel=*/false, /*op_offload=*/true);
            impl_->sched_nodes = gs;
            if (!impl_->sched) {
                DA_LOG("Backend::compute: ggml_backend_sched_new failed");
                impl_->pending.clear();
                impl_->captures.clear();
                impl_->roots.clear();
                ggml_free(ctx);
                return false;
            }
        }
        ggml_backend_sched_reset(impl_->sched);
        alloc_ok = ggml_backend_sched_alloc_graph(impl_->sched, gf);
        if (!alloc_ok) DA_LOG("Backend::compute: ggml_backend_sched_alloc_graph failed");
    } else {
        // Fast path: CPU, or a GPU whose backend supports every op in this graph.
        // Persistent gallocr over the active backend's buffer type, lazily created
        // and reused on every subsequent call (it only reallocates the underlying
        // buffer when the graph grows beyond the current high-water mark).
        if (!impl_->galloc) {
            impl_->galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(impl_->backend));
            if (!impl_->galloc) {
                DA_LOG("Backend::compute: ggml_gallocr_new failed");
                impl_->pending.clear();
                impl_->captures.clear();
                impl_->roots.clear();
                ggml_free(ctx);
                return false;
            }
        }
        alloc_ok = ggml_gallocr_alloc_graph(impl_->galloc, gf);
        if (!alloc_ok) DA_LOG("Backend::compute: ggml_gallocr_alloc_graph failed");
    }
    if (!alloc_ok) {
        impl_->pending.clear();
        impl_->captures.clear();
        impl_->roots.clear();
        ggml_free(ctx);
        return false;
    }

    // Inputs are allocated now (->buffer/->data set): push host data in. Skip any
    // registered input the graph never referenced — the allocator leaves it without
    // a buffer (e.g. cam_src_in when forward_mv runs with a single view), and
    // uploading to it would assert "tensor buffer not set". Unreachable inputs feed
    // nothing, so there is nothing to upload.
    for (const PendingInput& pi : impl_->pending) {
        ggml_backend_buffer_t ib = pi.tensor->view_src ? pi.tensor->view_src->buffer
                                                        : pi.tensor->buffer;
        if (!ib) continue;
        ggml_backend_tensor_set(pi.tensor, pi.host, 0, pi.nbytes);
    }
    impl_->pending.clear();

    enum ggml_status status = need_sched
        ? ggml_backend_sched_graph_compute(impl_->sched, gf)
        : ggml_backend_graph_compute(impl_->backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        DA_LOG("Backend::compute: ggml_backend_graph_compute failed (status=%d)",
               (int)status);
        impl_->captures.clear();
        impl_->roots.clear();
        ggml_free(ctx);
        return false;
    }

    // Read back any captured intermediates, then the final output.
    for (const PendingCapture& pc : impl_->captures) {
        size_t cn = (size_t)ggml_nelements(pc.tensor);
        pc.dst->resize(cn);
        ggml_backend_tensor_get(pc.tensor, pc.dst->data(), 0, cn * sizeof(float));
    }
    impl_->captures.clear();
    impl_->roots.clear();

    bool consume_ok = true;
    if (consume) {
        ggml_backend_t output_backend = impl_->backend;
        if (need_sched && impl_->sched) {
            ggml_backend_t placed = ggml_backend_sched_get_tensor_backend(impl_->sched, output);
            if (placed) output_backend = placed;
        }
        consume_ok = (*consume)(output_backend, output);
    }

    if (out) {
        size_t n = (size_t)ggml_nelements(output);
        out->resize(n);
        ggml_backend_tensor_get(output, out->data(), 0, n * sizeof(float));
    }

    ggml_free(ctx);
    return consume_ok;
}

bool Backend::compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                      std::vector<float>& out, size_t graph_nodes) {
    return compute_impl(build, &out, nullptr, graph_nodes);
}

bool Backend::compute_device(
        const std::function<ggml_tensor*(ggml_context*)>& build,
        const std::function<bool(ggml_backend_t, const ggml_tensor*)>& consume,
        size_t graph_nodes) {
    if (!consume) return false;
    return compute_impl(build, nullptr, &consume, graph_nodes);
}

bool Backend::forward_capture(const std::function<ggml_tensor*(ggml_context*)>& build,
                              std::vector<float>& out, size_t graph_nodes) {
    // Same path as compute(); captures registered during build are honored.
    return compute(build, out, graph_nodes);
}

}  // namespace da
