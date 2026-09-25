#pragma once

namespace da {
// The graph builder has a few CPU/GPU-specialized implementation choices.
// Keep the mode thread-local and scope it to Backend::compute(), so independent
// CPU and accelerator Engine instances can safely execute concurrently.
void set_gpu_mode(bool on);
bool gpu_mode();

class ScopedComputeMode {
public:
    explicit ScopedComputeMode(bool gpu) : previous_(gpu_mode()) { set_gpu_mode(gpu); }
    ~ScopedComputeMode() { set_gpu_mode(previous_); }
    ScopedComputeMode(const ScopedComputeMode&) = delete;
    ScopedComputeMode& operator=(const ScopedComputeMode&) = delete;
private:
    bool previous_;
};
} // namespace da
