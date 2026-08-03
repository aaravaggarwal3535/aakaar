#pragma once
#include <cuda_runtime.h>
#include <memory>

struct GraphHandle {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStream_t stream = nullptr;
    bool captured = false;
    bool capture_started = false;  // true once begin_capture succeeds, false after end_capture (success OR abort)

    ~GraphHandle() {
        // If a capture was started but never properly ended (e.g. an
        // exception was thrown mid-capture), abort it here so the stream
        // isn't left in a permanently-broken "capturing" state that poisons
        // the whole CUDA context for the rest of the process.
        if (capture_started && !captured && stream) {
            cudaGraph_t dangling_graph = nullptr;
            cudaStreamEndCapture(stream, &dangling_graph);  // best-effort abort
            if (dangling_graph) cudaGraphDestroy(dangling_graph);
        }
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
    }
};

std::shared_ptr<GraphHandle> cuda_graph_begin_capture();
void cuda_graph_end_capture(std::shared_ptr<GraphHandle> handle);
void cuda_graph_replay(std::shared_ptr<GraphHandle> handle);
void cuda_graph_replay_two(std::shared_ptr<GraphHandle> h1, std::shared_ptr<GraphHandle> h2);
void cuda_graph_replay_full_step(std::shared_ptr<GraphHandle> fwd,
                                  std::shared_ptr<GraphHandle> bwd_data,
                                  std::shared_ptr<GraphHandle> bwd_filter);
void cuda_graph_synchronize(std::shared_ptr<GraphHandle> handle);