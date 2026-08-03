#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "tensor.h"
#include "cuda_graph.h"

std::shared_ptr<GraphHandle> cuda_graph_begin_capture() {
    auto handle = std::make_shared<GraphHandle>();
    cudaError_t err = cudaStreamCreate(&handle->stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda_graph_begin_capture: cudaStreamCreate failed: ") + cudaGetErrorString(err));

    err = cudaStreamBeginCapture(handle->stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda_graph_begin_capture: cudaStreamBeginCapture failed: ") + cudaGetErrorString(err));

    handle->capture_started = true;
    return handle;
}

void cuda_graph_end_capture(std::shared_ptr<GraphHandle> handle) {
    if (!handle->stream)
        throw std::runtime_error("cuda_graph_end_capture: capture was never started (no stream).");

    cudaError_t err = cudaStreamEndCapture(handle->stream, &handle->graph);
    handle->capture_started = false;
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda_graph_end_capture: cudaStreamEndCapture failed: ") + cudaGetErrorString(err) +
                                  ". This usually means an operation inside the captured region did something "
                                  "capture-incompatible (e.g. allocated new memory, synchronized the stream, "
                                  "or used a stream other than the one passed to graph_capture()).");

    err = cudaGraphInstantiate(&handle->exec, handle->graph, nullptr, nullptr, 0);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda_graph_end_capture: cudaGraphInstantiate failed: ") + cudaGetErrorString(err));

    handle->captured = true;
}

void cuda_graph_replay(std::shared_ptr<GraphHandle> handle) {
    if (!handle->captured)
        throw std::runtime_error("cuda_graph_replay: this graph was never successfully captured. "
                                  "Call graph_capture() first and ensure end_capture() succeeded.");
    cudaError_t err = cudaGraphLaunch(handle->exec, handle->stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda_graph_replay: cudaGraphLaunch failed: ") + cudaGetErrorString(err));
}

void cuda_graph_replay_two(std::shared_ptr<GraphHandle> h1, std::shared_ptr<GraphHandle> h2) {
    if (!h1->captured || !h2->captured)
        throw std::runtime_error("cuda_graph_replay_two: one or both graphs were never successfully captured.");
    cudaError_t err = cudaGraphLaunch(h1->exec, h1->stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda_graph_replay_two: first launch failed: ") + cudaGetErrorString(err));
    err = cudaGraphLaunch(h2->exec, h2->stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda_graph_replay_two: second launch failed: ") + cudaGetErrorString(err));
}

void cuda_graph_replay_full_step(std::shared_ptr<GraphHandle> fwd,
                                  std::shared_ptr<GraphHandle> bwd_data,
                                  std::shared_ptr<GraphHandle> bwd_filter) {
    if (!fwd->captured || !bwd_data->captured || !bwd_filter->captured)
        throw std::runtime_error("cuda_graph_replay_full_step: one or more graphs were never captured.");
    cudaError_t err = cudaGraphLaunch(fwd->exec, fwd->stream);
    if (err != cudaSuccess) throw std::runtime_error(std::string("full_step fwd launch failed: ") + cudaGetErrorString(err));
    err = cudaGraphLaunch(bwd_data->exec, bwd_data->stream);
    if (err != cudaSuccess) throw std::runtime_error(std::string("full_step bwd_data launch failed: ") + cudaGetErrorString(err));
    err = cudaGraphLaunch(bwd_filter->exec, bwd_filter->stream);
    if (err != cudaSuccess) throw std::runtime_error(std::string("full_step bwd_filter launch failed: ") + cudaGetErrorString(err));
}

void cuda_graph_synchronize(std::shared_ptr<GraphHandle> handle) {
    cudaError_t err = cudaStreamSynchronize(handle->stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cuda_graph_synchronize failed: ") + cudaGetErrorString(err));
}