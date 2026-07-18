#include <cudnn.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include "tensor.h"
#include <unordered_map>

struct VecIntHash {
    size_t operator()(const std::vector<int>& v) const {
        size_t h = v.size();
        for (int x : v) h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Expose the global TF32 flag from matmul_kernel.cu
extern bool get_tf32_enabled();

class CudnnManager {
public:
    static CudnnManager& get_instance() {
        static CudnnManager instance;
        return instance;
    }
    cudnnHandle_t handle;
private:
    CudnnManager() {
        cudnnStatus_t st = cudnnCreate(&handle);
        if (st != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(
                "cudnnCreate failed — is libcudnn installed and does its version match your CUDA toolkit?");
    }
    ~CudnnManager() { cudnnDestroy(handle); }
};

static void check_cudnn(cudnnStatus_t st, const char* what) {
    if (st != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuDNN error in ") + what + ": " + cudnnGetErrorString(st));
}

#include <map>
#include <tuple>

struct Conv1dDescs {
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
};

// cuDNN's convolution algorithms are reliably implemented for 4D (NCHW)
// descriptors; the generic N-d path (nbDims=3) silently lacks algorithm
// support for many/most algos, surfacing as CUDNN_STATUS_NOT_SUPPORTED at
// the workspace-size query. torch's own Conv1d works around this exact gap
// by describing the op as a 2D convolution with a singleton height
// dimension — same fix applied here: (B, C, L) -> (B, C, 1, L).
static Conv1dDescs make_descs(int B, int C_in, int L_in, int C_out, int K,
                               int S, int P, int D, int L_out) {
    Conv1dDescs d{};
    check_cudnn(cudnnCreateTensorDescriptor(&d.xDesc), "create xDesc");
    check_cudnn(cudnnCreateTensorDescriptor(&d.yDesc), "create yDesc");
    check_cudnn(cudnnCreateFilterDescriptor(&d.wDesc), "create wDesc");
    check_cudnn(cudnnCreateConvolutionDescriptor(&d.convDesc), "create convDesc");

    check_cudnn(cudnnSetTensor4dDescriptor(
        d.xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C_in, 1, L_in), "set xDesc");

    check_cudnn(cudnnSetTensor4dDescriptor(
        d.yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C_out, 1, L_out), "set yDesc");

    check_cudnn(cudnnSetFilter4dDescriptor(
        d.wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, 1, K), "set wDesc");

    check_cudnn(cudnnSetConvolution2dDescriptor(
        d.convDesc, 0, P, 1, S, 1, D, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "set convDesc");
    // pad_h=0, pad_w=P, stride_h=1, stride_w=S, dilation_h=1, dilation_w=D
    // — height dim is always singleton/untouched, all real params on width.

    return d;
}

struct Conv1dCacheEntry {
    Conv1dDescs descs;
    void* algo_data = nullptr;      // opaque storage for whichever algo enum applies
    std::shared_ptr<Tensor> workspace;  // reused via CachingAllocator (float32 CUDA alloc)
};

// Keyed on every shape/param that affects descriptor or algo choice. Separate
// caches per op since forward/bwd-data/bwd-filter each pick their own algo.
static std::unordered_map<std::vector<int>, Conv1dCacheEntry, VecIntHash> g_fwd_cache;
static std::unordered_map<std::vector<int>, Conv1dCacheEntry, VecIntHash> g_bwd_data_cache;
static std::unordered_map<std::vector<int>, Conv1dCacheEntry, VecIntHash> g_bwd_filter_cache;

static std::shared_ptr<Tensor> get_workspace(size_t ws_bytes) {
    if (ws_bytes == 0) return nullptr;
    // Round up to float-count; CachingAllocator (used by all float32 CUDA
    // Tensor allocations) tracks in float units, so this reuses the same
    // pooled-allocation machinery instead of a raw cudaMalloc/cudaFree pair
    // every call — that round trip was the dominant cost at these sizes.
    int n_floats = (int)((ws_bytes + sizeof(float) - 1) / sizeof(float));
    return std::make_shared<Tensor>(std::vector<int>{n_floats}, std::string("cuda"), DType::FLOAT32);
}

// Scratch workspace used only during algorithm search (FindAlgorithmEx times
// real candidate kernels against this buffer). Allocated once via
// CachingAllocator and reused across every shape's search.
static std::shared_ptr<Tensor> get_search_workspace() {
    static std::shared_ptr<Tensor> ws;
    if (!ws) {
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        // Use up to 50% of currently-free memory, capped at 1GB — generous
        // enough that workspace-hungry algorithms (Winograd/FFT-tiled) aren't
        // excluded from the search purely due to an arbitrary fixed budget.
        size_t bytes = std::min(free_bytes / 2, (size_t)1024ull * 1024 * 1024);
        int n_floats = (int)(bytes / sizeof(float));
        ws = std::make_shared<Tensor>(std::vector<int>{n_floats}, std::string("cuda"), DType::FLOAT32);
    }
    return ws;
}

static void destroy_descs(Conv1dDescs& d) {
    cudnnDestroyTensorDescriptor(d.xDesc);
    cudnnDestroyTensorDescriptor(d.yDesc);
    cudnnDestroyFilterDescriptor(d.wDesc);
    cudnnDestroyConvolutionDescriptor(d.convDesc);
}

std::shared_ptr<Tensor> run_cudnn_conv1d_forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                   int stride, int padding, int dilation, int L_out) {
    int B = x->shape[0], C_in = x->shape[1], L_in = x->shape[2];
    int C_out = w->shape[0], K = w->shape[2];
    std::vector<int> key = {B, C_in, L_in, C_out, K, stride, padding, dilation, L_out};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_fwd_cache.find(key);
    if (it == g_fwd_cache.end()) {
        Conv1dCacheEntry entry;
        entry.descs = make_descs(B, C_in, L_in, C_out, K, stride, padding, dilation, L_out);

        // Apply TF32 math type preference
        extern bool get_cudnn_tf32_enabled();  // instead of get_tf32_enabled()

        // inside all three run_cudnn_conv1d_* cache-miss branches:
        cudnnMathType_t math = get_cudnn_tf32_enabled() ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
        check_cudnn(cudnnSetConvolutionMathType(entry.descs.convDesc, math), "set math type");

        // Empirical search (like torch's cudnn.benchmark=True) instead of a
        // static heuristic: times real candidate kernels on the actual
        // x/w tensors and picks the measured-fastest one, rather than
        // trusting a lookup-table guess. This is what closes the gap to
        // torch's real achieved speed rather than its worst-case default.
        auto y_probe = std::make_shared<Tensor>(std::vector<int>{B, C_out, L_out}, std::string("cuda"));
        auto search_ws = get_search_workspace();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxCandidates];
        int returned = 0;
        check_cudnn(cudnnFindConvolutionForwardAlgorithmEx(
            handle,
            entry.descs.xDesc, x->fptr(),
            entry.descs.wDesc, w->fptr(),
            entry.descs.convDesc,
            entry.descs.yDesc, y_probe->fptr(),
            kMaxCandidates, &returned, perf_results,
            search_ws->data_ptr, search_ws_bytes), "find fwd algo (Ex)");

        if (returned == 0)
            throw std::runtime_error("conv1d_cudnn: no forward algorithm candidates succeeded during search.");

        // perf_results is sorted fastest-first, but skip any that reported
        // failure (e.g. numerically unstable candidates get flagged, not
        // silently ranked) rather than blindly trusting index 0.
        int best = -1;
        for (int i = 0; i < returned; ++i) {
            if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
        }
        if (best == -1)
            throw std::runtime_error("conv1d_cudnn: all forward algorithm candidates reported failure status.");
        fprintf(stderr, "[conv1d_cudnn] shape=(%d,%d,%d)->(%d,%d,%d) K=%d chosen algo=%d time=%.4fms workspace=%zuMB\n",
            B, C_in, L_in, B, C_out, L_out, K, (int)perf_results[best].algo,
            perf_results[best].time, perf_results[best].memory / (1024*1024));
        fflush(stderr);

        auto* algo_storage = new cudnnConvolutionFwdAlgo_t(perf_results[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(
            handle, entry.descs.xDesc, entry.descs.wDesc, entry.descs.convDesc,
            entry.descs.yDesc, perf_results[best].algo, &ws_size), "get fwd workspace size");
        entry.workspace = get_workspace(ws_size);

        it = g_fwd_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionFwdAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto y = std::make_shared<Tensor>(std::vector<int>{B, C_out, L_out}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    check_cudnn(cudnnConvolutionForward(
        handle, &alpha, entry.descs.xDesc, x->fptr(), entry.descs.wDesc, w->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.yDesc, y->fptr()),
        "conv forward");
    return y;
}

std::shared_ptr<Tensor> run_cudnn_conv1d_backward_data(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                                         int B, int C_in, int L_in, int stride, int padding,
                                                         int dilation, int L_out) {
    int C_out = w->shape[0], K = w->shape[2];
    std::vector<int> key = {B, C_in, L_in, C_out, K, stride, padding, dilation, L_out};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_bwd_data_cache.find(key);
    if (it == g_bwd_data_cache.end()) {
        Conv1dCacheEntry entry;
        entry.descs = make_descs(B, C_in, L_in, C_out, K, stride, padding, dilation, L_out);

        // Apply TF32 math type preference
        cudnnMathType_t math = get_tf32_enabled() ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
        check_cudnn(cudnnSetConvolutionMathType(entry.descs.convDesc, math), "set math type");

        auto grad_x_probe = std::make_shared<Tensor>(std::vector<int>{B, C_in, L_in}, std::string("cuda"));
        auto search_ws = get_search_workspace();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionBwdDataAlgoPerf_t perf_results[kMaxCandidates];
        int returned = 0;
        check_cudnn(cudnnFindConvolutionBackwardDataAlgorithmEx(
            handle,
            entry.descs.wDesc, w->fptr(),
            entry.descs.yDesc, grad_y->fptr(),
            entry.descs.convDesc,
            entry.descs.xDesc, grad_x_probe->fptr(),
            kMaxCandidates, &returned, perf_results,
            search_ws->data_ptr, search_ws_bytes), "find bwd-data algo (Ex)");

        if (returned == 0)
            throw std::runtime_error("conv1d_cudnn: no backward-data algorithm candidates succeeded during search.");

        int best = -1;
        for (int i = 0; i < returned; ++i) {
            if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
        }
        if (best == -1)
            throw std::runtime_error("conv1d_cudnn: all backward-data algorithm candidates reported failure status.");

        auto* algo_storage = new cudnnConvolutionBwdDataAlgo_t(perf_results[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, entry.descs.wDesc, entry.descs.yDesc, entry.descs.convDesc,
            entry.descs.xDesc, perf_results[best].algo, &ws_size), "get bwd-data workspace size");
        entry.workspace = get_workspace(ws_size);

        it = g_bwd_data_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionBwdDataAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto grad_x = std::make_shared<Tensor>(std::vector<int>{B, C_in, L_in}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    check_cudnn(cudnnConvolutionBackwardData(
        handle, &alpha, entry.descs.wDesc, w->fptr(), entry.descs.yDesc, grad_y->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.xDesc, grad_x->fptr()),
        "conv backward data");
    return grad_x;
}

std::shared_ptr<Tensor> run_cudnn_conv1d_backward_filter(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                                           int C_out, int K, int stride, int padding,
                                                           int dilation, int L_out) {
    int B = x->shape[0], C_in = x->shape[1], L_in = x->shape[2];
    std::vector<int> key = {B, C_in, L_in, C_out, K, stride, padding, dilation, L_out};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_bwd_filter_cache.find(key);
    if (it == g_bwd_filter_cache.end()) {
        Conv1dCacheEntry entry;
        entry.descs = make_descs(B, C_in, L_in, C_out, K, stride, padding, dilation, L_out);

        // Apply TF32 math type preference
        cudnnMathType_t math = get_tf32_enabled() ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
        check_cudnn(cudnnSetConvolutionMathType(entry.descs.convDesc, math), "set math type");

        auto grad_w_probe = std::make_shared<Tensor>(std::vector<int>{C_out, C_in, K}, std::string("cuda"));
        auto search_ws = get_search_workspace();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionBwdFilterAlgoPerf_t perf_results[kMaxCandidates];
        int returned = 0;
        check_cudnn(cudnnFindConvolutionBackwardFilterAlgorithmEx(
            handle,
            entry.descs.xDesc, x->fptr(),
            entry.descs.yDesc, grad_y->fptr(),
            entry.descs.convDesc,
            entry.descs.wDesc, grad_w_probe->fptr(),
            kMaxCandidates, &returned, perf_results,
            search_ws->data_ptr, search_ws_bytes), "find bwd-filter algo (Ex)");

        if (returned == 0)
            throw std::runtime_error("conv1d_cudnn: no backward-filter algorithm candidates succeeded during search.");

        int best = -1;
        for (int i = 0; i < returned; ++i) {
            if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
        }
        if (best == -1)
            throw std::runtime_error("conv1d_cudnn: all backward-filter algorithm candidates reported failure status.");

        auto* algo_storage = new cudnnConvolutionBwdFilterAlgo_t(perf_results[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle, entry.descs.xDesc, entry.descs.yDesc, entry.descs.convDesc,
            entry.descs.wDesc, perf_results[best].algo, &ws_size), "get bwd-filter workspace size");
        entry.workspace = get_workspace(ws_size);

        it = g_bwd_filter_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionBwdFilterAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto grad_w = std::make_shared<Tensor>(std::vector<int>{C_out, C_in, K}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    check_cudnn(cudnnConvolutionBackwardFilter(
        handle, &alpha, entry.descs.xDesc, x->fptr(), entry.descs.yDesc, grad_y->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.wDesc, grad_w->fptr()),
        "conv backward filter");
    return grad_w;
}