#include "cudnn_manager.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <array>
#include "tensor.h"
#include "allocator.h"

extern bool get_cudnn_tf32_enabled();  // shared flag from conv_cudnn.cu

static void check_cudnn2d(cudnnStatus_t st, const char* what) {
    if (st != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuDNN error in ") + what + ": " + cudnnGetErrorString(st));
}

struct Conv2dDescs {
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
};

struct Conv2dCacheEntry {
    Conv2dDescs descs;
    void* algo_data = nullptr;
    std::shared_ptr<Tensor> workspace;
};

// Fixed-size, stack-allocated cache keys — the shape/param tuple is always
// exactly N ints, so a std::vector (heap-allocating on every single
// forward/backward call, cache hit or miss) was pure unnecessary overhead.
using ConvKeyFwd2D = std::array<int, 15>;
using ConvKeyBwd2D = std::array<int, 16>;  // +1 for require_capturable

template <size_t N>
struct ArrayIntHash2D {
    size_t operator()(const std::array<int, N>& k) const {
        size_t h = N;
        for (int x : k) h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static std::unordered_map<ConvKeyFwd2D, Conv2dCacheEntry, ArrayIntHash2D<15>> g_fwd2d_cache;
static std::unordered_map<ConvKeyBwd2D, Conv2dCacheEntry, ArrayIntHash2D<16>> g_bwd2d_data_cache;
static std::unordered_map<ConvKeyBwd2D, Conv2dCacheEntry, ArrayIntHash2D<16>> g_bwd2d_filter_cache;

static std::shared_ptr<Tensor> get_search_workspace_2d() {
    CachingAllocator::get_instance().empty_cache();

    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    // Default search budget: up to half of currently-free memory, capped
    // at 2GB (not the previous 1GB) — 1GB was silently excluding candidate
    // algorithms at larger shapes (confirmed: cuDNN reports
    // CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE for any candidate whose
    // workspace requirement exceeds what this search buffer provides,
    // rather than a clearer "insufficient workspace" status). Override via
    // AAKAAR_CUDNN_SEARCH_WS_MB for models with unusually large conv
    // layers where even 2GB isn't enough headroom.
    size_t cap_bytes = (size_t)2048ull * 1024 * 1024;
    if (const char* env = std::getenv("AAKAAR_CUDNN_SEARCH_WS_MB")) {
        long mb = std::strtol(env, nullptr, 10);
        if (mb > 0) cap_bytes = (size_t)mb * 1024 * 1024;
    }
    size_t bytes = std::min(free_bytes / 2, cap_bytes);

    int n_floats = (int)(bytes / sizeof(float));
    return std::make_shared<Tensor>(std::vector<int>{n_floats}, std::string("cuda"), DType::FLOAT32);
}

static std::shared_ptr<Tensor> get_workspace_2d(size_t ws_bytes) {
    if (ws_bytes == 0) return nullptr;
    int n_floats = (int)((ws_bytes + sizeof(float) - 1) / sizeof(float));
    return std::make_shared<Tensor>(std::vector<int>{n_floats}, std::string("cuda"), DType::FLOAT32);
}

static Conv2dDescs make_descs_2d(int B, int C_in, int H, int W, int C_out, int KH, int KW,
                                  int SH, int SW, int PH, int PW, int DH, int DW, int OH, int OW) {
    Conv2dDescs d{};
    check_cudnn2d(cudnnCreateTensorDescriptor(&d.xDesc), "create xDesc");
    check_cudnn2d(cudnnCreateTensorDescriptor(&d.yDesc), "create yDesc");
    check_cudnn2d(cudnnCreateFilterDescriptor(&d.wDesc), "create wDesc");
    check_cudnn2d(cudnnCreateConvolutionDescriptor(&d.convDesc), "create convDesc");

    check_cudnn2d(cudnnSetTensor4dDescriptor(d.xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C_in, H, W), "set xDesc");
    check_cudnn2d(cudnnSetTensor4dDescriptor(d.yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C_out, OH, OW), "set yDesc");
    check_cudnn2d(cudnnSetFilter4dDescriptor(d.wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, KH, KW), "set wDesc");
    check_cudnn2d(cudnnSetConvolution2dDescriptor(d.convDesc, PH, PW, SH, SW, DH, DW,
                                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "set convDesc");

    cudnnMathType_t math = get_cudnn_tf32_enabled() ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
    check_cudnn2d(cudnnSetConvolutionMathType(d.convDesc, math), "set math type");
    return d;
}

std::shared_ptr<Tensor> run_cudnn_conv2d_forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                 int SH, int SW, int PH, int PW, int DH, int DW,
                                                 int OH, int OW) {
    int B = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    int C_out = w->shape[0], KH = w->shape[2], KW = w->shape[3];
    ConvKeyFwd2D key = {{B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW}};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_fwd2d_cache.find(key);
    
    if (it == g_fwd2d_cache.end()) {
        Conv2dCacheEntry entry;
        entry.descs = make_descs_2d(B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);

        auto y_probe = std::make_shared<Tensor>(std::vector<int>{B, C_out, OH, OW}, std::string("cuda"));
        auto search_ws = get_search_workspace_2d();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxCandidates];

        // Lambda to try finding an algorithm with a specific math type
        auto try_find_algo = [&](bool use_tf32) -> int {
            cudnnMathType_t math = use_tf32 ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
            check_cudnn2d(cudnnSetConvolutionMathType(entry.descs.convDesc, math), "set math type");

            int returned = 0;
            check_cudnn2d(cudnnFindConvolutionForwardAlgorithmEx(
                handle, entry.descs.xDesc, x->fptr(), entry.descs.wDesc, w->fptr(),
                entry.descs.convDesc, entry.descs.yDesc, y_probe->fptr(),
                kMaxCandidates, &returned, perf_results, search_ws->data_ptr, search_ws_bytes), "find fwd algo");

            // Defensive: cuDNN's internal algorithm-timing mechanism for
            // FindConvolutionForwardAlgorithmEx has been observed (on this
            // cuDNN 9.2 build) to leave the stream in an unexpected capture-related
            // state after the search completes, causing the NEXT unrelated cudaMalloc
            // to fail with "operation not permitted when stream is capturing" even
            // though no graph capture was ever explicitly requested by this code.
            // Query and defensively end any stray capture before proceeding.
            cudaStreamCaptureStatus capture_status;
            cudaStreamIsCapturing(0, &capture_status);  // 0 = default/legacy stream
            if (capture_status == cudaStreamCaptureStatusActive) {
                cudaGraph_t stray_graph = nullptr;
                cudaStreamEndCapture(0, &stray_graph);
                if (stray_graph) cudaGraphDestroy(stray_graph);
            }

            if (aakaar_cudnn_debug_enabled()) {
                printf("[conv2d_cudnn debug] TF32=%d returned=%d candidates:\n", (int)use_tf32, returned);
                for (int i = 0; i < returned; ++i) {
                    printf("  [%d] algo=%d status=%d (%s) time=%.4f memory=%zu\n",
                           i, (int)perf_results[i].algo, (int)perf_results[i].status,
                           cudnnGetErrorString(perf_results[i].status),
                           perf_results[i].time, perf_results[i].memory);
                }
                fflush(stdout);
            }

            int best = -1;
            for (int i = 0; i < returned; ++i) {
                if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
                // See conv_cudnn.cu's identical guard for the full rationale: FFT-based
                // algorithms decompose one logical call into many kernel launches,
                // which cuDNN's own isolated-timing search doesn't capture. Confirmed
                // via nsys for Conv1d/ConvTranspose1d at a large shape; applied here
                // preventively since the mechanism is identical.
                if (perf_results[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT ||
                    perf_results[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) continue;
                best = i;
                break;
            }
            if (best == -1) {
                for (int i = 0; i < returned; ++i) {
                    if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
                }
            }

            if (best != -1) {
                int chosen = best;
                if (chosen != -1) {
                    int best_small = -1;
                    for (int i = 0; i < returned; ++i) {
                        if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
                        if (perf_results[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT ||
                            perf_results[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) continue;
                        // Widened to 1.5x (from 1.25x) specifically because cuDNN's
                        // one-shot search timing is noisy enough, run to run, that a
                        // tighter margin caused this exact tiebreak to flip between two
                        // candidates non-deterministically across separate process runs
                        // at the same shape -- confirmed directly: two candidates whose
                        // true costs are close enough that measurement noise alone moved
                        // one in and out of a 1.25x window across six back-to-back runs.
                        bool much_smaller_workspace = perf_results[i].memory < perf_results[best].memory / 10;
                        bool close_enough_time = perf_results[i].time <= perf_results[best].time * 1.5;
                        if (much_smaller_workspace && close_enough_time) {
                            // Among all qualifying candidates, deterministically prefer
                            // the smallest workspace, not just the first one encountered
                            // in the (fastest-first) sorted order -- removes order/noise
                            // dependence entirely for the final choice.
                            if (best_small == -1 || perf_results[i].memory < perf_results[best_small].memory) {
                                best_small = i;
                            }
                        }
                    }
                    if (best_small != -1) chosen = best_small;
                }
                best = chosen;
            }

            return best;
        };

        // 1. Try with the requested TF32 setting
        int best = try_find_algo(get_cudnn_tf32_enabled());
        
        // 2. If it failed and TF32 was requested, fallback to standard FP32
        if (best == -1 && get_cudnn_tf32_enabled()) {
            best = try_find_algo(false);
        }
        
        // 3. If both failed, we cannot proceed
        if (best == -1) {
            throw std::runtime_error("conv2d_cudnn: all forward algorithm candidates reported failure status "
                                     "(tried both TF32 and standard FP32 math modes).");
        }

        auto* algo_storage = new cudnnConvolutionFwdAlgo_t(perf_results[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn2d(cudnnGetConvolutionForwardWorkspaceSize(
            handle, entry.descs.xDesc, entry.descs.wDesc, entry.descs.convDesc,
            entry.descs.yDesc, perf_results[best].algo, &ws_size), "get fwd workspace size");
        entry.workspace = get_workspace_2d(ws_size);

        it = g_fwd2d_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionFwdAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto y = std::make_shared<Tensor>(std::vector<int>{B, C_out, OH, OW}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    
    check_cudnn2d(cudnnConvolutionForward(
        handle, &alpha, entry.descs.xDesc, x->fptr(), entry.descs.wDesc, w->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.yDesc, y->fptr()), "conv forward");
        
    return y;
}

std::shared_ptr<Tensor> run_cudnn_conv2d_backward_data(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                                       int B, int C_in, int H, int W, int SH, int SW,
                                                       int PH, int PW, int DH, int DW, int OH, int OW,
                                                       bool require_capturable) {
    int C_out = w->shape[0], KH = w->shape[2], KW = w->shape[3];
    ConvKeyBwd2D key = {{B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW, (int)require_capturable}};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_bwd2d_data_cache.find(key);
    if (it == g_bwd2d_data_cache.end()) {
        Conv2dCacheEntry entry;
        entry.descs = make_descs_2d(B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);

        auto gx_probe = std::make_shared<Tensor>(std::vector<int>{B, C_in, H, W}, std::string("cuda"));
        auto search_ws = get_search_workspace_2d();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionBwdDataAlgoPerf_t perf_results[kMaxCandidates];
        int returned = 0;
        check_cudnn2d(cudnnFindConvolutionBackwardDataAlgorithmEx(
            handle, entry.descs.wDesc, w->fptr(), entry.descs.yDesc, grad_y->fptr(),
            entry.descs.convDesc, entry.descs.xDesc, gx_probe->fptr(),
            kMaxCandidates, &returned, perf_results, search_ws->data_ptr, search_ws_bytes), "find bwd-data algo");

        cudaStreamCaptureStatus capture_status;
        cudaStreamIsCapturing(0, &capture_status);
        if (capture_status == cudaStreamCaptureStatusActive) {
            cudaGraph_t stray_graph = nullptr;
            cudaStreamEndCapture(0, &stray_graph);
            if (stray_graph) cudaGraphDestroy(stray_graph);
        }

        int best = -1;
        if (require_capturable) {
            // Only pay the per-candidate capture-probe cost when the caller actually intends
            // to record this op into a CUDA graph
            for (int i = 0; i < returned; ++i) {
                if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;

                size_t probe_ws_size = 0;
                cudnnStatus_t ws_st = cudnnGetConvolutionBackwardDataWorkspaceSize(
                    handle, entry.descs.wDesc, entry.descs.yDesc, entry.descs.convDesc,
                    entry.descs.xDesc, perf_results[i].algo, &probe_ws_size);
                if (ws_st != CUDNN_STATUS_SUCCESS) continue;

                auto probe_ws = get_workspace_2d(probe_ws_size);
                void* probe_ws_ptr = probe_ws ? probe_ws->data_ptr : nullptr;

                cudaStream_t probe_stream;
                cudaStreamCreateWithFlags(&probe_stream, cudaStreamNonBlocking);
                cudnnSetStream(handle, probe_stream);
                cudaGraph_t probe_graph = nullptr;
                cudaError_t cap_err = cudaStreamBeginCapture(probe_stream, cudaStreamCaptureModeThreadLocal);

                bool capture_ok = false;
                if (cap_err == cudaSuccess) {
                    float alpha = 1.0f, beta = 0.0f;
                    cudnnStatus_t conv_st = cudnnConvolutionBackwardData(
                        handle, &alpha, entry.descs.wDesc, w->fptr(), entry.descs.yDesc, gx_probe->fptr(),
                        entry.descs.convDesc, perf_results[i].algo, probe_ws_ptr, probe_ws_size,
                        &beta, entry.descs.xDesc, gx_probe->fptr());
                    capture_ok = (conv_st == CUDNN_STATUS_SUCCESS);
                }
                if (cap_err == cudaSuccess) {
                    cudaStreamEndCapture(probe_stream, &probe_graph);
                    if (probe_graph) cudaGraphDestroy(probe_graph);
                }
                cudnnSetStream(handle, 0);
                cudaStreamDestroy(probe_stream);

                if (capture_ok) { best = i; break; }
            }
            if (best == -1)
                throw std::runtime_error("conv2d_cudnn: no backward-data algorithm is both fast AND CUDA-graph-capturable for this shape.");
        } else {
            for (int i = 0; i < returned; ++i) {
                if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
                if (perf_results[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT ||
                    perf_results[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING) continue;
                best = i;
                break;
            }
            if (best == -1) {
                for (int i = 0; i < returned; ++i) {
                    if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
                }
            }

            if (best != -1) {
                int chosen = best;
                if (chosen != -1) {
                    int best_small = -1;
                    for (int i = 0; i < returned; ++i) {
                        if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
                        if (perf_results[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT ||
                            perf_results[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING) continue;
                        // Widened to 1.5x (from 1.25x) specifically because cuDNN's
                        // one-shot search timing is noisy enough, run to run, that a
                        // tighter margin caused this exact tiebreak to flip between two
                        // candidates non-deterministically across separate process runs
                        // at the same shape -- confirmed directly: two candidates whose
                        // true costs are close enough that measurement noise alone moved
                        // one in and out of a 1.25x window across six back-to-back runs.
                        bool much_smaller_workspace = perf_results[i].memory < perf_results[best].memory / 10;
                        bool close_enough_time = perf_results[i].time <= perf_results[best].time * 1.5;
                        if (much_smaller_workspace && close_enough_time) {
                            // Among all qualifying candidates, deterministically prefer
                            // the smallest workspace, not just the first one encountered
                            // in the (fastest-first) sorted order -- removes order/noise
                            // dependence entirely for the final choice.
                            if (best_small == -1 || perf_results[i].memory < perf_results[best_small].memory) {
                                best_small = i;
                            }
                        }
                    }
                    if (best_small != -1) chosen = best_small;
                }
                best = chosen;
            }

            if (best == -1)
                throw std::runtime_error("conv2d_cudnn: all backward-data algorithm candidates reported failure status.");

            if (aakaar_cudnn_debug_enabled()) {
                printf("[conv2d_cudnn_bwd_data debug] returned=%d candidates:\n", returned);
                for (int i = 0; i < returned; ++i) {
                    printf("  [%d] algo=%d status=%d (%s) time=%.4f memory=%zu\n",
                        i, (int)perf_results[i].algo, (int)perf_results[i].status,
                        cudnnGetErrorString(perf_results[i].status),
                        perf_results[i].time, perf_results[i].memory);
                }
                printf("  chosen: algo=%d time=%.4f\n", (int)perf_results[best].algo, perf_results[best].time);
                fflush(stdout);
            }
        }

        auto* algo_storage = new cudnnConvolutionBwdDataAlgo_t(perf_results[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn2d(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, entry.descs.wDesc, entry.descs.yDesc, entry.descs.convDesc,
            entry.descs.xDesc, perf_results[best].algo, &ws_size), "get bwd-data workspace size");
        entry.workspace = get_workspace_2d(ws_size);

        it = g_bwd2d_data_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionBwdDataAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto grad_x = std::make_shared<Tensor>(std::vector<int>{B, C_in, H, W}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    check_cudnn2d(cudnnConvolutionBackwardData(
        handle, &alpha, entry.descs.wDesc, w->fptr(), entry.descs.yDesc, grad_y->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.xDesc, grad_x->fptr()), "conv backward data");
    return grad_x;
}

std::shared_ptr<Tensor> run_cudnn_conv2d_backward_filter(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                                         int C_out, int KH, int KW, int SH, int SW,
                                                         int PH, int PW, int DH, int DW, int OH, int OW,
                                                         bool require_capturable) {
    int B = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    ConvKeyBwd2D key = {{B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW, (int)require_capturable}};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_bwd2d_filter_cache.find(key);
    if (it == g_bwd2d_filter_cache.end()) {
        Conv2dCacheEntry entry;
        entry.descs = make_descs_2d(B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);

        auto gw_probe = std::make_shared<Tensor>(std::vector<int>{C_out, C_in, KH, KW}, std::string("cuda"));
        auto search_ws = get_search_workspace_2d();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionBwdFilterAlgoPerf_t perf_results[kMaxCandidates];
        int returned = 0;
        check_cudnn2d(cudnnFindConvolutionBackwardFilterAlgorithmEx(
            handle, entry.descs.xDesc, x->fptr(), entry.descs.yDesc, grad_y->fptr(),
            entry.descs.convDesc, entry.descs.wDesc, gw_probe->fptr(),
            kMaxCandidates, &returned, perf_results, search_ws->data_ptr, search_ws_bytes), "find bwd-filter algo");

        cudaStreamCaptureStatus capture_status;
        cudaStreamIsCapturing(0, &capture_status); 
        if (capture_status == cudaStreamCaptureStatusActive) {
            cudaGraph_t stray_graph = nullptr;
            cudaStreamEndCapture(0, &stray_graph);
            if (stray_graph) cudaGraphDestroy(stray_graph);
        }

        int best = -1;
        if (require_capturable) {
            for (int i = 0; i < returned; ++i) {
                if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;

                size_t probe_ws_size = 0;
                cudnnStatus_t ws_st = cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    handle, entry.descs.xDesc, entry.descs.yDesc, entry.descs.convDesc,
                    entry.descs.wDesc, perf_results[i].algo, &probe_ws_size);
                if (ws_st != CUDNN_STATUS_SUCCESS) continue;

                auto probe_ws = get_workspace_2d(probe_ws_size);
                void* probe_ws_ptr = probe_ws ? probe_ws->data_ptr : nullptr;

                cudaStream_t probe_stream;
                cudaStreamCreateWithFlags(&probe_stream, cudaStreamNonBlocking);
                cudnnSetStream(handle, probe_stream);
                cudaGraph_t probe_graph = nullptr;
                cudaError_t cap_err = cudaStreamBeginCapture(probe_stream, cudaStreamCaptureModeThreadLocal);

                bool capture_ok = false;
                if (cap_err == cudaSuccess) {
                    float alpha = 1.0f, beta = 0.0f;
                    cudnnStatus_t conv_st = cudnnConvolutionBackwardFilter(
                        handle, &alpha, entry.descs.xDesc, x->fptr(), entry.descs.yDesc, grad_y->fptr(),
                        entry.descs.convDesc, perf_results[i].algo, probe_ws_ptr, probe_ws_size,
                        &beta, entry.descs.wDesc, gw_probe->fptr());
                    capture_ok = (conv_st == CUDNN_STATUS_SUCCESS);
                }

                if (cap_err == cudaSuccess) {
                    cudaStreamEndCapture(probe_stream, &probe_graph);
                    if (probe_graph) cudaGraphDestroy(probe_graph);
                }
                cudnnSetStream(handle, 0);
                cudaStreamDestroy(probe_stream);

                if (capture_ok) { best = i; break; }
            }
            if (best == -1)
                throw std::runtime_error("conv2d_cudnn: no backward-filter algorithm is both fast AND CUDA-graph-capturable for this shape.");
        } else {
            for (int i = 0; i < returned; ++i) {
                if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
                if (perf_results[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT ||
                    perf_results[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING) continue;
                best = i;
                break;
            }
            if (best == -1) {
                for (int i = 0; i < returned; ++i) {
                    if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
                }
            }

            if (best != -1) {
                int chosen = best;
                if (chosen != -1) {
                    int best_small = -1;
                    for (int i = 0; i < returned; ++i) {
                        if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
                        if (perf_results[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT ||
                            perf_results[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING) continue;
                        // Widened to 1.5x (from 1.25x) specifically because cuDNN's
                        // one-shot search timing is noisy enough, run to run, that a
                        // tighter margin caused this exact tiebreak to flip between two
                        // candidates non-deterministically across separate process runs
                        // at the same shape -- confirmed directly: two candidates whose
                        // true costs are close enough that measurement noise alone moved
                        // one in and out of a 1.25x window across six back-to-back runs.
                        bool much_smaller_workspace = perf_results[i].memory < perf_results[best].memory / 10;
                        bool close_enough_time = perf_results[i].time <= perf_results[best].time * 1.5;
                        if (much_smaller_workspace && close_enough_time) {
                            // Among all qualifying candidates, deterministically prefer
                            // the smallest workspace, not just the first one encountered
                            // in the (fastest-first) sorted order -- removes order/noise
                            // dependence entirely for the final choice.
                            if (best_small == -1 || perf_results[i].memory < perf_results[best_small].memory) {
                                best_small = i;
                            }
                        }
                    }
                    if (best_small != -1) chosen = best_small;
                }
                best = chosen;
            }

            if (best == -1)
                throw std::runtime_error("conv2d_cudnn: all backward-data algorithm candidates reported failure status.");

            if (aakaar_cudnn_debug_enabled()) {
                printf("[conv2d_cudnn_bwd_filter debug] returned=%d candidates:\n", returned);
                for (int i = 0; i < returned; ++i) {
                    printf("  [%d] algo=%d status=%d (%s) time=%.4f memory=%zu\n",
                        i, (int)perf_results[i].algo, (int)perf_results[i].status,
                        cudnnGetErrorString(perf_results[i].status),
                        perf_results[i].time, perf_results[i].memory);
                }
                printf("  chosen: algo=%d time=%.4f\n", (int)perf_results[best].algo, perf_results[best].time);
                fflush(stdout);
            }
        }
        
        auto* algo_storage = new cudnnConvolutionBwdFilterAlgo_t(perf_results[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn2d(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle, entry.descs.xDesc, entry.descs.yDesc, entry.descs.convDesc,
            entry.descs.wDesc, perf_results[best].algo, &ws_size), "get bwd-filter workspace size");
        entry.workspace = get_workspace_2d(ws_size);

        it = g_bwd2d_filter_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionBwdFilterAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto grad_w = std::make_shared<Tensor>(std::vector<int>{C_out, C_in, KH, KW}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    check_cudnn2d(cudnnConvolutionBackwardFilter(
        handle, &alpha, entry.descs.xDesc, x->fptr(), entry.descs.yDesc, grad_y->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.wDesc, grad_w->fptr()), "conv backward filter");
    return grad_w;
}

void run_cudnn_conv2d_forward_into(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w, std::shared_ptr<Tensor> y,
                                    int SH, int SW, int PH, int PW, int DH, int DW) {
    int B = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    int C_out = w->shape[0], KH = w->shape[2], KW = w->shape[3];
    int OH = y->shape[2], OW = y->shape[3];
    ConvKeyFwd2D key = {{B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW}};

    auto it = g_fwd2d_cache.find(key);
    if (it == g_fwd2d_cache.end())
        throw std::runtime_error("run_cudnn_conv2d_forward_into: algorithm cache not warm for this shape. "
            "Call run_cudnn_conv2d_forward() once, uncaptured, first.");

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionFwdAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    float alpha = 1.0f, beta = 0.0f;
    check_cudnn2d(cudnnConvolutionForward(
        CudnnManager::get_instance().handle, &alpha, entry.descs.xDesc, x->fptr(),
        entry.descs.wDesc, w->fptr(), entry.descs.convDesc, algo, ws_ptr, ws_bytes,
        &beta, entry.descs.yDesc, y->fptr()), "conv forward (into)");
}

void run_cudnn_conv2d_backward_data_into(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                          std::shared_ptr<Tensor> grad_x, int SH, int SW, int PH, int PW, int DH, int DW) {
    int B = grad_x->shape[0], C_in = grad_x->shape[1], H = grad_x->shape[2], W = grad_x->shape[3];
    int C_out = w->shape[0], KH = w->shape[2], KW = w->shape[3];
    int OH = grad_y->shape[2], OW = grad_y->shape[3];
    // Hardcoded true (not a parameter here): this function is only ever
    // called from inside CUDA graph capture, which by definition requires
    // the capturable-validated algorithm — must match the key that
    // CapturedConv2d's warmup phase stored under require_capturable=true.
    ConvKeyBwd2D key = {{B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW, 1}};

    auto it = g_bwd2d_data_cache.find(key);
    if (it == g_bwd2d_data_cache.end())
        throw std::runtime_error("run_cudnn_conv2d_backward_data_into: algorithm cache not warm for this shape. "
            "Run an uncaptured backward pass with require_capturable=true first.");

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionBwdDataAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    float alpha = 1.0f, beta = 0.0f;
    check_cudnn2d(cudnnConvolutionBackwardData(
        CudnnManager::get_instance().handle, &alpha, entry.descs.wDesc, w->fptr(),
        entry.descs.yDesc, grad_y->fptr(), entry.descs.convDesc, algo, ws_ptr, ws_bytes,
        &beta, entry.descs.xDesc, grad_x->fptr()), "conv backward data (into)");
}

void run_cudnn_conv2d_backward_filter_into(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                            std::shared_ptr<Tensor> grad_w, int SH, int SW, int PH, int PW, int DH, int DW) {
    int B = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    int C_out = grad_w->shape[0], KH = grad_w->shape[2], KW = grad_w->shape[3];
    int OH = grad_y->shape[2], OW = grad_y->shape[3];
    ConvKeyBwd2D key = {{B, C_in, H, W, C_out, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW, 1}};

    auto it = g_bwd2d_filter_cache.find(key);
    if (it == g_bwd2d_filter_cache.end())
        throw std::runtime_error("run_cudnn_conv2d_backward_filter_into: algorithm cache not warm for this shape. "
            "Run an uncaptured backward pass with require_capturable=true first.");

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionBwdFilterAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    float alpha = 1.0f, beta = 0.0f;
    check_cudnn2d(cudnnConvolutionBackwardFilter(
        CudnnManager::get_instance().handle, &alpha, entry.descs.xDesc, x->fptr(),
        entry.descs.yDesc, grad_y->fptr(), entry.descs.convDesc, algo, ws_ptr, ws_bytes,
        &beta, entry.descs.wDesc, grad_w->fptr()), "conv backward filter (into)");
}