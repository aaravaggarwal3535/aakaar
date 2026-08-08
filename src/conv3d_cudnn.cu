#include <cudnn.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdlib>
#include "tensor.h"
#include "cudnn_manager.h"   // brings in aakaar_cudnn_debug_enabled()
#include "allocator.h"

extern bool get_cudnn_tf32_enabled();  // shared flag from matmul_kernel.cu

static void check_cudnn3d(cudnnStatus_t st, const char* what) {
    if (st != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuDNN error in ") + what + ": " + cudnnGetErrorString(st));
}

struct Conv3dDescs {
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
};

struct Conv3dCacheEntry {
    Conv3dDescs descs;
    void* algo_data = nullptr;
    std::shared_ptr<Tensor> workspace;
};

// B,Cin,D,H,W,Cout,KD,KH,KW,SD,SH,SW,PD,PH,PW,DD,DH,DW,OD,OH,OW = 21 ints.
// Stack-allocated from the start — this codebase's Conv2d cache learned
// the hard way that std::vector keys heap-allocate on every single call.
using ConvKey3D = std::array<int, 21>;
struct ArrayIntHash3D {
    size_t operator()(const ConvKey3D& k) const {
        size_t h = k.size();
        for (int x : k) h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static std::unordered_map<ConvKey3D, Conv3dCacheEntry, ArrayIntHash3D> g_fwd3d_cache;
static std::unordered_map<ConvKey3D, Conv3dCacheEntry, ArrayIntHash3D> g_bwd3d_data_cache;
static std::unordered_map<ConvKey3D, Conv3dCacheEntry, ArrayIntHash3D> g_bwd3d_filter_cache;

// 2GB default cap (not 1GB — Conv2d's original 1GB cap was confirmed, via
// nsys trace, to silently exclude valid faster candidates at large shapes).
// Override via AAKAAR_CUDNN_SEARCH_WS_MB for unusually large volumetric
// models where even 2GB isn't enough headroom.
static std::shared_ptr<Tensor> get_search_workspace_3d() {
    CachingAllocator::get_instance().empty_cache();
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t cap_bytes = (size_t)2048ull * 1024 * 1024;
    if (const char* env = std::getenv("AAKAAR_CUDNN_SEARCH_WS_MB")) {
        long mb = std::strtol(env, nullptr, 10);
        if (mb > 0) cap_bytes = (size_t)mb * 1024 * 1024;
    }
    size_t bytes = std::min(free_bytes / 2, cap_bytes);
    int n_floats = (int)(bytes / sizeof(float));
    return std::make_shared<Tensor>(std::vector<int>{n_floats}, std::string("cuda"), DType::FLOAT32);
}

static std::shared_ptr<Tensor> get_workspace_3d(size_t ws_bytes) {
    if (ws_bytes == 0) return nullptr;
    int n_floats = (int)((ws_bytes + sizeof(float) - 1) / sizeof(float));
    return std::make_shared<Tensor>(std::vector<int>{n_floats}, std::string("cuda"), DType::FLOAT32);
}

static void set_tensor_desc_5d(cudnnTensorDescriptor_t desc, int N, int C, int D, int H, int W) {
    int dimA[5] = {N, C, D, H, W};
    int strideA[5];
    strideA[4] = 1;
    strideA[3] = W;
    strideA[2] = H * W;
    strideA[1] = D * H * W;
    strideA[0] = C * D * H * W;
    check_cudnn3d(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 5, dimA, strideA), "set tensor nd desc");
}

static Conv3dDescs make_descs_3d(int B, int Cin, int D, int H, int W, int Cout, int KD, int KH, int KW,
                                  int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW,
                                  int OD, int OH, int OW) {
    Conv3dDescs desc{};
    check_cudnn3d(cudnnCreateTensorDescriptor(&desc.xDesc), "create xDesc");
    check_cudnn3d(cudnnCreateTensorDescriptor(&desc.yDesc), "create yDesc");
    check_cudnn3d(cudnnCreateFilterDescriptor(&desc.wDesc), "create wDesc");
    check_cudnn3d(cudnnCreateConvolutionDescriptor(&desc.convDesc), "create convDesc");

    set_tensor_desc_5d(desc.xDesc, B, Cin, D, H, W);
    set_tensor_desc_5d(desc.yDesc, B, Cout, OD, OH, OW);

    int filterDimA[5] = {Cout, Cin, KD, KH, KW};
    check_cudnn3d(cudnnSetFilterNdDescriptor(desc.wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filterDimA),
                  "set filter nd desc");

    int padA[3] = {PD, PH, PW};
    int strideA[3] = {SD, SH, SW};
    int dilationA[3] = {DD, DH, DW};
    check_cudnn3d(cudnnSetConvolutionNdDescriptor(desc.convDesc, 3, padA, strideA, dilationA,
                                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "set conv nd desc");

    cudnnMathType_t math = get_cudnn_tf32_enabled() ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
    check_cudnn3d(cudnnSetConvolutionMathType(desc.convDesc, math), "set math type");
    return desc;
}

std::shared_ptr<Tensor> run_cudnn_conv3d_forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                  int SD, int SH, int SW, int PD, int PH, int PW,
                                                  int DD, int DH, int DW, int OD, int OH, int OW) {
    int B = x->shape[0], Cin = x->shape[1], D = x->shape[2], H = x->shape[3], W = x->shape[4];
    int Cout = w->shape[0], KD = w->shape[2], KH = w->shape[3], KW = w->shape[4];
    ConvKey3D key = {{B, Cin, D, H, W, Cout, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW}};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_fwd3d_cache.find(key);
    if (it == g_fwd3d_cache.end()) {
        Conv3dCacheEntry entry;
        entry.descs = make_descs_3d(B, Cin, D, H, W, Cout, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);

        auto y_probe = std::make_shared<Tensor>(std::vector<int>{B, Cout, OD, OH, OW}, std::string("cuda"));
        auto search_ws = get_search_workspace_3d();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionFwdAlgoPerf_t perf[kMaxCandidates];
        int returned = 0;
        check_cudnn3d(cudnnFindConvolutionForwardAlgorithmEx(
            handle, entry.descs.xDesc, x->fptr(), entry.descs.wDesc, w->fptr(),
            entry.descs.convDesc, entry.descs.yDesc, y_probe->fptr(),
            kMaxCandidates, &returned, perf, search_ws->data_ptr, search_ws_bytes), "find fwd3d algo");

        int best = -1;
        for (int i = 0; i < returned; ++i) {
            if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
            // FFT-based algorithms decompose one logical call into many kernel launches,
            // which cuDNN's own isolated-timing search doesn't capture well.
            if (perf[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT ||
                perf[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) continue;
            best = i;
            break;
        }
        if (best == -1) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
            }
        }

        if (best != -1) {
            int chosen = best;
            if (chosen != -1) {
                int best_small = -1;
                for (int i = 0; i < returned; ++i) {
                    if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
                    if (perf[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT ||
                        perf[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) continue;
                    // Widened to 1.5x (from 1.25x) specifically because cuDNN's
                    // one-shot search timing is noisy enough, run to run, that a
                    // tighter margin caused this exact tiebreak to flip between two
                    // candidates non-deterministically across separate process runs
                    // at the same shape -- confirmed directly: two candidates whose
                    // true costs are close enough that measurement noise alone moved
                    // one in and out of a 1.25x window across six back-to-back runs.
                    bool much_smaller_workspace = perf[i].memory < perf[best].memory / 10;
                    bool close_enough_time = perf[i].time <= perf[best].time * 1.5;
                    if (much_smaller_workspace && close_enough_time) {
                        // Among all qualifying candidates, deterministically prefer
                        // the smallest workspace, not just the first one encountered
                        // in the (fastest-first) sorted order -- removes order/noise
                        // dependence entirely for the final choice.
                        if (best_small == -1 || perf[i].memory < perf[best_small].memory) {
                            best_small = i;
                        }
                    }
                }
                if (best_small != -1) chosen = best_small;
            }
            best = chosen;
        }

        if (best == -1) throw std::runtime_error("conv3d_cudnn: no forward algorithm candidates succeeded.");

        if (aakaar_cudnn_debug_enabled()) {
            printf("[conv3d_cudnn debug] returned=%d candidates:\n", returned);
            for (int i = 0; i < returned; ++i)
                printf("  [%d] algo=%d status=%d time=%.4f memory=%zu\n", i, (int)perf[i].algo,
                       (int)perf[i].status, perf[i].time, perf[i].memory);
            printf("  chosen: algo=%d time=%.4f\n", (int)perf[best].algo, perf[best].time);
            fflush(stdout);
        }

        auto* algo_storage = new cudnnConvolutionFwdAlgo_t(perf[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn3d(cudnnGetConvolutionForwardWorkspaceSize(
            handle, entry.descs.xDesc, entry.descs.wDesc, entry.descs.convDesc,
            entry.descs.yDesc, perf[best].algo, &ws_size), "get fwd3d workspace size");
        entry.workspace = get_workspace_3d(ws_size);

        it = g_fwd3d_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionFwdAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto y = std::make_shared<Tensor>(std::vector<int>{B, Cout, OD, OH, OW}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    check_cudnn3d(cudnnConvolutionForward(
        handle, &alpha, entry.descs.xDesc, x->fptr(), entry.descs.wDesc, w->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.yDesc, y->fptr()), "conv3d forward");
    return y;
}

std::shared_ptr<Tensor> run_cudnn_conv3d_backward_data(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                                        int B, int Cin, int D, int H, int W,
                                                        int SD, int SH, int SW, int PD, int PH, int PW,
                                                        int DD, int DH, int DW, int OD, int OH, int OW) {
    int Cout = w->shape[0], KD = w->shape[2], KH = w->shape[3], KW = w->shape[4];
    ConvKey3D key = {{B, Cin, D, H, W, Cout, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW}};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_bwd3d_data_cache.find(key);
    if (it == g_bwd3d_data_cache.end()) {
        Conv3dCacheEntry entry;
        entry.descs = make_descs_3d(B, Cin, D, H, W, Cout, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);

        auto gx_probe = std::make_shared<Tensor>(std::vector<int>{B, Cin, D, H, W}, std::string("cuda"));
        auto search_ws = get_search_workspace_3d();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionBwdDataAlgoPerf_t perf[kMaxCandidates];
        int returned = 0;
        check_cudnn3d(cudnnFindConvolutionBackwardDataAlgorithmEx(
            handle, entry.descs.wDesc, w->fptr(), entry.descs.yDesc, grad_y->fptr(),
            entry.descs.convDesc, entry.descs.xDesc, gx_probe->fptr(),
            kMaxCandidates, &returned, perf, search_ws->data_ptr, search_ws_bytes), "find bwd3d-data algo");

        int best = -1;
        for (int i = 0; i < returned; ++i) {
            if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
            if (perf[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT ||
                perf[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING) continue;
            best = i;
            break;
        }
        if (best == -1) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
            }
        }
        if (best == -1) throw std::runtime_error("conv3d_cudnn: no backward-data algorithm candidates succeeded.");

        if (aakaar_cudnn_debug_enabled()) {
            printf("[conv3d_cudnn_bwd_data debug] returned=%d candidates:\n", returned);
            for (int i = 0; i < returned; ++i)
                printf("  [%d] algo=%d status=%d time=%.4f memory=%zu\n", i, (int)perf[i].algo,
                       (int)perf[i].status, perf[i].time, perf[i].memory);
            printf("  chosen: algo=%d time=%.4f\n", (int)perf[best].algo, perf[best].time);
            fflush(stdout);
        }

        auto* algo_storage = new cudnnConvolutionBwdDataAlgo_t(perf[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn3d(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, entry.descs.wDesc, entry.descs.yDesc, entry.descs.convDesc,
            entry.descs.xDesc, perf[best].algo, &ws_size), "get bwd3d-data workspace size");
        entry.workspace = get_workspace_3d(ws_size);

        it = g_bwd3d_data_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionBwdDataAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto grad_x = std::make_shared<Tensor>(std::vector<int>{B, Cin, D, H, W}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    check_cudnn3d(cudnnConvolutionBackwardData(
        handle, &alpha, entry.descs.wDesc, w->fptr(), entry.descs.yDesc, grad_y->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.xDesc, grad_x->fptr()), "conv3d backward data");
    return grad_x;
}

std::shared_ptr<Tensor> run_cudnn_conv3d_backward_filter(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                                          int Cout, int KD, int KH, int KW,
                                                          int SD, int SH, int SW, int PD, int PH, int PW,
                                                          int DD, int DH, int DW, int OD, int OH, int OW) {
    int B = x->shape[0], Cin = x->shape[1], D = x->shape[2], H = x->shape[3], W = x->shape[4];
    ConvKey3D key = {{B, Cin, D, H, W, Cout, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW}};

    auto handle = CudnnManager::get_instance().handle;
    auto it = g_bwd3d_filter_cache.find(key);
    if (it == g_bwd3d_filter_cache.end()) {
        Conv3dCacheEntry entry;
        entry.descs = make_descs_3d(B, Cin, D, H, W, Cout, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);

        auto gw_probe = std::make_shared<Tensor>(std::vector<int>{Cout, Cin, KD, KH, KW}, std::string("cuda"));
        auto search_ws = get_search_workspace_3d();
        size_t search_ws_bytes = (size_t)search_ws->size * sizeof(float);

        const int kMaxCandidates = 8;
        cudnnConvolutionBwdFilterAlgoPerf_t perf[kMaxCandidates];
        int returned = 0;
        check_cudnn3d(cudnnFindConvolutionBackwardFilterAlgorithmEx(
            handle, entry.descs.xDesc, x->fptr(), entry.descs.yDesc, grad_y->fptr(),
            entry.descs.convDesc, entry.descs.wDesc, gw_probe->fptr(),
            kMaxCandidates, &returned, perf, search_ws->data_ptr, search_ws_bytes), "find bwd3d-filter algo");

        int best = -1;
        for (int i = 0; i < returned; ++i) {
            if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
            if (perf[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT ||
                perf[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING) continue;
            best = i;
            break;
        }
        if (best == -1) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS) { best = i; break; }
            }
        }
        if (best == -1) throw std::runtime_error("conv3d_cudnn: no backward-filter algorithm candidates succeeded.");

        if (aakaar_cudnn_debug_enabled()) {
            printf("[conv3d_cudnn_bwd_filter debug] returned=%d candidates:\n", returned);
            for (int i = 0; i < returned; ++i)
                printf("  [%d] algo=%d status=%d time=%.4f memory=%zu\n", i, (int)perf[i].algo,
                       (int)perf[i].status, perf[i].time, perf[i].memory);
            printf("  chosen: algo=%d time=%.4f\n", (int)perf[best].algo, perf[best].time);
            fflush(stdout);
        }

        auto* algo_storage = new cudnnConvolutionBwdFilterAlgo_t(perf[best].algo);
        entry.algo_data = algo_storage;

        size_t ws_size = 0;
        check_cudnn3d(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle, entry.descs.xDesc, entry.descs.yDesc, entry.descs.convDesc,
            entry.descs.wDesc, perf[best].algo, &ws_size), "get bwd3d-filter workspace size");
        entry.workspace = get_workspace_3d(ws_size);

        it = g_bwd3d_filter_cache.emplace(key, std::move(entry)).first;
    }

    auto& entry = it->second;
    auto algo = *static_cast<cudnnConvolutionBwdFilterAlgo_t*>(entry.algo_data);
    void* ws_ptr = entry.workspace ? entry.workspace->data_ptr : nullptr;
    size_t ws_bytes = entry.workspace ? (size_t)entry.workspace->size * sizeof(float) : 0;

    auto grad_w = std::make_shared<Tensor>(std::vector<int>{Cout, Cin, KD, KH, KW}, std::string("cuda"));
    float alpha = 1.0f, beta = 0.0f;
    check_cudnn3d(cudnnConvolutionBackwardFilter(
        handle, &alpha, entry.descs.xDesc, x->fptr(), entry.descs.yDesc, grad_y->fptr(),
        entry.descs.convDesc, algo, ws_ptr, ws_bytes, &beta, entry.descs.wDesc, grad_w->fptr()), "conv3d backward filter");
    return grad_w;
}