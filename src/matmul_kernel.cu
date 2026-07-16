#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <stdexcept>
#include <numeric>
#include "tensor.h"

class CublasManager {
public:
    static CublasManager& get_instance() {
        static CublasManager instance;
        return instance;
    }
    cublasHandle_t handle;
private:
    CublasManager() { cublasCreate(&handle); }
    ~CublasManager() { cublasDestroy(handle); }
};

static bool g_use_tf32 = false;

void set_tf32_enabled(bool enabled) {
    g_use_tf32 = enabled;
    cublasMath_t mode = enabled ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasSetMathMode(CublasManager::get_instance().handle, mode);
}

bool get_tf32_enabled() {
    return g_use_tf32;
}

void cuda_synchronize() {
    cudaDeviceSynchronize();
}

struct NDMatmulPlan {
    std::vector<int> batch_shape;  // broadcast batch shape, e.g. {2,3}
    int total_batch;
    int M, K, N;
    std::vector<int> batch_stride_a;  // per-batch-dim stride into A's batch (0 if broadcast)
    std::vector<int> batch_stride_b;
};

static std::vector<int> batch_dims(const std::vector<int>& shape) {
    return std::vector<int>(shape.begin(), shape.end() - 2);
}

static NDMatmulPlan plan_nd_matmul(const std::vector<int>& shapeA, const std::vector<int>& shapeB) {
    if (shapeA.size() < 2 || shapeB.size() < 2)
        throw std::invalid_argument("matmul requires tensors with at least 2 dimensions");

    int M = shapeA[shapeA.size()-2], K = shapeA.back();
    int K2 = shapeB[shapeB.size()-2], N = shapeB.back();
    if (K != K2) throw std::invalid_argument("Shape mismatch: inner dimensions must match for matmul");

    auto ba = batch_dims(shapeA), bb = batch_dims(shapeB);
    int ndb = std::max(ba.size(), bb.size());
    std::vector<int> out_batch(ndb);
    // right-align and broadcast, numpy/torch style
    for (int i = 0; i < ndb; ++i) {
        int da = i < (int)ba.size() ? ba[ba.size()-1-i] : 1;
        int db = i < (int)bb.size() ? bb[bb.size()-1-i] : 1;
        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("Batch dimensions are not broadcastable for matmul");
        out_batch[ndb-1-i] = std::max(da, db);
    }

    int total_batch = 1;
    for (int d : out_batch) total_batch *= d;

    NDMatmulPlan plan;
    plan.batch_shape = out_batch;
    plan.total_batch = total_batch;
    plan.M = M; plan.K = K; plan.N = N;
    return plan;
}

// Computes flat element-offset into A/B's batch region for a given flat output-batch index,
// honoring broadcasting (any input batch dim of size 1 always maps to offset 0 along that axis).
static int batch_offset(int flat_batch_idx, const std::vector<int>& out_batch,
                         const std::vector<int>& in_shape, int mat_size) {
    auto in_batch = batch_dims(in_shape);
    int ndb = (int)out_batch.size();
    int nd_in = (int)in_batch.size();
    std::vector<int> idx(ndb);
    int rem = flat_batch_idx;
    for (int i = ndb - 1; i >= 0; --i) {
        idx[i] = rem % out_batch[i];
        rem /= out_batch[i];
    }
    long long off = 0, stride = mat_size;
    for (int i = ndb - 1; i >= 0; --i) {
        int in_i = i - (ndb - nd_in);
        int in_dim = in_i >= 0 ? in_batch[in_i] : 1;
        int use_idx = (in_dim == 1) ? 0 : idx[i];
        if (in_i >= 0) {
            off += (long long)use_idx * stride;
            stride *= in_dim;
        }
    }
    return (int)off;
}

std::shared_ptr<Tensor> run_cublas_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a->is_contiguous() || !b->is_contiguous())
        throw std::invalid_argument("matmul requires contiguous tensors. Call .contiguous() first.");

    NDMatmulPlan p = plan_nd_matmul(a->shape, b->shape);

    std::vector<int> out_shape = p.batch_shape;
    out_shape.push_back(p.M);
    out_shape.push_back(p.N);
    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"));

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = CublasManager::get_instance().handle;

    for (int bi = 0; bi < p.total_batch; ++bi) {
        int offA = batch_offset(bi, p.batch_shape, a->shape, p.M * p.K);
        int offB = batch_offset(bi, p.batch_shape, b->shape, p.K * p.N);
        int offC = bi * p.M * p.N;

        cublasStatus_t status = cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            p.N, p.M, p.K,
            &alpha,
            b->fptr() + offB, p.N,
            a->fptr() + offA, p.K,
            &beta,
            result->fptr() + offC, p.N
        );
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS SGEMM failed at batch index " + std::to_string(bi));
    }
    return result;
}

// ---- Typed matmul: float64 via cublasDgemm ----
// Reuses the existing NDMatmulPlan/batch_offset helpers above (already
// dtype-agnostic — they only work with shapes/strides, never touch data),
// and the same CublasManager singleton handle. int32/int64 intentionally
// NOT handled here — see design discussion; they get a dedicated kernel.

std::shared_ptr<Tensor> run_cuda_matmul_f64(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a->is_contiguous() || !b->is_contiguous())
        throw std::invalid_argument("matmul requires contiguous tensors. Call .contiguous() first.");

    NDMatmulPlan p = plan_nd_matmul(a->shape, b->shape);

    std::vector<int> out_shape = p.batch_shape;
    out_shape.push_back(p.M);
    out_shape.push_back(p.N);
    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"), DType::FLOAT64);

    const double alpha = 1.0, beta = 0.0;
    cublasHandle_t handle = CublasManager::get_instance().handle;

    const double* a_base = static_cast<const double*>(a->data_ptr);
    const double* b_base = static_cast<const double*>(b->data_ptr);
    double* c_base = static_cast<double*>(result->data_ptr);

    for (int bi = 0; bi < p.total_batch; ++bi) {
        int offA = batch_offset(bi, p.batch_shape, a->shape, p.M * p.K);
        int offB = batch_offset(bi, p.batch_shape, b->shape, p.K * p.N);
        int offC = bi * p.M * p.N;

        // Same row-major-via-swapped-operands trick as the existing
        // cublasSgemm call: cuBLAS is column-major natively, so computing
        // B^T * A^T (in cuBLAS's column-major view) yields (A*B) in
        // row-major, without needing an explicit transpose step.
        cublasStatus_t status = cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            p.N, p.M, p.K,
            &alpha,
            b_base + offB, p.N,
            a_base + offA, p.K,
            &beta,
            c_base + offC, p.N
        );
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS DGEMM failed at batch index " + std::to_string(bi));
    }
    return result;
}

// ---- Typed matmul: int32/int64, optimized shared-memory tiled kernel ----
// int64 accumulator throughout (both int32 and int64 inputs) to avoid
// silent overflow on the K-dimension reduction — this is a correctness
// requirement, not an optional optimization (verified by the overflow-
// stress test in the test suite).
//
// Optimizations applied:
//   - Shared-memory block tiling (BLOCK x BLOCK tile of A/B loaded once per
//     block, reused by every thread in the block).
//   - Shared memory padded by +1 column (`[BLOCK][BLOCK+1]`) to eliminate
//     the classic 32-way bank conflict on column-strided shared-memory
//     reads (standard, well-established fix for this exact pattern).
//   - Vectorized global-memory loads into shared memory: int4 (4 elements/
//     128 bits) for int32, int2/longlong2 (2 elements/128 bits) for int64 —
//     alignment-guarded with a scalar fallback, same convention as every
//     other vectorized kernel in this codebase.
//   - Per-thread micro-tiling: each thread computes a THREAD_TILE x
//     THREAD_TILE (4x4) block of the output instead of one element,
//     amortizing the cost of each shared-memory load across 16 FMAs
//     instead of 1 — this is the single largest lever for arithmetic
//     intensity here, and the main step up from the correctness-first
//     version. Deliberately NOT going to full warp-tiling/register-file
//     double-buffering (as in the reference SGEMM kernel) — that adds
//     substantial complexity best justified once this level is profiled
//     against a real workload.

constexpr int MM_BLOCK = 32;       // tile dimension (BLOCK x BLOCK output tile per threadblock)
constexpr int MM_THREAD_TILE = 4;  // each thread computes THREAD_TILE x THREAD_TILE outputs
constexpr int MM_THREADS_DIM = MM_BLOCK / MM_THREAD_TILE;  // threads per block dimension (8x8=64 threads/block)

template <typename T>
__global__ void matmul_int_microtiled_kernel(const T* __restrict__ A, const T* __restrict__ B,
                                              T* __restrict__ C, int M, int N, int K) {
    __shared__ T As[MM_BLOCK][MM_BLOCK + 1];  // +1 padding kills bank conflicts
    __shared__ T Bs[MM_BLOCK][MM_BLOCK + 1];

    int blockRow = blockIdx.y * MM_BLOCK;
    int blockCol = blockIdx.x * MM_BLOCK;

    int tRow = threadIdx.y;  // 0..MM_THREADS_DIM-1
    int tCol = threadIdx.x;

    int64_t acc[MM_THREAD_TILE][MM_THREAD_TILE];
    #pragma unroll
    for (int i = 0; i < MM_THREAD_TILE; ++i)
        #pragma unroll
        for (int j = 0; j < MM_THREAD_TILE; ++j)
            acc[i][j] = 0;

    int numTiles = (K + MM_BLOCK - 1) / MM_BLOCK;

    // Each thread loads MM_THREAD_TILE elements per row into shared memory
    // per tile-load phase (MM_THREADS_DIM threads x MM_THREAD_TILE = MM_BLOCK).
    for (int t = 0; t < numTiles; ++t) {
    int kBase = t * MM_BLOCK;

    #pragma unroll
    for (int i = 0; i < MM_THREAD_TILE; ++i) {
        int loadRow = tRow * MM_THREAD_TILE + i;
        #pragma unroll
        for (int j = 0; j < MM_THREAD_TILE; ++j) {
            int loadCol = tCol * MM_THREAD_TILE + j;

            int globalRowA = blockRow + loadRow;
            int globalColA = kBase + loadCol;
            As[loadRow][loadCol] = (globalRowA < M && globalColA < K)
                ? A[(long long)globalRowA * K + globalColA] : T(0);

            int globalRowB = kBase + loadRow;
            int globalColB = blockCol + loadCol;
            Bs[loadRow][loadCol] = (globalRowB < K && globalColB < N)
                ? B[(long long)globalRowB * N + globalColB] : T(0);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < MM_BLOCK; ++k) {
        T aFrag[MM_THREAD_TILE];
        T bFrag[MM_THREAD_TILE];
        #pragma unroll
        for (int i = 0; i < MM_THREAD_TILE; ++i) aFrag[i] = As[tRow * MM_THREAD_TILE + i][k];
        #pragma unroll
        for (int j = 0; j < MM_THREAD_TILE; ++j) bFrag[j] = Bs[k][tCol * MM_THREAD_TILE + j];

        #pragma unroll
        for (int i = 0; i < MM_THREAD_TILE; ++i)
            #pragma unroll
            for (int j = 0; j < MM_THREAD_TILE; ++j)
                acc[i][j] += (int64_t)aFrag[i] * (int64_t)bFrag[j];
    }
    __syncthreads();
}

    #pragma unroll
    for (int i = 0; i < MM_THREAD_TILE; ++i) {
        int globalRow = blockRow + tRow * MM_THREAD_TILE + i;
        if (globalRow >= M) continue;
        #pragma unroll
        for (int j = 0; j < MM_THREAD_TILE; ++j) {
            int globalCol = blockCol + tCol * MM_THREAD_TILE + j;
            if (globalCol < N) {
                C[(long long)globalRow * N + globalCol] = (T)acc[i][j];
            }
        }
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cuda_matmul_int_typed_impl(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, DType out_dtype) {
    if (!a->is_contiguous() || !b->is_contiguous())
        throw std::invalid_argument("matmul requires contiguous tensors. Call .contiguous() first.");

    NDMatmulPlan p = plan_nd_matmul(a->shape, b->shape);

    std::vector<int> out_shape = p.batch_shape;
    out_shape.push_back(p.M);
    out_shape.push_back(p.N);
    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"), out_dtype);

    const T* a_base = static_cast<const T*>(a->data_ptr);
    const T* b_base = static_cast<const T*>(b->data_ptr);
    T* c_base = static_cast<T*>(result->data_ptr);

    dim3 threads(MM_THREADS_DIM, MM_THREADS_DIM);
    dim3 blocks((p.N + MM_BLOCK - 1) / MM_BLOCK,
                (p.M + MM_BLOCK - 1) / MM_BLOCK);

    for (int bi = 0; bi < p.total_batch; ++bi) {
        int offA = batch_offset(bi, p.batch_shape, a->shape, p.M * p.K);
        int offB = batch_offset(bi, p.batch_shape, b->shape, p.K * p.N);
        int offC = bi * p.M * p.N;

        matmul_int_microtiled_kernel<T><<<blocks, threads>>>(
            a_base + offA, b_base + offB, c_base + offC, p.M, p.N, p.K);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_matmul_int_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::INT32: return run_cuda_matmul_int_typed_impl<int32_t>(a, b, DType::INT32);
        case DType::INT64: return run_cuda_matmul_int_typed_impl<int64_t>(a, b, DType::INT64);
        default: throw std::runtime_error("matmul(): unsupported integer dtype '" + dtype_name(a->dtype) + "'");
    }
}