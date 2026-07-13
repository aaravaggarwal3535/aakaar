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
