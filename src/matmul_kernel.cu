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

struct MatmulPlan {
    int batch, M, K, N;
    long long strideA, strideB, strideC;
};

// Figures out batch/M/K/N and broadcast strides from two input shapes.
static MatmulPlan plan_matmul(const std::vector<int>& shapeA, const std::vector<int>& shapeB) {
    auto last2 = [](const std::vector<int>& s) { return std::make_pair(s[s.size()-2], s[s.size()-1]); };

    int ndimA = (int)shapeA.size();
    int ndimB = (int)shapeB.size();
    if (ndimA < 2 || ndimB < 2)
        throw std::invalid_argument("matmul requires tensors with at least 2 dimensions");
    if (ndimA > 3 || ndimB > 3)
        throw std::invalid_argument("matmul currently supports at most one batch dimension (2D or 3D tensors)");

    auto [M, K] = last2(shapeA);
    auto [K2, N] = last2(shapeB);
    if (K != K2)
        throw std::invalid_argument("Shape mismatch: inner dimensions must match for matmul");

    int batchA = ndimA == 3 ? shapeA[0] : 1;
    int batchB = ndimB == 3 ? shapeB[0] : 1;

    int batch;
    long long strideA, strideB;
    if (batchA == batchB) {
        batch = batchA;
        strideA = (long long)M * K * (batchA > 1 ? 1 : 0);
        strideB = (long long)K * N * (batchB > 1 ? 1 : 0);
    } else if (batchA == 1) {
        batch = batchB;
        strideA = 0;  // broadcast: reuse same A for every batch
        strideB = (long long)K * N;
    } else if (batchB == 1) {
        batch = batchA;
        strideA = (long long)M * K;
        strideB = 0;  // broadcast: reuse same B for every batch
    } else {
        throw std::invalid_argument("Batch dimensions must match or one must be 1 (broadcast); got " +
                                     std::to_string(batchA) + " and " + std::to_string(batchB));
    }

    return {batch, M, K, N, strideA, strideB, (long long)M * N};
}

std::shared_ptr<Tensor> run_cublas_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a->is_contiguous() || !b->is_contiguous())
        throw std::invalid_argument("matmul requires contiguous tensors. Call .contiguous() first.");

    MatmulPlan p = plan_matmul(a->shape, b->shape);

    std::vector<int> out_shape = p.batch > 1 ? std::vector<int>{p.batch, p.M, p.N}
                                              : std::vector<int>{p.M, p.N};
    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"));

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = CublasManager::get_instance().handle;

    // Row-major C = A*B via cuBLAS's column-major convention: compute C^T = B^T * A^T
    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        p.N, p.M, p.K,
        &alpha,
        b->data_ptr, p.N, p.strideB,
        a->data_ptr, p.K, p.strideA,
        &beta,
        result->data_ptr, p.N, p.strideC,
        p.batch
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS SGEMM (batched) failed");
    cudaDeviceSynchronize();
    return result;
}