#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <stdexcept>
#include "tensor.h"

// Reuse a persistent cuBLAS handle, same pattern as your CUDARandomManager
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

// C = A @ B, where A is (M, K) and B is (K, N), both row-major float32
std::shared_ptr<Tensor> run_cublas_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::invalid_argument("matmul currently supports 2D tensors only");
    }
    int M = a->shape[0];
    int K = a->shape[1];
    int K2 = b->shape[0];
    int N = b->shape[1];
    if (K != K2) {
        throw std::invalid_argument("Shape mismatch: A is (" + std::to_string(M) + ", " +
                                     std::to_string(K) + "), B is (" + std::to_string(K2) +
                                     ", " + std::to_string(N) + ") — inner dimensions must match");
    }

    auto result = std::make_shared<Tensor>(std::vector<int>{M, N}, std::string("cuda"));

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = CublasManager::get_instance().handle;

    // cuBLAS is column-major internally; to compute row-major C = A*B we swap
    // operand order and dimensions (standard trick: C^T = B^T * A^T).
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b->data_ptr, N,
        a->data_ptr, K,
        &beta,
        result->data_ptr, N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS SGEMM failed");
    }
    cudaDeviceSynchronize();
    return result;
}