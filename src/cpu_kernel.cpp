#include <random>
#include <memory>
#include <stdexcept>
#include "tensor.h"
#include <omp.h>

void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < t->size; ++i) {
        t->data_ptr[i] = dis(gen);
    }
}

std::shared_ptr<Tensor> run_cpu_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a->is_contiguous() || !b->is_contiguous())
        throw std::invalid_argument("matmul requires contiguous tensors. Call .contiguous() first.");

    auto last2 = [](const std::vector<int>& s) { return std::make_pair(s[s.size()-2], s[s.size()-1]); };
    int ndimA = (int)a->shape.size(), ndimB = (int)b->shape.size();
    if (ndimA < 2 || ndimB < 2)
        throw std::invalid_argument("matmul requires tensors with at least 2 dimensions");
    if (ndimA > 3 || ndimB > 3)
        throw std::invalid_argument("matmul currently supports at most one batch dimension (2D or 3D tensors)");

    auto [M, K] = last2(a->shape);
    auto [K2, N] = last2(b->shape);
    if (K != K2) throw std::invalid_argument("Shape mismatch: inner dimensions must match");

    int batchA = ndimA == 3 ? a->shape[0] : 1;
    int batchB = ndimB == 3 ? b->shape[0] : 1;
    int batch;
    if (batchA == batchB) batch = batchA;
    else if (batchA == 1) batch = batchB;
    else if (batchB == 1) batch = batchA;
    else throw std::invalid_argument("Batch dimensions must match or one must be 1");

    std::vector<int> out_shape = batch > 1 ? std::vector<int>{batch, M, N} : std::vector<int>{M, N};
    auto result = std::make_shared<Tensor>(out_shape, std::string("cpu"));

    // 1. Multithreading: Parallelize across the batch and M dimensions
    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < batch; ++bi) {
        for (int i = 0; i < M; ++i) {
            // Recalculate pointers inside the loop for thread safety
            const float* a_ptr = a->data_ptr + (batchA > 1 ? bi * M * K : 0);
            const float* b_ptr = b->data_ptr + (batchB > 1 ? bi * K * N : 0);
            float* c_ptr = result->data_ptr + bi * M * N;

            // Initialize the current row of C to 0
            for (int j = 0; j < N; ++j) {
                c_ptr[i * N + j] = 0.0f;
            }

            // 2. Loop Reordering: Swap the j and k loops (i-k-j instead of i-j-k)
            for (int k = 0; k < K; ++k) {
                // Cache the scalar value of A to avoid re-reading it
                float a_ik = a_ptr[i * K + k]; 
                
                // 3. Auto-Vectorization: Inner loop now accesses memory sequentially
                // A modern compiler will vectorize this loop using AVX/SSE
                for (int j = 0; j < N; ++j) {
                    c_ptr[i * N + j] += a_ik * b_ptr[k * N + j];
                }
            }
        }
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) {
        throw std::invalid_argument("Shape mismatch in elementwise op");
    }
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        result->data_ptr[i] = a->get_scalar_flat(i) + b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) {
        throw std::invalid_argument("Shape mismatch in elementwise op");
    }
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        result->data_ptr[i] = a->get_scalar_flat(i) - b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) throw std::invalid_argument("Shape mismatch in elementwise op");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) * b->get_scalar_flat(i);
    return result;
}

std::shared_ptr<Tensor> run_cpu_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) throw std::invalid_argument("Shape mismatch in elementwise op");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) / b->get_scalar_flat(i);
    return result;
}

std::shared_ptr<Tensor> run_cpu_add_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) + s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_sub_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) - s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_mul_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) * s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_div_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) / s;
    return result;
}
