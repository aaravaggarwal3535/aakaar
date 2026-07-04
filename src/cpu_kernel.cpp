#include <random>
#include <memory>
#include <stdexcept>
#include "tensor.h"

void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < t->size; ++i) {
        t->data_ptr[i] = dis(gen);
    }
}

std::shared_ptr<Tensor> run_cpu_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::invalid_argument("matmul currently supports 2D tensors only");
    }
    int M = a->shape[0];
    int K = a->shape[1];
    int K2 = b->shape[0];
    int N = b->shape[1];
    if (K != K2) {
        throw std::invalid_argument("Shape mismatch: inner dimensions must match");
    }

    auto result = std::make_shared<Tensor>(std::vector<int>{M, N}, std::string("cpu"));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a->data_ptr[i * K + k] * b->data_ptr[k * N + j];
            }
            result->data_ptr[i * N + j] = sum;
        }
    }
    return result;
}