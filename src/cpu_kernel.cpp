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