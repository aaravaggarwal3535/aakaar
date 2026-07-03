#include <random>
#include <memory>
#include "tensor.h"

void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < t->size; ++i) {
        t->data_ptr[i] = dis(gen);
    }
}