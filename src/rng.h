#pragma once
#include <curand.h>
#include <stdexcept>

class CUDARandomManager {
private:
    curandGenerator_t generator;
    bool initialized;
    unsigned long long last_seed;
    bool has_seed;

    CUDARandomManager() : initialized(false), has_seed(false), last_seed(0) {}

public:
    static CUDARandomManager& get_instance() {
        static CUDARandomManager instance;
        return instance;
    }

    curandGenerator_t get_generator(unsigned long long seed) {
        if (!initialized) {
            curandStatus_t status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
            if (status != CURAND_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuRAND generator");
            }
            initialized = true;
        }

        // Only pay the (expensive) reseed cost when the seed actually changed.
        if (!has_seed || seed != last_seed) {
            curandSetPseudoRandomGeneratorSeed(generator, seed);
            last_seed = seed;
            has_seed = true;
        }

        return generator;
    }

    ~CUDARandomManager() {
        if (initialized) {
            curandDestroyGenerator(generator);
        }
    }
};