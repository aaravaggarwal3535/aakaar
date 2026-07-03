#pragma once
#include <curand.h>
#include <stdexcept>

class CUDARandomManager {
private:
    curandGenerator_t generator;
    bool initialized;

    // Private constructor (Singleton)
    CUDARandomManager() : initialized(false) {}

public:
    static CUDARandomManager& get_instance() {
        static CUDARandomManager instance;
        return instance;
    }

    // Initialize the generator only once
    // Initialize the generator only once
    curandGenerator_t get_generator(unsigned long long seed) {
        if (!initialized) {
            curandStatus_t status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
            if (status != CURAND_STATUS_SUCCESS) {  // <--- FIXED HERE
                throw std::runtime_error("Failed to create cuRAND generator");
            }
            initialized = true;
        }
        
        curandSetPseudoRandomGeneratorSeed(generator, seed);
        return generator;
    }

    // Clean up the generator when the engine shuts down
    ~CUDARandomManager() {
        if (initialized) {
            curandDestroyGenerator(generator);
        }
    }
};