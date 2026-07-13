#include <cuda_runtime.h>
#include <curand.h>
#include <memory>
#include "tensor.h"
#include "rng.h" // Import the random manager

void run_curand_uniform(std::shared_ptr<Tensor> t, unsigned long long seed) {
    // Fetch the persistent global generator
    curandGenerator_t generator = CUDARandomManager::get_instance().get_generator(seed);

    // Generate random numbers directly into the existing tensor's buffer
    curandStatus_t status = curandGenerateUniform(generator, static_cast<float*>(t->fptr()), t->size);
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("cuRAND generation failed");
    }

    // Ensure the GPU finishes execution before returning control to Python
}
