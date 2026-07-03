#include <cuda_runtime.h>
#include <curand.h>
#include <memory>
#include "tensor.h" // Import our new class

// Return a shared pointer to our Tensor so Pybind11 can manage its lifecycle safely
std::shared_ptr<Tensor> run_curand_uniform(int size, unsigned long long seed) {
    
    // 1. Create a new Tensor (this automatically triggers cudaMalloc)
    auto tensor = std::make_shared<Tensor>(size);

    // 2. Setup cuRAND
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, seed);

    // 3. Generate floats directly into the Tensor's GPU memory
    curandGenerateUniform(generator, tensor->data_ptr, size);

    // 4. Cleanup generator (but DO NOT free the tensor memory!)
    curandDestroyGenerator(generator);

    // Return the remote control back to Python
    return tensor;
}