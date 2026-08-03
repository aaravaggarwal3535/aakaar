#pragma once
#include <cudnn.h>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

class CudnnManager {
public:
    static CudnnManager& get_instance() {
        static CudnnManager instance;
        return instance;
    }
    cudnnHandle_t handle;
private:
    CudnnManager() {
        cudnnStatus_t st = cudnnCreate(&handle);
        if (st != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(
                "cudnnCreate failed — is libcudnn installed and does its version match your CUDA toolkit?");
    }
    ~CudnnManager() { cudnnDestroy(handle); }
};

inline void warm_cudnn() {
    CudnnManager::get_instance();  // constructs on first call, no-op after
}

// Off by default. Set AAKAAR_CUDNN_VERBOSE=1 in the environment to see
// which cuDNN algorithm was chosen for each new (shape, params) combination
// and how long each candidate took — useful when debugging unexpectedly
// slow convolutions, not needed for normal training runs.
inline bool aakaar_cudnn_debug_enabled() {
    static bool enabled = []() {
        const char* env = std::getenv("AAKAAR_CUDNN_VERBOSE");
        return env != nullptr && std::strcmp(env, "0") != 0 && env[0] != '\0';
    }();
    return enabled;
}