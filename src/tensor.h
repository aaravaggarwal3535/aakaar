#pragma once
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

class Tensor {
public:
    float* data_ptr;
    int size;

    // Constructor: When created, allocate GPU memory
    Tensor(int s) : size(s) {
        cudaError_t err = cudaMalloc((void**)&data_ptr, size * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA Malloc failed!");
        }
    }

    // Destructor: When Python deletes this object, free the GPU memory!
    ~Tensor() {
        cudaFree(data_ptr);
    }

    // Helper method: Bring data across the PCI-e bus to the CPU as a NumPy array
    py::array_t<float> cpu() {
        // Allocate an empty NumPy array
        py::array_t<float> result(size);
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);

        // Copy from Device (GPU) to Host (CPU)
        cudaMemcpy(ptr, data_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        return result;
    }
};