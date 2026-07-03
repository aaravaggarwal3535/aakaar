#pragma once
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "allocator.h" // Your CachingAllocator

namespace py = pybind11;

class Tensor {
public:
    float* data_ptr;
    int size;
    std::string device;

    Tensor(int s, std::string dev) : size(s), device(dev) {
        if (device == "cuda") {
            data_ptr = CachingAllocator::get_instance().allocate(size);
        } else {
            data_ptr = new float[size]; // Standard heap allocation
        }
    }

    ~Tensor() {
        if (device == "cuda") {
            CachingAllocator::get_instance().free(data_ptr, size);
        } else {
            delete[] data_ptr;
        }
    }

    // Copy data back to a numpy array on the host, regardless of device
    py::array_t<float> to_numpy() {
        py::array_t<float> result(size);
        auto buf = result.mutable_data();

        if (device == "cuda") {
            cudaError_t err = cudaMemcpy(buf, data_ptr, size * sizeof(float),
                                          cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy failed in to_numpy(): " +
                                          std::string(cudaGetErrorString(err)));
            }
        } else {
            std::memcpy(buf, data_ptr, size * sizeof(float));
        }
        return result;
    }
};