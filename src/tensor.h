// tensor.h
#pragma once
#include <string>
#include <stdexcept>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifndef AAKAAR_NO_CUDA
#include <cuda_runtime.h>
#include "allocator.h"
#endif

namespace py = pybind11;

class Tensor {
public:
    float* data_ptr;
    int size;
    std::string device;

    Tensor(int s, std::string dev) : size(s), device(dev) {
        if (device == "cuda") {
#ifdef AAKAAR_NO_CUDA
            throw std::runtime_error("This build of aakaar was compiled without CUDA support "
                                      "(no CUDA toolkit was found at install time). "
                                      "Use device='cpu' instead.");
#else
            data_ptr = CachingAllocator::get_instance().allocate(size);
#endif
        } else {
            data_ptr = new float[size];
        }
    }

    ~Tensor() {
        if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
            CachingAllocator::get_instance().free(data_ptr, size);
#endif
        } else {
            delete[] data_ptr;
        }
    }

    py::array_t<float> to_numpy() {
        py::array_t<float> result(size);
        auto buf = result.mutable_data();
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(buf, data_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
            return result;
        }
#endif
        std::memcpy(buf, data_ptr, size * sizeof(float));
        return result;
    }
    // Add these methods inside the Tensor class in tensor.h

float get_item(int index) {
    if (index < 0 || index >= size) {
        throw std::out_of_range("Tensor index out of range");
    }
    float value;
#ifndef AAKAAR_NO_CUDA
    if (device == "cuda") {
        cudaMemcpy(&value, data_ptr + index, sizeof(float), cudaMemcpyDeviceToHost);
        return value;
    }
#endif
    return data_ptr[index];
}

std::string repr() {
    // Preview first few elements, like torch's tensor([...]) repr
    std::string out = "aakaar.Tensor([";
    int preview = size < 6 ? size : 6;
    for (int i = 0; i < preview; ++i) {
        out += std::to_string(get_item(i));
        if (i < preview - 1) out += ", ";
    }
    if (size > preview) out += ", ...";
    out += "], device='" + device + "', size=" + std::to_string(size) + ")";
    return out;
}
};

