#pragma once
#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef AAKAAR_NO_CUDA
#include <cuda_runtime.h>
#include "allocator.h"
#endif

namespace py = pybind11;

class Tensor {
public:
    float* data_ptr;
    int size;                     // total element count (product of shape)
    std::vector<int> shape;       // e.g. {100, 100}
    std::string device;

    // Construct from a shape vector, e.g. {100, 100} or {50}
    Tensor(std::vector<int> shp, std::string dev) : shape(shp), device(dev) {
        size = 1;
        for (int d : shape) {
            if (d <= 0) throw std::invalid_argument("Tensor dimensions must be positive");
            size *= d;
        }
        if (device == "cuda") {
#ifdef AAKAAR_NO_CUDA
            throw std::runtime_error("This build of aakaar was compiled without CUDA support. "
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
        py::array_t<float> result(shape);   // gives numpy the real shape, not flat
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

    std::string shape_str() {
        std::string s = "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i < shape.size() - 1) s += ", ";
        }
        s += shape.size() == 1 ? ",)" : ")";
        return s;
    }

    std::string repr() {
        std::string out = "aakaar.Tensor([";
        int preview = size < 6 ? size : 6;
        for (int i = 0; i < preview; ++i) {
            out += std::to_string(get_item(i));
            if (i < preview - 1) out += ", ";
        }
        if (size > preview) out += ", ...";
        out += "], device='" + device + "', shape=" + shape_str() + ")";
        return out;
    }
};