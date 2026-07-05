#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef AAKAAR_NO_CUDA
#include <cuda_runtime.h>
#include "allocator.h"
#endif

namespace py = pybind11;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    float* data_ptr;               // points at the first element THIS tensor/view sees
    int size;                      // element count of this view
    std::vector<int> shape;
    std::vector<int> strides;      // in elements
    std::string device;

    std::shared_ptr<Tensor> base;  // non-null => this is a view; keeps owner alive
    float* owned_ptr = nullptr;    // the actually-allocated buffer (only if base == nullptr)

    static std::vector<int> contiguous_strides(const std::vector<int>& shp) {
        std::vector<int> st(shp.size());
        int acc = 1;
        for (int i = (int)shp.size() - 1; i >= 0; --i) {
            st[i] = acc;
            acc *= shp[i];
        }
        return st;
    }

    // Fresh allocation (used by rand(), matmul() results, etc.)
    Tensor(std::vector<int> shp, std::string dev) : shape(shp), device(dev), base(nullptr) {
        size = 1;
        for (int d : shape) {
            if (d <= 0) throw std::invalid_argument("Tensor dimensions must be positive");
            size *= d;
        }
        strides = contiguous_strides(shape);
        if (device == "cuda") {
#ifdef AAKAAR_NO_CUDA
            throw std::runtime_error("This build of aakaar was compiled without CUDA support. Use device='cpu' instead.");
#else
            owned_ptr = CachingAllocator::get_instance().allocate(size);
#endif
        } else {
            owned_ptr = new float[size];
        }
        data_ptr = owned_ptr;
    }

    // View constructor: shares memory of `parent` at element offset `off`
    Tensor(std::shared_ptr<Tensor> parent, int off, std::vector<int> shp, std::vector<int> strd)
        : shape(shp), strides(strd), device(parent->device), base(parent) {
        size = 1;
        for (int d : shape) size *= d;
        data_ptr = parent->data_ptr + off;
    }

    ~Tensor() {
        if (base == nullptr) {   // only owners free memory; views just drop their shared_ptr ref
            if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
                CachingAllocator::get_instance().free(owned_ptr, size);
#endif
            } else {
                delete[] owned_ptr;
            }
        }
    }

    bool is_contiguous() const {
        return strides == contiguous_strides(shape);
    }

    std::shared_ptr<Tensor> contiguous() {
        if (is_contiguous()) {
            return shared_from_this();  // already contiguous, return self reference
        }
        auto result = std::make_shared<Tensor>(shape, device);
        std::vector<int> idx(shape.size(), 0);
        for (int flat = 0; flat < size; ++flat) {
            float val = get_scalar(idx);
            if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
                cudaMemcpy(result->data_ptr + flat, &val, sizeof(float), cudaMemcpyHostToDevice);
#endif
            } else {
                result->data_ptr[flat] = val;
            }
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return result;
    }

    // === NEW TO_DEVICE METHOD ADDED HERE ===
    std::shared_ptr<Tensor> to_device(std::string target_device) {
        if (target_device == device) {
            return shared_from_this();  // no-op, already on the target device
        }

#ifdef AAKAAR_NO_CUDA
        if (target_device == "cuda") {
            throw std::runtime_error("This build of aakaar has no CUDA support.");
        }
#endif

        // Materialize source into a flat host buffer first (handles views/strides correctly)
        std::vector<float> host_buf(size);
        std::vector<int> idx(shape.size(), 0);
        for (int flat = 0; flat < size; ++flat) {
            host_buf[flat] = get_scalar(idx);
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }

        auto result = std::make_shared<Tensor>(shape, target_device);

#ifndef AAKAAR_NO_CUDA
        if (target_device == "cuda") {
            cudaMemcpy(result->data_ptr, host_buf.data(), size * sizeof(float), cudaMemcpyHostToDevice);
            return result;
        }
#endif
        std::memcpy(result->data_ptr, host_buf.data(), size * sizeof(float));
        return result;
    }

    float get_scalar(const std::vector<int>& idx) {
        if (idx.size() != shape.size())
            throw std::out_of_range("Index dimensionality does not match tensor shape");
        int off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] < 0 || idx[i] >= shape[i])
                throw std::out_of_range("Tensor index out of range");
            off += idx[i] * strides[i];
        }
        float value;
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(&value, data_ptr + off, sizeof(float), cudaMemcpyDeviceToHost);
            return value;
        }
#endif
        return data_ptr[off];
    }

    // Access element at flat logical index i, respecting strides (handles views correctly)
    float get_scalar_flat(int flat_idx) {
        std::vector<int> idx(shape.size());
        int remaining = flat_idx;
        for (int d = (int)shape.size() - 1; d >= 0; --d) {
            idx[d] = remaining % shape[d];
            remaining /= shape[d];
        }
        return get_scalar(idx);
    }

    // Correctly handles non-contiguous strided views on both CPU and CUDA
    py::array_t<float> to_numpy() {
        py::array_t<float> result(shape);
        float* out = result.mutable_data();

        std::vector<float> host_buf;
        const float* src;

        if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
            int max_off = 0;
            for (size_t i = 0; i < shape.size(); ++i)
                if (shape[i] > 0) max_off += (shape[i] - 1) * strides[i];
            host_buf.resize(max_off + 1);
            cudaMemcpy(host_buf.data(), data_ptr, (max_off + 1) * sizeof(float), cudaMemcpyDeviceToHost);
            src = host_buf.data();
#else
            throw std::runtime_error("CUDA tensor on a CPU-only build");
#endif
        } else {
            src = data_ptr;
        }

        std::vector<int> idx(shape.size(), 0);
        for (int flat = 0; flat < size; ++flat) {
            int off = 0;
            for (size_t d = 0; d < shape.size(); ++d) off += idx[d] * strides[d];
            out[flat] = src[off];
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return result;
    }

    std::string shape_str() const {
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
        std::vector<int> idx(shape.size(), 0);
        for (int i = 0; i < preview; ++i) {
            out += std::to_string(get_scalar(idx));
            if (i < preview - 1) out += ", ";
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        if (size > preview) out += ", ...";
        out += "], device='" + device + "', shape=" + shape_str() + ")";
        return out;
    }
};
